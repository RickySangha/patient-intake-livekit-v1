import logging
import os
import sys
from dotenv import load_dotenv
from typing import Optional

from livekit.agents import (
    AgentSession,
    JobProcess,
    JobContext,
    WorkerOptions,
    cli,
    metrics,
)
from livekit.plugins import deepgram, openai, silero, groq, azure
from livekit.plugins.turn_detector.english import EnglishModel
from livekit.plugins.openai import LLM as OPENAI_LLM
from livekit.agents import metrics

from agents.entry import EntryAgent
from custom_types.patient_context import PatientContext
from plugin_kokoro.kokoro import KokoroTTS
from db.firestore_client import prepare_initial_patient_context

logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)
# # Configure basic logging if not done elsewhere
# logging.basicConfig(
#     level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
# )


load_dotenv()


def prewarm(proc: JobProcess):
    """Function called before the worker process starts handling jobs."""
    logger.info("Prewarming process...")
    proc.userdata["vad"] = silero.VAD.load(activation_threshold=1)


async def entrypoint(ctx: JobContext):
    """Main entry point for the agent job."""
    # TODO: Hardcoded for now. When Telephony is implemented, this will be provided on agent dispatch.
    appointment_id = "appt1"

    try:
        # Prepare context with full patient history
        initial_context = await prepare_initial_patient_context(appointment_id)

        logger.info(
            f"Successfully initialized PatientContext for appointment: {appointment_id}"
        )

        # Log information about loaded context
        patient_name = (
            f"{initial_context.first_name} {initial_context.last_name}"
            if initial_context.first_name
            else "Unknown Patient"
        )
        doctor_name = (
            f"{initial_context.doctor.firstName} {initial_context.doctor.lastName}"
            if initial_context.doctor
            else "Unknown Doctor"
        )
        clinic_name = (
            initial_context.clinic.name if initial_context.clinic else "Unknown Clinic"
        )

        logger.info(
            f"Call with {patient_name}, doctor: {doctor_name}, clinic: {clinic_name}"
        )

        # Log if we have medical history
        has_history = initial_context.medical_history is not None and bool(
            initial_context.medical_history
        )
        logger.info(f"Loaded with medical history: {has_history}")

    except RuntimeError as e:
        logger.critical(
            f"Stopping job execution due to Firestore client init error during context initialization: {e}"
        )
        raise
    except Exception as e:
        logger.exception(
            f"Unexpected error initializing context for appointment {appointment_id}: {e}",
            exc_info=True,
        )
        # Create a fallback context. TODO: we need to handle this better. Although this is unlikely to happen.
        initial_context = PatientContext(
            id=None, first_name="Error", last_name="Loading Context"
        )

    try:
        await ctx.connect()
        logger.info("Job context connected to room: %s", ctx.room.name)
    except Exception as e:
        logger.exception("Failed to connect job context to room.", exc_info=True)
        return

    vad_plugin = ctx.proc.userdata.get("vad")
    if not vad_plugin:
        logger.warning("VAD plugin not found in prewarmed data, loading again.")
        vad_plugin = silero.VAD.load()

    azure_llm = OPENAI_LLM.with_azure(
        model="gpt-4o-mini",
        temperature=0.2,
    )
    logger.info("Azure LLM initialized.")

    tts = deepgram.TTS()
    logger.info("Deepgram TTS initialized.")

    # Create agent session with PatientContext
    session = AgentSession[PatientContext](
        vad=vad_plugin,
        llm=azure_llm,
        stt=deepgram.STT(),
        tts=tts,
        userdata=initial_context,
        turn_detection=EnglishModel(),
    )

    @session.on("metrics_collected")
    def _on_metrics_collected(mtrcs: metrics.AgentMetrics):
        metrics.log_metrics(mtrcs)

    logger.info("Starting initial agent: EntryAgent")

    # Extract info for EntryAgent initialization
    agent_name = (
        initial_context.clinic.open_settings.agent_name
        if (
            initial_context.clinic
            and initial_context.clinic.open_settings
            and initial_context.clinic.open_settings.agent_name
        )
        else "Automated Assistant"
    )

    clinic_name = (
        initial_context.clinic.name if initial_context.clinic else "Medical Clinic"
    )

    doctor_name = "Doctor"
    if initial_context.doctor:
        doctor_first = initial_context.doctor.firstName or ""
        doctor_last = initial_context.doctor.lastName or ""
        doctor_name = (
            f"Dr. {doctor_last}"
            if doctor_last
            else f"Dr. {doctor_first}" if doctor_first else "Doctor"
        )

    patient_first_name = initial_context.first_name or "Patient"
    patient_last_name = initial_context.last_name or ""
    patient_age = initial_context.age or 0
    patient_gender = initial_context.gender or "unknown"

    # Create the entry agent with all the context information
    entry_agent = EntryAgent(
        agent_name=agent_name,
        clinic_name=clinic_name,
        doctor_name=doctor_name,
        patient_first_name=patient_first_name,
        patient_last_name=patient_last_name,
        patient_age=patient_age,
        patient_gender=patient_gender,
    )

    # Start the agent session
    await session.start(
        agent=entry_agent,
        room=ctx.room,
    )
    logger.info("Agent session finished or closed for room: %s", ctx.room.name)


if __name__ == "__main__":
    # Ensure GOOGLE_APPLICATION_CREDENTIALS is set in the environment
    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        print(
            "CRITICAL ERROR: GOOGLE_APPLICATION_CREDENTIALS environment variable not set.",
            file=sys.stderr,
        )
        sys.exit(1)

    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        )
    )
