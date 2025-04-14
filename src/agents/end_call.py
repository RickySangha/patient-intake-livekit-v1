import logging
from typing import Optional
import datetime
import uuid

from livekit.agents import Agent, ChatContext
from livekit.agents.voice import RunContext
from livekit.agents.llm import function_tool

# Import updated types
from custom_types.patient_context import PatientContext
from custom_types.call_details import CallDetails

# Import DB client and functions
from db.firestore_client import (
    save_complaint_data,
    save_medical_history,
    save_call_details,
    update_call_with_collected_data,
)

logger = logging.getLogger(__name__)


class EndCallAgent(Agent):
    """
    Agent responsible for saving collected data to Firestore
    and ending the call session.
    """

    def __init__(
        self, *, chat_ctx: Optional[ChatContext] = None, system_prompt: str
    ) -> None:
        self.SYSTEM_PROMPT = system_prompt
        CURRENT_TASK = "You have gathered all necessary information. Ask the patient if they have final questions. Answer briefly or defer to the doctor. Then, politely end the call. Thank them, mention the doctor will review the info, and wish them well. CRITICALLY, you MUST call 'end_call' to finalize."

        super().__init__(
            instructions=f"{self.SYSTEM_PROMPT}\n\n{CURRENT_TASK}",
            chat_ctx=chat_ctx,
        )

    async def on_enter(self):
        """Called when the agent becomes active. Stores the session context."""
        logger.info("EndCallAgent entered")

    @function_tool()
    async def end_call(self, context: RunContext[PatientContext]):
        """
        Saves the collected data, then ends the call session.
        This function MUST be called to properly conclude the interaction."""

        # --- Prepare Data ---
        patient_context: PatientContext = context.userdata
        call_details: CallDetails = patient_context.call_details
        patient_id = patient_context.id
        call_id = call_details.call_id or f"call_{uuid.uuid4()}"

        # Ensure call_id is set
        if not call_details.call_id:
            logger.info(f"Setting new call_id: {call_id}")
            call_details.call_id = call_id

        # Update call details with end-of-call information
        call_details.call_end_time = datetime.datetime.now(datetime.timezone.utc)
        call_details.call_status = "completed"

        # Calculate duration if possible
        start_time = call_details.call_start_time
        end_time = call_details.call_end_time
        duration = None
        if start_time and end_time and isinstance(start_time, datetime.datetime):
            duration = (end_time - start_time).total_seconds() / 60
        call_details.duration_minutes = duration

        # Generate a simple summary if none exists
        if not call_details.final_summary:
            call_details.final_summary = f"Patient {patient_context.first_name} {patient_context.last_name} completed pre-appointment screening."

        # --- Save Data to Database ---
        complaint_id = None
        history_id = None

        if patient_id:
            logger.info(f"Saving data for patient ID: {patient_id}")

            # Save complaint details if present
            if patient_context.complaint_details:
                complaint_id = await save_complaint_data(
                    patient_id, patient_context.complaint_details, call_id
                )
                logger.info(f"Saved complaint details, ID: {complaint_id}")

            # Save medical history if present
            if patient_context.medical_history:
                history_id = await save_medical_history(
                    patient_id, patient_context.medical_history, call_id
                )
                logger.info(f"Saved medical history, ID: {history_id}")
        else:
            logger.error("Missing patient ID in context. Cannot save patient data.")

        # --- Save Call Details ---
        success = await save_call_details(call_details)
        if success:
            logger.info(f"Successfully saved call details for call ID: {call_id}")

            # Update call with references to saved data
            await update_call_with_collected_data(
                call_id,
                complaint_id=complaint_id,
                history_id=history_id,
                final_summary=call_details.final_summary,
            )
        else:
            logger.error(f"Failed to save call details for call ID: {call_id}")

        # --- End Agent Session ---
        logger.info(f"Closing agent session for call ID: {call_id}...")
        try:
            await self.session.aclose()
            logger.info(f"Agent session closed successfully for call ID: {call_id}.")
        except Exception as e:
            logger.exception(
                f"Error occurred during agent session close for call ID: {call_id}.",
                exc_info=True,
            )
