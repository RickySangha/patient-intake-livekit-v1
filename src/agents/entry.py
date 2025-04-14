import logging
from typing import Annotated
from livekit.agents import Agent
from livekit.agents.voice import RunContext
from livekit.agents.llm import function_tool
from pydantic import Field
from custom_types.patient_context import PatientContext

from .issue import IssueAgent

logger = logging.getLogger(__name__)

# TODO: We can add in additional context that will be useful for the agent in the system prompt.
# Questions: what additional context is useful?
# -Current time?
# -Patient history?
# -Custom doctor instructions?
# -Custom clinic instructions?
# -Any other context?


class EntryAgent(Agent):
    def __init__(
        self,
        agent_name: str,
        clinic_name: str,
        doctor_name: str,
        patient_first_name: str,
        patient_last_name: str,
        patient_age: int,
        patient_gender: str,
    ) -> None:
        self.patient_first_name = patient_first_name
        self.patient_last_name = patient_last_name
        self.SYSTEM_PROMPT = f"""You are {agent_name}, a professional medical assistant calling on behalf of {doctor_name} from {clinic_name}. Your job is to collect important information from patients before their upcoming appointments. This helps the doctor understand the patient's situation and provide better care.

You should:
 - Speak in a warm, professional manner using natural language
 - Keep your responses concise and focused on collecting information
 - No need to repeat the users responses back to them unless you are unclear. A short 'thank you' or 'got it' is sufficient.
 - Ask one question at a time
 - Be empathetic but professional
 - If you dont understand the patient's response, ask for clarification

Important guidelines:
 - DO NOT provide medical advice or diagnoses
 - If the patient seems distressed or reports severe symptoms, acknowledge their concern and note it
 - If the patient refuses to continue or wants to speak directly with the doctor, respect their decision and end the call gracefully
 - If the patient gets off topic, gently bring them back to the current questions
 - DONOT SAY ANYTHING WHEN RUNNING FUNCTIONS. JUST CALL THE FUNCTION.
 - ALl FUNCTION PARAMETERS ARE REQUIRED. UNLESS SPECIFIED OTHERWISE.
 - DO NOT SUMMARISE THE INFORMATION YOU HAVE COLLECTED BACK TO THE PATIENT, UNLESS SPECIFIED IN YOUR CURRENT TASK OTHERWISE.

Below is information on the patient you are calling:
Name: {patient_first_name} {patient_last_name}
Age: {patient_age}
Gender: {patient_gender}

Your current task is:
 """

        CURRENT_TASK = """You're first task is to greet the patient and confirm you are speaking with the correct person. If not ask to be transferred to the correct person. Then explain the purpose of the call and ask if they consent to proceed with the pre-appointment phone screening. Use the available tools based on their answer."""
        super().__init__(
            instructions=f"{self.SYSTEM_PROMPT}\n\n{CURRENT_TASK}",
        )

    async def on_enter(self):
        logger.info("EntryAgent entered")
        await self.session.say(
            f"Hello, I'm calling from your doctor's office for a quick pre-appointment check-in. Am I speaking with {self.patient_first_name} {self.patient_last_name}?"
        )
        # await self.session.generate_reply(
        #     instructions="Greet the patient and confirm you are talking with the correcy person. If not ask to be transferred to the correct person."
        # )

    @function_tool()
    async def handle_consent(
        self,
        context: RunContext[PatientContext],
        consent: Annotated[
            bool,
            Field(
                description="Whether the patient has given consent to proceed with the pre-appointment phone screening."
            ),
        ],
    ) -> tuple[Agent, str]:
        """Call this function ONLY WHEN the patient has explicity given or refused consent to proceed with the pre-appointment phone screening."""

        if consent:
            logger.info("Consent given by patient.")
            context.userdata.call_details.consent_given = True
            new_chat_ctx = self.chat_ctx
            next_agent = IssueAgent(
                chat_ctx=new_chat_ctx, system_prompt=self.SYSTEM_PROMPT
            )

            return next_agent
        else:
            logger.info("Consent refused by patient.")
            context.userdata.call_details.consent_given = False
            await self.session.say(
                "Okay, I understand. We will not proceed with the screening. Please contact the office if you have any questions. Goodbye."
            )
            await self.session.aclose()
