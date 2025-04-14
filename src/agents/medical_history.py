import logging
from typing import Annotated, Optional, List
from pydantic import Field
from livekit.agents import Agent, ChatContext
from livekit.agents.voice import RunContext
from livekit.agents.llm import function_tool
from custom_types.patient_context import PatientContext
from .end_call import EndCallAgent

logger = logging.getLogger(__name__)


class MedicalHistoryAgent(Agent):
    def __init__(
        self, *, chat_ctx: Optional[ChatContext] = None, system_prompt: str
    ) -> None:
        self.SYSTEM_PROMPT = system_prompt
        CURRENT_TASK = """The patient has already provided details about their main concern. Your task is now to ask general medical history questions. 

Ask about:
1. Past medical conditions (e.g., diabetes, high blood pressure)
2. Previous surgeries
3. Current medications (including dosage and frequency if possible)
4. Known allergies (medications, food, environmental)
Ask these questions conversationally, one at a time.

Call the capture_medical_history function when you have gathered all the information to proceed.
"""
        super().__init__(
            instructions=f"{self.SYSTEM_PROMPT}\n\n{CURRENT_TASK}",
            chat_ctx=chat_ctx,
        )

    async def on_enter(self):
        logger.info("MedicalHistoryAgent entered")
        # Start with the first general question
        await self.session.say(
            "Thanks for providing those details. Now, I'd like to ask a few general questions about your medical history. Do you have any past medical conditions I should be aware of, like diabetes or high blood pressure?"
        )

    @function_tool()
    async def capture_medical_history(
        self,
        context: RunContext[PatientContext],
        past_conditions: Annotated[
            List[str],
            Field(description="List of past medical conditions the patient has."),
        ],
        surgeries: Annotated[
            List[str],
            Field(description="List of previous surgeries the patient has had."),
        ],
        medications: Annotated[
            List[str],
            Field(description="List of current medications the patient is taking."),
        ],
        allergies: Annotated[
            List[str], Field(description="List of known allergies the patient has.")
        ],
    ) -> str:
        """Call this function to capture the patient's general medical history."""
        logger.info("General medical history captured.")

        # Store the details in userdata (or the appropriate context object)
        context.userdata.medical_history = {
            "past_conditions": past_conditions,
            "surgeries": surgeries,
            "medications": medications,
            "allergies": allergies,
        }

        next_agent = EndCallAgent(
            chat_ctx=self.session.history, system_prompt=self.SYSTEM_PROMPT
        )
        return next_agent
