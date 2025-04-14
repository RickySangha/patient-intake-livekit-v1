import logging
from typing import Annotated, Optional
from pydantic import Field
from livekit.agents import Agent, ChatContext
from livekit.agents.voice import RunContext
from livekit.agents.llm import function_tool

from custom_types.patient_context import PatientContext
from agents.medical_history import MedicalHistoryAgent
from custom_types.complaint_details import BackPainDetails

logger = logging.getLogger(__name__)


class BackPainAgent(Agent):
    def __init__(
        self, *, chat_ctx: Optional[ChatContext] = None, system_prompt: str
    ) -> None:
        self.SYSTEM_PROMPT = system_prompt
        CURRENT_TASK = """The patient has indicated back pain as their main concern. Acknowledge this and explain that you'll ask some specific questions about their back pain to help the doctor understand their situation better.

Carefully assess back pain characteristics. Ask about:
1. Location of pain (lower, middle, upper back)
2. Quality of pain (sharp, dull, shooting, etc.)
3. Severity (1-10 scale)
4. Duration and onset (sudden or gradual)
5. What makes it better or worse
6. Any associated symptoms (numbness, tingling, weakness)
Ask these questions conversationally, one at a time.

Call the handle_back_pain_details function when you have gathered all the information to proceed.

Monitor the patients responses for any emergency symptoms. For example, severe back pain with loss of bladder/bowel control, numbness in the groin area, progressive leg weakness, fever with severe back pain, or back pain from a major trauma/fall. If the patient is experiencing an emergency, call the handle_emergency function straight away, no need to ask any more questions."""
        super().__init__(
            instructions=f"{self.SYSTEM_PROMPT}\n\n{CURRENT_TASK}",
            chat_ctx=chat_ctx,
        )

    async def on_enter(self):
        logger.info("BackPainAgent entered")
        await self.session.say(
            "Okay, I see you're having back pain. Can you tell me where exactly on your back it hurts?"
        )

    @function_tool()
    async def handle_back_pain_details(
        self,
        context: RunContext[PatientContext],
        location: Annotated[
            str, Field(description="The specific location of the back pain.")
        ],
        quality: Annotated[str, Field(description="The quality of the back pain.")],
        severity: Annotated[str, Field(description="The severity of the back pain.")],
        duration: Annotated[str, Field(description="The duration of the back pain.")],
        onset: Annotated[
            str, Field(description="The onset of the back pain. (sudden or gradual)")
        ],
        aggravating_factors: Annotated[
            list[str], Field(description="What makes the back pain worse.")
        ],
        relieving_factors: Annotated[
            list[str], Field(description="What makes the back pain better.")
        ],
        associated_symptoms: Annotated[
            list[str], Field(description="Any associated symptoms of the back pain.")
        ],
    ) -> tuple[Agent, str]:
        """Call this function to save back pain details and proceed to the next step."""
        logger.info("Back pain details captured")

        try:
            back_pain_data = BackPainDetails(
                location=location,
                quality=quality,
                severity=severity,
                duration=duration,
                onset=onset,
                aggravating_factors=aggravating_factors,
                relieving_factors=relieving_factors,
                associated_symptoms=associated_symptoms,
            )
            context.userdata.complaint_details = back_pain_data

        except Exception as e:
            logger.exception(
                f"Error creating or setting BackPainDetails: {e}", exc_info=True
            )
            # Handle the error appropriately, maybe set complaint_details to None or an error state

        next_agent = MedicalHistoryAgent(
            chat_ctx=self.session.history, system_prompt=self.SYSTEM_PROMPT
        )
        return next_agent
