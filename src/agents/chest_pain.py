import logging
from typing import Annotated, Optional
from pydantic import Field
from livekit.agents import Agent, ChatContext
from livekit.agents.voice import RunContext
from livekit.agents.llm import function_tool

from custom_types.patient_context import PatientContext
from agents.medical_history import MedicalHistoryAgent
from custom_types.complaint_details import ChestPainDetails

logger = logging.getLogger(__name__)


class ChestPainAgent(Agent):
    def __init__(
        self, *, chat_ctx: Optional[ChatContext] = None, system_prompt: str
    ) -> None:
        self.SYSTEM_PROMPT = system_prompt
        CURRENT_TASK = """The patient has indicated they are experiencing chest pain. Your task is to ask clarifying questions about the chest pain. Carefully assess chest pain characteristics. Ask about:
1. Location and radiation of pain
2. Quality of pain (sharp, dull, pressure, etc.)
3. Severity (1-10 scale)
4. Associated symptoms
5. What makes it better or worse
Ask these questions conversationally, one at a time.

Call the handle_chest_pain_details function when you have gathered all the information to proceed.

Monitor the patients responses for any emergency symptoms. For example, severe very severe chest pain, shortness of breath, nausea, dizziness, or pain in the jaw, neck, or arms. If the patient is experiencing an emergency, call the handle_emergency function straight away, no need to ask any more questions."""
        super().__init__(
            instructions=f"{self.SYSTEM_PROMPT}\n\n{CURRENT_TASK}",
            chat_ctx=chat_ctx,
        )

    async def on_enter(self):
        logger.info("ChestPainAgent entered")
        # Ask the first question
        await self.session.say(
            "Okay, I understand you're experiencing chest pain. Can you tell me where exactly the pain is located?"
        )

    @function_tool()
    async def handle_chest_pain_details(
        self,
        context: RunContext[PatientContext],
        location: Annotated[
            str,
            Field(
                description="The location of the chest pain described by the patient."
            ),
        ],
        quality: Annotated[
            str,
            Field(
                description="The quality of the chest pain described by the patient."
            ),
        ],
        severity: Annotated[
            str,
            Field(
                description="The severity of the chest pain described by the patient."
            ),
        ],
        associated_symptoms: Annotated[
            list[str],
            Field(
                description="The associated symptoms of the chest pain described by the patient."
            ),
        ],
        relieving_factors: Annotated[
            list[str],
            Field(description="What makes the chest pain better."),
        ],
        aggravating_factors: Annotated[
            list[str],
            Field(description="What makes the chest pain worse."),
        ],
        is_emergency: Annotated[
            bool,
            Field(description="Whether the patient is experiencing an emergency."),
        ],
        emergency_reason: Annotated[
            str,
            Field(description="The reason for the emergency."),
        ],
        # Add more parameters here as needed (e.g., duration)
    ) -> tuple[Agent, str]:
        """Call this function to save chest pain details and proceed to the next step."""
        logger.info("Chest pain details captured")

        try:
            chest_pain_details = ChestPainDetails(
                location=location,
                quality=quality,
                severity=severity,
                associated_symptoms=associated_symptoms,
                relieving_factors=relieving_factors,
                aggravating_factors=aggravating_factors,
                is_emergency=is_emergency,
                emergency_reason=emergency_reason,
            )
            context.userdata.complaint_details = chest_pain_details
        except Exception as e:
            logger.exception(
                f"Error creating or setting ChestPainDetails: {e}", exc_info=True
            )

        next_agent = MedicalHistoryAgent(
            chat_ctx=self.session.history, system_prompt=self.SYSTEM_PROMPT
        )
        return next_agent
