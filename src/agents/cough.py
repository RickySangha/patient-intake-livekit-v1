import logging
from typing import Annotated, Optional
from pydantic import Field
from livekit.agents import Agent, ChatContext
from livekit.agents.voice import RunContext
from livekit.agents.llm import function_tool

from custom_types.patient_context import PatientContext
from agents.medical_history import MedicalHistoryAgent
from custom_types.complaint_details import CoughDetails

logger = logging.getLogger(__name__)


class CoughAgent(Agent):
    def __init__(
        self, *, chat_ctx: Optional[ChatContext] = None, system_prompt: str
    ) -> None:
        self.SYSTEM_PROMPT = system_prompt
        CURRENT_TASK = """The patient has indicated respiratory symptoms as their main concern. Acknowledge this and explain that you'll ask some specific questions about their symptoms to help the doctor understand their situation better.

Carefully assess respiratory symptoms. Ask about:
1. Type of symptoms (cough, shortness of breath, wheezing, etc.)
2. Duration of symptoms
3. Any triggers or patterns they've noticed
4. Associated symptoms (fever, chest pain, fatigue)
5. Severity of symptoms and impact on daily activities
Ask these questions conversationally, one at a time.

Call the handle_cough_details function when you have gathered all the information to proceed.

Monitor the patients responses for any emergency symptoms. For example, severe difficulty breathing, coughing up blood, severe chest pain, high fever with confusion, or inability to speak in full sentences due to breathlessness. If the patient is experiencing an emergency, call the handle_emergency function straight away, no need to ask any more questions.
"""
        super().__init__(
            instructions=f"{self.SYSTEM_PROMPT}\n\n{CURRENT_TASK}",
            chat_ctx=chat_ctx,
        )

    async def on_enter(self):
        logger.info("CoughAgent entered")
        await self.session.say(
            "Okay, you mentioned having a cough. How long have you had it?"
        )

    @function_tool()
    async def handle_cough_details(
        self,
        context: RunContext[PatientContext],
        duration: Annotated[
            str, Field(description="How long the patient has had the cough.")
        ],
        cough_type: Annotated[
            str, Field(description="Type of cough (e.g., dry, productive, hacking).")
        ],
        triggers_patterns: Annotated[
            list[str],
            Field(description="Any triggers or patterns noticed for the cough."),
        ],
        associated_symptoms: Annotated[
            list[str],
            Field(
                description="Associated symptoms like fever, chest pain, fatigue, shortness of breath."
            ),
        ],
        severity: Annotated[
            str,
            Field(
                description="Severity of the cough and its impact on daily activities."
            ),
        ],
    ) -> tuple[Agent, str]:
        """Call this function to save cough details and proceed to the next step."""
        logger.info("Cough details captured")

        try:
            cough_details = CoughDetails(
                duration=duration,
                cough_type=cough_type,
                triggers_patterns=triggers_patterns,
                associated_symptoms=associated_symptoms,
                severity=severity,
            )
            context.userdata.complaint_details = cough_details
        except Exception as e:
            logger.exception(
                f"Error creating or setting CoughDetails: {e}", exc_info=True
            )

        # Transition to the next agent
        next_agent = MedicalHistoryAgent(
            chat_ctx=self.session.history, system_prompt=self.SYSTEM_PROMPT
        )
        return next_agent
