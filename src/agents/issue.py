import logging
from typing import Annotated, Optional
from livekit.agents.voice import RunContext
from livekit.agents import Agent, ChatContext
from livekit.agents.llm import function_tool
from pydantic import Field

from custom_types.patient_context import PatientContext

# Import the new agents
from .chest_pain import ChestPainAgent
from .cough import CoughAgent
from .back_pain import BackPainAgent

logger = logging.getLogger(__name__)


class IssueAgent(Agent):
    def __init__(
        self, *, chat_ctx: Optional[ChatContext] = None, system_prompt: str
    ) -> None:
        self.SYSTEM_PROMPT = system_prompt
        CURRENT_TASK = """Your next task is to ask the patient for the main reason for their upcoming appointment or their primary health concern. 

Once you have identified the main issue Use the capture_main_issue tool to proceed. DO NOT give them funtion parameters as options to select. Let the patinet answer in their own words. 
        """
        super().__init__(
            instructions=f"{self.SYSTEM_PROMPT}\n\n{CURRENT_TASK}",
            chat_ctx=chat_ctx,
        )

    async def on_enter(self):
        logger.info("IssueAgent entered")
        self.session.generate_reply()  # Might not be needed if ConsentAgent passes a message

    @function_tool()
    async def capture_main_issue(
        self,
        context: RunContext[PatientContext],
        issue: Annotated[
            str,
            Field(
                description="The patient's main issue.",
                enum=["back_pain", "cough", "chest_pain"],
            ),
        ],
    ) -> tuple[Agent, str] | str:
        """Call this function to record the patient's main issue and determine the next step."""
        logger.info(f"Main issue captured: {issue}")
        context.userdata.main_issue = issue

        # Determine the next agent based on the issue
        if "back_pain" in issue.lower():
            next_agent = BackPainAgent(
                chat_ctx=context.session.history, system_prompt=self.SYSTEM_PROMPT
            )
        elif "cough" in issue.lower() or "respiratory" in issue.lower():
            next_agent = CoughAgent(
                chat_ctx=context.session.history, system_prompt=self.SYSTEM_PROMPT
            )
        elif "chest_pain" in issue.lower():
            next_agent = ChestPainAgent(
                chat_ctx=context.session.history, system_prompt=self.SYSTEM_PROMPT
            )
        # Add more conditions for other agents

        return next_agent
