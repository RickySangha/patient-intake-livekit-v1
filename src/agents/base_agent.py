import asyncio
import logging
from typing import AsyncIterable, List, cast, Optional
from custom_types.patient_context import PatientContext

# Necessary imports from livekit
from livekit.agents import Agent
from livekit.agents.llm import ChatContext

logger = logging.getLogger(__name__)

# TODO: we can create a base agent that has overriden llm_node and tts_node and then have the other agents inherit from it. This will allow us to add emotion to the tts_node. Will only work with openai tts. We should also add custom pernounciations here, depending on the TTS we use.


class BaseAgent(Agent):
    """
    Base agent class that overrides llm_node and tts_node to add emotion if using openai tts.
    """

    def __init__(
        self,
        next_task: str,
        chat_ctx: Optional[ChatContext] = None,
    ) -> None:
        SYSTEM_PROMPT = """You are Amy, a professional medical assistant calling on behalf of Dr. Smith from Surrey Medical Centre. Your job is to collect important information from patients before their upcoming appointments.

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
 - ONLY CALL FUNCTIONS THAT ARE CURRENTLY AVAILABLE. DO NOT CALL FUNCTIONS THAT ARE NOT AVAILABLE OR FUNCTIONS YOU HAVE CALLED PREVIOUSLY. ONLY CALL EACH FUNCTION ONCE."""
        super().__init__(
            instructions=f"{SYSTEM_PROMPT}\n\n{next_task}",
            chat_ctx=chat_ctx,
        )

    # async def llm_node(
    #     self,
    #     chat_ctx: ChatContext,
    #     tools: list[FunctionTool],
    #     model_settings: ModelSettings,
    # ):
    #     llm = cast(openai.LLM, self.llm)
    #     tool_choice = model_settings.tool_choice if model_settings else NOT_GIVEN
    #     async with llm.chat(
    #         chat_ctx=chat_ctx,
    #         tools=tools,
    #         tool_choice=tool_choice,
    #         response_format=ResponseEmotion,
    #     ) as stream:
    #         async for chunk in stream:
    #             yield chunk

    # async def tts_node(self, text: AsyncIterable[str], model_settings: ModelSettings):
    #     instruction_updated = False

    #     def output_processed(resp: ResponseEmotion):
    #         nonlocal instruction_updated
    #         if (
    #             resp.get("voice_instructions")
    #             and resp.get("response")
    #             and not instruction_updated
    #         ):
    #             # when the response isn't empty, we can assume voice_instructions is complete.
    #             # (if the LLM sent the fields in the right order)
    #             instruction_updated = True
    #             logger.info(
    #                 f"Applying TTS instructions before generating response audio: "
    #                 f'"{resp["voice_instructions"]}"'
    #             )

    #             tts = cast(openai.TTS, self.tts)
    #             tts.update_options(instructions=resp["voice_instructions"])

    #     return super().tts_node(
    #         process_structured_output(text, callback=output_processed), model_settings
    #     )

    # async def transcription_node(
    #     self, text: AsyncIterable[str], model_settings: ModelSettings
    # ):
    #     async for delta in process_structured_output(text):
    #         yield delta
