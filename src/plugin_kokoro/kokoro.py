from __future__ import annotations

import asyncio
import json
import os
import weakref
from dataclasses import dataclass
from typing import Optional

import numpy as np
import websockets

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    tokenize,
    tts,
    utils,
)
from livekit import rtc

# Import logger similar to the Deepgram implementation
from livekit.agents.log import logger


@dataclass
class _TTSOptions:
    voice_id: str
    speed: float
    language: str
    sample_rate: int
    sentence_tokenizer: tokenize.SentenceTokenizer


class KokoroTTS(tts.TTS):
    def __init__(
        self,
        *,
        ws_url: str = "ws://localhost:8000/tts",
        voice_id: str = "af_heart",
        speed: float = 1.2,
        language: str = "en-us",
        sample_rate: int = 24000,
        sentence_tokenizer: tokenize.SentenceTokenizer = tokenize.basic.SentenceTokenizer(),
        api_key: str | None = None,
    ) -> None:
        """
        Create a new instance of Kokoro TTS.

        Args:
            ws_url (str): WebSocket URL of the Kokoro TTS service. Defaults to "ws://localhost:8000/tts".
            voice_id (str): Voice ID to use for synthesis. Defaults to "af_heart".
            speed (float): Speech rate multiplier. Defaults to 1.2.
            language (str): Language code. Defaults to "en-us".
            sample_rate (int): Sample rate of audio. Defaults to 24000.
            sentence_tokenizer (tokenize.SentenceTokenizer): Tokenizer for processing text.
                Defaults to basic SentenceTokenizer.
            api_key (str): Optional API key. If not provided, will look for KOKORO_API_KEY in environment.
        """
        super().__init__(
            capabilities=tts.TTSCapabilities(
                # Set streaming to False since Kokoro doesn't support word-by-word streaming
                # But we still support sentence-by-sentence streaming using the sentence tokenizer
                streaming=False
            ),
            sample_rate=sample_rate,
            num_channels=1,  # Mono audio
        )

        api_key = api_key or os.environ.get("KOKORO_API_KEY")
        if not api_key:
            logger.warning(
                "No Kokoro API key provided. Set KOKORO_API_KEY or provide api_key."
            )

        self._opts = _TTSOptions(
            voice_id=voice_id,
            speed=speed,
            language=language,
            sample_rate=sample_rate,
            sentence_tokenizer=sentence_tokenizer,
        )
        self._ws_url = ws_url
        self._api_key = api_key
        self._streams = weakref.WeakSet()
        self._pool = utils.ConnectionPool[websockets.WebSocketClientProtocol](
            connect_cb=self._connect_ws,
            close_cb=self._close_ws,
            max_session_duration=3600,  # 1 hour
            mark_refreshed_on_get=False,
        )

    async def _connect_ws(self) -> websockets.WebSocketClientProtocol:
        try:
            # WebSocket connection with proper header format for the websockets library
            # Note: The websockets library expects headers as a list of tuples, not a dict

            # Create headers list only if API key exists
            headers = None
            if self._api_key:
                headers = [("Authorization", f"Bearer {self._api_key}")]

            # Connect to WebSocket with proper parameters
            ws = await asyncio.wait_for(
                websockets.connect(
                    uri=self._ws_url,
                    extra_headers=headers,  # This will be passed correctly to the websockets library
                ),
                timeout=self._conn_options.timeout,
            )

            # Send initial settings
            settings = {
                "settings": {
                    "voice_id": self._opts.voice_id,
                    "speed": self._opts.speed,
                    "language": self._opts.language,
                    "sample_rate": self._opts.sample_rate,
                }
            }
            await ws.send(json.dumps(settings))

            # Wait for settings confirmation
            response = await asyncio.wait_for(
                ws.recv(), timeout=self._conn_options.timeout
            )
            data = json.loads(response)
            if "settings_updated" not in data:
                raise APIConnectionError(f"Failed to set initial settings: {data}")

            return ws
        except asyncio.TimeoutError as e:
            raise APITimeoutError() from e
        except Exception as e:
            logger.error(f"Connection error: {str(e)}")
            raise APIConnectionError() from e

    async def _close_ws(self, ws: websockets.WebSocketClientProtocol):
        try:
            await ws.close()
        except Exception as e:
            logger.warning(f"Error closing WebSocket: {e}")

    def update_options(
        self,
        *,
        voice_id: str | None = None,
        speed: float | None = None,
        language: str | None = None,
        sample_rate: int | None = None,
    ) -> None:
        """
        Update the TTS options. Changes will apply to new connections.

        Args:
            voice_id (str): Voice ID to use.
            speed (float): Speech rate multiplier.
            language (str): Language code.
            sample_rate (int): Sample rate of audio.
        """
        if voice_id is not None:
            self._opts.voice_id = voice_id
        if speed is not None:
            self._opts.speed = speed
        if language is not None:
            self._opts.language = language
        if sample_rate is not None:
            self._opts.sample_rate = sample_rate
            # Update the sample rate for the TTS instance
            self._sample_rate = sample_rate

        # Invalidate the pool to get a new connection with updated options
        self._pool.invalidate()

    def synthesize(
        self,
        text: str,
        *,
        conn_options: Optional[APIConnectOptions] = None,
    ) -> "ChunkedStream":
        """
        Synthesize speech from text in a single request.

        Args:
            text (str): Text to synthesize.
            conn_options (Optional[APIConnectOptions]): Connection options.

        Returns:
            ChunkedStream: Stream of synthesized audio.
        """
        return ChunkedStream(
            tts=self,
            input_text=text,
            ws_url=self._ws_url,
            api_key=self._api_key,
            conn_options=conn_options,
            opts=self._opts,
        )

    def stream(
        self, *, conn_options: Optional[APIConnectOptions] = None
    ) -> "SynthesizeStream":
        """
        Create a streaming synthesis stream.

        Args:
            conn_options (Optional[APIConnectOptions]): Connection options.

        Returns:
            SynthesizeStream: Stream for streaming synthesis.
        """
        stream = SynthesizeStream(
            tts=self,
            pool=self._pool,
            opts=self._opts,
        )
        self._streams.add(stream)
        return stream

    def prewarm(self) -> None:
        """Prewarm the connection pool."""
        self._pool.prewarm()

    async def aclose(self) -> None:
        """Close all resources."""
        for stream in list(self._streams):
            await stream.aclose()
        self._streams.clear()
        await self._pool.aclose()
        await super().aclose()


class ChunkedStream(tts.ChunkedStream):
    def __init__(
        self,
        *,
        tts: KokoroTTS,
        ws_url: str,
        api_key: str,
        input_text: str,
        opts: _TTSOptions,
        conn_options: Optional[APIConnectOptions] = None,
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._opts = opts
        self._ws_url = ws_url
        self._api_key = api_key

    async def _run(self) -> None:
        request_id = utils.shortuuid()

        try:
            # Connect to the WebSocket using the correct format for headers
            headers = None
            if self._api_key:
                headers = [("Authorization", f"Bearer {self._api_key}")]

            async with websockets.connect(
                uri=self._ws_url,
                close_timeout=5.0,
                ping_interval=20.0,
                ping_timeout=10.0,
                max_size=10 * 1024 * 1024,  # 10MB max
            ) as ws:
                # Send initial settings
                settings = {
                    "settings": {
                        "voice_id": self._opts.voice_id,
                        "speed": self._opts.speed,
                        "language": self._opts.language,
                        "sample_rate": self._opts.sample_rate,
                    }
                }
                await ws.send(json.dumps(settings))

                # Wait for settings confirmation
                response = await asyncio.wait_for(
                    ws.recv(), timeout=self._conn_options.timeout
                )
                data = json.loads(response)
                if "settings_updated" not in data:
                    raise APIConnectionError(f"Failed to set initial settings: {data}")

                # Send the text request
                await ws.send(
                    json.dumps({"text": self._input_text, "request_id": request_id})
                )

                # Process WebSocket responses
                while True:
                    message = await asyncio.wait_for(
                        ws.recv(), timeout=self._conn_options.timeout
                    )

                    if isinstance(message, str):
                        try:
                            data = json.loads(message)
                            if data.get("request_id") != request_id:
                                continue

                            if "event" in data:
                                if data["event"] == "tts_started":
                                    continue
                                elif data["event"] in [
                                    "tts_completed",
                                    "tts_stopped",
                                ]:  # Support both event names
                                    break
                                elif "error" in data:
                                    raise APIStatusError(
                                        message=data["error"],
                                        status_code=500,
                                        request_id=request_id,
                                        body=data,
                                    )
                        except json.JSONDecodeError:
                            logger.warning(f"Received invalid JSON: {message[:100]}")
                    elif isinstance(message, bytes):
                        # Convert bytes to audio frame
                        audio_data = np.frombuffer(message, dtype=np.int16)
                        frame = rtc.AudioFrame(
                            data=audio_data.tobytes(),
                            sample_rate=self._opts.sample_rate,
                            num_channels=1,
                            samples_per_channel=len(audio_data),
                        )
                        self._event_ch.send_nowait(
                            tts.SynthesizedAudio(request_id=request_id, frame=frame)
                        )

        except asyncio.TimeoutError as e:
            raise APITimeoutError() from e
        except websockets.exceptions.ConnectionClosed as e:
            logger.error(f"WebSocket connection closed unexpectedly: {str(e)}")
            raise APIConnectionError("WebSocket connection closed unexpectedly") from e
        except Exception as e:
            logger.error(f"Error in ChunkedStream._run: {str(e)}")
            raise APIConnectionError() from e


class SynthesizeStream(tts.SynthesizeStream):
    def __init__(
        self,
        *,
        tts: KokoroTTS,
        opts: _TTSOptions,
        pool: utils.ConnectionPool[websockets.WebSocketClientProtocol],
    ):
        super().__init__(tts=tts)
        self._opts = opts
        self._pool = pool
        self._segments_ch = utils.aio.Chan[tokenize.SentenceStream]()
        self._active_request_id = None

    async def _run(self) -> None:
        request_id = utils.shortuuid()

        @utils.log_exceptions(logger=logger)
        async def _tokenize_input():
            # Converts incoming text into SentenceStreams and sends them into _segments_ch
            # Unlike word-level tokenization, we need to buffer until we have complete sentences
            sentence_stream = None
            text_buffer = ""

            async for input in self._input_ch:
                if isinstance(input, str):
                    # Accumulate text in buffer
                    text_buffer += input

                    # Create sentence stream if needed
                    if sentence_stream is None:
                        sentence_stream = self._opts.sentence_tokenizer.stream()
                        self._segments_ch.send_nowait(sentence_stream)

                    # Process the buffer to extract complete sentences
                    sentences = self._opts.sentence_tokenizer.tokenize(text_buffer)

                    # If we have complete sentences, push them to the stream
                    if (
                        len(sentences) > 1
                    ):  # More than one means we have at least one complete sentence
                        for sentence in sentences[:-1]:
                            if sentence.strip():
                                sentence_stream.push_text(
                                    sentence + " "
                                )  # Add space after sentence

                        # Keep the last (possibly incomplete) sentence in the buffer
                        text_buffer = sentences[-1] if sentences else ""

                elif isinstance(input, self._FlushSentinel):
                    # On flush, send any remaining text as a complete sentence
                    if text_buffer.strip() and sentence_stream:
                        sentence_stream.push_text(text_buffer)
                        text_buffer = ""

                    if sentence_stream:
                        sentence_stream.end_input()
                    sentence_stream = None

            # Handle any remaining text at the end
            if text_buffer.strip() and sentence_stream:
                sentence_stream.push_text(text_buffer)

            self._segments_ch.close()

        @utils.log_exceptions(logger=logger)
        async def _run_segments():
            async for sentence_stream in self._segments_ch:
                await self._run_ws(sentence_stream, request_id)

        tasks = [
            asyncio.create_task(_tokenize_input()),
            asyncio.create_task(_run_segments()),
        ]
        try:
            await asyncio.gather(*tasks)
        except asyncio.TimeoutError as e:
            raise APITimeoutError() from e
        except Exception as e:
            logger.error(f"Error in SynthesizeStream._run: {str(e)}")
            raise APIConnectionError() from e
        finally:
            await utils.aio.gracefully_cancel(*tasks)

    async def _run_ws(self, sentence_stream: tokenize.SentenceStream, request_id: str):
        segment_id = utils.shortuuid()

        async def send_task(ws: websockets.WebSocketClientProtocol):
            # Collect complete sentences from the sentence stream
            async for sentence in sentence_stream:
                # Skip empty sentences
                if not sentence.token.strip():
                    continue

                # We got a complete sentence - Kokoro processes at sentence level
                complete_sentence = sentence.token.strip()

                # Generate a unique ID for this sentence request
                segment_request_id = f"{request_id}_{segment_id}_{utils.shortuuid()}"
                self._active_request_id = segment_request_id

                # Send the complete sentence to Kokoro
                speak_msg = {
                    "text": complete_sentence,
                    "request_id": segment_request_id,
                }
                self._mark_started()

                # Log what we're sending to help with debugging
                logger.debug(
                    f"Sending sentence to Kokoro: {complete_sentence[:50]}{'...' if len(complete_sentence) > 50 else ''}"
                )

                await ws.send(json.dumps(speak_msg))

                # Wait a small amount to ensure proper message ordering on the server
                await asyncio.sleep(0.05)

        async def recv_task(ws: websockets.WebSocketClientProtocol):
            emitter = tts.SynthesizedAudioEmitter(
                event_ch=self._event_ch,
                request_id=request_id,
                segment_id=segment_id,
            )

            while True:
                try:
                    message = await asyncio.wait_for(
                        ws.recv(), timeout=self._conn_options.timeout
                    )

                    if isinstance(message, str):
                        try:
                            data = json.loads(message)
                            req_id = data.get("request_id", "")
                            if req_id and not req_id.startswith(
                                f"{request_id}_{segment_id}"
                            ):
                                continue

                            if "event" in data:
                                if data["event"] == "tts_started":
                                    continue
                                elif data["event"] in [
                                    "tts_completed",
                                    "tts_stopped",
                                ]:  # Support both event names
                                    emitter.flush()
                                    break
                                elif "error" in data:
                                    raise APIStatusError(
                                        message=data["error"],
                                        status_code=500,
                                        request_id=request_id,
                                        body=data,
                                    )
                        except json.JSONDecodeError:
                            logger.warning(f"Received invalid JSON: {message[:100]}")
                    elif isinstance(message, bytes):
                        # Convert bytes to audio frame
                        audio_data = np.frombuffer(message, dtype=np.int16)
                        frame = rtc.AudioFrame(
                            data=audio_data.tobytes(),
                            sample_rate=self._opts.sample_rate,
                            num_channels=1,
                            samples_per_channel=len(audio_data),
                        )
                        emitter.push(frame)
                except asyncio.TimeoutError:
                    logger.warning(
                        f"Timeout waiting for WebSocket response in segment {segment_id}"
                    )
                    break
                except Exception as e:
                    logger.error(f"Error in recv_task: {str(e)}")
                    raise

        # For SynthesizeStream, we'll avoid using connection pool's context manager
        # and handle websocket manually to work around extra_headers issue
        try:
            ws = await self._pool.get()

            tasks = [
                asyncio.create_task(send_task(ws)),
                asyncio.create_task(recv_task(ws)),
            ]

            try:
                await asyncio.gather(*tasks)
            except asyncio.TimeoutError as e:
                raise APITimeoutError() from e
            except websockets.exceptions.ConnectionClosed as e:
                logger.error(f"WebSocket connection closed unexpectedly: {e}")
                raise APIConnectionError(
                    "WebSocket connection closed unexpectedly"
                ) from e
            except Exception as e:
                logger.error(f"Error in _run_ws: {e}")
                raise APIConnectionError() from e
            finally:
                self._active_request_id = None
                await utils.aio.gracefully_cancel(*tasks)
                # Return the websocket to the pool
                await self._pool.put(ws)
        except Exception as e:
            logger.error(f"Error getting WebSocket from pool: {e}")
            raise APIConnectionError("Failed to get WebSocket connection") from e

    async def interrupt(self) -> None:
        """Interrupt the current synthesis."""
        if self._active_request_id:
            try:
                # Get a WebSocket from the pool
                ws = await self._pool.get()
                try:
                    # Send interrupt command
                    await ws.send(
                        json.dumps(
                            {
                                "event": "interrupt",
                                "request_id": self._active_request_id,
                            }
                        )
                    )

                    # Wait for confirmation of interruption
                    try:
                        response = await asyncio.wait_for(ws.recv(), timeout=1.0)
                        data = json.loads(response)
                        if data.get("event") != "interrupted":
                            logger.warning(f"Unexpected interruption response: {data}")
                    except asyncio.TimeoutError:
                        logger.warning("Timeout waiting for interruption confirmation")
                    except Exception as e:
                        logger.warning(f"Error processing interrupt confirmation: {e}")

                finally:
                    # Return the WebSocket to the pool
                    await self._pool.put(ws)
            except Exception as e:
                logger.warning(f"Error during interrupt: {e}")
