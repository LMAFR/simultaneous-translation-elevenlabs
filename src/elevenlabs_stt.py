from __future__ import annotations

import asyncio
import base64
import json
import os
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Optional

import websockets


@dataclass
class TranscriptEvent:
    text: str
    is_final: bool
    received_time_ms: float
    utterance_id: str


class ElevenLabsRealtimeSTT:
    """ElevenLabs Realtime STT over WebSocket (per-utterance sessions).

    We intentionally open a fresh WS session per utterance. In practice this is much
    more reliable than trying to segment multiple utterances inside a single long-lived
    STT session (which can stop emitting partials after the first commit).

    Notes:
    - The official ElevenLabs Python SDK currently focuses on non-realtime STT APIs.
      For low-latency mic->text we keep using the realtime WebSocket endpoint.
    - We also provide an SDK-based non-realtime fallback (`transcribe_pcm16_sdk`) that
      can be used to debug correctness.
    """

    def __init__(
        self,
        api_key: str,
        input_lang: str = "en",
        sample_rate: int = 16000,
        model_id: str = "scribe_v2_realtime",
        timestamps_granularity: str = "word",
    ):
        self.api_key = api_key
        self.input_lang = input_lang
        self.sample_rate = sample_rate
        self.model_id = model_id
        self.timestamps_granularity = timestamps_granularity

        self.ws_url = os.getenv("ELEVENLABS_STT_WS_URL") or "wss://api.elevenlabs.io/v1/speech-to-text/realtime"

    async def run_one_utterance(
        self,
        audio_frames: "asyncio.Queue[bytes | None]",
        utterance_id: str,
    ) -> AsyncGenerator[TranscriptEvent, None]:
        """Run a dedicated STT websocket session for a single utterance.

        audio_frames yields raw PCM16 mono frames (bytes). The producer must send
        None as a sentinel to indicate end-of-utterance.
        """
        headers = {"xi-api-key": self.api_key}

        print(f"[STT] (utt) Connecting {utterance_id}: {self.ws_url}")

        async with websockets.connect(self.ws_url, additional_headers=headers, ping_interval=20) as ws:
            print(f"[STT] (utt) Connected {utterance_id}")

            out_q: asyncio.Queue[TranscriptEvent | None] = asyncio.Queue()

            async def recv_loop() -> None:
                try:
                    async for msg in ws:
                        t_ms = asyncio.get_running_loop().time() * 1000.0
                        ev = self._parse_event(msg, t_ms, utterance_id)
                        if ev is not None:
                            await out_q.put(ev)
                except Exception as e:
                    print(f"[STT] (utt) recv error {utterance_id}: {type(e).__name__}: {e}")
                finally:
                    await out_q.put(None)

            async def send_loop() -> None:
                bytes_per_sec = self.sample_rate * 2
                min_commit_bytes = int(bytes_per_sec * 0.32)
                uncommitted = bytearray()
                try:
                    while True:
                        frame = await audio_frames.get()
                        if frame is None:
                            # commit on boundary
                            if uncommitted and len(uncommitted) < min_commit_bytes:
                                uncommitted.extend(b"\x00" * (min_commit_bytes - len(uncommitted)))
                            if uncommitted:
                                await ws.send(
                                    json.dumps(
                                        {
                                            "message_type": "input_audio_chunk",
                                            "audio_base_64": base64.b64encode(bytes(uncommitted)).decode("ascii"),
                                            "commit": True,
                                            "sample_rate": self.sample_rate,
                                            **({"language_code": self.input_lang} if self.input_lang else {}),
                                        }
                                    )
                                )
                            # Important: close so recv_loop terminates promptly.
                            await ws.close()
                            break

                        uncommitted.extend(frame)
                        await ws.send(
                            json.dumps(
                                {
                                    "message_type": "input_audio_chunk",
                                    "audio_base_64": base64.b64encode(frame).decode("ascii"),
                                    "commit": False,
                                    "sample_rate": self.sample_rate,
                                    **({"language_code": self.input_lang} if self.input_lang else {}),
                                }
                            )
                        )
                except Exception as e:
                    print(f"[STT] (utt) send error {utterance_id}: {type(e).__name__}: {e}")

            recv_task = asyncio.create_task(recv_loop())
            send_task = asyncio.create_task(send_loop())

            try:
                while True:
                    ev = await out_q.get()
                    if ev is None:
                        break
                    yield ev
            finally:
                send_task.cancel()
                recv_task.cancel()

    async def transcribe_pcm16_sdk(self, pcm16_bytes: bytes) -> str:
        """SDK-based non-realtime STT for debugging correctness.

        This is not low-latency; it sends the full utterance at once.
        """
        try:
            from elevenlabs import ElevenLabs
        except Exception as e:
            raise RuntimeError(f"elevenlabs sdk not installed/importable: {e}")

        client = ElevenLabs(api_key=self.api_key)
        # The SDK STT API surface may vary by version; keep this best-effort.
        # If this raises, we'll adjust to the exact SDK version installed.
        audio_format = "pcm"  # best-effort hint
        resp = client.speech_to_text.convert(
            audio=pcm16_bytes,
            model_id=os.getenv("ELEVENLABS_STT_MODEL_ID", "scribe_v1"),
            language_code=self.input_lang or None,
            audio_format=audio_format,
            sample_rate=self.sample_rate,
        )
        # resp might be dict-like or have `.text`
        if isinstance(resp, dict):
            return str(resp.get("text") or "")
        return str(getattr(resp, "text", "") or "")

    def _parse_event(self, msg: Any, t_ms: float, utterance_id: str) -> Optional[TranscriptEvent]:
        if not isinstance(msg, str):
            return None
        try:
            obj = json.loads(msg)
        except Exception:
            return None

        mt = (obj.get("message_type") or obj.get("type") or "").lower()

        if mt in ("input_error", "error"):
            print(f"[STT] Error: {obj}")
            return None

        if mt == "session_started":
            return None

        if mt == "partial_transcript":
            text = obj.get("text") or ""
            if text:
                return TranscriptEvent(text=text, is_final=False, received_time_ms=t_ms, utterance_id=utterance_id)
            return None

        if mt in ("committed_transcript", "committed_transcript_with_timestamps"):
            text = obj.get("text") or ""
            if text:
                return TranscriptEvent(text=text, is_final=True, received_time_ms=t_ms, utterance_id=utterance_id)
            return None

        return None