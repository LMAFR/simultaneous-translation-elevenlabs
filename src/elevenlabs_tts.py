from __future__ import annotations

import os
from typing import AsyncGenerator

import httpx


class ElevenLabsTTS:
    def __init__(
        self,
        api_key: str,
        voice_id: str,
        model_id: str = "eleven_turbo_v2_5",
        output_format: str = "pcm_16000",
        timeout_s: float = 30.0,
        chunk_size: int = 4096,
        use_sdk: bool | None = None,
    ):
        self.api_key = api_key
        self.voice_id = voice_id
        self.model_id = model_id
        self.output_format = output_format
        self.timeout_s = timeout_s
        self.chunk_size = chunk_size

        # By default, prefer the SDK if installed.
        if use_sdk is None:
            use_sdk = True
        self.use_sdk = use_sdk

    async def stream(self, text: str) -> AsyncGenerator[bytes, None]:
        """Yields audio bytes.

        Prefers the official ElevenLabs Python SDK (reliable full audio), while
        preserving the existing streaming interface used by latency_loop.

        If SDK is unavailable or fails, falls back to the HTTP streaming endpoint.
        """
        if self.use_sdk:
            try:
                # Imported lazily so the repo works without the SDK installed.
                from elevenlabs import ElevenLabs

                client = ElevenLabs(api_key=self.api_key)
                audio = client.text_to_speech.convert(
                    voice_id=self.voice_id,
                    model_id=self.model_id,
                    text=text,
                    output_format=self.output_format,
                )

                # SDK may return bytes or an iterator of bytes.
                if isinstance(audio, (bytes, bytearray)):
                    data = bytes(audio)
                    for i in range(0, len(data), self.chunk_size):
                        yield data[i : i + self.chunk_size]
                    return

                # Iterator of chunks
                if hasattr(audio, "__iter__"):
                    for chunk in audio:
                        if isinstance(chunk, (bytes, bytearray)) and chunk:
                            yield bytes(chunk)
                    return
            except Exception:
                # Fall back to HTTP streaming
                pass

        # HTTP streaming fallback
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}/stream"
        headers = {
            "xi-api-key": self.api_key,
            "accept": "application/octet-stream",
            "content-type": "application/json",
        }
        params = {
            "output_format": self.output_format,
            "optimize_streaming_latency": "4",
        }
        payload = {
            "text": text,
            "model_id": self.model_id,
        }

        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            async with client.stream("POST", url, headers=headers, params=params, json=payload) as r:
                r.raise_for_status()
                async for chunk in r.aiter_bytes(chunk_size=2048):
                    if chunk:
                        yield chunk


def resolve_voice_id() -> str:
    # Best-effort default voice if not provided.
    return os.getenv("ELEVENLABS_VOICE_ID") or "21m00Tcm4TlvDq8ikWAM"  # Rachel (commonly available)