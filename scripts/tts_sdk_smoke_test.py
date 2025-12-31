from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

# Allow `python scripts/tts_sdk_smoke_test.py` to import from /src
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.audio_io import JitterBufferPlayer


def _iter_bytes(audio) -> bytes:
    """Best-effort conversion of SDK 'audio' output to raw bytes."""
    if audio is None:
        return b""
    if isinstance(audio, (bytes, bytearray)):
        return bytes(audio)
    # SDK may return an iterator / generator of bytes
    if hasattr(audio, "__iter__"):
        out = bytearray()
        for chunk in audio:
            if isinstance(chunk, (bytes, bytearray)):
                out.extend(chunk)
        return bytes(out)
    return b""


async def main() -> None:
    load_dotenv()

    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        raise SystemExit("Missing ELEVENLABS_API_KEY (see .env.example)")

    voice_id = os.getenv("ELEVENLABS_VOICE_ID") or "21m00Tcm4TlvDq8ikWAM"
    model_id = os.getenv("ELEVENLABS_MODEL_ID") or "eleven_turbo_v2_5"

    text = os.getenv(
        "TTS_TEST_TEXT",
        "Hello. This is an ElevenLabs SDK text to speech test. It should read this entire sentence.",
    )

    print("WARNING: Use headphones to avoid feedback loops.")
    print(f"Voice: {voice_id}")
    print(f"Model: {model_id}")
    print(f"Text: {text}")

    # Import here so the repo still imports without the SDK installed
    try:
        from elevenlabs import ElevenLabs
    except Exception as e:
        raise SystemExit(f"Failed to import elevenlabs SDK: {e}")

    client = ElevenLabs(api_key=api_key)

    # Prefer decoded PCM to avoid mp3 decoding dependencies.
    # If this fails for your account, set ELEVENLABS_OUTPUT_FORMAT=mp3_44100_128 and I'll add decoding.
    output_format = os.getenv("ELEVENLABS_OUTPUT_FORMAT", "pcm_16000")

    try:
        audio = client.text_to_speech.convert(
            voice_id=voice_id,
            model_id=model_id,
            text=text,
            output_format=output_format,
        )
    except TypeError:
        # Older SDK signature fallback
        audio = client.text_to_speech.convert(
            voice=voice_id,
            model=model_id,
            text=text,
            output_format=output_format,
        )

    pcm = _iter_bytes(audio)
    print(f"SDK: received bytes: {len(pcm)}")

    if output_format.startswith("pcm_16000"):
        approx_s = len(pcm) / (16000 * 2)
        print(f"SDK: approx duration (pcm16@16k): {approx_s:.2f}s")

    # Play
    player = JitterBufferPlayer(sample_rate=16000, channels=1, dtype="int16", prebuffer_ms=0)
    await player.start()

    # If pcm is empty, bail
    if not pcm:
        print("ERROR: SDK returned no audio bytes")
        await asyncio.sleep(0.2)
        await player.close()
        return

    # Enqueue as a few chunks to mimic streaming
    chunk_size = 4096
    for i in range(0, len(pcm), chunk_size):
        await player.enqueue(pcm[i : i + chunk_size])

    t_play = await player.wait_first_play(timeout_s=5.0)
    if t_play is None:
        print("WARNING: audio never started playing (check output device)")
        await asyncio.sleep(2.0)
    else:
        # Wait enough time to finish
        await asyncio.sleep(max(2.0, len(pcm) / (16000 * 2) + 1.0))

    await player.close()


if __name__ == "__main__":
    asyncio.run(main())
