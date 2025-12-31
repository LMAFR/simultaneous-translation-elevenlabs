from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import sounddevice as sd
from dotenv import load_dotenv

# Allow `python scripts/tts_smoke_test.py` to import from /src
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.audio_io import JitterBufferPlayer
from src.elevenlabs_tts import ElevenLabsTTS, resolve_voice_id


async def main() -> None:
    load_dotenv()

    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        raise SystemExit("Missing ELEVENLABS_API_KEY (see .env.example)")

    voice_id = resolve_voice_id()

    text = os.getenv(
        "TTS_TEST_TEXT",
        "Hello. This is a text to speech smoke test. If you can hear this clearly, "
        "then streaming TTS and playback are working correctly.",
    )

    print("WARNING: Use headphones to avoid feedback loops.")
    print(f"Voice: {voice_id}")
    print(f"Text: {text}")

    # Optional output device selection
    out_dev = os.getenv("OUTPUT_DEVICE")
    if out_dev is not None:
        try:
            out_dev_idx = int(out_dev)
            sd.default.device = (sd.default.device[0], out_dev_idx)
            print(f"Using OUTPUT_DEVICE={out_dev_idx}")
        except Exception:
            print(f"WARNING: invalid OUTPUT_DEVICE={out_dev}; ignoring")

    print("Available audio devices:")
    try:
        for i, d in enumerate(sd.query_devices()):
            if d.get("max_output_channels", 0) > 0:
                print(f"  [{i}] {d['name']} (out_ch={d['max_output_channels']})")
    except Exception as e:
        print(f"WARNING: could not list devices: {e}")

    player = JitterBufferPlayer(
        sample_rate=16000,
        channels=1,
        dtype="int16",
        prebuffer_ms=int(os.getenv("TTS_PREBUFFER_MS", "500")),
    )
    await player.start()

    tts = ElevenLabsTTS(api_key=api_key, voice_id=voice_id, output_format="pcm_16000")

    total_bytes = 0
    first = True
    async for chunk in tts.stream(text):
        if first:
            first = False
            print("TTS: first audio bytes arrived")
        total_bytes += len(chunk)
        await player.enqueue(chunk)

    print(f"TTS: total bytes enqueued: {total_bytes}")
    approx_pcm_seconds = total_bytes / (16000 * 2)
    print(f"TTS: approx pcm duration (if pcm16@16k): {approx_pcm_seconds:.2f}s")

    # If the total audio is shorter than the prebuffer target, we will never be "ready".
    # In that case, just proceed to attempt playback.
    prebuffer_ms = int(os.getenv("TTS_PREBUFFER_MS", "500"))
    prebuffer_target_bytes = int(16000 * (prebuffer_ms / 1000.0) * 2)
    if total_bytes > 0 and total_bytes < prebuffer_target_bytes:
        print(
            f"TTS: total audio ({approx_pcm_seconds:.2f}s) is shorter than prebuffer ({prebuffer_ms}ms); "
            "disabling prebuffer for this run."
        )
        player._ready_to_play.set()  # type: ignore[attr-defined]

    ok = await player.wait_buffered(min_ms=min(prebuffer_ms, int(approx_pcm_seconds * 1000)), timeout_s=5.0)
    print(f"TTS: prebuffer reached ({prebuffer_ms}ms): {ok}")

    t_play = await player.wait_first_play(timeout_s=5.0)
    if t_play is None:
        print("WARNING: audio never started playing (check output device)")
        await asyncio.sleep(2.0)
    else:
        # Drain time: be intentionally generous to rule out early shutdown.
        drain_s = float(os.getenv("TTS_DRAIN_SECONDS", "15"))
        print(f"TTS: draining for {drain_s:.1f}s...")
        # simple progress wait
        steps = int(drain_s / 0.5)
        for _ in range(max(1, steps)):
            await asyncio.sleep(0.5)

    await player.close()


if __name__ == "__main__":
    asyncio.run(main())
