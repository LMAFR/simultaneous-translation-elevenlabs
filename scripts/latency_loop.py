from __future__ import annotations

import asyncio
import os
import re
import signal
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Allow `python scripts/latency_loop.py` to import from /src
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv

from src.audio_io import AudioConfig, EnergyVAD, JitterBufferPlayer, mic_pcm16_stream
from src.elevenlabs_stt import ElevenLabsRealtimeSTT, TranscriptEvent
from src.elevenlabs_tts import ElevenLabsTTS, resolve_voice_id
from src.metrics import MetricsLogger


PUNCT_RE = re.compile(r"[\.!\?\,\;\:]")


@dataclass
class TranslationConfig:
    mode: str
    input_lang: str
    output_lang: str


async def translate(text: str, cfg: TranslationConfig) -> str:
    mode = (cfg.mode or "none").lower()
    if mode == "none":
        return text
    if mode == "dummy":
        # Tiny placeholder so we can test latency behavior without external calls
        dictionary = {
            "hola": "hello",
            "gracias": "thanks",
            "adios": "goodbye",
        }
        words = [dictionary.get(w.lower(), w) for w in text.split()]
        return "[TRANSLATED] " + " ".join(words)
    if mode == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return text
        try:
            import importlib

            openai_mod = importlib.import_module("openai")
            AsyncOpenAI = getattr(openai_mod, "AsyncOpenAI")

            client = AsyncOpenAI(api_key=api_key)
            prompt = (
                f"Translate from {cfg.input_lang} to {cfg.output_lang}. "
                "Return only the translated text.\n\n"
                f"Text: {text}"
            )
            resp = await client.responses.create(
                model="gpt-4o-mini",
                input=prompt,
            )
            out = resp.output_text
            return out.strip() if out else text
        except Exception:
            return text

    return text


def chunk_text_delta(prev: str, curr: str, min_chars: int = 24) -> str:
    if not curr:
        return ""
    if curr.startswith(prev):
        delta = curr[len(prev) :]
    else:
        # model may revise partials; in this prototype, reset
        delta = curr
    delta_stripped = delta.strip()
    if not delta_stripped:
        return ""

    if len(delta_stripped) >= min_chars or PUNCT_RE.search(delta_stripped):
        return delta_stripped
    return ""


async def main() -> None:
    load_dotenv()

    print("WARNING: Use headphones to avoid feedback loops.")

    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        raise SystemExit("Missing ELEVENLABS_API_KEY (see .env.example)")

    input_lang = os.getenv("INPUT_LANG", "es")
    output_lang = os.getenv("OUTPUT_LANG", "en")
    translation_mode = os.getenv("TRANSLATION_MODE", "none")

    voice_id = resolve_voice_id()

    audio_cfg = AudioConfig(sample_rate=16000, channels=1, dtype="int16", block_ms=20)
    # Treat hangover as "still speaking"; silence cutoff for utterance end is handled below.
    vad = EnergyVAD(threshold_rms=0.012, hangover_ms=250, sample_rate=audio_cfg.sample_rate, block_ms=audio_cfg.block_ms)

    stt = ElevenLabsRealtimeSTT(api_key=api_key, input_lang=input_lang, sample_rate=audio_cfg.sample_rate)
    tts = ElevenLabsTTS(api_key=api_key, voice_id=voice_id, output_format="pcm_16000")

    metrics = MetricsLogger()

    player = JitterBufferPlayer(sample_rate=audio_cfg.sample_rate, channels=1, dtype="int16")
    await player.start()

    # Queue carries (pcm_bytes, is_speech, utterance_id)
    audio_q: asyncio.Queue[tuple[bytes, bool, Optional[str]]] = asyncio.Queue(maxsize=400)

    # Per-utterance audio queues for the new per-utterance STT sessions
    utt_audio_q: dict[str, asyncio.Queue[bytes | None]] = {}

    stop_event = asyncio.Event()

    def _handle_sigint(*_):
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_sigint)

    # Utterance state shared by tasks
    utt_counter = 0
    current_utt_id: Optional[str] = None
    in_speech = False
    silence_frames = 0

    end_silence_ms = 1200  # be more conservative than 1s to avoid chopping
    end_silence_frames = max(1, int(end_silence_ms / audio_cfg.block_ms))

    min_speech_ms = 400  # don't treat <400ms as a real utterance
    min_speech_frames = max(1, int(min_speech_ms / audio_cfg.block_ms))
    speech_frames_in_utt = 0

    # Track whether playback is allowed (only after silence)
    playback_allowed = asyncio.Event()
    playback_allowed.set()  # not speaking initially

    # Buffers per utterance (collect what we will say)
    pending_say_by_utt: dict[str, str] = {}
    # Track latest full STT hypothesis per utterance
    latest_stt_by_utt: dict[str, str] = {}
    ended_utt_q: asyncio.Queue[str] = asyncio.Queue()

    # Track when an STT session for an utterance has finished (queue drained + ws closed)
    stt_done_evt_by_utt: dict[str, asyncio.Event] = {}

    # Debug: count frames per utterance we actually forwarded
    forwarded_frames_by_utt: dict[str, int] = {}

    async def stt_consumer_task(utt_id: str, q: "asyncio.Queue[bytes | None]") -> None:
        """Consumes STT events for a single utterance session.

        Keep only the latest full hypothesis.
        """
        done_evt = stt_done_evt_by_utt.setdefault(utt_id, asyncio.Event())
        tr_cfg = TranslationConfig(mode=translation_mode, input_lang=input_lang, output_lang=output_lang)
        last_translated_src = ""

        try:
            async for ev in stt.run_one_utterance(q, utt_id):
                if stop_event.is_set():
                    break

                if ev.text:
                    latest_stt_by_utt[utt_id] = ev.text

                # Optional: keep the console preview, but do not trust it for final playback.
                if (not ev.is_final) and ev.text:
                    if metrics.get(utt_id).t_partial_transcript_ms is None:
                        metrics.set_if_none(utt_id, "t_partial_transcript_ms", ev.received_time_ms)

                    if ev.text != last_translated_src:
                        last_translated_src = ev.text
                        translated = await translate(ev.text, tr_cfg)
                        pending_say_by_utt[utt_id] = translated.strip()
                        print(
                            f"[{ev.received_time_ms:,.0f} ms] SAY (buffering, not playing yet) "
                            f"({utt_id}): {pending_say_by_utt[utt_id]}"
                        )
                    continue

                if ev.is_final and ev.text:
                    metrics.set_if_none(utt_id, "t_final_transcript_ms", ev.received_time_ms)
                    latest_stt_by_utt[utt_id] = ev.text
        finally:
            done_evt.set()

    async def mic_task():
        nonlocal utt_counter, current_utt_id, in_speech, silence_frames, speech_frames_in_utt

        async for pcm_bytes, t_ms, is_speech in mic_pcm16_stream(audio_cfg, vad=vad, include_silence=True):
            if stop_event.is_set():
                break

            # speech start
            if is_speech and not in_speech:
                in_speech = True
                silence_frames = 0
                speech_frames_in_utt = 0

                utt_counter += 1
                current_utt_id = f"utt_{utt_counter}"
                playback_allowed.clear()

                # create per-utterance audio queue + stt consumer
                q: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=800)
                utt_audio_q[current_utt_id] = q
                asyncio.create_task(stt_consumer_task(current_utt_id, q))

                metrics.set_if_none(current_utt_id, "t_first_audio_captured_ms", t_ms)
                print(f"[{t_ms:,.0f} ms] MIC: speech start ({current_utt_id})")

            if in_speech and is_speech:
                speech_frames_in_utt += 1

            # speech end detection (silence)
            if not is_speech and in_speech:
                silence_frames += 1
                if silence_frames >= end_silence_frames:
                    # Ignore extremely short utterances
                    if speech_frames_in_utt < min_speech_frames:
                        print(
                            f"[{t_ms:,.0f} ms] MIC: discard short utterance ({current_utt_id}) "
                            f"speech_ms≈{speech_frames_in_utt * audio_cfg.block_ms}"
                        )
                        # close STT utterance queue
                        if current_utt_id and current_utt_id in utt_audio_q:
                            try:
                                utt_audio_q[current_utt_id].put_nowait(None)
                            except asyncio.QueueFull:
                                pass
                            utt_audio_q.pop(current_utt_id, None)

                        pending_say_by_utt.pop(current_utt_id or "", None)
                        in_speech = False
                        silence_frames = 0
                        playback_allowed.set()
                        current_utt_id = None
                        continue

                    in_speech = False
                    silence_frames = 0
                    playback_allowed.set()
                    print(f"[{t_ms:,.0f} ms] MIC: speech end ({current_utt_id})")

                    # close STT utterance queue (signals commit)
                    if current_utt_id and current_utt_id in utt_audio_q:
                        try:
                            utt_audio_q[current_utt_id].put_nowait(None)
                        except asyncio.QueueFull:
                            pass
                        utt_audio_q.pop(current_utt_id, None)

                    # Trigger TTS for this utterance
                    if current_utt_id:
                        try:
                            ended_utt_q.put_nowait(current_utt_id)
                        except asyncio.QueueFull:
                            pass

                    current_utt_id = None
                    continue

            # Forward audio frames into the current utterance queue
            if current_utt_id is None:
                continue

            if in_speech or (not is_speech and silence_frames > 0):
                q = utt_audio_q.get(current_utt_id)
                if q is None:
                    continue
                try:
                    q.put_nowait(pcm_bytes)
                    forwarded_frames_by_utt[current_utt_id] = forwarded_frames_by_utt.get(current_utt_id, 0) + 1
                except asyncio.QueueFull:
                    pass

    async def playback_task():
        tr_cfg = TranslationConfig(mode=translation_mode, input_lang=input_lang, output_lang=output_lang)

        # Give STT a moment after mic end-of-utterance to finalize the last hypothesis.
        post_utt_delay_s = float(os.getenv("POST_UTTERANCE_DELAY_S", "1.0"))

        while not stop_event.is_set():
            utt_id = await ended_utt_q.get()
            frames = forwarded_frames_by_utt.get(utt_id, 0)

            if post_utt_delay_s > 0:
                await asyncio.sleep(post_utt_delay_s)

            # Wait for the STT session to finish so we really have the last partial/final.
            done_evt = stt_done_evt_by_utt.get(utt_id)
            if done_evt is not None:
                try:
                    await asyncio.wait_for(done_evt.wait(), timeout=3.0)
                except asyncio.TimeoutError:
                    pass

            # Translate the latest hypothesis at playback time (authoritative).
            src_text = (latest_stt_by_utt.get(utt_id) or "").strip()
            if not src_text:
                # fall back to whatever preview we had
                src_text = (pending_say_by_utt.get(utt_id) or "").strip()
                if not src_text:
                    print(f"[PLAYBACK] ({utt_id}) No buffered text to speak. (frames_forwarded={frames})")
                    continue
                final_to_say = src_text
            else:
                final_to_say = (await translate(src_text, tr_cfg)).strip()

            print(f"[PLAYBACK] SAY FINAL ({utt_id}): {final_to_say}")
            await playback_allowed.wait()
            await speak_streaming(tts, final_to_say, metrics, player, utt_id)

            pending_say_by_utt.pop(utt_id, None)
            latest_stt_by_utt.pop(utt_id, None)
            stt_done_evt_by_utt.pop(utt_id, None)
            metrics.commit_utterance(utt_id)
            print(metrics.summary_str())

    async def speak_streaming(
        tts_client: ElevenLabsTTS,
        text: str,
        metrics: MetricsLogger,
        player: JitterBufferPlayer,
        utt_id: str,
        is_partial: bool = False,
    ) -> None:
        if not text.strip():
            return

        speak_text = text if not is_partial else (text.strip() + " ")

        first_chunk = True
        async for audio_chunk in tts_client.stream(speak_text):
            if first_chunk:
                first_chunk = False
                if metrics.get(utt_id).t_first_tts_audio_bytes_ms is None:
                    t = metrics.mark(utt_id, "t_first_tts_audio_bytes_ms")
                    print(f"[{t:,.0f} ms] TTS: first audio bytes arrived ({utt_id})")
            await player.enqueue(audio_chunk)

        if metrics.get(utt_id).t_first_audio_play_ms is None:
            t_play = await player.wait_first_play(timeout_s=2.0)
            if t_play is not None:
                metrics.set_if_none(utt_id, "t_first_audio_play_ms", t_play)
                lat = metrics.get(utt_id).e2e_latency_ms()
                if lat is not None:
                    print(f"[{t_play:,.0f} ms] PLAY: first audio played ({utt_id}) | e2e≈{lat:.0f} ms")

    tasks = [
        asyncio.create_task(mic_task()),
        asyncio.create_task(playback_task()),
    ]

    await stop_event.wait()

    for t in tasks:
        t.cancel()
    await player.close()

    print("\nStopped.")
    print(metrics.summary_str())


if __name__ == "__main__":
    asyncio.run(main())