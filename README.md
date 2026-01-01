# simultaneous-translation-elevenlabs
Simple project to test simultaneous translation feature from elevenLabs API (locally).

## Local latency loop (MIC → STT → (translate) → Streaming TTS → play)

This repo contains a local, single-machine Python prototype to measure perceived end-to-end latency for a “simultaneous translation” style loop.

**Important:** use **headphones** (otherwise you will create a feedback loop).

### Structure
- `scripts/latency_loop.py` – main entry
- `src/audio_io.py` – mic capture + playback
- `src/elevenlabs_stt.py` – ElevenLabs realtime STT (WebSocket, best-effort)
- `src/elevenlabs_tts.py` – ElevenLabs streaming TTS (HTTP streaming)
- `src/metrics.py` – timestamps + summary metrics

### Setup
1. Create a virtualenv and install deps
2. Copy env file:
   - `cp .env.example .env`
3. Fill in:
   - `ELEVENLABS_API_KEY`
   - (optional) `ELEVENLABS_VOICE_ID`

### Run
- `python scripts/latency_loop.py`

You should see logs like:
- MIC first speech frame captured
- STT partial deltas / final
- TTS first audio bytes arrived
- PLAY first audio played + approximate e2e mic→play latency

## Architecture notes (updated)

### STT sessions are per utterance
This prototype opens a **fresh ElevenLabs realtime STT WebSocket session per utterance** (speech segment).
This is intentional: in practice, segmenting multiple utterances inside a single long-lived session can stop producing partials after the first commit.

### TTS uses the ElevenLabs Python SDK (preferred)
The TTS client (`src/elevenlabs_tts.py`) maintains an `async for chunk in tts.stream(text)` interface, but it now **prefers the official ElevenLabs Python SDK** when installed.
If the SDK is unavailable or fails, it falls back to the HTTP streaming endpoint.

### Environment variables
- `ELEVENLABS_API_KEY` (required)
- `ELEVENLABS_VOICE_ID` (optional; defaults to a commonly-available voice)
- `INPUT_LANG` (default `es`)
- `OUTPUT_LANG` (default `en`)
- `TRANSLATION_MODE`: `none` | `dummy` | `openai`
  - `none`: echoes text
  - `dummy`: tiny placeholder translation (no extra keys)
  - `openai`: uses `OPENAI_API_KEY` if present; otherwise falls back to echo

Optional tuning:
- `OUTPUT_DEVICE`: integer output device index for sounddevice
- `TTS_PREBUFFER_MS`: jitter buffer prebuffer for playback
- `TTS_DRAIN_SECONDS`: how long the smoke test waits before exiting

### Using OpenAI only for translation
Set:
- `TRANSLATION_MODE=openai`
- `OPENAI_API_KEY=...`

Pipeline:
`MIC -> ElevenLabs Realtime STT -> OpenAI translation -> ElevenLabs Streaming TTS -> Playback`

If `OPENAI_API_KEY` is missing, the code will automatically fall back to echoing the original text so the loop still runs.

## Troubleshooting

- **Echo/feedback:** use headphones, reduce speaker volume, and/or lower mic gain.
- **Latency tuning:**
  - Smaller `block_ms` (e.g. 10–20ms) can help, but may increase CPU overhead.
  - TTS param `optimize_streaming_latency` is set to a high value.
- **STT WebSocket:** ElevenLabs realtime STT endpoint/payload may differ by account/model.
  If STT doesn’t connect or parse results, edit `src/elevenlabs_stt.py` only.
- **Audio format:** TTS requests `pcm_16000` for simplest streaming playback.
  If your account doesn’t support PCM streaming, switch to mp3 output and add a decoder.
- **Truncated TTS audio (only first word/syllables):** this can happen if your ElevenLabs account/API key has **insufficient credits** or is rate-limited.
  Verify your credit balance and re-run the smoke test.
