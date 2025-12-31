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

### Environment variables
- `ELEVENLABS_API_KEY` (required)
- `ELEVENLABS_VOICE_ID` (optional; defaults to a commonly-available voice)
- `INPUT_LANG` (default `es`)
- `OUTPUT_LANG` (default `en`)
- `TRANSLATION_MODE`: `none` | `dummy` | `openai`
  - `none`: echoes text
  - `dummy`: tiny placeholder translation (no extra keys)
  - `openai`: uses `OPENAI_API_KEY` if present; otherwise falls back to echo

### Using OpenAI only for translation
Set:
- `TRANSLATION_MODE=openai`
- `OPENAI_API_KEY=...`

Pipeline:
`MIC -> ElevenLabs Realtime STT -> OpenAI translation -> ElevenLabs Streaming TTS -> Playback`

If `OPENAI_API_KEY` is missing, the code will automatically fall back to echoing the original text so the loop still runs.

### Notes / troubleshooting
- **Echo/feedback:** use headphones, reduce speaker volume, and/or lower mic gain.
- **Latency tuning:**
  - Smaller `block_ms` (e.g. 10–20ms) can help, but may increase CPU overhead.
  - TTS param `optimize_streaming_latency` is set to a high value.
- **STT WebSocket:** ElevenLabs realtime STT endpoint/payload may differ by account/model.
  If STT doesn’t connect or parse results, edit `src/elevenlabs_stt.py` only.
- **Audio format:** TTS requests `pcm_16000` for simplest streaming playback.
  If your account doesn’t support PCM streaming, switch to mp3 output and add a decoder.
