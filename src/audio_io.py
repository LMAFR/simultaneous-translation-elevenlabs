from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass
from typing import AsyncGenerator, Optional

import numpy as np
import sounddevice as sd


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    channels: int = 1
    dtype: str = "int16"  # PCM16
    block_ms: int = 20  # 20ms frames

    @property
    def block_frames(self) -> int:
        return int(self.sample_rate * self.block_ms / 1000)


class EnergyVAD:
    def __init__(
        self,
        threshold_rms: float = 0.015,
        hangover_ms: int = 250,
        sample_rate: int = 16000,
        block_ms: int = 20,
    ):
        self.threshold_rms = threshold_rms
        self.hangover_frames = max(1, int(hangover_ms / block_ms))
        self._hang = 0

    def is_speech(self, pcm16: np.ndarray) -> bool:
        # pcm16: shape (frames,), int16
        x = pcm16.astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(x * x) + 1e-12))
        if rms >= self.threshold_rms:
            self._hang = self.hangover_frames
            return True
        if self._hang > 0:
            self._hang -= 1
            return True
        return False


async def mic_pcm16_stream(
    config: AudioConfig,
    vad: Optional[EnergyVAD] = None,
    include_silence: bool = False,
) -> AsyncGenerator[tuple[bytes, float, bool], None]:
    """Yields (pcm16_bytes, capture_time_ms, is_speech)."""

    loop = asyncio.get_running_loop()
    q: asyncio.Queue[tuple[np.ndarray, float]] = asyncio.Queue(maxsize=100)

    def callback(indata, frames, time_info, status):
        if status:
            pass
        # indata is (frames, channels)
        mono = indata[:, 0].copy()
        t_ms = time.perf_counter() * 1000.0
        try:
            loop.call_soon_threadsafe(q.put_nowait, (mono, t_ms))
        except asyncio.QueueFull:
            # Drop if overloaded
            pass

    stream = sd.InputStream(
        samplerate=config.sample_rate,
        channels=config.channels,
        dtype=config.dtype,
        blocksize=config.block_frames,
        callback=callback,
    )

    with stream:
        while True:
            pcm16, t_ms = await q.get()
            is_speech = vad.is_speech(pcm16) if vad else True
            if include_silence or is_speech:
                yield (pcm16.tobytes(), t_ms, is_speech)


class JitterBufferPlayer:
    """Very small jitter buffer + playback using sounddevice OutputStream."""

    def __init__(self, sample_rate: int, channels: int = 1, dtype: str = "int16", prebuffer_ms: int = 400):
        self.sample_rate = sample_rate
        self.channels = channels
        self.dtype = dtype
        self.prebuffer_ms = prebuffer_ms
        self._prebuffer_bytes = int(self.sample_rate * (self.prebuffer_ms / 1000.0) * 2 * self.channels)

        self._byte_q: asyncio.Queue[bytes] = asyncio.Queue(maxsize=400)
        self._closed = asyncio.Event()

        self._started = threading.Event()
        self._played_first = threading.Event()
        self._first_play_time_ms: Optional[float] = None

        self._buffer = bytearray()
        self._lock = threading.Lock()

        # Gate actual playback until we have enough buffered bytes
        self._ready_to_play = threading.Event()

        self._stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=self.dtype,
            blocksize=0,  # let portaudio decide
            callback=self._callback,
        )

    def _callback(self, outdata, frames, time_info, status):
        need_bytes = frames * 2 * self.channels  # int16

        # Prebuffer to avoid immediate underruns with streamed audio
        if self.prebuffer_ms > 0 and not self._ready_to_play.is_set():
            outdata[:] = np.zeros((frames, self.channels), dtype=np.int16)
            return

        with self._lock:
            if len(self._buffer) < need_bytes:
                # underrun: output zeros
                outdata[:] = np.zeros((frames, self.channels), dtype=np.int16)
                return
            chunk = self._buffer[:need_bytes]
            del self._buffer[:need_bytes]

        out = np.frombuffer(chunk, dtype=np.int16).reshape(frames, self.channels)
        outdata[:] = out
        if not self._played_first.is_set():
            self._first_play_time_ms = time.perf_counter() * 1000.0
            self._played_first.set()

    async def start(self) -> None:
        if self._started.is_set():
            return
        self._stream.start()
        self._started.set()
        asyncio.create_task(self._pump())

    async def _pump(self) -> None:
        # Pull bytes from asyncio queue and append to internal buffer for callback thread.
        while not self._closed.is_set():
            try:
                data = await asyncio.wait_for(self._byte_q.get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue
            with self._lock:
                self._buffer.extend(data)
                if self.prebuffer_ms > 0 and not self._ready_to_play.is_set() and len(self._buffer) >= self._prebuffer_bytes:
                    self._ready_to_play.set()

    async def enqueue(self, pcm16_bytes: bytes) -> None:
        if self._closed.is_set():
            return
        try:
            self._byte_q.put_nowait(pcm16_bytes)
        except asyncio.QueueFull:
            # Drop oldest by draining a bit
            _ = self._byte_q.get_nowait()
            self._byte_q.put_nowait(pcm16_bytes)

    async def wait_first_play(self, timeout_s: float = 5.0) -> Optional[float]:
        # wait in thread-friendly way
        start = time.time()
        while time.time() - start < timeout_s:
            if self._played_first.is_set():
                return self._first_play_time_ms
            await asyncio.sleep(0.01)
        return None

    async def wait_buffered(self, min_ms: int = 400, timeout_s: float = 5.0) -> bool:
        """Wait until at least `min_ms` of audio is buffered."""
        target = int(self.sample_rate * (min_ms / 1000.0) * 2 * self.channels)
        start = time.time()
        while time.time() - start < timeout_s:
            with self._lock:
                if len(self._buffer) >= target:
                    return True
            await asyncio.sleep(0.01)
        return False

    async def close(self) -> None:
        self._closed.set()
        await asyncio.sleep(0)  # allow pump to exit
        try:
            self._stream.stop()
            self._stream.close()
        except Exception:
            pass