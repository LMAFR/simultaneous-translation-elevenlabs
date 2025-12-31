from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


def now_ms() -> float:
    return time.perf_counter() * 1000.0


@dataclass
class UtteranceMetrics:
    utterance_id: str
    t_first_audio_captured_ms: Optional[float] = None
    t_partial_transcript_ms: Optional[float] = None
    t_final_transcript_ms: Optional[float] = None
    t_first_tts_audio_bytes_ms: Optional[float] = None
    t_first_audio_play_ms: Optional[float] = None

    def e2e_latency_ms(self) -> Optional[float]:
        if self.t_first_audio_captured_ms is None or self.t_first_audio_play_ms is None:
            return None
        return self.t_first_audio_play_ms - self.t_first_audio_captured_ms


@dataclass
class MetricsLogger:
    _utterances: Dict[str, UtteranceMetrics] = field(default_factory=dict)
    _e2e_latencies_ms: List[float] = field(default_factory=list)

    def get(self, utterance_id: str) -> UtteranceMetrics:
        if utterance_id not in self._utterances:
            self._utterances[utterance_id] = UtteranceMetrics(utterance_id=utterance_id)
        return self._utterances[utterance_id]

    def mark(self, utterance_id: str, field_name: str) -> float:
        t = now_ms()
        u = self.get(utterance_id)
        if getattr(u, field_name) is None:
            setattr(u, field_name, t)
        return t

    def set_if_none(self, utterance_id: str, field_name: str, t_ms: float) -> float:
        u = self.get(utterance_id)
        if getattr(u, field_name) is None:
            setattr(u, field_name, t_ms)
        return float(getattr(u, field_name))

    def commit_utterance(self, utterance_id: str) -> None:
        u = self._utterances.get(utterance_id)
        if not u:
            return
        lat = u.e2e_latency_ms()
        if lat is not None:
            self._e2e_latencies_ms.append(lat)

    def summary_str(self) -> str:
        if not self._e2e_latencies_ms:
            return "No complete utterances recorded."
        vals = sorted(self._e2e_latencies_ms)
        avg = sum(vals) / len(vals)
        p50 = vals[int(0.50 * (len(vals) - 1))]
        p90 = vals[int(0.90 * (len(vals) - 1))]
        p95 = vals[int(0.95 * (len(vals) - 1))]
        return (
            f"Utterances: {len(vals)} | e2e mic→play (ms): "
            f"avg={avg:.0f} p50={p50:.0f} p90={p90:.0f} p95={p95:.0f}"
        )