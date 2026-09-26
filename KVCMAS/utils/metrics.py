from __future__ import annotations

import json
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Dict, Optional

from loguru import logger


@dataclass(slots=True)
class GenerationResult:
    """Container describing a single model generation outcome."""

    text: str
    mode: str
    ttft: float
    e2e_latency: float = 0.0
    raw_output: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class RequestMetricsRecorder:
    """Tracks per-request agent outputs, reuse rates, and mode-specific TTFT."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._requests: Dict[str, Dict[str, Any]] = {}
        self._total_calls: int = 0
        self._total_reuse_calls: int = 0
        # Handed-over token counts, for the serving replay; the reported reuse ratio is
        # request-wise and does not use them.
        self._total_reusable_tokens: int = 0
        self._total_reused_tokens: int = 0
        # Source hops (the request's first-executed agent) cannot reuse: no upstream output exists
        # and it IS the harvest source.
        self._total_source_calls: int = 0
        self._ttft_stats: Dict[str, Dict[str, float]] = {}
        self._tt_stats: Dict[str, Dict[str, float]] = {}

    def start_request(
        self,
        *,
        request_uid: str,
        batch_index: Optional[int],
        task: Optional[str],
        execution_mode: str,
    ) -> None:
        """Initialise bookkeeping for a new request."""
        with self._lock:
            self._requests[request_uid] = {
                "batch_index": batch_index,
                "task": task,
                "execution_mode": execution_mode,
                "agents": [],
                "kv_reuse_count": 0,
                "total_count": 0,
            }

    def record_agent_output(
        self,
        *,
        request_uid: str,
        agent_id: str,
        agent_name: str,
        agent_role: str,
        generation: GenerationResult | None,
    ) -> None:
        """Log an agent's generation output and update reuse statistics."""
        if generation is None:
            return

        with self._lock:
            request_entry = self._requests.setdefault(
                request_uid,
                {
                    "batch_index": None,
                    "task": None,
                    "execution_mode": "unknown",
                    "agents": [],
                    "kv_reuse_count": 0,
                    "total_count": 0,
                },
            )

            agent_record: Dict[str, Any] = {
                "agent_id": agent_id,
                "agent_name": agent_name,
                "agent_role": agent_role,
                "mode": generation.mode,
                "ttft": generation.ttft,
                "e2e_latency": generation.e2e_latency,
            }
            if agent_name != "CopyMachine":
                agent_record["text"] = generation.text
            if generation.metadata:
                agent_record["metadata"] = generation.metadata
                _md = generation.metadata
                if "reusable_tokens" in _md:
                    self._total_reusable_tokens += int(_md.get("reusable_tokens", 0))
                    self._total_reused_tokens += int(_md.get("reused_tokens", 0))
                    if _md.get("is_source_hop"):
                        self._total_source_calls += 1
            agents_list = request_entry["agents"]
            replaced = False
            for idx, existing in enumerate(agents_list):
                if existing["agent_id"] == agent_id:
                    agents_list[idx] = agent_record
                    replaced = True
                    break
            if not replaced:
                agents_list.append(agent_record)


            request_entry["total_count"] = len(agents_list)
            request_entry["kv_reuse_count"] = sum(
                1 for entry in agents_list if entry.get("mode") == "kv_reuse"
            )


            stats = self._ttft_stats.setdefault(
                generation.mode, {"sum": 0.0, "count": 0.0}
            )
            stats["sum"] += generation.ttft
            stats["count"] += 1
            avg_ttft = stats["sum"] / stats["count"] if stats["count"] else 0.0

            tt_stats = self._tt_stats.setdefault(
                generation.mode, {"sum": 0.0, "count": 0.0}
            )
            tt_stats["sum"] += generation.e2e_latency
            tt_stats["count"] += 1
            avg_tt = tt_stats["sum"] / tt_stats["count"] if tt_stats["count"] else 0.0

            metadata = generation.metadata or {}
            latency_payload: Dict[str, Any] = {
                "request_uid": request_uid,
                "agent_id": agent_id,
                "agent_name": agent_name,
                "agent_role": agent_role,
                "mode": generation.mode,
            }
            preprocess_latency = metadata.get("preprocess_latency")
            if preprocess_latency is not None:
                latency_payload["preprocess_latency"] = preprocess_latency
            generation_ttft = metadata.get("generation_ttft")
            if generation_ttft is not None:
                latency_payload["generation_ttft"] = generation_ttft
            latency_payload["ttft"] = generation.ttft
            latency_payload["mode_avg_ttft"] = avg_ttft
            latency_payload["tt"] = generation.e2e_latency
            latency_payload["mode_avg_tt"] = avg_tt
            logger.opt(colors=True).info(
                "<cyan>[LATENCY:{mode}]</cyan> {}",
                json.dumps(latency_payload, ensure_ascii=False),
                mode=generation.mode,
            )

            if agent_name != "CopyMachine":
                output_payload = {
                    "request_uid": request_uid,
                    "agent_id": agent_id,
                    "agent_name": agent_name,
                    "agent_role": agent_role,
                    "mode": generation.mode,
                    "text": generation.text,
                }
                logger.opt(colors=True).info(
                    "<green>[AGENT OUTPUT]</green> {}",
                    json.dumps(output_payload, ensure_ascii=False),
                )

    def finalize_request(self, request_uid: str) -> Optional[float]:
        """Compute and log per-request reuse statistics."""
        with self._lock:
            request_entry = self._requests.pop(request_uid, None)
            if request_entry is None:
                return None

            kv_reuse = request_entry.get("kv_reuse_count", 0)
            total = request_entry.get("total_count", 0)
            _src = sum(1 for a in request_entry.get("agents", [])
                       if (a.get("metadata") or {}).get("is_source_hop"))
            _capable = max(0, total - _src)          # calls that COULD reuse
            reuse_rate = (kv_reuse / _capable) if _capable else 0.0

            is_copy = any(
                a.get("agent_name") == "CopyMachine"
                for a in request_entry.get("agents", [])
            )
            # Per-request bookkeeping only; the reported ratio is the run-level one in
            # log_cumulative.
            payload: Dict[str, Any] = {
                "request_uid": request_uid,
                "batch_index": request_entry.get("batch_index"),
                "execution_mode": request_entry.get("execution_mode"),
                "kv_reuse_calls": kv_reuse,
                "reuse_capable_calls": _capable,
                "total_agents": total,
            }
            if not is_copy:
                payload["task"] = request_entry.get("task")
            logger.opt(colors=True).info(
                "<magenta>[REQUEST REUSE]</magenta> {}",
                json.dumps(payload, ensure_ascii=False),
            )

            self._total_calls += total
            self._total_reuse_calls += kv_reuse
            return reuse_rate

    def log_cumulative(self, *, batch_index: Optional[int]) -> float:
        """Log the run-level reuse ratio across all processed requests."""
        with self._lock:
            _capable = max(0, self._total_calls - self._total_source_calls)
            cumulative = (self._total_reuse_calls / _capable) if _capable else 0.0

            payload = {
                "batch_index": batch_index,
                # Reuse ratio (rho): reuse-capable agent calls that reused, over all
                # reuse-capable agent calls. Source hops are outside the denominator.
                "reuse_ratio": cumulative,
                "kv_reuse_calls": self._total_reuse_calls,
                "reuse_capable_calls": _capable,
                "total_agent_calls": self._total_calls,
                "source_hop_calls": self._total_source_calls,
            }
            logger.opt(colors=True).info(
                "<yellow>[CUMULATIVE REUSE]</yellow> {}",
                json.dumps(payload, ensure_ascii=False),
            )
            return cumulative


metrics_recorder = RequestMetricsRecorder()
