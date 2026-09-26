import argparse
import asyncio
import sys, os
# Segment layout of the controlled CopyMachine trajectory, in tokens.
os.environ.setdefault("SYSTEM_PROMPT", "1024")     # pf0, the cacheable prefix
os.environ.setdefault("USER_PROMPT", "512")        # ph0, the shared NON-decoded span
os.environ.setdefault("RETRIEVED_TEXT", "512")     # priv, per-agent, never shared
os.environ.setdefault("GENERATED_TEXT", "128")     # ph_i, the DECODED span relayed on
os.environ.setdefault("INTER_AGENT_TEXT", "16")    # pf_i, glue before each peer output
# Long-context warmup repeatedly allocates/frees ~8 GB caches, which fragments the CUDA pool
# (reserved-but-unallocated grows until a contiguous alloc.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout.reconfigure(encoding='utf-8')
import random
import json
import time
from pathlib import Path
from typing import List, Literal, Union, Dict, Any

import numpy as np
import torch
from KVCMAS.graph.graph import Graph
from dataclasses import replace
from KVCMAS.llm.config import KVCommConfig
from KVCMAS.utils.log import configure_logging, logger
from KVCMAS.utils.prefix_cache_shim import (
    enabled as _prefix_cache_enabled,
    maybe_install as _install_prefix_shim,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Steady-phase peak, set by evaluate() and read by the summary writer.
_STEADY_PEAK_GB = [0.0]

SEED = int(os.getenv("SEED", 42))
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


def parse_args():
    parser = argparse.ArgumentParser(description="KVCMAS Efficiency Benchmark (TTFT + Throughput)")
    parser.add_argument(
        "--mode",
        type=str,
        default="FullConnected",
        choices=["DirectAnswer", "FullConnected", "Random", "Chain", "Debate", "Layered", "Star", "Mesh"],
        help="The communication topology among agents.",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--agent_names",
        nargs="+",
        type=str,
        default=["CopyMachine"],
        help="List of agent names to include in the graph."
    )
    parser.add_argument(
        "--agent_nums",
        nargs="+",
        type=int,
        default=[5],
        help="List of counts corresponding to each agent name."
    )
    parser.add_argument("--llm_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--domain", type=str, default="COPY")
    parser.add_argument("--decision_method", type=str, default=None)
    parser.add_argument(
        "--execution_mode",
        type=str,
        default="allow_kv_reuse",
        choices=["default", "allow_kv_reuse"],
        help="Execution strategy for the graph.",
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save the output results. Defaults to runs/efficiency_<mode>.")
    parser.add_argument("--prefix", type=str, default="", help="Lead-in text before the user question in the standalone encode. Empty under the declared segment layout (copy_machine.py): ph0 is the task alone, so the encode must cover the task alone, and any lead-in here would be undeclared tokens charged to the shared span.")
    parser.add_argument("--samples", type=int, default=10, help="Number of query samples")
    parser.add_argument("--warmup", type=int, default=None, help="Number of warmup samples for anchor building (default: --kv-max-anchor-num, so warmup fills the pool to capacity).")
    parser.add_argument("--reuse-ratio", type=float, default=1.0, help="Fraction of STEADY agent executions that reuse (vs dense re-prefill). 1.0 = all reuse (current). E.g. 0.8 => 20%% of the N*samples executions re-prefill, assigned agent-0-first then agent-1, etc. Only affects allow_kv_reuse mode.")
    parser.add_argument("--kv-threshold", type=float, default=2.0, help="Threshold for key-value memory usage. >=2 bulletproofs the entropy gate against fp32 noise so steady state stays strict kv_reuse.")
    parser.add_argument("--kv-max-anchor-num", type=int, default=10, help="Maximum number of anchors for key-value memory.")
    parser.add_argument("--kv-window-size", type=int, default=5, help="Window size for key-value memory update.")
    parser.add_argument("--kv-thread-workers", type=int, default=None, help="Number of thread workers for key-value memory processing.")
    parser.add_argument("--kv-worker-timeout", type=float, default=None, help="Timeout for key-value memory workers processing.")
    parser.add_argument("--rank-ph", dest="svd_rank_ph", type=int, default=32, help="KVCMAS SVD rank for the PLACEHOLDER (ph) delta (default 8; 0 = disable). ph eff-rank >> pf.")
    parser.add_argument("--rank-pf", dest="svd_rank_pf", type=int, default=0, help="KVCMAS SVD rank for the PREFIX (pf) delta (default 8; 0 = disable). pf eff-rank << ph (often <4), so pf can take a smaller rank.")
    parser.add_argument("--rank-base-key", dest="svd_rank_base_key", type=int, default=32, help="SVD rank for the BASE KEY embedding (prefill-only matching; default None = disabled). base_key eff-rank << base_value, so it compresses well.")
    parser.add_argument("--rank-base-value", dest="svd_rank_base_value", type=int, default=32, help="SVD rank for the BASE VALUE embedding (feeds the entropy gate; default None = disabled). Keep HIGH: base_value is high-rank; too low flattens reuse.")
    parser.add_argument("--hot-anchor-num", dest="hot_anchor_num", type=int, default=10, help="kvcmas-sa selective anchoring: keep top-k anchors per (segment, layer) for delta reconstruction (None disables = all anchors). e.g. 4.")
    parser.add_argument("--no-drop-shared-cache", dest="drop_shared_cache_on_finalize", action="store_false", default=True, help="Disable end-of-request eviction of shared KV buckets (input/response/condition). On by default.")
    parser.add_argument("--num_rounds", type=int, default=1, help="Number of temporal rounds per sample.")

    args = parser.parse_args()
    # NonShared baseline: hold every agent's full KV resident, so peak memory reflects the true
    # N-times-context cost that the sharing methods are compared against.
    if args.execution_mode == "default":
        os.environ.setdefault("KVCMAS_HOLD_KV", "1")
    if args.output_dir is None:
        mode_suffix = "kvcomm" if args.execution_mode == "allow_kv_reuse" else "default"
        if args.execution_mode == "allow_kv_reuse" and (args.svd_rank_ph > 0 or args.svd_rank_pf > 0):
            mode_suffix = f"kvcmas_ph{args.svd_rank_ph}_pf{args.svd_rank_pf}"
        args.output_dir = str(PROJECT_ROOT / "runs" / f"efficiency_{mode_suffix}")
    result_path = Path(args.output_dir)
    result_path.mkdir(parents=True, exist_ok=True)
    if len(args.agent_names) != len(args.agent_nums):
        parser.error("The number of agent names must match the number of agent counts.")
    return args

def _make_random_token_sequence(length: int, index: int = 0) -> str:
    """Deterministic task text of exactly ``length`` tokens, distinct per ``index``. """
    rng = random.Random(0x5EED ^ (index * 2654435761))
    return " ".join(rng.choice(("Δ", "Ω")) for _ in range(length))


def _get_peak_memory_gb() -> float:
    """Return peak GPU memory allocated in GB. Returns 0.0 if CUDA unavailable."""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024 ** 3)


async def evaluate(
        graph: Graph,
        *,
        samples: int,
        warmup: int,
        execution_mode: str,
        output_dir: str,
        num_rounds: int = 1,
        reuse_ratio: float = 1.0,
        **kwargs
        ) -> List[Dict[str, Any]]:

    graph.spatial_logits.requires_grad_ = False
    graph.temporal_logits.requires_grad_ = False

    if execution_mode == "default":
        # NonShared has no anchor pool to build, but keep a 1-sample measurement warmup so first-
        # call one-time costs (flash-attn/CUDA/cuDNN JIT, caching-.
        warmup = 1

    # After the dense (anchor-building) warmup, run one REUSE iteration to warm the FLRA/reuse
    # decode kernel (one-time JIT load + persistent.
    reuse_warm = 1 if (execution_mode == "allow_kv_reuse" and warmup > 0) else 0
    total = warmup + reuse_warm + samples
    user_prompt_len = int(os.environ["USER_PROMPT"])
    # Pass the sample index: without it every task string is identical, and since the task IS the
    # shared-KV bucket key that would fire the standalone.
    data = [{"task": _make_random_token_sequence(user_prompt_len, i)} for i in range(total)]

    # Reset peak memory stats before benchmark
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    _mem_warmup_peak_always = None

    # Force dense_prefill during warmup by setting threshold=0.0 so all
    # warmup samples become anchors (emulates diverse real-world queries).
    original_configs: Dict[str, Any] = {}
    if warmup > 0:
        for node_id, node in graph.nodes.items():
            original_configs[node_id] = node.llm.config
            node.llm.config = replace(node.llm.config, threshold=0.0)

    # Reuse-ratio schedule (steady, allow_kv_reuse only): make (1 - reuse_ratio) of the steady
    # SAMPLES fully dense (every agent re-prefills), the rest.
    _reuse_nodes = list(graph.nodes.items())          # (node_id, node) in agent order 0,1,2,...
    _n_agents = len(_reuse_nodes)
    _dense_samples = 0
    if execution_mode == "allow_kv_reuse" and reuse_ratio < 1.0 and _n_agents > 0:
        _dense_samples = round((1.0 - reuse_ratio) * samples)
        print(f"[REUSE-RATIO] {reuse_ratio:.3f} -> {_dense_samples}/{samples} steady samples fully dense "
              f"({_dense_samples * _n_agents}/{samples * _n_agents} execs); rest fully reuse", flush=True)

    all_results: List[Dict[str, Any]] = []
    for i, input_dict in enumerate(data):
        # Restore original threshold once warmup is done
        if i == warmup and original_configs:
            for node_id, node in graph.nodes.items():
                node.llm.config = original_configs[node_id]

        if i == warmup and torch.cuda.is_available():
            # Steady-only peak: snapshot the warmup peak, then reset so the peak measured from
            # here on is the steady phase alone.
            _mem_warmup_peak_always = torch.cuda.max_memory_allocated() / (1024 ** 3)
            torch.cuda.reset_peak_memory_stats()
        phase = "warmup" if i < warmup else ("reuse_warm" if i < warmup + reuse_warm else "steady")
        print(f"[{phase.upper()} {i+1}/{total}] " + 60*'-')

        # Reuse-ratio schedule (sample-level): the first `_dense_samples` steady samples
        # are fully dense (all agents threshold=0 -> re-prefill); the rest fully reuse.
        if _dense_samples > 0 and phase == "steady":
            _sj = i - warmup - reuse_warm
            _sample_dense = _sj < _dense_samples
            for _nid, _node in _reuse_nodes:
                _thr = 0.0 if _sample_dense else original_configs[_nid].threshold
                _node.llm.config = replace(_node.llm.config, threshold=_thr)
            print(f"[REUSE-RATIO steady j={_sj}] "
                  f"{'DENSE (all agents re-prefill)' if _sample_dense else 'reuse (all agents)'}", flush=True)

        graph.reset_state()
        tasks = [asyncio.create_task(graph.arun(
            input_dict, num_rounds, mode=execution_mode, output_dir=output_dir, **kwargs
        ))]
        raw_results = await asyncio.gather(*tasks)
        all_results.extend(raw_results)

        # Warmup repeatedly allocates and frees the real cache, leaving a large reserved-but-
        # unallocated pool; empty_cache() returns it so the steady peak is not fragmentation.
        if i < warmup and torch.cuda.is_available():
            torch.cuda.empty_cache()
    print("Done!")

    latency_file = Path(output_dir)
    try:
        _write_per_agent_latency(latency_file)
    except Exception as e:
        logger.warning("Failed to write per-agent latency JSONs: {}", e)

    steady_peak_gb = _get_peak_memory_gb()   # post-reset => steady phase only
    peak_memory_gb = max(steady_peak_gb, _mem_warmup_peak_always or 0.0)
    _STEADY_PEAK_GB[0] = steady_peak_gb
    _print_efficiency_summary(latency_file, warmup, total, execution_mode, peak_memory_gb, reuse_warmup=reuse_warm)

    return all_results


def _write_per_agent_latency(latency_file: Path) -> None:
    if not latency_file.exists():
        logger.warning("Latency file not found at {}", str(latency_file))
        return
    try:
        with open(latency_file, "r", encoding="utf-8") as f:
            records = json.load(f)
    except Exception as e:
        logger.warning("Could not read latency file: {}", e)
        return
    by_agent: Dict[str, List[Dict[str, Any]]] = {}
    for rec in records if isinstance(records, list) else []:
        agent_id = rec.get("agent_id") or "unknown"
        by_agent.setdefault(agent_id, []).append(rec)
    out_dir = latency_file.parent / "agent_latency"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = latency_file.stem  # e.g. "latency_260521_133045"
    for agent_id, items in by_agent.items():
        agent_file = out_dir / f"agent_{agent_id}_{stem}.json"
        with open(agent_file, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
    combined = out_dir / f"per_agent_{stem}.json"
    with open(combined, "w", encoding="utf-8") as f:
        json.dump(by_agent, f, ensure_ascii=False, indent=2)


def _print_efficiency_summary(
    latency_file: Path,
    warmup: int,
    total_samples: int,
    execution_mode: str,
    peak_memory_gb: float,
    reuse_warmup: int = 0,
) -> None:
    """Parse latency file and print throughput / TTFT summary for a single execution mode."""
    if not latency_file.exists():
        print(f"No latency file found at {latency_file} -- skipping efficiency summary.")
        return
    with open(latency_file, "r", encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list) or not records:
        print("latency.json is empty -- skipping efficiency summary.")
        return

    # Group records by request_uid
    uid_order: List[str] = []
    uid_to_records: Dict[str, List[Dict[str, Any]]] = {}
    for rec in records:
        uid = rec.get("request_uid", "")
        if uid not in uid_to_records:
            uid_order.append(uid)
            uid_to_records[uid] = []
        uid_to_records[uid].append(rec)

    num_samples_found = len(uid_order)
    if num_samples_found == 0:
        print("No request_uids found -- skipping efficiency summary.")
        return

    # Compute per-sample metrics
    sample_metrics: List[Dict[str, Any]] = []
    for sample_idx, uid in enumerate(uid_order):
        recs = uid_to_records[uid]
        total_input_tokens = 0
        total_output_tokens = 0
        e2e_latency_sum = 0.0
        ttft_values: List[float] = []

        for rec in recs:
            total_input_tokens += rec.get("input_tokens", 0)
            total_output_tokens += rec.get("output_tokens", 0)
            ttft_val = rec.get("ttft")
            if ttft_val is not None:
                ttft_values.append(ttft_val)

            # Use e2e_latency (default mode) or kvcomm_end_to_end_latency (kv_reuse mode from agen_kvcomm)
            if "e2e_latency" in rec:
                e2e_latency_sum += rec["e2e_latency"]
            elif "kvcomm_end_to_end_latency" in rec:
                e2e_latency_sum += rec["kvcomm_end_to_end_latency"]

        total_tokens = total_input_tokens + total_output_tokens
        sample_metrics.append({
            "sample_idx": sample_idx,
            "request_uid": uid,
            "total_tokens": total_tokens,
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "e2e_latency": e2e_latency_sum,
            "mean_ttft": float(np.mean(ttft_values)) if ttft_values else None,
            "num_agents": len(recs),
        })

    # Split into phases
    warmup_samples = [s for s in sample_metrics if s["sample_idx"] < warmup]
    # Samples in [warmup, warmup+reuse_warmup) are the reuse kernel warm-up — excluded
    # from steady so the one-time FLRA JIT/descriptor cost doesn't spike the p99.
    steady_samples = [s for s in sample_metrics if s["sample_idx"] >= warmup + reuse_warmup]

    def _aggregate(samples: List[Dict[str, Any]], label: str) -> Dict[str, Any]:
        agg: Dict[str, Any] = {"label": label, "n": len(samples)}
        if not samples:
            print(f"\n  [{label}] No samples.")
            return agg
        total_tokens = sum(s["total_tokens"] for s in samples)
        total_input = sum(s["input_tokens"] for s in samples)
        total_output = sum(s["output_tokens"] for s in samples)
        n = len(samples)

        e2e_latencies = [s["e2e_latency"] for s in samples if s["e2e_latency"] > 0]
        ttft_values = [s["mean_ttft"] for s in samples if s["mean_ttft"] is not None]

        print(f"\n  [{label}]  samples={n}  total_tokens={total_tokens}  (input={total_input}, output={total_output})")

        if e2e_latencies:
            total_latency = sum(e2e_latencies)
            throughput = total_tokens / total_latency if total_latency > 0 else float("inf")
            print(f"    Throughput  : {throughput:,.1f} tok/s  |  total_latency = {total_latency:.3f}s")
            agg["throughput_tok_s"] = throughput
            agg["total_latency_s"] = total_latency
            agg["avg_latency_per_sample_s"] = total_latency / n

        if ttft_values:
            print(f"    TTFT        : mean = {np.mean(ttft_values):.4f}s  |  p50 = {np.median(ttft_values):.4f}s  |  p99 = {np.percentile(ttft_values, 99):.4f}s")
            agg["ttft_mean"] = float(np.mean(ttft_values))
            agg["ttft_p50"] = float(np.median(ttft_values))
            agg["ttft_p99"] = float(np.percentile(ttft_values, 99))

        return agg

    print("\n" + "=" * 80)
    print(f"EFFICIENCY SUMMARY  (execution_mode={execution_mode})")
    print("=" * 80)
    agg_all = _aggregate(sample_metrics, f"ALL")
    agg_warmup = _aggregate(warmup_samples, f"WARMUP")
    agg_steady = _aggregate(steady_samples, f"STEADY-STATE")

    if peak_memory_gb > 0:
        print(f"\n  [GPU MEMORY]  Peak allocated : {peak_memory_gb:.2f} GB")
    print("=" * 80)


    # Split by agent CALL: agent 0 is the source hop (no sharable cache yet), the rest are
    # the reuse-capable hops.
    def _aid_int(a) -> int:
        a = str(a)
        return int(a) if a.lstrip("-").isdigit() else 1 << 30

    _uid_order: List[str] = []
    for _r in records:
        _u = _r.get("request_uid", "")
        if _u not in _uid_order:
            _uid_order.append(_u)
    _uid_to_idx = {u: i for i, u in enumerate(_uid_order)}
    _source_id = (min((str(r.get("agent_id", "")) for r in records), key=_aid_int)
                  if records else "")
    _skip_upto = warmup + reuse_warmup

    def _lat(rec) -> float:
        for k in ("e2e_latency", "kvcomm_end_to_end_latency"):
            if k in rec and rec[k]:
                return float(rec[k])
        return 0.0

    _src_recs, _proc_recs = [], []
    for _r in records:
        if _uid_to_idx.get(_r.get("request_uid", ""), 0) < _skip_upto:
            continue
        (_src_recs if str(_r.get("agent_id", "")) == _source_id else _proc_recs).append(_r)

    def _aggregate_calls(recs, label: str) -> Dict[str, Any]:
        agg: Dict[str, Any] = {"label": label, "n_calls": len(recs)}
        if not recs:
            print(f"\n  [{label}] No agent calls.")
            return agg
        tok_in = sum(r.get("input_tokens", 0) for r in recs)
        tok_out = sum(r.get("output_tokens", 0) for r in recs)
        lats = [_lat(r) for r in recs if _lat(r) > 0]
        ttfts = [r["ttft"] for r in recs if r.get("ttft") is not None]
        agents = sorted({str(r.get("agent_id", "")) for r in recs}, key=_aid_int)
        print(f"\n  [{label}]  agent_calls={len(recs)}  agents={agents}  "
              f"tokens={tok_in + tok_out} (in={tok_in}, out={tok_out})")
        if lats:
            tot = sum(lats)
            # Output-only throughput: decode tok/s is what the table reports, and mixing the
            # prefilled input tokens in would let a longer shared context inflate.
            agg.update(throughput_tok_s=((tok_in + tok_out) / tot if tot > 0 else float("inf")),
                       throughput_tok_s_output_only=(tok_out / tot if tot > 0 else float("inf")),
                       total_latency_s=tot, avg_latency_per_call_s=tot / len(recs))
            print(f"    Throughput  : {agg['throughput_tok_s']:,.1f} tok/s (in+out)  "
                  f"|  total_latency = {tot:.3f}s")
            # Anchor construction (SVD at dense hops) is inside e2e but outside TTFT; report
            # its share so the two columns can be reconciled.
            anc = sum(float(r.get("anchor_create_latency") or 0.0) for r in recs)
            agg.update(anchor_create_latency_s=anc, anchor_create_share=(anc / tot if tot > 0 else 0.0))
            if anc > 0:
                print(f"    AnchorBuild : {anc:.3f}s inside e2e ({100.0 * anc / tot:.1f}% of total_latency; not in TTFT)")
        if ttfts:
            agg.update(ttft_mean=float(np.mean(ttfts)), ttft_p50=float(np.median(ttfts)),
                       ttft_p99=float(np.percentile(ttfts, 99)))
            print(f"    TTFT        : mean = {agg['ttft_mean']:.4f}s  |  p50 = {agg['ttft_p50']:.4f}s")
        return agg

    agg_src = _aggregate_calls(_src_recs, f"STEADY - source agent {_source_id} (no sharable cache)")
    agg_proc = _aggregate_calls(_proc_recs, "STEADY - downstream agents (reuse consumers)")
    agg_calls_all = _aggregate_calls(_src_recs + _proc_recs, "STEADY - all agents (call-level)")

    summary = {
        "execution_mode": execution_mode,
        "total_samples": total_samples,
        "warmup_samples": warmup,
        "peak_memory_allocated_GB": peak_memory_gb,
        "steady_peak_memory_GB": _STEADY_PEAK_GB[0],
        "source_agent_id": _source_id,
        "aggregated": {
            "all": agg_all,
            "warmup": agg_warmup,
            "steady_state": agg_steady,
            # Same three keys the cacheblend/droidspeak forks emit, so one parser serves every
            # branch.
            "steady_all": agg_calls_all,
            "steady_source_agent": agg_src,
            "steady_proceeding_agents": agg_proc,
        },
        "per_sample": sample_metrics,
    }
    stem = latency_file.stem  # e.g. "latency_260521_133045"
    summary_path = latency_file.parent / f"efficiency_summary_{stem}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nDetailed per-sample metrics saved to: {summary_path}")


async def main():
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    agent_names = [name for name, num in zip(args.agent_names, args.agent_nums) for _ in range(num)]
    kwargs = get_kwargs(args.mode, len(agent_names))
    kv_config = KVCommConfig.from_env().apply_overrides(
        threshold=args.kv_threshold,
        max_anchor_num=args.kv_max_anchor_num,
        window_size=args.kv_window_size,
        thread_pool_workers=args.kv_thread_workers,
        worker_timeout=args.kv_worker_timeout,
        svd_rank_ph=args.svd_rank_ph,
        svd_rank_pf=args.svd_rank_pf,
        svd_rank_base_key=args.svd_rank_base_key,
        svd_rank_base_value=args.svd_rank_base_value,
        hot_anchor_num=args.hot_anchor_num,
        drop_shared_cache_on_finalize=args.drop_shared_cache_on_finalize,
    )

    graph = Graph(
        domain=args.domain,
        llm_name=args.llm_name,
        agent_names=agent_names,
        kv_config=kv_config,
        **kwargs,
    )

    # Warmup defaults to the max anchor pool size so the warmup phase fills the
    # pool exactly to capacity (rather than a separate hardcoded number).
    warmup = args.warmup if args.warmup is not None else kv_config.max_anchor_num

    timestamp = time.strftime("%y%m%d_%H%M%S", time.localtime())
    latency_path = output_dir / f"latency_{timestamp}.json"

    configure_logging(log_path=output_dir / f"logs/log_{timestamp}.txt")

    # True cross-request prefix caching (EFF_TRUE_PREFIX_CACHE=1).
    _prefix_shim = None
    if _prefix_cache_enabled():
        _nodes = list(graph.nodes.values())
        if _nodes:
            _prefix_shim = _install_prefix_shim(_nodes[0].llm.model)
            print(f"[PREFIX-CACHE] enabled: shim installed on the shared model "
                  f"({len(_nodes)} agents share it)", flush=True)
    _ = await evaluate(
        graph=graph,
        samples=args.samples,
        warmup=warmup,
        execution_mode=args.execution_mode,
        output_dir=str(latency_path),
        num_rounds=args.num_rounds,
        reuse_ratio=args.reuse_ratio,
        prefix=args.prefix,
    )

    if _prefix_shim is not None:
        # hits/misses matter: a shim that installed but never hit is worse than absent (it
        # pays a learning forward for nothing), and that failure is silent in the timings.
        print(_prefix_shim.report(), flush=True)


def get_kwargs(
    mode: Union[
        Literal["DirectAnswer"],
        Literal["FullConnected"],
        Literal["Random"],
        Literal["Chain"],
        Literal["Debate"],
        Literal["Layered"],
        Literal["Star"],
        Literal["Mesh"],
    ],
    N: int,
):
    fixed_spatial_masks: List[List[int]] = None
    fixed_temporal_masks: List[List[int]] = None
    node_kwargs = None

    def generate_layered_graph(n, layer_num=2):
        adj_matrix = [[0] * n for _ in range(n)]
        base_size = n // layer_num
        remainder = n % layer_num
        layers: List[int] = []
        for i in range(layer_num):
            size = base_size + (1 if i < remainder else 0)
            layers.extend([i] * size)
        random.shuffle(layers)
        for i in range(n):
            current_layer = layers[i]
            for j in range(n):
                if layers[j] == current_layer + 1:
                    adj_matrix[i][j] = 1
        return adj_matrix

    def generate_mesh_graph(n):
        adj_matrix = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                adj_matrix[i][j] = 1
        return adj_matrix

    def generate_star_graph(n):
        adj_matrix = [[0] * n for _ in range(n)]
        for i in range(1, n):
            adj_matrix[0][i] = 1
        return adj_matrix

    if mode == "DirectAnswer":
        fixed_spatial_masks = [[0]]
        fixed_temporal_masks = [[0]]
        node_kwargs = [{"role": "Normal"}]
    elif mode == "FullConnected":
        fixed_spatial_masks = [[1 if i != j else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1 for _ in range(N)] for _ in range(N)]
    elif mode == "Random":
        fixed_spatial_masks = [[random.randint(0, 1) if i != j else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[random.randint(0, 1) for _ in range(N)] for _ in range(N)]
    elif mode == "Chain":
        fixed_spatial_masks = [[1 if i == j + 1 else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1 if i == 0 and j == N - 1 else 0 for i in range(N)] for j in range(N)]
    elif mode == "Debate":
        fixed_spatial_masks = [[0 for _ in range(N)] for _ in range(N)]
        fixed_temporal_masks = [[1 for _ in range(N)] for _ in range(N)]
    elif mode == "Layered":
        fixed_spatial_masks = generate_layered_graph(N)
        fixed_temporal_masks = [[1 for _ in range(N)] for _ in range(N)]
    elif mode == "Mesh":
        fixed_spatial_masks = generate_mesh_graph(N)
        fixed_temporal_masks = [[1 for _ in range(N)] for _ in range(N)]
    elif mode == "Star":
        fixed_spatial_masks = generate_star_graph(N)
        fixed_temporal_masks = [[1 for _ in range(N)] for _ in range(N)]
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return {
        "fixed_spatial_masks": fixed_spatial_masks,
        "fixed_temporal_masks": fixed_temporal_masks,
        "node_kwargs": node_kwargs,
    }


if __name__ == "__main__":
    asyncio.run(main())
