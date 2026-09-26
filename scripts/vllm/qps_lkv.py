#!/usr/bin/env python
"""QPS sweep on vLLM for the KVCMAS method families. """

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Which trace variant a method's reused requests behave like, and how p is spent.
FAMILY = {
    "nonshared":    ("none", None),
    "fullshared":   ("none", None),
    "droidspeak":   ("layerwise", 1.0),    # index of the first critical layer
    "cacheblend":   ("tokenwise", 1.0),    # 1-layer check band
    "relaycaching": ("tokenwise", 4.0),    # 4-layer check band
    "graphflow":    ("delta_encode", "none"),   # per-request context-free base encode
    "kvcomm":       ("delta_encode", "full"),   # full-dimensional standalone base encode
    "kvcmas":       ("delta", "lowrank"),
}
N_LAYERS, N_KV, HEAD_DIM = 32, 8, 128
SYSTEM_PROMPT = 1024           # copy_machine.py pf0; the base segment, never delta-corrected
ANCHORS, RANK = 10, 32          # --kv-max-anchor-num 10 --hot-anchor-num 10, rank-ph 32


def prefill_cost_fraction(fam, knob, p, block_size):
    """Fraction of the handed-over span a method must actually re-prefill. """
    if fam == "delta_encode":
        # The base encode is a full prefill of the shared span, and it recurs on every
        # reusing request because the span changes per request.
        return 1.0
    if fam == "layerwise":
        return (N_LAYERS - knob) / N_LAYERS
    if fam == "tokenwise":
        pages = 1.0 - (1.0 - p) ** block_size
        band = knob / N_LAYERS
        return band + (1.0 - band) * pages
    return 0.0


def _pct(xs, q):
    if not xs:
        return 0.0
    ys = sorted(xs)
    k = (len(ys) - 1) * q / 100.0
    lo = int(k)
    return ys[lo] if lo + 1 >= len(ys) else ys[lo] + (ys[lo + 1] - ys[lo]) * (k - lo)


def load_trace(path):
    return [r for r in json.load(open(path)) if r["agent"] != "e2e"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=sorted(FAMILY))
    ap.add_argument("--p", type=float, required=True)
    ap.add_argument("--trace-dir", required=True)
    ap.add_argument("--context", type=int, default=32768, help="UP, the shared question span")
    ap.add_argument("--decode", type=int, default=128, help="GEN, each agent's relayed output")
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--gpu-mem-util", type=float, default=0.30)
    ap.add_argument("--qps-list", default="0.25,0.5,1,2,4,8")
    ap.add_argument("--repeats", type=int, default=5,
                    help="repetitions of each rate run; percentiles pool their latencies")
    ap.add_argument("--qps-requests", type=int, default=68)
    ap.add_argument("--qps0-requests", type=int, default=0,
                    help="requests in the qps=0 column; 0 = one trajectory")
    ap.add_argument("--out", default="")
    ap.add_argument("--pf-rank", type=int, default=0,
                    help="KVCMAS only: rank of the PREFIX (pf) delta, i.e. --rank-pf with "
                         "KVCMAS_PF_DELTA=1. 0 (default) = the shipped setting, prefix reused "
                         "rotate-only with no correction. A reusing hop then pays a SECOND "
                         "low-rank reconstruction, over its pf span: the template text after "
                         "the question (priv + glue + tail), which the trace gives as "
                         "`prefill - SYSTEM_PROMPT` -- the system prompt is the base segment "
                         "and never carries a delta.")
    ap.add_argument("--anchor-svd", choices=["batched", "per-layer", "none"], default="batched",
                    help="KVCMAS only: the anchor construction a DENSE hop pays after it "
                         "finishes (truncated SVD of delta K/V and base K/V, 32 layers, per "
                         "placeholder). It runs synchronously and holds the GPU before the "
                         "next request can be served, so at qps>0 it is charged as scheduler "
                         "time (every in-flight request's TTFT sees it); it is never added to "
                         "the dense request's own first-token time, matching the engine's "
                         "e2e-not-TTFT accounting. batched = the engine's layer-batched "
                         "range finder (default); per-layer = torch.svd_lowrank per layer, "
                         "the pre-2026-09-20 engine; none = the old harness (not charged).")
    ap.add_argument("--delta-mode", choices=["chained", "nonchained"], default="chained",
                    help="KVCMAS only. chained: every hop corrects against the PREVIOUS "
                         "hop's KV, one low-rank delta per hop, no encode (the shipped "
                         "method). nonchained: the first agent also runs a context-free "
                         "encode of the shared span (the base), every later hop's delta is "
                         "taken against that base, and materialising it means hop i must "
                         "first SUBTRACT hop i-1's delta from the cache and then ADD its "
                         "own -- two reconstructions per hop after the first.")
    args = ap.parse_args()

    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    import torch
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    fam, knob = FAMILY[args.method]
    nonchained = args.delta_mode == "nonchained"
    if nonchained:
        if args.method != "kvcmas":
            raise SystemExit("--delta-mode nonchained only applies to --method kvcmas")
        # Non-chained KVCMAS has the SAME encode shape as GraphFlow/KVComm (one context-free base
        # encode of the shared span.
        fam = "delta_encode"
    # The copy_machine layout (pf0 system | ph0 question | priv | (glue + output)*), as
    # scripts/vllm/make_traces.py writes it: make_traces.py --prefill CONTEXT --decode DECODE.
    _tag = f"eff_{args.context}_d{args.decode}"
    a_rows = load_trace(os.path.join(args.trace_dir, f"{_tag}_a.json"))
    b_rows = load_trace(os.path.join(args.trace_dir, f"{_tag}_b.json"))
    turns = b_rows if args.method == "nonshared" else a_rows
    max_total = max(r["total"] for r in a_rows)

    # Handed-over span: what another agent contributed, i.e. what a reusing method gets
    # for free and a dense one must redo. It is exactly the gap between the variants.
    handover = [max(0, tb["prefill"] - ta["prefill"]) for ta, tb in zip(a_rows, b_rows)]

    frac = prefill_cost_fraction(fam, knob, args.p, args.block_size)
    encode_span = 0
    if fam == "delta_encode":
        # ONE context-free encode of the shared question span, charged to hop 0.
        _ca = [r["cached"] for r in a_rows]
        _up = (2 * _ca[1] - _ca[2]) if len(_ca) > 2 else (_ca[1] if len(_ca) > 1 else 0)
        encode_span = max(0, _up)
        recompute = [0] * len(handover)
    else:
        recompute = [int(round(h * frac)) for h in handover]
    print(f"[model] {args.method} p={args.p} family={fam} knob={knob} "
          f"delta_mode={args.delta_mode} prefill_cost_fraction={frac:.4f} "
          f"tokens={sum(recompute)}/{sum(handover)} encode_span={encode_span}",
          flush=True)

    llm = LLM(model=args.model, max_model_len=max_total + 256,
              block_size=args.block_size, enable_prefix_caching=True,
              gpu_memory_utilization=args.gpu_mem_util,
              compilation_config={"cudagraph_mode": "PIECEWISE"}, disable_log_stats=True)
    engine = llm.llm_engine

    rng = random.Random(0)
    dev, dt = "cuda", torch.bfloat16
    D = N_KV * HEAD_DIM

    # Correction buffers, allocated once. Sizing them by the longest reused span is the
    # honest worst case and keeps allocation out of the timed region.
    corr = None
    if fam.startswith("delta") and knob != "none":
        T = max_total
        if knob == "full":
            corr = {"W": torch.randn(ANCHORS, device=dev, dtype=dt),
                    "Dm": torch.randn(ANCHORS, T, D, device=dev, dtype=dt)}
        else:
            corr = {"W": torch.randn(ANCHORS, device=dev, dtype=dt),
                    "A": torch.randn(ANCHORS, T, RANK, device=dev, dtype=dt),
                    "B": torch.randn(RANK, D, device=dev, dtype=dt)}

    # Prefix-delta buffers (--pf-rank > 0): same weighted-sum-of-low-rank-factors shape as
    # the placeholder delta, at the prefix rank and over the pf span.
    pf_corr = None
    if fam.startswith("delta") and knob == "lowrank" and args.pf_rank > 0:
        pf_corr = {"W": torch.randn(ANCHORS, device=dev, dtype=dt),
                   "A": torch.randn(ANCHORS, max_total, args.pf_rank, device=dev, dtype=dt),
                   "B": torch.randn(args.pf_rank, D, device=dev, dtype=dt)}

    def correction(ntok: int) -> None:
        """The per-layer delta reconstruction a reusing request pays."""
        if corr is None or ntok <= 0:
            return
        for _ in range(N_LAYERS):
            if knob == "full":
                torch.einsum("a,atd->td", corr["W"], corr["Dm"][:, :ntok])
            else:
                torch.einsum("a,atr->tr", corr["W"], corr["A"][:, :ntok]) @ corr["B"]

    def pf_correction(ntok: int) -> None:
        """The prefix-span delta a reusing hop adds when --pf-rank > 0."""
        if pf_corr is None or ntok <= 0:
            return
        for _ in range(N_LAYERS):
            torch.einsum("a,atr->tr", pf_corr["W"], pf_corr["A"][:, :ntok]) @ pf_corr["B"]

    # Anchor construction a dense hop pays: one upstream-output span per preceding hop.
    svd_src = None
    if fam.startswith("delta") and knob == "lowrank" and args.anchor_svd != "none":
        svd_src = torch.randn(max_total, D, device=dev, dtype=dt)

    def _orth(Y):
        for _ in range(2):
            G = Y.transpose(-1, -2) @ Y
            eye = torch.eye(G.shape[-1], device=G.device, dtype=G.dtype)
            G = G + eye * (1e-6 * G.diagonal(dim1=-2, dim2=-1).mean(-1, keepdim=True).unsqueeze(-1))
            L, info = torch.linalg.cholesky_ex(G)
            if bool((info != 0).any()):
                return torch.linalg.qr(Y).Q
            Y = torch.linalg.solve_triangular(L.transpose(-1, -2), Y, upper=True, left=False)
        return Y

    def _rsvd(M, rank, niter=2):
        # Mirrors KVCMAS/llm/kvcmas_engine.py::_rsvd_batched; keep the two in sync.
        r = max(1, min(rank, M.shape[-2], M.shape[-1]))
        Q = _orth(M @ torch.randn(M.shape[-1], r, device=M.device, dtype=M.dtype))
        for _ in range(niter):
            Q = _orth(M @ _orth(M.transpose(-1, -2) @ Q))
        return Q, Q.transpose(-1, -2) @ M

    def anchor_build(spans) -> None:
        """The factorization a dense hop pays for its `spans` (token counts)."""
        if svd_src is None:
            return
        budget = 1 << 30
        for ntok in spans:
            ntok = int(min(ntok, max_total))
            if ntok <= 0:
                continue
            for _ in range(4):                       # delta K, delta V, base K, base V
                if args.anchor_svd == "per-layer":
                    for _l in range(N_LAYERS):
                        torch.svd_lowrank(svd_src[:ntok].float(), q=min(RANK, ntok))
                else:
                    C = max(1, min(N_LAYERS, budget // (ntok * D * 4)))
                    for l0 in range(0, N_LAYERS, C):
                        c = min(C, N_LAYERS - l0)
                        M = svd_src[:ntok].unsqueeze(0).expand(c, ntok, D).float()
                        _rsvd(M, RANK)
                        del M

    traj_pool = {}

    def stream(k):
        if k not in traj_pool:
            traj_pool[k] = [rng.randrange(1000, 30000) for _ in range(max_total + 16)]
        return traj_pool[k]

    sp1 = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True)

    def build(n_req):
        """Per hop: (tokens, is_dense, reused_tokens, recompute_tokens, sub_tokens, encode). """
        out = []
        prev_span = 0                 # span of the delta currently materialised in the cache
        for k in range(n_req):
            i = k % len(turns)
            t = turns[i]
            end = t["cached"] + t["prefill"]
            if i == 0:
                prev_span = 0         # new request: cache is the fresh base, nothing applied
            # NonShared reuses nothing.
            dense = args.method == "nonshared"
            if fam.startswith("delta"):
                # N*p of the requests run fully dense. Deterministic count, spread
                # evenly so a rate run does not put them all in one burst.
                dense = (int(k * args.p) != int((k - 1) * args.p)) if k else (args.p > 0)
            if args.method == "nonshared":
                dense = True
            toks = stream(k // len(turns))[:end]
            if dense:
                toks = stream(10_000 + k)[:end]      # unseen -> forced full prefill
            elif recompute[i]:
                # The recompute rides on the MEASURED request, and it REPLACES the tail rather
                # than extending it: the request keeps its real length.
                rc = min(recompute[i], end)
                toks = toks[:end - rc] + stream(20_000 + k)[:rc]
            reused = 0 if dense else t["cached"]
            # Materialisation.
            sub = prev_span if (nonchained and reused and prev_span) else 0
            if dense:
                prev_span = 0
            elif reused:
                prev_span = reused
            enc = (fam == "delta_encode" and i == 0 and encode_span > 0)
            # Anchor construction owed by a dense hop: one span per upstream output it
            # embeds (hop i embeds i of them), each a decode length.
            anc = [turns[j]["decode"] for j in range(i)] if (dense and fam.startswith("delta")) else []
            # pf span of this hop: the template text after the question. SYSTEM_PROMPT is the
            # base segment (no delta), so prefill - SYSTEM is priv + glue*i + tail.
            pf = max(0, t["prefill"] - SYSTEM_PROMPT) if (not dense and reused) else 0
            out.append((toks, dense, reused, recompute[i], sub, enc, anc, pf))
        return out

    def run_rate(qps):
        engine.reset_prefix_cache()
        n0 = args.qps0_requests if args.qps0_requests > 0 else len(turns)
        reqs = build(args.qps_requests if qps > 0 else n0)
        if qps == 0:
            # No queueing: one request at a time, on the same engine path and clock as the
            # rate runs.
            lat = []
            for j, (toks, dense, reused, rc, sub, enc, anc, pf) in enumerate(reqs):
                if not dense and reused:
                    if sub:
                        correction(sub)          # remove the previous hop's delta
                    correction(reused)           # apply this hop's delta
                    pf_correction(pf)            # and the prefix delta, when --pf-rank > 0
                t0 = time.perf_counter()
                if enc:
                    # context-free base encode: its own short sequence, unseen, so it
                    # attends only over itself -- exactly what a standalone encode does.
                    engine.add_request(f"e{j}", TokensPrompt(
                        prompt_token_ids=stream(30_000 + j)[:encode_span]), sp1)
                engine.add_request(f"r{j}", TokensPrompt(prompt_token_ids=toks), sp1)
                first_t = {}
                while engine.has_unfinished_requests():
                    for o in engine.step():
                        rid = getattr(o, "request_id", None)
                        if rid is not None and rid not in first_t:
                            first_t[rid] = time.perf_counter() - t0
                v = first_t.get(f"r{j}", 0.0)
                if enc and f"e{j}" in first_t:
                    v = max(v, first_t[f"e{j}"])   # hop 0 is not usable until its base exists
                lat.append(v)
                if dense and anc:
                    anchor_build(anc)            # after the hop; outside its own latency
                    torch.cuda.synchronize()
            return lat
        arrival, first, pending_anchor = {}, {}, {}
        t0 = time.perf_counter()
        nxt, last = 0, time.perf_counter()
        while nxt < len(reqs) or engine.has_unfinished_requests():
            now = time.perf_counter() - t0
            while nxt < len(reqs) and nxt / qps <= now:
                toks, dense, reused, rc, sub, enc, anc, pf = reqs[nxt]
                # The correction has to finish before the reused KV can be attended to, so it is
                # charged to this request's TTFT (arrival is the ideal send time.
                if not dense and reused:
                    if sub:
                        correction(sub)
                    correction(reused)
                    pf_correction(pf)
                if enc:
                    # The base encode is submitted as its own request at the same arrival; hop 0's
                    # TTFT is taken as the LATER of the two first tokens (see the collection.
                    engine.add_request(f"e{nxt}", TokensPrompt(
                        prompt_token_ids=stream(30_000 + nxt)[:encode_span]), sp1)
                    arrival[f"e{nxt}"] = nxt / qps
                engine.add_request(f"r{nxt}", TokensPrompt(prompt_token_ids=toks), sp1)
                arrival[f"r{nxt}"] = nxt / qps
                if dense and anc:
                    pending_anchor[f"r{nxt}"] = anc
                nxt += 1
                last = time.perf_counter()
            if engine.has_unfinished_requests():
                for o in engine.step():
                    rid = getattr(o, "request_id", None)
                    if rid in arrival and rid not in first:
                        first[rid] = time.perf_counter() - t0 - arrival[rid]
                        last = time.perf_counter()
                        anc = pending_anchor.pop(rid, None)
                        if anc:
                            # The dense hop is done (max_tokens=1): its anchor construction now
                            # holds the GPU.
                            anchor_build(anc)
                            torch.cuda.synchronize()
            elif nxt < len(reqs):
                time.sleep(min(0.002, max(0.0, nxt / qps - (time.perf_counter() - t0))))
                last = time.perf_counter()
            if time.perf_counter() - last > 180:
                raise RuntimeError(f"stalled: {nxt}/{len(reqs)} sent, {len(first)} done")
        out = []
        for k in sorted((r for r in first if r[0] == "r"), key=lambda r: int(r[1:])):
            v = first[k]
            e = "e" + k[1:]
            if e in first:
                v = max(v, first[e])     # hop 0 is not usable until its base exists
            out.append(v)
        return out

    run_rate(0)                                     # warm
    sweep = []
    for q in [float(x) for x in args.qps_list.split(",")]:
        lat, runs = [], []
        for _rep in range(max(1, args.repeats)):
            l = run_rate(q)
            lat += l
            runs.append({"p50": round(_pct(l, 50), 4), "p90": round(_pct(l, 90), 4)})
        sweep.append({"qps": q, "n": len(lat), "repeats": max(1, args.repeats),
                      "p10": round(_pct(lat, 10), 4), "p50": round(_pct(lat, 50), 4),
                      "p90": round(_pct(lat, 90), 4), "p99": round(_pct(lat, 99), 4),
                      "runs": runs})
        _r50 = [r["p50"] for r in runs]; _r90 = [r["p90"] for r in runs]
        print(f"  qps={q}: n={len(lat)} ({len(runs)} runs) p50={sweep[-1]['p50']:.4f} "
              f"p90={sweep[-1]['p90']:.4f} | per-run p50 {min(_r50):.4f}-{max(_r50):.4f} "
              f"p90 {min(_r90):.4f}-{max(_r90):.4f}", flush=True)
    res = {"method": args.method, "p": args.p, "family": fam, "knob": knob,
           "delta_mode": args.delta_mode, "anchor_svd": args.anchor_svd,
           "pf_rank": args.pf_rank,
           "context": args.context, "sweep": sweep,
           "recompute_tokens": sum(recompute), "handover": sum(handover),
           "encode_span": encode_span}
    print("RESULT " + json.dumps(res))
    if args.out:
        json.dump(res, open(args.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
