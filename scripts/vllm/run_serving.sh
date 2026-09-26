#!/bin/bash
# Concurrent-serving sweep, one cell per <method>@<p>. CONTEXT (32768) is the shared span and
# DECODE (128) each agent's output; OUT_DIR (runs/serving) takes the cells.
#   VLLM_PYTHON=/path/to/.venv-vllm/bin/python bash scripts/vllm/run_serving.sh <gpu> kvcmas@0.2 nonshared@0.2
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
V=${VLLM_PYTHON:?set VLLM_PYTHON to the python of your vLLM environment}
export PATH="$(dirname "$V"):$PATH"     # vLLM's JIT kernels need that env's ninja
CTX=${CONTEXT:-32768}; GEN=${DECODE:-128}
case $CTX in 2048) TAG=2k ;; 8192) TAG=8k ;; 32768) TAG=32k ;; *) TAG=$CTX ;; esac
[ "$GEN" = 128 ] || TAG="${TAG}_d${GEN}"
TD=${TRACE_DIR:-traces}
# HOPS 4 = the three task agents plus the FinalRefer hop, the topology the benchmarks run.
[ -f "$TD/eff_${CTX}_d${GEN}_a.json" ] || $V scripts/vllm/make_traces.py \
    --prefill "$CTX" --decode "$GEN" --hops "${HOPS:-4}" --out-dir "$TD"
D=${OUT_DIR:-runs/serving}; mkdir -p "$D"
GPU=$1; shift
for spec in "$@"; do
  M="${spec%%@*}"; P="${spec##*@}"
  CUDA_VISIBLE_DEVICES=$GPU timeout 21600 $V scripts/vllm/qps_lkv.py \
    --method "$M" --p "$P" --trace-dir "$TD" --context "$CTX" --decode "$GEN" --gpu-mem-util 0.30 \
    --qps-requests ${QPS_REQUESTS:-68} --repeats ${REPEATS:-5} \
    --out "$D/${M}_p${P}_${TAG}.json" > "$D/${M}_p${P}_${TAG}.log" 2>&1
  echo "[gpu$GPU] done $M p=$P ctx=$CTX exit=$?"
  sleep 20      # let CUDA memory come back before the next engine sizes its cache
done
