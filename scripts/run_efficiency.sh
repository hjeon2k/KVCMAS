#!/bin/bash
# Efficiency on the controlled CopyMachine trajectory: `up` sweeps the shared USER_PROMPT span,
# `decode` the relayed GENERATED_TEXT. Grid: UP_VALUES / DECODE_VALUES; budget: P.
#   bash scripts/run_efficiency.sh <up|decode> <nonshared|kvcmas> <gpu> [args...]
set -eu
AXIS=${1:?axis}; METHOD=${2:?method}; GPU=${3:?gpu}; shift 3

case "$AXIS" in
  up)     VALUES=${UP_VALUES:-32768}; VAR=USER_PROMPT ;;
  decode) VALUES=${DECODE_VALUES:-"64 128 256 512 1024 2048"};          VAR=GENERATED_TEXT ;;
  *) echo "unknown axis: $AXIS (up|decode)" >&2; exit 2 ;;
esac

# efficiency.py has no --chained flag; chaining is KVCMAS_CHAIN_DELTA, which defaults to 1.
# Pinned here anyway so the arm cannot change under a different environment.
export KVCMAS_CHAIN_DELTA=${KVCMAS_CHAIN_DELTA:-1}
# Ranks and pool come from efficiency.py's defaults (the paper values). The gate is
# left wide open here: on this trace p is realized by --reuse-ratio, not by the gate.
KVCMAS_CFG="--kv-threshold 2.0"

# warmup: KVCMAS needs at least pool-many requests before the anchor pool is full and the
# steady window means anything. NonShared has no pool and needs only JIT/allocator warmth.
case "$METHOD" in
  nonshared) ARGS="--execution_mode default";                          WARMUP=${WARMUP:-1}  ;;
  kvcmas)    ARGS="--execution_mode allow_kv_reuse $KVCMAS_CFG";       WARMUP=${WARMUP:-10} ;;
  *) echo "unknown method: $METHOD (nonshared|kvcmas)" >&2; exit 2 ;;
esac

# p is realized at sample granularity: --reuse-ratio 0.8 runs 8 of 10 samples through the
# reuse path and 2 fully dense. NonShared ignores it (it is dense by definition).
P=${P:-0.2}
[ "$METHOD" = kvcmas ] && ARGS="$ARGS --reuse-ratio $(python -c "print(round(1-$P,2))")"

export CUDA_VISIBLE_DEVICES=$GPU
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for v in $VALUES; do
  OUT=runs/eff_${AXIS}${v}_${METHOD}_p${P}
  echo "=== $AXIS=$v  method=$METHOD  p=$P -> $OUT"
  env "$VAR=$v" python experiments/efficiency.py \
    --mode FullConnected --agent_names CopyMachine --agent_nums 3 --num_rounds 1 \
    --llm_name "${LLM:-meta-llama/Llama-3.1-8B-Instruct}" \
    $ARGS --warmup "$WARMUP" --samples "${SAMPLES:-10}" --output_dir "$OUT" "$@"
done
