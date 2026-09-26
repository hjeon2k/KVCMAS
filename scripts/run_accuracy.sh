#!/bin/bash
# Accuracy: one benchmark x one method on one GPU.
#   bash scripts/run_accuracy.sh <gsm8k|mmlu|humaneval|mathvista|videomme> <nonshared|kvcmas> <gpu> [args...]
set -eu
BENCH=${1:?benchmark}; METHOD=${2:?method}; GPU=${3:?gpu}; shift 3

case "$BENCH" in
  gsm8k)     RUNNER=run_gsm8k.py;     AGENTS=MathSolver;   LLM="meta-llama/Llama-3.1-8B-Instruct" ; VLM=0 ;;
  mmlu)      RUNNER=run_mmlu.py;      AGENTS=AnalyzeAgent; LLM="meta-llama/Llama-3.1-8B-Instruct" ; VLM=0 ;;
  humaneval) RUNNER=run_humaneval.py; AGENTS=CodeWriting;  LLM="Qwen/Qwen2.5-Coder-7B-Instruct"   ; VLM=0 ;;
  mathvista) RUNNER=run_mathvista.py; AGENTS=MathSolver;   LLM="llava-hf/llava-onevision-qwen2-7b-ov-hf"; VLM=1 ;;
  videomme)  RUNNER=run_videomme.py;  AGENTS=AnalyzeAgent; LLM="llava-hf/llava-onevision-qwen2-7b-ov-hf"; VLM=1 ;;
  *) echo "unknown benchmark: $BENCH (gsm8k|mmlu|humaneval|mathvista|videomme)" >&2; exit 2 ;;
esac

# Runner defaults: rank 32/0/32/32, pool 10/10, gate .29/.32/.34/.50/.60 (mmlu/gsm8k/
# humaneval/mathvista/videomme). Append a flag to override any of them.
case "$METHOD" in
  nonshared) ARGS="--execution_mode default" ;;
  kvcmas)    ARGS="--execution_mode allow_kv_reuse --chained" ;;
  *) echo "unknown method: $METHOD (nonshared|kvcmas)" >&2; exit 2 ;;
esac

export CUDA_VISIBLE_DEVICES=$GPU
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export KVCMAS_ATTN_IMPL=${KVCMAS_ATTN_IMPL:-flash_attention_2}  # pinned: sdpa moves 15% of VLM predictions

OUT=${OUT:-runs/acc_${BENCH}_${METHOD}}
set -x
python experiments/$RUNNER --mode FullConnected --agent_names "$AGENTS" --agent_nums 3 \
  --llm_name "$LLM" $ARGS --output_dir "$OUT" "$@"
