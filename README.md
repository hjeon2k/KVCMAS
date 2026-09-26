# KVCMAS

KV cache reuse for prompt-specialized multi-agent systems: an agent reuses the cache a
previous agent produced, corrected by a low-rank delta matched against an anchor pool.

| Method | Switch |
|---|---|
| NonShared | `nonshared` |
| KVCMAS | `kvcmas` |

## Layout

```
KVCMAS/llm/kvcmas_engine.py   anchor pool, matching, low-rank delta, reuse gate
KVCMAS/llm/gpt_chat.py        prompt assembly, prefill/decode paths
KVCMAS/agents/                agent roles and the decision node
experiments/                  benchmark runners and the single-stream harness
scripts/                      entry points for the three experiments
```

## Install

```bash
# accuracy and single-stream (Python 3.10, CUDA 12.x)
python3.10 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export HF_HOME=/path/to/hf_cache

# concurrent serving: a vLLM environment, kept on PATH when running (its JIT needs ninja)
```

Video-MME needs its clips first: `python datasets/videomme/download_videos.py`.

## Accuracy

```bash
bash scripts/run_accuracy.sh gsm8k kvcmas 0
bash scripts/run_accuracy.sh gsm8k nonshared 1
```

`<benchmark> <method> <gpu>`, benchmark one of `gsm8k | mmlu | humaneval | mathvista |
videomme`, which fixes the model and the agent roster. Every hyperparameter is a runner
default and can be overridden by appending the flag.

## Efficiency

```bash
# single-stream: up sweeps the shared span, decode sweeps the relayed agent output
UP_VALUES=32768 bash scripts/run_efficiency.sh up kvcmas 0
UP_VALUES=32768 bash scripts/run_efficiency.sh up nonshared 1

# concurrent serving (vLLM environment), <method>@<p> with rho = 1 - p
VLLM_PYTHON=/path/to/.venv-vllm/bin/python CONTEXT=32768 DECODE=128 \
  bash scripts/vllm/run_serving.sh 0 kvcmas@0.2 nonshared@0.2
```

Both need an idle GPU. The serving sweep runs 68 requests per rate, repeated 5 times
(`REPEATS`), and pools the repetitions for the percentiles; `scripts/vllm/report_lkv.py`
tabulates the cells. The trajectory for a (`CONTEXT`, `DECODE`) point is built on first use by
`scripts/vllm/make_traces.py --prefill <CONTEXT> --decode <DECODE>` into `traces/`.

## Acknowledgements

Built on the open-source code of [KVCOMM](https://github.com/FastMAS/KVCOMM),
[GPTSwarm](https://github.com/metauto-ai/GPTSwarm) and
[AgentPrune](https://github.com/yanweiyue/AgentPrune).
