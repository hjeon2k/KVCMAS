#!/usr/bin/env python3
"""Generate efficiency traces for a (PREFILL, DECODE) grid point. """
import argparse, json, os

SYSTEM, PRIV, GLUE, TAIL = 1024, 512, 16, 5


def build(up: int, gen: int, hops: int):
    a, b = [], []
    for i in range(hops):
        prompt = SYSTEM + up + PRIV + (GLUE + gen) * i + TAIL
        cached = (up + gen * i) if i else 0
        # total carries this hop's own decode: the committed eff_8192/eff_32768 traces do,
        # qps_lkv.py sizes max_model_len from the trace, so the field must be present.
        total = prompt + gen
        a.append({"agent": f"agent{i}", "prefill": prompt - cached, "cached": cached,
                  "decode": gen, "total": total})
        b.append({"agent": f"agent{i}", "prefill": prompt, "cached": 0,
                  "decode": gen, "total": total})
    return a, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefill", type=int, required=True, help="UP, the shared question span")
    ap.add_argument("--decode", type=int, default=64, help="GEN, each agent's relayed output")
    ap.add_argument("--hops", type=int, default=4)
    ap.add_argument("--out-dir", default="traces")
    args = ap.parse_args()
    a, b = build(args.prefill, args.decode, args.hops)
    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"eff_{args.prefill}_d{args.decode}"
    for suf, rows in (("a", a), ("b", b)):
        p = os.path.join(args.out_dir, f"{tag}_{suf}.json")
        with open(p, "w") as f:
            json.dump(rows, f, indent=4)
    print(f"{tag}: totals={[r['total'] for r in a]} cached={[r['cached'] for r in a]} "
          f"a_prefill={[r['prefill'] for r in a]}")


if __name__ == "__main__":
    main()
