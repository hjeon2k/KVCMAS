#!/usr/bin/env python
"""Render the LKV QPS sweep: one table per percentile, methods grouped by family. """

from __future__ import annotations

import argparse
import glob
import json
import os

ORDER = [
    ("nonshared", "NonShared", None),
    ("fullshared", "FullShared", None),
    ("droidspeak", "DroidSpeak", "recompute"),
    ("cacheblend", "CacheBlend", "recompute"),
    ("relaycaching", "RelayCaching", "recompute"),
    ("graphflow", "GraphFlow", "delta"),
    ("kvcomm", "KVComm", "delta"),
    ("kvcmas", "KVCMAS", "delta"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--pct", default="p10,p50,p90,p99")
    args = ap.parse_args()

    got = {}
    for f in glob.glob(os.path.join(args.dir, "*.json")):
        try:
            d = json.load(open(f))
        except Exception:  # noqa: BLE001
            continue
        if "sweep" in d:
            got[(d["method"], round(float(d["p"]), 3))] = d

    ps = sorted({k[1] for k in got})
    rates = []
    for d in got.values():
        for r in d["sweep"]:
            if r["qps"] not in rates:
                rates.append(r["qps"])
    rates.sort()
    if not rates:
        print("no results")
        return 1

    for pct in args.pct.split(","):
        print(f"\n=== TTFT {pct} (s) vs offered load (QPS), 17.3k context ===")
        print("  " + f"{'method':<15}{'p':>5}" + "".join(f"{q:>9g}" for q in rates))
        for key, label, fam in ORDER:
            for p in ps:
                d = got.get((key, p))
                if d is None:
                    if fam is None and p != ps[0]:
                        continue          # p-independent methods print once
                    continue
                cells = {r["qps"]: r[pct] for r in d["sweep"]}
                row = "".join(f"{cells[q]:>9.3f}" if q in cells else f"{'--':>9}"
                              for q in rates)
                tag = "-" if fam is None else f"{p:g}"
                print(f"  {label:<15}{tag:>5}{row}")
                if fam is None:
                    break
    # p-independent methods (NonShared, FullShared) are run once and are not "missing"
    # at the other p values.
    missing = [f"{lbl} p={p:g}" for k, lbl, fam in ORDER for p in ps
               if fam is not None and (k, p) not in got]
    if any(fam is None and not any((k, p) in got for p in ps) for k, _l, fam in ORDER):
        missing += [l for k, l, fam in ORDER
                    if fam is None and not any((k, p) in got for p in ps)]
    if missing:
        print("\n  missing: " + ", ".join(missing))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
