"""Random search for counterexamples to Fox's trapezoidal conjecture.

Samples random prime alternating knots (spherogram's random 4-valent planar
map model), computes the exact Alexander polynomial via Fox calculus, and
checks the coefficient shape.  Everything interesting is appended to JSONL
files in the output directory:

  counterexample.jsonl   -- unimodality violations (disproves Fox!)
  strict.jsonl           -- flat spot off the central plateau
                            (violates the strict trapezoidal form)
  logconcave.jsonl       -- log-concavity violations (Stoimenow's conjecture)
  db.jsonl.gz            -- every sampled knot: n, coefficients, PD code
                            (fuel for the connected-sum product attack)

Any unimodality violation is immediately re-verified with the independent
Seifert-matrix computation before being reported.
"""

import argparse
import gzip
import json
import random
import sys
import time
from collections import defaultdict

import spherogram

from alexander import alexander_seifert, alexander_wirtinger, analyze


def lc_ratio(b):
    """max of b[k-1]*b[k+1] / b[k]^2; >= 1 means a log-concavity violation."""
    worst = 0.0
    for k in range(1, len(b) - 1):
        if b[k]:
            worst = max(worst, b[k - 1] * b[k + 1] / (b[k] * b[k]))
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default="24,32,40,48,56,64",
                    help="comma-separated diagram sizes to request")
    ap.add_argument("--min-n", type=int, default=12,
                    help="discard prime pieces smaller than this")
    ap.add_argument("--seconds", type=float, default=600)
    ap.add_argument("--twist", action="store_true",
                    help="generate diagrams with consistent twist regions")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True, help="output directory")
    args = ap.parse_args()

    import os
    os.makedirs(args.out, exist_ok=True)
    targets = [int(x) for x in args.targets.split(",")]
    rng = random.Random(args.seed)
    random.seed(args.seed)  # spherogram uses the global random module

    db = gzip.open(f"{args.out}/db.jsonl.gz", "at")
    files = {}

    def emit(kind, rec):
        if kind not in files:
            files[kind] = open(f"{args.out}/{kind}.jsonl", "a")
        files[kind].write(json.dumps(rec) + "\n")
        files[kind].flush()

    t0 = time.time()
    count = 0
    per_n = defaultdict(int)
    min_margin, min_margin_rec = None, None
    max_lc, max_lc_rec = 0.0, None
    not_alternating = 0
    last_report = t0

    while time.time() - t0 < args.seconds:
        target = rng.choice(targets)
        try:
            L = spherogram.random_link(target, num_components=1, alternating=True,
                                       consistent_twist_regions=args.twist)
        except Exception:
            continue
        n = len(L.crossings)
        if n < args.min_n:
            continue
        if not L.is_alternating():
            not_alternating += 1
            continue
        pd = [list(t) for t in L.PD_code()]
        try:
            coeffs = alexander_wirtinger(pd)
        except (ValueError, AssertionError) as e:
            emit("anomaly", {"pd": pd, "error": str(e)})
            continue
        info = analyze(coeffs)
        count += 1
        per_n[n] += 1
        rec = {"n": n, "coeffs": coeffs, "pd": pd}
        db.write(json.dumps(rec) + "\n")

        # structural theorems for alternating knots -- failure means a bug
        if not (info["palindromic"] and info["signs_alternate"]
                and abs(sum(coeffs)) == 1):
            check = alexander_seifert(L)
            emit("anomaly", {**rec, "seifert_check": check,
                             "info": {k: v for k, v in info.items() if k != "coeffs"}})
            continue

        b = [abs(c) for c in coeffs]
        r = lc_ratio(b)
        if r > max_lc:
            max_lc, max_lc_rec = r, rec
        m = info["margin"]
        if m is not None and (min_margin is None or m < min_margin):
            min_margin, min_margin_rec = m, rec

        if not info["unimodal"]:
            check = alexander_seifert(L)
            emit("counterexample", {**rec, "seifert_check": check,
                                    "confirmed": check == coeffs})
            print(f"!!! UNIMODALITY VIOLATION n={n} coeffs={coeffs} "
                  f"seifert_confirms={check == coeffs}", flush=True)
        elif not info["trapezoidal"]:
            emit("strict", rec)
            print(f"** strict-form violation n={n} coeffs={coeffs}", flush=True)
        if not info["log_concave"]:
            emit("logconcave", {**rec, "lc_ratio": r})
            print(f"* log-concavity violation n={n} ratio={r:.4f}", flush=True)

        if time.time() - last_report > 60:
            last_report = time.time()
            db.flush()
            rate = count / (time.time() - t0)
            sizes = sorted(per_n.items())
            big = [f"{k}:{v}" for k, v in sizes if k >= 30][-8:]
            print(f"[{time.time()-t0:7.0f}s] {count} knots ({rate:.1f}/s) "
                  f"min_margin={min_margin} max_lc={max_lc:.4f} "
                  f"nonalt={not_alternating} big_n={big}", flush=True)

    db.close()
    summary = {
        "count": count,
        "per_n": dict(per_n),
        "min_margin": min_margin,
        "min_margin_rec": min_margin_rec,
        "max_lc_ratio": max_lc,
        "max_lc_rec": max_lc_rec,
        "not_alternating_skipped": not_alternating,
    }
    with open(f"{args.out}/summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ("min_margin_rec", "max_lc_rec")}))


if __name__ == "__main__":
    sys.exit(main())
