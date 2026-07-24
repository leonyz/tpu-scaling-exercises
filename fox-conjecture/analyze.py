"""Aggregate statistics over the search databases -> RESULTS.md numbers."""

import glob
import gzip
import json
import sys
from collections import Counter
from fractions import Fraction

from alexander import analyze
from climb import off_plateau_lc


def load(paths):
    for p in paths:
        try:
            with gzip.open(p, "rt") as f:
                for line in f:
                    try:
                        yield json.loads(line)
                    except Exception:
                        break
        except Exception:
            continue


def main():
    paths = sys.argv[1:] or glob.glob(
        "/tmp/claude-0/*/*/scratchpad/*/db.jsonl.gz")
    total = 0
    n_hist = Counter()
    deg_hist = Counter()
    margin_hist = Counter()
    seen = set()
    max_lc = Fraction(0)
    max_lc_rec = None
    closest = []  # (margin, -degree) best low-margin high-degree specimens
    non_trap = non_uni = non_lc = 0
    max_n = max_deg = max_det = 0
    for rec in load(paths):
        total += 1
        b = tuple(abs(c) for c in rec["coeffs"])
        info = analyze(rec["coeffs"])
        n_hist[rec["n"]] += 1
        deg_hist[info["degree"]] += 1
        max_n = max(max_n, rec["n"])
        max_deg = max(max_deg, info["degree"])
        max_det = max(max_det, info["det"])
        if not info["unimodal"]:
            non_uni += 1
        if not info["trapezoidal"]:
            non_trap += 1
        if not info["log_concave"]:
            non_lc += 1
        m = info["margin"]
        if m is not None:
            margin_hist[min(m, 10)] += 1
        if b in seen:
            continue
        seen.add(b)
        lc = off_plateau_lc(list(b))
        if lc > max_lc:
            max_lc, max_lc_rec = lc, rec
        if m is not None and info["degree"] >= 6:
            closest.append((m, -info["degree"], info["det"], b[:20]))
    closest.sort()
    print(json.dumps({
        "total_knots": total,
        "unique_coeff_seqs": len(seen),
        "max_diagram_crossings": max_n,
        "max_degree": max_deg,
        "max_det": max_det,
        "violations": {"unimodal": non_uni, "trapezoid": non_trap,
                       "log_concave": non_lc},
        "margin_histogram": dict(sorted(margin_hist.items())),
        "max_offplateau_lc_ratio": f"{float(max_lc):.6f}",
        "max_lc_example": (max_lc_rec or {}).get("coeffs", [])[:15],
        "size_histogram_20plus": {k: v for k, v in sorted(n_hist.items())
                                  if k >= 20 and k % 10 == 0},
    }, indent=1))
    print("\nclosest calls (margin, degree, det, |coeffs| prefix):")
    for m, negd, det, bpre in closest[:12]:
        print(f"  margin={m} deg={-negd} det={det} b={list(bpre)}")


if __name__ == "__main__":
    main()
