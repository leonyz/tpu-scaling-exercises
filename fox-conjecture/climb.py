"""Directed hill-climb toward counterexamples of Fox's trapezoidal conjecture.

State: embedded Tait graphs of alternating knots.  Moves: series / parallel
edge extensions applied twice (one full twist), which preserve alternation
and the single-component property while growing genus fast and determinant
slowly -- steering into the high-genus low-determinant region where the
coefficient staircase climbs in the smallest steps.

Objectives, lexicographically:
  margin   -- smallest ascent step before the first coefficient maximum;
              0 kills the (strict) trapezoidal conjecture, < 0 kills even
              unimodality
  lc       -- max off-plateau b[k-1]*b[k+1]/b[k]^2; > 1 kills Stoimenow's
              log-concavity strengthening (and arms the connected-sum attack)

Any margin <= 0 or lc > 1 hit is re-verified with the independent
Seifert-matrix computation and dumped with its PD code.
"""

import argparse
import json
import random
import time
from fractions import Fraction

import spherogram

from alexander import alexander_wirtinger, alexander_seifert, analyze
from taitgraph import PlanarGraph, tait_graph


def off_plateau_lc(b):
    """max b[k-1]b[k+1]/b[k]^2 over triples not entirely at the peak value."""
    peak = max(b)
    best = Fraction(0)
    for k in range(1, len(b) - 1):
        if b[k - 1] == b[k] == b[k + 1] == peak:
            continue
        if b[k]:
            best = max(best, Fraction(b[k - 1] * b[k + 1], b[k] * b[k]))
    return best


def lc_surplus(b):
    """max integer b[k-1]b[k+1] - b[k]^2 over off-plateau triples.

    Log-concavity fails iff this is > 0.  Arithmetic-progression ascents
    sit at exactly -1, the sharpest LC-consistent value, so this is the
    right pressure gauge for the climb (the ratio saturates at 1 on its
    own as coefficients grow)."""
    peak = max(b)
    best = None
    for k in range(1, len(b) - 1):
        if b[k - 1] == b[k] == b[k + 1] == peak:
            continue
        s = b[k - 1] * b[k + 1] - b[k] * b[k]
        if best is None or s > best:
            best = s
    return best if best is not None else -(10 ** 9)


def evaluate(G):
    """Build the diagram, compute Delta, return (score, record) or None."""
    L = G.link()
    if len(L.link_components) != 1:
        return None
    coeffs = alexander_wirtinger(L)
    info = analyze(coeffs)
    if not (info["palindromic"] and info["signs_alternate"]
            and abs(sum(coeffs)) == 1):
        return None  # should not happen; treated as invalid rather than crash
    b = [abs(c) for c in coeffs]
    m = info["margin"]
    if m is None:
        return None
    lc = off_plateau_lc(b)
    # lexicographic score, smaller is better: margin, then LC surplus
    score = (m, -lc_surplus(b), Fraction(info["det"], info["degree"] + 1),
             -info["degree"])
    rec = {"n": len(L.crossings), "coeffs": coeffs, "det": info["det"],
           "degree": info["degree"], "margin": m, "lc": float(lc),
           "unimodal": info["unimodal"], "trapezoidal": info["trapezoidal"],
           "log_concave": info["log_concave"],
           "pd": [list(t) for t in L.PD_code()]}
    return score, rec, L


def mutate(G, rng, moves=1, p_series=0.75):
    H = G.copy()
    for _ in range(moves):
        e = rng.randrange(len(H.edges))
        if rng.random() < p_series:
            H.series(e)
            H.series(len(H.edges) - 1)
        else:
            H.parallel(e)
            H.parallel(len(H.edges) - 1)
    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--seconds", type=float, default=1800)
    ap.add_argument("--max-crossings", type=int, default=90)
    ap.add_argument("--pop", type=int, default=48)
    ap.add_argument("--det-cap-pow", type=float, default=0,
                    help="reject children with det > (degree+2)**pow; 0 = off")
    ap.add_argument("--p-series", type=float, default=0.75)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import os
    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(args.seed)
    random.seed(args.seed)

    hits = open(f"{args.out}/hits.jsonl", "a")

    def report_hit(kind, rec, L):
        check = alexander_seifert(L)
        rec = {**rec, "kind": kind, "seifert_check": check,
               "confirmed": check == rec["coeffs"]}
        hits.write(json.dumps(rec) + "\n")
        hits.flush()
        print(f"!!! {kind} n={rec['n']} confirmed={rec['confirmed']} "
              f"coeffs={rec['coeffs']}", flush=True)

    # seed population from small random alternating knots
    pop = []
    while len(pop) < args.pop // 2:
        try:
            L = spherogram.random_link(rng.choice([8, 10, 12, 14, 16]),
                                       num_components=1, alternating=True)
        except Exception:
            continue
        if not (4 <= len(L.crossings) and len(L.link_components) == 1
                and L.is_alternating()):
            continue
        try:
            G = tait_graph(L)
            r = evaluate(G)
        except AssertionError:
            continue
        if r:
            pop.append((r[0], G))
    pop.sort(key=lambda x: x[0])

    t0 = time.time()
    evals = 0
    best_seen = pop[0][0]
    last_report = t0
    while time.time() - t0 < args.seconds:
        # tournament pick biased to the front, occasional fresh restart
        i = min(rng.randrange(len(pop)), rng.randrange(len(pop)))
        G = pop[i][1]
        H = mutate(G, rng, moves=rng.choice([1, 1, 1, 2, 3]),
                   p_series=args.p_series)
        if len(H.edges) > args.max_crossings:
            continue
        try:
            r = evaluate(H)
        except AssertionError:
            continue
        evals += 1
        if r is None:
            continue
        score, rec, L = r
        if args.det_cap_pow and rec["det"] > (rec["degree"] + 2) ** args.det_cap_pow:
            continue
        if rec["margin"] <= 0:
            report_hit("MARGIN", rec, L)
        if rec["lc"] > 1:
            report_hit("LOGCONCAVITY", rec, L)
        pop.append((score, H))
        pop.sort(key=lambda x: x[0])
        del pop[args.pop:]
        if score < best_seen:
            best_seen = score
            b = [abs(c) for c in rec["coeffs"]]
            print(f"[{time.time()-t0:6.0f}s] best margin={rec['margin']} "
                  f"lc={rec['lc']:.4f} n={rec['n']} deg={rec['degree']} "
                  f"det={rec['det']} b={b}", flush=True)
        if time.time() - last_report > 120:
            last_report = time.time()
            print(f"[{time.time()-t0:6.0f}s] {evals} evals, frontier margin="
                  f"{pop[0][0][0]} surplus={-pop[0][0][1]}", flush=True)

    with open(f"{args.out}/final_pop.jsonl", "w") as f:
        for score, G in pop[:10]:
            r = evaluate(G)
            if r:
                f.write(json.dumps(r[1]) + "\n")
    print(f"done: {evals} evals")


if __name__ == "__main__":
    main()
