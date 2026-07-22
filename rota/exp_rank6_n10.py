"""Cost model + pruning experiments for an exhaustive n=10, rank-6 search.

Part 1 benchmarks the two per-matroid costs an exhaustive sweep would pay:
  - unimodality check via the generic flats engine (rank oracle)
  - unimodality check via a 1024-entry rank table (what a C sweep would do:
    a matroid on 10 elements IS a 1KB lookup table)

Part 2 implements the truncation prune. Any rank-6 counterexample M on 10
elements is simple (else its simplification lives on <= 9 elements, all
verified) and dips exactly at k=4: W_3 > W_4 < W_5. Its rank-5 truncation
T(M) has the same W_0..W_4, and M is an erection of T(M), so
W_5(M) <= H_free(T(M)) — the free erection has the finest hyperplane
partition (Crapo). Hence:

    a rank-6 counterexample on 10 elements exists
    iff some simple rank-5 matroid on 10 elements has
    W_4 < W_3 and free-erection hyperplane count H_free > W_4.

So we never need to enumerate rank-6 matroids at all: enumerate thin-top
rank-5 matroids and free-erect each once. Here we sweep the sparse-paving
cell of that space: rank-5 sparse paving matroids need Z >= 23
circuit-hyperplanes (W_4 = 210 - 4Z < 120 = W_3), and Z <= A(10,4,5) = 36,
achieved by the residual of the Witt design S(4,5,11).
"""

import random
import time
from itertools import combinations
from math import comb

from matroids import (
    dip_score, erected_whitney, is_unimodal, popcount, sparse_paving_rank,
    whitney2,
)
from search import ternary_golay_hexads

N = 10
R5_SETS = [sum(1 << i for i in c) for c in combinations(range(N), 5)]


# ---------------------------------------------------------------------------
# Part 1: per-matroid cost of the unimodality check
# ---------------------------------------------------------------------------

def random_sparse_paving_chs(n, r, rng):
    chosen = []
    cands = [sum(1 << i for i in c) for c in combinations(range(n), r)]
    rng.shuffle(cands)
    for c in cands:
        if all(popcount(c & o) <= r - 2 for o in chosen):
            chosen.append(c)
    return chosen


def rank_table(n, rank):
    return bytes(rank(S) for S in range(1 << n))


def whitney_from_table(n, table):
    """The inner loop a compiled exhaustive sweep would run: S is a flat iff
    every e outside S raises the rank. ~n * 2^n table lookups."""
    W = [0] * (table[(1 << n) - 1] + 1)
    for S in range(1 << n):
        rS = table[S]
        for e in range(n):
            b = 1 << e
            if not (S & b) and table[S | b] == rS:
                break
        else:
            W[rS] += 1
    return W


def part1_benchmark():
    print("[1] per-matroid cost, rank-6 matroids on n=10")
    rng = random.Random(1)
    ms = []
    for _ in range(30):
        chs = random_sparse_paving_chs(10, 6, rng)
        ms.append(sparse_paving_rank(10, 6, chs))

    t0 = time.perf_counter()
    for rank in ms:
        W = whitney2(10, rank)
        assert is_unimodal(W)
    t_oracle = (time.perf_counter() - t0) / len(ms)

    tables = [rank_table(10, rank) for rank in ms]
    t0 = time.perf_counter()
    for tb in tables:
        W = whitney_from_table(10, tb)
        assert is_unimodal(W)
    t_table = (time.perf_counter() - t0) / len(ms)

    # cross-check the two paths agree
    for rank, tb in zip(ms, tables):
        assert whitney2(10, rank) == whitney_from_table(10, tb)

    ops = 10 * 1024  # table lookups per matroid
    print(f"  flats-engine check : {t_oracle*1e3:8.2f} ms/matroid")
    print(f"  rank-table check   : {t_table*1e3:8.2f} ms/matroid in Python "
          f"({ops} table lookups -> ~2-5 us/matroid compiled)")
    for name, t in [("python table", t_table), ("compiled est.", 3e-6)]:
        total = 4.9e9 * t
        print(f"  4.9e9 matroids via {name:14s}: {total/3600:10.1f} core-hours")


# ---------------------------------------------------------------------------
# Part 2: the truncation prune, swept over the sparse-paving cell
# ---------------------------------------------------------------------------

def witt_residual_36():
    """Blocks of S(4,5,11) avoiding point 10: the optimal constant-weight
    code A(10,4,5) = 36, i.e. the max sparse-paving CH family."""
    hexads = ternary_golay_hexads()
    blocks11 = [h & ~(1 << 11) for h in hexads if h >> 11 & 1]
    res = [b for b in blocks11 if not (b >> 10 & 1)]
    assert len(res) == 36
    assert all(popcount(a & b) <= 3 for a in res for b in res if a != b)
    return res


def greedy_stable_set(rng, target=None):
    """Random greedy + (1-out, 2-in) local search on the Johnson graph J(10,5)."""
    cands = R5_SETS[:]
    rng.shuffle(cands)
    S = []
    for c in cands:
        if all(popcount(c & o) <= 3 for o in S):
            S.append(c)
    improved = True
    while improved and (target is None or len(S) < target):
        improved = False
        for out in list(S):
            rest = [x for x in S if x != out]
            adds = [c for c in cands if c not in rest
                    and all(popcount(c & o) <= 3 for o in rest)]
            two = [(a, b) for i, a in enumerate(adds) for b in adds[i + 1:]
                   if popcount(a & b) <= 3]
            if two:
                a, b = rng.choice(two)
                S = rest + [a, b]
                improved = True
                break
    return S


def part2_truncation_prune():
    print("\n[2] truncation prune: sparse-paving rank-5 cell on n=10")
    print("    need Z >= 23 (thin: W_4 = 210-4Z < 120) and H_free > W_4")
    witt = witt_residual_36()
    rng = random.Random(5)

    results = {}  # Z -> (tried, erectable_count, best_H, threshold)
    trials = []
    # subsets of the Witt-optimal code cover Z = 23..36 systematically
    for Z in range(23, 37):
        for _ in range(25 if Z < 36 else 1):
            trials.append(("witt-sub", rng.sample(witt, Z)))
    # independent greedy/local-search families for structural diversity
    for _ in range(60):
        S = greedy_stable_set(rng)
        if len(S) >= 23:
            trials.append(("greedy", S))

    t0 = time.perf_counter()
    n_erectable = 0
    for kind, chs in trials:
        Z = len(chs)
        rank = sparse_paving_rank(10, 5, chs)
        W = whitney2(10, rank)
        assert W == [1, 10, 45, 120, 210 - 4 * Z, 1], (Z, W)
        EW = erected_whitney(10, rank, 5)
        tried, erect, best = results.get(Z, (0, 0, -1))
        if EW is not None:
            n_erectable += 1
            H = EW[5]
            best = max(best, H)
            erect += 1
            if not is_unimodal(EW):
                print(f"  COUNTEREXAMPLE: Z={Z} {kind}: {EW}")
                raise SystemExit(1)
        results[Z] = (tried + 1, erect, best)
    dt = time.perf_counter() - t0

    print(f"  {len(trials)} rank-5 sparse pavings free-erected in {dt:.1f}s "
          f"({dt/len(trials)*1e3:.0f} ms each)")
    print(f"  erectable: {n_erectable}/{len(trials)}")
    print("   Z  tried  erectable  best H_free  needed (>W_4)")
    for Z in sorted(results):
        tried, erect, best = results[Z]
        thr = 210 - 4 * Z
        b = best if best >= 0 else "-"
        print(f"  {Z:2d}  {tried:5d}  {erect:9d}  {str(b):>11s}  {thr}")


def part3_erectability_frontier():
    """Where does erectability die as circuit-hyperplanes accumulate?
    Thinness needs Z >= 23; measure H_free for Z = 0..22 to see how far
    below that the erection property already fails."""
    print("\n[3] erectability frontier in Z (thinness needs Z >= 23)")
    EW = erected_whitney(10, sparse_paving_rank(10, 5, []), 5)
    assert EW == [1, 10, 45, 120, 210, 252, 1]  # U(5,10) erects to U(6,10)
    print("   Z=0: U(5,10) -> U(6,10), H_free = 252 (engine sanity check)")

    witt = witt_residual_36()
    rng = random.Random(9)
    print("   Z  tried  erectable  H_free range   W_4 (dip needs H > W_4)")
    for Z in range(1, 23):
        outcomes = []
        for t in range(12):
            if t % 2 == 0:
                chs = rng.sample(witt, Z)
            else:
                S = greedy_stable_set(rng)
                if len(S) < Z:
                    continue
                chs = rng.sample(S, Z)
            EW = erected_whitney(10, sparse_paving_rank(10, 5, chs), 5)
            outcomes.append(EW[5] if EW else None)
        er = [h for h in outcomes if h is not None]
        span = f"{min(er)}..{max(er)}" if er else "-"
        print(f"  {Z:3d}  {len(outcomes):5d}  {len(er):9d}  {span:>12s}   "
              f"{210 - 4 * Z}")


if __name__ == "__main__":
    part1_benchmark()
    part2_truncation_prune()
    part3_erectability_frontier()
