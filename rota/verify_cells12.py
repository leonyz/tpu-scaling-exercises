"""Machine verification of every counting ingredient in theorem_cells12.md,
plus randomized cross-validation of the theorem against the cascade code.

Parts:
  A. exact packing numbers on tiny ground sets (exhaustive B&B)
  B. closed 8-/9-point class impossibility: exhaustive max-savings searches
  C. the 10-point case analysis maxima (a = 2, 1 six-blocks)
  D. the cell-1 arithmetic kill
  E. empirical cross-check: random/adversarial paving families with
     savings >= 91 are never erectable; report the erectable frontier.
"""

import random
import time
from itertools import combinations

from matroids import erected_whitney, paving_rank_from_blocks, popcount, whitney2
from exp_rank6_n10 import witt_residual_36


def ksubsets(points, k):
    return [sum(1 << i for i in c) for c in combinations(points, k)]


def max_weight_family(cands, weights, compatible, ub_stop=None):
    """Exhaustive branch-and-bound: maximum total weight of a subfamily of
    `cands` that is pairwise `compatible`. Small instances only."""
    n = len(cands)
    order = sorted(range(n), key=lambda i: -weights[i])
    cands = [cands[i] for i in order]
    weights = [weights[i] for i in order]
    comp = [0] * n
    for i in range(n):
        for j in range(n):
            if i != j and compatible(cands[i], cands[j]):
                comp[i] |= 1 << j
    suffix = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix[i] = suffix[i + 1] + weights[i]
    best = 0

    def dfs(avail, lo, val):
        nonlocal best
        if val > best:
            best = val
            if ub_stop is not None and best >= ub_stop:
                return
        i = lo
        while i < n:
            if not (avail >> i) & 1:
                i += 1
                continue
            rest = val
            a = avail >> i
            j = i
            while a:
                if a & 1:
                    rest += weights[j]
                a >>= 1
                j += 1
            if rest <= best:
                return
            dfs(avail & comp[i], i + 1, val + weights[i])
            avail &= ~(1 << i)
            i += 1

    dfs((1 << n) - 1, 0, 0)
    return best


def meet_le(k):
    return lambda a, b: popcount(a & b) <= k


# ---------------------------------------------------------------------------
def part_a():
    print("[A] packing numbers (exhaustive)")
    a745 = max_weight_family(ksubsets(range(7), 5), [1] * 21, meet_le(3))
    assert a745 == 3, a745
    print(f"  A(7,4,5) = {a745}")
    a845 = max_weight_family(ksubsets(range(8), 5), [1] * 56, meet_le(3))
    assert a845 == 8, a845
    print(f"  A(8,4,5) = {a845}")
    for pts, expect in [(6, 4), (7, 7), (8, 8)]:
        t = max_weight_family(ksubsets(range(pts), 3),
                              [1] * len(ksubsets(range(pts), 3)), meet_le(1))
        assert t == expect, (pts, t)
        print(f"  max triples pairwise<=1 on {pts} points = {t}")
    a844 = max_weight_family(ksubsets(range(8), 4), [1] * 70, meet_le(2))
    assert a844 == 14, a844
    print(f"  A(8,4,4) = {a844}")
    # A(9,4,5) <= A(8,4,4) + A(8,4,5) = 22 (point split); true value is 18
    print(f"  A(9,4,5) <= {a844} + {a845} = {a844 + a845}  (need <= 22)")


# ---------------------------------------------------------------------------
def savings(block):
    from math import comb
    return comb(popcount(block), 4) - 1


def big_block_candidates(points, sizes):
    out = []
    for s in sizes:
        out.extend(ksubsets(points, s))
    return out


def max5(points, fixed_blocks, extra=None):
    """Exact max number of 5-blocks within `points`, pairwise meet <= 3 and
    meeting every block in `fixed_blocks` in <= 3 (or an `extra` filter)."""
    cands = [s for s in ksubsets(points, 5)
             if all(popcount(s & b) <= 3 for b in fixed_blocks)
             and (extra is None or extra(s))]
    if not cands:
        return 0
    return max_weight_family(cands, [1] * len(cands), meet_le(3))


def part_b():
    print("\n[B] closed 8/9-point classes force savings < 91 "
          "(subcase-exhaustive)")
    B = lambda pts: sum(1 << i for i in pts)

    # ---- m = 9 (K = points 0..8): every big block lies inside K ----
    # any two 6-subsets of a 9-set meet in >= 3; pairwise <= 3 iff their
    # 3-point complements are disjoint (verify exhaustively):
    sixes = ksubsets(range(9), 6)
    K9 = B(range(9))
    for i, x in enumerate(sixes):
        for y in sixes[i + 1:]:
            assert (popcount(x & y) <= 3) == ((K9 ^ x) & (K9 ^ y) == 0)
    print("  9pts: 6-blocks pairwise<=3 <=> disjoint complements "
          "(so at most 3)")

    cases = {}
    # 8-block: check nothing else fits, savings 69
    b8 = B(range(8))
    assert max5(range(9), [b8]) == 0
    assert all(popcount(b8 & s) > 3 for s in ksubsets(range(9), 6))
    cases["8-block"] = 69
    # 7-block: no 6-block fits; exact max 5-blocks
    b7 = B(range(7))
    assert all(popcount(b7 & s) > 3 for s in ksubsets(range(9), 6))
    cases["7-block"] = 34 + 4 * max5(range(9), [b7])
    # three 6-blocks (complements = disjoint triples, unique up to iso)
    T = [B([0, 1, 2]), B([3, 4, 5]), B([6, 7, 8])]
    fixed3 = [K9 ^ t for t in T]
    cases["three 6-blocks"] = 42 + 4 * max5(range(9), fixed3)
    cases["two 6-blocks"] = 28 + 4 * max5(range(9), fixed3[:2])
    cases["one 6-block"] = 14 + 4 * max5(range(9), fixed3[:1])
    # pure 5-blocks: A(9,4,5) <= A(8,4,4) + A(8,4,5) = 22 (part A)
    cases["pure 5-blocks"] = 4 * 22
    for name, sav in cases.items():
        assert sav < 91, (name, sav)
        print(f"  9pts, {name:15s}: max savings {sav:3d}  (< 91)")

    # ---- m = 8 (K = 0..7, complement {u,v}) ----
    # 5-blocks not inside K contain both u,v; their triples inside K are
    # pairwise <= 1: at most 8 of them (part A), contributing <= 32,
    # independent of the inside configuration. Inside K:
    cases = {}
    b7 = B(range(7))
    assert max5(range(8), [b7]) == 0
    assert all(popcount(b7 & s) > 3 for s in ksubsets(range(8), 6))
    cases["7-block"] = 34 + 32
    h6 = B(range(6))
    assert all(popcount(h6 & s) > 3 for s in ksubsets(range(8), 6)
               if s != h6)
    cases["6-block"] = 14 + 4 * max5(range(8), [h6]) + 32
    cases["pure 5-blocks"] = 4 * 8 + 32  # A(8,4,5) = 8 from part A
    for name, sav in cases.items():
        assert sav < 91, (name, sav)
        print(f"  8pts, {name:15s}: max savings {sav:3d}  (< 91)")


# ---------------------------------------------------------------------------
def part_c():
    print("\n[C] 10-point case analysis (6-blocks present)")
    # a = 2: two 6-blocks meeting in exactly 2 cover all 10 points; no
    # 5-block can meet both in <= 2. Verify directly.
    h1 = sum(1 << i for i in range(6))            # {0..5}
    h2 = sum(1 << i for i in [0, 1, 6, 7, 8, 9])  # meets h1 in {0,1}
    ok5 = [s for s in ksubsets(range(10), 5)
           if popcount(s & h1) <= 2 and popcount(s & h2) <= 2]
    assert not ok5
    print("  a=2: no compatible 5-block exists; savings = 28 < 91")

    # a = 1: 5-blocks meet H6 in <= 2, pairwise <= 3: exact max
    cands = [s for s in ksubsets(range(10), 5) if popcount(s & h1) <= 2]
    c1 = max_weight_family(cands, [1] * len(cands), meet_le(3))
    assert c1 <= 13, c1
    print(f"  a=1: max 5-blocks = {c1} (needs >= 20; savings <= "
          f"{14 + 4 * c1} < 91)")


# ---------------------------------------------------------------------------
def part_d():
    print("\n[D] cell-1 (a=0, sparse paving) arithmetic")
    for Z in range(23, 37):
        min_pairs = 10 * Z - 120  # t_tau in {1,2} spread; t_tau <= 3
        cap = 2 * Z               # Lemma C: <= 4 per CH, halved
        assert min_pairs > cap, Z
    print("  10Z-120 > 2Z for all Z in [23,36]: contradiction confirmed")


# ---------------------------------------------------------------------------
def random_paving_family(rng, six_blocks):
    """Random paving block family on 10 points: `six_blocks` 6-blocks plus a
    greedy random maximal family of compatible 5-blocks."""
    blocks = []
    all6 = ksubsets(range(10), 6)
    rng.shuffle(all6)
    for b in all6:
        if len(blocks) == six_blocks:
            break
        if all(popcount(b & o) <= 3 for o in blocks):
            blocks.append(b)
    all5 = ksubsets(range(10), 5)
    rng.shuffle(all5)
    for s in all5:
        if all(popcount(s & o) <= 3 for o in blocks):
            blocks.append(s)
    return blocks


def part_e(trials=400):
    print("\n[E] cross-validation against the cascade code")
    rng = random.Random(17)
    t0 = time.time()
    checked = thin = 0
    best_erectable_sav = -1
    witt = witt_residual_36()
    for i in range(trials):
        kind = i % 4
        if kind == 0:
            blocks = random_paving_family(rng, 0)
        elif kind == 1:
            blocks = random_paving_family(rng, 1)
        elif kind == 2:
            blocks = random_paving_family(rng, 2)
        else:
            blocks = rng.sample(witt, rng.randrange(10, 37))
        sav = sum(savings(b) for b in blocks)
        rank = paving_rank_from_blocks(10, 5, blocks)
        EW = erected_whitney(10, rank, 5)
        checked += 1
        if sav >= 91:
            thin += 1
            assert EW is None, (sorted(blocks), EW)  # the theorem
        elif EW is not None:
            best_erectable_sav = max(best_erectable_sav, sav)
    print(f"  {checked} random paving families in {time.time()-t0:.0f}s; "
          f"{thin} thin (sav>=91): all non-erectable  [theorem holds]")
    print(f"  largest savings seen on an ERECTABLE family: "
          f"{best_erectable_sav}  (theorem forbids >= 91)")


if __name__ == "__main__":
    part_a()
    part_b()
    part_c()
    part_d()
    part_e()
    print("\nall verifications passed")
