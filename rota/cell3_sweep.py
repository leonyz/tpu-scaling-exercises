"""Cell 3: non-paving rank-5 truncations on 10 elements.

The theorem in theorem_cells12.md closes the paving cells. What remains:
simple rank-5 NON-paving matroids T on 10 elements with W_4 < W_3, whose
free erection has H > W_4. Structural prunes proved here-ish:

  * (rank argument) T is non-erectable if it has a rank-4 flat with >= 9
    points, since a rank-4 flat of the erection has <= 8 points.
  * (V9-contraction prune) in a counterexample N, every single-element
    contraction N/x is a matroid whose simplification has <= 9 elements;
    granting the verified n<=9 catalogue, each flat-count vector
    (#rank-k flats of N containing x)_k must itself be unimodal.

Sweeps below:
  A. random GF(2)-represented rank-5 simple matroids on 10 points
  B. random GF(3)/GF(5) ditto
  C. truncations of random rank-6 GF(2) matroids — these T's are erectable
     BY CONSTRUCTION, so any thin one is a live counterexample candidate:
     check H_free vs W_4 directly.
  D. structured configurations engineered for thinness (points packed
     into few rank-4 flats), the n=10 analogue of truncated PG(5,2).
"""

import random
import time
from itertools import combinations

from matroids import (
    dip_score, erected_whitney, flats_by_rank, free_erection, is_unimodal,
    linear_rank_gf2, linear_rank_gfp, popcount, truncate, uniform_rank,
    whitney2,
)

STATS = {}


def is_paving_rank5(n, rank):
    """Paving iff every 4-set is independent."""
    for c in combinations(range(n), 4):
        if rank(sum(1 << i for i in c)) < 4:
            return False
    return True


def contraction_vectors(n, rank):
    """For each point x: (#flats of rank k containing x)_k. In a genuine
    rank-6 counterexample each such vector must be unimodal (n<=9 data)."""
    levels = flats_by_rank(n, rank)
    out = []
    for x in range(n):
        b = 1 << x
        out.append([sum(1 for F in lvl if F & b) for lvl in levels])
    return out


def process_T(desc, n, rank, log_all=False):
    """Thinness-filter a rank-5 matroid; free-erect if thin. Returns the
    erected Whitney vector if a dip candidate materializes."""
    W = whitney2(n, rank)
    if len(W) != 6:
        return None
    key = desc.split()[0]
    s = STATS.setdefault(key, {"seen": 0, "thin": 0, "erectable_thin": 0})
    s["seen"] += 1
    thin = W[4] < W[3]
    if not thin:
        return None
    s["thin"] += 1
    EW = erected_whitney(n, rank, 5)
    if EW is None:
        if log_all:
            print(f"    {desc}: thin W={W}, not erectable")
        return None
    s["erectable_thin"] += 1
    print(f"    {desc}: thin W={W} -> erected {EW} "
          f"(H={EW[5]} vs W_4={EW[4]})")
    if not is_unimodal(EW):
        print(f"    *** COUNTEREXAMPLE: {desc}: {EW}")
        raise SystemExit(1)
    return EW


# ---------------------------------------------------------------------------

def sweep_gf2(trials, rng):
    print(f"\n[A] random GF(2) rank-5 matroids on 10 points ({trials} trials)")
    for i in range(trials):
        cols = tuple(rng.sample(range(1, 32), 10))
        rank = linear_rank_gf2(cols)
        if rank(1023) != 5:
            continue
        if is_paving_rank5(10, rank):
            continue  # paving cell: closed by theorem
        process_T(f"GF(2) cols={cols}", 10, rank)


def sweep_gfp(p, trials, rng):
    print(f"\n[B] random GF({p}) rank-5 matroids on 10 points ({trials} trials)")
    # projective points: normalize first nonzero coordinate to 1
    pts = []
    for v in range(1, p ** 5):
        vec = tuple((v // p ** i) % p for i in range(5))
        first = next(x for x in vec if x)
        if first == 1:
            pts.append(vec)
    for i in range(trials):
        cols = tuple(rng.sample(pts, 10))
        rank = linear_rank_gfp(cols, p)
        if rank(1023) != 5:
            continue
        if is_paving_rank5(10, rank):
            continue
        process_T(f"GF({p}) sample#{i}", 10, rank)


def sweep_truncations(trials, rng):
    print(f"\n[C] truncations of random rank-6 GF(2) matroids ({trials} trials)")
    print("    (erectable by construction: thin ones are live candidates)")
    for i in range(trials):
        cols = tuple(rng.sample(range(1, 64), 10))
        rank6 = linear_rank_gf2(cols)
        if rank6(1023) != 6:
            continue
        W6 = whitney2(10, rank6)
        assert is_unimodal(W6), (cols, W6)
        T = truncate(rank6, 5)
        process_T(f"trunc6 cols={cols}", 10, T)


def sweep_structured():
    print("\n[D] structured thin attempts")
    rng = random.Random(23)
    # two rank-4 GF(2) subspaces of F_2^5 meeting in a rank-3 subspace:
    # points drawn 5+5 from the two solids -> few rank-4 flats
    A = [v for v in range(1, 32) if v & 16 == 0]          # <e1..e4>: 15 pts
    Bs = [v for v in range(1, 32) if (v & 8 == 0)]        # <e1,e2,e3,e5>
    shared = [v for v in A if v in Bs]
    onlyA = [v for v in A if v not in shared]
    onlyB = [v for v in Bs if v not in shared]
    for t in range(40):
        ka = rng.randrange(3, 7)
        cols = rng.sample(onlyA, ka) + rng.sample(onlyB, 8 - ka) \
            + rng.sample(shared, 2)
        rank = linear_rank_gf2(tuple(cols))
        if rank(1023) != 5 or is_paving_rank5(10, rank):
            continue
        process_T(f"2solids sample#{t}", 10, rank, log_all=(t < 3))

    # cone: point + 9 points in a hyperplane through it (few rank-4 flats
    # by design); and near-pencil style extremes
    for t in range(40):
        base = rng.sample([v for v in range(1, 32) if v & 16 == 0], 9)
        cols = tuple(base + [16 | rng.randrange(0, 16)])
        rank = linear_rank_gf2(cols)
        if rank(1023) != 5 or is_paving_rank5(10, rank):
            continue
        process_T(f"cone sample#{t}", 10, rank)


def check_9point_flat_prune():
    print("\n[E] rank-argument prune: 9-point rank-4 flat => non-erectable")

    def rank(S):  # U_{4,9} plus a coloop on point 9
        return min(popcount(S & 511), 4) + (S >> 9 & 1)

    W = whitney2(10, rank)
    assert W == [1, 10, 45, 120, 210 - 126 + 1, 1], W  # W_4 = 85: thin!
    EW = erected_whitney(10, rank, 5)
    assert EW is None
    print(f"    U(4,9)+coloop: W = {W} (thin, W_4=85) -> not erectable, "
          "as the rank argument demands")


def demo_contraction_prune():
    print("\n[F] V9-contraction prune demo (on U(6,10) as a stand-in N)")
    vecs = contraction_vectors(10, uniform_rank(6))
    assert all(is_unimodal(v) for v in vecs)
    print(f"    all 10 contraction vectors unimodal, e.g. {vecs[0]}")


def main():
    t0 = time.time()
    rng = random.Random(41)
    check_9point_flat_prune()
    demo_contraction_prune()
    sweep_gf2(4000, rng)
    sweep_gfp(3, 700, rng)
    sweep_gfp(5, 300, rng)
    sweep_truncations(3000, rng)
    sweep_structured()
    print(f"\n=== summary ({time.time()-t0:.0f}s) ===")
    for k, s in STATS.items():
        print(f"  {k:8s}: {s['seen']:5d} non-paving rank-5, "
              f"{s['thin']:4d} thin, {s['erectable_thin']:3d} erectable-thin")
    print("no counterexample found" if True else "")


if __name__ == "__main__":
    main()
