"""Search campaigns for a counterexample to Rota's unimodality conjecture.

Strategy (see README.md): known theorems make most of matroid space provably
safe. Top-heaviness (Braden-Huh-Matherne-Proudfoot-Wang) applied to
truncations forces W_k <= W_{k+1} whenever 2k+1 <= r, and de Bruijn-Erdos
gives W_1 <= W_2 always; so an interior dip W_{k-1} > W_k < W_{k+1} needs
2k > r+1 and k <= r-2. The MINIMAL possible counterexample is therefore a
rank-6 matroid with its dip at k=4: W_3 > W_4 < W_5 ("many planes, few
rank-4 flats, many hyperplanes"). A thin rank-4 layer forces design-like
structure, so the campaigns center on Steiner designs, the paving matroids
they define, and Crapo/Knuth free erections (which maximize the number of
hyperplanes one rank up — exactly the rise a dip needs).
"""

import random
import time
from itertools import combinations, product
from math import comb

from matroids import (
    dip_score, direct_sum_whitney, erected_whitney, free_erection,
    is_unimodal, linear_rank_gf2, paving_rank_from_blocks, popcount,
    sparse_paving_rank, truncate, uniform_rank, whitney2,
)

NEAR_MISSES = []  # (score, description, W)


def report(desc, W):
    if W is None:
        print(f"  {desc}: not erectable")
        return
    s = dip_score(W)
    NEAR_MISSES.append((s, desc, W))
    flag = "  <-- COUNTEREXAMPLE!" if not is_unimodal(W) else ""
    print(f"  {desc}: W = {W}  dip = {s:.3f}{flag}")
    if not is_unimodal(W):
        raise SystemExit(f"NON-UNIMODAL WHITNEY SEQUENCE FOUND: {desc} {W}")


# ---------------------------------------------------------------------------
# Finite fields and inversive (Mobius) planes S(3, q+1, q^2+1)
# ---------------------------------------------------------------------------

class GF:
    """GF(p^k) as tuples of coefficients mod an irreducible polynomial."""

    def __init__(self, p, irr):
        self.p, self.irr = p, irr
        self.k = len(irr) - 1
        self.q = p ** self.k
        self.elems = [tuple(t) for t in product(range(p), repeat=self.k)]
        self.zero = tuple([0] * self.k)
        self.one = tuple([1] + [0] * (self.k - 1))

    def add(self, a, b):
        return tuple((x + y) % self.p for x, y in zip(a, b))

    def mul(self, a, b):
        k, p = self.k, self.p
        prod = [0] * (2 * k - 1)
        for i, x in enumerate(a):
            if x:
                for j, y in enumerate(b):
                    prod[i + j] = (prod[i + j] + x * y) % p
        for d in range(2 * k - 2, k - 1, -1):
            c = prod[d]
            if c:
                prod[d] = 0
                for j in range(self.k):
                    prod[d - k + j] = (prod[d - k + j] - c * self.irr[j]) % p
        return tuple(prod[:k])

    def inv(self, a):
        for b in self.elems:
            if self.mul(a, b) == self.one:
                return b
        raise ZeroDivisionError

    def subfield_gf_p(self):
        """The prime subfield: constant polynomials."""
        return [tuple([c] + [0] * (self.k - 1)) for c in range(self.p)]


def mobius_blocks(F, sub):
    """Points: elements of F plus 'inf' (index len(F.elems)). Circles: images of
    sub ∪ {inf} under all Mobius maps z -> (az+b)/(cz+d), ad-bc != 0."""
    idx = {e: i for i, e in enumerate(F.elems)}
    INF = len(F.elems)
    base = [idx[e] for e in sub] + [INF]
    inv_table = {e: F.inv(e) for e in F.elems if e != F.zero}

    def apply(a, b, c, d, ptidx):
        if ptidx == INF:
            if c == F.zero:
                return INF
            return idx[F.mul(a, inv_table[c])]
        z = F.elems[ptidx]
        num = F.add(F.mul(a, z), b)
        den = F.add(F.mul(c, z), d)
        if den == F.zero:
            return INF
        return idx[F.mul(num, inv_table[den])]

    blocks = set()
    for a, b, c, d in product(F.elems, repeat=4):
        det = F.add(F.mul(a, d), tuple((-x) % F.p for x in F.mul(b, c)))
        if det == F.zero:
            continue
        blocks.add(frozenset(apply(a, b, c, d, p) for p in base))
    return [sum(1 << i for i in bl) for bl in blocks], INF + 1


def check_steiner_3design(blocks, n, block_size):
    """Every 3-subset of [n] lies in exactly one block."""
    seen = {}
    for bl in blocks:
        elts = [i for i in range(n) if bl >> i & 1]
        assert len(elts) == block_size, (bin(bl), block_size)
        for t in combinations(elts, 3):
            assert t not in seen, f"triple {t} in two blocks"
            seen[t] = bl
    assert len(seen) == comb(n, 3), (len(seen), comb(n, 3))


# ---------------------------------------------------------------------------
# Campaign 1: sparse paving matroids (formula-level; provably safe, verify)
# ---------------------------------------------------------------------------

def campaign_sparse_paving_formula():
    print("\n[1] Sparse paving scan (greedy max circuit-hyperplane packings)")
    print("    W = (binomials..., C(n,r-1) - (r-1)Z, 1); analysis says no dip is possible.")
    rng = random.Random(7)
    worst = None
    for n in range(8, 21):
        for r in range(4, min(n - 3, 9)):
            # greedy random independent set in the Johnson graph J(n, r)
            best_z = 0
            for _ in range(3):
                chosen = []
                cands = list(combinations(range(n), r))
                rng.shuffle(cands)
                for c in cands:
                    m = sum(1 << i for i in c)
                    if all(popcount(m & o) <= r - 2 for o in chosen):
                        chosen.append(m)
                best_z = max(best_z, len(chosen))
            W = [comb(n, k) for k in range(r - 1)] + \
                [comb(n, r - 1) - (r - 1) * best_z, 1]
            s = dip_score(W)
            if worst is None or s > worst[0]:
                worst = (s, f"sparse paving n={n} r={r} Z={best_z}", W)
            assert is_unimodal(W), (n, r, best_z, W)
    NEAR_MISSES.append(worst)
    print(f"  all unimodal; closest call: {worst[1]}: dip = {worst[0]:.3f}")


# ---------------------------------------------------------------------------
# Campaign 2: random represented matroids and their truncations
# ---------------------------------------------------------------------------

def campaign_random_gf2(samples=120):
    print("\n[2] Random GF(2)-represented matroids + truncations")
    rng = random.Random(11)
    worst = (0.0, "", None)
    for _ in range(samples):
        dim = rng.choice([5, 6, 7])
        n = rng.choice([10, 11, 12])
        cols = [rng.randrange(1, 1 << dim) for _ in range(n)]
        rank = linear_rank_gf2(tuple(cols))
        r = rank((1 << n) - 1)
        if r < 4:
            continue
        W = whitney2(n, rank)
        assert is_unimodal(W), (cols, W)
        s = dip_score(W)
        if s > worst[0]:
            worst = (s, f"GF(2) n={n} dim={dim}", W)
        for k in range(4, r):
            Wt = whitney2(n, truncate(rank, k))
            assert is_unimodal(Wt), (cols, k, Wt)
            s = dip_score(Wt)
            if s > worst[0]:
                worst = (s, f"GF(2) n={n} dim={dim} trunc->{k}", Wt)
    NEAR_MISSES.append(worst)
    print(f"  {samples} samples, all unimodal; closest: {worst[1]}: "
          f"W = {worst[2]}  dip = {worst[0]:.3f}")


# ---------------------------------------------------------------------------
# Campaign 3: Steiner-design paving matroids and their free erections
# ---------------------------------------------------------------------------

def campaign_design_erections():
    print("\n[3] Free erections of rank<=4 design pavings (obstruction check)")
    print("    A rank-5 erection of a rank-4 M with W_3 < W_2 would violate the")
    print("    (proven) top-heavy theorem, so these MUST all be non-erectable;")
    print("    confirming that empirically validates the erection machinery.")

    # --- SQS(8) = AG(3,2): blocks are 4-subsets of F_2^3 with zero sum
    blocks8 = [sum(1 << x for x in c) for c in combinations(range(8), 4)
               if c[0] ^ c[1] ^ c[2] ^ c[3] == 0]
    check_steiner_3design(blocks8, 8, 4)
    rank8 = paving_rank_from_blocks(8, 4, blocks8)
    print(f"  SQS(8)=AG(3,2) paving: W = {whitney2(8, rank8)}")
    report("free erection of SQS(8) paving", erected_whitney(8, rank8, 4))

    # --- SQS(10): the Miquelian inversive plane of order 3, S(3,4,10),
    # from PG(1,9) under PGL(2,9). n=10 is BEYOND the exhaustively
    # enumerated range (matroids are only fully enumerated through n=9).
    F9 = GF(3, [1, 0, 1])  # GF(9) = GF(3)[x]/(x^2+1)
    blocks10, n10 = mobius_blocks(F9, F9.subfield_gf_p())
    assert n10 == 10 and len(blocks10) == 30
    check_steiner_3design(blocks10, 10, 4)
    rank10 = paving_rank_from_blocks(10, 4, blocks10)
    print(f"  S(3,4,10) inversive-plane paving: W = {whitney2(10, rank10)}")
    report("free erection of S(3,4,10) paving", erected_whitney(10, rank10, 4))

    # --- S(3,5,17): inversive plane of order 4, from PG(1,16) under PGL(2,16).
    F16 = GF(2, [1, 1, 0, 0, 1])  # GF(16) = GF(2)[x]/(x^4+x+1)
    # the GF(4) subfield = solutions of x^4 = x
    sub4 = [e for e in F16.elems
            if F16.mul(F16.mul(e, e), F16.mul(e, e)) == e]
    assert len(sub4) == 4
    blocks17, n17 = mobius_blocks(F16, sub4)
    assert n17 == 17 and len(blocks17) == 68, (n17, len(blocks17))
    check_steiner_3design(blocks17, 17, 5)
    rank17 = paving_rank_from_blocks(17, 4, blocks17)
    print(f"  S(3,5,17) inversive-plane paving: W = {whitney2(17, rank17)}")
    t0 = time.time()
    report("free erection of S(3,5,17) paving", erected_whitney(17, rank17, 4))
    print(f"    ({time.time() - t0:.1f}s)")

    # --- PG(3,2) itself, erected toward rank 5
    pg32 = linear_rank_gf2(tuple(range(1, 16)))
    report("free erection of PG(3,2)", erected_whitney(15, pg32, 4))

    # --- STS(13) as a rank-3 paving, erected toward rank 4 (pipeline demo;
    # rank 4 cannot violate Rota, but erectability itself is informative)
    base = [(0, 1, 4), (0, 2, 7)]  # cyclic STS(13) difference family mod 13
    blocks13 = sorted({frozenset(((a + s) % 13, (b + s) % 13, (c + s) % 13))
                       for a, b, c in base for s in range(13)})
    assert len(blocks13) == 26
    rank13 = paving_rank_from_blocks(13, 3, [sum(1 << i for i in b)
                                             for b in blocks13])
    print(f"  STS(13) paving: W = {whitney2(13, rank13)}")
    report("free erection of STS(13) paving", erected_whitney(13, rank13, 3))

    # --- STS(15) = lines of PG(3,2): erect rank-3 paving twice
    lines15 = []
    for a in range(1, 16):
        for b in range(a + 1, 16):
            c = a ^ b
            if c > b:
                lines15.append((1 << (a - 1)) | (1 << (b - 1)) | (1 << (c - 1)))
    assert len(lines15) == 35
    rank15 = paving_rank_from_blocks(15, 3, lines15)
    print(f"  STS(15)=PG(3,2)-lines paving: W = {whitney2(15, rank15)}")
    hyps = free_erection(15, rank15, 3)
    if hyps is None:
        print("  STS(15) paving: not erectable")
    else:
        W = whitney2(15, rank15)[:-1] + [len(hyps), 1]
        report("free erection of STS(15) paving", W)
        # erect the erection: rank oracle for the erected matroid
        hypset = list(hyps)

        def rank_erected(S):
            r3 = rank15(S)
            if r3 <= 2:
                return r3
            for H in hypset:
                if S | H == H:
                    return 3
            return 4

        report("double erection of STS(15) paving",
               erected_whitney(15, rank_erected, 4))


# ---------------------------------------------------------------------------
# Campaign 4: direct sums (Whitney = convolution) over an extreme library
# ---------------------------------------------------------------------------

def library_vectors():
    lib = {}
    for q in [2, 3, 4, 5, 7, 8, 9, 11, 13]:  # projective planes
        m = q * q + q + 1
        lib[f"PP({q})"] = [1, m, m, 1]
    for n in [5, 8, 13, 21, 40]:  # near-pencils
        lib[f"NP({n})"] = [1, n, n, 1]
    lib["PG(3,2)"] = [1, 15, 35, 15, 1]
    lib["PG(4,2)"] = [1, 31, 155, 155, 31, 1]
    lib["PG(5,2)"] = [1, 63, 651, 1395, 651, 63, 1]
    lib["PG(3,3)"] = [1, 40, 130, 40, 1]
    lib["PG(4,3)"] = [1, 121, 1210, 1210, 121, 1]
    lib["SQS(8)"] = [1, 8, 28, 14, 1]
    lib["SQS(10)"] = [1, 10, 45, 30, 1]
    lib["SQS(14)"] = [1, 14, 91, 91, 1]
    lib["SQS(16)"] = [1, 16, 120, 140, 1]
    lib["S(3,5,17)"] = [1, 17, 136, 68, 1]
    lib["Vamos"] = [1, 8, 28, 41, 1]
    for n, r in [(6, 3), (8, 4), (10, 5), (12, 4)]:
        lib[f"U({r},{n})"] = [comb(n, k) for k in range(r)] + [1]
    for n in [7, 9, 13, 15, 19, 21, 25]:  # Steiner triple systems as pavings
        lib[f"STS({n})"] = [1, n, n * (n - 1) // 6, 1]
    return lib


def campaign_direct_sums():
    print("\n[4] Direct sums / truncations over an extreme-W library")
    lib = library_vectors()
    names = list(lib)
    worst = (0.0, "", None)
    count = 0
    for i, a in enumerate(names):
        for b in names[i:]:
            for c in [None] + names:
                W = direct_sum_whitney(lib[a], lib[b])
                desc = f"{a} + {b}"
                if c is not None:
                    W = direct_sum_whitney(W, lib[c])
                    desc += f" + {c}"
                variants = [(desc, W)] + [
                    (desc + f" trunc->{k}", W[:k] + [1])
                    for k in range(3, len(W) - 1)]
                for d, V in variants:
                    count += 1
                    assert is_unimodal(V), (d, V)
                    s = dip_score(V)
                    if s > worst[0]:
                        worst = (s, d, V)
    NEAR_MISSES.append(worst)
    print(f"  {count} sums/truncations checked, all unimodal; "
          f"closest: {worst[1]}: dip = {worst[0]:.3f}")


# ---------------------------------------------------------------------------
# Campaign 5: THE FRONTIER — rank-5 matroids with a thin rank-4 layer,
# erected toward rank 6 (the minimal shape a counterexample can have)
# ---------------------------------------------------------------------------

def ternary_golay_hexads():
    """The 132 hexads of S(5,6,12) as supports of weight-6 words of the
    extended ternary Golay code [12,6,6]."""
    A = [[0, 1, 1, 1, 1, 1],
         [1, 0, 1, 2, 2, 1],
         [1, 1, 0, 1, 2, 2],
         [1, 2, 1, 0, 1, 2],
         [1, 2, 2, 1, 0, 1],
         [1, 1, 2, 2, 1, 0]]
    G = [[1 if i == j else 0 for j in range(6)] + A[i] for i in range(6)]
    hexads = set()
    for coeffs in product(range(3), repeat=6):
        w = [sum(c * G[i][j] for i, c in enumerate(coeffs)) % 3
             for j in range(12)]
        supp = [j for j in range(12) if w[j]]
        if len(supp) == 6:
            hexads.add(sum(1 << j for j in supp))
    return sorted(hexads)


def check_steiner_t_design(blocks, n, block_size, t):
    seen = set()
    for bl in blocks:
        elts = [i for i in range(n) if bl >> i & 1]
        assert len(elts) == block_size
        for sub in combinations(elts, t):
            assert sub not in seen, f"{t}-set {sub} in two blocks"
            seen.add(sub)
    assert len(seen) == comb(n, t)


def campaign_thin_rank4_erections():
    print("\n[5] FRONTIER: erecting rank-5 matroids with W_4 < W_3 toward rank 6")
    print("    An erection with H > W_4 would give W_3 > W_4 < H: a dip at k=4,")
    print("    the minimal shape not excluded by known theorems.")

    hexads = ternary_golay_hexads()
    assert len(hexads) == 132
    check_steiner_t_design(hexads, 12, 6, 5)

    # S(4,5,11): derived design (blocks through point 11, with 11 removed)
    blocks11 = [h & ~(1 << 11) for h in hexads if h >> 11 & 1]
    assert len(blocks11) == 66
    check_steiner_t_design(blocks11, 11, 5, 4)

    rank11 = paving_rank_from_blocks(11, 5, blocks11)
    W11 = whitney2(11, rank11)
    print(f"  S(4,5,11) Witt paving (rank 5): W = {W11}")
    report("free erection of S(4,5,11) paving", erected_whitney(11, rank11, 5))

    # relaxations: drop k blocks (each relaxed circuit-hyperplane becomes a
    # basis; slack may let the closure cascade halt)
    rng = random.Random(3)
    for k in [1, 2, 4, 8, 16, 33]:
        keep = rng.sample(blocks11, 66 - k)
        rk = paving_rank_from_blocks(11, 5, keep)
        report(f"erection of S(4,5,11) minus {k} random blocks",
               erected_whitney(11, rk, 5))

    # S(5,6,12): rank-6 paving, erected toward rank 7 (dip would sit at k=5)
    rank12 = paving_rank_from_blocks(12, 6, hexads)
    W12 = whitney2(12, rank12)
    print(f"  S(5,6,12) Witt paving (rank 6): W = {W12}")
    report("free erection of S(5,6,12) paving", erected_whitney(12, rank12, 6))
    for k in [1, 4, 16, 66]:
        keep = rng.sample(hexads, 132 - k)
        rk = paving_rank_from_blocks(12, 6, keep)
        report(f"erection of S(5,6,12) minus {k} random blocks",
               erected_whitney(12, rk, 6))

    # AG(4,2): rank 5, n=16, W = (1,16,120,140,30,1) — very thin top layer
    ag42 = linear_rank_gf2(tuple(16 + x for x in range(16)))
    W16 = whitney2(16, ag42)
    print(f"  AG(4,2) (rank 5): W = {W16}")
    t0 = time.time()
    report("free erection of AG(4,2)", erected_whitney(16, ag42, 5))
    print(f"    ({time.time() - t0:.1f}s)")


# ---------------------------------------------------------------------------

def main():
    t0 = time.time()
    campaign_sparse_paving_formula()
    campaign_random_gf2()
    campaign_design_erections()
    campaign_direct_sums()
    campaign_thin_rank4_erections()
    print(f"\n=== near-miss leaderboard (dip > 1 would refute Rota) ===")
    for s, d, W in sorted(NEAR_MISSES, reverse=True)[:8]:
        print(f"  {s:.3f}  {d}  {W}")
    print(f"\ntotal time: {time.time() - t0:.1f}s — no counterexample found")


if __name__ == "__main__":
    main()
