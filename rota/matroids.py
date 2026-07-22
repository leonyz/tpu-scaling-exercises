"""Core machinery for hunting counterexamples to Rota's unimodality conjecture.

Everything works over bitmask subsets of {0, ..., n-1}. A matroid is given by a
rank oracle rank(mask) -> int. From that we compute closures, the full lattice
of flats stratified by rank, and the Whitney numbers of the second kind
W_k = #{flats of rank k}.

Rota's conjecture: (W_0, ..., W_r) is unimodal for every matroid.
A counterexample needs an interior dip: W_{k-1} > W_k < W_{k+1} for some k.
"""

from functools import lru_cache
from itertools import combinations
from math import comb


def popcount(x: int) -> int:
    return x.bit_count()


def bits(mask: int):
    while mask:
        b = mask & -mask
        yield b.bit_length() - 1
        mask ^= b


# ---------------------------------------------------------------------------
# Flats and Whitney numbers from a rank oracle
# ---------------------------------------------------------------------------

def closure(n, rank, S):
    """cl(S) = {e : rank(S + e) = rank(S)}."""
    r = rank(S)
    T = S
    for e in range(n):
        b = 1 << e
        if not (T & b) and rank(T | b) == r:
            T |= b
    return T


def flats_by_rank(n, rank):
    """BFS up the lattice of flats: every rank-(k+1) flat is cl(F + e) for some
    rank-k flat F. Returns a list of sets of masks, indexed by rank."""
    bottom = closure(n, rank, 0)
    levels = [{bottom}]
    full = (1 << n) - 1
    while full not in levels[-1]:
        nxt = set()
        for F in levels[-1]:
            rf = rank(F)
            for e in range(n):
                b = 1 << e
                if not (F & b):
                    G = closure(n, rank, F | b)
                    if rank(G) == rf + 1:
                        nxt.add(G)
        levels.append(nxt)
    return levels


def whitney2(n, rank):
    return [len(level) for level in flats_by_rank(n, rank)]


def is_unimodal(W):
    seq = list(W)
    i = 0
    while i + 1 < len(seq) and seq[i + 1] >= seq[i]:
        i += 1
    while i + 1 < len(seq) and seq[i + 1] <= seq[i]:
        i += 1
    return i == len(seq) - 1


def dip_score(W):
    """max over interior k of min(W_{k-1}, W_{k+1}) / W_k.
    A value > 1 means non-unimodal, i.e. a counterexample to Rota."""
    best = 0.0
    for k in range(1, len(W) - 1):
        best = max(best, min(W[k - 1], W[k + 1]) / W[k])
    return best


# ---------------------------------------------------------------------------
# Rank oracles
# ---------------------------------------------------------------------------

def uniform_rank(r):
    return lambda S: min(popcount(S), r)


def sparse_paving_rank(n, r, circuit_hyperplanes):
    """Sparse paving matroid of rank r: every r-set is a basis except the given
    circuit-hyperplanes (r-sets pairwise intersecting in <= r-2 elements)."""
    ch = frozenset(circuit_hyperplanes)

    def rank(S):
        p = popcount(S)
        if p < r:
            return p
        if p == r:
            return r - 1 if S in ch else r
        return r

    return rank


def paving_rank_from_blocks(n, r, blocks):
    """Paving matroid of rank r whose hyperplanes are `blocks` (a d-partition:
    each (r-1)-subset lies in exactly one block, blocks pairwise intersect in
    <= r-2 elements). rank(S) = min(|S|, r) except sets inside a block cap at r-1."""
    blocks = [b for b in blocks]

    def rank(S):
        p = popcount(S)
        if p <= r - 1:
            return p
        for b in blocks:
            if S | b == b:  # S subset of block
                return r - 1
        return r

    return rank


def linear_rank_gf2(columns):
    """Matroid of a set of GF(2) column vectors, each an int bitmask."""

    @lru_cache(maxsize=None)
    def rank(S):
        basis = []
        for i in bits(S):
            v = columns[i]
            for b in basis:
                v = min(v, v ^ b)
            if v:
                basis.append(v)
                basis.sort(reverse=True)
        return len(basis)

    return rank


def linear_rank_gfp(columns, p):
    """Matroid of column vectors (tuples) over GF(p)."""

    @lru_cache(maxsize=None)
    def rank(S):
        rows = [list(columns[i]) for i in bits(S)]
        r = 0
        m = len(columns[0]) if columns else 0
        for c in range(m):
            piv = next((i for i in range(r, len(rows)) if rows[i][c] % p), None)
            if piv is None:
                continue
            rows[r], rows[piv] = rows[piv], rows[r]
            inv = pow(rows[r][c], p - 2, p)
            rows[r] = [(x * inv) % p for x in rows[r]]
            for i in range(len(rows)):
                if i != r and rows[i][c] % p:
                    f = rows[i][c]
                    rows[i] = [(x - f * y) % p for x, y in zip(rows[i], rows[r])]
            r += 1
        return r

    return rank


def graphic_rank(n_vertices, edges):
    """Cycle matroid of a graph. `edges` is a list of (u, v); element i = edge i."""

    def rank(S):
        parent = list(range(n_vertices))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        r = 0
        for i in bits(S):
            u, v = edges[i]
            ru, rv = find(u), find(v)
            if ru != rv:
                parent[ru] = rv
                r += 1
        return r

    return rank


def truncate(rank, k):
    return lambda S: min(rank(S), k)


def direct_sum_whitney(W1, W2):
    """Whitney numbers of a direct sum = convolution of the summands'."""
    out = [0] * (len(W1) + len(W2) - 1)
    for i, a in enumerate(W1):
        for j, b in enumerate(W2):
            out[i + j] += a * b
    return out


# ---------------------------------------------------------------------------
# Free erection (Crapo / Knuth)
# ---------------------------------------------------------------------------
# Given M of rank r, an *erection* is a matroid N of rank r+1 whose truncation
# is M. N's flats of rank <= r-1 are exactly M's, so
#     W(N) = (W_0(M), ..., W_{r-1}(M), H, 1)
# where H is the number of hyperplanes of N. The free erection maximizes H.
#
# Construction: each new hyperplane is a union of M-bases, closed under
# "if T \subseteq K spans an M-hyperplane, then that hyperplane \subseteq K",
# and every M-basis lies in exactly one new hyperplane. Start with each basis
# as its own seed, close, and merge seeds whose closures share a basis, until
# stable. If a class closes to the whole ground set, M has no proper erection.

def m_closure_r1(n, rank, r, hyperplanes, S):
    """Close S under: any subset of S of rank r-1 pulls in its M-hyperplane."""
    changed = True
    while changed:
        changed = False
        for H in hyperplanes:
            if H & ~S and rank(S & H) == r - 1:
                S |= H
                changed = True
    return S


def free_erection(n, rank, r, verbose=False):
    """Return (list of new hyperplanes) or None if M is not erectable.
    M must be given by its rank oracle and have rank r."""
    full = (1 << n) - 1
    levels = flats_by_rank(n, rank)
    hyperplanes = list(levels[r - 1])
    bases = [sum(1 << i for i in c)
             for c in combinations(range(n), r)]
    bases = [B for B in bases if rank(B) == r]

    # union-find over bases
    idx = {B: i for i, B in enumerate(bases)}
    parent = list(range(len(bases)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    closure_of = {}
    changed = True
    while changed:
        changed = False
        groups = {}
        for i, B in enumerate(bases):
            groups.setdefault(find(i), 0)
            groups[find(i)] |= B
        new_closures = {}
        for g, S in groups.items():
            S = m_closure_r1(n, rank, r, hyperplanes, S)
            new_closures[g] = S
            if S == full and len(groups) > 1:
                pass  # keep going; may all merge anyway
        closure_of = new_closures
        # merge any group with any basis contained (and spanning) in another's closure
        for g, S in closure_of.items():
            for j, B in enumerate(bases):
                if B & ~S == 0 and find(j) != g:
                    union(j, g)
                    changed = True
    groups = {}
    for i in range(len(bases)):
        groups.setdefault(find(i), 0)
        groups[find(i)] |= bases[i]
    hyps = sorted({m_closure_r1(n, rank, r, hyperplanes, S)
                   for S in groups.values()})
    if any(h == full for h in hyps):
        return None
    return hyps


def verify_matroid_axioms(n, rank, samples=20000, seed=0):
    """Brute-force check of the rank axioms (for certifying any 'hit'):
    0 <= r(S) <= |S|;  S <= T => r(S) <= r(T);  submodularity
    r(S|T) + r(S&T) <= r(S) + r(T). Exhaustive for n <= 12, sampled above."""
    import random as _random
    rng = _random.Random(seed)
    full = 1 << n
    if full * full <= samples or n <= 6:
        pairs = [(S, T) for S in range(full) for T in range(full)]
    else:
        pairs = [(rng.randrange(full), rng.randrange(full))
                 for _ in range(samples)]
    for S, T in pairs:
        rS, rT = rank(S), rank(T)
        assert 0 <= rS <= popcount(S)
        if T & ~S == 0:
            assert rank(T) <= rS
        assert rank(S | T) + rank(S & T) <= rS + rT, (bin(S), bin(T))
    return True


def erected_whitney(n, rank, r):
    """Whitney numbers of the free erection, or None if not erectable."""
    hyps = free_erection(n, rank, r)
    if hyps is None:
        return None
    W = whitney2(n, rank)
    return W[:-1] + [len(hyps), 1]
