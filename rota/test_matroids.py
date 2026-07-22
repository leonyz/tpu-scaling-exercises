"""Sanity checks for the engine against matroids with known Whitney numbers."""

from math import comb

from matroids import (
    closure, direct_sum_whitney, dip_score, erected_whitney, flats_by_rank,
    free_erection, graphic_rank, is_unimodal, linear_rank_gf2, linear_rank_gfp,
    paving_rank_from_blocks, popcount, sparse_paving_rank, uniform_rank,
    whitney2,
)


def test_uniform():
    # U_{r,n}: W_k = C(n,k) for k < r, then 1.
    for n, r in [(6, 3), (7, 4), (9, 5)]:
        W = whitney2(n, uniform_rank(r))
        assert W == [comb(n, k) for k in range(r)] + [1], (n, r, W)


def test_fano_and_pg32():
    # Fano = PG(2,2): W = (1,7,7,1). PG(3,2): W = (1,15,35,15,1).
    fano = linear_rank_gf2(list(range(1, 8)))
    assert whitney2(7, fano) == [1, 7, 7, 1]
    pg32 = linear_rank_gf2(list(range(1, 16)))
    assert whitney2(15, pg32) == [1, 15, 35, 15, 1]


def test_graphic_k5():
    # M(K5): lattice = partition lattice Pi_5, W = (1,10,25,15,1).
    edges = [(u, v) for u in range(5) for v in range(u + 1, 5)]
    assert whitney2(10, graphic_rank(5, edges)) == [1, 10, 25, 15, 1]


def test_vamos():
    # Vamos: rank-4 sparse paving on 8 elements, 5 circuit-hyperplanes.
    # W_3 = C(8,3) - 5*(4-1) = 41.
    pair = lambda i: (1 << (2 * i)) | (1 << (2 * i + 1))  # a,b,c,d = 0,1,2,3
    chs = [pair(0) | pair(1), pair(0) | pair(2), pair(0) | pair(3),
           pair(1) | pair(2), pair(1) | pair(3)]  # all but c+d
    W = whitney2(8, sparse_paving_rank(8, 4, chs))
    assert W == [1, 8, 28, 41, 1], W


def test_ag32_paving():
    # AG(3,2) = SQS(8): blocks are 4-subsets of F_2^3 with zero sum.
    from itertools import combinations
    blocks = [sum(1 << x for x in c) for c in combinations(range(8), 4)
              if (lambda s: s[0] ^ s[1] ^ s[2] ^ s[3] == 0)(list(c))]
    assert len(blocks) == 14
    W = whitney2(8, paving_rank_from_blocks(8, 4, blocks))
    assert W == [1, 8, 28, 14, 1], W


def test_unimodality_helpers():
    assert is_unimodal([1, 5, 9, 4, 1])
    assert is_unimodal([1, 5, 5, 1])
    assert not is_unimodal([1, 9, 4, 9, 1])
    assert dip_score([1, 9, 4, 9, 1]) > 1
    assert dip_score([1, 5, 9, 4, 1]) <= 1


def test_direct_sum():
    W = direct_sum_whitney([1, 3, 1], [1, 4, 1])  # U_{2,3} + U_{2,4}
    assert W == [1, 7, 14, 7, 1]
    assert sum(W) == 5 * 6  # product of lattice sizes


def _check_erection_valid(n, rank, r, hyps):
    """Every M-basis lies in exactly one new hyperplane; hyperplanes are
    m_closure-closed sets of M-rank r."""
    from itertools import combinations
    for c in combinations(range(n), r):
        B = sum(1 << i for i in c)
        if rank(B) == r:
            assert sum(1 for H in hyps if B & ~H == 0) == 1, (bin(B), "not in exactly one")
    for H in hyps:
        assert rank(H) == r


def test_free_erection_uniform():
    # Free erection of U_{3,5} is U_{4,5}.
    hyps = free_erection(5, uniform_rank(3), 3)
    assert hyps is not None and len(hyps) == 10  # all 3-subsets
    _check_erection_valid(5, uniform_rank(3), 3, hyps)
    assert erected_whitney(5, uniform_rank(3), 3) == [1, 5, 10, 10, 1]


def test_free_erection_pg22():
    # Erecting the rank-3 paving matroid whose lines are the lines of PG(2,2)
    # should NOT exist inside rank 4 with small hyperplanes... compute and
    # check validity of whatever comes out.
    fano = linear_rank_gf2(list(range(1, 8)))
    hyps = free_erection(7, fano, 3)
    if hyps is not None:
        _check_erection_valid(7, fano, 3, hyps)
        print("Fano free erection hyperplanes:", len(hyps))
    else:
        print("Fano is not erectable")


def test_axiom_checker():
    from matroids import verify_matroid_axioms
    assert verify_matroid_axioms(6, uniform_rank(3))
    fano = linear_rank_gf2(list(range(1, 8)))
    assert verify_matroid_axioms(7, fano)
    # a non-matroid rank function must be caught
    bad = lambda S: popcount(S) if popcount(S) != 2 else 1
    try:
        verify_matroid_axioms(5, bad)
        assert False, "axiom checker missed a non-matroid"
    except AssertionError as e:
        if "missed" in str(e):
            raise


if __name__ == "__main__":
    import sys
    mod = sys.modules["__main__"]
    for name in sorted(dir(mod)):
        if name.startswith("test_"):
            getattr(mod, name)()
            print(f"ok  {name}")
    print("all tests passed")
