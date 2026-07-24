"""Validation of the Alexander pipeline before trusting any 'counterexample'.

  1. Known Alexander polynomials from the Rolfsen table.
  2. Seifert-matrix method vs Wirtinger/Fox-calculus method on random
     alternating knots (independent code paths, must agree).
  3. Euler characteristic of knot Floer homology (third independent
     implementation, from the knot_floer_homology C++ library).
  4. Structural sanity: Delta(1) = +-1 and palindromicity everywhere.
"""

import random
import sys

import spherogram

from alexander import alexander_seifert, alexander_wirtinger, analyze, normalize

ROLFSEN = {
    "3_1": [1, -1, 1],
    "4_1": [1, -3, 1],
    "5_1": [1, -1, 1, -1, 1],
    "5_2": [2, -3, 2],
    "6_1": [2, -5, 2],
    "6_2": [1, -3, 3, -3, 1],
    "6_3": [1, -3, 5, -3, 1],
    "7_1": [1, -1, 1, -1, 1, -1, 1],
    "7_4": [4, -7, 4],
    "7_7": [1, -5, 9, -5, 1],
    "8_18": [1, -5, 10, -13, 10, -5, 1],
}


def hfk_alexander(link):
    """Alexander coefficients from knot Floer homology Euler characteristics."""
    link = link.copy()
    link.simplify("level")  # HFK rejects R1-reducible diagrams
    ranks = link.knot_floer_homology()["ranks"]
    if not ranks:
        return [1]
    amin = min(a for a, m in ranks)
    amax = max(a for a, m in ranks)
    coeffs = [0] * (amax - amin + 1)
    for (a, m), r in ranks.items():
        coeffs[a - amin] += (-1) ** m * r
    return normalize(coeffs)


def delta_at_one(coeffs):
    return sum(coeffs)


def main():
    failures = 0

    print("== 1. Rolfsen table ==")
    for name, expected in ROLFSEN.items():
        L = spherogram.Link(name)
        got_s = alexander_seifert(L)
        got_w = alexander_wirtinger(L)
        ok = got_s == expected == got_w
        if not ok:
            failures += 1
        print(f"  {name:6s} expected {expected}  seifert {got_s}  wirtinger {got_w}  {'OK' if ok else 'MISMATCH'}")

    print("== 2/3/4. random alternating knots: cross-method + HFK + sanity ==")
    rng = random.Random(20260724)
    for trial in range(40):
        target = rng.choice([8, 10, 12, 14, 16])
        L = spherogram.random_link(target, num_components=1, alternating=True)
        n = len(L.crossings)
        if n < 3:
            continue
        a_s = alexander_seifert(L)
        a_w = alexander_wirtinger(L)
        a_h = hfk_alexander(L) if n <= 14 else None
        info = analyze(a_s)
        ok = (a_s == a_w
              and (a_h is None or a_h == a_s)
              and abs(delta_at_one(a_s)) == 1
              and info["palindromic"]
              and info["signs_alternate"])
        if not ok:
            failures += 1
            print(f"  FAIL n={n}: seifert={a_s} wirtinger={a_w} hfk={a_h} "
                  f"D(1)={delta_at_one(a_s)} pal={info['palindromic']} alt={info['signs_alternate']}")
        else:
            print(f"  ok n={n:3d} deg={info['degree']:3d} det={info['det']:8d} "
                  f"hfk={'✓' if a_h is not None else '-'} trap={info['trapezoidal']}")

    print(f"\n{'ALL CHECKS PASSED' if failures == 0 else f'{failures} FAILURES'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
