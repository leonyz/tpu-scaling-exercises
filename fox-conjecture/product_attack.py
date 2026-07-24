"""Connected-sum attack on the *strict* trapezoidal form of Fox's conjecture.

A connected sum of alternating knots is an alternating knot, and Alexander
polynomials multiply.  Coefficient signs alternate in each factor, so the
absolute coefficient sequence of a product is the plain convolution of the
factors' absolute sequences.

By Wintner's theorem, the convolution of symmetric unimodal sequences is
symmetric unimodal, so connected sums can never violate unimodality (the
weak form of Fox's conjecture is closed under connected sum).  But strict
trapezoidality -- strictly increase / one plateau / strictly decrease -- is
NOT obviously closed under convolution.  This script convolves pairs and
triples of sampled Alexander polynomials looking for a product with a flat
step off the central plateau.  Any hit is realized by an explicit composite
alternating knot (connected sum of the sampled PD codes).
"""

import glob
import gzip
import itertools
import json
import sys

from alexander import is_trapezoidal, is_unimodal


def convolve(a, b):
    out = [0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i + j] += x * y
    return out


def load_unique(paths, max_deg=None):
    seen = {}
    for path in paths:
        with gzip.open(path, "rt") as f:
            for line in f:
                rec = json.loads(line)
                b = tuple(abs(c) for c in rec["coeffs"])
                if len(b) < 2:
                    continue
                if max_deg and len(b) > max_deg + 1:
                    continue
                if b not in seen:
                    seen[b] = rec
    return seen


def main():
    dbs = sys.argv[1:]
    if not dbs:
        dbs = glob.glob("/tmp/**/run*/db.jsonl.gz", recursive=True)
    uniq = load_unique(dbs)
    print(f"{len(uniq)} unique |coefficient| sequences from {len(dbs)} dbs")

    seqs = sorted(uniq, key=len)
    hits = []

    def check(prod, provenance):
        if not is_unimodal(prod):
            print(f"!!! UNIMODALITY BROKEN BY PRODUCT (should be impossible): {provenance}")
            hits.append(("unimodal", prod, provenance))
        elif not is_trapezoidal(prod):
            hits.append(("strict", prod, provenance))

    # all pairs (convolution is cheap; cap the quadratic work if huge)
    cap = 4000
    small = seqs[:cap]
    total = len(small) * (len(small) + 1) // 2
    print(f"checking {total} pairwise products of the {len(small)} smallest-degree seqs")
    done = 0
    pair_hits_seqs = []
    for i in range(len(small)):
        for j in range(i, len(small)):
            p = convolve(small[i], small[j])
            check(p, (small[i], small[j]))
            done += 1
        if i % 200 == 0:
            print(f"  row {i}, {done}/{total}, strict hits so far: "
                  f"{sum(1 for h in hits if h[0]=='strict')}", flush=True)

    # triples: seed with any strict hits' factors plus the lowest-degree seqs
    seed = seqs[:60]
    print(f"checking {len(seed)**3 // 6}-ish triple products of {len(seed)} seeds")
    for a, b, c in itertools.combinations_with_replacement(seed, 3):
        check(convolve(convolve(a, b), c), (a, b, c))

    print(f"\ntotal hits: {len(hits)}")
    with open("product_hits.jsonl", "w") as f:
        for kind, prod, prov in hits[:1000]:
            f.write(json.dumps({
                "kind": kind, "product_abs": prod,
                "factors_abs": [list(x) for x in prov],
                "factor_knots": [uniq[tuple(x)] for x in prov],
            }) + "\n")
    for kind, prod, prov in hits[:10]:
        print(f"  [{kind}] {prod}\n    factors: {[list(x) for x in prov]}")


if __name__ == "__main__":
    sys.exit(main())
