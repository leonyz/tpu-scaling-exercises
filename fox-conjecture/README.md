# Hunting a counterexample to Fox's trapezoidal conjecture

Fox (1962): for an alternating knot, the absolute values of the
coefficients of the Alexander polynomial form a **trapezoidal** sequence —
strictly increasing, then a single constant plateau, then strictly
decreasing:

    a_1 < a_2 < ... < a_k = ... = a_m > a_{m+1} > ... > a_n

Still open. Proven for: two-bridge knots (Hartley), alternating algebraic
knots (Murasugi), genus ≤ 2 (Ozsváth–Szabó, Jong, Hirasawa–Murasugi),
special alternating links (Hafner–Mészáros–Vidinas 2023, via Lorentzian
polynomials — they get log-concavity), and certain Murasugi sums
(Azarpendar–Juhász–Kálmán 2024; dimer proof by
Mészáros–Sherman-Bennett–Vidinas 2025). Stoimenow's **strong Fox
conjecture** upgrades trapezoidal to log-concave with no internal zeros.

## Structure of the hunt

**Key reduction.** For a symmetric positive sequence, log-concavity
implies the strict trapezoid shape, and conversely *any* Fox violation
(a flat step below the peak, or a dip) forces a log-concavity violation
at that position: if `b[k] = b[k+1] < b[k+2]` then
`b[k+1]^2 < b[k]*b[k+2]`. So every possible counterexample to Fox lives
inside a counterexample to strong Fox: **the single quantity to chase is
the off-plateau ratio `b[k-1]*b[k+1] / b[k]^2` exceeding 1.**

**Where the wall is thinnest.** The ascent steps of the staircase are
bounded on average by det/degree (determinant = sum of |coefficients| =
spanning-tree count of the Tait graph). Random alternating diagrams have
exponentially many spanning trees — huge steps, nowhere near violating.
The tight region is **high genus + low determinant**: long staircases
climbing in steps of 1. A counterexample must also dodge every proven
class (non-algebraic, non-special, genus ≥ 3).

## Pipeline

- `alexander.py` — exact Alexander polynomials, two independent routes:
  Seifert matrix (`det(V − tVᵀ)`) and Wirtinger presentation + Fox
  calculus from the PD code. Exact integer determinants (FLINT) at
  integer points + Newton interpolation. Coefficient-shape checks:
  sign alternation, palindromicity, unimodality, strict trapezoid,
  log-concavity, and the ascent margin.
- `validate.py` — Rolfsen table, cross-method agreement, knot Floer
  homology Euler characteristics (third independent implementation),
  Δ(1) = ±1, palindromicity. All pass.
- `search.py` — mass random sampling of prime alternating knots
  (spherogram's random planar-map model), monitoring all three
  violation levels + closest-call statistics.
- `taitgraph.py` — embedded checkerboard graphs: extract from any
  alternating diagram, rebuild the diagram as the medial (round-trip
  verified to preserve Δ; Matrix–Tree count = knot determinant on every
  sample). Series/parallel edge moves = twist-region surgery.
- `climb.py` — evolutionary hill-climb over Tait graphs, minimizing the
  ascent margin with the LC ratio as tiebreaker, det-capped to steer
  into the high-genus/low-det region. Margin 0 kills Fox; ratio > 1
  kills strong Fox and arms the product attack.
- `product_attack.py` — connected sums multiply Alexander polynomials;
  |coefficients| convolve. By Wintner's theorem symmetric unimodality
  survives convolution, so composites can only break the *strict* form —
  and only if some prime factor already violates log-concavity.

## Verdict so far

See `RESULTS.md` for the numbers from the runs in this session.
