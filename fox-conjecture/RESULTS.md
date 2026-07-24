# Results: no counterexample found — and a sharp picture of why

Session date: 2026-07-24. Hardware: 4 cores, ~2.5 hours of compute.

## What was searched

| Attack | Scale | Violations |
|---|---|---|
| Random prime alternating knots (3 samplers: broad, large-only, twist-structured) | 352,439 knots, diagrams up to 96 crossings, Alexander degree up to 68 (genus 34), determinants up to 3.3·10²⁰ | 0 |
| Directed evolutionary climb over embedded Tait graphs (twist surgery, det-capped, margin + LC-surplus objective) | 2,032,984 evaluations across 3 runs | 0 |
| Connected-sum product attack (convolutions of sampled polynomials) | 8,002,000 pairwise + 36,000 triple products over 315,919 unique sequences | 0 |

Every knot was checked for: unimodality of |coefficients| (weak Fox),
strict trapezoidal shape (Fox as stated), and log-concavity (Stoimenow's
strong Fox). **No violation of any of the three was ever observed.**
312,701 distinct coefficient sequences were examined.

## The sharp wall

The search did not just fail — it located the exact boundary of failure.

1. **Every Fox counterexample is an LC counterexample.** For symmetric
   positive sequences, a flat step below the peak (or a dip) forces
   `b[k-1]·b[k+1] > b[k]²` at that spot. The hunt reduces to making the
   integer surplus `b[k-1]·b[k+1] − b[k]²` reach 0 off the plateau.

2. **The climber reached surplus −1 and stayed there for 2M evaluations.**
   Directed search into the high-genus/low-determinant region converges
   onto *arithmetic-progression staircases* — coefficient sequences like

       1, 3, 4, 5, 5, 5, 4, 3, 1        (10 crossings, det 31, genus 4)
       40, 81, 82, 83, 83, ..., 83, 82, 81, 40   (92 crossings, det 987)

   whose consecutive-step triples satisfy `b[k-1]·b[k+1] = b[k]² − 1`
   **exactly** — log-concavity holding with the minimum possible slack.
   These families are reachable by twist surgery in profusion, extend to
   arbitrary genus, and never cross. The extremal specimen above was
   verified by three independent computations (Fox calculus, Seifert
   matrix, knot Floer homology Euler characteristics); its PD code is in
   `climb_surplus/final_pop.jsonl`.

3. **Random alternating knots are nowhere near the wall.** Large random
   diagrams have exponentially many spanning trees, so their staircases
   climb steeply: the large-only sampler (up to 96 crossings) never saw
   an ascent margin below 14 or an off-plateau LC ratio above 0.9708.
   Over the full random corpus the max off-plateau LC ratio was 0.9966,
   attained — like every close call — by low-determinant specimens.

4. **Composites cannot save the day.** By Wintner's theorem, symmetric
   unimodality survives convolution, so connected sums can never break
   weak Fox; and since every sampled prime factor is log-concave, their
   products are too, closing the strict form as well. The empirical
   product sweep agreed: zero hits.

## Honest verdict

The conjecture did not fall, and the data explains why it shouldn't:
the coefficient staircase of an alternating knot behaves like a
log-concave sequence with integer slack ≥ 1 at every step, and the
extremal families sit at slack exactly 1 without crossing. This is
precisely the shape of a theorem-in-waiting of Lorentzian/dimer type —
the Hafner–Mészáros–Vidinas program proved exactly this (log-concavity)
for the special alternating case. A counterexample would have to appear
in a non-algebraic, non-special, genus ≥ 3, low-determinant corner while
evading a structure that held with equality-tight sharpness across
2.4 million examined knots. We searched that corner directionally, at
scale, with exact arithmetic and independently cross-validated
invariants, and the wall never moved.

Verification hierarchy (all exact integer arithmetic):
Fox calculus ↔ Seifert matrix ↔ knot Floer homology on the Rolfsen
table and random samples; Δ(1) = ±1, palindromicity, sign alternation
on all 352k knots; Matrix–Tree spanning-tree count = |Δ(−1)| for every
Tait-graph-constructed diagram; diagram → Tait graph → diagram
round-trips preserve Δ.
