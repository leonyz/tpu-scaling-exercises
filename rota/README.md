# Hunting a counterexample to Rota's unimodality conjecture

An attempt to do to Rota's conjecture what has lately been done to a number of
open conjectures: refute it by targeted computational search. Spoiler: no
counterexample here — but the exercise produces a sharp map of *where* one
would have to live, working machinery for searching that region, and an
empirical characterization of the obstruction that makes the region hard to
populate.

## The conjecture

For a matroid $M$ of rank $r$, the **Whitney numbers of the second kind**
$W_0, W_1, \dots, W_r$ count the flats of each rank ($W_0 = W_r = 1$;
for a simple matroid $W_1 = n$ points, $W_2$ = lines, $W_3$ = planes, ...,
$W_{r-1}$ = hyperplanes).

> **Rota (1971).** $(W_0, \dots, W_r)$ is unimodal: it rises to a peak and
> then falls. (Stronger folklore variant: it is log-concave.)

Three neighboring statements are **theorems**, which matters a lot below:

- **Whitney numbers of the first kind** (characteristic-polynomial
  coefficients) are log-concave — Adiprasito–Huh–Katz (2015). That
  celebrated result did *not* touch the second kind.
- **Top-heaviness** (Dowling–Wilson conjecture): $W_k \le W_{r-k}$ for
  $k \le r/2$ — Braden–Huh–Matherne–Proudfoot–Wang (2020).
- **de Bruijn–Erdős / Motzkin / Basterfield–Kelly**: $W_1 \le W_2$ (rank
  $\ge 3$) and $W_1 \le W_{r-1}$.

The second-kind unimodality conjecture remains open. Matroids are fully
enumerated only through 9 elements, and the conjecture holds there, so any
counterexample needs $n \ge 10$ — beyond exhaustive search, which is exactly
why a *targeted* search is the only sensible move.

## Step 1: carve away the provably safe region

A failure of unimodality means an **interior dip**: some $k$ with
$W_{k-1} > W_k < W_{k+1}$. Since the lattice of flats of $M$ equals that of
its simplification, assume $M$ simple. Two observations shrink the target
dramatically:

**Paving matroids are safe.** If $M$ is paving of rank $r$, every set of size
$\le r-1$ is independent, so $W_k = \binom{n}{k}$ for $k \le r-2$, and the
hyperplanes form a $d$-partition, so $W_{r-1} \le \binom{n}{r-1}$. A dip at
$k \le r-2$ would need the binomial sequence to fall at $k$ and then rise
into $W_{r-1}$, impossible since falling means $\binom{n}{k} <
\binom{n}{k-1}$ forces $\binom{n}{k+1} < \binom{n}{k}$ too and
$W_{r-1} \le \binom{n}{r-1}$; a dip at $k = r-1$ would need $W_r = 1 >
W_{r-1} \ge 1$. So no dip exists. Since conjecturally almost every matroid is
(sparse) paving, **almost all of matroid space is safe and random sampling is
hopeless**. Campaigns must be structural.

**The dip can only sit just below the top.** Truncating $M$ to rank $2k+1$
preserves all flats of rank $\le 2k$, and top-heaviness applied to the
truncation gives $W_k \le W_{k+1}$ whenever $2k+1 \le r$: the *proved*
top-heavy theorem forces the bottom half of the sequence to be monotone
increasing. Hence a dip at $k$ (which needs $W_{k-1} > W_k$) requires

$$2k > r + 1, \qquad 2 \le k \le r-2 .$$

These force $r \ge 6$, and for $r = 6$ the only admissible position is
$k = 4$. So:

> **The minimal possible counterexample is a rank-6 simple matroid with
> $W_3 > W_4 < W_5$** — many planes, a *thin layer of rank-4 flats*, many
> hyperplanes — on at least 10 elements. In general rank, dips are confined
> to $(r+1)/2 < k \le r-2$.

This shape is an *hourglass pinched at rank 4*, the opposite curvature from
modular geometries like $PG(5,q)$ whose top half decreases monotonically.

## Step 2: a tool that manufactures exactly that shape — free erections

A thin rank-4 layer under a *fat* rank-5 layer is precisely what a
**matroid erection** produces. If $N$ (rank $r+1$) is an erection of $M$
(rank $r$), i.e. $T(N) = M$, then

$$W(N) = \big(W_0(M), \dots, W_{r-1}(M),\; H,\; 1\big)$$

where $H$ is the number of hyperplanes of $N$. Among all erections, Crapo's
**free erection** has the *finest* hyperplane partition, hence the maximal
$H$ — it is the optimal one-step move toward a dip. Concretely: erect a
rank-5 matroid $M$ with $W_4(M) < W_3(M)$; a counterexample appears iff the
free erection exists and has $H > W_4(M)$.

`matroids.py` implements the whole pipeline over bitmask subsets:

- flats / Whitney numbers from any rank oracle (`whitney2`, `flats_by_rank`);
- oracles: uniform, graphic, GF(2)/GF(p) linear, (sparse) paving from blocks,
  truncation, relaxation, direct sums (Whitney vectors convolve);
- **free erection** (Knuth-style): seed each basis, close under
  "a spanning subset of an $M$-hyperplane pulls in that hyperplane", merge
  seeds whose closures share a basis, iterate to fixpoint; `None` means not
  erectable;
- `verify_matroid_axioms` — brute-force rank-axiom certification, so that any
  future "hit" can be independently checked before anyone gets excited;
- `dip_score(W)` = $\max_k \min(W_{k-1}, W_{k+1})/W_k$ — a continuous
  progress measure (> 1 ⟺ counterexample).

`test_matroids.py` pins the engine to known ground truth: $U_{r,n}$, Fano,
$PG(3,2)$, $M(K_5)$ (partition lattice), Vámos, $AG(3,2)$, and the classical
facts that the free erection of $U_{3,5}$ is $U_{4,5}$ and that Fano is not
erectable.

## Step 3: campaigns (`search.py`) and what happened

| # | Campaign | Result |
|---|----------|--------|
| 1 | Sparse paving, greedy max circuit-hyperplane packings, $n \le 20$, $r \le 8$ | all unimodal (as proved above); best dip 0.667 |
| 2 | Random GF(2)-represented matroids + all truncations, $n \le 12$ | all unimodal; best dip 0.729 |
| 3 | Free erections of rank-$\le4$ design pavings: $SQS(8)=AG(3,2)$, $S(3,4,10)$ (inversive plane of order 3, built from $PGL_2(9)$ — note $n=10$ is beyond the enumerated range), $S(3,5,17)$ (inversive plane of order 4), $PG(3,2)$, cyclic $STS(13)$, $STS(15)$ | **all non-erectable** — as they must be: a rank-5 erection with $W_2 > W_3$ would contradict the proven top-heavy theorem. The machinery confirming the obstruction empirically is a strong validity check. Bonus: the free erection of the $STS(15)$-paving reconstructs $PG(3,2)$ exactly, $(1,15,35,15,1)$ |
| 4 | Direct sums + truncations over a library of extreme Whitney vectors (projective planes, $PG(d,q)$, Steiner pavings, Vámos, near-pencils) — 209 161 sequences | all unimodal; best dip 0.773 ($PG(5,2)^{\oplus 3}$) |
| 5 | **The frontier**: rank-5 matroids with thin $W_4$, erected toward rank 6 — Witt design $S(4,5,11)$ paving $(1,11,55,165,66,1)$ built from the ternary Golay code, its relaxations (1–33 blocks dropped), $S(5,6,12)$ paving $(1,12,66,220,495,132,1)$ + relaxations, $AG(4,2)$ $(1,16,120,140,30,1)$ | **all non-erectable**, even with half the blocks relaxed |

Reproduce with `python3 test_matroids.py && python3 search.py`
(~2.5 min, pure Python, no dependencies).

## What the negative results actually teach

Campaign 5 is the interesting one. The dip shape demands a rank-4 layer that
is **thin** ($W_4 < W_3$ forces every 4-set to lie in a large rank-4 flat —
design-like tightness) yet **erectable** (the closure cascade "a spanning
subset of a hyperplane absorbs the hyperplane" must halt before swallowing
the ground set — which demands looseness). In every candidate, tightness won:
each 5-set's five 4-subsets sit in big blocks whose absorption creates new
4-subsets faster than the cascade can terminate. Relaxing blocks (which
provably preserves matroidness) did not open enough slack even at 50%
density. This tension — *thin enough to dip, loose enough to lift* — is, in
compressed empirical form, why the conjecture is hard.

## Feasibility of an exhaustive n = 10, rank-6 sweep (`exp_rank6_n10.py`)

Estimates in the billions for 10-element matroid counts are consistent with
the Bansal–Pendavingh–van der Pol asymptotics ($\log_2\log_2 m_n \le n -
\tfrac32\log_2 n + O(\log\log n)$ gives $\sim 2^{32} \approx 4\times10^9$ at
$n=10$), but no enumeration exists — the frontier is $n = 9$ (Mayhew–Royle,
~383M matroids). The census would have to be *generated*, and that, not the
unimodality check, is the entire cost:

- **Checking is trivial.** A matroid on 10 elements is a 1024-entry rank
  table; counting flats by rank is $\sim n \cdot 2^n$ table lookups.
  Measured: 7 ms/matroid (flats engine), 0.7 ms (table walk, pure Python),
  ~2–5 µs compiled — i.e. **~4 core-hours for 4.9B matroids** in C.
- **Generation is not.** Isomorph-free exhaustive generation (canonical
  augmentation from the 9-element catalogue) costs orders of magnitude more
  per matroid and all of the engineering.

**The prune that removes rank-6 generation entirely.** Any 10-element
counterexample is simple (its simplification lives on $\le 9$ elements, all
verified) and rank 6 with its dip at $k=4$. Its rank-5 truncation $T(M)$
shares $W_0..W_4$, and $M$ is an erection of $T(M)$, so $W_5(M) \le
H_{\mathrm{free}}(T(M))$. Hence a rank-6 counterexample on 10 elements
exists **iff** some simple rank-5 matroid on 10 elements has $W_4 < W_3$
and a free erection with $H_{\mathrm{free}} > W_4$. One free-erection
computation settles *all* rank-6 matroids sharing that truncation, and the
thinness filter $W_4 < W_3 \le 120$ applies before erecting.

Measured on the sparse-paving cell of that truncation space (rank-5 sparse
paving needs $Z \ge 23$ circuit-hyperplanes for thinness; $Z \le
A(10,4,5) = 36$, the Witt-residual optimum):

- 386 sampled families with $Z \in [23, 36]$ (Witt-code subfamilies and
  independent local-search packings): **all non-erectable** — 35 ms each.
- Sweeping $Z = 0..22$: erectability itself dies by $Z \approx 8$, and
  $H_{\mathrm{free}}$ exceeds $W_4$ only for $Z \le 2$. Each added
  circuit-hyperplane costs the free erection ~20 hyperplanes while the
  dip threshold falls by only 4 — the two requirements diverge with
  slope $\sim -21$ vs $-4$ per block.

So within this cell the "thin enough to dip" ($Z \ge 23$) and "liftable with
a rise" ($Z \le 2$) regimes are separated by a factor of ten. Not yet a
theorem — the samples should become an exhaustive sweep over stable-set
isomorphism classes of $J(10,5)$ (a 252-vertex graph; standard clique/orbit
tooling), and the paving-with-larger-blocks and non-paving truncation cells
remain open. Those two cells are the actual frontier of an exhaustive
$n = 10$ verdict.

## Where to push next

1. **Skip erections; encode the target directly.** A rank-6 counterexample is
   a solution to a constraint system: choose the rank-4 flat family and
   hyperplane family on $n \approx 12$–$20$ points subject to the flat-lattice
   axioms plus $W_3 > W_4 < W_5$. This is SAT/ILP-shaped, and modern solvers
   (or LLM-guided search à la FunSearch over block-family generators, with
   `dip_score` as fitness and `verify_matroid_axioms` as the referee) are the
   natural next tools. The near-miss leaderboard gives the warm starts.
2. **Truncated big geometries.** $T(PG(5,2))$ to rank 5 has
   $W_4 = 651 \ll W_3 = 1395$ and *is* erectable by construction (PG itself
   is an erection); the question is whether its free erection beats
   $H = 651$ rather than just recovering $W_5 = 63$. At $n = 63$ this needs
   the bitset engine ported to a fast language — the clearest concrete open
   computation this repo sets up.
3. **Higher rank.** Dips are allowed at all $(r+1)/2 < k \le r-2$; rank 7–8
   candidates (e.g. relaxations of Dowling lattices, $q$-analog designs)
   widen the target zone at modest $n$.

A sober note on the framing: the recent wave of machine-found counterexamples
has mostly landed on conjectures with low-structure search spaces. Here,
proven theorems (top-heaviness above all) actively fence off most of the
space — which cuts both ways: random search is provably useless, but the
fence tells us exactly where to aim, and the region it leaves open is small,
concrete, and computationally attackable. That is how you begin.
