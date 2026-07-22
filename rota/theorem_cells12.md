# Theorem: no rank-6 counterexample on 10 elements has a paving truncation

Setting: a counterexample to Rota's conjecture on 10 elements must be a
simple rank-6 matroid $N$ with $W_3 > W_4 < W_5$ (README). Its rank-5
truncation $T$ shares $W_0..W_4$ and $N$ is an erection of $T$. This note
proves:

> **Theorem.** No rank-5 paving matroid on 10 elements with $W_4 < 120$
> admits an erection to rank 6. Hence every 10-element counterexample (if
> any) has a **non-paving** truncation.

Throughout, $T$ is rank-5 paving on $E$, $|E| = 10$: every 4-set is
independent, the hyperplanes ("blocks") form a d-partition — each 4-set
lies in exactly one block, blocks pairwise meet in $\le 3$ points. Blocks
of size 4 are "free" 4-sets; write $s_1, s_2, \dots \ge 5$ for the sizes of
the big blocks. Since $W_3 = \binom{10}{3} = 120$ and
$W_4 = 210 - \sum_i(\binom{s_i}{4} - 1)$, thinness ($W_4 < W_3$) is

$$\mathrm{sav} := \sum_i \left(\tbinom{s_i}{4} - 1\right) \ \ge\ 91. \tag{thin}$$

Suppose $N$ is an erection of $T$ (any erection, not just the free one).
Its hyperplanes ("classes") partition the bases of $T$: every basis lies in
exactly one class $K$, and $K$ is a flat of $N$, hence:

* **(closure)** if $A \subseteq K$ with $|A| = 4$ then $\mathrm{cl}_T(A)
  \subseteq K$; since a 4-set's closure is its block, this reads: every
  block meets $K$ in $\le 3$ points or lies inside $K$.
* **(absorption)** consequently the class of a basis $B$ contains the
  cascade closure of $B$: repeatedly add any block meeting the current set
  in $\ge 4$ points.

**Fact 0 (class sizes).** A class is a proper flat: $|K| \le 9$. A rank-4
flat of $N$ has $\le 8$ points (else adding the $\le 1$ remaining point
cannot raise rank from 4 to 6); likewise blocks of $T$ that are to survive
as rank-4 flats of $N$ have $\le 8$ points.

**Fact 1 (seed classes).** For every block $H$ with $|H| \ge 5$ and every
$b \notin H$: pick 4 independent points $A \subset H$; $A \cup b$ extends to
a basis $B \subseteq H \cup b$, and $\mathrm{cl}_T(A) = H$ forces the class
of $B$ to contain $H \cup \{b\}$. So **some class contains $H \cup \{b\}$**,
for every such pair $(H, b)$.

## Step 1: classes have at most 7 points

Let $K$ be a class, $m = |K|$.

*$m = 9$:* complement $\{x\}$. A block $H \not\subseteq K$ has
$|H \cap K| \le 3$, so $|H| \le 4$: **every big block lies in $K$**.
Enumerate big-block families inside 9 points (pairwise $\cap \le 3$,
disjoint 4-set coverage):

- an 8-block: no other big block fits (a 5-block would need $\ge 2$ points
  outside it, only 1 exists); sav $= 69$.
- a 7-block $B_7$: no 6-block ($\cap \ge 4$); 5-blocks contain both points
  of $K \setminus B_7$ plus a triple of $B_7$, triples pairwise $\le 1$:
  $\le 7$; sav $\le 34 + 28 = 62$.
- 6-blocks pairwise meet in exactly 3 inside 9 points, so their 3-point
  complements in $K$ are disjoint: at most 3 of them.
  - three 6-blocks: complements $T_1T_2T_3$ partition $K$; a 5-block would
    need $|S \cap T_i| \ge 2$ for all $i$ (to meet each 6-block
    $K \setminus T_i$ in $\le 3$), impossible for $|S| = 5$; sav $= 42$.
  - two: 5-blocks need $\ge 2$ in each of $T_1, T_2$, i.e. a pair in each;
    distinct blocks can't repeat a (pair, pair) combo ($\cap \ge 4$):
    $\le 9$; sav $\le 28 + 36 = 64$.
  - one 6-block $H_6$: 5-blocks meet $H_6$ in $\le 3$, forcing $\ge 2$ of
    the 3 outside points; per outside pair: triples of $H_6$ pairwise
    $\le 1$: $\le 4$ each ($3 \times 4$); outside-triple blocks: disjoint
    pairs of $H_6$: $\le 3$; total $\le 15$; sav $\le 14 + 60 = 74$.
  - none: pure 5-blocks: $Z \le A(9,4,5) \le A(8,4,4) + A(8,4,5) \le
    14 + 8 = 22$ (split by a point; both bounds are pure counting, below);
    sav $\le 88$.

All cases give sav $\le 88 < 91$: **no 9-point class in the thin regime**.

*$m = 8$:* complement $\{u, v\}$. Blocks $\not\subseteq K$ of size 5
contain both $u, v$ ($\ge 5 - 3$), of size $\ge 6$ are impossible
($\ge 3$ points in a 2-set): big blocks are inside $K$ or are 5-blocks
through $\{u,v\}$. Through-$uv$ 5-blocks have triples in $K$ pairwise
$\le 1$: $\le 8$. Inside $K$ (8 points): a 7-block excludes all other big
blocks inside (5-block would need $\ge 2$ of the single outside point);
sav $\le 34 + 32 = 66$. A 6-block: inside 5-blocks = (2 outside pts of
$K \setminus H_6$) + triple of $H_6$ pairwise $\le 1$: $\le 4$; sav
$\le 14 + 16 + 32 = 62$. Neither: inside 5-blocks $\le A(8,4,5) = 8$;
sav $\le 32 + 32 = 64$. All $< 91$: **no 8-point class**.

So every class has $\le 7$ points.

## Step 2: big blocks are nearly impossible

- $|H| \ge 7$: Fact 1 gives a class $\supseteq H \cup b$ with $\ge 8$
  points. Contradiction. **No blocks of size $\ge 7$.**
- Two 6-blocks with $|H_6 \cap H_6'| = 3$: Fact 1's class at
  $b \in H_6' \setminus H_6$ absorbs $H_6'$ ($\ge 4$ points shared with
  $H_6 \cup b$): class $\supseteq H_6 \cup H_6'$, 9 points. Contradiction.
  In 10 points $|H_6 \cap H_6'| \ge 2$, so **6-blocks pairwise meet in
  exactly 2**; complements are disjoint 4-sets: **at most two 6-blocks**.
- 5-block $S$ with $|S \cap H_6| = 3$: $S$ has 2 points outside $H_6$;
  taking $b \in S \setminus H_6$, the class of $(H_6, b)$ absorbs $S$:
  $\ge 8$ points. Contradiction. **5-blocks meet 6-blocks in $\le 2$.**

## Step 3: case analysis on the number $a$ of 6-blocks

Let $c$ = number of 5-blocks; sav $= 14a + 4c \ge 91$.

*$a = 2$:* $|H_6 \cup H_6'| = 10$, so a 5-block satisfies
$5 = |S \cap H_6| + |S \cap H_6'| - |S \cap (H_6 \cap H_6')| \le 2 + 2$.
Contradiction: $c = 0$, sav $= 28 < 91$. Dead.

*$a = 1$:* need $c \ge \lceil 77/4 \rceil = 20$. Each 5-block has $\le 2$
points in $H_6$, hence $\ge 3$ in the 4 points outside. Outside-part a
4-set: at most one such block. Outside-part a triple (4 choices): the
$H_6$-parts are pairs, pairwise disjoint ($\cap \le 3$): $\le 3$ each.
Total $c \le 1 + 12 = 13 < 20$. Dead.

*$a = 0$:* pure sparse paving, $c = Z \ge \lceil 91/4 \rceil = 23$.

**Lemma C.** For each CH $C$ (5-block), at most 4 CHs meet $C$ in 3
points. *Proof:* such a $C'$ has exactly 2 points $\{b_i, b_j\}$ among the
5 outside $C$; the class of $(C, b_i)$ absorbs $C'$, giving a closed
$K = C \cup \{b_i, b_j\}$ of size 7. If $b_i$ paired with two different
partners, the class would have 8 points — so the pairs form a matching on
5 points: $\le 2$ pairs. Each 7-set holds $\le A(7,4,5) = 3$ CHs
(complements are disjoint pairs in 7 points), so $\le 2$ CHs per pair
besides $C$: $\le 4$ total. ∎

**Lemma D (counting kill).** For a 3-set $\tau$, let $t_\tau$ = #CHs
containing $\tau$; CHs through $\tau$ have disjoint 2-point remainders, so
$t_\tau \le 3$. $\sum_\tau t_\tau = 10Z$ over $\binom{10}{3} = 120$
triples. Two CHs share a triple iff they meet in exactly 3 points, and
then share exactly one triple, so the number of 3-intersecting CH pairs is
$\sum_\tau \binom{t_\tau}{2} \ge 10Z - 120$ (minimized by spreading
$t_\tau \in \{1,2\}$). Lemma C caps this at $4Z/2 = 2Z$. But
$10Z - 120 > 2Z \iff Z > 15$, and $Z \ge 23$. Contradiction. ∎

All cases dead: the theorem follows. $\blacksquare$

## Counting facts used (all machine-verified in `verify_cells12.py`)

- $A(7,4,5) = 3$: complements of 5-sets in 7 points are 2-sets; pairwise
  $\cap \le 3$ ⟺ complements disjoint; max matching in $K_7$ is 3.
- $A(8,4,5) = 8$: complements are 3-sets pairwise $\le 1$; each point of 8
  lies in $\le \lfloor 7/2 \rfloor = 3$ such triples; $3T \le 24$.
- $A(8,4,4) \le 14$: 4-sets pairwise $\le 2$ share no triple;
  $4T \le \binom{8}{3} = 56$.
- triples pairwise $\le 1$: $\le 7$ in 7 points, $\le 8$ in 8 points,
  $\le 4$ in 6 points (same degree count).
- $A(9,4,5) \le 22$ by the point split above (the true value is 18, the
  Witt residual; only $\le 22$ is needed).

Every case bound in Steps 1–3 is additionally confirmed by exhaustive
branch-and-bound over the corresponding tiny configuration spaces, and the
theorem is cross-validated against the cascade code on randomized paving
families (savings $\ge 91$ ⟹ non-erectable, no exceptions found).
