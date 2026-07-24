"""Exact Alexander polynomials of knots + Fox trapezoidal-conjecture checks.

Two independent computations of the Alexander polynomial (up to units):

  1. Seifert matrix V (from spherogram):  Delta(t) = det(V - t*V^T)
  2. Wirtinger presentation + Fox calculus from the PD code:
     Delta(t) = det of the (n-1)x(n-1) minor of the Alexander matrix

Both use exact integer determinants (FLINT) at integer evaluation points,
recovered by Newton interpolation over the rationals.  Agreement of the two
methods, plus Delta(1) = +-1, palindromicity, and (for small knots) the
Euler characteristic of knot Floer homology, validates the pipeline.

Fox's trapezoidal conjecture (1962): for an alternating knot the absolute
values of the coefficients of Delta strictly increase, stabilize, then
strictly decrease.  We check three nested conditions on |coefficients|:

  * unimodal        -- never strictly rises after a strict fall
                       (a violation disproves even the weak form of Fox)
  * trapezoidal     -- strictly increasing, one central plateau, strictly
                       decreasing (a violation disproves the strict form)
  * log-concave     -- b_i^2 >= b_{i-1} b_{i+1} (Stoimenow's stronger
                       conjecture; a violation maps the safety margin)
"""

from fractions import Fraction

from flint import fmpz_mat


# ---------------------------------------------------------------------------
# exact determinant of a matrix of linear polynomials, by eval + interpolation
# ---------------------------------------------------------------------------

def _det_poly(rows_const, rows_lin, size):
    """det(C + t*L) as list of int coefficients (low degree first).

    rows_const, rows_lin: flat lists (row-major) of ints, length size*size.
    """
    if size == 0:
        return [1]
    npts = size + 1
    xs = list(range(2, 2 + npts))
    ys = []
    for x in xs:
        entries = [c + x * l for c, l in zip(rows_const, rows_lin)]
        ys.append(int(fmpz_mat(size, size, entries).det()))
    # Newton divided differences over Fraction
    coeffs = [Fraction(y) for y in ys]
    for j in range(1, npts):
        for i in range(npts - 1, j - 1, -1):
            coeffs[i] = (coeffs[i] - coeffs[i - 1]) / (xs[i] - xs[i - j])
    # expand Newton form to monomial basis
    poly = [Fraction(0)] * npts
    poly[0] = coeffs[npts - 1]
    deg = 0
    for k in range(npts - 2, -1, -1):
        # poly <- poly * (t - xs[k]) + coeffs[k]
        for i in range(deg + 1, 0, -1):
            poly[i] = poly[i - 1] - xs[k] * poly[i]
        poly[0] = -xs[k] * poly[0] + coeffs[k]
        deg += 1
    out = []
    for c in poly:
        assert c.denominator == 1, "interpolation produced non-integer"
        out.append(int(c))
    return out


def normalize(coeffs):
    """Strip trailing/leading zeros, make the lowest-degree coefficient > 0."""
    c = list(coeffs)
    while c and c[-1] == 0:
        c.pop()
    lead = 0
    while lead < len(c) and c[lead] == 0:
        lead += 1
    c = c[lead:]
    if not c:
        return c
    if c[0] < 0:
        c = [-x for x in c]
    return c


# ---------------------------------------------------------------------------
# method 1: Seifert matrix
# ---------------------------------------------------------------------------

def alexander_seifert(link):
    """Alexander coefficients (normalized) from spherogram's Seifert matrix."""
    V = link.seifert_matrix()
    m = len(V)
    const, lin = [], []
    for i in range(m):
        for j in range(m):
            const.append(int(V[i][j]))
            lin.append(-int(V[j][i]))
    return normalize(_det_poly(const, lin, m))


# ---------------------------------------------------------------------------
# method 2: Wirtinger presentation + Fox calculus, from the PD code
# ---------------------------------------------------------------------------

def _arcs_and_signs(pd):
    """From a knot PD code, return (arc ids per edge label, crossing data).

    PD tuples are (a, b, c, d): a = incoming understrand, then
    counterclockwise.  The overstrand occupies b and d; its direction
    (d->b vs b->d) determines the sign (KnotTheory convention:
    over d->b is positive).
    """
    n = len(pd)
    labels = sorted({x for t in pd for x in t})
    assert len(labels) == 2 * n
    base = labels[0]
    N = 2 * n

    def nxt(x):
        return (x - base + 1) % N + base

    parent = {x: x for x in labels}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        parent[find(x)] = find(y)

    for (a, b, c, d) in pd:
        union(b, d)  # overstrand passes through

    crossings = []
    for (a, b, c, d) in pd:
        if nxt(d) == b:
            sign = +1  # over runs d -> b
        elif nxt(b) == d:
            sign = -1  # over runs b -> d
        else:
            raise ValueError("overstrand edges not consecutive: knot PD expected")
        crossings.append((sign, find(b), find(a), find(c)))  # (sign, over, in, out)
    arc_ids = {x: i for i, x in enumerate(sorted({find(x) for x in labels}))}
    return arc_ids, crossings


def alexander_wirtinger(link_or_pd):
    """Alexander coefficients (normalized) via Fox calculus on the PD code."""
    pd = link_or_pd if isinstance(link_or_pd, list) else link_or_pd.PD_code()
    pd = [tuple(t) for t in pd]
    n = len(pd)
    if n == 0:
        return [1]
    arc_ids, crossings = _arcs_and_signs(pd)
    assert len(arc_ids) == n, "expected n arcs for an n-crossing knot diagram"
    # entries are c + l*t; build full n x n then delete last row and column
    size = n - 1
    const = [0] * (size * size)
    lin = [0] * (size * size)

    def add(row, col, c, l):
        if row >= size or col >= size:
            return
        const[row * size + col] += c
        lin[row * size + col] += l

    for row, (sign, over, a_in, a_out) in enumerate(crossings):
        o, i, j = arc_ids[over], arc_ids[a_in], arc_ids[a_out]
        if sign > 0:
            # relation x_out = x_over x_in x_over^-1 : row (1-t, t, -1)
            add(row, o, 1, -1)
            add(row, i, 0, 1)
            add(row, j, -1, 0)
        else:
            # relation x_out = x_over^-1 x_in x_over : row (t-1, 1, -t)
            add(row, o, -1, 1)
            add(row, i, 1, 0)
            add(row, j, 0, -1)
    return normalize(_det_poly(const, lin, size))


# ---------------------------------------------------------------------------
# coefficient-shape checks
# ---------------------------------------------------------------------------

def signs_alternate(coeffs):
    return all(coeffs[k] * coeffs[k + 1] < 0 for k in range(len(coeffs) - 1))


def is_palindromic(coeffs):
    return coeffs == coeffs[::-1]


def is_unimodal(b):
    """No strict rise after a strict fall."""
    fallen = False
    for k in range(len(b) - 1):
        if b[k + 1] < b[k]:
            fallen = True
        elif b[k + 1] > b[k] and fallen:
            return False
    return True


def is_trapezoidal(b):
    """Strictly increasing, then constant, then strictly decreasing."""
    n = len(b)
    k = 0
    while k + 1 < n and b[k + 1] > b[k]:
        k += 1
    l = k
    while l + 1 < n and b[l + 1] == b[l]:
        l += 1
    while l + 1 < n and b[l + 1] < b[l]:
        l += 1
    return l == n - 1


def is_log_concave(b):
    return all(b[k] * b[k] >= b[k - 1] * b[k + 1] for k in range(1, len(b) - 1))


def unimodality_margin(b):
    """Smallest ascent step strictly before the first maximum (the sequence
    is palindromic, so the descent mirrors the ascent).  For a unimodal
    sequence this is >= 0; it is 0 exactly when there is a flat spot off the
    central plateau (a strict-form violation), and > 0 for a true trapezoid.
    None for degenerate (length <= 2 or already-maximal-at-0) sequences."""
    first_max = b.index(max(b))
    if first_max == 0:
        return None
    return min(b[k + 1] - b[k] for k in range(first_max))


def analyze(coeffs):
    b = [abs(c) for c in coeffs]
    return {
        "coeffs": coeffs,
        "degree": len(coeffs) - 1,
        "det": sum(b),  # |Delta(-1)| when signs alternate
        "signs_alternate": signs_alternate(coeffs),
        "palindromic": is_palindromic(coeffs),
        "unimodal": is_unimodal(b),
        "trapezoidal": is_trapezoidal(b),
        "log_concave": is_log_concave(b),
        "margin": unimodality_margin(b),
    }
