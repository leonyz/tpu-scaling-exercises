"""Embedded Tait (checkerboard) graphs of alternating link diagrams.

An embedded planar multigraph is stored as a rotation system:

  edges:  list of (u, v) vertex pairs; dart (e, 0) sits at u, (e, 1) at v
  rot:    dict vertex -> cyclic (counterclockwise) list of darts

The alternating diagram of the graph is its medial: one crossing per edge,
glued corner-to-corner around each vertex.  Going the other way, the Tait
graph of an alternating diagram is recovered from a checkerboard coloring
of its faces.

Series (subdivide an edge) and parallel (double an edge) moves correspond
to extending twist regions of the diagram, preserve alternation, and grow
the determinant slowly -- the right moves for exploring high-genus,
low-determinant alternating knots where Fox's trapezoidal conjecture is
tightest.
"""

import random

import spherogram
from flint import fmpz_mat


class PlanarGraph:
    def __init__(self, edges, rot):
        self.edges = [tuple(e) for e in edges]
        self.rot = {v: list(ds) for v, ds in rot.items()}

    def copy(self):
        return PlanarGraph(self.edges, self.rot)

    def dart_vertex(self, dart):
        e, k = dart
        return self.edges[e][k]

    def check(self):
        from collections import Counter
        darts = [(e, k) for e in range(len(self.edges)) for k in (0, 1)]
        listed = [d for ds in self.rot.values() for d in ds]
        assert Counter(darts) == Counter(listed), "rotation system malformed"
        for v, ds in self.rot.items():
            for d in ds:
                assert self.dart_vertex(d) == v

    # -- invariants ---------------------------------------------------------

    def spanning_trees(self):
        """Kirchhoff matrix-tree count (= knot determinant for the medial)."""
        vs = sorted(self.rot)
        idx = {v: i for i, v in enumerate(vs)}
        m = len(vs)
        if m <= 1:
            return 1
        Lap = [[0] * m for _ in range(m)]
        for (u, v) in self.edges:
            iu, iv = idx[u], idx[v]
            Lap[iu][iu] += 1
            Lap[iv][iv] += 1
            Lap[iu][iv] -= 1
            Lap[iv][iu] -= 1
        entries = [Lap[i][j] for i in range(1, m) for j in range(1, m)]
        return int(fmpz_mat(m - 1, m - 1, entries).det())

    # -- medial construction ------------------------------------------------

    def link(self):
        """The alternating link diagram (spherogram.Link) of the medial."""
        crossings = [spherogram.Crossing(f"e{e}") for e in range(len(self.edges))]

        def corner_after(dart):
            e, k = dart
            return crossings[e], (0 if k == 0 else 2)

        def corner_before(dart):
            e, k = dart
            return crossings[e], (3 if k == 0 else 1)

        for v, ds in self.rot.items():
            for i, d in enumerate(ds):
                nxt = ds[(i + 1) % len(ds)]
                ca, sa = corner_after(d)
                cb, sb = corner_before(nxt)
                ca[sa] = cb[sb]
        return spherogram.Link(crossings)

    # -- mutations ----------------------------------------------------------

    def series(self, e):
        """Subdivide edge e with a new degree-2 vertex."""
        u, v = self.edges[e]
        w = max(self.rot) + 1
        e2 = len(self.edges)
        self.edges[e] = (u, w)          # dart (e,1) moves to w
        self.edges.append((w, v))       # darts (e2,0) at w, (e2,1) at v
        self.rot[w] = [(e, 1), (e2, 0)]
        ds = self.rot[v]
        ds[ds.index((e, 1))] = (e2, 1)

    def parallel(self, e):
        """Add an edge parallel to e, forming a bigon."""
        u, v = self.edges[e]
        e2 = len(self.edges)
        self.edges.append((u, v))
        du = self.rot[u]
        du.insert(du.index((e, 0)) + 1, (e2, 0))
        dv = self.rot[v]
        dv.insert(dv.index((e, 1)), (e2, 1))


def tait_graph(link):
    """Embedded checkerboard graph of an alternating diagram.

    Faces are 2-colored; black faces become vertices, crossings become
    edges.  The cyclic order of crossings around a black face is read off
    the face boundary, giving the rotation system for free.
    """
    faces = link.faces()
    # face lookup: which face contains a given CrossingStrand
    face_of = {}
    for fi, f in enumerate(faces):
        for cs in f:
            face_of[(cs.crossing, cs.strand_index)] = fi
    # adjacency for 2-coloring: faces meeting across a strand-edge
    color = {0: 0}
    stack = [0]
    seen = {0}
    while stack:
        fi = stack.pop()
        for cs in faces[fi]:
            op = cs.opposite()
            fj = face_of[(op.crossing, op.strand_index)]
            if fj not in seen:
                seen.add(fj)
                color[fj] = 1 - color[fi]
                stack.append(fj)
            else:
                assert color[fj] == 1 - color[fi], "faces not 2-colorable"
    black = [fi for fi in range(len(faces)) if color[fi] == 0]
    black_index = {fi: i for i, fi in enumerate(black)}

    # each crossing borders exactly two black faces, at opposite corners
    edges = []
    rot = {i: [] for i in range(len(black))}
    edge_of_crossing = {}
    ends_assigned = {}
    for fi in black:
        vtx = black_index[fi]
        for cs in faces[fi]:
            c = cs.crossing
            if c not in edge_of_crossing:
                e = len(edges)
                edge_of_crossing[c] = e
                edges.append([vtx, None])
                ends_assigned[c] = 0
                rot[vtx].append((e, 0))
            else:
                e = edge_of_crossing[c]
                assert ends_assigned[c] == 0, "crossing on >2 black faces"
                ends_assigned[c] = 1
                edges[e][1] = vtx
                rot[vtx].append((e, 1))
    assert all(x[1] is not None for x in edges)
    G = PlanarGraph([tuple(x) for x in edges], rot)
    G.check()
    return G
