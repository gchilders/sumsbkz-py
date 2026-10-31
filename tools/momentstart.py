#!/usr/bin/env python3
"""Descents started in the cosets of the zero-moment sublattice (the fix for forced moments).

    python tools/momentstart.py -k 89 -j 1 [-N 100] [--t 2,3,4,6,8] -o results/complete/mstart-k89.txt

On the relation lattice L_{k,N}, every short vector has j-th moment mu_j(c) = sum c_x x^j = 0 when
M_j(k) is large compared with sqrt(T) N^j (Proposition 8.1), so the sieve database, and every
descent that uses it, stays in the sublattice mu_j = 0, where relations with a small side and a
long side do not exist.  Here the starts are relations with mu_j = -t g, g the generator of
mu_j(L) (a multiple of M_j(k)), built from a reduced basis by the extended gcd; the database
moves preserve mu_j, so the descent explores the coset mu_j = -t g.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from eslp.lattice import bkz, relation_basis  # noqa: E402
from eslp.solutions import Solution  # noqa: E402
from tools.compare import load_current  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("-k", type=int, required=True)
ap.add_argument("-j", type=int, required=True)
ap.add_argument("-N", type=int, default=100)
ap.add_argument("--t", default="1,2,3,4,6,8,12")
ap.add_argument("--per-t", type=int, default=60)
ap.add_argument("--seed", type=int, default=0, help="varies the random database shifts of the starts")
ap.add_argument("-o", "--out", required=True)
ap.add_argument("--db", default=None, help="database file (default results/complete/db-k{k}-N{N}.npy)")
a = ap.parse_args()
k, j, N = a.k, a.j, a.N
cur = load_current("normal")
D = np.load(a.db or f"results/complete/db-k{k}-N{N}.npy")
R = bkz(relation_basis(list(range(1, N + 1)), k), 20).astype(object)
xj = np.array([x**j for x in range(1, N + 1)], dtype=object)
mu = [int(r.dot(xj)) for r in R]
# extended gcd over the basis rows: a lattice vector g_vec with mu_j = g = gcd of all mu_j
g, gv = 0, np.zeros(N, dtype=object)
for r, m in zip(R, mu):
    if m == 0:
        continue
    if g == 0:
        g, gv = m, r.copy()
        continue
    a0, b0, x0, x1, y0, y1 = g, m, 1, 0, 0, 1
    while b0:
        qq = a0 // b0
        a0, b0 = b0, a0 - qq * b0
        x0, x1 = x1, x0 - qq * x1
        y0, y1 = y1, y0 - qq * y1
    g, gv = a0, x0 * gv + y0 * r
if g < 0:
    g, gv = -g, -gv
print(f"k={k} j={j}: mu_j(L) = {g} Z (M_j = see congruence.py); generator |c|_1 {int(sum(abs(int(v)) for v in gv))}", flush=True)
# shorten the generator within its coset by subtracting zero-moment rows (Babai on the rows with mu=0)
from fpylll import FPLLL, GSO, IntegerMatrix  # noqa: E402
Z = [list(map(int, r)) for r, m in zip(R, mu) if m == 0]
B0 = IntegerMatrix.from_matrix(bkz(np.array(Z, dtype=np.int64), 20).tolist()) if len(Z) else None


def shorten(c):
    if B0 is None:
        return c
    old = FPLLL.get_precision()
    FPLLL.set_precision(max(53, int(max(abs(int(v)) for v in c)).bit_length() + 64))
    try:
        G = GSO.Mat(B0, float_type="mpfr")
        G.update_gso()
        for _ in range(3):
            coef = G.babai([int(v) for v in c])
            close = np.array(B0.multiply_left(coef), dtype=object)
            if not any(close):
                break
            c = c - close
    finally:
        FPLLL.set_precision(old)
    return c


rng = np.random.default_rng(k + 1000 * a.seed)
starts = []
for t in (int(s) for s in a.t.split(",")):
    for sgn in (1, -1):
        c = shorten(sgn * t * gv)
        if max(abs(int(v)) for v in c) >= 2**31:
            continue
        c = np.array([int(v) for v in c], dtype=np.int64)
        starts.append(c)
        sh = D[rng.permutation(len(D))[: a.per_t]].astype(np.int64)
        starts.extend(list(c[None, :] + sh))
starts = np.array(starts)
print(f"  {len(starts)} starts; start |c|_1 median {np.median(np.abs(starts).sum(1)):.0f}", flush=True)
from eslp.descent_gpu import descend_gpu  # noqa: E402
fr = descend_gpu(starts, D)
cells = []
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
with open(a.out, "a") as f:
    f.write(f"// momentstart k={k} j={j} N={N} t={a.t} seed={a.seed}\n")
    for m, (n, v) in sorted(fr.items()):
        if m > 40:
            continue
        s = Solution(k, {x + 1: int(c) for x, c in enumerate(v) if c}).normalized()
        assert s.is_valid()
        r = cur.get(k, s.m)
        tag = f" // NEW record (was {r})" if r is not None and s.n < r else ""
        f.write(s.format() + tag + "\n")
        if s.m <= 16:
            cells.append(f"{s.m}:{s.n}" + ("" if r is None else ("<%d!" % r if s.n < r else "/%d" % r)))
print(f"  smallest m: {min(fr) if fr else None};  k={k}  " + "  ".join(cells), flush=True)
