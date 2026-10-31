#!/usr/bin/env python3
"""Descent started from known identities (are the records local minima for the descent?).

    python tools/recordstart.py -k 50 -N 117 --per 25 -o results/k50exp/recstart-k50-N117.txt

The starts are the identities for k in Meyrignac's database and the 2003 archive with bases at
most N (and m <= 40), each alone and plus or minus random vectors of the saved database
results/complete/db-k{k}-N{N}.npy; the moves are that database.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from eslp.solutions import Solution, read_solutions  # noqa: E402
from tools.compare import load_current  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("-k", type=int, required=True)
ap.add_argument("-N", type=int, required=True)
ap.add_argument("--per", type=int, default=25, help="random database shifts per identity")
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--db", default=None, help="database file (default results/complete/db-k{k}-N{N}.npy)")
ap.add_argument("-o", "--out", required=True)
a = ap.parse_args()
k, N = a.k, a.N
D = np.load(a.db or f"results/complete/db-k{k}-N{N}.npy")
rng = np.random.default_rng(a.seed)
ids = []
for p in ("data/euler_free_fr/solutions_normal.txt", "data/archive/database.txt"):
    for s in read_solutions(p):
        if s.k == k and s.m <= 40 and max(s.coeffs) <= N and s.is_valid():
            v = np.zeros(N, dtype=np.int64)
            for x, c in s.coeffs.items():
                v[x - 1] = c
            ids.append(v)
ids = np.unique(np.array(ids), axis=0)
known = {tuple(np.sign(v[np.flatnonzero(v)[0]]) * v) for v in ids}
starts = []
for v in ids:
    starts.append(v)
    sh = D[rng.permutation(len(D))[: a.per]].astype(np.int64)
    starts.extend(list(v[None, :] + sh))
    starts.extend(list(v[None, :] - sh))
starts = np.array(starts)
print(f"k={k} N={N}: {len(ids)} identities, {len(starts)} starts", flush=True)
from eslp.descent_gpu import descend_gpu  # noqa: E402
fr = {}
CH = 580                     # starts per call: rows = starts x 7 weights must stay below 4096
for i in range(0, len(starts), CH):
    for m, (n, v) in descend_gpu(starts[i:i + CH], D).items():
        if m not in fr or n < fr[m][0]:
            fr[m] = (n, v)
    print(f"  {min(i + CH, len(starts))}/{len(starts)} starts done", flush=True)
cur = load_current("normal")
cells = []
with open(a.out, "a") as f:
    f.write(f"// recordstart k={k} N={N} per={a.per} seed={a.seed}\n")
    for m, (n, v) in sorted(fr.items()):
        if m > 40:
            continue
        s = Solution(k, {x + 1: int(c) for x, c in enumerate(v) if c}).normalized()
        assert s.is_valid()
        if tuple(np.sign(v[np.flatnonzero(v)[0]]) * v) in known:
            continue                     # a starting identity itself, not a new one
        f.write(s.format() + "\n")
        r = cur.get(k, s.m)
        if s.m <= 16:
            cells.append(f"{s.m}:{s.n}" + ("" if r is None else ("<%d!" % r if s.n < r else "/%d" % r)))
print(f"  smallest m: {min(fr) if fr else None};  " + "  ".join(cells), flush=True)
