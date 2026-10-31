#!/usr/bin/env python3
"""Which moments vanish on the short relations of L_{k,N}?  (The obstruction of Section 8.2.)

    python tools/forcedmoments.py [-k 17 100] [-N 60 70 80 90 100] [-o paper/data/forcedmoments.json]

For each k and N, the rows of a BKZ-20 basis of L_{k,N} serve as a sample of short relations.
For j = 1..12 with M_j(k) > 1, a moment mu_j(c) = sum_x c_x x^j is "forced" when it vanishes on
at least 90% of the rows.  Every descent whose moves are short relations then stays in the
sublattice mu_j = 0, where relations with a small side and a long side cannot exist; the cure is
to start in the cosets mu_j = -t g (tools/momentstart.py).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from eslp.lattice import bkz, relation_basis  # noqa: E402
from tools.congruence import M as Mj  # noqa: E402


def forced(k, N, jmax=12, thresh=0.9, detail=False):
    R = bkz(relation_basis(list(range(1, N + 1)), k), 20).astype(object)
    out = []
    for j in range(1, min(k, jmax + 1)):
        if Mj(k, j) <= 2:
            continue
        xj = np.array([x**j for x in range(1, N + 1)], dtype=object)
        z = float(np.mean(R.dot(xj) == 0))
        if z >= thresh or (detail and z > 0.5):
            out.append((j, Mj(k, j), z))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-k", type=int, nargs=2, default=[17, 100])
    ap.add_argument("-N", type=int, nargs="+", default=[60, 70, 80, 90, 100])
    ap.add_argument("-o", "--out", default="paper/data/forcedmoments.json")
    a = ap.parse_args()
    res = {}
    for k in range(a.k[0], a.k[1] + 1):
        row = {}
        for N in a.N:
            row[N] = forced(k, N)
        res[k] = row
        print(f"k={k:3d}  " + "  ".join(f"N={N}: " + (",".join(f"j{j}" for j, _, _ in v) or "-") for N, v in row.items()),
              flush=True)
    json.dump({str(k): {str(N): v for N, v in r.items()} for k, r in res.items()}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
