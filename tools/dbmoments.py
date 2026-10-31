#!/usr/bin/env python3
"""Forced moments on the saved sieve databases, and database statistics for given powers.

    python tools/dbmoments.py -o paper/data/dbmoments.json [--stats 17 18 19]

For every power with a forced moment (paper/data/forcedmoments.json, N=100) and a saved
database results/complete/db-k{k}-N100.npy, the fraction of database vectors with mu_j = 0.
With --stats, also the statistics of tools/dbcompare.py for the listed powers (whether or not a
moment is forced), together with the distribution of min(p,q).
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

N = 100


def load(k):
    p = f"results/complete/db-k{k}-N{N}.npy"
    return np.load(p, mmap_mode="r") if os.path.exists(p) else None


def zero_fraction(D, j, chunk=1 << 20):
    x = np.arange(1, N + 1, dtype=np.int64) ** j
    z = 0
    for i in range(0, len(D), chunk):
        z += int(((np.asarray(D[i:i + chunk], dtype=np.int64) * x).sum(1) == 0).sum())
    return z / len(D)


def stats(D):
    C = np.asarray(D, dtype=np.int64)
    A = np.abs(C)
    p = np.where(C > 0, C, 0).sum(1)
    q = np.where(C < 0, -C, 0).sum(1)
    T, m = p + q, np.minimum(p, q)
    return {"size": int(len(C)), "T_median": float(np.median(T)),
            "minpq_quantiles": [int(np.quantile(m, s)) for s in (0, 0.001, 0.01, 0.5)],
            "minpq_le16": int((m <= 16).sum()), "rel_imbalance_median": float(np.median(np.abs(p - q) / T)),
            "mass_quarters": [round(float(A[:, a:b].sum() / A.sum()), 3) for a, b in
                              ((0, 25), (25, 50), (50, 75), (75, 100))]}


ap = argparse.ArgumentParser()
ap.add_argument("-o", "--out", required=True)
ap.add_argument("--stats", type=int, nargs="*", default=[])
a = ap.parse_args()
forced = json.load(open("paper/data/forcedmoments.json"))
out = {"zero_fraction": {}, "stats": {}}
for k, row in sorted(forced.items(), key=lambda t: int(t[0])):
    v = row.get(str(N), [])
    if not v:
        continue
    D = load(int(k))
    if D is None:
        continue
    j = v[0][0]
    out["zero_fraction"][k] = {"j": j, "size": int(len(D)), "zero": zero_fraction(D, j)}
    print(k, out["zero_fraction"][k], flush=True)
for k in a.stats:
    D = load(k)
    if D is not None:
        out["stats"][str(k)] = stats(D)
        print(k, out["stats"][str(k)], flush=True)
json.dump(out, open(a.out, "w"), indent=1)
