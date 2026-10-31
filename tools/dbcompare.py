#!/usr/bin/env python3
"""Compare saved sieve databases of L_{k,100} (why does k=50 resist coset starts?).

    python tools/dbcompare.py 50:2 82:2 76:4 89:1 98:2

For each k:j it reports, for the database D (results/complete/db-k{k}-N100.npy):
  - the fraction of vectors with mu_j = 0 and the cosets mu_j/g of the others;
  - the mass profile: the share of sum |c_x| on the bases in each quarter of 1..N;
  - the term-count change p-q of a move (|p-q| mean and quantiles), overall and for the moves that
    keep mu_j = 0: within a coset, a descent can shorten the long side only through moves with
    p-q != 0, which trade many small bases for fewer large ones;
  - for those moves, the root-mean-square base of their positive and negative parts.
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tools.congruence import M  # noqa: E402

N = 100


def stats(k, j, sample=1_000_000, seed=1):
    D = np.load(f"results/complete/db-k{k}-N{N}.npy", mmap_mode="r")
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(D), size=min(sample, len(D)), replace=False))
    C = np.asarray(D[idx], dtype=np.int64)
    x = np.arange(1, N + 1, dtype=np.int64)
    mu = (C * (x ** j)).sum(1)                       # fits in int64 for N=100, j<=4
    g = 0
    for v in np.unique(np.abs(mu[mu != 0]))[:2000]:
        g = math.gcd(g, int(v))
    A = np.abs(C)
    T = A.sum(1)
    p = np.where(C > 0, C, 0).sum(1)
    q = np.where(C < 0, -C, 0).sum(1)
    d = p - q
    z = mu == 0
    out = {"k": k, "j": j, "M": M(k, j), "g_sample": g, "size": len(D), "zero": float(z.mean())}
    if (~z).any():
        t = mu[~z] // g if g else mu[~z]
        vals, cnt = np.unique(np.abs(t), return_counts=True)
        out["cosets"] = {int(a): int(b) for a, b in zip(vals[:6], cnt[:6])}
    q4 = [A[:, a:b].sum() / A.sum() for a, b in ((0, 25), (25, 50), (50, 75), (75, 100))]
    out["mass_quarters"] = [round(float(v), 3) for v in q4]
    out["T_median"] = float(np.median(T))
    for name, sel in (("all", slice(None)), ("mu0", z)):
        dd = np.abs(d[sel])
        out[f"|p-q| {name}"] = (round(float(dd.mean()), 2), [int(np.quantile(dd, s)) for s in (0.5, 0.9, 0.99)],
                                round(float((dd >= 4).mean()), 4))
    # for mu=0 moves with p != q: rms base of the side with more terms vs the side with fewer
    sel = z & (d != 0)
    if sel.any():
        Cs = C[sel]
        sgn = np.sign(d[sel])[:, None]
        more = np.where(Cs * sgn > 0, np.abs(Cs), 0)
        fewer = np.where(Cs * sgn < 0, np.abs(Cs), 0)
        rms = lambda W: float(np.sqrt((W * x ** 2).sum() / W.sum()))
        out["rms base (more side, fewer side)"] = (round(rms(more), 1), round(rms(fewer), 1))
    return out


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        k, j = (int(v) for v in arg.split(":"))
        if not os.path.exists(f"results/complete/db-k{k}-N{N}.npy"):
            print(f"k={k}: no database", flush=True)
            continue
        print(stats(k, j), flush=True)
