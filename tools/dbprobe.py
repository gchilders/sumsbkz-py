#!/usr/bin/env python3
"""Which moments vanish identically on the saved N=100 databases?

    python tools/dbprobe.py K [K ...]

For j = 1..8: the fraction of database vectors c with sum_x c_x x^j = 0, and M_j(k).
A fraction near 1 means that, for relations as short as the sieve's, the congruence of
Proposition 8.1 forces the j-th moment to vanish, although the worst-case bound T N^j
exceeds M_j(k).
"""
import sys

import numpy as np

sys.path.insert(0, ".")
from tools.congruence import M as Mj  # noqa: E402

x = np.arange(1, 101, dtype=object)
for k in map(int, sys.argv[1:]):
    D = np.load(f"results/complete/db-k{k}-N100.npy")
    S = D[np.random.default_rng(0).permutation(len(D))[:20000]].astype(object)
    out = []
    for j in range(1, 9):
        mom = S.dot(x**j)
        z = float(np.mean(mom == 0))
        if z > 0.01 or j <= 2:
            out.append(f"j={j}: {z:.2f} (M_j={Mj(k, j)})")
    print(f"k={k}: " + "; ".join(out), flush=True)
