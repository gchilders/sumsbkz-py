#!/usr/bin/env python3
"""Does the descent's step limit explain the high powers where no small m is reached?

    python tools/steptest.py K [K ...]

On the saved sieve database of L_{k,100}, run the descent of method C (starts: BKZ-20 rows and
300 database vectors) with max_steps 500 and 5000, and report the smallest m reached.
"""
import sys
import time

import numpy as np

sys.path.insert(0, ".")
from eslp.descent_gpu import descend_gpu  # noqa: E402
from eslp.lattice import bkz, relation_basis  # noqa: E402

for k in map(int, sys.argv[1:]):
    D = np.load(f"results/complete/db-k{k}-N100.npy")
    R = bkz(relation_basis(list(range(1, 101)), k), 20)
    rng = np.random.default_rng(1)
    starts = np.concatenate([R, D[rng.permutation(len(D))[:300]].astype(np.int64)])
    for ms in (500, 5000):
        t = time.time()
        fr = descend_gpu(starts, D, max_steps=ms)
        small = sorted(m for m in fr if m <= 40)
        print(f"k={k} max_steps={ms}: smallest m {min(fr) if fr else None}, m<=40: {small[:8]}  ({time.time() - t:.0f}s)",
              flush=True)
