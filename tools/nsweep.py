#!/usr/bin/env python3
"""Frontier against the number of bases N at a fixed time budget.

    python tools/nsweep.py -k 28 -N 70 --radius 1.45 --budget 3600 -o paper/data/nsweep/k28-N70.json

Repeats method C of tools/ablation.py (BKZ-20, progressive bgj1 sieve, descent with the
database as moves) on L_{k,N}, each run on a new random ordering of the bases, starting
a new run while less than `budget` seconds have passed.  Records the best n for each
m <= 40 over all runs, with the run count and the time used.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eslp.lattice import bkz, relation_basis, verify  # noqa: E402
from eslp.solutions import Solution  # noqa: E402
from tools.ablation import run_C  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-k", type=int, required=True)
    ap.add_argument("-N", type=int, required=True)
    ap.add_argument("--radius", type=float, required=True)
    ap.add_argument("--budget", type=float, default=3600)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    xs = list(range(1, a.N + 1))
    best, runs, t0 = {}, [], time.time()
    while time.time() - t0 < a.budget:
        perm = xs[:]
        rng.shuffle(perm)
        t1 = time.time()
        R = bkz(relation_basis(perm, a.k), 20, seed=rng.randrange(2**31))
        _, fr = run_C(R, a.radius, rng.randrange(2**31), a.threads)
        inv = np.argsort(perm)          # back to the order 1..N
        for m, (n, v) in fr.items():
            if m <= 40 and (m not in best or n < best[m][0]):
                w = np.asarray(v, dtype=np.int64)[inv]
                assert verify(xs, a.k, w)
                best[m] = (n, w)
        runs.append(round(time.time() - t1, 1))
        print(f"run {len(runs)} {runs[-1]}s  " + " ".join(f"{m}:{best[m][0]}" for m in sorted(best) if m <= 16), flush=True)
    out = {"k": a.k, "N": a.N, "radius": a.radius, "budget": a.budget, "threads": a.threads,
           "runs": runs, "seconds": round(time.time() - t0, 1),
           "frontier": {m: n for m, (n, _) in sorted(best.items())},
           "identities": [Solution(a.k, {x: int(c) for x, c in zip(xs, v) if c}).normalized().format()
                          for _, (_, v) in sorted(best.items())]}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
