#!/usr/bin/env python3
"""Progressive versus random-start sieving, and the contents of the sieve database.

    python tools/dbstats.py -k 28 -N 70 --radius 1.45 --seed 1 -o paper/data/dbstats/k28-N70-s1.json

On L_{k,N} after BKZ-20: (P) the progressive sieve of eslp/sieve.sieve_db, and (R) the
full-dimensional sieve started from random samples (start_dim >= d).  For each, record
the time, the database statistics of tools/ablation.db_stats and the frontier (m <= 16)
of the descent of method C with the same starts seed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eslp.descent import descend  # noqa: E402
from eslp.lattice import bkz, relation_basis  # noqa: E402
from eslp.sieve import frontier_of, sieve_db  # noqa: E402
from tools.ablation import MMAX, db_stats, steps_summary  # noqa: E402
from tools.compare import load_current  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-k", type=int, required=True)
    ap.add_argument("-N", type=int, default=70)
    ap.add_argument("--radius", type=float, default=1.45)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()
    cur = load_current("normal")
    xs = list(range(1, a.N + 1))
    R = bkz(relation_basis(xs, a.k), 20)
    d = R.shape[0]
    out = {"k": a.k, "N": a.N, "radius": a.radius, "seed": a.seed, "threads": a.threads}
    for name, sd in (("progressive", 50), ("random", d + 1)):
        t0 = time.time()
        D = sieve_db(R, threads=a.threads, alg="bgj1", saturation_radius=a.radius, start_dim=sd)
        ts = time.time() - t0
        st = db_stats(D, a.k, a.N, a.radius)
        rng = np.random.default_rng(a.seed)
        starts = np.concatenate([R, D[rng.permutation(len(D))[:300]]])
        t0 = time.time()
        fr = descend(starts, D)
        td = time.time() - t0
        for m, (n, i) in frontier_of(D).items():
            if m not in fr or n < fr[m][0]:
                fr[m] = (n, D[i])
        front = {m: fr[m][0] for m in range(1, MMAX + 1) if m in fr}
        out[name] = {"sieve_s": round(ts, 1), "descent_s": round(td, 1), "db": st,
                     "descent_steps": steps_summary(), "frontier": front,
                     "improved": sum(1 for m, n in front.items()
                                     if cur.get(a.k, m) is not None and n < cur.get(a.k, m))}
        print(name, json.dumps(out[name]), flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
