#!/usr/bin/env python3
"""Move-set experiment: is the sieve database better because of its size or its content?

    python tools/moveset.py -k 28 -N 60 --seed 1 -o paper/data/moveset/k28-s1.json

On L_{k,N} (BKZ-20, bgj1 sieve with radius 1.5, as method C of tools/ablation.py) the same
starts (the BKZ-20 rows and 300 random database vectors) are descended with different move sets:
  full        the whole database D (method C);
  rand_E      a random sample of D of the size of method E's move set for this k and seed
              (from paper/data/ablation/k{k}-s{seed}.json);
  short_E     the shortest |E| vectors of D;
  frac_f      random samples of D of size f|D| for f = 1/16, 1/8, 1/4, 1/2;
and, with the whole database, two variants of the frontier update:
  path        only the relations on the accepted path update the frontier;
  ends        only the end points of the descents.
For each variant the frontier (m <= 16), the number of entries below the table and the descent
time are written to the JSON file.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from eslp.descent import descend  # noqa: E402
from eslp.lattice import bkz, relation_basis  # noqa: E402
from eslp.sieve import sieve_db  # noqa: E402
from tools.compare import load_current  # noqa: E402

MMAX = 16

ap = argparse.ArgumentParser()
ap.add_argument("-k", type=int, required=True)
ap.add_argument("-N", type=int, default=60)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--radius", type=float, default=1.5)
ap.add_argument("--threads", type=int, default=os.cpu_count())
ap.add_argument("-o", "--out", required=True)
a = ap.parse_args()
k, N = a.k, a.N
cur = load_current("normal")
E_size = json.load(open(f"paper/data/ablation/k{k}-s{a.seed}.json"))["E_moves"]
rng = np.random.default_rng(a.seed)
R = bkz(relation_basis(list(range(1, N + 1)), k), 20, seed=a.seed)
t0 = time.time()
D = sieve_db(R, threads=a.threads, alg="bgj1", saturation_radius=a.radius)
out = {"k": k, "N": N, "seed": a.seed, "db": int(len(D)), "E_moves": int(E_size),
       "sieve_s": round(time.time() - t0, 1), "variants": {}}
starts = np.concatenate([R, D[rng.permutation(len(D))[:300]]])
sq = (D.astype(np.int64) ** 2).sum(1)


def run(name, moves, record=0):
    t = time.time()
    fr = descend(starts, moves, record=record)
    front = {m: n for m, (n, _) in fr.items() if 1 <= m <= MMAX}
    imp = sum(1 for m, n in front.items() if cur.get(k, m) is not None and n < cur.get(k, m))
    out["variants"][name] = {"moves": int(len(moves)), "record": record, "frontier": front,
                             "improved": imp, "descent_s": round(time.time() - t, 1)}
    print(f"k={k} seed={a.seed} {name:8s} |moves|={len(moves):8d} improved={imp:2d} "
          f"reached={len(front):2d} {time.time() - t:7.1f}s", flush=True)


run("full", D)
run("path", D, record=1)
run("ends", D, record=2)
nE = min(E_size, len(D))
run("rand_E", D[rng.permutation(len(D))[:nE]])
run("short_E", D[np.argsort(sq, kind="stable")[:nE]])
for f in (16, 8, 4, 2):
    run(f"frac_{f}", D[rng.permutation(len(D))[: len(D) // f]])
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
json.dump(out, open(a.out, "w"), indent=1)
