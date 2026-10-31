#!/usr/bin/env python3
"""Sensitivity of the combination (Algorithm 2) to the tie limit tau, on the pools R u H.

    python tools/tausweep.py -k 20 28 30 --tau 8 16 64 256 -o paper/data/tausweep.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from eslp.solutions import read_solutions  # noqa: E402
from tools.combine import combine_pool  # noqa: E402
from tools.compare import load_current  # noqa: E402
from tools.controls import HIST, RUNS, files, load_pool, score  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("-k", type=int, nargs="+", default=[20, 28, 30])
ap.add_argument("--tau", type=int, nargs="+", default=[8, 16, 64, 256])
ap.add_argument("--maxbase", type=int, default=120)
ap.add_argument("-o", "--out", required=True)
a = ap.parse_args()
cur = load_current("normal")
hist = [s for p in files(HIST) for s in read_solutions(p)]
runs = [s for p in files(RUNS) for s in read_solutions(p) if "[combine]" not in s.note]
out = {}
for k in a.k:
    V = np.concatenate([load_pool(runs, k, a.maxbase), load_pool(hist, k, a.maxbase)])
    for t in a.tau:
        t0 = time.time()
        F, info = combine_pool(V, ties=t)
        tot = sum(n for m, (n, _) in F.items() if m <= 16)
        out[f"{k}/{t}"] = {"improved_saved": score(F, cur, k), "sum_n_m_le16": int(tot),
                          "rounds": info.get("rounds"), "seconds": round(time.time() - t0, 1)}
        print(k, t, out[f"{k}/{t}"], flush=True)
json.dump(out, open(a.out, "w"), indent=1)
