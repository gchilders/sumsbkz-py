#!/usr/bin/env python3
"""Sieve L_{k,N} with G6K-GPU (settings of the N=100 runs) and save the database.

    python tools/makedb.py -k 25 -N 100 -o results/complete/db-k25-N100.npy
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from eslp.sieve import sieve_db  # noqa: E402
from sievesearch import reduced_basis  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("-k", type=int, required=True)
ap.add_argument("-N", type=int, default=100)
ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--gpus", type=int, default=1, help="GPUs used by the sieve")
ap.add_argument("--radius", type=float, default=4 / 3, help="saturation radius rho")
ap.add_argument("-o", "--out", required=True)
a = ap.parse_args()
R = reduced_basis(list(range(1, a.N + 1)), a.k, 20, "bases")
D = sieve_db(R, threads=a.threads, alg="gpu", saturation_radius=a.radius, gpu_bucketer="bdgl",
             saturation_ratio=0.375, db_size_factor=2.77, gpus=a.gpus, gpu_params={"multi_bucket": 2, "max_nr_buckets": 0})
np.save(a.out + ".tmp.npy", D)
os.replace(a.out + ".tmp.npy", a.out)
print(f"k={a.k} N={a.N}: {len(D)} vectors", flush=True)
