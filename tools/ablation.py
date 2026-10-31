#!/usr/bin/env python3
"""Ablation for the paper: what each stage of the pipeline contributes, on fixed lattices.

    python tools/ablation.py -k 28 -N 60 --seed 1 --radius 1.5 -o paper/data/ablation/k28-s1.csv

On the relation lattice of bases 1..N, for one k and one seed:
  B  the sieve database alone (best n per m among its vectors);
  C  sieve + descent (moves = database, starts = BKZ-20 rows + 300 database vectors);
  D  three C runs (different random starts, same reduced basis), their frontier
     identities then combined by tools/combine.combine_pool;
  A  BKZ-20 + Yura's routine (the 2003 method) on fresh random orderings of the bases,
     in parallel processes (one per thread C uses) for the wall time of D;
  E  reduction only, with current methods: BKZ 2.0 with block size 40 (after BKZ-20) on
     fresh random orderings, in parallel processes, for half the wall time of D; the
     reduced rows of all passes are then the moves, and the starts are the rows of one
     basis and 300 random rows, of the same descent as C.
Writes one row per (method, m) with the best n, and a JSON file with timings and
statistics (database lengths, saturation, descent steps).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eslp import descent as descent_mod  # noqa: E402
from eslp import sieve as sieve_mod  # noqa: E402
from eslp.descent import descend  # noqa: E402
from eslp.greedy import yura  # noqa: E402
from eslp.lattice import bkz, relation_basis, verify  # noqa: E402
from eslp.sieve import frontier_of, sieve_db  # noqa: E402
from eslp.solutions import Solution  # noqa: E402
from tools.combine import combine_pool  # noqa: E402
from tools.compare import load_current  # noqa: E402

MMAX = 16
E_BLOCK = 40


def gh2(k, N, exact=True):
    """GH(L_{k,N})^2: exact ball volume (as G6K uses) or the asymptotic form of the paper."""
    d = N - 1
    logdet = 0.5 * math.log(sum(x ** (2 * k) for x in range(1, N + 1)))
    if exact:
        return math.exp(2 * (math.lgamma(d / 2 + 1) + logdet) / d) / math.pi
    return d / (2 * math.pi * math.e) * math.exp(2 * logdet / d)


def _A_worker(args):
    """One worker of method A: BKZ-20 + Yura's step on fresh random orderings until the deadline."""
    xs, k, seed, deadline = args
    rng = random.Random(seed)
    out, passes = [], 0
    while time.time() < deadline:
        perm = list(range(len(xs)))
        rng.shuffle(perm)
        RA = bkz(relation_basis([xs[i] for i in perm], k), 20, seed=rng.randrange(2**31))
        C = yura(RA)
        Cfull = np.zeros((len(C), len(xs)), dtype=np.int64)
        Cfull[:, perm] = C
        out.append(Cfull)
        passes += 1
    return np.concatenate(out) if out else np.zeros((0, len(xs)), dtype=np.int64), passes


def _E_worker(args):
    """One worker of method E: BKZ-20 then BKZ 2.0 block size 40 on fresh orderings until the deadline."""
    xs, k, seed, deadline = args
    rng = random.Random(seed)
    out, passes = [], 0
    while time.time() < deadline:
        perm = list(range(len(xs)))
        rng.shuffle(perm)
        RE = bkz(bkz(relation_basis([xs[i] for i in perm], k), 20, seed=rng.randrange(2**31)),
                 E_BLOCK, seed=rng.randrange(2**31))
        Efull = np.zeros_like(RE)
        Efull[:, perm] = RE
        out.append(Efull)
        passes += 1
    return np.concatenate(out) if out else np.zeros((0, len(xs)), dtype=np.int64), passes


def canonical(V):
    V = V[np.any(V != 0, axis=1)]
    first = np.argmax(V != 0, axis=1)
    V = V * np.sign(V[np.arange(len(V)), first])[:, None]
    return np.unique(V, axis=0)


def best_rows(fr):
    return {m: n for m, (n, _) in fr.items() if m <= MMAX}


def steps_summary():
    s = descent_mod.LAST_STEPS
    return {"tasks": int(len(s)), "median": float(np.median(s)), "p99": float(np.percentile(s, 99)),
            "max": int(s.max()), "at_cap": int((s >= 500).sum())}


def db_stats(D, k, N, radius):
    sq = (D.astype(np.int64) ** 2).sum(axis=1)
    pos = np.where(D > 0, D, 0).sum(axis=1)
    neg = np.where(D < 0, -D, 0).sum(axis=1)
    T, m = pos + neg, np.minimum(pos, neg)
    g2, g2a = gh2(k, N), gh2(k, N, exact=False)
    q = lambda x: [float(np.percentile(x, p)) for p in (0, 10, 50, 90, 100)]  # noqa: E731
    d = N - 1
    inside = int((sq <= radius * g2).sum())
    return {"size": int(len(D)), "gh2_exact": g2, "gh2_asym": g2a,
            "norm2_over_gh2_q": q(sq / g2), "frac_outside_radius": float((sq > radius * g2).mean()),
            "sat_achieved": inside / (0.5 * radius ** (d / 2)),
            "T_q": q(T), "minpq_q": q(m), "minpq_le16": int((m <= 16).sum()),
            "max_T": int(T.max()), "max_norm2": int(sq.max()),
            "sieves": sieve_mod.LAST_STATS.get("sieves"), "unsaturated": sieve_mod.LAST_STATS.get("unsaturated"),
            "final_unsaturated": bool(sieve_mod.LAST_STATS.get("full_unsaturated"))}


def run_C(R, radius, seed, threads, stats=None, k=None, N=None):
    t0 = time.time()
    D = sieve_db(R, threads=threads, alg="bgj1", saturation_radius=radius)
    ts = time.time() - t0
    if stats is not None:
        stats["db"] = db_stats(D, k, N, radius)
        stats["sieve_s"] = round(ts, 1)
    rng = np.random.default_rng(seed)
    starts = np.concatenate([R, D[rng.permutation(len(D))[:300]]])
    fr = descend(starts, D)
    if stats is not None:
        stats["descent_steps"] = steps_summary()
    for m, (n, i) in frontier_of(D).items():
        if m not in fr or n < fr[m][0]:
            fr[m] = (n, D[i])
    return D, fr


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-k", type=int, required=True)
    ap.add_argument("-N", type=int, default=60)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--radius", type=float, default=1.5)
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("-o", "--out", required=True, help="CSV; statistics go to the same name with .json")
    ap.add_argument("--identities", default="results/ablation_identities.txt")
    a = ap.parse_args()
    k, s0 = a.k, 100 * a.seed
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    cur = load_current("normal")
    xs = list(range(1, a.N + 1))
    stats = {"k": k, "N": a.N, "seed": a.seed, "radius": a.radius, "threads": a.threads}
    R = bkz(relation_basis(xs, k), 20)

    # C (timed), B (its database) and D (three C runs + combination)
    t0 = time.time()
    D1, fr1 = run_C(R, a.radius, s0 + 1, a.threads, stats, k, a.N)
    tC = time.time() - t0
    frB, frC = best_rows(frontier_of(D1)), best_rows(fr1)
    pool = [v for _, v in fr1.values()]
    for sd in (s0 + 2, s0 + 3):
        _, fr = run_C(R, a.radius, sd, a.threads)
        pool += [v for _, v in fr.values()]
    t1 = time.time()
    frcomb, info = combine_pool(np.array(pool, dtype=np.int32))
    stats["combine"] = dict(info, seconds=round(time.time() - t1, 1))
    tD = time.time() - t0
    frD = best_rows(frcomb)
    found = pool + [v for _, v in frcomb.values()]

    import multiprocessing
    Pool = multiprocessing.get_context("spawn").Pool
    # A: the 2003 method, on as many processes as C has threads, for the wall time of D
    t0 = time.time()
    with Pool(a.threads) as P:
        parts = P.map(_A_worker, [(xs, k, 10_000 * a.seed + 100 * k + i, t0 + tD) for i in range(a.threads)])
    tA = time.time() - t0
    VA = np.concatenate([v for v, _ in parts])
    stats["A_passes"] = sum(p for _, p in parts)
    frA = best_rows(frontier_of(VA))
    found += list(VA)

    # E: BKZ-40 reduction for half of D's time, then the descent over all reduced rows
    t0 = time.time()
    with Pool(a.threads) as P:
        parts = P.map(_E_worker, [(xs, k, 20_000 * a.seed + 100 * k + i, t0 + tD / 2) for i in range(a.threads)])
    VE = canonical(np.concatenate([v for v, _ in parts]))
    stats["E_passes"], stats["E_moves"] = sum(p for _, p in parts), int(len(VE))
    rng = np.random.default_rng(s0 + 4)
    starts = np.concatenate([R, VE[rng.permutation(len(VE))[:300]]])
    frE = descend(starts, VE)
    stats["E_descent_steps"] = steps_summary()
    for m, (n, i) in frontier_of(VE).items():
        if m not in frE or n < frE[m][0]:
            frE[m] = (n, VE[i])
    tE = time.time() - t0
    found += [v for _, v in frE.values()]
    frE = best_rows(frE)

    # save the best identity per m found by any method, verified exactly
    allbest = {}
    for v in found:
        v = np.asarray(v, dtype=np.int64)
        if not v.any():
            continue
        pp, qq = int(v[v > 0].sum()), int(-v[v < 0].sum())
        m, n = min(pp, qq), max(pp, qq)
        if m <= 40 and (m not in allbest or n < allbest[m][0]):
            allbest[m] = (n, v)
    with open(a.identities, "a") as fh:
        fh.write(f"// ablation k={k} N={a.N} radius={a.radius} seed={a.seed}\n")
        for m, (n, v) in sorted(allbest.items()):
            assert verify(xs, k, v)
            fh.write(Solution(k, {x: int(c) for x, c in zip(xs, v) if c}).normalized().format() + "\n")

    rows = []
    for name, fr, t in (("A", frA, tA), ("B", frB, stats["sieve_s"]), ("C", frC, tC),
                        ("D", frD, tD), ("E", frE, tE)):
        wins = sum(1 for m, n in fr.items() if cur.get(k, m) is not None and n < cur.get(k, m))
        for m in range(1, MMAX + 1):
            rows.append({"k": k, "N": a.N, "seed": a.seed, "method": name, "m": m, "n": fr.get(m, ""),
                         "record": cur.get(k, m) or "", "seconds": round(t, 1)})
        print(f"k={k} seed={a.seed} {name}: {wins:2d} entries below the table, time {t:7.1f}s", flush=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    json.dump(stats, open(os.path.splitext(a.out)[0] + ".json", "w"), indent=1)
    print(json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
