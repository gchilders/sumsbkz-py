#!/usr/bin/env python3
"""sievesearch.py -- sieve + descent search for equal sums of like powers.

For each power k and each chosen set of bases, build the relation lattice,
BKZ-reduce it, run a G6K sieve to saturate a ball of radius sqrt(radius)*GH,
then walk relations downhill in w*m + total using the sieve database as the
move set (eslp/descent.py).  Every relation examined is checked against the
current euler.free.fr record table; ties and improvements are written out
after exact verification.

Examples:
    # all bases 1..50, k = 20..30, one lattice per k
    python sievesearch.py -k 20 30 -N 50 --radius 1.5
    # 10 random 50-element subsets of 1..80 per k
    python sievesearch.py -k 20 30 -N 80 --subset 50 --runs 10 --radius 1.5
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time

import numpy as np

from eslp.descent import descend
from eslp.lattice import bkz, first_primes, relation_basis, verify
from eslp.sieve import frontier_of, sieve_db
from eslp.solutions import Solution
from tools.compare import load, load_current

HERE = os.path.dirname(os.path.abspath(__file__))


def choose(N, subset, primes, rng, keep_low=0, divisible_by=(), extra=()):
    pool = first_primes(N) if primes else list(range(1, N + 1))
    if divisible_by:
        # sumsbkzmod: 1 plus multiples of the given primes, sidestepping the
        # x^k = 0/1 (mod p) count constraint for primes p with (p-1) | k
        pool, x = [1], 1
        while len(pool) < N:
            x += 1
            if any(x % p == 0 for p in divisible_by):
                pool.append(x)
    if extra:
        # add multiples p*j (j <= M) beyond the range: for primes p with (p-1) | k the
        # counts of terms prime to p on the two sides must agree mod p, and lopsided
        # (low m) solutions need many multiples of p to satisfy that
        pool = sorted(set(pool) | {p * j for p, M in extra for j in range(1, M + 1)})
    if subset and subset < len(pool):
        low, rest = pool[:keep_low], pool[keep_low:]
        xs = sorted(low + rng.sample(rest, subset - len(low)))
    else:
        xs = pool
    return xs


def reduced_basis(xs, k, block, cache=None):
    """BKZ-reduced relation basis, optionally cached on disk (keeps a GPU from waiting on BKZ)."""
    path = None
    if cache:
        import hashlib
        tag = hashlib.sha1(",".join(map(str, xs)).encode()).hexdigest()[:10]
        path = os.path.join(cache, f"k{k}_n{len(xs)}_b{block}_{tag}.npy")
        if os.path.exists(path):
            return np.load(path)
    R = bkz(relation_basis(xs, k), block)
    if path:
        os.makedirs(cache, exist_ok=True)
        np.save(path + ".tmp.npy", R)
        os.replace(path + ".tmp.npy", path)
    return R


def search_one(xs, k, a, R=None):
    t0 = time.time()
    if R is None:
        R = reduced_basis(xs, k, a.block, a.basis_cache)
    D = sieve_db(R, threads=a.threads, alg=a.alg, saturation_radius=a.radius,
                 gpu_bucketer=a.gpu_bucketer, db_cap=a.db_cap, gpus=a.gpus,
                 db_size_factor=a.db_size_factor, saturation_ratio=a.saturation_ratio,
                 gpu_params={k: v for k, v in (("multi_bucket", a.multi_bucket),
                                               ("max_nr_buckets", a.max_nr_buckets)) if v is not None})
    t1 = time.time()
    if a.db_cap and len(D) > a.db_cap:
        D = D[np.argsort((D.astype(np.int64) ** 2).sum(1), kind="stable")[: a.db_cap]]
    moves = D
    if a.moves and len(D) > a.moves:
        moves = D[np.argsort((D.astype(np.int64) ** 2).sum(1), kind="stable")[: a.moves]]
    rng = np.random.default_rng(a.seed_np.integers(2**63))
    pick = rng.permutation(len(D))[: a.starts]
    starts = np.concatenate([R, D[pick]])
    if a.gpu_descent:
        from eslp.descent_gpu import descend_gpu
        fr = descend_gpu(starts, moves, max_steps=a.max_steps)
    else:
        fr = descend(starts, moves, max_steps=a.max_steps)
    for m, (n, i) in frontier_of(D).items():
        if m not in fr or n < fr[m][0]:
            fr[m] = (n, D[i])
    return fr, len(D), t1 - t0, time.time() - t1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-k", type=int, nargs="+", required=True, help="power, or kmin kmax")
    ap.add_argument("-N", type=int, default=50, help="bases drawn from 1..N (or the first N primes)")
    ap.add_argument("--subset", type=int, default=0, help="use random subsets of this size (0 = all N)")
    ap.add_argument("--keep-low", type=int, default=0,
                    help="with --subset, always include the smallest this-many bases")
    ap.add_argument("--divisible-by", type=lambda t: [int(x) for x in t.split(",")], default=[],
                    help="bases = 1 plus the first N-1 multiples of any of these primes (e.g. 3,19 for k=18)")
    ap.add_argument("--extra", type=lambda t: [tuple(int(y) for y in x.split(":")) for x in t.split(",")],
                    default=[], help="add multiples p*j, j<=M, as 'p:M,q:M2' (e.g. 19:8,7:20 for k=18)")
    ap.add_argument("--runs", type=int, default=1, help="lattices per k")
    ap.add_argument("--primes", action="store_true")
    ap.add_argument("--radius", type=float, default=1.5, help="G6K saturation radius (squared, x GH^2)")
    ap.add_argument("--alg", default="bgj1", help="G6K sieve: bgj1 / bdgl / gauss")
    ap.add_argument("--gpus", type=int, default=1, help="with --alg gpu: GPUs used by one sieve")
    ap.add_argument("--db-size-factor", type=float, default=3.2, help="G6K db size = factor * base^dim")
    ap.add_argument("--saturation-ratio", type=float, default=0.5)
    ap.add_argument("--multi-bucket", type=int, default=None, help="G6K-GPU multi_bucket (authors used 2)")
    ap.add_argument("--max-nr-buckets", type=int, default=None, help="G6K-GPU max_nr_buckets (authors used 0)")
    ap.add_argument("--gpu-bucketer", default="bgj1", help="with --alg gpu: bgj1 or bdgl")
    ap.add_argument("--gpu-descent", action="store_true", help="run the descent on the GPU (numba.cuda)")
    ap.add_argument("--prefetch", type=int, default=0,
                    help="reduce all bases up front in this many worker processes (keeps a GPU fed)")
    ap.add_argument("--basis-cache", default=None, help="directory to save/reuse reduced bases")
    ap.add_argument("--bases-only", action="store_true", help="only compute (and cache) the reduced bases")
    ap.add_argument("--db-cap", type=int, default=0, help="keep only the shortest this-many sieve vectors")
    ap.add_argument("--block", type=int, default=20, help="BKZ preprocessing block size")
    ap.add_argument("--starts", type=int, default=300, help="sieve vectors used as descent starts")
    ap.add_argument("--moves", type=int, default=0, help="cap on move-set size (shortest first); 0 = whole DB")
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--slack", type=int, default=0, help="write solutions within this many terms of the record")
    ap.add_argument("--mmax", type=int, default=40)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("-o", "--out", default=None, help="append solutions here (default results/sieve_<kind>.txt)")
    a = ap.parse_args(argv)

    kmin, kmax = (a.k * 2)[:2]
    kind = "prime" if a.primes else "normal"
    out_path = a.out or os.path.join(HERE, "results", f"sieve_{kind}.txt")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    rng = random.Random(a.seed)
    a.seed_np = np.random.default_rng(a.seed)

    cur = load_current(kind)
    # also count what this tool has already written, so reruns only report fresh gains
    mine = load([out_path]) if os.path.exists(out_path) else None

    with open(out_path, "a") as out:
        out.write(f"// sievesearch k={kmin}..{kmax} N={a.N} subset={a.subset} runs={a.runs} radius={a.radius} "
                  f"alg={a.alg} starts={a.starts} moves={a.moves} seed={a.seed} {time.strftime('%Y-%m-%d %H:%M')}\n")
        # (k, run) -> bases, chosen up front so --prefetch can reduce them in parallel
        plan = [(k, run, choose(a.N, a.subset, a.primes, rng, a.keep_low, a.divisible_by, a.extra))
                for k in range(kmin, kmax + 1) for run in range(a.runs)]
        futures = {}
        if a.prefetch:
            from concurrent.futures import ProcessPoolExecutor
            pool = ProcessPoolExecutor(max_workers=a.prefetch)
            futures = {(k, run): pool.submit(reduced_basis, xs, k, a.block, a.basis_cache) for k, run, xs in plan}
        if a.bases_only:
            for (k, run), f in futures.items():
                f.result()
                print(f"basis k={k} run={run} done", flush=True)
            return 0
        todo = iter(plan)
        for k in range(kmin, kmax + 1):
            best: dict[int, tuple[int, Solution]] = {}
            for run in range(a.runs):
                _, _, xs = next(todo)
                R = futures[(k, run)].result() if futures else None
                fr, dbn, ts, td = search_one(xs, k, a, R)
                for m, (n, v) in fr.items():
                    if m > a.mmax:
                        continue
                    if not verify(xs, k, v):
                        raise AssertionError(f"non-relation k={k} m={m}")
                    s = Solution(k, {x: int(c) for x, c in zip(xs, v) if c}).normalized()
                    if a.primes and not s.is_prime():
                        continue
                    if m not in best or s.n < best[m][0]:
                        best[m] = (s.n, s)
                print(f"k={k} run {run + 1}/{a.runs}  bases {xs[0]}..{xs[-1]} ({len(xs)})  db {dbn}  "
                      f"sieve {ts:.1f}s  descent {td:.1f}s", flush=True)
            cells = []
            for m in sorted(best):
                n, s = best[m]
                r = cur.get(k, m)
                prev = mine.get(k, m) if mine else None
                if r is None or n <= r + a.slack:
                    tag = "" if r is None or n >= r else f" // NEW record (was {r})"
                    if prev is None or n < prev or tag:
                        out.write(s.format() + tag + "\n")
                if r is None:
                    cells.append(f"{m}:{n}+")
                elif n < r:
                    cells.append(f"{m}:{n}<{r}!")
                elif n == r:
                    cells.append(f"{m}:{n}=")
                else:
                    cells.append(f"{m}:{n}/{r}")
            out.flush()
            print(f"  k={k:3d}  " + "  ".join(c for c in cells[:20]), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
