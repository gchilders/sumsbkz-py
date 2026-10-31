#!/usr/bin/env python3
"""Combine known solutions across runs: NewLC/ResultLC with the compiled descent.

    python tools/combine.py FILES... -k 19 30 [--maxbase 100] [-o results/combine_normal.txt]

For each k, every verified solution in FILES with all bases <= maxbase becomes a
vector in Z^maxbase (one sign, duplicates removed).  combine_pool() then runs the
descent (eslp/descent.py) with the pool as both starts and moves, under several
orderings of the pool (the descent breaks ties by position), and adds to the pool up
to `ties` relations attaining the current frontier value for each m <= mmax.  It
stops when a round changes none of these sets.  Results that tie
or beat the euler.free.fr table are written out.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eslp.descent import descend  # noqa: E402
from eslp.solutions import Solution, read_solutions  # noqa: E402
from tools.compare import load_current  # noqa: E402


def canon(v: np.ndarray) -> np.ndarray:
    """Sign-normalize: first nonzero entry positive."""
    nz = np.flatnonzero(v)
    return -v if v[nz[0]] < 0 else v


def mn(v):
    p, q = int(v[v > 0].sum()), int(-v[v < 0].sum())
    return min(p, q), max(p, q)


def combine_pool(V: np.ndarray, mmax: int = 40, orderings: int = 4, ties: int = 64, seed: int = 0,
                 max_rounds: int = 50, log=None):
    """Combine a pool of relations to a fixed point.  Returns ({m: (n, vector)}, info).

    The pool is the initial set P0 together with, for each m <= mmax, a set S_m of at most
    `ties` distinct relations attaining the current frontier value F(m).  Each round runs
    the descent with the pool as starts and moves, once for each of `orderings` orderings
    of the pool (the first is the given order, the others random), lowers F(m) where a
    better relation was found (emptying S_m), and adds tied relations to S_m until it is
    full.  The procedure stops after a round in which no S_m changes.
    """
    rng = np.random.default_rng(seed)
    P0 = {}
    for v in V:
        v = canon(np.asarray(v, dtype=np.int32))
        P0[v.tobytes()] = v
    F, S = {}, {}

    def offer(m, n, v):
        """Update F and S with relation v of type (m, n); return True if S changed."""
        if m > mmax:
            return False
        if m not in F or n < F[m]:
            F[m], S[m] = n, {}
        if n == F[m] and len(S[m]) < ties:
            v = canon(v)
            key = v.tobytes()
            if key not in S[m]:
                S[m][key] = v
                return True
        return False

    for v in P0.values():
        offer(*mn(v), v)
    rounds, changed = 0, True
    while changed and rounds < max_rounds:
        rounds += 1
        pool = dict(P0)
        for Sm in S.values():
            pool.update(Sm)
        P = np.array(list(pool.values()))
        changed = False
        for o in range(orderings):
            Q = P if o == 0 else P[rng.permutation(len(P))]
            for m, n, v in sorted(descend(Q, Q, all_tasks=True), key=lambda t: (t[0], t[1])):
                changed |= offer(m, n, v)
        if log:
            log(f"    round {rounds}: pool {len(P)}, frontier sum "
                f"{sum(n for m, n in F.items() if m <= 16)}")
    best = {m: (n, next(iter(S[m].values()))) for m, n in F.items()}
    return best, {"rounds": rounds, "pool": len(P), "capped": changed}


def load_pool(sols, k, maxbase, primes=False):
    vecs = {}
    for s in sols:
        if s.k != k or max(s.coeffs) > maxbase or (primes and not s.is_prime()):
            continue
        if not s.is_valid():
            continue
        v = np.zeros(maxbase, dtype=np.int32)
        for b, c in s.normalized().coeffs.items():
            v[b - 1] = c
        v = canon(v)
        vecs[v.tobytes()] = v
    return np.array(list(vecs.values())) if vecs else np.zeros((0, maxbase), dtype=np.int32)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("-k", type=int, nargs=2, required=True)
    ap.add_argument("--maxbase", type=int, default=100)
    ap.add_argument("--primes", action="store_true")
    ap.add_argument("--mmax", type=int, default=40)
    ap.add_argument("--orderings", type=int, default=4)
    ap.add_argument("--ties", type=int, default=64)
    ap.add_argument("-o", "--out", default=None)
    a = ap.parse_args()
    kind = "prime" if a.primes else "normal"
    cur = load_current(kind)
    sols = [s for p in a.files for s in read_solutions(p)]
    out = open(a.out, "a") if a.out else None

    for k in range(a.k[0], a.k[1] + 1):
        V = load_pool(sols, k, a.maxbase, a.primes)
        if len(V) < 2:
            continue
        fr, info = combine_pool(V, a.mmax, a.orderings, a.ties)
        cells = []
        for m, (n, v) in sorted(fr.items()):
            s = Solution(k, {i + 1: int(c) for i, c in enumerate(v) if c}).normalized()
            assert s.is_valid() and (s.m, s.n) == (m, n)
            r = cur.get(k, m)
            if r is not None and n <= r:
                cells.append(f"{m}:{n}{'<%d!' % r if n < r else '='}")
                if out and n < r:
                    out.write(f"{s.format()} // NEW record (was {r}) [combine]\n")
        print(f"k={k:3d}  {len(V):5d} solutions  {info['rounds']} rounds  pool {info['pool']}  "
              + "  ".join(cells), flush=True)
    if out:
        out.close()


if __name__ == "__main__":
    main()
