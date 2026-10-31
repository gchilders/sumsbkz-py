"""Relation lattices for equal sums of like powers, reduced with fplll.

For bases x_1..x_n and power k, the relation lattice is

    L = { c in Z^n : sum c_i x_i^k = 0 },   rank n-1.

The original sumsbkz built the embedding [I | W*x^k] and ran NTL's BKZ_FP on it;
rows whose last coordinate came out 0 were relations.  Here we LLL the embedding
once to get an exact basis of L (the last column is huge, so LLL zeroes it on
n-1 rows), then do all further work -- BKZ, sieving, enumeration -- in L itself,
where the entries are small.
"""

from __future__ import annotations

import random

import numpy as np
from fpylll import BKZ, LLL, GSO, IntegerMatrix
from fpylll.fplll.bkz_param import BKZParam, load_strategies_json


def _strategies():
    """fplll's default BKZ 2.0 pruning strategies.

    The fpylll macOS wheel points BKZ.DEFAULT_STRATEGY at its CI build directory,
    so fall back to a system fplll install (Homebrew, /usr) when that path is missing.
    """
    import os
    candidates = [os.environ.get("FPLLL_STRATEGIES", ""),
                  BKZ.DEFAULT_STRATEGY.decode() if isinstance(BKZ.DEFAULT_STRATEGY, bytes) else BKZ.DEFAULT_STRATEGY,
                  os.path.join(os.path.dirname(__file__), "..", "data", "fplll", "default.json"),
                  "/opt/homebrew/share/fplll/strategies/default.json",
                  "/usr/local/share/fplll/strategies/default.json",
                  "/usr/share/fplll/strategies/default.json"]
    for c in candidates:
        if c and os.path.exists(c):
            return load_strategies_json(c.encode())
    raise FileNotFoundError("fplll strategies/default.json not found; set FPLLL_STRATEGIES")


_STRATEGIES = None


def first_primes(count: int, skip: int = 0) -> list[int]:
    out, c = [], 2
    while len(out) < count + skip:
        if all(c % p for p in out if p * p <= c):
            out.append(c)
        c += 1
    return out[skip:]


def choose_bases(n: int, rng: random.Random, primes: bool = False,
                 workterms: int = 0, includelow: bool = False) -> list[int]:
    """Bases in the order they become lattice rows, following sumsbkz's options.

    Without workterms the set is 1..n (or the first n primes) and only the row
    order is random.  With workterms, a random subset of that size is used and
    includelow forces the five smallest candidates into it.
    """
    pool = first_primes(n) if primes else list(range(1, n + 1))
    if includelow:
        low, rest = pool[:5], pool[5:]
        rng.shuffle(rest)
        size = workterms or n
        xs = rest[: size - 5] + low
    else:
        xs = pool[:]
        rng.shuffle(xs)
        if workterms:
            xs = xs[:workterms]
    return xs


def relation_basis(xs: list[int], k: int, weight: int = 10000) -> np.ndarray:
    """LLL-reduced basis of the relation lattice, as an (n-1) x n int64 array."""
    n = len(xs)
    A = IntegerMatrix(n, n + 1)
    for i, x in enumerate(xs):
        A[i, i] = 1
        A[i, n] = weight * x**k
    LLL.reduction(A)
    rows = [[A[i, j] for j in range(n)] for i in range(n) if A[i, n] == 0]
    if len(rows) != n - 1:
        raise RuntimeError(f"expected {n-1} relations, got {len(rows)}")
    return np.array(rows, dtype=np.int64)


def bkz(B: np.ndarray, block_size: int, max_loops: int = 8,
        auto_abort: bool = True, seed: int | None = None) -> np.ndarray:
    """fplll BKZ 2.0 (default pruning strategies) on the rows of B."""
    A = IntegerMatrix.from_matrix(B.tolist())
    if block_size >= 2:
        flags = BKZ.MAX_LOOPS | (BKZ.AUTO_ABORT if auto_abort else 0)
        global _STRATEGIES
        if _STRATEGIES is None:
            _STRATEGIES = _strategies()
        par = BKZParam(block_size=min(block_size, A.nrows), strategies=_STRATEGIES,
                       max_loops=max_loops, flags=flags)
        if seed is not None:
            from fpylll import FPLLL
            FPLLL.set_random_seed(seed)
        BKZ.reduction(A, par)
    return np.array([[A[i, j] for j in range(A.ncols)] for i in range(A.nrows)], dtype=np.int64)


def randomize(B: np.ndarray, rng: random.Random, density: int = 3) -> np.ndarray:
    """Apply a random sparse unimodular transform (rerandomization between BKZ tours)."""
    B = B.copy()
    r = B.shape[0]
    order = list(range(r))
    rng.shuffle(order)
    B = B[order]
    for i in range(r):
        for _ in range(density):
            j = rng.randrange(r)
            if j != i:
                B[i] += rng.choice((-1, 1)) * B[j]
    return B


def verify(xs: list[int], k: int, c: np.ndarray) -> bool:
    return sum(int(ci) * x**k for x, ci in zip(xs, c) if ci) == 0
