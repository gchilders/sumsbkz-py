"""Sieve the relation lattice with G6K and harvest every database vector as a candidate.

BKZ hands back n-1 short vectors; a sieve saturates a ball of radius
sqrt(saturation_radius) * GH, which for these dimensions (40-100) holds tens of
thousands of short relations.  Records are judged by term counts rather than the
Euclidean norm, but a solution with T terms, mostly of multiplicity 1, has
||c||^2 close to T, and the record solutions sit near the Gaussian heuristic, so
a saturated database contains most of them, or vectors one greedy step away.
"""

from __future__ import annotations

import numpy as np
from fpylll import IntegerMatrix

from eslp.greedy import one_side_len, side_counts, total_len

# statistics of the last sieve_db call: number of sieve calls and how many of them
# stopped without reaching the saturation target
LAST_STATS = {"sieves": 0, "unsaturated": 0}


def sieve_db(R: np.ndarray, threads: int = 4, alg: str = "bgj1",
             saturation_radius: float = 4 / 3, saturation_ratio: float = 0.5,
             db_size_factor: float = 3.2, gpu_bucketer: str = "bgj1",
             db_cap: int = 0, chunk: int = 1 << 20, start_dim: int = 50, gpus: int = 1,
             gpu_params: dict | None = None) -> np.ndarray:
    """Run a full sieve on the lattice spanned by the rows of R; return the database as vectors.

    The database is returned in the ambient coordinates (one relation per row,
    sign-normalized so the first nonzero entry is positive, duplicates removed).
    """
    from g6k import Siever, SieverParams

    d = R.shape[0]
    A = IntegerMatrix.from_matrix(R.tolist())
    params = SieverParams(threads=threads, default_sieve=alg,
                          saturation_radius=saturation_radius,
                          saturation_ratio=saturation_ratio,
                          db_size_base=saturation_radius ** 0.5,
                          db_size_factor=db_size_factor)
    if alg == "gpu":            # G6K-GPU-Tensor build only
        params.gpus = gpus
        params.gpu_bucketer = gpu_bucketer
        for key, val in (gpu_params or {}).items():   # e.g. multi_bucket, max_nr_buckets
            setattr(params, key, val)
    g = Siever(A, params)
    # Progressive sieving ("pump up"): sieve the last start_dim coordinates of the
    # projected lattice, then extend left one dimension at a time, re-sieving from the
    # current database.  Far cheaper than filling a full-dimensional database with
    # random samples and sieving it from scratch.
    from g6k.siever import SaturationError

    LAST_STATS.update(sieves=0, unsaturated=0, full_unsaturated=False)

    def run():
        # as in G6K's pump: an unsaturated database is still a good database
        LAST_STATS["sieves"] += 1
        try:
            g(alg=alg)
            return True
        except SaturationError:
            LAST_STATS["unsaturated"] += 1
            return False

    g.initialize_local(0, max(0, d - start_dim), d)
    ok = run()
    while g.l > 0:
        g.extend_left(1)
        ok = run()
    LAST_STATS["full_unsaturated"] = not ok
    if db_cap and len(g) > db_cap:
        g.shrink_db(db_cap)     # drops the longest vectors first
    # G6K may have size-reduced/LLL'd its basis; map through its own B, not R.
    B = np.array([[g.M.B[i, j] for j in range(g.M.B.ncols)] for i in range(g.M.B.nrows)],
                 dtype=np.int64)
    # extract in chunks as int16 so a 10^7-vector database fits in memory
    from itertools import islice
    it = g.itervalues()
    parts = []
    while True:
        X = np.array(list(islice(it, chunk)), dtype=np.int64)
        if len(X) == 0:
            break
        V = X @ B
        if np.abs(V).max() > 32000:
            raise OverflowError("relation coefficient exceeds int16")
        parts.append(V.astype(np.int16))
    return canonical(np.concatenate(parts))


def canonical(V: np.ndarray) -> np.ndarray:
    """Drop zero rows, fix sign (first nonzero entry > 0), remove duplicates."""
    V = V[np.any(V != 0, axis=1)]
    first = np.argmax(V != 0, axis=1)
    sgn = np.sign(V[np.arange(len(V)), first])
    V = V * sgn[:, None]
    return np.unique(V, axis=0)


def frontier_of(V: np.ndarray) -> dict[int, tuple[int, int]]:
    """Best n per m among rows of V: {m: (n, row index)}."""
    pos, neg = side_counts(V)
    m = np.minimum(pos, neg)
    n = np.maximum(pos, neg)
    best: dict[int, tuple[int, int]] = {}
    order = np.lexsort((n, m))
    for i in order:
        mi, ni = int(m[i]), int(n[i])
        if mi not in best:
            best[mi] = (ni, int(i))
    return best


def greedy_against(v: np.ndarray, D: np.ndarray, measure, max_steps: int = 200) -> np.ndarray:
    """Vectorized Yura step: repeatedly replace v by the best v +/- D[j] while measure drops."""
    cur = measure(v)
    for _ in range(max_steps):
        cand = np.concatenate([v + D, v - D])
        mc = measure(cand)
        mc = np.where(np.any(cand != 0, axis=1), mc, np.iinfo(np.int64).max)
        j = int(np.argmin(mc))
        if mc[j] >= cur:
            break
        v, cur = cand[j], mc[j]
    return v


def lex_measure(weight: int):
    """m first, then total: weight*m + total (weight large => strict m priority)."""
    def f(V):
        return weight * one_side_len(V) + total_len(V)
    return f


class FrontierTracker:
    """Best (n, vector) per m over every relation seen."""

    def __init__(self) -> None:
        self.best: dict[int, tuple[int, np.ndarray]] = {}

    def update(self, V: np.ndarray) -> None:
        if len(V) == 0:
            return
        pos, neg = side_counts(V)
        m, n = np.minimum(pos, neg), np.maximum(pos, neg)
        for mi in np.unique(m):
            idx = np.nonzero(m == mi)[0]
            j = idx[np.argmin(n[idx])]
            cur = self.best.get(int(mi))
            if cur is None or n[j] < cur[0]:
                self.best[int(mi)] = (int(n[j]), V[j].copy())

    def table(self) -> dict[int, int]:
        return {m: v[0] for m, v in sorted(self.best.items())}


def greedy_sweep(starts: np.ndarray, D: np.ndarray, weights=(0, 1, 2, 4, 8, 16, 1000),
                 tracker: FrontierTracker | None = None, chunk: int = 16,
                 max_steps: int = 500) -> FrontierTracker:
    """ResultLC-style descent with the sieve database as the move set.

    For each weight w, every start vector is walked downhill in w*m + total by
    steepest +/- D[j] moves; every vector on every path is offered to the tracker.
    """
    tracker = tracker or FrontierTracker()
    D = D.astype(np.int32)
    M = np.concatenate([D, -D])                        # (2|D|, N) moves
    tracker.update(starts)
    for w in weights:
        for s0 in range(0, len(starts), chunk):
            V = starts[s0:s0 + chunk].astype(np.int32)
            pos, neg = side_counts(V)
            cur = w * np.minimum(pos, neg) + pos + neg
            active = np.ones(len(V), bool)
            for _ in range(max_steps):
                if not active.any():
                    break
                ia = np.nonzero(active)[0]
                C = V[ia, None, :] + M[None, :, :]       # (a, 2|D|, N)
                p = np.where(C > 0, C, 0).sum(-1)
                q = np.where(C < 0, -C, 0).sum(-1)
                f = w * np.minimum(p, q) + p + q
                f = np.where((p + q) > 0, f, np.iinfo(np.int32).max)
                j = np.argmin(f, axis=1)
                fj = f[np.arange(len(ia)), j]
                better = fj < cur[ia]
                if not better.any():
                    break
                ib = ia[better]
                V[ib] = C[np.nonzero(better)[0], j[better]]
                cur[ib] = fj[better]
                active[ia[~better]] = False
                tracker.update(V[ib])
    return tracker


def sieve_db_projected(R: np.ndarray, free: int, threads: int = 4, alg: str = "bgj1",
                       saturation_radius: float = 4 / 3, saturation_ratio: float = 0.5,
                       db_size_factor: float = 3.2) -> np.ndarray:
    """Sieve only the projected sublattice [free, d) and Babai-lift the database.

    Dimensions for free: on a BKZ-reduced basis the first `free` Gram-Schmidt
    vectors are short, so a vector that is short in the projection lifts to a
    reasonably short vector of the full lattice.  Cost follows d - free, not d.
    """
    from g6k import Siever, SieverParams

    d = R.shape[0]
    A = IntegerMatrix.from_matrix(R.tolist())
    params = SieverParams(threads=threads, default_sieve=alg,
                          saturation_radius=saturation_radius,
                          saturation_ratio=saturation_ratio,
                          db_size_base=saturation_radius ** 0.5,
                          db_size_factor=db_size_factor)
    g = Siever(A, params)
    g.initialize_local(0, free, d)
    g(alg=alg)
    X = np.array(list(g.itervalues()), dtype=np.int64)          # coords on rows free..d-1
    B = np.array([[g.M.B[i, j] for j in range(g.M.B.ncols)] for i in range(g.M.B.nrows)],
                 dtype=np.int64)
    Y = X @ B[free:]
    if free:
        # Babai nearest plane against rows 0..free-1 (float GSO of the prefix)
        P = B[:free].astype(np.float64)
        Q = np.zeros_like(P)
        for i in range(free):
            v = P[i].copy()
            for j in range(i):
                v -= (P[i] @ Q[j]) / (Q[j] @ Q[j]) * Q[j]
            Q[i] = v
        Yf = Y.astype(np.float64)
        for i in range(free - 1, -1, -1):
            c = np.rint((Yf @ Q[i]) / (Q[i] @ Q[i]))
            Y -= c.astype(np.int64)[:, None] * B[i]
            Yf -= c[:, None] * B[i].astype(np.float64)
    return canonical(Y)
