#!/usr/bin/env python3
"""Congruence solutions for the completion method: bases t with sum +-t^k == 0 (mod M^k).

    python tools/congsolve.py -k 85 -M 16 -C 120 -o results/congsolve/k85-M16.txt

For a modulus M and the first C positive integers prime to M as candidate bases T, the
congruence solutions are the vectors of the lattice

    Lambda = { c in Z^T : sum_t c_t t^k == 0 (mod M^k) },

of rank C and determinant M^k.  A basis comes from the embedding
[I | lam (t^k mod M^k)] with the extra row [0 | lam M^k], after LLL.  The basis is reduced
with BKZ, sieved with G6K when C <= --sieve-max, and the weighted descent of
eslp/descent.py is run inside Lambda (moves: the database, or the BKZ rows) to find solutions
with a small side min(p, q).  Each solution found (the frontier for every m, and the shortest
database vectors) is written as an input line for tools/complete.py --signed:

    (k,a-b) (t1+...+ta-s1-...-sb)/M

with the a terms of the positive side first.  The completion then writes
(sum t^k - sum s^k)/M^k as a short combination of x^k, x <= N, and multiplies these bases by M.
Bases of Lambda are prime to M, so they never coincide with the bases M x of the completion.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np
from fpylll import LLL, IntegerMatrix

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eslp.descent import descend  # noqa: E402
from eslp.lattice import bkz  # noqa: E402


def candidates(M, C, start=1):
    out, t = [], start
    while len(out) < C:
        if math.gcd(t, M) == 1:
            out.append(t)
        t += 1
    return out


def strong_moments(k, N=100, threshold=-2.0):
    """(j, M_j(k)) for the moments that Proposition 8.1 makes nearly forced on L_{k,N}.

    Every relation of L_{k,N} satisfies sum c_x x^j == 0 mod M_j(k), the product of the primes p
    with (p-1) | (k-j).  When M_j(k) is not small compared with T N^j (T the number of terms of a
    short relation), a coset of L_{k,N} contains short vectors only if its j-th moment is close
    to a multiple of M_j(k).  These are the j with log10(M_j / (T N^j)) > threshold.
    """
    from tools.congruence import M as Mj, gh2
    T = math.sqrt(N * gh2(k, N))
    out = []
    for j in range(1, k):
        m = Mj(k, j)
        if m > 1 and math.log10(m / (T * N**j)) > threshold:
            out.append((j, m))
    return out


def congruence_basis(T, k, M, moments=(), lam=None):
    """Basis of {c : sum c_t t^k == 0 mod M^k L}, L the lcm of the m_j in moments.

    The completion writes D = -sum c_t t^k / M^k as sum_x d_x x^k.  For each prime p | m_j,
    sum d_x x^j == sum d_x x^k = D (mod p), so the j-th moment of every completion is D mod m_j,
    and the coset of completions contains short vectors only if D == 0 (mod m_j).  Requiring
    M^k L | sum c_t t^k makes D divisible by L.
    """
    L = 1
    for _, mj in moments:
        L = L * mj // math.gcd(L, mj)
    Q = M**k * L
    C = len(T)
    # the weight must exceed the length of the short congruence vectors, about Q^(1/C)
    lam = lam or max(10**6, 2 ** (Q.bit_length() // C + 20))
    A = IntegerMatrix(C + 1, C + 1)
    for i, t in enumerate(T):
        A[i, i] = 1
        A[i, C] = lam * (pow(t, k, Q))
    A[C, C] = lam * Q
    LLL.reduction(A)
    rows = [[A[i, j] for j in range(C)] for i in range(C + 1) if A[i, C] == 0]
    if len(rows) != C:
        raise RuntimeError(f"expected {C} congruence rows, got {len(rows)}")
    B = np.array(rows, dtype=np.int64)
    for r in B[:5]:
        assert sum(int(c) * pow(t, k, Q) for c, t in zip(r, T)) % Q == 0
    return B


def fmt(k, M, T, c):
    pos = sorted([t for t, v in zip(T, c) for _ in range(v) if v > 0], reverse=True)
    neg = sorted([t for t, v in zip(T, c) for _ in range(-v) if v < 0], reverse=True)
    if len(pos) > len(neg):
        pos, neg = neg, pos
    body = "+".join(map(str, pos)) + "".join(f"-{t}" for t in neg) if pos else \
        "-".join(map(str, neg))
    if not pos:     # all terms on one side: Wroblewski's one-sided format, read without --signed
        return f"({k},{len(neg)}) ({'+'.join(map(str, neg))})/{M}"
    return f"({k},{len(pos)}-{len(neg)}) ({body})/{M}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-k", type=int, required=True)
    ap.add_argument("-M", type=int, required=True)
    ap.add_argument("-C", type=int, required=True, help="number of candidate bases (the rank)")
    ap.add_argument("--start", type=int, default=1, help="smallest candidate base")
    ap.add_argument("--block", type=int, default=30)
    ap.add_argument("--sieve-max", type=int, default=100)
    ap.add_argument("--radius", type=float, default=1.4)
    ap.add_argument("--alg", default="bgj1")
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--shortest", type=int, default=200, help="also write this many shortest vectors")
    ap.add_argument("--mmax", type=int, default=16)
    ap.add_argument("--no-moments", action="store_true", help="omit the moment conditions of strong_moments()")
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()
    k, M, C = a.k, a.M, a.C
    t0 = time.time()
    T = candidates(M, C, a.start)
    moments = [] if a.no_moments else strong_moments(k)
    print(f"moment conditions: {[(j, mj) for j, mj in moments]}", flush=True)
    B = congruence_basis(T, k, M, moments)
    R = bkz(bkz(B, 20), a.block) if a.block > 20 else bkz(B, 20)
    d = C
    gh = math.sqrt(d / (2 * math.pi * math.e)) * (M ** (k / d))
    print(f"k={k} M={M} C={C}: bases {T[0]}..{T[-1]}, GH {gh:.1f}, "
          f"BKZ-{a.block} first row |c|^2 {int((R[0].astype(np.int64)**2).sum())}  {time.time() - t0:.0f}s", flush=True)
    moves = R
    if C <= a.sieve_max:
        from eslp.sieve import sieve_db
        t1 = time.time()
        kw = dict(threads=a.threads, alg=a.alg, saturation_radius=a.radius)
        if a.alg == "gpu":
            kw.update(gpu_bucketer="bdgl", saturation_ratio=0.375, db_size_factor=2.77,
                      gpu_params={"multi_bucket": 2, "max_nr_buckets": 0})
        moves = sieve_db(R, **kw)
        print(f"  sieve: {len(moves)} vectors  {time.time() - t1:.0f}s", flush=True)
    rng = np.random.default_rng(k * 1000 + M)
    starts = np.concatenate([R, moves[rng.permutation(len(moves))[:300]]])
    if a.alg == "gpu":
        from eslp.descent_gpu import descend_gpu
        fr = descend_gpu(starts, moves)
    else:
        fr = descend(starts, moves)
    sols = {}
    for m, (n, v) in sorted(fr.items()):
        if m <= a.mmax:
            sols[np.asarray(v).tobytes()] = np.asarray(v, dtype=np.int64)
    sq = (moves.astype(np.int64) ** 2).sum(1)
    for i in np.argsort(sq, kind="stable")[: a.shortest]:
        sols[moves[i].tobytes()] = moves[i].astype(np.int64)
    Q = M**k
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        f.write(f"// congsolve k={k} M={M} C={C} start={a.start} {time.strftime('%Y-%m-%d %H:%M')}\n")
        for v in sols.values():
            if not v.any():
                continue
            assert sum(int(c) * pow(t, k, Q) for c, t in zip(v, T)) % Q == 0
            f.write(fmt(k, M, T, v) + "\n")
    print(f"  frontier (m: n): " + " ".join(f"{m}:{n}" for m, (n, _) in sorted(fr.items()) if m <= a.mmax)
          + f";  {len(sols)} inputs written  {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
