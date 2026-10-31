#!/usr/bin/env python3
"""Structured congruence solutions, level by level in the p-adic valuation of the bases.

    python tools/structsolve.py -k 61 -p 2 -a 9 -o results/congsolve/struct-k61-M512.txt

To make S = sum t^k divisible by M^k, M = p^a, the bases are taken as t = p^v s with s prime to p
and 0 <= v < a.  Writing A_v = sum (over the bases of level v) of u_s s^k, we have
S = sum_v p^(kv) A_v, and S/M^k is the carry c_a of the recursion c_0 = 0,
c_(v+1) = (c_v + A_v)/p^k.  So level v only needs A_v == -c_v (mod p^k): a congruence in a lattice
of determinant p^k (instead of p^(ka) for all levels at once), which has far shorter solutions.
This is the structure of J. Wroblewski's congruence solutions of 2002.

At the last level the condition is taken modulo p^k M1', where M1' is the part prime to p of the
modulus L of the strong moments of tools/congsolve.strong_moments; then D = S/M^k == R (mod M1'),
so the completion over the bases M x lies in a coset with short vectors whose first moment is R
(a negative R makes the completion lean to the negative side; see tools/joint.py).

Each level: the lattice {u : sum u_s s^k == 0 mod Q} on the candidate units s of that level is
reduced and sieved; an affine start u0 (Babai from a one-term solution) is descended inside the
coset with the database as moves, and the vector with the fewest negative entries (then fewest
terms) is kept.  The output lines are inputs for tools/complete.py (--both).
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np
from fpylll import FPLLL, GSO, IntegerMatrix

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eslp.descent import descend  # noqa: E402
from eslp.lattice import bkz  # noqa: E402
from tools.congsolve import strong_moments  # noqa: E402


def units(p, lo, hi, C):
    """Up to C integers in [lo, hi] prime to p, the largest first."""
    out = [s for s in range(hi, lo - 1, -1) if s % p][:C]
    return sorted(out)


def affine_short(B, S, k, Q, tau):
    """A short u with sum u_i S_i^k == tau (mod Q): a one-term solution, Babai-reduced against B."""
    i1 = next(i for i, s in enumerate(S) if math.gcd(s, Q) == 1)
    u = np.zeros(len(S), dtype=object)
    u[i1] = (tau * pow(pow(S[i1], k, Q), -1, Q)) % Q
    BL = IntegerMatrix.from_matrix(B.tolist())
    old = FPLLL.get_precision()
    FPLLL.set_precision(max(53, int(u[i1]).bit_length() + 64))
    try:
        G = GSO.Mat(BL, float_type="mpfr")
        G.update_gso()
        for _ in range(3):
            coef = G.babai([int(v) for v in u])
            close = np.array(BL.multiply_left(coef), dtype=object)
            if not any(close):
                break
            u = u - close
    finally:
        FPLLL.set_precision(old)
    return np.array([int(v) for v in u], dtype=np.int64)


def solve_level(S, k, Q, tau, threads, radius, alg, rng):
    """A vector u with sum u_s s^k == tau (mod Q), few negative entries and few terms."""
    B = bkz(_level_basis(S, k, Q), 20)
    moves = B
    if len(S) >= 30:
        from eslp.sieve import sieve_db
        moves = sieve_db(B, threads=threads, alg=alg, saturation_radius=radius)
    sample = moves[rng.permutation(len(moves))[:120]].astype(np.int64)
    if tau % Q == 0:
        # no carry: any nonzero vector of the lattice will do (the zero vector is excluded below)
        u0 = None
        starts = np.concatenate([sample, -sample, B.astype(np.int64)])
    else:
        u0 = affine_short(B, S, k, Q, tau)
        starts = np.concatenate([u0[None, :], u0[None, :] + sample[:60], u0[None, :] - sample[60:]])
    fr = descend(starts, moves, all_tasks=True)
    cands = [np.asarray(v, dtype=np.int64) for _, _, v in fr] + ([u0] if u0 is not None else [])
    best = None
    for v in cands:
        if not v.any() or sum(int(c) * pow(s, k, Q) for c, s in zip(v, S)) % Q != tau % Q:
            continue
        key = int(np.abs(v).sum()) + 4 * int(-v[v < 0].sum())    # short, and on the positive side
        if best is None or key < best[0]:
            best = (key, v)
    return best[1]


def _level_basis(S, k, Q):
    """Basis of {u : sum u_s s^k == 0 mod Q} (the embedding of tools/congsolve, one modulus)."""
    from fpylll import LLL
    C = len(S)
    lam = max(10**6, 2 ** (Q.bit_length() // C + 20))
    A = IntegerMatrix(C + 1, C + 1)
    for i, s in enumerate(S):
        A[i, i] = 1
        A[i, C] = lam * pow(s, k, Q)
    A[C, C] = lam * Q
    LLL.reduction(A)
    rows = [[A[i, j] for j in range(C)] for i in range(C + 1) if A[i, C] == 0]
    if len(rows) != C:
        raise RuntimeError(f"expected {C} rows, got {len(rows)}")
    return np.array(rows, dtype=np.int64)


def one_try(args):
    """Solve all levels once (no moment conditions); return (positive bases, negative bases, D)."""
    k, p, A, C, bound, radius, alg, seed = args
    rng = np.random.default_rng(seed)
    c, pos, neg = 0, [], []
    for v in range(A):
        hi = max(bound // p**v, 2 * C + 2)
        allu = [s for s in range(1, hi + 1) if s % p]
        # a random choice of C units, biased to the large ones (they cost nothing extra)
        pick = rng.choice(len(allu), size=min(C, len(allu)), replace=False,
                          p=np.linspace(1, 3, len(allu)) / np.linspace(1, 3, len(allu)).sum())
        S = sorted(allu[i] for i in pick)
        Q = p**k
        u = solve_level(S, k, Q, (-c) % Q, 1, radius, alg, rng)
        Av = sum(int(x) * s**k for x, s in zip(u, S))
        c = (c + Av) // p**k
        for x, s in zip(u, S):
            (pos if x > 0 else neg).extend([p**v * s] * abs(int(x)))
    return pos, neg, c


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-k", type=int, required=True)
    ap.add_argument("-p", type=int, default=2)
    ap.add_argument("-a", type=int, required=True, help="M = p^a")
    ap.add_argument("-C", type=int, default=48, help="candidate units per level")
    ap.add_argument("--bound", type=int, default=None, help="largest base of level 0 (default 32 M)")
    ap.add_argument("--window", default=None,
                    help="lo,hi: keep solutions whose D, reduced mod M1', lies in [lo, hi] (default from the table)")
    ap.add_argument("--N", type=int, default=100, help="completion range, for the strong moments")
    ap.add_argument("--radius", type=float, default=1.4)
    ap.add_argument("--alg", default="bgj1")
    ap.add_argument("--threads", type=int, default=os.cpu_count(), help="parallel tries")
    ap.add_argument("--tries", type=int, default=400)
    ap.add_argument("--keep", type=int, default=20, help="stop after this many solutions in the window")
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()
    k, p, A = a.k, a.p, a.a
    M = p**A
    bound = a.bound or 32 * M
    L = 1
    for _, mj in strong_moments(k, a.N):
        L = L * mj // math.gcd(L, mj)
    M1 = L
    while M1 % p == 0:
        M1 //= p
    if a.window:
        lo, hi = (int(x) for x in a.window.split(","))
    else:
        from tools.compare import load_current
        n1 = load_current("normal").get(k, 1) or 1000
        lo, hi = -(a.N // 2) * n1, -(a.N // 2) * n1 // 8      # completions of n1/8 to n1 terms
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    out = open(a.out, "a")
    out.write(f"// structsolve k={k} M={p}^{A} C={a.C} bound={bound} M1'={M1} window=[{lo},{hi}] "
              f"{time.strftime('%Y-%m-%d %H:%M')}\n")
    import multiprocessing
    t0, kept, done = time.time(), 0, 0
    jobs = [(k, p, A, a.C, bound, a.radius, a.alg, 1000003 * k + 7919 * A + i) for i in range(a.tries)]
    with multiprocessing.get_context("spawn").Pool(a.threads) as P:
        for pos, neg, D in P.imap_unordered(one_try, jobs):
            done += 1
            Sfull = sum(t**k for t in pos) - sum(t**k for t in neg)
            assert Sfull == D * M**k
            r = D % M1 if M1 > 1 else 0
            if r > M1 // 2:
                r -= M1
            # This tests D mod M1', as in the runs of the paper.  The completion's j-th moment is in
            # fact determined modulo g = g_j(k,N), a multiple of 2*M1' (g = 2*M1' for k = 85, 97 and
            # 72*M1' for k = 75), so the test also keeps solutions whose moments are r + M1' (mod 2*M1'),
            # none of them in the window.  The window only constrains r when M1' exceeds its width.
            if M1 > hi - lo and not (lo <= r <= hi):
                continue
            pos.sort(reverse=True)
            neg.sort(reverse=True)
            if neg:
                line = f"({k},{len(pos)}-{len(neg)}) ({'+'.join(map(str, pos))}{''.join(f'-{t}' for t in neg)})/{M}"
            else:
                line = f"({k},{len(pos)}) ({'+'.join(map(str, pos))})/{M}"
            out.write(line + "\n")
            out.flush()
            kept += 1
            print(f"try {done}: {len(pos)} positive, {len(neg)} negative bases, D mod M1' = {r}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
            if kept >= a.keep:
                P.terminate()
                break
    print(f"{kept} solutions in the window from {done} tries ({time.time() - t0:.0f}s)", flush=True)
    out.close()


if __name__ == "__main__":
    main()
