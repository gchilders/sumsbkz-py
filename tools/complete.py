#!/usr/bin/env python3
"""Complete congruence solutions to identities: the 2002 method with a sieve move set.

    python tools/complete.py FILES... -k 61 [-N 100] [--alg gpu] -o results/complete/k61.txt

Each input line "(k,c) (t1+...+tc)/M" (J. Wroblewski's format) gives bases t_i with
S = sum t_i^k divisible by M^k.  As in decompose.cpp (2002), M is first extended to the
largest product of integers i < 500 with i^k | S, and D = S / M^k.  Writing
D = sum_{x<=N} c_x x^k gives the identity

    sum_{c_x>0} c_x (Mx)^k = sum_i t_i^k + sum_{c_x<0} |c_x| (Mx)^k.

A particular solution c0 comes from the embedding basis [I | 0 | lam x^k ; 0 | 1 | -lam D]
(Kannan's embedding, as in decompose.cpp) after LLL and BKZ-20.  The descent then runs on
c with the sieve database of L_{k,N} as moves; the t_i are fixed terms on the negative
side (q_off = their number).  Bases t_i that equal M*y with y <= N are absorbed into c.
Every identity found is assembled, verified in exact arithmetic and written out.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
import time

import numpy as np
from fpylll import BKZ, LLL, IntegerMatrix

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eslp.solutions import Solution  # noqa: E402
from tools.compare import load_current  # noqa: E402

LINE = re.compile(r"^\((\d+),(\d+)\)\s*\(([\d+]+)\)/(\d+)")
# signed inputs: "(19,1-3) (57-37-31-21)/2" (signs written out) and "(31,1+5) (203+127+...)/2"
# (the first 1 term positive, the other 5 negative, all written with '+')
SIGNED = re.compile(r"^\((\d+),(\d+)([+-])(\d+)\)\s*\(([\d+-]+)\)/(\d+)")


def read_inputs(paths, k, signed=False):
    """Distinct inputs (positive terms, negative terms, M) for power k from Wroblewski's files.

    By default only the one-sided lines "(k,c) (t1+...+tc)/M" are read (all terms positive);
    with signed=True only the signed lines are read, so the two input sets, and the order
    of each (used by --shard), do not depend on each other.
    """
    seen, out = set(), []
    for p in paths:
        for line in open(p, errors="replace"):
            line = line.strip()
            if not signed:
                mt = LINE.match(line)
                if not mt or int(mt.group(1)) != k:
                    continue
                terms = tuple(sorted((int(t) for t in mt.group(3).split("+")), reverse=True))
                if len(terms) != int(mt.group(2)):
                    continue
                key = (terms, ())
                M0 = int(mt.group(4))
            else:
                mt = SIGNED.match(line)
                if not mt or int(mt.group(1)) != k:
                    continue
                na, op, nb = int(mt.group(2)), mt.group(3), int(mt.group(4))
                toks = re.findall(r"([+-]?)(\d+)", mt.group(5))
                if len(toks) != na + nb:
                    continue
                if op == "-":       # signs written out; the first term is positive
                    pos = [int(t) for i, (sg, t) in enumerate(toks) if sg != "-"]
                    neg = [int(t) for sg, t in toks if sg == "-"]
                    if len(pos) != na:
                        continue
                else:               # first na terms positive, the rest negative
                    pos = [int(t) for _, t in toks[:na]]
                    neg = [int(t) for _, t in toks[na:]]
                key = (tuple(sorted(pos, reverse=True)), tuple(sorted(neg, reverse=True)))
                M0 = int(mt.group(6))
            if key in seen:
                continue
            seen.add(key)
            out.append((key[0], key[1], M0))
    return out


def modulus(S, k, M0):
    """Largest M (a multiple of M0, extended by factors i < 500 as decompose.cpp did) with M^k | S."""
    if S % M0**k:
        return None
    M, rest = M0, S // M0**k
    i = 2
    while i < 500:
        if rest % i**k == 0:
            M *= i
            rest //= i**k
        else:
            i += 1
    return M


def particular(D, k, N, lam=None):
    """Short solutions c of sum_{x<=N} c_x x^k = D from the reduced embedding basis.

    Rows of the reduced basis with last coordinate 0 are vectors (c, e) with
    sum c_x x^k = e D.  Rows with e = +-1 give solutions directly.  If there are none, an
    integer combination of rows with e = 1 is formed from the extended gcd of their e's,
    and shortened by nearest-plane (Babai) reduction against the rows with e = 0.
    """
    from fpylll import GSO
    lam = lam or 10**6
    A = IntegerMatrix(N + 1, N + 2)
    for x in range(1, N + 1):
        A[x - 1, x - 1] = 1
        A[x - 1, N + 1] = lam * x**k
    A[N, N] = 1
    A[N, N + 1] = -lam * D
    LLL.reduction(A)
    BKZ.reduction(A, BKZ.Param(block_size=20, max_loops=4, flags=BKZ.MAX_LOOPS))
    rows = [[A[i, j] for j in range(N + 1)] for i in range(A.nrows) if A[i, N + 1] == 0]
    sols = [np.array([r[N] * r[j] for j in range(N)], dtype=object) for r in rows if abs(r[N]) == 1]
    if not sols:
        # extended gcd over the e-coordinates of the rows with e != 0
        comb, g = None, 0
        for r in rows:
            e = r[N]
            if e == 0:
                continue
            if comb is None:
                comb, g = np.array(r, dtype=object), e
                continue
            # find a, b with a*g + b*e = gcd(g, e)
            x0, x1, y0, y1, aa, bb = 1, 0, 0, 1, g, e
            while bb:
                q = aa // bb
                aa, bb = bb, aa - q * bb
                x0, x1 = x1, x0 - q * x1
                y0, y1 = y1, y0 - q * y1
            comb = x0 * comb + y0 * np.array(r, dtype=object)
            g = aa
            if abs(g) == 1:
                break
        rel = [r[:N] for r in rows if r[N] == 0]
        if comb is not None and abs(g) == 1 and len(rel) == N - 1:
            c = comb[:N] * g          # now sum c_x x^k = D
        else:
            # no usable rows: D e_1 is a solution (1^k = 1), shortened against L_{k,N} below
            from eslp.lattice import bkz, relation_basis
            c = np.zeros(N, dtype=object)
            c[0] = D
            rel = bkz(relation_basis(list(range(1, N + 1)), k), 20).tolist()
        from fpylll import FPLLL
        B = IntegerMatrix.from_matrix(rel)
        old = FPLLL.get_precision()
        FPLLL.set_precision(max(53, int(max(abs(int(v)) for v in c)).bit_length() + 64))
        try:
            G = GSO.Mat(B, float_type="mpfr")
            G.update_gso()
            for _ in range(3):            # nearest plane, repeated while it still shortens c
                coef = G.babai([int(v) for v in c])
                close = np.array(B.multiply_left(coef), dtype=object)
                if not any(close):
                    break
                c = c - close
        finally:
            FPLLL.set_precision(old)
        sols = [c]
    out = []
    for c in sols:
        assert sum(int(c[x - 1]) * x**k for x in range(1, N + 1)) == D
        if max(abs(int(v)) for v in c) < 2**31:
            out.append(np.array([int(v) for v in c], dtype=np.int64))
    return out


def _claim(path):
    """Create path exclusively; True if this process created it (and so should build the file)."""
    try:
        os.close(os.open(path, os.O_CREAT | os.O_EXCL))
        return True
    except FileExistsError:
        return False


def _pool_worker(args):
    """Rows of one BKZ reduction of L_{k,N} with the bases in a random order, in the order 1..N."""
    from eslp.lattice import bkz, relation_basis
    k, N, block, seed = args
    rng = np.random.default_rng(seed)
    perm = rng.permutation(N) + 1
    R = bkz(bkz(relation_basis([int(x) for x in perm], k), 20, seed=seed), block, seed=seed)
    out = np.zeros_like(R)
    out[:, perm - 1] = R
    if np.abs(out).max() > 32000:
        out = out[np.abs(out).max(axis=1) <= 32000]
    return out


def assemble(k, M, c, pos_fixed, neg_fixed=()):
    """The identity as a Solution: c on the bases M*x, the fixed bases with their signs."""
    coeffs = {}
    for x, cx in enumerate(c, 1):
        if cx:
            coeffs[M * x] = coeffs.get(M * x, 0) + int(cx)
    for t in pos_fixed:
        coeffs[t] = coeffs.get(t, 0) - 1
    for t in neg_fixed:
        coeffs[t] = coeffs.get(t, 0) + 1
    coeffs = {b: v for b, v in coeffs.items() if v}
    return Solution(k, coeffs).normalized()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("-k", required=True,
                    help="the power; a trailing 's' (e.g. 19s) is the same as --signed, so that signed "
                         "tasks can share a task list with one-sided ones")
    ap.add_argument("-N", type=int, default=100)
    ap.add_argument("--radius", type=float, default=4 / 3)
    ap.add_argument("--alg", default="gpu")
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--starts", type=int, default=100, help="random database shifts of each particular solution")
    ap.add_argument("--basis-cache", default="bases")
    ap.add_argument("--db-npy", default=None, help="load/save the sieve database here")
    ap.add_argument("--pad-db", default=None,
                    help="instead of sieving L_{k,N}: use this database of L_{k,N0}, N0 < N, padded with zeros")
    ap.add_argument("--bkz-pool", type=int, default=0,
                    help="add the rows of this many BKZ reductions of L_{k,N} in random orders to the moves")
    ap.add_argument("--bkz-block", type=int, default=30)
    ap.add_argument("--mmax", type=int, default=40)
    ap.add_argument("--limit", type=int, default=0, help="use only the first this many inputs (testing)")
    ap.add_argument("--seed", type=int, default=0, help="varies the random database shifts of the starts")
    ap.add_argument("--signed", action="store_true", help="use the signed inputs (one or more left terms)")
    ap.add_argument("--both", action="store_true", help="use the one-sided and the signed inputs")
    ap.add_argument("--shard", type=int, nargs=2, default=None, metavar=("I", "S"),
                    help="process only inputs i with i %% S == I (inputs are in a fixed order)")
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()
    if a.k.endswith("s"):
        a.signed, a.k = True, a.k[:-1]
        if a.db_npy:            # the move set depends only on k and N
            a.db_npy = a.db_npy.replace(f"db-k{a.k}s-", f"db-k{a.k}-")
    a.k = int(a.k)
    k, N = a.k, a.N
    cur = load_current("normal")
    inputs = (read_inputs(a.files, k) + read_inputs(a.files, k, True)) if a.both else read_inputs(a.files, k, a.signed)
    if a.limit:
        inputs = inputs[: a.limit]
    if a.shard:
        inputs = inputs[a.shard[0]:: a.shard[1]]
    print(f"k={k}: {len(inputs)} distinct congruence inputs", flush=True)
    if not inputs:
        return

    # the move set: the sieve database of L_{k,N}
    t0 = time.time()
    if a.pad_db:
        D0 = np.load(a.pad_db)
        D = np.zeros((len(D0), N), dtype=D0.dtype)
        D[:, : D0.shape[1]] = D0
    elif a.db_npy and (os.path.exists(a.db_npy) or not _claim(a.db_npy + ".lock")):
        while not os.path.exists(a.db_npy):     # another process is sieving this lattice
            time.sleep(30)
        time.sleep(5)
        D = np.load(a.db_npy)
    else:
        from eslp.sieve import sieve_db
        from sievesearch import reduced_basis
        R = reduced_basis(list(range(1, N + 1)), k, 20, a.basis_cache)
        kw = dict(threads=a.threads, alg=a.alg, saturation_radius=a.radius)
        if a.alg == "gpu":
            kw.update(gpu_bucketer="bdgl", saturation_ratio=0.375, db_size_factor=2.77,
                      gpu_params={"multi_bucket": 2, "max_nr_buckets": 0})
        D = sieve_db(R, **kw)
        if a.db_npy:
            np.save(a.db_npy + ".tmp.npy", D)
            os.replace(a.db_npy + ".tmp.npy", a.db_npy)
    if a.bkz_pool:
        import multiprocessing
        from eslp.sieve import canonical
        t2 = time.time()
        with multiprocessing.get_context("spawn").Pool(a.threads) as P:
            parts = P.map(_pool_worker, [(k, N, a.bkz_block, 1000 * k + i) for i in range(a.bkz_pool)])
        D = canonical(np.concatenate([D] + [p.astype(D.dtype) for p in parts]))
        print(f"BKZ-{a.bkz_block} pool: {a.bkz_pool} reductions, {time.time() - t2:.0f}s", flush=True)
    print(f"database {len(D)} vectors, {time.time() - t0:.0f}s", flush=True)
    if a.alg == "gpu":
        from eslp.descent_gpu import descend_gpu as run_descent
    else:
        from eslp.descent import descend as run_descent

    rng = np.random.default_rng(k + 1000 * a.seed)
    best = {}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    out = open(a.out, "a")
    out.write(f"// complete k={k} N={N} inputs={len(inputs)} {time.strftime('%Y-%m-%d %H:%M')}\n")
    for idx, (pos, neg, M0) in enumerate(inputs):
        t1 = time.time()
        terms = pos + neg
        S = sum(t**k for t in pos) - sum(t**k for t in neg)
        M = modulus(S, k, M0) if S else None
        if M is None:
            print(f"  input {idx}: sum not divisible by {M0}^{k}; skipped", flush=True)
            continue
        Dk = S // M**k
        absorbed = np.zeros(N, dtype=np.int64)
        pos_fixed, neg_fixed = [], []
        for t, sign, fixed in [(t, 1, pos_fixed) for t in pos] + [(t, -1, neg_fixed) for t in neg]:
            if t % M == 0 and t // M <= N:
                absorbed[t // M - 1] += sign
            else:
                fixed.append(t)
        sols = particular(Dk, k, N)
        if not sols:
            print(f"  input {idx}: no particular solution found; skipped", flush=True)
            continue
        base = np.array([c - absorbed for c in sols])
        shifts = D[rng.permutation(len(D))[: a.starts]].astype(np.int64)
        starts = np.concatenate([base] + [base[:1] + sh[None, :] for sh in shifts]
                                + [base[:1] - sh[None, :] for sh in shifts])
        fr = run_descent(starts, D, p_off=len(neg_fixed), q_off=len(pos_fixed))
        found = 0
        for m, (n, c) in fr.items():
            if m > a.mmax:
                continue
            s = assemble(k, M, c, pos_fixed, neg_fixed)
            if not s.is_valid():
                raise AssertionError(f"assembled identity fails verification k={k} input {idx}")
            if (s.m, s.n) != (m, n):        # bases merged across the two parts
                m, n = s.m, s.n
            if m not in best or n < best[m][0]:
                best[m] = (n, s)
                r = cur.get(k, m)
                tag = f" // NEW record (was {r})" if r is not None and n < r else ""
                out.write(f"{s.format()}{tag} [complete M={M} c={len(terms)}]\n")
                out.flush()
                found += 1
        print(f"  input {idx}: c={len(pos)}+{len(neg)} M={M} fixed={len(pos_fixed)}+{len(neg_fixed)} "
              f"D~10^{len(str(abs(Dk))) - 1} "
              f"particular={len(sols)} new frontier entries {found}  {time.time() - t1:.0f}s", flush=True)
    cells = []
    for m in sorted(best):
        if m > 16:
            continue
        n = best[m][0]
        r = cur.get(k, m)
        cells.append(f"{m}:{n}" + ("+" if r is None else "<%d!" % r if n < r else "=" if n == r else "/%d" % r))
    print(f"  k={k:3d}  " + "  ".join(cells), flush=True)


if __name__ == "__main__":
    main()
