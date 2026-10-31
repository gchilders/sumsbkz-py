#!/usr/bin/env python3
"""sumsbkz.py -- port of SumsBKZ 1.3.3 (Childers, 2000-2003) to fpylll.

Searches for equal sums of like powers sum c_i x_i^k = 0 by reducing the relation
lattice of x_1..x_n with BKZ, then improving each relation with Yura's greedy
add/subtract pass.  Solutions that tie or beat a reference record table are
written in the standard '(k,m,n) lhs=rhs' format.

Example (the classic setting: bases 1..40, k = 9..20, BKZ-20, one pass):
    python sumsbkz.py -n 40 -k 9 20 -b 20 --passes 1 --ref data/archive/database.txt
"""

from __future__ import annotations

import argparse
import random
import sys
import time
import numpy as np

from eslp.greedy import yura
from eslp.lattice import bkz, choose_bases, first_primes, relation_basis, verify
from eslp.solutions import Frontier, Solution, read_solutions
from tools.compare import load, report


def candidates(xs, k, block, max_loops, seed):
    R = relation_basis(xs, k)
    R = bkz(R, block, max_loops=max_loops, seed=seed)
    return yura(R)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", type=int, nargs="+", required=True, help="number of bases, or nmin nmax")
    ap.add_argument("-k", type=int, nargs="+", required=True, help="power, or kmin kmax")
    ap.add_argument("-b", "--block", type=int, default=20, help="BKZ block size (default 20)")
    ap.add_argument("--max-loops", type=int, default=8, help="BKZ tours (default 8, with auto-abort)")
    ap.add_argument("--workterms", type=int, default=0, help="reduce only this many of the n bases")
    ap.add_argument("--includelow", action="store_true", help="with --workterms, always include the 5 smallest")
    ap.add_argument("--primes", action="store_true", help="use the first n primes as bases")
    ap.add_argument("--passes", type=int, default=1, help="randomized passes over all (n, k); 0 = forever")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--ref", nargs="*", default=[], help="solution files giving the record table to compare against")
    ap.add_argument("--ref-maxbase", type=int, default=None,
                    help="only count reference solutions whose bases are all <= this (default: n max)")
    ap.add_argument("--slack", type=int, default=0, help="also write solutions within this many terms of the record")
    ap.add_argument("-o", "--out", default=None, help="append solutions here (default output.txt / poutput.txt)")
    ap.add_argument("-q", "--quiet", action="store_true")
    a = ap.parse_args(argv)

    nmin, nmax = (a.n * 2)[:2]
    kmin, kmax = (a.k * 2)[:2]
    out_path = a.out or ("poutput.txt" if a.primes else "output.txt")
    rng = random.Random(a.seed)

    pool_max = first_primes(nmax)[-1] if a.primes else nmax
    ref = load(a.ref, a.ref_maxbase or pool_max, a.primes)

    found = Frontier()
    seen: set[tuple] = set()
    t0 = time.time()

    with open(out_path, "a") as out:
        out.write(f"// sumsbkz.py n={nmin}..{nmax} k={kmin}..{kmax} block={a.block} "
                  f"workterms={a.workterms} includelow={int(a.includelow)} primes={int(a.primes)} seed={a.seed}\n")
        p = 0
        while a.passes == 0 or p < a.passes:
            p += 1
            for n in range(nmin, nmax + 1):
                xs = choose_bases(n, rng, a.primes, a.workterms, a.includelow)
                for k in range(kmin, kmax + 1):
                    tk = time.time()
                    C = candidates(xs, k, a.block, a.max_loops, rng.randrange(2**31))
                    nsol = 0
                    for c in C:
                        if not c.any():
                            continue
                        s = Solution(k, {x: int(v) for x, v in zip(xs, c) if v}).normalized()
                        key = (k, tuple(sorted(s.coeffs.items())))
                        if key in seen:
                            continue
                        seen.add(key)
                        if not verify(xs, k, c):
                            raise AssertionError(f"non-relation from lattice: k={k} {s.format()}")
                        found.add(s)
                        r = ref.get(k, s.m)
                        if r is None or s.n <= r + a.slack:
                            tag = "" if r is None or s.n >= r else " // beats reference"
                            out.write(s.format() + tag + "\n")
                            nsol += 1
                    out.flush()
                    if not a.quiet:
                        print(f"pass {p}  n {n}  k {k}  rel {len(C)}  written {nsol}  "
                              f"{time.time() - tk:.2f}s", flush=True)

    print(f"\nDone in {time.time() - t0:.1f}s; {len(seen)} distinct relations.")
    if ref.best:
        print("Best found vs reference (m:n; '=' tie, '<ref!' beat, '/ref' worse, '+' no ref entry):")
        report(found, ref, (kmin, kmax))
    return 0


if __name__ == "__main__":
    sys.exit(main())
