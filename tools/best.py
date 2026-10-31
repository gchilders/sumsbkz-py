#!/usr/bin/env python3
"""Consolidate solution files: verify every line, keep the best per (k, m), compare to the record table.

    python tools/best.py results/*.txt [--primes] [-o results/best_normal.txt]

Prints each (k, m) cell where a verified solution beats or ties the current
euler.free.fr table, and writes the best solution per beaten cell to -o.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eslp.solutions import read_solutions  # noqa: E402
from tools.compare import load_current  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--primes", action="store_true")
    ap.add_argument("--mmax", type=int, default=16, help="only report cells with m <= this (the site's grid)")
    ap.add_argument("-o", "--out", default=None)
    a = ap.parse_args()
    kind = "prime" if a.primes else "normal"
    cur = load_current(kind)

    best = {}
    bad = 0
    for path in a.files:
        for s in read_solutions(path):
            if not s.is_valid() or (a.primes and not s.is_prime()):
                bad += 1
                continue
            key = (s.k, s.m)
            if key not in best or s.n < best[key].n:
                best[key] = s
    if bad:
        print(f"warning: {bad} lines failed verification and were skipped", file=sys.stderr)

    beats, ties = [], []
    for (k, m), s in sorted(best.items()):
        if m > a.mmax:
            continue
        r = cur.get(k, m)
        if r is None:
            continue
        if s.n < r:
            beats.append((k, m, s.n, r, s))
        elif s.n == r:
            ties.append((k, m, s.n))
    print(f"{len(beats)} cells below the euler.free.fr table, {len(ties)} ties (m <= {a.mmax})")
    for k, m, n, r, s in beats:
        print(f"  ({k},{m},{n})  record {r}  (-{r - n})   max base {max(s.coeffs)}")
    if a.out:
        with open(a.out, "w") as f:
            f.write(f"// best verified solutions below the euler.free.fr {kind} table (as scraped 2026-10-27)\n")
            for k, m, n, r, s in beats:
                f.write(f"{s.format()} // record was ({k},{m},{r})\n")
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
