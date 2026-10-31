#!/usr/bin/env python3
"""Compare the record frontier of solution files against a reference.

    python tools/compare.py RESULTS... --ref data/archive/database.txt [--maxbase 50] [-k 9 20]

For every k, prints the reference record n for each m (restricted to solutions
whose largest base is <= --maxbase, so a search over bases 1..N is judged only
against what was reachable from 1..N) and the best n found in RESULTS.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eslp.solutions import Frontier, read_solutions  # noqa: E402


def load(paths, maxbase=None, primes=False, kr=None):
    f = Frontier()
    for p in paths:
        for s in read_solutions(p):
            if not s.is_valid():
                continue
            if maxbase and max(s.coeffs) > maxbase:
                continue
            if primes and not s.is_prime():
                continue
            if kr and not (kr[0] <= s.k <= kr[1]):
                continue
            f.add(s)
    return f


def load_current(kind="normal", path=None):
    """Current euler.free.fr record table (records.csv) as a Frontier.

    Uses the smaller of the grid value and the best database entry for each (k, m).
    """
    import csv
    path = path or os.path.join(os.path.dirname(__file__), "..", "data", "euler_free_fr", "records.csv")
    f = Frontier()
    with open(path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["kind"] != kind or not r["n"]:
                continue
            k, m, n = int(r["k"]), int(r["m"]), int(r["n"])
            if r.get("db_best_n"):
                n = min(n, int(r["db_best_n"]))
            cur = f.best.get((k, m))
            if cur is None or n < cur:
                f.best[(k, m)] = n
    return f


def report(found: Frontier, ref: Frontier, kr, mmax=12, out=sys.stdout):
    """Return (cells_matched, cells_beaten, cells_missed, new_cells) over reference cells."""
    tot = [0, 0, 0, 0]
    for k in range(kr[0], kr[1] + 1):
        ms = sorted(set(ref.ms(k)) | set(found.ms(k)))
        ms = [m for m in ms if m <= mmax]
        if not ms:
            continue
        cells = []
        for m in ms:
            r, f = ref.get(k, m), found.get(k, m)
            if r is None:
                cells.append(f"{m}:{f}+")
                tot[3] += 1
            elif f is None:
                cells.append(f"{m}:-/{r}")
                tot[2] += 1
            elif f < r:
                cells.append(f"{m}:{f}<{r}!")
                tot[1] += 1
            elif f == r:
                cells.append(f"{m}:{f}=")
                tot[0] += 1
            else:
                cells.append(f"{m}:{f}/{r}")
                tot[2] += 1
        print(f"k={k:3d}  " + "  ".join(cells), file=out)
    print(f"reference cells matched {tot[0]}, beaten {tot[1]}, missed {tot[2]};  "
          f"cells with no reference entry {tot[3]}", file=out)
    return tuple(tot)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="+")
    ap.add_argument("--ref", nargs="+", required=True)
    ap.add_argument("--maxbase", type=int, default=None)
    ap.add_argument("--primes", action="store_true")
    ap.add_argument("-k", type=int, nargs=2, default=[2, 100])
    ap.add_argument("--mmax", type=int, default=12)
    a = ap.parse_args()
    ref = load(a.ref, a.maxbase, a.primes, a.k)
    found = load(a.results, a.maxbase, a.primes, a.k)
    print("cell = m:found  ('=' ties ref, '<ref!' beats it, '/ref' worse, '-' none found, '+' no ref entry)")
    report(found, ref, a.k, a.mmax)


if __name__ == "__main__":
    main()
