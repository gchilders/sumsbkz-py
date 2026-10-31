#!/usr/bin/env python3
"""Verify the identities of the data archive.  Python 3 standard library only.

    python3 verify.py [identities.txt] [table_reference.csv]

If totals.txt and totals_reference.csv are present, the identities in totals.txt are checked in
the same way, and their total m+n is compared with the least m+n of an identity for the same k
in Meyrignac's database and the 2003 archive.

For each line "(k,m,n) lhs=rhs" of identities.txt, where a side is a sum of terms
"b" or "b*c" (c copies of b^k), the script checks in exact integer arithmetic that
sum lhs b^k = sum rhs b^k, that no base occurs on both sides, that the two sides
have m and n terms with m <= n, and that the bases have no common factor.  It then
compares n with the reference value for (k, m) in table_reference.csv, the smaller of
the table entry and the least n of an identity of that type in Meyrignac's database,
and reports how many entries are improved, separately for entries attained by an
identity in the table and for entries that are derived bounds.  Exit status 0 means every check passed.
"""

import csv
import re
import sys
from math import gcd

LINE = re.compile(r"^\((\d+),(\d+),(\d+)\)\s+([0-9*+]+)=([0-9*+]+)\s*$")


def side(text):
    terms = {}
    for t in text.split("+"):
        b, _, c = t.partition("*")
        b, c = int(b), int(c) if c else 1
        if b <= 0 or c <= 0:
            raise ValueError(f"bad term {t!r}")
        terms[b] = terms.get(b, 0) + c
    return terms


def check(k, m, n, L, R):
    """The checks common to both files; returns a list of errors."""
    errs = []
    if sum(c * b ** k for b, c in L.items()) != sum(c * b ** k for b, c in R.items()):
        errs.append("sides differ")
    if set(L) & set(R):
        errs.append("a base occurs on both sides")
    p, q = sum(L.values()), sum(R.values())
    if (min(p, q), max(p, q)) != (m, n):
        errs.append(f"term counts are ({min(p, q)},{max(p, q)}), not ({m},{n})")
    g = 0
    for b in list(L) + list(R):
        g = gcd(g, b)
    if g != 1:
        errs.append(f"bases have common factor {g}")
    return errs


def totals(path="totals.txt", ref="totals_reference.csv"):
    """Check the identities that lower the least known total m+n; returns the number of failures."""
    import os
    if not (os.path.exists(path) and os.path.exists(ref)):
        return 0
    least = {}
    with open(ref, newline="") as fh:
        for r in csv.DictReader(fh):
            least[int(r["k"])] = int(r["least_total_of_database_identity"])
    ok = bad = 0
    for lineno, line in enumerate(open(path), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        mt = LINE.match(line)
        if not mt:
            print(f"{path} line {lineno}: cannot parse"); bad += 1; continue
        k, m, n = (int(mt.group(i)) for i in (1, 2, 3))
        errs = check(k, m, n, side(mt.group(4)), side(mt.group(5)))
        if k not in least:
            errs.append("no reference total")
        elif m + n >= least[k]:
            errs.append(f"m+n = {m + n} is not below the least total {least[k]}")
        if errs:
            print(f"{path} line {lineno} ({k},{m},{n}): " + "; ".join(errs)); bad += 1
        else:
            ok += 1
    print(f"{ok} identities lower the least known total m+n for their k, {bad} failed")
    return bad


def main():
    ids = sys.argv[1] if len(sys.argv) > 1 else "identities.txt"
    ref = sys.argv[2] if len(sys.argv) > 2 else "table_reference.csv"
    table = {}
    with open(ref, newline="") as fh:
        for r in csv.DictReader(fh):
            n = int(r["table_n"])
            if r["least_n_of_database_identity"]:
                n = min(n, int(r["least_n_of_database_identity"]))
            table[(int(r["k"]), int(r["m"]))] = (n, r["kind"])
    ok = bad = 0
    improved = {}
    for lineno, line in enumerate(open(ids), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        mt = LINE.match(line)
        if not mt:
            print(f"line {lineno}: cannot parse"); bad += 1; continue
        k, m, n = (int(mt.group(i)) for i in (1, 2, 3))
        errs = check(k, m, n, side(mt.group(4)), side(mt.group(5)))
        t = table.get((k, m))
        if t is None:
            errs.append("no table entry")
        elif n >= t[0]:
            errs.append(f"n = {n} does not improve the reference value {t[0]}")
        if errs:
            print(f"line {lineno} ({k},{m},{n}): " + "; ".join(errs)); bad += 1
        else:
            ok += 1
            improved[(k, m)] = (t[0] - n, t[1])
    kinds = {}
    for _, kind in improved.values():
        kinds[kind] = kinds.get(kind, 0) + 1
    print(f"{ok} identities verified, {bad} failed")
    print(f"{len(improved)} table entries improved, {sum(s for s, _ in improved.values())} terms saved")
    for kind, c in sorted(kinds.items()):
        print(f"  {c} improve an entry of kind: {kind}")
    bad += totals()
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
