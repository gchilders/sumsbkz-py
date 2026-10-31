#!/usr/bin/env python3
"""Build the data archive that accompanies the paper (directory archive/).

    python tools/make_archive.py      # after tools/make_tables.py

Writes
  archive/identities.txt       every improved entry (m <= 16, 17 <= k <= 100), one per line
  archive/improvements.csv     k, m, n, the table entry it improves and its kind, the
                               saving, the largest base and the signed relation vector
  archive/table_reference.csv  the table entries with m <= 16 and 17 <= k <= 100 used for the
                               comparison (value, kind, least n of a database identity)
  archive/totals.txt           identities whose total m+n is below that of every identity for the same k
                               in Meyrignac's database and the 2003 archive (any m)
  archive/totals_reference.csv k and the least m+n of an identity in Meyrignac's database and the 2003 archive
  archive/verify.py            a standalone checker (Python 3 standard library only)
  archive/SHA256SUMS           checksums of the archive files and of the snapshot of
                               Meyrignac's pages and database from which the reference
                               table was extracted (data/euler_free_fr/raw, not included)
"""

from __future__ import annotations

import csv
import glob
import hashlib
import math
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eslp.solutions import read_solutions  # noqa: E402

HERE = os.path.join(os.path.dirname(__file__), "..")
A = os.path.join(HERE, "archive")
KIND = {"found": "identity", "derived_bound": "derived bound", "conjectured": "identity (conjectured minimal)",
        "proved": "identity (proved minimal)", "db_only": "identity (database only)"}


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def primitive(s):
    g = 0
    for b in s.coeffs:
        g = math.gcd(g, b)
    return g == 1


def totals():
    """Identities whose m+n is below every identity for the same k in the database and the 2003 archive."""
    hist = {}
    for p in HIST:
        for s in read_solutions(os.path.join(HERE, p)):
            if 17 <= s.k <= 100 and s.is_valid() and (s.k not in hist or s.total < hist[s.k]):
                hist[s.k] = s.total
    best = {}
    for p in sorted(glob.glob(os.path.join(HERE, "results/**/*.txt"), recursive=True)):
        for s in read_solutions(p):
            if s.k in hist and s.total < hist[s.k] and (s.k not in best or s.total < best[s.k].total) \
                    and s.is_valid() and primitive(s):
                best[s.k] = s
    with open(os.path.join(A, "totals_reference.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["k", "least_total_of_database_identity"])
        for k in sorted(hist):
            w.writerow([k, hist[k]])
    with open(os.path.join(A, "totals.txt"), "w") as f:
        f.write("# Identities whose total number of terms m+n is below that of every identity for the same k\n"
                "# in Meyrignac's database and the 2003 archive (any m).  Same format as identities.txt.\n")
        for k in sorted(best):
            s = best[k]
            s.note = ""
            f.write(s.format() + "\n")
    return len(best)


HIST = ["data/euler_free_fr/solutions_normal.txt", "data/archive/database.txt", "data/archive/datahigh.txt"]


def main():
    os.makedirs(A, exist_ok=True)
    meta = {}
    with open(os.path.join(HERE, "data/euler_free_fr/records.csv"), encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["kind"] == "normal" and r["n"] and 17 <= int(r["k"]) <= 100 and int(r["m"]) <= 16:
                meta[(int(r["k"]), int(r["m"]))] = r
    with open(os.path.join(A, "table_reference.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["k", "m", "table_n", "kind", "least_n_of_database_identity"])
        for (k, m), r in sorted(meta.items()):
            w.writerow([k, m, r["n"], KIND.get(r["status"], r["status"]), r.get("db_best_n", "")])

    sols = read_solutions(os.path.join(HERE, "paper/data/new_records.txt"))
    with open(os.path.join(A, "identities.txt"), "w") as fi, \
            open(os.path.join(A, "improvements.csv"), "w", newline="") as fc:
        fi.write("# Improved entries of Meyrignac's tables (m <= 16, 17 <= k <= 100), one identity per line.\n"
                 "# Format: (k,m,n) a1+a2*2+...=b1+... where b*c stands for c copies of b^k.\n")
        w = csv.writer(fc)
        w.writerow(["k", "m", "n", "reference_n", "table_kind", "saving", "max_base", "relation"])
        for s in sorted(sols, key=lambda s: (s.k, s.m)):
            s.note = ""
            r = meta[(s.k, s.m)]
            rel = " ".join(f"{x}:{c:+d}" for x, c in sorted(s.normalized().coeffs.items()))
            fi.write(s.format() + "\n")
            ref = min(int(r["n"]), int(r["db_best_n"])) if r.get("db_best_n") else int(r["n"])
            w.writerow([s.k, s.m, s.n, ref, KIND.get(r["status"], r["status"]), ref - s.n,
                        max(s.coeffs), rel])
    ntot = totals()
    shutil.copy(os.path.join(HERE, "tools/archive_verify.py"), os.path.join(A, "verify.py"))

    lines = []
    for name in ("README.md", "identities.txt", "improvements.csv", "table_reference.csv", "totals.txt",
                 "totals_reference.csv", "verify.py"):
        lines.append(f"{sha(os.path.join(A, name))}  {name}")
    raw = os.path.join(HERE, "data/euler_free_fr/raw")
    for name in sorted(os.listdir(raw)):
        lines.append(f"{sha(os.path.join(raw, name))}  snapshot/{name}")
    with open(os.path.join(A, "SHA256SUMS"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"archive: {len(sols)} identities, {len(meta)} reference entries, {ntot} lower totals")


if __name__ == "__main__":
    main()
