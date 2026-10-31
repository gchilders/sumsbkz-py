#!/usr/bin/env python3
"""Which method and settings produced the best identity for each table entry?

    python tools/attribution.py [-o paper/data/attribution.json]

Every result file is mapped to a method:
  sieve:N     sievesearch (sieve + descent) with bases 1..N (from the file's header or name)
  coset       moment-coset starts (tools/momentstart.py)
  wcomplete   completions of Wroblewski's congruence inputs (tools/complete.py)
  struct      completions of structured congruence solutions (tools/structsolve.py)
  recstart    descents started from known identities (tools/recordstart.py)
  other       anything else (combination, ablation, ...)
For each power the method attaining the best n in the most entries (m <= 16) is reported,
with the counts, so that further runs can use the most productive settings.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from eslp.solutions import parse_line  # noqa: E402

HERE = os.path.join(os.path.dirname(__file__), "..")
HDR = re.compile(r"^//\s*sievesearch .*?\bN=(\d+)")


def method_of(path):
    b = os.path.basename(path)
    if b.startswith("recstart-"):
        return "recstart"
    if "/rerun" in path:                      # reruns: results/rerun*/ with the method in the name
        m = re.match(r"k\d+-N(\d+)-s\d+\.txt$", b)
        if m:
            return f"sieve:{m.group(1)}"
        for pre, meth in (("mstart-", "coset"), ("wcomplete-", "wcomplete"), ("struct-", "struct")):
            if b.startswith(pre):
                return meth
    if b.startswith("mstart-"):
        return "coset"
    if b.startswith("comp-struct-"):
        return "struct"
    if "/complete/" in path and re.match(r"k\d+s?-N\d+", b):
        return "wcomplete"
    m = re.match(r"k\d+-N(\d+)\.txt$", b)
    if m and ("/t4/" in path or "/cpu/" in path or "/h200/" in path or "/bench/" in path):
        return f"sieve:{m.group(1)}"
    if "sieve_normal" in b or "/pvc/" in path:
        return "sieve:?"
    if "nsweep" in b or "ablation" in b:
        return "sieve:exp"
    return "other"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--out", default="paper/data/attribution.json")
    a = ap.parse_args()
    best = {}                                   # (k, m) -> (n, set of methods attaining it)
    files = sorted({p for pat in ("results/**/*.txt",) for p in glob.glob(os.path.join(HERE, pat), recursive=True)})
    for p in files:
        meth0 = method_of(p)
        N = None
        for line in open(p, errors="replace"):
            h = HDR.match(line)
            if h:
                N = h.group(1)
                continue
            try:
                s = parse_line(line)
            except ValueError:
                continue
            if s is None or s.m > 16 or not s.is_valid():
                continue
            meth = f"sieve:{N}" if meth0 == "sieve:?" and N else meth0
            key = (s.k, s.m)
            if key not in best or s.n < best[key][0]:
                best[key] = (s.n, {meth})
            elif s.n == best[key][0]:
                best[key][1].add(meth)
    perk = defaultdict(Counter)
    for (k, m), (n, ms) in best.items():
        for meth in ms:
            perk[k][meth] += 1
    out = {}
    for k in sorted(perk):
        if not (17 <= k <= 100):
            continue
        out[k] = perk[k].most_common()
        print(f"k={k:3d}  " + "  ".join(f"{m}:{c}" for m, c in perk[k].most_common(4)), flush=True)
    json.dump({str(k): v for k, v in out.items()}, open(os.path.join(HERE, a.out), "w"), indent=1)


if __name__ == "__main__":
    main()
