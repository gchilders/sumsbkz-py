#!/usr/bin/env python3
"""Control experiments for the combination step: where do the improvements come from?

    python tools/controls.py [-k 17 30] [--maxbase 120]

Pools, for each k (identities with bases <= maxbase only):
  H    the historical identities (Meyrignac's database and the 2003 archive), as they are;
  H*   H combined by tools/combine.combine_pool, with no new identities;
  Ha*  as H*, but with every historical identity for k, whatever its largest base;
  R    the identities stored by our runs (sieve + descent), not combined;
  R*   R combined, without the historical identities;
  RH*  R and H combined (the final result).
For each pool the number of table entries with m <= 16 that are improved and the terms
saved are printed and written to paper/tables/controls.tex; the identities of the
combined pools are appended to results/controls_<pool>.txt after exact verification.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eslp.solutions import Solution, read_solutions  # noqa: E402
from tools.combine import combine_pool, load_pool, mn  # noqa: E402
from tools.compare import load_current  # noqa: E402

HERE = os.path.join(os.path.dirname(__file__), "..")
HIST = ["data/euler_free_fr/solutions_normal.txt", "data/archive/database.txt"]
# identities stored by individual runs; combined files and best-of summaries excluded
RUNS = ["results/sieve_normal.txt", "results/pvc/*.txt", "results/nrp/*.log",
        "results/turing/cpu/*.txt", "results/turing/bench/*.txt", "results/turing/h200/*.txt",
        "results/turing/t4/*.txt", "results/turing/rerun*/*.txt", "results/turing/complete/*.txt",
        "results/turing/sieve_normal.txt", "results/turing/pvc/*.txt", "results/turing/nrp/*.log",
        "results/turing/smallN/*.txt",
        "results/ablation_identities.txt", "results/ablation_nrp_identities.txt", "results/nsweep_identities.txt"]
MMAX = 16


def files(pats):
    # descents started from known identities (tools/recordstart.py) use the historical pool, so they
    # are not part of R
    return sorted({p for pat in pats for p in glob.glob(os.path.join(HERE, pat), recursive=True)
                   if "recstart" not in os.path.basename(p)})


def score(F, cur, k):
    imp = saved = 0
    for m in range(1, MMAX + 1):
        r = cur.get(k, m)
        if r is not None and m in F and F[m][0] < r:
            imp += 1
            saved += r - F[m][0]
    return imp, saved


def frontier(V, mmax=40):
    F = {}
    for v in V:
        m, n = mn(v)
        if m <= mmax and (m not in F or n < F[m][0]):
            F[m] = (n, v)
    return F


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-k", type=int, nargs=2, default=[17, 30])
    ap.add_argument("--maxbase", type=int, default=120)
    ap.add_argument("--orderings", type=int, default=4)
    ap.add_argument("--json", default="paper/data/controls.json")
    a = ap.parse_args()
    cur = load_current("normal")
    hist = [s for p in files(HIST) for s in read_solutions(p)]
    runs = [s for p in files(RUNS) for s in read_solutions(p) if "[combine]" not in s.note]
    res = {}
    for k in range(a.k[0], a.k[1] + 1):
        H = load_pool(hist, k, a.maxbase)
        R = load_pool(runs, k, a.maxbase)
        row = {"nH": len(H), "nR": len(R)}
        row["H"] = score(frontier(H), cur, k)
        row["R"] = score(frontier(R), cur, k)
        mb = max([max(s.coeffs) for s in hist if s.k == k] + [a.maxbase])
        Fa, _ = combine_pool(load_pool(hist, k, mb), orderings=a.orderings)
        row["Ha*"], row["Ha maxbase"] = score(Fa, cur, k), mb
        for name, V in (("H*", H), ("R*", R), ("RH*", np.concatenate([R, H]) if len(H) else R)):
            t0 = time.time()
            F, info = combine_pool(V, orderings=a.orderings) if len(V) >= 2 else (frontier(V), {})
            row[name] = score(F, cur, k)
            row[name + " info"] = dict(info, seconds=round(time.time() - t0, 1))
            with open(os.path.join(HERE, f"results/controls_{name.rstrip('*')}.txt"), "a") as fh:
                fh.write(f"// controls k={k} pool {name}\n")
                for m, (n, v) in sorted(F.items()):
                    s = Solution(k, {i + 1: int(c) for i, c in enumerate(v) if c}).normalized()
                    assert s.is_valid() and (s.m, s.n) == (m, n)
                    r = cur.get(k, m)
                    if r is not None and n < r:
                        fh.write(f"{s.format()} // NEW record (was {r}) [combine {name}]\n")
        res[k] = row
        print(f"k={k}  |H|={len(H)} |R|={len(R)}  " + "  ".join(
            f"{n}: {row[n][0]}/{row[n][1]}" for n in ("H", "H*", "Ha*", "R", "R*", "RH*"))
            + f"   RH* rounds {row['RH* info'].get('rounds')} pool {row['RH* info'].get('pool')}"
            f" {row['RH* info'].get('seconds')}s", flush=True)
    os.makedirs(os.path.dirname(os.path.join(HERE, a.json)), exist_ok=True)
    json.dump({str(k): v for k, v in res.items()}, open(os.path.join(HERE, a.json), "w"), indent=1)
    cols = ("H", "H*", "Ha*", "R", "R*", "RH*")
    with open(os.path.join(HERE, "paper/tables/controls.tex"), "w") as f:
        f.write("% generated by tools/controls.py -- do not edit\n")
        f.write("\\begin{tabular}{l" + "r" * len(cols) + "}\n\\toprule\n")
        f.write("pool & " + " & ".join(f"${c.replace('*', '^{*}')}$".replace("Ha", "H_{\\mathrm{all}}") for c in cols)
                + " \\\\\n\\midrule\n")
        f.write("entries improved & " + " & ".join(str(sum(r[c][0] for r in res.values())) for c in cols) + " \\\\\n")
        f.write("terms saved & " + " & ".join(f"{sum(r[c][1] for r in res.values()):,}" for c in cols) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    with open(os.path.join(HERE, "paper/tables/controlsnumbers.tex"), "w") as f:
        f.write("% generated by tools/controls.py -- do not edit\n")
        f.write(f"\\newcommand{{\\CtrlRImp}}{{{sum(r['R'][0] for r in res.values())}}}\n")
        f.write(f"\\newcommand{{\\CtrlRSaved}}{{{sum(r['R'][1] for r in res.values()):,}}}\n")


if __name__ == "__main__":
    main()
