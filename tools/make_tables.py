#!/usr/bin/env python3
"""Generate the paper's tables and numbers from the verified result files.

    python tools/make_tables.py            # reads results/**, writes paper/tables/ and paper/data/

Every identity is re-verified in exact arithmetic before it is counted.  Outputs:
  paper/tables/numbers.tex   \\newcommand macros for every headline number used in the text
  paper/tables/perk.tex      per-power summary (cells improved, terms saved, largest gain)
  paper/tables/improved.tex  every improved (k, m) cell: record n, new n, saving, largest base
  paper/tables/gh.tex        Gaussian-heuristic estimate of the shortest relation vs N
  paper/data/perk.dat        per-power counts for the bar chart (pgfplots)
  paper/data/new_records.txt the improved identities in the euler.free.fr (k,m,n) format
"""

from __future__ import annotations

import glob
import math
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eslp.solutions import read_solutions  # noqa: E402
from tools.compare import load_current  # noqa: E402

HERE = os.path.join(os.path.dirname(__file__), "..")
SOURCES = ["results/sieve_normal.txt", "results/combine_normal.txt", "results/pvc/*.txt",
           "results/nrp/*.log", "results/turing/**/*.txt", "results/ablation_identities.txt",
           "results/ablation_nrp_identities.txt", "results/nsweep_identities.txt"]
MMAX = 16
KS = range(17, 31)


def gh2(k, N):
    d = N - 1
    logdet = 0.5 * math.log(sum(x ** (2 * k) for x in range(1, N + 1)))
    return d / (2 * math.pi * math.e) * math.exp(2 * logdet / d)


def table_meta():
    """(k, m) -> (status, best database identity n or None) from the scraped records.csv."""
    import csv
    meta = {}
    with open(os.path.join(HERE, "data/euler_free_fr/records.csv"), encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["kind"] != "normal" or not r["n"]:
                continue
            meta[(int(r["k"]), int(r["m"]))] = (r["status"], int(r["db_best_n"]) if r.get("db_best_n") else None)
    return meta


def main():
    cur = load_current("normal")
    meta = table_meta()
    files = sorted({p for pat in SOURCES for p in glob.glob(os.path.join(HERE, pat), recursive=True)})
    best, checked, bad = {}, 0, 0
    for path in files:
        for s in read_solutions(path):
            checked += 1
            if not s.is_valid():
                bad += 1
                continue
            key = (s.k, s.m)
            if key not in best or s.n < best[key].n:
                best[key] = s
    beats_all = {key: s for key, s in best.items()
                 if key[1] <= MMAX and cur.get(*key) is not None and s.n < cur.get(*key)}
    # the main table (17 <= k <= 32) and the high-power table (33 <= k <= 100) are reported separately
    beats = {key: s for key, s in beats_all.items() if 17 <= key[0] <= 30}
    k31 = {key: s for key, s in beats_all.items() if key[0] in (31, 32)}
    high = {key: s for key, s in beats_all.items() if 33 <= key[0] <= 100}
    ties = [key for key, s in best.items()
            if 17 <= key[0] <= 30 and key[1] <= MMAX and cur.get(*key) is not None
            and s.n == cur.get(*key) and key not in beats]

    cells = [(k, m) for k in range(17, 31) for m in range(1, MMAX + 1) if cur.get(k, m) is not None]
    derived = [key for key in beats if meta.get(key, ("", None))[0] == "derived_bound"]
    no_ident = [key for key in beats if meta.get(key, ("", None))[1] is None]
    ties_derived = [key for key in ties if meta.get(key, ("", None))[0] == "derived_bound"]
    # the table's convention f(k, m+c) <= f(k, m) + c, applied to our own frontier (not used for counting)
    clos = {}
    for (k, m) in cells:
        vals = [best[(k, mm)].n + (m - mm) for mm in range(1, m + 1) if (k, mm) in best]
        if vals:
            clos[(k, m)] = min(vals)
    clos_extra = [key for key in cells if key in clos and clos[key] < cur.get(*key) and key not in beats]
    clos_lower = [key for key in beats if clos[key] < beats[key].n]

    os.makedirs(os.path.join(HERE, "paper/tables"), exist_ok=True)
    os.makedirs(os.path.join(HERE, "paper/data"), exist_ok=True)

    saved = {key: cur.get(*key) - s.n for key, s in beats.items()}
    top = max(beats, key=lambda key: saved[key])
    perk = defaultdict(lambda: [0, 0, 0, 0])         # k -> [cells, terms saved, largest gain, derived]
    for (k, m), s in beats.items():
        perk[k][0] += 1
        perk[k][1] += saved[(k, m)]
        perk[k][2] = max(perk[k][2], saved[(k, m)])
        perk[k][3] += (k, m) in derived
    full = [k for k in KS if perk[k][0] == MMAX]

    def word(n):
        return {0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
                7: "seven", 8: "eight", 9: "nine", 10: "ten"}.get(n, str(n))

    def andlist(items):
        items = list(items)
        return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]

    with open(os.path.join(HERE, "paper/tables/numbers.tex"), "w") as f:
        f.write("% generated by tools/make_tables.py -- do not edit\n")
        f.write(f"\\newcommand{{\\NumImproved}}{{{len(beats)}}}\n")
        f.write(f"\\newcommand{{\\NumTies}}{{{len(ties)}}}\n")
        f.write(f"\\newcommand{{\\TermsSaved}}{{{sum(saved.values()):,}}}\n")
        f.write(f"\\newcommand{{\\MaxGain}}{{{saved[top]}}}\n")
        f.write(f"\\newcommand{{\\MaxGainCell}}{{$({top[0]},{top[1]})$}}\n")
        f.write(f"\\newcommand{{\\MaxGainOld}}{{{cur.get(*top)}}}\n")
        f.write(f"\\newcommand{{\\MaxGainNew}}{{{beats[top].n}}}\n")
        f.write(f"\\newcommand{{\\MaxBase}}{{{max(max(s.coeffs) for s in beats.values())}}}\n")
        f.write(f"\\newcommand{{\\NumChecked}}{{{checked:,}}}\n")
        f.write(f"\\newcommand{{\\NumFullK}}{{{word(len(full))}}}\n")
        f.write(f"\\newcommand{{\\FullKList}}{{{', '.join(map(str, full))}}}\n")
        f.write(f"\\newcommand{{\\KMinImproved}}{{{min(k for k, _ in beats)}}}\n")
        f.write(f"\\newcommand{{\\KMaxImproved}}{{{max(k for k, _ in beats)}}}\n")
        f.write(f"\\newcommand{{\\NumCells}}{{{len(cells)}}}\n")
        f.write(f"\\newcommand{{\\NumNotImproved}}{{{len(cells) - len(beats)}}}\n")
        f.write(f"\\newcommand{{\\NumDerived}}{{{len(derived)}}}\n")
        f.write(f"\\newcommand{{\\NumBeatIdent}}{{{len(beats) - len(no_ident)}}}\n")
        f.write(f"\\newcommand{{\\NoIdentCells}}{{{andlist(f'$({k},{m})$' for k, m in sorted(no_ident))}}}\n")
        f.write(f"\\newcommand{{\\NumTiesDerived}}{{{len(ties_derived)}}}\n")
        nt = len(ties_derived)
        tds = ("" if nt == 0 else
               "One of the matched entries is also a derived bound, and for it our identity is the first "
               "of its type." if nt == 1 else
               f"{word(nt).capitalize()} of the matched entries are also derived bounds, and for them our "
               "identities are the first of their type.")
        f.write(f"\\newcommand{{\\TiesDerivedSentence}}{{{tds}}}\n")
        f.write(f"\\newcommand{{\\ClosureExtra}}{{{andlist(f'$({k},{m})$' for k, m in sorted(clos_extra))}}}\n")
        f.write(f"\\newcommand{{\\NumClosureExtra}}{{{word(len(clos_extra))}}}\n")
        f.write(f"\\newcommand{{\\NumClosureLower}}{{{len(clos_lower)}}}\n")
        cells_derived = [key for key in cells if meta.get(key, ("", None))[0] == "derived_bound"]
        f.write(f"\\newcommand{{\\NumCellsDerived}}{{{len(cells_derived)}}}\n")
        f.write(f"\\newcommand{{\\NumCellsIdent}}{{{len(cells) - len(cells_derived)}}}\n")
        f.write(f"\\newcommand{{\\NumImprovedIdent}}{{{len(beats) - len(derived)}}}\n")
        f.write(f"\\newcommand{{\\TermsSavedIdent}}{{{sum(v for key, v in saved.items() if key not in derived):,}}}\n")
        f.write(f"\\newcommand{{\\TermsSavedDerived}}{{{sum(v for key, v in saved.items() if key in derived):,}}}\n")

    # the high-power table, 33 <= k <= 100
    hcells = [(k, m) for k in range(33, 101) for m in range(1, MMAX + 1) if cur.get(k, m) is not None]
    hderived = [key for key in high if meta.get(key, ("", None))[0] == "derived_bound"]
    hcells_derived = [key for key in hcells if meta.get(key, ("", None))[0] == "derived_bound"]
    hsaved = {key: cur.get(*key) - s.n for key, s in high.items()}
    htop = max(high, key=lambda key: hsaved[key])
    hperk = defaultdict(lambda: [0, 0, 0, 0, 0])      # k -> [improved, saved, largest, derived, cells]
    for (k, m) in hcells:
        hperk[k][4] += 1
    for (k, m), s in high.items():
        hperk[k][0] += 1
        hperk[k][1] += hsaved[(k, m)]
        hperk[k][2] = max(hperk[k][2], hsaved[(k, m)])
        hperk[k][3] += (k, m) in hderived
    hks = sorted({k for k, _ in hcells})
    with open(os.path.join(HERE, "paper/tables/numbers.tex"), "a") as f:
        f.write(f"\\newcommand{{\\HNumCells}}{{{len(hcells):,}}}\n")
        f.write(f"\\newcommand{{\\HNumCellsDerived}}{{{len(hcells_derived)}}}\n")
        f.write(f"\\newcommand{{\\HNumCellsIdent}}{{{len(hcells) - len(hcells_derived)}}}\n")
        f.write(f"\\newcommand{{\\HNumImproved}}{{{len(high):,}}}\n")
        f.write(f"\\newcommand{{\\HNumDerived}}{{{len(hderived)}}}\n")
        f.write(f"\\newcommand{{\\HNumImprovedIdent}}{{{len(high) - len(hderived)}}}\n")
        f.write(f"\\newcommand{{\\HTermsSaved}}{{{sum(hsaved.values()):,}}}\n")
        f.write(f"\\newcommand{{\\HMaxGain}}{{{hsaved[htop]:,}}}\n")
        f.write(f"\\newcommand{{\\HMaxGainCell}}{{$({htop[0]},{htop[1]})$}}\n")
        f.write(f"\\newcommand{{\\HMaxGainOld}}{{{cur.get(*htop):,}}}\n")
        f.write(f"\\newcommand{{\\HMaxGainNew}}{{{high[htop].n:,}}}\n")
        f.write(f"\\newcommand{{\\HNumK}}{{{len(hks)}}}\n")
        f.write(f"\\newcommand{{\\HNumFullK}}{{{sum(1 for k in hks if hperk[k][0] == hperk[k][4])}}}\n")
        none = [str(k) for k in hks if hperk[k][0] == 0]
        f.write(f"\\newcommand{{\\HNoneK}}{{{', '.join(none)}}}\n")
        f.write("\\newcommand{\\HNoneSentence}{" + (f"No entry is improved for $k\\in\\{{{', '.join(none)}\\}}$."
                if none else "Every power has at least one improved entry.") + "}\n")
        f.write(f"\\newcommand{{\\HMaxBase}}{{{max(max(s.coeffs) for s in high.values()):,}}}\n")
        f.write(f"\\newcommand{{\\AllImproved}}{{{len(beats) + len(high) + len(k31):,}}}\n")
        for kk, name in ((31, "One"), (32, "Two")):
            sub = {key: v for key, v in k31.items() if key[0] == kk}
            f.write(f"\\newcommand{{\\KThirty{name}Improved}}{{{len(sub)}}}\n")
            f.write(f"\\newcommand{{\\KThirty{name}Saved}}{{{sum(cur.get(*key) - v.n for key, v in sub.items())}}}\n")
    with open(os.path.join(HERE, "paper/tables/perk_high.tex"), "w") as f:
        f.write("% generated by tools/make_tables.py -- do not edit\n")
        half = (len(hks) + 1) // 2
        f.write("\\begin{tabular}{rrrrr@{\\hspace{1.5em}}rrrrr}\n\\toprule\n"
                "$k$ & ident. & bound & saved & largest & $k$ & ident. & bound & saved & largest \\\\\n\\midrule\n")

        def cellrow(k):
            c, t, g, dv, _ = hperk[k]
            return f"{k} & {c - dv} & {dv} & {t:,} & {f'{g:,}' if c else '--'}"
        for i in range(half):
            left = cellrow(hks[i])
            right = cellrow(hks[i + half]) if i + half < len(hks) else "& & & &"
            f.write(f"{left} & {right} \\\\\n")
        f.write("\\midrule\n")
        f.write(f"\\multicolumn{{10}}{{l}}{{all: {len(high) - len(hderived)} identity entries and {len(hderived)} "
                f"bounds improved, {sum(hsaved.values()):,} terms saved}} \\\\\n\\bottomrule\n\\end{{tabular}}\n")

    with open(os.path.join(HERE, "paper/tables/perk.tex"), "w") as f:
        f.write("% generated by tools/make_tables.py -- do not edit\n")
        f.write("\\begin{tabular}{rrrrrr}\n\\toprule\n& \\multicolumn{3}{c}{entries improved} & & \\\\\n"
                "\\cmidrule(lr){2-4}\n$k$ & identity & bound & total & terms saved & largest saving \\\\\n\\midrule\n")
        for k in KS:
            c, t, g, dv = perk[k]
            f.write(f"{k} & {c - dv} & {dv} & {c} & {t} & {g if c else '--'} \\\\\n")
        f.write("\\midrule\n")
        f.write(f"all & {len(beats) - len(derived)} & {len(derived)} & {len(beats)} & {sum(saved.values()):,} & {saved[top]} "
                f"\\\\\n\\bottomrule\n\\end{{tabular}}\n")

    with open(os.path.join(HERE, "paper/data/perk.dat"), "w") as f:
        f.write("k cells full\n")
        for k in KS:
            f.write(f"{k} {perk[k][0]} {1 if perk[k][0] == MMAX else 0}\n")

    with open(os.path.join(HERE, "paper/tables/improved.tex"), "w") as f:
        f.write("% generated by tools/make_tables.py -- do not edit\n")
        f.write("\\begin{longtable}{rrrrrr}\n\\caption{Every improved cell with $m \\le 16$. "
                "``Record'' is the least $n$ in Meyrignac's table (accessed 2026-10-27); "
                "``max'' is the largest base in the new identity.}\\label{tab:improved}\\\\\n"
                "\\toprule\n$k$ & $m$ & record $n$ & new $n$ & saved & max \\\\\n\\midrule\n\\endfirsthead\n"
                "\\toprule\n$k$ & $m$ & record $n$ & new $n$ & saved & max \\\\\n\\midrule\n\\endhead\n")
        last = None
        for (k, m) in sorted(beats):
            s = beats[(k, m)]
            if last is not None and k != last:
                f.write("\\addlinespace\n")
            f.write(f"{k} & {m} & {cur.get(k, m)} & {s.n} & {saved[(k, m)]} & {max(s.coeffs)} \\\\\n")
            last = k
        f.write("\\bottomrule\n\\end{longtable}\n")

    # one-page grid: new n / record n for improved cells, record n (grey) otherwise
    with open(os.path.join(HERE, "paper/tables/grid.tex"), "w") as f:
        f.write("% generated by tools/make_tables.py -- do not edit\n")
        f.write("\\setlength{\\tabcolsep}{1.6pt}\n\\begin{tabular}{r" + "c" * MMAX + "}\n\\toprule\n")
        f.write("$k\\backslash m$ & " + " & ".join(str(m) for m in range(1, MMAX + 1)) + " \\\\\n\\midrule\n")
        gridb = {**beats, **k31}
        for k in range(17, 33):
            row = []
            for m in range(1, MMAX + 1):
                r = cur.get(k, m)
                if (k, m) in gridb:
                    row.append(f"\\textbf{{{gridb[(k, m)].n}}}/{r}")
                elif r is None:
                    row.append("")
                else:
                    row.append(f"\\textcolor{{gray}}{{{r}}}")
            f.write(f"{k} & " + " & ".join(row) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    Ns = [50, 70, 100, 150, 200, 250]
    with open(os.path.join(HERE, "paper/tables/gh.tex"), "w") as f:
        f.write("% generated by tools/make_tables.py -- do not edit\n")
        f.write("\\begin{tabular}{r" + "r" * len(Ns) + "r}\n\\toprule\n")
        f.write("$k$ & " + " & ".join(f"$N={N}$" for N in Ns) + " & best $N$ \\\\\n\\midrule\n")
        for k in (20, 22, 24, 26, 28, 30, 32):
            bestN = min(range(40, 400), key=lambda N: gh2(k, N))
            f.write(f"{k} & " + " & ".join(f"{gh2(k, N):.0f}" for N in Ns) + f" & {bestN} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    with open(os.path.join(HERE, "paper/data/new_records.txt"), "w") as f:
        f.write("// improved cells (m <= 16) vs euler.free.fr, accessed 2026-10-27; every identity verified exactly\n")
        for key in sorted({**beats, **high, **k31}):
            s = beats.get(key) or high.get(key) or k31[key]
            s.note = ""
            f.write(f"{s.format()} // record was ({key[0]},{key[1]},{cur.get(*key)})\n")

    print(f"{len(files)} files, {checked} identities read ({bad} failed verification), "
          f"{len(beats)} improved cells, {len(ties)} ties, {sum(saved.values())} terms saved; "
          f"largest gain {saved[top]} at {top}; fully improved k: {full}; cells {len(cells)}; "
          f"vs derived bounds {len(derived)}; no database identity {sorted(no_ident)}; "
          f"ties with derived {len(ties_derived)}; closure would add {sorted(clos_extra)}")


if __name__ == "__main__":
    main()
