#!/usr/bin/env python3
"""Statistics from the run logs: frontier against the lattice dimension, and compute used.

    python tools/logstats.py     # writes paper/tables/resources.tex and paper/data/logstats.json

Reads the NRP job logs (results/pvc/*.log), the Turing CPU job logs
(results/turing/slurm/eslp-k*-N*.out) and the H200 logs.  Each log has lines
    k=K run i/r  bases 1..N (N)  db S  sieve Xs  descent Ys
followed, after the runs for K, by the frontier line of those runs
    k= K  1:117<162!  2:119<146! ...
(cells are m:n, then '<rec!', '=', or '/rec').  Only full ranges of bases 1..N are used
for the dimension comparison.  Compute used is taken from the logs (wall time of each
job times the threads it was given) and from the Slurm accounting in
results/turing/sacct.txt.
"""

from __future__ import annotations

import glob
import json
import os
import re
import statistics
import sys
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools.compare import load_current  # noqa: E402

HERE = os.path.join(os.path.dirname(__file__), "..")
RUN = re.compile(r"^k=(\d+) run \d+/\d+\s+bases (\d+)\.\.(\d+) \((\d+)\)\s+db (\d+)\s+sieve ([\d.]+)s\s+descent ([\d.]+)s")
FRONT = re.compile(r"^\s+k=\s*(\d+)\s+(.*)$")
CELL = re.compile(r"(\d+):(\d+)")
MMAX = 16
BANDS = ((1, 4), (5, 8), (9, 12), (13, 16))


def parse(path):
    """Yield (k, N, full_range, runs, frontier{m: n}, seconds) for each k in a log."""
    pending = defaultdict(list)
    for line in open(path, errors="replace"):
        m = RUN.match(line)
        if m:
            k, lo, hi, cnt = (int(m.group(i)) for i in (1, 2, 3, 4))
            pending[k].append((lo, hi, cnt, float(m.group(6)) + float(m.group(7))))
            continue
        m = FRONT.match(line)
        if m and pending.get(int(m.group(1))):
            k = int(m.group(1))
            runs = pending.pop(k)
            lo, hi, cnt, _ = runs[0]
            full = all(r[0] == 1 and r[2] == r[1] for r in runs) and len({r[1] for r in runs}) == 1
            front = {int(a): int(b) for a, b in CELL.findall(m.group(2)) if int(a) <= MMAX}
            yield k, hi, full, len(runs), front, sum(r[3] for r in runs)


def job_core_hours():
    """NRP jobs: (done - start) x threads, from the '== TAG start/done' lines."""
    tot = 0.0
    logs = {os.path.basename(p): p for p in glob.glob(os.path.join(HERE, "results/nrp/*.log"))}
    logs.update({os.path.basename(p): p for p in glob.glob(os.path.join(HERE, "results/pvc/*.log"))})
    for p in logs.values():
        txt = open(p, errors="replace").read()
        s = re.search(r"== (\S+) start (.+?) on \S+ threads=(\d+)", txt)
        d = s and re.search(rf"== {re.escape(s.group(1))} done (.+)", txt)
        if not (s and d):
            continue
        fmt = "%a %b %d %H:%M:%S %Z %Y"
        t0, t1 = datetime.strptime(s.group(2).strip(), fmt), datetime.strptime(d.group(1).strip(), fmt)
        tot += (t1 - t0).total_seconds() / 3600 * int(s.group(3))
    return tot


def slurm_hours():
    """Turing: CPU core-hours and GPU-hours from sacct (Elapsed x AllocCPUS or x gres/gpu)."""
    path = os.path.join(HERE, "results/turing/sacct.txt")
    cpu = gpu = 0.0
    if not os.path.exists(path):
        return None, None
    for line in open(path):
        f = line.split()
        if len(f) < 6 or "eslp" not in f[1]:
            continue
        el = re.search(r"\s((?:\d+-)?\d\d:\d\d:\d\d)\s", line).group(1)
        days, _, hms = el.rpartition("-")
        h, mi, s = (int(x) for x in hms.split(":"))
        hours = (int(days) if days else 0) * 24 + h + mi / 60 + s / 3600
        g = re.search(r"gres/gpu=(\d+)", line)
        if g:
            gpu += hours * int(g.group(1))
        else:
            cpu += hours * int(f[3])
    return cpu, gpu


def main():
    cur = load_current("normal")
    logs = sorted(glob.glob(os.path.join(HERE, "results/pvc/*.log"))
                  + glob.glob(os.path.join(HERE, "results/turing/slurm/eslp-k*-N*.out")))
    best = defaultdict(dict)        # (k, N) -> {m: best n over runs at this N}
    secs = defaultdict(list)        # N -> seconds per run
    nruns = defaultdict(int)
    for p in logs:
        for k, N, full, runs, front, s in parse(p):
            if not full:
                continue
            nruns[(k, N)] += runs
            secs[N].append(s / runs)
            for m, n in front.items():
                if m not in best[(k, N)] or n < best[(k, N)][m]:
                    best[(k, N)][m] = n
    Ns = sorted({N for _, N in best})
    ks = sorted({k for k, _ in best})
    # For each k, band and N: mean over m in the band of n(N) - min over N of n.  Only k for
    # which every N in Ns was run, and m reached at every N, are used.
    rows = {}
    for lo, hi in BANDS:
        exc = defaultdict(list)
        wins = defaultdict(int)
        for k in ks:
            if not all((k, N) in best for N in Ns):
                continue
            for m in range(lo, hi + 1):
                vals = {N: best[(k, N)].get(m) for N in Ns}
                if any(v is None for v in vals.values()):
                    continue
                b = min(vals.values())
                for N in Ns:
                    exc[N].append(vals[N] - b)
                    wins[N] += vals[N] == b
        rows[(lo, hi)] = {N: (statistics.mean(exc[N]) if exc[N] else None, wins[N], len(exc[N])) for N in Ns}
    common_k = [k for k in ks if all((k, N) in best for N in Ns)]
    out = {"Ns": Ns, "common_k": common_k, "runs": {f"{k},{N}": c for (k, N), c in nruns.items()},
           "median_seconds_per_run": {N: statistics.median(secs[N]) for N in Ns},
           "bands": {f"{lo}-{hi}": {N: v for N, v in r.items()} for (lo, hi), r in rows.items()},
           "best": {f"{k},{N}": f for (k, N), f in best.items()}}
    nrp = job_core_hours()
    cpu, gpu = slurm_hours()
    out["core_hours"] = {"nrp": nrp, "turing_cpu": cpu, "turing_gpu_hours": gpu}
    json.dump(out, open(os.path.join(HERE, "paper/data/logstats.json"), "w"), indent=1)
    with open(os.path.join(HERE, "paper/tables/resources.tex"), "w") as f:
        f.write("% generated by tools/logstats.py -- do not edit\n")
        f.write(f"\\newcommand{{\\CoreHoursNRP}}{{{nrp:,.0f}}}\n")
        f.write(f"\\newcommand{{\\CoreHoursTuring}}{{{cpu or 0:,.0f}}}\n")
        f.write(f"\\newcommand{{\\CoreHoursTotal}}{{{nrp + (cpu or 0):,.0f}}}\n")
        f.write(f"\\newcommand{{\\GPUHours}}{{{gpu or 0:,.0f}}}\n")
    print(json.dumps({k: v for k, v in out.items() if k != "best"}, indent=1, default=str))


if __name__ == "__main__":
    main()
