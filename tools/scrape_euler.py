#!/usr/bin/env python3
"""Parse Jean-Charles Meyrignac's "Computing Minimal Equal Sums Of Like Powers"
site (http://euler.free.fr, plus the high-power companion http://euler2000.free.fr)
from raw HTML/text saved under data/euler_free_fr/raw/ into machine-readable form.

Stdlib only.  Re-run with:  python3 tools/scrape_euler.py [--data-dir DIR]

Inputs (in DIR/raw/):
  index.html          euler.free.fr main page: normal grid k=2..32, m=1..16, + m+n table
  prime.htm           prime-solution grid k=2..32, m=1..16, + m+n table
  euler2000_index.html high-power grid k=33.., m=1..16, + m+n table
  database.txt        normal solution database      (euler.free.fr/database.txt)
  datahigh.txt        high-power solution database  (euler2000.free.fr/datahigh.txt)
  dataprime.txt       prime solution database       (euler.free.fr/dataprime.txt)

Outputs (in DIR/):
  records.csv          one row per (kind,k,m) grid cell with a value, plus rows for
                       (k,m) with m>16 that only appear in the databases
  min_terms.csv        the "LOWEST NUMBER OF TERMS" (best m+n per k) tables
  solutions_normal.txt database.txt + datahigh.txt, normalised, one per line
  solutions_prime.txt  dataprime.txt, normalised, one per line
"""
import argparse
import csv
import html
import os
import re
import sys
from html.parser import HTMLParser

STATUS_BY_COLOR = {
    "3366ff": "proved",        # best lower bound proved
    "33ccff": "conjectured",   # best lower bound conjectured
    "66ffff": "found",         # best currently found (has a discoverer)
    "ccffff": "derived_bound", # known bound derived from neighbour (no own solution)
    "cccccc": "unexplored",
    "ff0000": "not_needed",
}


# --------------------------------------------------------------------------- HTML

class TableGrabber(HTMLParser):
    """Collect every <table> as a list of rows; each cell is a dict with
    text, bold flag, bgcolor (cell's own or inherited from <tr>).  Also records,
    for each table, the most recent big heading text seen before it."""

    def __init__(self):
        HTMLParser.__init__(self, convert_charrefs=True)
        self.tables = []      # list of dicts: {heading, rows}
        self.stack = []       # nested tables
        self.cur_row = None
        self.cur_cell = None
        self.row_color = None
        self.heading = ""
        self.in_font_big = False
        self.font_buf = []
        self.bold_depth = 0

    def handle_starttag(self, tag, attrs):
        a = dict((k.lower(), (v or "")) for k, v in attrs)
        if tag == "table":
            t = {"heading": self.heading, "rows": []}
            self.tables.append(t)
            self.stack.append((t, self.cur_row, self.cur_cell, self.row_color))
            self.cur_row = self.cur_cell = None
        elif tag == "tr" and self.stack:
            self.cur_row = []
            self.stack[-1][0]["rows"].append(self.cur_row)
            self.row_color = a.get("bgcolor", "").lstrip("#").lower() or None
        elif tag in ("td", "th") and self.stack:
            if self.cur_row is None:
                self.cur_row = []
                self.stack[-1][0]["rows"].append(self.cur_row)
            color = a.get("bgcolor", "").lstrip("#").lower() or self.row_color
            self.cur_cell = {"parts": [], "bold": False, "color": color}
            self.cur_row.append(self.cur_cell)
        elif tag == "b":
            self.bold_depth += 1
        elif tag == "br" and self.cur_cell is not None:
            self.cur_cell["parts"].append("\n")
        elif tag == "font" and a.get("size", "").strip() in ("+2", "+3"):
            self.in_font_big = True
            self.font_buf = []

    def handle_endtag(self, tag):
        if tag == "table" and self.stack:
            _, self.cur_row, self.cur_cell, self.row_color = self.stack.pop()
        elif tag in ("td", "th"):
            self.cur_cell = None
        elif tag == "tr":
            self.cur_row = None
        elif tag == "b":
            self.bold_depth = max(0, self.bold_depth - 1)
        elif tag == "font" and self.in_font_big:
            self.in_font_big = False
            self.heading = " ".join("".join(self.font_buf).split())

    def handle_data(self, data):
        if self.in_font_big:
            self.font_buf.append(data)
        if self.cur_cell is not None:
            self.cur_cell["parts"].append(data)
            if self.bold_depth and data.strip():
                self.cur_cell["bold"] = True


def cell_text(c):
    return "".join(c["parts"]).replace("\xa0", " ")


def parse_cell(c):
    """Return (n, initials, bold, status) or None for empty cells."""
    txt = cell_text(c)
    m = re.match(r"\s*(\d+)\s*(?:\(([^)]*)\))?\s*$", " ".join(txt.split()))
    if not m:
        return None
    n = int(m.group(1))
    ini = (m.group(2) or "").strip()
    status = STATUS_BY_COLOR.get(c["color"] or "", c["color"] or "")
    return n, ini, c["bold"], status


def load_html(path):
    with open(path, "rb") as f:
        raw = f.read()
    return raw.decode("latin-1")


def parse_page(path):
    g = TableGrabber()
    g.feed(load_html(path))
    g.close()
    return g.tables


def is_header(row, label):
    return row and " ".join(cell_text(row[0]).split()).lower() == label


def extract_grid(tables):
    """The 'KNOWN LOWER BOUNDS' table: header row 'k\\m', 1..16; rows k, cells."""
    for t in tables:
        rows = t["rows"]
        if rows and is_header(rows[0], "k\\m"):
            ms = [int(cell_text(c).strip()) for c in rows[0][1:]]
            out = []
            for r in rows[1:]:
                if not r:
                    continue
                k = int(cell_text(r[0]).strip())
                for m, c in zip(ms, r[1:]):
                    p = parse_cell(c)
                    if p:
                        out.append((k, m) + p)
            return out
    raise ValueError("grid not found")


def extract_min_terms(tables):
    """'LOWEST NUMBER OF TERMS' tables: pairs of rows k / m+n.  Only the first
    block of tables under the first such heading is used (prime.htm repeats a
    stale copy later)."""
    out = {}
    seen_heading = False
    for t in tables:
        if t["heading"].upper() != "LOWEST NUMBER OF TERMS":
            if seen_heading and out:
                break
            continue
        rows = t["rows"]
        if len(rows) >= 2 and is_header(rows[0], "k") and is_header(rows[1], "m+n"):
            seen_heading = True
            ks = [int(cell_text(c).strip()) for c in rows[0][1:]]
            for k, c in zip(ks, rows[1][1:]):
                p = parse_cell(c)
                if p and k not in out:
                    out[k] = p
    return out


def legend(tables):
    """Initials -> full name, from the cells 'XX: Full Name'."""
    names = {}
    for t in tables:
        for r in t["rows"]:
            for c in r:
                m = re.match(r"^\s*([A-Z][A-Z0-9]{1,3}):\s*(.+?)\s*$", " ".join(cell_text(c).split()))
                if m:
                    names.setdefault(m.group(1), m.group(2))
    return names


# --------------------------------------------------------------------- databases

HEAD_RE = re.compile(r"^\((\d+),(\d+),(\d+)\)\s*(.*)$")
TERMS_RE = re.compile(r"^\d+(\*\d+)?(\+\d+(\*\d+)?)*$")


def parse_side(s):
    terms = []
    for t in s.split("+"):
        if "*" in t:
            b, c = t.split("*")
            terms.append((int(b), int(c)))
        else:
            terms.append((int(t), 1))
    return terms


def split_name_date(attr):
    """'Laurent Lucas, 07/25/2015' -> ('Laurent Lucas', '07/25/2015')."""
    attr = attr.strip()
    m = re.match(r"^(.*?)[,\s]*\b(\d{1,2}/\d{1,2}/\d{2,4}|\d{4})\s*$", attr)
    if m and m.group(1).strip():
        return m.group(1).strip().rstrip(","), m.group(2)
    return attr, ""


def date_key(d):
    m = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{2,4})$", d)
    if m:
        y = int(m.group(3))
        y += 1900 if y < 100 and y > 50 else (2000 if y < 100 else 0)
        return (y, int(m.group(1)), int(m.group(2)))
    if re.match(r"^\d{4}$", d):
        return (int(d), 0, 0)
    return (9999, 0, 0)


def parse_db(path, kind, source):
    """Return (entries, problems).  Lines starting with ';' are commented out on
    the site and are skipped (counted in problems)."""
    entries, problems = [], []
    with open(path, "rb") as f:
        text = f.read().decode("latin-1")
    for lineno, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        if line.startswith(";"):
            problems.append((source, lineno, "commented", line[:80]))
            continue
        m = HEAD_RE.match(line)
        if not m:
            problems.append((source, lineno, "unparsed", line[:80]))
            continue
        k, mm, nn = int(m.group(1)), int(m.group(2)), int(m.group(3))
        rest = m.group(4)
        pm = re.match(r"^([0-9*+=\s]+)(.*)$", rest)
        eq, tail = (pm.group(1), pm.group(2).strip()) if pm else (rest, "")
        am = re.match(r"^\(([^()]*)\)\s*(.*)$", tail)
        if am:
            attr = am.group(1).strip()
            if am.group(2):
                problems.append((source, lineno, "trailing_garbage:" + am.group(2)[:20], line[:60]))
        else:
            attr = tail.strip("() ")
            if tail:
                problems.append((source, lineno, "unbalanced_attr", tail[:40]))
        eq = re.sub(r"\s+", "", eq)
        if eq.count("=") != 1:
            problems.append((source, lineno, "bad_equation", line[:80]))
            continue
        lhs, rhs = eq.split("=")
        if not (TERMS_RE.match(lhs) and TERMS_RE.match(rhs)):
            problems.append((source, lineno, "bad_terms", line[:80]))
            continue
        L, R = parse_side(lhs), parse_side(rhs)
        cl, cr = sum(c for _, c in L), sum(c for _, c in R)
        ok = sum(c * b ** k for b, c in L) == sum(c * b ** k for b, c in R)
        name, date = split_name_date(attr)
        e = {
            "kind": kind, "k": k, "m": mm, "n": nn, "lhs": lhs, "rhs": rhs,
            "name": name, "date": date, "attr": attr, "source": source,
            "line": lineno, "count_ok": (cl == mm and cr == nn), "sum_ok": ok,
            "all_prime_or_1": None,
        }
        if kind == "prime":
            e["all_prime_or_1"] = all(b == 1 or is_prime(b) for b, _ in L + R)
        if not e["count_ok"]:
            problems.append((source, lineno, "count_mismatch(%d,%d)" % (cl, cr), line[:80]))
        if not ok:
            problems.append((source, lineno, "sum_mismatch", line[:80]))
        entries.append(e)
    return entries, problems


def is_prime(n):
    """Miller-Rabin; deterministic below 3.3e24, probabilistic-strong above."""
    if n < 2:
        return False
    small = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41)
    for p in small:
        if n % p == 0:
            return n == p
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in small:
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def fmt_solution(e):
    s = "(%d,%d,%d) %s=%s" % (e["k"], e["m"], e["n"], e["lhs"], e["rhs"])
    if e["attr"]:
        s += " (%s)" % e["attr"]
    return s


# ---------------------------------------------------------------------------- main

def expand_initials(ini, names):
    if not ini:
        return []
    return [names.get(p.strip(), p.strip()) for p in re.split(r"[&,]", ini) if p.strip()]


def name_matches(entry_name, fullnames):
    en = entry_name.lower()
    for fn in fullnames:
        toks = [t for t in re.split(r"[\s.]+", fn.lower()) if len(t) > 2 and t not in ("and",)]
        surname = toks[-1] if toks else fn.lower()
        if surname and surname in en:
            return True
    return False


def pick_solution(cands, fullnames):
    if not cands:
        return None
    good = [e for e in cands if e["sum_ok"] and e["count_ok"]] or cands
    named = [e for e in good if name_matches(e["name"], fullnames)] if fullnames else []
    pool = named or good
    return sorted(pool, key=lambda e: (date_key(e["date"]), e["line"]))[0]


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.path.join(here, "..", "data", "euler_free_fr"))
    args = ap.parse_args()
    d = os.path.abspath(args.data_dir)
    raw = os.path.join(d, "raw")

    pages = {
        "normal": parse_page(os.path.join(raw, "index.html")),
        "high": parse_page(os.path.join(raw, "euler2000_index.html")),
        "prime": parse_page(os.path.join(raw, "prime.htm")),
    }
    names = {}
    for p in pages.values():
        for kk, vv in legend(p).items():
            names.setdefault(kk, vv)

    # last-updated note
    idx = load_html(os.path.join(raw, "index.html"))
    m = re.search(r'href="index\.htm">Index</a></td>\s*<td[^>]*>([^<]+)<', idx)
    print("index.htm 'Last updated' column:", m.group(1).strip() if m else "?")
    news = re.findall(r"<b>(\d+\w*\s+of\s+\w+\s+\d{4}):</b>", idx)
    if news:
        print("latest news item on index:", news[0])

    # databases
    normal_db, p1 = parse_db(os.path.join(raw, "database.txt"), "normal", "database.txt")
    high_db, p2 = parse_db(os.path.join(raw, "datahigh.txt"), "normal", "datahigh.txt")
    prime_db, p3 = parse_db(os.path.join(raw, "dataprime.txt"), "prime", "dataprime.txt")
    problems = p1 + p2 + p3
    db = {"normal": normal_db + high_db, "prime": prime_db}
    by_kmn = {}
    best_n = {}
    for kind, es in db.items():
        for e in es:
            by_kmn.setdefault((kind, e["k"], e["m"], e["n"]), []).append(e)
            if e["sum_ok"] and e["count_ok"]:
                key = (kind, e["k"], e["m"])
                best_n[key] = min(best_n.get(key, 10 ** 9), e["n"])

    # solutions files
    for kind, fn in (("normal", "solutions_normal.txt"), ("prime", "solutions_prime.txt")):
        with open(os.path.join(d, fn), "w", encoding="utf-8") as f:
            for e in db[kind]:
                f.write(fmt_solution(e) + "\n")

    # records
    grids = [("normal", "index.html", extract_grid(pages["normal"])),
             ("normal", "euler2000_index.html", extract_grid(pages["high"])),
             ("prime", "prime.htm", extract_grid(pages["prime"]))]
    cols = ["kind", "k", "m", "n", "total", "discoverer", "date", "solution",
            "initials", "status", "k_best", "source", "db_matches", "db_best_n",
            "sol_verified"]
    rows = []
    covered = set()
    for kind, src, cells in grids:
        for (k, mm, n, ini, bold, status) in cells:
            if status in ("not_needed", "unexplored") and not ini:
                continue
            covered.add((kind, k, mm))
            full = expand_initials(ini, names)
            cands = by_kmn.get((kind, k, mm, n), []) if (ini or status in ("proved", "conjectured")) else []
            e = pick_solution(cands, full)
            rows.append({
                "kind": kind, "k": k, "m": mm, "n": n, "total": mm + n,
                "discoverer": " & ".join(full),
                "date": e["date"] if e else "",
                "solution": fmt_solution(e) if e else "",
                "initials": ini, "status": status, "k_best": int(bool(bold)),
                "source": src, "db_matches": len(cands),
                "db_best_n": best_n.get((kind, k, mm), ""),
                "sol_verified": (int(e["sum_ok"] and e["count_ok"]) if e else ""),
            })
    # (k,m) present in databases but outside the grids (m > 16 etc.)
    for (kind, k, mm), n in sorted(best_n.items()):
        if (kind, k, mm) in covered:
            continue
        e = pick_solution(by_kmn[(kind, k, mm, n)], [])
        rows.append({
            "kind": kind, "k": k, "m": mm, "n": n, "total": mm + n,
            "discoverer": e["name"], "date": e["date"], "solution": fmt_solution(e),
            "initials": "", "status": "db_only", "k_best": "",
            "source": e["source"], "db_matches": len(by_kmn[(kind, k, mm, n)]),
            "db_best_n": n, "sol_verified": int(e["sum_ok"] and e["count_ok"]),
        })
    rows.sort(key=lambda r: (r["kind"], r["k"], r["m"]))
    with open(os.path.join(d, "records.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    # best m+n per k
    with open(os.path.join(d, "min_terms.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["kind", "k", "total", "initials", "discoverer", "status", "source"])
        for kind, key, src in (("normal", "normal", "index.html"),
                               ("normal", "high", "euler2000_index.html"),
                               ("prime", "prime", "prime.htm")):
            for k, (tot, ini, bold, status) in sorted(extract_min_terms(pages[key]).items()):
                w.writerow([kind, k, tot, ini, " & ".join(expand_initials(ini, names)), status, src])

    # report
    for kind in ("normal", "prime"):
        rs = [r for r in rows if r["kind"] == kind]
        g = [r for r in rs if r["status"] != "db_only"]
        print("%s: %d rows (%d grid, %d db_only), k=%d..%d; grid k=%d..%d" % (
            kind, len(rs), len(g), len(rs) - len(g),
            min(r["k"] for r in rs), max(r["k"] for r in rs),
            min(r["k"] for r in g), max(r["k"] for r in g)))
        print("  solutions: %d lines" % len(db[kind]))
    better = [r for r in rows if r["db_best_n"] != "" and r["status"] != "db_only"
              and r["db_best_n"] < r["n"]]
    nosol = [r for r in rows if r["initials"] and not r["solution"]]
    print("grid cells where database has a smaller n:", len(better))
    for r in better[:40]:
        print("   ", r["kind"], (r["k"], r["m"]), "grid n=%d db n=%d" % (r["n"], r["db_best_n"]))
    print("grid cells with discoverer but no exact DB solution:", len(nosol))
    for r in nosol[:60]:
        print("   ", r["kind"], (r["k"], r["m"], r["n"]), r["initials"])
    kinds = {}
    for p in problems:
        kinds[(p[0], p[2].split("(")[0])] = kinds.get((p[0], p[2].split("(")[0]), 0) + 1
    print("database problems:", kinds)
    for p in problems:
        if p[2] not in ("commented",):
            print("   ", p)
    np = [e for e in prime_db if e["all_prime_or_1"] is False]
    print("prime-db entries with a non-prime (non-1) base:", len(np))
    unknown = sorted(set(i for r in rows for i in re.split(r"[&,]", r["initials"]) if i and i not in names))
    print("initials not in legend:", unknown)


if __name__ == "__main__":
    sys.exit(main())
