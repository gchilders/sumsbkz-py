"""Parsing, formatting, verification and record frontiers for equal sums of like powers.

A solution is stored as a dict {base: coefficient}; positive coefficients are one
side, negative the other.  The text format is the one used by the euler.free.fr
tables and the original sumsbkz/NewLC/ResultLC programs:

    (k,m,n) 64+62*2+...=63+61*3+... (Name, date)

where m <= n are the left/right term counts with multiplicity and "b*c" means
c copies of b^k.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field

_HEAD = re.compile(r"^\s*\((\d+),(\d+),(\d+)\)\s*(.*)$")
_TERM = re.compile(r"^(\d+)(?:\*(\d+))?$")


@dataclass
class Solution:
    k: int
    coeffs: dict[int, int]
    note: str = ""
    raw_mn: tuple[int, int] | None = field(default=None, compare=False)

    @property
    def counts(self) -> tuple[int, int]:
        pos = sum(c for c in self.coeffs.values() if c > 0)
        neg = -sum(c for c in self.coeffs.values() if c < 0)
        return pos, neg

    @property
    def m(self) -> int:
        return min(self.counts)

    @property
    def n(self) -> int:
        return max(self.counts)

    @property
    def total(self) -> int:
        return sum(self.counts)

    def is_valid(self) -> bool:
        return bool(self.coeffs) and sum(c * b**self.k for b, c in self.coeffs.items()) == 0

    def is_prime(self) -> bool:
        return all(_is_prime(b) for b in self.coeffs)

    def normalized(self) -> "Solution":
        """Left side = the side with fewer terms (ties: side holding the largest base)."""
        pos, neg = self.counts
        flip = pos > neg or (pos == neg and self.coeffs[max(self.coeffs)] < 0)
        c = {b: (-v if flip else v) for b, v in self.coeffs.items() if v}
        return Solution(self.k, c, self.note)

    def format(self) -> str:
        s = self.normalized()
        def side(sign):
            terms = [(b, abs(c)) for b, c in sorted(s.coeffs.items(), reverse=True) if c * sign > 0]
            return "+".join(f"{b}*{c}" if c > 1 else f"{b}" for b, c in terms)
        out = f"({s.k},{s.m},{s.n}) {side(+1)}={side(-1)}"
        return f"{out} {s.note}".rstrip() if s.note else out


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13):
        if n % p == 0:
            return n == p
    d = 17
    while d * d <= n:
        if n % d == 0:
            return False
        d += 2
    return True


def _parse_side(text: str) -> dict[int, int]:
    out: dict[int, int] = defaultdict(int)
    text = text.strip()
    if not text:
        return out
    for tok in text.split("+"):
        mt = _TERM.match(tok.strip())
        if not mt:
            raise ValueError(f"bad term {tok!r}")
        out[int(mt.group(1))] += int(mt.group(2) or 1)
    return out


def parse_line(line: str) -> Solution | None:
    """Parse one '(k,m,n) lhs=rhs [note]' line.  Returns None for non-solution lines."""
    mh = _HEAD.match(line)
    if not mh:
        return None
    k, m, n, rest = int(mh.group(1)), int(mh.group(2)), int(mh.group(3)), mh.group(4)
    eq, _, note = rest.partition(" ")
    # Some archive lines have spaces inside the equation; fall back to splitting on '('.
    if "=" not in eq:
        eq, _, note = rest.partition("(")
        note = f"({note}" if note else ""
        eq = eq.replace(" ", "")
    lhs, _, rhs = eq.partition("=")
    coeffs: dict[int, int] = defaultdict(int)
    for b, c in _parse_side(lhs).items():
        coeffs[b] += c
    for b, c in _parse_side(rhs).items():
        coeffs[b] -= c
    coeffs = {b: c for b, c in coeffs.items() if c}
    return Solution(k, coeffs, note.strip(), raw_mn=(m, n))


def read_solutions(path: str, encoding: str = "latin-1") -> list[Solution]:
    sols = []
    with open(path, encoding=encoding) as f:
        for line in f:
            try:
                s = parse_line(line)
            except ValueError:
                continue
            if s is not None and s.coeffs:
                sols.append(s)
    return sols


class Frontier:
    """Best known n for each (k, m): the record table."""

    def __init__(self) -> None:
        self.best: dict[tuple[int, int], int] = {}

    def add(self, s: Solution) -> bool:
        """Record s; return True if it ties or beats the current entry for (k, m)."""
        key = (s.k, s.m)
        cur = self.best.get(key)
        if cur is None or s.n <= cur:
            self.best[key] = s.n if cur is None else min(cur, s.n)
            return True
        return False

    def get(self, k: int, m: int) -> int | None:
        return self.best.get((k, m))

    def min_total(self, k: int) -> int | None:
        tots = [m + n for (kk, m), n in self.best.items() if kk == k]
        return min(tots) if tots else None

    def ms(self, k: int) -> list[int]:
        return sorted(m for (kk, m) in self.best if kk == k)

    @classmethod
    def from_solutions(cls, sols) -> "Frontier":
        f = cls()
        for s in sols:
            f.add(s)
        return f
