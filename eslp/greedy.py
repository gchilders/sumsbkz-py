"""Yura's greedy post-processing, as used in sumsbkz 1.3.3.

BKZ minimizes the Euclidean norm, but a record is judged by term counts: m (the
smaller side) first, then the total.  For each relation, keep adding or
subtracting other relations while that strictly lowers the target measure.
"""

from __future__ import annotations

import numpy as np


def side_counts(V: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pos = np.where(V > 0, V, 0).sum(axis=-1)
    neg = np.where(V < 0, -V, 0).sum(axis=-1)
    return pos, neg


def one_side_len(V: np.ndarray) -> np.ndarray:
    pos, neg = side_counts(V)
    return np.minimum(pos, neg)


def total_len(V: np.ndarray) -> np.ndarray:
    return np.abs(V).sum(axis=-1)


def _improve(v: np.ndarray, others: np.ndarray, measure) -> np.ndarray:
    """Greedy descent of measure(v) by +/- the rows of others (sumsbkz's inner loop)."""
    cur = measure(v)
    changed = True
    while changed:
        changed = False
        for j in range(others.shape[0] - 1, -1, -1):
            for sgn in (1, -1):
                while True:
                    t = v + sgn * others[j]
                    mt = measure(t)
                    if mt < cur and t.any():
                        v, cur, changed = t, mt, True
                    else:
                        break
    return v


def yura(R: np.ndarray) -> np.ndarray:
    """Return R plus, for each row, its one-side-greedy and total-greedy improvements.

    Mirrors searchlll(): 'norig' originals, then a one-side pass, then a total pass,
    each starting from the original row and combining with all other originals.
    """
    out = [R]
    for measure in (one_side_len, total_len):
        imp = np.empty_like(R)
        for i in range(R.shape[0] - 1, -1, -1):
            others = np.delete(R, i, axis=0)
            imp[i] = _improve(R[i].copy(), others, measure)
        out.append(imp)
    return np.concatenate(out)
