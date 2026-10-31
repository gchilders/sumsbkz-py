"""Compiled ResultLC/Yura descent: walk relations downhill in w*m + total using a move set.

Every vector visited is a valid relation (a sum of relations), so the best n
seen for each m is tracked along every path, as ResultLC printed every
improvement it passed through.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

MMAX = 96  # track frontier for m < MMAX


@njit(cache=True, inline="always")
def _counts(v):
    p = 0
    q = 0
    for i in range(v.shape[0]):
        x = v[i]
        if x > 0:
            p += x
        else:
            q -= x
    return p, q


@njit(cache=True)
def _descend_one(v0, moves, w, max_steps, best_n, best_vec, p_off, q_off, record):
    N = v0.shape[0]
    K = moves.shape[0]
    v = v0.copy()
    t = np.empty(N, np.int32)
    p, q = _counts(v)
    p += p_off
    q += q_off
    cur = w * min(p, q) + p + q
    m = min(p, q)
    if record != 2 and m < MMAX and max(p, q) < best_n[m]:
        best_n[m] = max(p, q)
        best_vec[m, :] = v
    steps = 0
    for _ in range(max_steps):
        best_f = cur
        best_j = -1
        best_s = 0
        for j in range(K):
            for s in (1, -1):
                pp = p_off
                qq = q_off
                nz = False
                for i in range(N):
                    x = v[i] + s * moves[j, i]
                    if x > 0:
                        pp += x
                        nz = True
                    elif x < 0:
                        qq -= x
                        nz = True
                if not nz:
                    continue
                mm = pp if pp < qq else qq
                f = w * mm + pp + qq
                if f < best_f:
                    best_f = f
                    best_j = j
                    best_s = s
                # every candidate is a relation: record it if it improves the frontier
                # (record=0; record=1 keeps only the accepted path, record=2 only the end point)
                if record == 0 and mm < MMAX:
                    nn = pp if pp > qq else qq
                    if nn < best_n[mm]:
                        best_n[mm] = nn
                        for i in range(N):
                            best_vec[mm, i] = v[i] + s * moves[j, i]
        if best_j < 0:
            break
        for i in range(N):
            t[i] = v[i] + best_s * moves[best_j, i]
        v[:] = t
        cur = best_f
        steps += 1
        if record == 1:
            p, q = _counts(v)
            p += p_off
            q += q_off
            m = min(p, q)
            if m < MMAX and max(p, q) < best_n[m]:
                best_n[m] = max(p, q)
                best_vec[m, :] = v
    if record == 2:
        p, q = _counts(v)
        p += p_off
        q += q_off
        m = min(p, q)
        if m < MMAX and max(p, q) < best_n[m]:
            best_n[m] = max(p, q)
            best_vec[m, :] = v
    return steps


@njit(parallel=True, cache=True)
def _descend_all(starts, moves, weights, max_steps, best_n, best_vec, steps, p_off, q_off, record):
    S = starts.shape[0]
    W = weights.shape[0]
    for t in prange(S * W):
        s = t // W
        w = weights[t % W]
        steps[t] = _descend_one(starts[s], moves, w, max_steps, best_n[t], best_vec[t], p_off, q_off, record)


# number of accepted steps of every (start, weight) descent in the last call, for statistics
LAST_STEPS = np.zeros(0, dtype=np.int64)


def descend(starts: np.ndarray, moves: np.ndarray, weights=(0, 1, 2, 4, 8, 16, 1000),
            max_steps: int = 500, all_tasks: bool = False, p_off: int = 0, q_off: int = 0,
            record: int = 0):
    """Run the descent from every start under every weight; return {m: (n, vector)}.

    Each candidate move v +/- moves[j] examined is itself a relation, so the
    frontier is updated over the whole neighbourhood, not just the path taken.
    With all_tasks=True, return instead a list of (m, n, vector) holding the best
    relation for each m found by each (start, weight) descent, so that ties
    between descents are kept.  p_off and q_off are fixed terms on the positive and
    negative side that the moves never change (tools/complete.py); they are added to
    the counts of every vector.  record=0 (the method of the paper) updates the frontier
    with every candidate examined; record=1 only with the accepted path and record=2 only
    with the end point of each descent (for the ablation of tools/moveset.py).
    """
    global LAST_STEPS
    starts = np.ascontiguousarray(starts, dtype=np.int32)
    moves = np.ascontiguousarray(moves, dtype=np.int32)
    weights = np.asarray(weights, dtype=np.int64)
    T = len(starts) * len(weights)
    N = starts.shape[1]
    best_n = np.full((T, MMAX), np.iinfo(np.int32).max, dtype=np.int64)
    best_vec = np.zeros((T, MMAX, N), dtype=np.int32)
    steps = np.zeros(T, dtype=np.int64)
    _descend_all(starts, moves, weights, max_steps, best_n, best_vec, steps, p_off, q_off, record)
    LAST_STEPS = steps
    if all_tasks:
        t, m = np.nonzero(best_n < np.iinfo(np.int32).max)
        return [(int(mm), int(best_n[tt, mm]), best_vec[tt, mm].copy()) for tt, mm in zip(t, m)]
    out: dict[int, tuple[int, np.ndarray]] = {}
    for m in range(MMAX):
        t = int(np.argmin(best_n[:, m]))
        if best_n[t, m] < np.iinfo(np.int32).max:
            out[m] = (int(best_n[t, m]), best_vec[t, m].copy())
    return out
