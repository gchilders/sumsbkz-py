"""GPU version of eslp/descent.py (numba.cuda).

Same walk as the CPU descent: every row (a start vector paired with a weight w)
moves to the best v +/- moves[j] in w*m + total while that strictly improves,
and every candidate examined updates the global (m -> fewest n) frontier.
Ties break the same way (smallest j, then + before -), so on identical input the
two versions return identical frontiers.

One kernel launch evaluates all active rows against all moves for one step:
grid (ceil(K / TPB), rows); each thread scores one move in both signs.  The
best move per row and the frontier are packed into 64-bit keys and reduced
with atomicMin.  The host applies the chosen moves and decodes frontier hits.

Test without a GPU:  NUMBA_ENABLE_CUDASIM=1 python -m eslp.descent_gpu
"""

from __future__ import annotations

import numpy as np
from numba import cuda, uint64, int32

MMAX = 96
MAXN = 256          # max coordinates per vector (joint descents use C + N up to 256)
TPB = 256           # threads per block
U64MAX = np.uint64(0xFFFFFFFFFFFFFFFF)

# frontier key: n (20 bits) | row (12 bits) | j (31 bits) | sign (1 bit)
_ROW_BITS, _J_BITS = 12, 31


RT = 8              # rows per block: each move column is read once per RT rows


@cuda.jit
def _eval_kernel(V, active, MT, w, K, N, R, best_move, front, front_n, p_off, q_off):
    r0 = cuda.blockIdx.y * RT
    tid = cuda.threadIdx.x
    v = cuda.shared.array((RT, MAXN), int32)
    red = cuda.shared.array((RT, TPB), uint64)
    any_active = False
    for t in range(RT):
        if r0 + t < R and active[r0 + t] != 0:
            any_active = True
    if not any_active:
        return
    for t in range(RT):
        for i in range(tid, N, TPB):
            v[t, i] = V[r0 + t, i] if r0 + t < R else 0
    cuda.syncthreads()

    j = cuda.blockIdx.x * TPB + tid
    p1 = cuda.local.array(RT, int32)
    q1 = cuda.local.array(RT, int32)
    p2 = cuda.local.array(RT, int32)
    q2 = cuda.local.array(RT, int32)
    for t in range(RT):
        p1[t] = p_off
        q1[t] = q_off
        p2[t] = p_off
        q2[t] = q_off
    if j < K:
        for i in range(N):
            mi = int32(MT[i, j])
            for t in range(RT):
                a = v[t, i] + mi
                b = v[t, i] - mi
                if a > 0:
                    p1[t] += a
                else:
                    q1[t] -= a
                if b > 0:
                    p2[t] += b
                else:
                    q2[t] -= b
    for t in range(RT):
        r = r0 + t
        best = U64MAX
        if j < K and r < R and active[r] != 0:
            wr = w[r]
            for sb in range(2):
                if sb == 0:
                    pp = p1[t]
                    qq = q1[t]
                else:
                    pp = p2[t]
                    qq = q2[t]
                if pp + qq == p_off + q_off:
                    continue
                m = pp if pp < qq else qq
                n = pp if pp > qq else qq
                f = wr * m + pp + qq
                key = (uint64(f) << uint64(32)) | uint64(j * 2 + sb)
                if key < best:
                    best = key
                if m < MMAX and n < front_n[m]:
                    fk = ((uint64(n) << uint64(_ROW_BITS + _J_BITS + 1))
                          | (uint64(r) << uint64(_J_BITS + 1))
                          | (uint64(j) << uint64(1)) | uint64(sb))
                    cuda.atomic.min(front, m, fk)
        red[t, tid] = best
    cuda.syncthreads()
    s = TPB // 2
    while s > 0:
        if tid < s:
            for t in range(RT):
                if red[t, tid + s] < red[t, tid]:
                    red[t, tid] = red[t, tid + s]
        cuda.syncthreads()
        s //= 2
    if tid < RT:
        r = r0 + tid
        if r < R and red[tid, 0] != U64MAX:
            cuda.atomic.min(best_move, r, red[tid, 0])


def _counts(V):
    p = np.where(V > 0, V, 0).sum(-1)
    q = np.where(V < 0, -V, 0).sum(-1)
    return p, q


def descend_gpu(starts: np.ndarray, moves: np.ndarray, weights=(0, 1, 2, 4, 8, 16, 1000),
                max_steps: int = 500, verbose: bool = False, p_off: int = 0,
                q_off: int = 0) -> dict[int, tuple[int, np.ndarray]]:
    """Drop-in replacement for eslp.descent.descend running on the GPU.

    p_off and q_off are terms on the positive and negative side that the moves never
    change (the fixed part of a completion, tools/complete.py); they are added to the
    counts p and q of every vector, and the frontier is kept in these full counts.
    """
    S, N = starts.shape
    K = moves.shape[0]
    if N > MAXN:
        raise ValueError(f"at most {MAXN} coordinates")
    if K >= 1 << _J_BITS or S * len(weights) >= 1 << _ROW_BITS:
        raise ValueError("too many moves or rows for the key packing")
    W = np.asarray(weights, dtype=np.int32)
    V = np.repeat(starts.astype(np.int32), len(W), axis=0)          # row = start x weight
    wr = np.tile(W, S)
    R = len(V)
    Mh = np.ascontiguousarray(moves.astype(np.int16))
    MT = cuda.to_device(np.ascontiguousarray(Mh.T))
    d_w = cuda.to_device(wr)

    # frontier from the starts themselves
    best: dict[int, tuple[int, np.ndarray]] = {}
    p, q = _counts(V)
    p, q = p + p_off, q + q_off
    for i in range(R):
        m, n = int(min(p[i], q[i])), int(max(p[i], q[i]))
        if m < MMAX and (m not in best or n < best[m][0]):
            best[m] = (n, V[i].copy())
    cur = wr.astype(np.int64) * np.minimum(p, q) + p + q
    active = np.ones(R, np.uint8)

    front_n = np.full(MMAX, 1 << 19, np.int32)
    for m, (n, _) in best.items():
        front_n[m] = n
    grid = ((K + TPB - 1) // TPB, (R + RT - 1) // RT)
    for step in range(max_steps):
        if not active.any():
            break
        d_V = cuda.to_device(V)
        d_act = cuda.to_device(active)
        d_best = cuda.to_device(np.full(R, U64MAX, np.uint64))
        d_front = cuda.to_device(np.full(MMAX, U64MAX, np.uint64))
        d_fn = cuda.to_device(front_n)
        _eval_kernel[grid, TPB](d_V, d_act, MT, d_w, K, N, R, d_best, d_front, d_fn,
                                np.int32(p_off), np.int32(q_off))
        bm = d_best.copy_to_host()
        fr = d_front.copy_to_host()
        # frontier hits, decoded against V before this step's moves are applied
        for m in np.nonzero(fr != U64MAX)[0]:
            k = int(fr[m])
            n = k >> (_ROW_BITS + _J_BITS + 1)
            r = (k >> (_J_BITS + 1)) & ((1 << _ROW_BITS) - 1)
            j = (k >> 1) & ((1 << _J_BITS) - 1)
            sgn = -1 if k & 1 else 1
            if n < front_n[m]:
                front_n[m] = n
                best[int(m)] = (int(n), V[r] + sgn * Mh[j].astype(np.int32))
        # apply each row's best move if it improves
        f = (bm >> np.uint64(32)).astype(np.int64)
        ok = (bm != U64MAX) & (f < cur) & (active == 1)
        idx = np.nonzero(ok)[0]
        if len(idx):
            low = (bm[idx] & np.uint64(0xFFFFFFFF)).astype(np.int64)
            j = low >> 1
            sgn = np.where(low & 1, -1, 1)
            V[idx] += sgn[:, None] * Mh[j].astype(np.int32)
            cur[idx] = f[idx]
        active[~ok] = 0
        if verbose:
            print(f"step {step}: active {int(active.sum())}/{R}", flush=True)
    return {m: (n, v.astype(np.int64)) for m, (n, v) in sorted(best.items())}


if __name__ == "__main__":
    # Equivalence test against the CPU descent on a small random relation lattice.
    import time
    from eslp.descent import descend
    from eslp.lattice import bkz, relation_basis, verify
    xs = list(range(1, 25))
    R = bkz(relation_basis(xs, 12), 10)
    rng = np.random.default_rng(1)
    moves = np.concatenate([R, R[:-1] + R[1:], R[:-1] - R[1:]])
    starts = moves[rng.permutation(len(moves))[:6]]
    W = (0, 2, 1000)
    t = time.time(); a = descend(starts, moves, W, max_steps=50); ta = time.time() - t
    t = time.time(); b = descend_gpu(starts, moves, W, max_steps=50); tb = time.time() - t
    sa = {m: n for m, (n, _) in a.items()}
    sb = {m: n for m, (n, _) in b.items()}
    assert all(verify(xs, 12, v) for _, v in b.values()), "GPU frontier vector is not a relation"
    print("cpu ", sa, f"{ta:.2f}s")
    print("gpu ", sb, f"{tb:.2f}s")
    print("IDENTICAL" if sa == sb else "DIFFERENT")
