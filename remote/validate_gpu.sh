#!/bin/bash
# Same checks as the NRP GPU job: G6K-GPU sieve vs exact SVP, GPU descent vs CPU descent.
set -euo pipefail
python - <<'PY'
import time
import numpy as np
from fpylll import IntegerMatrix, LLL, GSO, Enumeration, FPLLL
from g6k import Siever, SieverParams
from eslp.lattice import bkz
ok = True
FPLLL.set_random_seed(1)
for n in (50, 52):
    A = IntegerMatrix.random(n, "qary", k=n // 2, bits=20); LLL.reduction(A)
    P = IntegerMatrix.from_matrix(bkz(np.array([[A[i, j] for j in range(n)] for i in range(n)], dtype=np.int64), 30).tolist())
    M = GSO.Mat(P); M.update_gso()
    ex = round(Enumeration(M).enumerate(0, n, M.get_r(0, 0), 0)[0][0])
    for bucketer in ("bgj1", "bdgl"):
        g = Siever(A.__copy__(), SieverParams(threads=4, gpus=1, gpu_bucketer=bucketer))
        g.initialize_local(0, 0, n)
        t = time.time(); g(alg="gpu")
        best = min(sum(x * x for x in g.M.B.multiply_left(c)) for c in g.itervalues())
        ok &= best == ex
        print(f"G6K-GPU n={n} {bucketer}: {'OK' if best == ex else 'MISMATCH'} {time.time() - t:.1f}s", flush=True)
assert ok, "sieve validation failed"
PY
python -m eslp.descent_gpu | tail -1
