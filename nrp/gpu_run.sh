#!/bin/bash
# GPU job entry point (nvidia/cuda:12.8.1-devel-ubuntu22.04 image).
#   gpu_run.sh TAG [sievesearch args...]
# Unpacks the prebuilt G6K-GPU env from the PVC (built on a CPU-only pod, so the GPU
# is never idle waiting on a compile), validates the GPU sieve and GPU descent,
# then runs sievesearch.  Results: /data/eslp/results/TAG.{txt,log}.
set -uo pipefail
TAG=$1; shift
T=${THREADS:-8}
RES=/data/eslp/results
mkdir -p $RES
LOG=$RES/$TAG.log
exec > >(tee -a $LOG) 2>&1
echo "== $TAG start $(date -u) on $(hostname)"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
apt-get update -qq > /dev/null && apt-get install -y -qq python3 libgmp10 libmpfr6 > /dev/null
cd / && tar xzf /data/eslp/gpu/g6kgpu-env.tgz
mkdir -p /w/proj && cd /w/proj
tar xf /data/eslp/src/eslp.tar 2>/dev/null
[ -f /opt/eslp-code/code.tgz ] && tar xzf /opt/eslp-code/code.tgz 2>/dev/null
source /opt/venv/bin/activate
export PYTHONPATH=/build/g6kgpu:/w/proj CUDA_HOME=/usr/local/cuda NUMBA_CUDA_USE_NVIDIA_BINDING=0
export NUMBA_NUM_THREADS=$T OMP_NUM_THREADS=1
echo "== env ready $(date -u)"

if [ "${VALIDATE:-1}" = 1 ]; then
python - <<'EOF' || { echo "== VALIDATION FAILED"; exit 1; }
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
        good = best == ex
        ok &= good
        print(f"validate G6K-GPU n={n} {bucketer}: {'OK' if good else 'MISMATCH %d vs %d' % (best, ex)} db={len(g)} {time.time() - t:.1f}s", flush=True)
assert ok
EOF
out=$(python -m eslp.descent_gpu 2>&1); echo "$out" | tail -3
echo "$out" | grep -q IDENTICAL || { echo "== GPU DESCENT MISMATCH"; exit 1; }
fi

echo "== search start $(date -u)"
python sievesearch.py "$@" --threads "$T" -o $RES/$TAG.txt 2>&1 | grep -v "^Warning"
echo "== $TAG solutions (verified; NEW = below the euler.free.fr table)"
grep -v "^//" $RES/$TAG.txt || true
echo "== $TAG done $(date -u)"
sleep 5
