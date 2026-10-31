#!/bin/bash
# NRP job entry point: build fpylll/G6K on local disk, then run sievesearch.py.
#   run.sh TAG [sievesearch args...]
# Sources come from /data/eslp/src on the PVC; results go to /data/eslp/results/TAG.{txt,log}.
# THREADS (env) = CPUs requested; VALIDATE=1 checks the G6K sieves against exact SVP first.
set -euo pipefail
TAG=$1; shift
T=${THREADS:-8}
export NUMBA_NUM_THREADS=$T OMP_NUM_THREADS=1
RES=/data/eslp/results
mkdir -p $RES
LOG=$RES/$TAG.log
exec > >(tee -a $LOG) 2>&1
echo "== $TAG start $(date -u) on $(hostname) threads=$T"
grep -m1 "model name" /proc/cpuinfo || true
grep -o -m1 -w "avx2\|avx512f" /proc/cpuinfo | sort -u | tr '\n' ' '; echo

W=/tmp/w
mkdir -p $W/proj
cd $W
apt-get update -qq
apt-get install -y -qq autoconf automake libtool libgmp-dev pkg-config patch >/dev/null
python -m venv /tmp/venv
source /tmp/venv/bin/activate
pip install -q --upgrade pip
pip install -q fpylll cysignals numpy numba scipy cython setuptools
tar xf /data/eslp/src/g6k.tar 2>/dev/null
tar xf /data/eslp/src/eslp.tar -C proj 2>/dev/null
# newer code from ConfigMap gc-eslp-code overrides the PVC copy
[ -f /opt/eslp-code/code.tgz ] && tar xzf /opt/eslp-code/code.tgz -C proj 2>/dev/null
cd g6k
patch -p1 < /data/eslp/src/g6k-macos-arm64.patch
autoreconf -i > /dev/null 2>&1
./configure --with-max-sieving-dim=128 > configure.log
make -j"$T" > make.log 2>&1 || { tail -30 make.log; exit 1; }
pip install -q --no-build-isolation .
cd $W/proj
echo "== build done $(date -u)"

if [ "${VALIDATE:-0}" = 1 ]; then
python - <<'EOF'
import time
import numpy as np
from fpylll import IntegerMatrix, LLL, GSO, Enumeration, FPLLL
from g6k import Siever, SieverParams
from eslp.lattice import bkz
FPLLL.set_random_seed(1)
n = 50
A = IntegerMatrix.random(n, "qary", k=n // 2, bits=20); LLL.reduction(A)
P = IntegerMatrix.from_matrix(bkz(np.array([[A[i, j] for j in range(n)] for i in range(n)], dtype=np.int64), 30).tolist())
M = GSO.Mat(P); M.update_gso()
ex = round(Enumeration(M).enumerate(0, n, M.get_r(0, 0), 0)[0][0])
for alg in ("gauss", "bgj1", "bdgl"):
    g = Siever(A.__copy__(), SieverParams(threads=4, default_sieve=alg))
    g.initialize_local(0, 0, n)
    t = time.time(); g(alg=alg)
    best = min(sum(x * x for x in g.M.B.multiply_left(c)) for c in g.itervalues())
    print(f"validate {alg}: {'OK' if best == ex else 'MISMATCH %d vs %d' % (best, ex)} {time.time() - t:.1f}s", flush=True)
EOF
fi

if [ "${1:-}" = "--py" ]; then
  # any other script of the project, e.g. run.sh TAG --py tools/ablation.py -k 28 ...
  shift
  python "$@" --threads "$T" 2>&1 | grep -v "^Warning"
else
  python sievesearch.py "$@" --threads "$T" -o $RES/$TAG.txt 2>&1 | grep -v "^Warning"
  echo "== $TAG solutions (verified; NEW = below the euler.free.fr table)"
  grep -v "^//" $RES/$TAG.txt || true
fi
echo "== $TAG done $(date -u)"
sleep 5  # let tee flush before the container exits
