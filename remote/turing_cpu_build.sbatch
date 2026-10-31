#!/bin/bash
#SBATCH --job-name=eslp-cpubuild
#SBATCH --partition=p_nsm_compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32gb
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
# Build mainline G6K (CPU sieves, portable AVX2) into envcpu, then validate against exact SVP.
module load gnu12/12.3.0 gmp/6.3.0-gcc-12.3.0
export LD_LIBRARY_PATH=/home/gchilders/apps/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=/home/gchilders/apps/lib:$LIBRARY_PATH
export CPATH=/home/gchilders/apps/include:$CPATH
cd ~/data/euler/sumsbkz-py
source envcpu/bin/activate
cd g6kcpu
autoreconf -i > ../g6kcpu-build.log 2>&1
CXXFLAGS="-O3 -march=x86-64-v3" CFLAGS="-O3 -march=x86-64-v3" ./configure --with-max-sieving-dim=128 --disable-native >> ../g6kcpu-build.log 2>&1
make -j16 >> ../g6kcpu-build.log 2>&1 || { tail -30 ../g6kcpu-build.log; exit 1; }
python setup.py build_ext --inplace >> ../g6kcpu-build.log 2>&1 || { tail -30 ../g6kcpu-build.log; exit 1; }
cd ..
export PYTHONPATH=$PWD/g6kcpu:$PWD
python - <<'PY'
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
for alg in ("bgj1", "bdgl"):
    g = Siever(A.__copy__(), SieverParams(threads=8, default_sieve=alg)); g.initialize_local(0, 0, n); g(alg=alg)
    best = min(sum(x * x for x in g.M.B.multiply_left(c)) for c in g.itervalues())
    print(f"validate cpu {alg}: {'OK' if best == ex else 'MISMATCH'}", flush=True)
PY
