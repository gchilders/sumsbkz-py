#!/bin/bash
# Build the GPU stack (fpylll + G6K-GPU-Tensor python3lwe + numba-cuda) in ./env and ./g6kgpu.
#   bash remote/setup_gpu.sh [SMS] [MAXDIM]      e.g.  bash remote/setup_gpu.sh 90 160   (H200)
# Needs: CUDA toolkit (nvcc) on PATH, python3 >= 3.10 with venv + headers, g++, git, libgmp-dev,
# libmpfr-dev, internet for pip/git.  Run from the sumsbkz-py directory.
set -euo pipefail
SMS=${1:-90}; MAXDIM=${2:-160}
ROOT=$(pwd)
JOBS=$(nproc)
command -v nvcc >/dev/null || { echo "nvcc not on PATH"; exit 1; }
python3 -m venv env
. env/bin/activate
pip install -q --upgrade pip
pip install -q fpylll cysignals numpy scipy "cython<3" setuptools numba numba-cuda
[ -d g6kgpu ] || git clone -q -b python3lwe https://github.com/WvanWoerden/G6K-GPU-Tensor.git g6kgpu
cd g6kgpu
[ -d parallel-hashmap ] || git clone -q https://github.com/cr-marcstevens/parallel-hashmap
sed -i "s/^SMS ?= .*/SMS ?= $SMS/" kernel/Makefile
grep -n "^SMS" kernel/Makefile
./rebuild.sh -f -y -m "$MAXDIM" -j "$JOBS" > "$ROOT/g6kgpu-build.log" 2>&1 || { tail -40 "$ROOT/g6kgpu-build.log"; exit 1; }
cd "$ROOT"
PYTHONPATH="$ROOT/g6kgpu" python -c "import g6k; from g6k import Siever; print('G6K-GPU import ok')"
echo "export PYTHONPATH=$ROOT/g6kgpu:$ROOT NUMBA_CUDA_USE_NVIDIA_BINDING=0 CUDA_HOME=\${CUDA_HOME:-/usr/local/cuda}" > env.sh
echo ". $ROOT/env/bin/activate" >> env.sh
echo "setup done: source env.sh, then bash remote/validate_gpu.sh"
