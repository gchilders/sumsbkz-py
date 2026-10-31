#!/bin/bash
# Login-node prep on turing (needs internet): uv, Python 3.12 venv with deps, G6K-GPU sources.
#   cd ~/data/euler/sumsbkz-py && bash remote/turing_prep.sh
set -euo pipefail
module load py3/3.13.3-gcc-12.3.0
command -v uv >/dev/null || PIP_CONFIG_FILE=/dev/null python3 -m pip install -q --user uv
export PATH=$HOME/.local/bin:$PATH
uv python install 3.10
[ -d env ] || uv venv --python 3.10 env
uv pip install --python env/bin/python -q fpylll cysignals "numpy<2" "scipy<1.15" "cython<3" setuptools numba numba-cuda
[ -d g6kgpu ] || git clone -q -b python3lwe https://github.com/WvanWoerden/G6K-GPU-Tensor.git g6kgpu
[ -d g6kgpu/parallel-hashmap ] || git clone -q https://github.com/cr-marcstevens/parallel-hashmap g6kgpu/parallel-hashmap
sed -i 's/^SMS ?= .*/SMS ?= 90/' g6kgpu/kernel/Makefile
sed -i 's/-march=native/-march=x86-64-v3/g' g6kgpu/kernel/Makefile
grep -n "^SMS\|march" g6kgpu/kernel/Makefile
env/bin/python -c "import fpylll, numba, numba_cuda; print('deps ok', fpylll.__version__, numba.__version__)"
