#!/bin/bash
# Login-node prep for the CPU (mainline G6K) environment: envcpu + patched g6k source.
set -euo pipefail
cd ~/data/euler/sumsbkz-py
export PATH=$HOME/.local/bin:$PATH
uv python install 3.12
[ -d envcpu ] || uv venv --python 3.12 envcpu
uv pip install --python envcpu/bin/python -q fpylll cysignals numpy scipy cython setuptools numba
[ -d g6kcpu ] || { tar xf remote/g6kcpu.tar; (cd g6kcpu && patch -p1 < ../patches/g6k-macos-arm64.patch); }
envcpu/bin/python -c "import fpylll, numba, numpy; print('envcpu ok', fpylll.__version__, numpy.__version__)"
