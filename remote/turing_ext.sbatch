#!/bin/bash
#SBATCH --job-name=eslp-ext
#SBATCH --partition=p_nsm_compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16gb
#SBATCH --time=01:00:00
#SBATCH --output=%x-%j.out
# Build only the G6K-GPU Cython extension (kernel/libG6K.so already built).
module load cuda/12.9.1 gnu12/12.3.0 gmp/6.3.0-gcc-12.3.0
export LD_LIBRARY_PATH=/home/gchilders/apps/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=/home/gchilders/apps/lib:$LIBRARY_PATH
export CPATH=/home/gchilders/apps/include:$CPATH
cd ~/data/euler/sumsbkz-py
source env/bin/activate
export CUDAPATH=$CUDA_HOME
cd g6kgpu
python setup.py build_ext --inplace > ../g6kgpu-ext.log 2>&1
echo "build_ext exit $?"
tail -3 ../g6kgpu-ext.log
cd ..
PYTHONPATH=$PWD/g6kgpu python -c "import g6k; from g6k import Siever; print('G6K-GPU import ok')"
