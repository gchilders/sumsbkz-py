#!/bin/bash
#SBATCH --job-name=eslp-build
#SBATCH --partition=p_nsm_compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32gb
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
# CPU-only compile of G6K-GPU (sm_90, max sieving dim 160). No GPU requested.
module load cuda/12.9.1 gnu12/12.3.0 gmp/6.3.0-gcc-12.3.0
export LD_LIBRARY_PATH=/home/gchilders/apps/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=/home/gchilders/apps/lib:$LIBRARY_PATH
export CPATH=/home/gchilders/apps/include:$CPATH
cd ~/data/euler/sumsbkz-py
source env/bin/activate
export CUDAPATH=$CUDA_HOME   # rebuild.sh defaults to /usr/local/cuda
cd g6kgpu
./rebuild.sh -f -y -m 160 -j 16 > ../g6kgpu-build.log 2>&1
echo "rebuild exit $?"
tail -5 ../g6kgpu-build.log
cd ..
PYTHONPATH=$PWD/g6kgpu python -c "import g6k; from g6k import Siever; print('G6K-GPU import ok')"
