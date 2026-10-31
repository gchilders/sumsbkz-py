#!/bin/bash
#SBATCH --job-name=eslp-t4bench
#SBATCH --partition=p_gpu_t4
#SBATCH --qos=q_gpu_t4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64gb
#SBATCH --time=06:00:00
#SBATCH --output=%x-%j.out
# Build G6K-GPU for the T4 (sm_75) in g6kgpu75/ (the H200 build in g6kgpu/ is untouched),
# validate it on the GPU, then time one N=100 power (k=24, cached basis) for comparison
# with the H200 (about 12 min of sieving) and with bdgl on 32 CPU cores (33 min).
module load cuda/12.9.1 gnu12/12.3.0 gmp/6.3.0-gcc-12.3.0
export LD_LIBRARY_PATH=/home/gchilders/apps/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=/home/gchilders/apps/lib:$LIBRARY_PATH
export CPATH=/home/gchilders/apps/include:$CPATH
cd ~/data/euler/sumsbkz-py
source env/bin/activate
export CUDAPATH=$CUDA_HOME
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
grep -m1 "model name" /proc/cpuinfo; grep -o -m1 -w avx2 /proc/cpuinfo
if [ ! -f g6kgpu75/g6k/siever.cpython-310-x86_64-linux-gnu.so ]; then
  rm -rf g6kgpu75 && cp -r g6kgpu g6kgpu75
  sed -i 's/^SMS ?= .*/SMS ?= 75/' g6kgpu75/kernel/Makefile
  (cd g6kgpu75 && ./rebuild.sh -f -y -m 160 -j 12 > ../g6kgpu75-build.log 2>&1); echo "rebuild exit $?"
  tail -3 g6kgpu75-build.log
fi
export PYTHONPATH=$PWD/g6kgpu75:$PWD NUMBA_CUDA_USE_NVIDIA_BINDING=0 CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=12
bash remote/validate_gpu.sh || { echo "T4 validation failed"; exit 1; }
mkdir -p results/t4
echo "start $(date)"; time python sievesearch.py -k 24 -N 100 --runs 1 --radius 1.3333 --alg gpu --gpu-bucketer bdgl \
  --saturation-ratio 0.375 --db-size-factor 2.77 --multi-bucket 2 --max-nr-buckets 0 \
  --gpu-descent --basis-cache bases --prefetch 1 --threads 11 \
  --seed 24100 -o results/t4/k24-N100.txt 2>&1 | grep -v "^Warning" | grep -v "^ *[0-9]* *[0-9]* " | tail -30
echo "done $(date)"
