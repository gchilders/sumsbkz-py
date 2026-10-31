#!/bin/bash
#SBATCH --partition=p_gpu_t4
#SBATCH --qos=q_gpu_t4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48gb
#SBATCH --output=%x-%A_%a.out
# One power per array task (K = array index) on one T4, G6K-GPU built for sm_75 in g6kgpu75/:
#   sbatch --job-name=eslp-t4-N100 --array=19,20,22-23,42-100%6 --time=3:00:00 \
#     --export=ALL,N=100 remote/turing_t4_array.sbatch
# Each task validates its GPU first and exits (freeing the GPU) if validation fails.
module load cuda/12.9.1 gnu12/12.3.0 gmp/6.3.0-gcc-12.3.0
export LD_LIBRARY_PATH=/home/gchilders/apps/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=/home/gchilders/apps/lib:$LIBRARY_PATH
export CPATH=/home/gchilders/apps/include:$CPATH
cd ~/data/euler/sumsbkz-py
source env/bin/activate
export PYTHONPATH=$PWD/g6kgpu75:$PWD NUMBA_CUDA_USE_NVIDIA_BINDING=0 CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=6
K=$SLURM_ARRAY_TASK_ID
mkdir -p results/t4
nvidia-smi --query-gpu=index,name,uuid --format=csv,noheader
bash remote/validate_gpu.sh > results/t4/validate-k$K.log 2>&1 && grep -q IDENTICAL results/t4/validate-k$K.log \
  || { echo "GPU validation failed"; cat results/t4/validate-k$K.log; exit 1; }
echo "start k=$K N=$N $(date)"
taskset -c 0-$(($(nproc --all) - 1)) python sievesearch.py -k $K -N $N --runs 1 --radius 1.3333 --alg gpu --gpu-bucketer bdgl \
  --saturation-ratio 0.375 --db-size-factor 2.77 --multi-bucket 2 --max-nr-buckets 0 \
  --gpu-descent --basis-cache bases --prefetch 1 --threads 5 \
  --seed $((K * 1000 + N)) -o results/t4/k$K-N$N.txt 2>&1 | grep -v "^Warning" | grep -v "^ *[0-9]* *[0-9]* \|^n   It\|max_nr_buckets\|^$"
echo "done k=$K N=$N $(date)"
