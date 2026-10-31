#!/bin/bash
#SBATCH --job-name=eslp-gpubench
#SBATCH --partition=p_gpu_h200
#SBATCH --qos=q_gpu_h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=60
#SBATCH --mem=400gb
#SBATCH --time=03:00:00
#SBATCH --output=%x-%j.out
# Four G6K-GPU configurations side by side (one per GPU) on the same lattice, k=28, N=100.
module load cuda/12.9.1 gnu12/12.3.0 gmp/6.3.0-gcc-12.3.0
export LD_LIBRARY_PATH=/home/gchilders/apps/lib:$LD_LIBRARY_PATH
cd ~/data/euler/sumsbkz-py
source env/bin/activate
export PYTHONPATH=$PWD/g6kgpu:$PWD NUMBA_CUDA_USE_NVIDIA_BINDING=0 CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=15
mkdir -p results/bench
common="-k 28 -N 100 --runs 1 --radius 1.3333 --alg gpu --gpu-descent --basis-cache bases --threads 15 --seed 777"
python sievesearch.py --bases-only --prefetch 1 -k 28 -N 100 --basis-cache bases -o /dev/null
run() { local g=$1 tag=$2; shift 2; ( export CUDA_VISIBLE_DEVICES=$g; s=$(date +%s); timeout 9000 python sievesearch.py $common "$@" -o results/bench/$tag.txt > results/bench/$tag.log 2>&1; echo "$tag exit=$? wall=$(( $(date +%s) - s ))s :: $(grep '^k=' results/bench/$tag.log | tail -1)" ); }
run 0 default --gpu-bucketer bdgl &
run 1 authors --gpu-bucketer bdgl --saturation-ratio 0.375 --db-size-factor 2.77 --multi-bucket 2 --max-nr-buckets 0 &
run 2 authors-bgj1 --gpu-bucketer bgj1 --saturation-ratio 0.375 --db-size-factor 2.77 --multi-bucket 2 --max-nr-buckets 0 &
run 3 authors-r128 --gpu-bucketer bdgl --saturation-ratio 0.375 --db-size-factor 2.77 --multi-bucket 2 --max-nr-buckets 0 --db-cap 8000000 &
wait
