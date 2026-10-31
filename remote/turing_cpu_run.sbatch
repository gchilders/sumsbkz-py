#!/bin/bash
#SBATCH --partition=p_nsm_compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --output=%x-%j.out
# CPU sieve + descent for one power:  sbatch --job-name=eslp-k28-N100 --mem=64gb --time=24:00:00 \
#   --export=ALL,K=28,N=100,RAD=1.3333 remote/turing_cpu_run.sbatch
module load gnu12/12.3.0 gmp/6.3.0-gcc-12.3.0
export LD_LIBRARY_PATH=/home/gchilders/apps/lib:$LD_LIBRARY_PATH
cd ~/data/euler/sumsbkz-py
source envcpu/bin/activate
export PYTHONPATH=$PWD/g6kcpu:$PWD OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=32
mkdir -p results/cpu
taskset -c 0-$(($(nproc --all) - 1)) python sievesearch.py -k $K -N $N --runs 1 --radius ${RAD:-1.3333} --alg ${ALG:-bdgl} --threads 32 \
  --basis-cache bases --seed $((K * 1000 + N)) -o results/cpu/k$K-N$N.txt 2>&1 | grep -v "^Warning"
