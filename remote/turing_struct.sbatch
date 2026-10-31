#!/bin/bash
#SBATCH --partition=p_nsm_compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64gb
#SBATCH --output=%x-%A_%a.out
# Structured congruence solutions (tools/structsolve.py), one (k, a) per array task from
# remote/struct_tasks.txt (line = array index): "k a tries keep [p]"; M = p^a (p = 2 by default).
#   sbatch --job-name=eslp-struct --array=1-11 --time=12:00:00 remote/turing_struct.sbatch
module load gnu12/12.3.0 gmp/6.3.0-gcc-12.3.0
export LD_LIBRARY_PATH=/home/gchilders/apps/lib:$LD_LIBRARY_PATH
cd ~/data/euler/sumsbkz-py
source envcpu/bin/activate
export PYTHONPATH=$PWD/g6kcpu:$PWD OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1
set -- $(sed -n "${SLURM_ARRAY_TASK_ID}p" remote/struct_tasks.txt)
k=$1; a=$2; tries=$3; keep=$4; p=${5:-2}
mkdir -p results/congsolve
taskset -c 0-$(($(nproc --all) - 1)) python tools/structsolve.py -k $k -p $p -a $a -C 48 --threads 32 \
  --tries $tries --keep $keep -o results/congsolve/struct-k$k-M$((p ** a)).txt 2>&1 | grep -v -i "warning"
