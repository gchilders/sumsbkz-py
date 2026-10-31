#!/bin/bash
#SBATCH --job-name=eslp-bases
#SBATCH --partition=p_nsm_compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64gb
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
# CPU-only: precompute BKZ-20 reduced relation bases so GPU jobs start sieving immediately.
cd ~/data/euler/sumsbkz-py
source env/bin/activate
export PYTHONPATH=$PWD OMP_NUM_THREADS=1
for N in 112 120 128 140; do
  python sievesearch.py -k 21 32 -N $N --basis-cache bases --bases-only --prefetch 12 -o /dev/null &
done
wait
ls bases | wc -l
