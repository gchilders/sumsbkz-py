#!/bin/bash
#SBATCH --partition=p_gpu_t4
#SBATCH --qos=q_gpu_t4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=%x-%j.out
# A pool of T4 workers completing J. Wroblewski's congruence solutions (tools/complete.py),
# one power at a time from the shared list remote/t4c_queue.txt (one "k N" per line; flock-protected, shared by all pool jobs).
#   sbatch --job-name=eslp-t4pool --gpus-per-node=5 --cpus-per-task=30 --mem=200gb \
#     --time=12:00:00 remote/turing_t4_pool.sbatch
# Each worker validates its GPU first and stops if validation fails; a worker exits when
# the list is empty, and the job ends when every worker has exited, so no GPU sits idle.
module load cuda/12.9.1 gnu12/12.3.0 gmp/6.3.0-gcc-12.3.0
export LD_LIBRARY_PATH=/home/gchilders/apps/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=/home/gchilders/apps/lib:$LIBRARY_PATH
export CPATH=/home/gchilders/apps/include:$CPATH
cd ~/data/euler/sumsbkz-py
source env/bin/activate
export PYTHONPATH=$PWD/g6kgpu75:$PWD NUMBA_CUDA_USE_NVIDIA_BINDING=0 CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=6
Q=remote/t4c140_queue.txt
mkdir -p results/complete results/t4
IFS=, read -ra DEVS <<< "$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,name,uuid --format=csv,noheader

pop() { flock $Q.lock bash -c "head -1 $Q; sed -i 1d $Q"; }

worker() {
  local i=$1
  export CUDA_VISIBLE_DEVICES=${DEVS[$i]}
  local tag=$SLURM_JOB_ID-gpu${DEVS[$i]}-w$i-$$
  # the sieves are heuristic and can miss the exact shortest vector now and then: allow one retry
  if ! { bash remote/validate_gpu.sh > results/t4/validate-$tag.log 2>&1 && grep -q IDENTICAL results/t4/validate-$tag.log; } \
     && ! { bash remote/validate_gpu.sh > results/t4/validate-$tag.log 2>&1 && grep -q IDENTICAL results/t4/validate-$tag.log; }; then
    echo "GPU ${DEVS[$i]} failed validation - worker stops"; return 1
  fi
  echo "GPU ${DEVS[$i]} validated $(date)"
  local J=legacy/third_party_private/decompose/sumsbkz/decompose
  while true; do
    t=$(pop); [ -z "$t" ] && break
    set -- $t; k=$1; N=$2
    echo "GPU ${DEVS[$i]} start completion k=$k N=$N $(date)"
    timeout 8h taskset -c 0-$(($(nproc --all) - 1)) python tools/complete.py $J/jw1.txt $J/jw2.txt $J/jw4.txt $J/jw5.txt $J/jw6.txt $J/jw/*.txt \
      -k $k -N $N --alg gpu --threads 5 --starts 100 $( [ $N -gt 100 ] && echo "--pad-db results/complete/db-k$k-N100.npy --bkz-pool 100 --bkz-block 30" || echo "--db-npy results/complete/db-k$k-N$N.npy") \
      -o results/complete/k$k-N$N.txt 2>&1 | grep -v "^Warning" | grep -v "^ *[0-9]* *[0-9]* \|^n   It\|max_nr_buckets\|^$" > results/complete/k$k-N$N.log
    echo "GPU ${DEVS[$i]} done  k=$k N=$N $(date) :: $(grep '^  k=' results/complete/k$k-N$N.log | tail -1 | cut -c1-200)"
  done
}

for i in "${!DEVS[@]}"; do worker $i & done
wait
echo "all workers done $(date)"
