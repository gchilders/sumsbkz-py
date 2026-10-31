#!/bin/bash
#SBATCH --partition=p_gpu_t4
#SBATCH --qos=q_gpu_t4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=%x-%j.out
# A pool of T4 workers (one per allocated GPU) taking powers from the shared list
# remote/t4_queue.txt (one "k N" per line; flock-protected, shared by all pool jobs).
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
Q=remote/t4_queue.txt
mkdir -p results/t4
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
  while true; do
    t=$(pop); [ -z "$t" ] && break
    set -- $t; k=$1; N=$2
    echo "GPU ${DEVS[$i]} start k=$k N=$N $(date)"
    timeout 4h taskset -c 0-$(($(nproc --all) - 1)) python sievesearch.py -k $k -N $N --runs 1 --radius ${RAD:-1.3333} --alg gpu --gpu-bucketer bdgl \
      --saturation-ratio 0.375 --db-size-factor 2.77 --multi-bucket 2 --max-nr-buckets 0 \
      --gpu-descent --basis-cache bases --prefetch 1 --threads 5 \
      --seed $((k * 1000 + N)) -o results/t4/k$k-N$N.txt 2>&1 | grep -v "^Warning" > results/t4/k$k-N$N.log
    echo "GPU ${DEVS[$i]} done  k=$k N=$N $(date) :: $(grep '^k=' results/t4/k$k-N$N.log | tail -1)"
  done
}

for i in "${!DEVS[@]}"; do worker $i & done
wait
echo "all workers done $(date)"
