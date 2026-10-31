#!/bin/bash
#SBATCH --job-name=eslp-h200
#SBATCH --partition=p_gpu_h200
#SBATCH --qos=q_gpu_h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=60
#SBATCH --mem=900gb
#SBATCH --time=48:00:00
#SBATCH --output=%x-%j.out
# Four GPU workers pull "k N" tasks from remote/h200_tasks.txt (one task per line) until it is
# empty, so all four GPUs stay busy and finish together; the job then exits.
# Each worker validates its GPU first and stops (freeing nothing to idle) if validation fails.
module load cuda/12.9.1 gnu12/12.3.0 gmp/6.3.0-gcc-12.3.0
export LD_LIBRARY_PATH=/home/gchilders/apps/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=/home/gchilders/apps/lib:$LIBRARY_PATH
export CPATH=/home/gchilders/apps/include:$CPATH
cd ~/data/euler/sumsbkz-py
source env/bin/activate
export PYTHONPATH=$PWD/g6kgpu:$PWD NUMBA_CUDA_USE_NVIDIA_BINDING=0 CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=15
QUEUE=remote/h200_queue.$SLURM_JOB_ID
cp remote/h200_tasks.txt $QUEUE
mkdir -p results/h200
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

pop() {   # atomically take the first line of the queue
  flock $QUEUE.lock bash -c "head -1 $QUEUE; sed -i 1d $QUEUE"
}

worker() {
  local g=$1
  export CUDA_VISIBLE_DEVICES=$g
  if ! bash remote/validate_gpu.sh > results/h200/validate-gpu$g.log 2>&1 || ! grep -q IDENTICAL results/h200/validate-gpu$g.log; then
    echo "GPU $g failed validation - cancelling the job so no GPU sits idle"
    cat results/h200/validate-gpu$g.log; scancel $SLURM_JOB_ID; return 1
  fi
  echo "GPU $g validated $(date)"
  while true; do
    t=$(pop); [ -z "$t" ] && break
    set -- $t; k=$1; N=$2
    echo "GPU $g start k=$k N=$N $(date)"
    lim=$([ "$N" -le 110 ] && echo 18000 || echo 72000)   # 5 h / 20 h cap: a hung sieve cannot hold a GPU
    timeout $lim python sievesearch.py -k $k -N $N --runs 1 --radius 1.3333 --alg gpu --gpu-bucketer bdgl \
      --saturation-ratio 0.375 --db-size-factor 2.77 --multi-bucket 2 --max-nr-buckets 0 \
      --gpu-descent --basis-cache bases --prefetch 1 --db-cap 30000000 --threads 15 \
      --seed $((k * 1000 + N)) -o results/h200/k$k-N$N.txt 2>&1 | grep -v "^Warning" > results/h200/k$k-N$N.log
    echo "GPU $g done  k=$k N=$N $(date) :: $(grep '^k=' results/h200/k$k-N$N.log | tail -1)"
  done
}

for g in 0 1 2 3; do worker $g & done
wait
echo "all workers done $(date)"
