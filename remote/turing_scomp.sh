#!/bin/bash
# Completions (tools/complete.py --both, N=100) of the structured congruence solutions listed in
# remote/scomp_queue.txt ("k file" per line); one worker per entry of CUDA_VISIBLE_DEVICES.
module load cuda/12.9.1 gnu12/12.3.0 gmp/6.3.0-gcc-12.3.0
export LD_LIBRARY_PATH=/home/gchilders/apps/lib:$LD_LIBRARY_PATH
cd ~/data/euler/sumsbkz-py
source env/bin/activate
export PYTHONPATH=$PWD/${G6K:-g6kgpu}:$PWD NUMBA_CUDA_USE_NVIDIA_BINDING=0 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=4
Q=remote/scomp_queue.txt
IFS=, read -ra DEVS <<< "$CUDA_VISIBLE_DEVICES"
pop() { flock $Q.lock bash -c "head -1 $Q; sed -i 1d $Q"; }
worker() {
  export CUDA_VISIBLE_DEVICES=${DEVS[$1]}
  while true; do
    t=$(pop); [ -z "$t" ] && break
    set -- $t; k=$1; f=$2; tag=$(basename $f .txt)
    echo "GPU $CUDA_VISIBLE_DEVICES start $tag $(date)"
    taskset -c 0-$(($(nproc --all) - 1)) timeout 10h python tools/complete.py $f -k $k -N 100 --both --alg gpu \
      --threads 4 --starts 30 --db-npy results/complete/db-k$k-N100.npy -o results/complete/comp-$tag.txt 2>&1 \
      | grep -v "Warning\|warn(\|^ *[0-9][0-9]* *[0-9][0-9]* \|^n   It\|max_nr_buckets\|^$" > results/complete/comp-$tag.log
    echo "GPU $CUDA_VISIBLE_DEVICES done  $tag $(date) :: $(grep '^  k=' results/complete/comp-$tag.log | tail -1 | cut -c1-200)"
  done
}
for i in "${!DEVS[@]}"; do worker $i & done
wait
