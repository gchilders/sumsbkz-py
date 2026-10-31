#!/bin/bash
# Congruence solutions (tools/congsolve.py) and their completions (tools/complete.py) for the
# tasks in remote/cong_queue.txt (one \"k M C\" per line, flock-protected).  One worker per entry of
# CUDA_VISIBLE_DEVICES (repeat a GPU to run several workers on it).  Run inside an allocation:
#   srun --jobid=J --overlap -N1 -n1 bash -c "export CUDA_VISIBLE_DEVICES=0,0,1,1 G6K=g6kgpu; bash remote/turing_cong.sh"
module load cuda/12.9.1 gnu12/12.3.0 gmp/6.3.0-gcc-12.3.0
export LD_LIBRARY_PATH=/home/gchilders/apps/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=/home/gchilders/apps/lib:$LIBRARY_PATH
export CPATH=/home/gchilders/apps/include:$CPATH
cd ~/data/euler/sumsbkz-py
source env/bin/activate
export PYTHONPATH=$PWD/${G6K:-g6kgpu75}:$PWD NUMBA_CUDA_USE_NVIDIA_BINDING=0 CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=4
Q=remote/cong_queue.txt
mkdir -p results/congsolve results/complete results/t4
IFS=, read -ra DEVS <<< "$CUDA_VISIBLE_DEVICES"
TS="taskset -c 0-$(($(nproc --all) - 1))"
pop() { flock $Q.lock bash -c "head -1 $Q; sed -i 1d $Q"; }

worker() {
  local i=$1
  export CUDA_VISIBLE_DEVICES=${DEVS[$i]}
  local tag=cong-gpu${DEVS[$i]}-w$i-$$
  if ! { bash remote/validate_gpu.sh > results/t4/validate-$tag.log 2>&1 && grep -q IDENTICAL results/t4/validate-$tag.log; } \
     && ! { bash remote/validate_gpu.sh > results/t4/validate-$tag.log 2>&1 && grep -q IDENTICAL results/t4/validate-$tag.log; }; then
    echo "GPU ${DEVS[$i]} failed validation - worker stops"; return 1
  fi
  while true; do
    t=$(pop); [ -z "$t" ] && break
    set -- $t; k=$1; M=$2; C=$3; tag=k$k-M$M-C$C
    echo "GPU ${DEVS[$i]} start congruence $tag $(date)"
    timeout 4h $TS python tools/congsolve.py -k $k -M $M -C $C --start $((M * 10 + 1)) --alg gpu --threads 4 \
      -o results/congsolve/$tag.txt > results/congsolve/$tag.log 2>&1
    timeout 10h $TS python tools/complete.py results/congsolve/$tag.txt -k $k -N 100 --both --alg gpu --threads 4 --starts 30 \
      --db-npy results/complete/db-k$k-N100.npy -o results/complete/cong-$tag.txt 2>&1 \
      | grep -v "^Warning" | grep -v "^ *[0-9][0-9]* *[0-9][0-9]* \|^n   It\|max_nr_buckets\|^$" > results/complete/cong-$tag.log
    echo "GPU ${DEVS[$i]} done  $tag $(date) :: $(grep '^  k=' results/complete/cong-$tag.log | tail -1 | cut -c1-200)"
  done
}

for i in "${!DEVS[@]}"; do worker $i & done
wait
echo "all workers done $(date)"
