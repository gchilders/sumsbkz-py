#!/bin/bash
# Moment-coset descents (tools/momentstart.py) for the "k j" pairs in remote/mstart_queue.txt,
# building the N=100 sieve database first where it is missing (tools/makedb.py).
module load cuda/12.9.1 gnu12/12.3.0 gmp/6.3.0-gcc-12.3.0
export LD_LIBRARY_PATH=/home/gchilders/apps/lib:$LD_LIBRARY_PATH
cd ~/data/euler/sumsbkz-py
source env/bin/activate
export PYTHONPATH=$PWD/g6kgpu:$PWD NUMBA_CUDA_USE_NVIDIA_BINDING=0 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=4
Q=remote/mstart_queue.txt
IFS=, read -ra DEVS <<< "$CUDA_VISIBLE_DEVICES"
TS="taskset -c 0-$(($(nproc --all) - 1))"
pop() { flock $Q.lock bash -c "head -1 $Q; sed -i 1d $Q"; }
worker() {
  export CUDA_VISIBLE_DEVICES=${DEVS[$1]}
  while true; do
    t=$(pop); [ -z "$t" ] && break
    set -- $t; k=$1; j=$2
    db=results/complete/db-k$k-N100.npy
    [ -f $db ] || $TS python tools/makedb.py -k $k -N 100 -o $db 2>&1 | grep "vectors"
    $TS timeout 3h python tools/momentstart.py -k $k -j $j --per-t 25 -o results/complete/mstart-k$k.txt 2>&1 \
      | grep "smallest m" | sed "s/^/k=$k j=$j /"
  done
}
for i in "${!DEVS[@]}"; do worker $i & done
wait
