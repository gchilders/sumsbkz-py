#!/bin/bash
# Generic GPU work queue: each line of $Q (default remote/cmd_queue.txt) is a shell command,
# run from the project directory with CUDA_VISIBLE_DEVICES set to the worker's GPU.  One worker
# per entry of CUDA_VISIBLE_DEVICES (repeat a GPU for several workers).  The G6K-GPU build is
# chosen by GPU model (g6kgpu75 for the T4s, g6kgpu otherwise).
module load cuda/12.9.1 gnu12/12.3.0 gmp/6.3.0-gcc-12.3.0
export LD_LIBRARY_PATH=/home/gchilders/apps/lib:$LD_LIBRARY_PATH
cd ~/data/euler/sumsbkz-py
source env/bin/activate
Q=${Q:-remote/cmd_queue.txt}
if nvidia-smi -L | grep -q T4; then G=g6kgpu75; else G=g6kgpu; fi
export PYTHONPATH=$PWD/$G:$PWD NUMBA_CUDA_USE_NVIDIA_BINDING=0 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=4
export TS="taskset -c 0-$(($(nproc --all) - 1))"
IFS=, read -ra DEVS <<< "$CUDA_VISIBLE_DEVICES"
pop() { flock $Q.lock bash -c "head -1 $Q; sed -i 1d $Q"; }
worker() {
  export CUDA_VISIBLE_DEVICES=${DEVS[$1]}
  while true; do
    c=$(pop); [ -z "$c" ] && break
    echo "GPU $CUDA_VISIBLE_DEVICES start $(date +%T) :: $c"
    bash -c "$c" > /dev/null 2>&1
    echo "GPU $CUDA_VISIBLE_DEVICES done  $(date +%T) :: $c"
  done
}
for i in "${!DEVS[@]}"; do worker $i & done
wait
echo "all workers done $(date)"
