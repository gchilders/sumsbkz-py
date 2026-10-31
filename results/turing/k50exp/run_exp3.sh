#!/bin/bash
module load cuda/12.9.1 gnu12/12.3.0 gmp/6.3.0-gcc-12.3.0
export LD_LIBRARY_PATH=/home/gchilders/apps/lib:$LD_LIBRARY_PATH LIBRARY_PATH=/home/gchilders/apps/lib:$LIBRARY_PATH CPATH=/home/gchilders/apps/include:$CPATH
cd ~/data/euler/sumsbkz-py; source env/bin/activate
export PYTHONPATH=$PWD/g6kgpu:$PWD NUMBA_CUDA_USE_NVIDIA_BINDING=0 CUDA_HOME=${CUDA_HOME:-/usr/local/cuda} OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=48
export CUDA_VISIBLE_DEVICES=1,2,3
TS="taskset -c 0-$(($(nproc --all) - 1))"
export CUDA_VISIBLE_DEVICES=1

export CUDA_VISIBLE_DEVICES=2

export CUDA_VISIBLE_DEVICES=1
$TS python tools/recordstart.py -k 50 -N 117 --per 12 -o results/k50exp/recstart-k50-N117.txt
export CUDA_VISIBLE_DEVICES=2
$TS python tools/recordstart.py -k 50 -N 100 --per 12 -o results/k50exp/recstart-k50-N100.txt
echo "exp3 done $(date)"
