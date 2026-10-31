#!/bin/bash
module load cuda/12.9.1 gnu12/12.3.0 gmp/6.3.0-gcc-12.3.0
export LD_LIBRARY_PATH=/home/gchilders/apps/lib:$LD_LIBRARY_PATH LIBRARY_PATH=/home/gchilders/apps/lib:$LIBRARY_PATH CPATH=/home/gchilders/apps/include:$CPATH
cd ~/data/euler/sumsbkz-py; source env/bin/activate
export PYTHONPATH=$PWD/g6kgpu:$PWD NUMBA_CUDA_USE_NVIDIA_BINDING=0 CUDA_HOME=${CUDA_HOME:-/usr/local/cuda} OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=48
export CUDA_VISIBLE_DEVICES=1,2,3
TS="taskset -c 0-$(($(nproc --all) - 1))"
echo "makedb start $(date)"
[ -f results/complete/db-k50-N117.npy ] || $TS python tools/makedb.py -k 50 -N 117 --gpus 3 --threads 48 -o results/complete/db-k50-N117.npy
echo "makedb done $(date)"
export CUDA_VISIBLE_DEVICES=1
$TS python tools/momentstart.py -k 50 -j 2 -N 117 --t 10,15 --per-t 25 --seed 11 -o results/k50exp/mstart-k50-N117-t10-15.txt &
export CUDA_VISIBLE_DEVICES=2
$TS python tools/momentstart.py -k 50 -j 2 -N 117 --t 3,5,8,12,20 --per-t 25 --seed 12 -o results/k50exp/mstart-k50-N117-tmix.txt &
wait
echo "exp2 done $(date)"
