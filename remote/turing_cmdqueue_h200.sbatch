#!/bin/bash
#SBATCH --partition=p_gpu_h200
#SBATCH --qos=q_gpu_h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=60
#SBATCH --mem=900gb
#SBATCH --output=%x-%j.out
# Generic command queue on the H200 node, three workers per GPU (Q selects the queue file).
export CUDA_VISIBLE_DEVICES=0,0,0,1,1,1,2,2,2,3,3,3
bash remote/turing_cmdqueue.sh
