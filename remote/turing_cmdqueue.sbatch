#!/bin/bash
#SBATCH --partition=p_gpu_t4
#SBATCH --qos=q_gpu_t4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=%x-%j.out
bash remote/turing_cmdqueue.sh
