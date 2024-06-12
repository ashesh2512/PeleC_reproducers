#!/bin/bash
#SBATCH --job-name=pmf
#SBATCH -p allgriz
#SBATCH --nodes=1
#SBATCH --time=0:10:00
#SBATCH --exclusive
#SBATCH --output=job.out
#SBATCH --mail-user=ashesh.sharma@hpe.com
#SBATCH --mail-type=BEGIN,END,FAIL

source ~/.bashrc
load_nvidia_env
export CUDA_VISIBLE_DEVICES=${SLURM_LOCALID}%4

srun -N1 -n1 -c1 --threads-per-core=1 fKernelSpec >& fKernelSpec.log

