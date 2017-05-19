#!/bin/bash -l

#SBATCH -N 2         #Use 2 nodes
#SBATCH -t 00:10:00  #Set 30 minute time limit
#SBATCH -p debug   #Submit to the regular 'partition'
#SBATCH -o fft3-cori-%j.out
#SBATCH -L SCRATCH   #Job requires $SCRATCH file system
#SBATCH -C haswell   #Use Haswell nodes

export MPICH_MAX_THREAD_SAFETY=multiple
srun --nodes=2 --ntasks=8 --ntasks-per-node=4 --cpus-per-task=8 ./fft3-cori.x -nl 8