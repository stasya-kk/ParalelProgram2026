#!/bin/bash
#SBATCH --job-name=lab5
#SBATCH --time=0:10:00
#SBATCH --ntasks-per-node=8
#SBATCH --partition=batch

module load intel/mpi4

mkdir -p results_mpi

for np in 1 2 4 8; do
    mpirun -r ssh -np $np ./lab5_mpi >> results_mpi/time_${np}proc.txt
done

echo "Job finished at $(date)"