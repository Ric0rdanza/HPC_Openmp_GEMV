#!/bin/bash
#SBATCH --job-name="hpc-mpi"
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --partition=normal
#SBATCH --output=hpc-mpi_%j.out
#SBATCH --error=hpc-mpi_%j.err
module purge
module load 2020
module load GCC/9.3.0

echo "OpenMP parallelism"
export OMP_NUM_THREADS=4
mpirun ./hpc-mpi
echo "DONE "
