#!/bin/bash

module load 2020
module load GCC/9.3.0
gcc -fopenmp -o gemv gemv.c -lm

mpiCC -O3 -Wall -W --std=c++11 -lm -Wno-cast-function-type -lmpi -fopenmp -o hpc-mpi hpc-mpi.c
