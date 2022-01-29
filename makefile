#!/bin/bash

module load 2020
module load GCC/9.3.0
gcc -fopenmp -o gemv gemv.c -lm
