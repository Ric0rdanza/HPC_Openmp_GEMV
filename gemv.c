#define N 47000
#define R 100

#include <omp.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// Used to save number of threads
int N_THREADS;

// Save matrix in 1-dimension vector
double* initialize_matrix() {
    double* matrix = (double*)malloc(sizeof(double) * N * N);
    if (matrix == NULL) {
        printf("Jesus");
    }
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            matrix[i * N + j] = (double)(rand() % 10);
        }
    }
    return matrix;
}

// Save matrix in 2-dimension vector, parallel assign random values
double** initialize_matrix_2dim() {
    double** matrix = (double**)malloc(sizeof(double*) * N);
    int i = 0;
#pragma omp parallel for
    for (i = 0; i < N; i++) {
        matrix[i] = (double*)malloc(sizeof(double) * N);
        for (int j = 0; j < N; j++) {
            matrix[i][j] = (double)(rand() % 10);
        }
    }
    return matrix;
}

// Print matrix (for debug)
void print_matrix(double* matrix) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%.2f, ", matrix[i * N + j]);
        }
        printf("\n");
    }
}

void print_matrix_2dim(double** matrix) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%.2f ", matrix[i][j]);
        }
        printf("\n");
    }
}

// Save vector in 1-dimension
void* initialize_vector() {
    double* vector = (double*)malloc(sizeof(double) * N);
    for (int i = 0; i < N; i++) {
        vector[i] = (double)(rand() % 10);
    }
    return vector;
}

// Print vector (for debug)
void print_vector(double* vector) {
    for (int i = 0; i < N; i++) {
        printf("%.2f ", vector[i]);
    }
    printf("\n");
}

// DEPRECATED for matrix size limited
// Naive gevm openmp approach -- appoximately 14x speedup
// @ matrix: Input matrix (1-dimension)
// @ vector: Input vector
// @ result: Save output vector
void gevm(double* matrix, double* vector, double* result) {
    int i = 0;
    //omp_set_num_threads(16);
#pragma omp parallel for
    for (i = 0; i < N; i+=1) {
        result[i] = 0;
        int j = 0;
        for (j = 0; j < N; j++) {
            result[i] += matrix[i * N + j] * vector[j];
        }
    }
}

// Naive gevm openmp approach -- appoximately ?x speedup
// @ matrix: Input matrix (2-dimension)
// @ vector: Input vector
// @ result: Save output vector
void gevm_2dim(double** matrix, double* vector, double* result) {
    int i = 0;
    //omp_set_num_threads(16);
#pragma omp parallel for
    for (i = 0; i < N; i++) {
        result[i] = 0;
        int j = 0;
        for (j = 0; j < N; j++) {
            result[i] += vector[j] * matrix[j][i];
        }
    }
}
// CACHE TEST:  N = 47000, R = 100
// CACHE OPTIMIZED  40.96
// NO CACHE         MORE THAN 5 MINUTES


// Helper: to count allocate pattern for matrix
// @ t: Number of threads to be used
// @ pattern: array to save patterns
void allocate_matrix(int t, int* pattern) {
    int length = N;
    int block = t;
    for (int i = 0; i < t; i++) {
        float x = (double)length / (double)block;
        int temp;
        if (length % block != 0) {
            temp = (int)(ceil(x));
            if (i == 0) {
                pattern[i] = temp;
            }
            else {
                pattern[i] = pattern[i - 1] + temp;
            }
            length -= temp;
            block -= 1;
            continue;
        }
        else {
            temp = (int)(length / block);
            if (i == 0) {
                pattern[i] = temp;
            }
            else {
                pattern[i] = pattern[i - 1] + temp;
            }
            length -= temp;
            block -= 1;
        }
    }
}

// Block parallel gevm openmp approach
// @ matrix: Input matrix (2-dimension)
// @ vector: Input vector
// @ result: Save output vector
// @ pattern saves pattern to allocate block of matrix
void gevm_block_2dim(double** matrix, double* vector, double* result, int* pattern) {
    int i = 0;
#pragma omp parallel for
    for (i = 0; i < N_THREADS; i++) {
        int x;
        if (i == 0) {
            x = 0;
        }
        else {
            x = pattern[i - 1];
        }
        for (x; x < pattern[i]; x++) {
            result[x] = 0;
            for (int y = 0; y < N; y++) {
                result[x] += vector[y] * matrix[x][y];
            }
        }
    }
    
}


// Change pointer x and y (used to exchange pointer of vector and result)
void swap(double** x, double** y) {
    double* temp = *x;
    *x = *y;
    *y = temp;
}

// Destroy matrix (2-dimension)
void destroy_matrix(double** matrix) {
    for (int i = 0; i < N; i++) {     
        free(matrix[i]);
    }
    free(matrix);
}

// Destroy vector
void destroy_vector(double* vector) {
    free(vector);
}

int main(int argc, char* argv[])
{
    //This is going to set the number of threads
    N_THREADS = omp_get_num_procs();
    /*
    //initialize variables
    double* matrix = initialize_matrix();
//    print_matrix(matrix);
    double* vector = initialize_vector();
    double* result = (double*)malloc(sizeof(double) * N);
    */

    double** matrix = initialize_matrix_2dim();
    double* vector = initialize_vector();
    double* result = (double*)malloc(sizeof(double) * N);

    //print_matrix_2dim(matrix);
    //print_vector(vector);

    for (int i = 0; i < N; i++) {
        result[i] = 0;
    }
//    print_vector(vector);
//    print_vector(result);
    
    // Get timing
    double start, end;
    start = omp_get_wtime();

    
    // START COMPUTING 

    // Testing naive approach
    /*
    for (int i = 0; i < R; i++) {
        gevm_2dim(matrix, vector, result);
        swap(&vector, &result);
    }
    */


    // Testing block parallel approach
    int* pattern = (int*)malloc(sizeof(int) * N_THREADS);
    allocate_matrix(N_THREADS, pattern);

    gevm_block_2dim(matrix, vector, result, pattern);
    swap(&vector, &result);
    for (int i = 0; i < R - 1; i++) {
        gevm_block_2dim(matrix, vector, result, pattern);
        swap(&vector, &result);
    }

    free(pattern);
    //print_vector(result);
    // Stop timing
    end = omp_get_wtime();

    //print_vector(vector);

    // Print result
    printf("obtained in %f seconds\n", end - start);

    destroy_matrix(matrix);
    destroy_vector(vector);
    destroy_vector(result);
}
