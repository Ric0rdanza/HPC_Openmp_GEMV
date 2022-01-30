#define R 100

#include <omp.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// Size of the vector
__int64 N = 47000;

/*
* Initialize matrix with shape N * N and random number from 0 to 9
* __int64 used to avoid overflow
*/
double* initialize_matrix() {
    double* matrix = (double*)malloc(sizeof(double) * N * N);
    __int64 i = 0;
#pragma omp parallel for
    for (i = 0; i < N; i++) {
        for (__int64 j = 0; j < N; j++) {
            __int64 x = i * N + j;
            matrix[i * N + j] = (double)(rand() % 10);
        }
    }
    return matrix;
}

/*
* Print the matrix
* Used for debug
*/
void print_matrix(double* matrix) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%.2f, ", matrix[i * N + j]);
        }
        printf("\n");
    }
}

/*
* Random initialize vector with number from 0 - 9
*/
double* initialize_vector() {
    double* vector = (double*)malloc(sizeof(double) * N);
    for (int i = 0; i < N; i++) {
        vector[i] = (double)(rand() % 10);
    }
    return vector;
}

/*
* Initialize a N-size vector with 0
*/
double* initialize_vector_zero() {
    double* result = (double*)malloc(sizeof(double) * N);
    for (int i = 0; i < N; i++) {
        result[i] = 0;
    }
    return result;
}

/*
* Print the vector
* Used for debug
*/
void print_vector(double* vector) {
    for (int i = 0; i < N; i++) {
        printf("%.2f ", vector[i]);
    }
    printf("\n");
}


/*
* The main function to compute V x M
* Each thread calculate one element in the result vector
* 
* @param matrix         The matrix M
* @param vector         The vector V
* @param result         Saving the result vector of V x M
*/
void gevm(double* matrix, double* vector, double* result) {
    __int64 i = 0;
    omp_set_num_threads(16);
#pragma omp parallel for
    for (i = 0; i < N; i+=1) {
        result[i] = 0;
        __int64 j = 0;
        for (j = 0; j < N; j++) {
            result[i] += matrix[i * N + j] * vector[j];
        }
    }
}

/*
* Helper function to transform result vector and make ready for next computing
* Used for transform result to vector
* 
* @param x          Pointer to one input vector
* @param y          Pointer to another input vetor
*/
void swap(double** x, double** y) {
    double* temp = *x;
    *x = *y;
    *y = temp;
}

/*
* Free the memory allocated before exit the program.
* 
* @param vector     The vector going to be free.
*/
void destroy(double* vector) {
    free(vector);
}

int main(int argc, char* argv[])
{
    // Initialize the input matrix and vector, and vector to save results
    double* matrix = initialize_matrix();
    double* vector = initialize_vector();
    double* result = initialize_vector_zero();
    
    // Get timing
    double start, end;
    start = omp_get_wtime();

    
    // START COMPUTING 

    // R times iteration.
    for (int i = 0; i < R; i++) {
        gevm(matrix, vector, result);
        swap(&vector, &result);
    }
    
    // Stop timing
    end = omp_get_wtime();


    // Print result
    printf("obtained in %f seconds\n", end - start);

    // Destroy the vector initialized
    destroy(matrix);
    destroy(vector);
    destroy(result);
}
