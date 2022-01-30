
#include <omp.h>
#include <stdio.h>
#include <mpi.h>
#include <stdbool.h>
#include <stdlib.h>
#include <math.h>
#define R 1
long N = 47000;

/**
 * Helper function to generate the matrix. The total number of its rows is  
 * depend on the variable "N". The matrix generates a part of the actual 
 * matrix. This is to make sure "each element of the matrix should reside
 * on only one processor".__int64 used to avoid overflow
 *
 * @param cols The column number of the matrix.
 * @param rank current processor number.
 * @return an initial matrix.
 */
double** initialize_matrix(long cols,int rank) {
    double** matrix = (double**)malloc(sizeof(double) * N);
    for(long i = 0;i<N;i++){
        matrix[i] = (double*)malloc(sizeof(double) * cols);
    }
    for (long i = 0; i < N; i++) {
        for (long j = 0; j < cols; j++) {
            matrix[i][j] = (double)(i) * N + (double)j + 1.0 + (rank)*cols;
        }
    }
    return matrix;
}

/**
 * Helper function to print the matrix. The total number of its rows is  
 * depend on the variable "N".
 *
 * @param cols The column amount number of the matrix.
 * @param matrix The matrix is needed to print.
 */
void print_matrix(double** matrix,long cols) {
    for (long i = 0; i < N; i++) {
        for (long j = 0; j < cols; j++) {
            printf("%.2f, ", matrix[i][j]);
        }
        printf("\n");
    }
}


/**
 * Helper function to generate a vector. The length is  
 * depend on the variable "N".
 *
 * @return an initial vector.
 */
double* initialize_vector() {
    double* vector = (double*)malloc(sizeof(double) * N);
    for (long i = 0; i < N; i++) {
        vector[i] = (double)i + 1.0;
    }
    return vector;
}

/**
 * Helper function to print the vector. The length is  
 * depend on the variable "N".
 *
 * @param vector The matrix is needed to print.
 */
void print_vector(double* vector) {
    for (long i = 0; i < N; i++) {
        printf("%.2f, ", vector[i]);
    }
}

/**
 * Free the memory allocated before exiting the program.
 * 
 * @param matrix The matrix is needed to destroy.
 */
void destroy_matrix(double** matrix) {
    for (long i = 0; i < N; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

/**
 * Free the memory allocated before exiting the program.
 * 
 * @param vector The vector is needed to destroy.
 */
void destroy_vector(double* vector) {
    free(vector);
}

/*
* Helper function to transform result vector and make ready for next computing
* Used for transform result to vector
* 
* @param x          Pointer to one input vector
* @param y          Pointer to another input vector
*/
void swap(double** x, double** y) {
    double* temp = *x;
    *x = *y;
    *y = temp;
}

/**
* use OpenMP and MPI to increase the efficiency of computing. First use 
* MPI_Bcast to replicate vector and every processor generate a part of 
* matrix to reduce the communication time. And use OpenMP to parallel 
* compute V x M. Finally, use MPI_Gather to get all the results from each 
* processor.
*/
int main(int argc, char *argv[]) {

    int rank;
    int size;

    /* Start up MPI */
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /*
        get the total number of column that the current processor need to be processed
    */
    long cols = 0;
    if (N > size){
        cols = N/size;
        if(rank < N % size && N % size != 0)
            cols+=1;
    }else{
        if(rank < N)
            cols = 1;
    }
        
    printf("Running as processor %d\n", rank);
    bool isMaster = rank == 0;
    double start = MPI_Wtime();

    //generate vector, matrix, result and final result
    double* vector = initialize_vector();
    double** matrix = initialize_matrix(cols,rank);
    double* result = (double*)malloc(sizeof(double) * cols);//it contains each processor's every iteration results
    double *final_res; // it contains every iteration results
    
    //start iteration
    for(int r = 0; r < R ; r++){

        MPI_Barrier(MPI_COMM_WORLD);
        //Broadcast the vector to every processor
        MPI_Bcast(vector, N ,MPI_DOUBLE,0,MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        long i = 0;
        //use OpenMP to parallel the computation
    #pragma omp parallel for
        for (i = 0; i < cols; i++) {
            result[i] = 0;
            long j = 0;
            for (j = 0; j < N; j++) {
                result[i] += matrix[j][i] * vector[j];
            }
        }

        if(r == 0 && isMaster){
            final_res = (double*)malloc(sizeof(double) * N);
        }
        //gather all the result from every processor to final_res
        MPI_Gather(result,cols,MPI_DOUBLE,final_res, cols,MPI_DOUBLE,0,MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);

        //swap the vector and final_res, because the next iteration, need to use final_res as the param to compute
        if(r != R -1 && isMaster){
            swap(&vector, &final_res);
        }
    }
    

    double finish = MPI_Wtime();
    printf("Processor %d has finished. This took %.1f seconds\n", rank, finish-start);

    destroy_matrix(matrix);
    destroy_vector(vector);
    destroy_vector(result);

    MPI_Finalize();

    return 0;
}
