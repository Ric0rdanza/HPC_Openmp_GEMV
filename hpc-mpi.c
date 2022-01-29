
#include <omp.h>
#include <stdio.h>
#include <mpi.h>
#include <stdbool.h>
#include <stdlib.h>
#include <math.h>
#define N 3
#define R 1

double** initialize_matrix(int cols) {
    double** matrix = (double**)malloc(sizeof(double) * N);
    for(int i = 0;i<N;i++){
        matrix[i] = (double*)malloc(sizeof(double) * cols);
    }
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i][j] = (double)i * N + (double)j + 1.0;
        }
    }
    return matrix;
}


void print_matrix(double** matrix,int cols) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%.2f, ", matrix[i][j]);
        }
        printf("\n");
    }
}

double* initialize_vector() {
    double* vector = (double*)malloc(sizeof(double) * N);
    for (int i = 0; i < N; i++) {
        vector[i] = (double)i + 1.0;
    }
    return vector;
}

void print_vector(double* vector) {
    for (int i = 0; i < N; i++) {
        printf("%.2f, ", vector[i]);
    }
}

void destroy_matrix(double** matrix) {
    for (int i = 0; i < N; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

void destroy_vector(double* vector) {
    free(vector);
}

void swap(double** x, double** y) {
    double* temp = *x;
    *x = *y;
    *y = temp;
}

int main(int argc, char *argv[]) {

    int rank;
    int size;
    double start;
    /* Start up MPI */
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int cols = N;
    if (cols > size){
        cols = N/size;

        if(N % size < rank){
            cols++;
        }   
    }
    
    printf("Running as processor %d\n", rank);
    bool isMaster = rank == 0;
    double* vector;
    if(isMaster){
        //initialize variables
        // double* matrix = initialize_matrix();
        //print_matrix(matrix);
        vector = initialize_vector();
        start = MPI_Wtime();

    }
    MPI_Bcast(&vector,N,MPI_DOUBLE,0,MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    double** matrix = initialize_matrix(cols);
    double* result = (double*)malloc(sizeof(double) * cols);
    
    for(int r = 0; r < R ; r++){
        int i = 0;
    #pragma omp parallel for
        for (i = 0; i < cols; i++) {
            result[i] = 0;
            int j = 0;
            for (j = 0; j < N; j++) {
                result[i] += matrix[j][i] * vector[j];
            }
        }
        if(r != R -1){
            swap(&vector, &result);
        }
    }
    

    double *final_res = NULL;

    if(isMaster){
        final_res = (double*)malloc(sizeof(double) * N);
        MPI_Gather(result,cols,MPI_DOUBLE,final_res,cols,MPI_DOUBLE,0,MPI_COMM_WORLD);
        const double finish = MPI_Wtime();
        printf("Processor %d has finished. This took %.1f seconds\n", rank, finish-start);
        print_vector(result);
    }else{
        MPI_Gather(result,cols,MPI_DOUBLE,final_res,cols,MPI_DOUBLE,0,MPI_COMM_WORLD);
    }
    destroy_matrix(matrix);
    destroy_vector(vector);
    destroy_vector(result);
    MPI_Barrier(MPI_COMM_WORLD);
    printf("Processor %d has finished\n", rank);
    MPI_Finalize();

    return 0;
}
