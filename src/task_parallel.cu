#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <omp.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define MAX_COUNT 10

__device__ void print_var(int* var, int size);
__device__ void print_matrix(int* matrix, int n);
 __device__ void print_triangle(int* matrix, int n, int* var, int size);

 __device__ int get_sum(int* matrix, int n, int* var, int size);

 __device__ void save_log(FILE* logfile, int** matrix, int n, int* var, int size, int sum, int min_sum);
__device__ void _variate(int* matrix, int n, int* var, int size, int min, int max, int deep, int* result_var, int* min_sum, FILE* logfile);

__device__ void new_variate(int* matrix, int n, int* var, int size, int min, int max, int true_max, int deep, int* result_var, int* min_sum, FILE* logfile, int* num);
 __global__ void variate(int* matrix, int n, int* var, int size, int min, int max, int deep, int* result_var, int* min_sum, FILE* logfile);
__global__ void startCompution(int* matrix, int n, int* var, int size, int* min_sum);
void host_print_var(int* var, int size);
void host_print_matrix(int* matrix, int n);
void host_print_triangle(int* matrix, int n, int* var, int size);

int host_get_sum(int* matrix, int n, int* var, int size);
//void new_variate(int* matrix, int n, int* var, int size, int min, int max, int deep, int* result_var, int* min_sum, FILE* logfile);
void new_variate(int* count, int* matrix, int n, int* var, int size, int min, int max, int deep, int* result_var, int* min_sum, FILE* logfile, int* device_var, int* device_result_var, int* device_matrix, int* device_min_sum);


int main(int argc, char* argv[]) {

    int n = 8;

    int* matrix = (int*)malloc(n * n * sizeof(int));

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            //fread(&matrix[i * n + j], sizeof(int), 1, file);
            matrix[i * n + j] = rand() % 10 + 1;
        }
    }

    if (n <= 20) host_print_matrix(matrix, n);

    int size = 3;

    int* var = (int*)malloc(size*sizeof(int));
    int* result_var = (int*)malloc(size*sizeof(int));
    int* device_var;
    cudaMalloc((void**)&device_var, sizeof(int) * size);
    int* device_result_var;
    cudaMalloc((void**)&device_result_var, sizeof(int) * size);

    int* device_matrix;
    cudaMalloc((void**)&device_matrix, sizeof(int) * n * n);
    cudaMemcpy(device_matrix, matrix, sizeof(int) * n * n, cudaMemcpyHostToDevice);

    int min_sum = INT_MAX;
    int* device_min_sum;
    cudaMalloc((void**)&device_min_sum, sizeof(int));
    cudaMemcpy(device_min_sum, &min_sum, sizeof(int), cudaMemcpyHostToDevice);

    




    int count = 0;

    new_variate(&count, matrix, n, var, size, 1, 8, 0, result_var, &min_sum, NULL, device_var, device_result_var, device_matrix, device_min_sum);











    // cudaMemcpy(result_var, device_result_var, sizeof(int) * size, cudaMemcpyDeviceToHost);
    // cudaMemcpy(&min_sum, device_min_sum, sizeof(int), cudaMemcpyDeviceToHost);

    //cudaDeviceSynchronize();

    // printf("\n---------------Output----------------\n");
    // printf("indexes:\n");
    // printf("[ ");
    // for (int i = 0; i < size; i++) {
    //     printf("%d ", result_var[i]);
    // }
    // printf("]");
    // printf("\n");
    // printf("submatrix:\n");
    // int row_index = 0, col_index = 0;
    // for (int i = result_var[row_index]; row_index < size; col_index = 0, row_index++, i = result_var[row_index]) {
    //     for (int j = result_var[col_index]; j <= i && col_index < size; col_index++, j = result_var[col_index]) {
    //         printf("%d ", matrix[(i - 1) * n + (j - 1)]);
    //     }
    //     printf("\n");
    // }
    // printf("min_sum: %d\n", min_sum);
    //printf("Time: %lf\n", finish - start);
    
    free(matrix);
    free(var);
    free(result_var);

    cudaFree(device_matrix);
    cudaFree(device_var);
    cudaFree(device_result_var);
    
    return 0;
}

__device__ void print_var(int* var, int size) {
    printf("[ ");
    for (int i = 0; i < size; i++) {
        printf("%d ", var[i]);
    }
    printf("]");
    printf("\n");
}

void host_print_var(int* var, int size) {
    printf("[ ");
    for (int i = 0; i < size; i++) {
        printf("%d ", var[i]);
    }
    printf("]");
    printf("\n");
}

__device__ void print_matrix(int* matrix, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%d ", matrix[i * n + j]);
        }
        printf("\n");
    }

}

void host_print_matrix(int* matrix, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%d ", matrix[i * n + j]);
        }
        printf("\n");
    }

}

 __device__ void print_triangle(int* matrix, int n, int* var, int size) {

    int row_index = 0, col_index = 0;
    for (int i = var[row_index]; row_index < size; col_index = 0, row_index++, i = var[row_index]) {
        for (int j = var[col_index]; j <= i && col_index < size; col_index++, j = var[col_index]) {
            printf("%d ", matrix[(i - 1) * n + (j - 1)]);
        }
        printf("\n");
    }
}

void host_print_triangle(int* matrix, int n, int* var, int size) {

    int row_index = 0, col_index = 0;
    for (int i = var[row_index]; row_index < size; col_index = 0, row_index++, i = var[row_index]) {
        for (int j = var[col_index]; j <= i && col_index < size; col_index++, j = var[col_index]) {
            printf("%d ", matrix[(i - 1) * n + (j - 1)]);
        }
        printf("\n");
    }
}

 __device__ int get_sum(int* matrix, int n, int* var, int size) {

    int sum = 0;

    int row_index = 0, col_index = 0;
    for (int i = var[row_index]; row_index < size; col_index = 0, row_index++, i = var[row_index]) {
        for (int j = var[col_index]; j <= i && col_index < size; col_index++, j = var[col_index]) {
            sum += matrix[(i - 1) * n + (j - 1)];
        }
    }
    return sum;

}

 int host_get_sum(int* matrix, int n, int* var, int size) {

    int sum = 0;

    int row_index = 0, col_index = 0;
    for (int i = var[row_index]; row_index < size; col_index = 0, row_index++, i = var[row_index]) {
        for (int j = var[col_index]; j <= i && col_index < size; col_index++, j = var[col_index]) {
            sum += matrix[(i - 1) * n + (j - 1)];
        }
    }
    return sum;

}

__global__ void startCompution(int* matrix, int n, int* var, int size, int* min_sum) {

    int threadId = threadIdx.x;  // Идентификатор потока внутри блока
    int blockId = blockIdx.x;    // Идентификатор блока внутри сетки
    printf("Thread ID: %d, Block ID: %d\n", threadId, blockId);

}

void new_variate(int* count, int* matrix, int n, int* var, int size, int min, int max, int deep, int* result_var, int* min_sum, FILE* logfile, int* device_var, int* device_result_var, int* device_matrix, int* device_min_sum) {

    for (int i = min; i <= max - size + 1; i++) {
        if (deep < size) {
            var[deep] = i;
            new_variate(count, matrix, n, var, size, i + 1, max + 1, deep + 1, result_var, min_sum, logfile, device_var, device_result_var, device_matrix, device_min_sum);
        }
        else {

            (*count)++;

            if (*count >= MAX_COUNT) {
                
                cudaDeviceSynchronize();
                cudaMemcpy(var, device_var, sizeof(int) * size, cudaMemcpyHostToDevice);

                startCompution<<<1,1>>>(matrix, n, device_var, size, device_min_sum);
                *count = 0;

            }

            host_print_var(var, size);
            
            int sum = host_get_sum(matrix, n, var, size);
            if (sum < *min_sum) {
                *min_sum = sum;
                for (int j = 0; j < size; j++) result_var[j] = var[j];
            }

            break;
        }
    }

    if (deep == 0) {
        cudaDeviceSynchronize();
        cudaMemcpy(var, device_var, sizeof(int) * size, cudaMemcpyHostToDevice);

        startCompution<<<1,1>>>(matrix, n, device_var, size, device_min_sum);
        *count = 0;
    }

}




