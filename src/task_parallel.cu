#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <omp.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define MAX_COUNT 10

void _variate(FILE* logfile,
                int matrix_rows_cols, int* temp_var, int* result_var, int var_size, int* min_sum, int* array,
                int* device_matrix, int* device_var, int* device_result_var, int* device_min_sum, int* device_array,
                int min, int max, int deep,
                int* count);

__global__ void compute(int* matrix, int matrix_rows_cols, int* array, int var_size, int* min_sum, int* result_var);

__device__ int get_sum(int* matrix, int matrix_rows_cols, int* temp_var, int var_size);

__device__ void save_log(FILE* logfile, int** matrix, int matrix_rows_cols, int* temp_var, int var_size, int sum, int min_sum);

void print_var(int* temp_var, int var_size);
void print_triangle(int* matrix, int matrix_rows_cols, int* temp_var, int var_size);
void print_array(int* array, int rows, int cols);

int main(int argc, char* argv[]) {

    if (argc < 3) {
        printf("Enter dataset and log filename as command prompt arguments\n");
        return 0;
    }

    char filename[255];
    strncpy(filename, argv[1], 255);

    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Dataset file not found\n");
        return 0;
    }

    fseek(file, 0, SEEK_END);
    long count = ftell(file) / sizeof(int);
    fseek(file, 0, SEEK_SET);

    printf("count: %d\n", count);

    int matrix_rows_cols = sqrt(count);

    int* matrix = (int*)malloc(matrix_rows_cols * matrix_rows_cols * sizeof(int));

    for (int i = 0; i < matrix_rows_cols; i++) {
        for (int j = 0; j < matrix_rows_cols; j++) {
            fread(&matrix[i * matrix_rows_cols + j], sizeof(int), 1, file);
        }
    }

    fclose(file);

    char log[255];
    strncpy(log, argv[2], 255);

    FILE* logfile = fopen(log, "w+");
    if (!logfile) {
        printf("Log file not found\n");
        return 0;
    }

    if (matrix_rows_cols <= 20) print_array(matrix, matrix_rows_cols, matrix_rows_cols);

    printf("Matrix size (1-%d): ", matrix_rows_cols);
    int var_size;
    if (!(scanf("%d", &var_size)==1 && var_size>=1 && var_size<=matrix_rows_cols)) {
        printf("Input error\n");
        return 0;
    }

    int* device_matrix;
    cudaMalloc((void**)&device_matrix, sizeof(int) * matrix_rows_cols * matrix_rows_cols);
    cudaMemcpy(device_matrix, matrix, sizeof(int) * matrix_rows_cols * matrix_rows_cols, cudaMemcpyHostToDevice);

    int* temp_var = (int*)malloc(var_size*sizeof(int));
    int* device_var;
    cudaMalloc((void**)&device_var, sizeof(int) * var_size);

    int* result_var = (int*)malloc(var_size*sizeof(int));
    int* device_result_var;
    cudaMalloc((void**)&device_result_var, sizeof(int) * var_size);

    int min_sum = INT_MAX;
    int* device_min_sum;
    cudaMalloc((void**)&device_min_sum, sizeof(int));
    cudaMemcpy(device_min_sum, &min_sum, sizeof(int), cudaMemcpyHostToDevice);

    int* array = (int*)malloc(sizeof(int)*MAX_COUNT*var_size);
    int* device_array;
    cudaMalloc((void**)&device_array, sizeof(int)*MAX_COUNT*var_size);
    
    int comb_count = 0; // счетчик числа сформированных комбинаций

    _variate(logfile, matrix_rows_cols, temp_var, result_var, var_size, &min_sum, array, device_matrix, device_var, device_result_var, device_min_sum, device_array, 1, matrix_rows_cols, 0, &comb_count);
    //_variate(&comb_count, matrix, n, temp_var, var_size, 1, n, 0, result_var, &min_sum, NULL, device_var, device_result_var, device_matrix, device_min_sum, array, device_array);

    // копируем результаты на хост
    cudaMemcpy(result_var, device_result_var, sizeof(int) * var_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(&min_sum, device_min_sum, sizeof(int), cudaMemcpyDeviceToHost);

    printf("\n---------------Output----------------\n");

    printf("indexes:\n");
    print_var(result_var, var_size); printf("\n");
    
    printf("submatrix:\n");
    print_triangle(matrix, matrix_rows_cols, result_var, var_size);

    printf("min_sum: %d\n", min_sum);


    free(matrix);
    free(temp_var);
    free(result_var);
    free(array);

    cudaFree(device_matrix);
    cudaFree(device_var);
    cudaFree(device_result_var);
    cudaFree(device_array);
    
    return 0;
}

void print_var(int* temp_var, int var_size) {
    printf("[ ");
    for (int i = 0; i < var_size; i++) {
        printf("%d ", temp_var[i]);
    }
    printf("]");
    printf("\n");
}

void print_matrix(int* matrix, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%d ", matrix[i * n + j]);
        }
        printf("\n");
    }

}


void print_triangle(int* matrix, int matrix_rows_cols, int* temp_var, int var_size) {

    int row_index = 0, col_index = 0;
    for (int i = temp_var[row_index]; row_index < var_size; col_index = 0, row_index++, i = temp_var[row_index]) {
        for (int j = temp_var[col_index]; j <= i && col_index < var_size; col_index++, j = temp_var[col_index]) {
            printf("%d ", matrix[(i - 1) * matrix_rows_cols + (j - 1)]);
        }
        printf("\n");
    }
}

 __device__ int get_sum(int* matrix, int matrix_rows_cols, int* temp_var, int var_size) {

    int sum = 0;

    int row_index = 0, col_index = 0;
    for (int i = temp_var[row_index]; row_index < var_size; col_index = 0, row_index++, i = temp_var[row_index]) {
        for (int j = temp_var[col_index]; j <= i && col_index < var_size; col_index++, j = temp_var[col_index]) {
            sum += matrix[(i - 1) * matrix_rows_cols + (j - 1)];
        }
    }
    return sum;

}

 int host_get_sum(int* matrix, int matrix_rows_cols, int* temp_var, int var_size) {

    int sum = 0;

    int row_index = 0, col_index = 0;
    for (int i = temp_var[row_index]; row_index < var_size; col_index = 0, row_index++, i = temp_var[row_index]) {
        for (int j = temp_var[col_index]; j <= i && col_index < var_size; col_index++, j = temp_var[col_index]) {
            sum += matrix[(i - 1) * matrix_rows_cols + (j - 1)];
        }
    }
    return sum;

}

void print_array(int* array, int rows, int cols) {
    for (int i = 0;i<rows;i++) {
        for (int j = 0;j<cols;j++) {
            printf("%d ", array[i * cols + j]);
        }
        printf("\n");
    }
}

__global__ void compute(int* matrix, int matrix_rows_cols, int* array, int var_size, int* min_sum, int* result_var) {

    int threadId = blockIdx.x * blockDim.x + threadIdx.x; // id текущего потока
    int N_threads = gridDim.x * blockDim.x; // всего потоков

    int array_index = threadId; // соответствующий id индекс

    // каждый поток обрабатывает несколько индексов, если это возможно
    while (array_index < MAX_COUNT) {

        if(array[array_index * var_size] == 0) break; // 0 - недопустимый номер строки

        int* temp_var = (int*) malloc (sizeof(int) * var_size);
        for (int i = 0 ; i < var_size; i++) {
            temp_var[i] = array[array_index * var_size + i]; 
        }

        int sum = get_sum(matrix, matrix_rows_cols, temp_var, var_size); // рассчитываем сумму элементов разреженной треугльной матрицы
        if (sum < *min_sum) {
            *min_sum = sum;
            for (int j = 0; j < var_size; j++) result_var[j] = temp_var[j];
        }

        printf("Process with id=%d computes array[%d]:\n  temp_var = %d %d %d\n  sum = %d\n", threadId, array_index, array[array_index * var_size + 0], array[array_index * var_size + 1],array[array_index * var_size + 2], sum);

        free(temp_var);

        array_index += N_threads; // следующий индекс

    }
}

void _variate(FILE* logfile,
                int matrix_rows_cols, int* temp_var, int* result_var, int var_size, int* min_sum, int* array,
                int* device_matrix, int* device_var, int* device_result_var, int* device_min_sum, int* device_array,
                int min, int max, int deep,
                int* count) {

    for (int i = min; i <= max - var_size + 1; i++) {

        // если комбинация построена не до конца (не до последнего числа)
        if (deep < var_size) {

            temp_var[deep] = i; // считаем очередное число

            // продолжаем рекурсию
             _variate(logfile, matrix_rows_cols, temp_var, result_var, var_size, min_sum, array, device_matrix, device_var, device_result_var, device_min_sum, device_array, i + 1, max + 1, deep + 1, count);
            //_variate(count, matrix, n, temp_var, var_size, i + 1, max + 1, deep + 1, result_var, min_sum, logfile, device_var, device_result_var, device_matrix, device_min_sum, array, device_array);
        
        }

        // иначе, если комбинация готова
        else {

            //сохраняем temp_var в array
            for (int j = 0; j < var_size; j++) {
                array[*count * var_size + j] = temp_var[j];
            }

            (*count)++; // увеличиваем значение счетчика построенных комбинаций

            // если построили уже MAX_COUNT комбинаций, сливаем их на обработку GPU
            if (*count >= MAX_COUNT) {

                cudaDeviceSynchronize(); // убеждаемся, что все процессы завершились

                // копируем массив комбинаций на девайс
                cudaMemcpy(device_array, array, sizeof(int) * var_size * MAX_COUNT, cudaMemcpyHostToDevice); 

                compute<<<3,10>>>(device_matrix, matrix_rows_cols, device_array, var_size, device_min_sum, device_result_var);

                // занулием массив
                for (int j = 0; j < MAX_COUNT; j++) {
                    for (int k = 0 ;k < var_size; k++) {
                        array[j*var_size + k] = 0;
                    }
                }

                *count = 0;

            }

            break;
        }
    }

    // если все комбинации уже построены, сливаем массив на обработку
    if (deep == 0) {

        cudaDeviceSynchronize(); // убеждаемся, что все процессы завершились

        // копируем массив комбинаций на девайс
        cudaMemcpy(device_array, array, sizeof(int) * var_size * MAX_COUNT, cudaMemcpyHostToDevice); 

        compute<<<3,10>>>(device_matrix, matrix_rows_cols, device_array, var_size, device_min_sum, device_result_var);

    }

}




