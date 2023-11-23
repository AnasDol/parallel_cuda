#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <omp.h>

void print_var(int* var, int size);
void print_matrix(int** matrix, int n);
void print_triangle(int** matrix, int n, int* var, int size);

int get_sum(int** matrix, int n, int* var, int size);
void _variate(int** matrix, int n, int* var, int size, int min, int max, int deep, int* result_var, int* min_sum, FILE* logfile);

void save_log(FILE* logfile, int** matrix, int n, int* var, int size, int sum, int min_sum);

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

    int n = sqrt(count);

    int** matrix = (int**)malloc(n * sizeof(int*));
    for (int i = 0; i < n; i++) {
        matrix[i] = (int*)malloc(n * sizeof(int));
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            fread(&matrix[i][j], sizeof(int), 1, file);
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

    if (n <= 20) print_matrix(matrix, n);

    printf("Matrix size (1-%d): ", n);
    int size;
    if (!(scanf("%d", &size)==1 && size>=1 && size<=n)) {
        printf("Input error\n");
        return 0;
    }

    int min_sum = INT_MAX;
    int* var = (int*)malloc(size*sizeof(int));
    int* result_var = (int*)malloc(size*sizeof(int));

    double start = omp_get_wtime();

    _variate(matrix, n, var, size, 1, 8, 0, result_var, &min_sum, logfile);

    double finish = omp_get_wtime();

    printf("\n---------------Output----------------\n");
    printf("indexes:\n");
    print_var(result_var, size);
    printf("matrix:\n");
    print_triangle(matrix, n, result_var, size);
    printf("min_sum: %d\n", min_sum);
    printf("Time: %lf\n", finish - start);

    printf("\nIntermediate results is saved in the log file.\n", count);
    
    for (int i = 0;i<n;i++) {
        free(matrix);
    }
    free(matrix);
    free(var);
    free(result_var);

    fclose(logfile);
    
    
    return 0;
}

void print_var(int* var, int size) {
    printf("[ ");
    for (int i = 0; i < size; i++) {
        printf("%d ", var[i]);
    }
    printf("]");
    printf("\n");
}

void print_matrix(int** matrix, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%d ", matrix[i][j]);
        }
        printf("\n");
    }

}

void print_triangle(int** matrix, int n, int* var, int size) {

    int row_index = 0, col_index = 0;
    for (int i = var[row_index]; row_index < size; col_index = 0, row_index++, i = var[row_index]) {
        for (int j = var[col_index]; j <= i && col_index < size; col_index++, j = var[col_index]) {
            printf("%d ", matrix[i - 1][j - 1]);
        }
        printf("\n");
    }
}

int get_sum(int** matrix, int n, int* var, int size) {

    int sum = 0;

    int row_index = 0, col_index = 0;
    for (int i = var[row_index]; row_index < size; col_index = 0, row_index++, i = var[row_index]) {
        for (int j = var[col_index]; j <= i && col_index < size; col_index++, j = var[col_index]) {
            sum += matrix[i - 1][j - 1];
        }
    }
    return sum;

}

void _variate(int** matrix, int n, int* var, int size, int min, int max, int deep, int* result_var, int* min_sum, FILE* logfile) {

    for (int i = min; i <= max - size + 1; i++) {
        if (deep < size) {
            var[deep] = i;
            _variate(matrix, n, var, size, i + 1, max + 1, deep + 1, result_var, min_sum, logfile);

        }
        else {
            
            int sum = get_sum(matrix, n, var, size);
            if (sum < *min_sum) {
                *min_sum = sum;
                for (int j = 0; j < size; j++) result_var[j] = var[j];
            }
            save_log(logfile, matrix, n, var, size, sum, *min_sum);
            break;
        }
    }
}

void save_log(FILE* logfile, int** matrix, int n, int* var, int size, int sum, int min_sum) {
    if (logfile != NULL) {
        fprintf(logfile, "[ ");
        for (int i = 0; i < size; i++) {
            fprintf(logfile, "%d ", var[i]);
        }
        fprintf(logfile, "]\nsubmatrix:\n");
        int row_index = 0, col_index = 0;
        for (int i = var[row_index]; row_index < size; col_index = 0, row_index++, i = var[row_index]) {
            for (int j = var[col_index]; j <= i && col_index < size; col_index++, j = var[col_index]) {
                fprintf(logfile, "%d ", matrix[i - 1][j - 1]);
            }
            fprintf(logfile, "\n");
        }
        fprintf(logfile, "sum: %d\n", sum);
        fprintf(logfile, "min_sum: %d\n", min_sum);
    }
}