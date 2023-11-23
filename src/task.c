#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <omp.h>

void print_var(int* var, int size);
void print_matrix(int** matrix, int n);
void print_triangle(int** matrix, int n, int* var, int size);

int get_sum(int** matrix, int n, int* var, int size);
void _variate(int** matrix, int n, int* var, int size, int min, int max, int deep, int* result_var, int* min_sum);

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

    // int a;
    // fread(&a, sizeof(int), 1, file);
    // printf("a = %d", a);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            fread(&matrix[i][j], sizeof(int), 1, file);
        }
    }

    

    print_matrix(matrix, n);

    // char log[255];
    // strncpy(log, argv[1], 255);

    // FILE* logfile = fopen(log, "w+");
    // if (!logfile) {
    //     printf("Log file not found\n");
    //     return 0;
    // }

    // printf("Row number (>=2): ");
    // int m; // размер области по вертикали
    // if (!(scanf("%d", &m)==1 && m>=2)) {
    //     printf("Input error\n");
    //     return 0;
    // }

    // printf("Column number (>=2): ");
    // int n; // размер области по горизонтали
    // if (!(scanf("%d", &n)==1 && n>=2)) {
    //     printf("Input error\n");
    //     return 0;
    // }

    // printf("Cut radius (1<R<%.2lf): ", (double)min(m,n)/2);
    // double R; // радиус круглого выреза
    // if (!(scanf("%lf", &R)==1 && R>=1 && R<=(double)min(m,n)/2)) {
    //     printf("Input error\n");
    //     return 0;
    // }

    // double** temperature = (double**)malloc(m * sizeof(double*));
    // for (int i = 0; i < m; i++) {
    //     temperature[i] = (double*)malloc(n * sizeof(double));
    // }

    // initialize(temperature, m, n, R);

    // if (m <= 30 && n <= 15) {
    //     printf("Initial state:\n");
    //     for (int i = 0; i < m; i++) {
    //         for (int j = 0; j < n; j++) {
    //             printf("%7.3lf ", temperature[i][j]);
    //         }
    //         printf("\n");
    //     }
    // }

    // double start = omp_get_wtime();

    // int count = compute_temperature(temperature, m, n, R, logfile);

    // double finish = omp_get_wtime();

    // printf("\nOutput:\n");
    // for (int i = 0; i < n; i++) {
    //     printf("%.3lf\t", temperature[0][i]);
    // }
    // printf("\nIntermediate results is saved in the log file.\n", count);
    // printf("Time: %lf\n", finish - start);
    // printf("Count: %d\n", count);

    // // Вывод распределения температур
    // // for (int i = 0; i < m; i++) {
    // //     for (int j = 0; j < n; j++) {
    // //         if (pow(i-round((double)(m/2-1)), 2) + pow(j-round((double)(n/2)), 2) <= R*R) printf("   \t");
    // //         else printf("%.2f\t", temperature[i][j]);
    // //     }
    // //     printf("\n");
    // // }


    // for (int i = 0;i<m;i++) {
    //     free(temperature[i]);
    // }
    // free(temperature);

    //fclose(logfile);
    fclose(file);
    
    // return 0;
}

void print_var(int* var, int size) {
    for (int i = 0; i < size; i++) {
        printf("%d ", var[i]);
    }
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

    print_triangle(matrix, n, var, size);

    int sum = 0;

    int row_index = 0, col_index = 0;
    for (int i = var[row_index]; row_index < size; col_index = 0, row_index++, i = var[row_index]) {
        for (int j = var[col_index]; j <= i && col_index < size; col_index++, j = var[col_index]) {
            sum += matrix[i - 1][j - 1];
        }
    }
    return sum;

}

void _variate(int** matrix, int n, int* var, int size, int min, int max, int deep, int* result_var, int* min_sum) {

    for (int i = min; i <= max - size + 1; i++) {
        if (deep < size) {
            var[deep] = i;
            _variate(matrix, n, var, size, i + 1, max + 1, deep + 1, result_var, min_sum);

        }
        else {
            printf("indexes: ");
            print_var(var, size);
            int sum = get_sum(matrix, n, var, size);
            printf("sum: %d\n", sum);
            if (sum < *min_sum) {
                printf("Less!\n");
                *min_sum = sum;
                for (int j = 0; j < size; j++) result_var[j] = var[j];
            }
            break;
        }
    }
}