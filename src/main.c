#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

#define SIZE 3

void init(int* var, int size) {
    for (int i = 0; i < size; i++) {
        var[i] = i + 1;
    }
}

void print(int* var, int size) {
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
            print(var, size);
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

int main() {

    int n = 8;
    int** matrix = (int**)malloc(n * sizeof(int*));
    for (int i = 0; i < n; i++) {
        matrix[i] = (int*)malloc(n * sizeof(int));
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            matrix[i][j] = rand() % 10 + 1;
        }
    }

    print_matrix(matrix, n);

    int min_sum = INT_MAX;
    int var[SIZE];
    int result_var[SIZE];
    init(var, SIZE);
    _variate(matrix, n, var, SIZE, 1, 8, 0, result_var, &min_sum);

    printf("\n---------------Output----------------\n");
    printf("indexes:\n");
    print(result_var, SIZE);
    printf("matrix:\n");
    print_triangle(matrix, n, result_var, SIZE);
    printf("min_sum: %d\n", min_sum);

}