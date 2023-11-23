#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>

int main(int argc, char* argv[]) {

    if (argc < 2) {
        printf("Enter filename as command prompt argument\n");
        return 0;
    }

    char filename[255];
    strncpy(filename, argv[1], 255);

    FILE* file = fopen(filename, "wb");
    if (!file) {
        printf("Error\n");
        return 0;
    }

    printf("Enter size of matrix: ");
    int count;
    if (!(scanf("%d", &count)==1 && count>=1)) {
        printf("Input error\n");
        return 0;
    }


    int** matrix = (int**)malloc(count * sizeof(int*));
    for (int i = 0; i < count; i++) {
        matrix[i] = (int*) malloc (count * sizeof(int));
    }

    for (int i = 0; i < count; i++) {
        for (int j = 0; j < count; j++) {
            matrix[i][j] = rand() % 30 + 1;
        }
    }

    // int a = 5;
    // fwrite((void*)&a, sizeof(int), 1, file);


    for (int i = 0; i<count; i++) {
        for (int j = 0; j < count; j++) {
            fwrite((void*)&matrix[i][j], sizeof(int), 1, file);
        }
    }

    for (int i = 0;i<count;i++) {
        free(matrix[i]);
    }
    free(matrix);

    fclose(file);

    return 0;
}