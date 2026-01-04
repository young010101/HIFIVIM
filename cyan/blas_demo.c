#include <stdio.h>
#include <cblas.h>

int main() {
    float A[6] = {1.0, 2.0, 3.0,
                  4.0, 5.0, 6.0}; // 2x3 matrix
    float B[6] = {7.0, 8.0, 9.0,
                  10.0, 11.0, 12.0}; // 3x2 matrix
    float C[4] = {0.0, 0.0,
                  0.0, 0.0}; // 2x2 result matrix
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                2, 2, 3,
                0.5, A, 3,
                B, 2,
                0.0, C, 2);
    for (int i = 0; i < 4; i++) {
        printf("%f ", C[i]);
    }
    printf("\n");
    return 0;
}
