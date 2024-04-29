const char* dgemv_desc = "Vectorized implementation of matrix-vector multiply.";

#include <omp.h>

/*
 * This routine performs a dgemv operation
 * Y :=  A * X + Y
 * where A is n-by-n matrix stored in row-major format, and X and Y are n by 1 vectors.
 * On exit, A and X maintain their input values.
 */
void my_dgemv(int n, double* A, double* x, double* y) {
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        double sum = 0.0;
        for (int j = 0; j < n; j += 4) {
            __builtin_prefetch(&A[i * n + j], 0, 3); // Prefetch A
            __builtin_prefetch(&x[j], 0, 3); // Prefetch x
            for (int k = 0; k < 4 && j + k < n; ++k) {
                sum += A[(j + k) * n + i] * x[j + k];
            }
        }
        y[i] += sum;
    }
}


