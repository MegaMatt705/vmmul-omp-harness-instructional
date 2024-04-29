const char* dgemv_desc = "Vectorized implementation of matrix-vector multiply.";

#include <omp.h>
#include <stdlib.h>

/*
 * This routine performs a dgemv operation
 * Y :=  A * X + Y
 * where A is n-by-n matrix stored in row-major format, and X and Y are n by 1 vectors.
 * On exit, A and X maintain their input values.
 */
void my_dgemv_simd(int n, double* A, double* x, double* y) {
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        __m256d sum = _mm256_setzero_pd();
        for (int j = 0; j < n; j += 4) {
            __m256d a = _mm256_loadu_pd(&A[i * n + j]);
            __m256d x_vec = _mm256_loadu_pd(&x[j]);
            sum = _mm256_add_pd(sum, _mm256_mul_pd(a, x_vec));
        }
        double temp[4];
        _mm256_storeu_pd(temp, sum);
        y[i] += temp[0] + temp[1] + temp[2] + temp[3];
    }
}

