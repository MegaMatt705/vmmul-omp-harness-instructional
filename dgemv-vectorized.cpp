const char* dgemv_desc = "Vectorized implementation of matrix-vector multiply.";

#include <omp.h>
// make report.txt

/*
 * This routine performs a dgemv operation
 * Y :=  A * X + Y
 * where A is n-by-n matrix stored in row-major format, and X and Y are n by 1 vectors.
 * On exit, A and X maintain their input values.
 */
void my_dgemv(int n, double* A, double* x, double* y) {
    const int blockSize = 64; // Choose a block size that fits into cache
    for (int ii = 0; ii < n; ii += blockSize) {
        for (int jj = 0; jj < n; jj += blockSize) {
            for (int i = ii; i < ii + blockSize && i < n; ++i) {
                double sum = 0.0;
                for (int j = jj; j < jj + blockSize && j < n; ++j) {
                    sum += A[i * n + j] * x[j];
                }
                y[i] += sum;
            }
        }
    }
}
