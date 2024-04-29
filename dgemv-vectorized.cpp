const char* dgemv_desc = "Vectorized implementation of matrix-vector multiply.";

#include <omp.h>

/*
 * This routine performs a dgemv operation
 * Y :=  A * X + Y
 * where A is n-by-n matrix stored in row-major format, and X and Y are n by 1 vectors.
 * On exit, A and X maintain their input values.
 */
void my_dgemv(int n, double* A, double* x, double* y) {
    const int blockSize = 4; // Tile size
    #pragma omp parallel for
    for (int i = 0; i < n; i += blockSize) {
        for (int j = 0; j < n; j += blockSize) {
            for (int ii = i; ii < i + blockSize && ii < n; ++ii) {
                double sum = 0.0;
                for (int jj = j; jj < j + blockSize && jj < n; ++jj) {
                    sum += A[ii * n + jj] * x[jj];
                }
                y[ii] += sum;
            }
        }
    }
}


