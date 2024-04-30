const char* dgemv_desc = "Vectorized implementation of matrix-vector multiply.";

#include <omp.h>
// make report.txt

/*
 * This routine performs a dgemv operation
 * Y :=  A * X + Y
 * where A is n-by-n matrix stored in row-major format, and X and Y are n by 1 vectors.
 * On exit, A and X maintain their input values.
 */
void my_dgemv(int n, double* A_col_major, double* x, double* y) {
    for (int i = 0; i < n; i++) {
        double sum = 0.0;
        for (int j = 0; j < n; j++) {
            sum += A_col_major[j * n + i] * x[j]; // Access in column-major order
        }
        y[i] += sum;
    }
}

