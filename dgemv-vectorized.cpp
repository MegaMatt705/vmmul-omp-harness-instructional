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
    double* temp = new double[n];
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            temp[j] = A[i * n + j] * x[j];
        }
        for (int j = 0; j < n; j++) {
            y[i] += temp[j];
        }
    }
    delete[] temp;
}


