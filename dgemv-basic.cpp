#include <iostream>

const char* dgemv_desc = "Basic implementation of matrix-vector multiply.";

/*
 * This routine performs a dgemv operation
 * Y := A * X + Y
 * where A is n-by-n matrix stored in row-major format, and X and Y are n by 1 vectors.
 * On exit, A and X maintain their input values.
 */
void my_dgemv(int n, double* A, double* x, double* y) {
    // Initialize a temporary variable to hold the sum of products
    double sum;

    // Loop through each row of the matrix A
    for (int i = 0; i < n; i++) {
        // Initialize the sum for the current row
        sum = 0.0;

        // Loop through each column of the matrix A
        for (int j = 0; j < n; j++) {
            // Multiply the current element of A by the corresponding element of X
            // and add it to the sum
            sum += A[i * n + j] * x[j];
        }

        // Add the sum to the corresponding element of Y
        y[i] += sum;
    }
}
