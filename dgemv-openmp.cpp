#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <job.in>

const char* dgemv_desc = "OpenMP dgemv.";

void my_dgemv(int n, double* A, double* x, double* y) {
    // Temporary array to hold result of A * X for each thread
    double* temp_y = (double*)malloc(n * sizeof(double));
    memset(temp_y, 0, n * sizeof(double));

    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int nthreads = OMP_NUM_THREADS;
        int chunk_size = n / nthreads;
        int start = thread_id * chunk_size;
        int end = (thread_id == nthreads - 1) ? n : start + chunk_size;

        // Matrix-vector multiplication
        for (int i = start; i < end; i++) {
            for (int j = 0; j < n; j++) {
                temp_y[i] += A[i * n + j] * x[j];
            }
        }
    }

    // Basic corresponding element addition
    for (int i = 0; i < n; i++) {
        y[i] += temp_y[i];
    }

    // Free the temp var
    free(temp_y);
}
