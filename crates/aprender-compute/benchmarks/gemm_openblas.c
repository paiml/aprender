/* C OpenBLAS GEMM benchmark for cgp compete.
 * Compile: gcc -O3 -march=native -o gemm_openblas benchmarks/gemm_openblas.c -lopenblas -lm
 * Run: ./gemm_openblas [size]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* OpenBLAS cblas interface */
extern void cblas_sgemm(int order, int transa, int transb,
                        int m, int n, int k,
                        float alpha, const float *a, int lda,
                        const float *b, int ldb,
                        float beta, float *c, int ldc);
#define CblasRowMajor 101
#define CblasNoTrans 111

static double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    int size = 1024;
    if (argc > 1) size = atoi(argv[1]);

    int n = size;
    float *a = (float *)malloc(n * n * sizeof(float));
    float *b = (float *)malloc(n * n * sizeof(float));
    float *c = (float *)calloc(n * n, sizeof(float));

    /* Init with random data */
    srand(42);
    for (int i = 0; i < n * n; i++) {
        a[i] = (float)rand() / RAND_MAX - 0.5f;
        b[i] = (float)rand() / RAND_MAX - 0.5f;
    }

    /* Warmup */
    for (int i = 0; i < 3; i++) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    n, n, n, 1.0f, a, n, b, n, 0.0f, c, n);
    }

    /* Benchmark: min-of-10 */
    double best = 1e9;
    for (int i = 0; i < 10; i++) {
        double t0 = get_time_sec();
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    n, n, n, 1.0f, a, n, b, n, 0.0f, c, n);
        double elapsed = get_time_sec() - t0;
        if (elapsed < best) best = elapsed;
    }

    double ms = best * 1000.0;
    double gflops = 2.0 * n * n * n / best / 1e9;
    printf("C/OpenBLAS GEMM (%dx%dx%d): %.2f ms (%.1f GFLOPS)\n", n, n, n, ms, gflops);

    free(a); free(b); free(c);
    return 0;
}
