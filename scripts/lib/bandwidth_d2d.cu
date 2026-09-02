// bandwidth_d2d.cu — MEASURE the device's memory bandwidth so the roofline
// ceiling stops being a vendor number (PP-23, PP-LLAMA-001 §2.4, §12 row 14).
//
// WHY MEASURED AND NOT VENDOR. §2.4's ceilings (~215 tok/s on RTX 4090,
// ~58 tok/s on GB10) are `[X]`: they divide the model size by a bandwidth
// figure taken from a spec sheet. A spec sheet is not an observation of this
// board, at this clock, with this driver — and PP-12 refuses an untagged
// vendor GB/s as a published figure. Until a `[V]` bandwidth is committed no
// percentage of a ceiling may be published at all.
//
// WHAT IS MEASURED. `cudaMemcpy(..., cudaMemcpyDeviceToDevice)` between two
// distinct 1 GiB device buffers. One copy moves BYTES out of DRAM and BYTES
// back into it, so the memory system sees 2*BYTES of traffic. The reported
// figure is that TRAFFIC divided by the elapsed time, which is the quantity
// comparable to a vendor peak (1008 GB/s on a 4090) and the quantity a decode
// step — which streams the whole model out of DRAM once per token — is
// bounded by. Reporting copy-size/time instead would halve the ceiling and
// make every decode rate look twice as close to it as it is.
//
// TIMING. CUDA events on the default stream, one synchronize per replicate.
//
// THE WARMUP IS A DURATION, NOT A COUNT, AND THAT IS THE WHOLE MEASUREMENT.
// Two untimed copies (~2 ms each) were the first version. They pay the context
// creation and the allocation faults, and they leave the board exactly where it
// was: idling in P8 at a 405 MHz memory clock. Measured on an RTX 4090 that way,
// n=15 came back TRIMODAL -- five replicates at ~673 GB/s, five at ~825, five at
// ~946 -- and the median landed on whichever cluster the ramp happened to be in.
// Two n=9 runs minutes apart reported 941.7 and 939.6 GB/s; the n=15 run reported
// 825.0. That is not a noisy measurement of one quantity, it is a clean
// measurement of the CLOCK RAMP, and taking its median is the bimodal-median
// trap: a summary statistic over a mixture describes neither mode.
//
// A decode loop runs the device continuously, so the quantity the roofline needs
// is the SUSTAINED bandwidth at a settled clock. The warmup therefore copies
// until at least `warmup_ms` of device time has elapsed, which drives the board
// to its steady P-state before the first timed replicate.
//
// EACH REPLICATE IS A BURST, NOT ONE COPY. A single 1 GiB D2D copy takes about
// 2.3 ms on an RTX 4090, which is short enough that one preemption by the
// display context ruins the sample. With the clock already pinned at its
// maximum (10501 MHz memory, read either side of the window), n=15 single-copy
// replicates still scattered 696-943 GB/s in no monotone order -- interleaved
// highs and lows, the signature of intermittent preemption rather than of a
// ramp. `copies_per_replicate` copies back to back inside ONE timed window, so
// each replicate measures a SUSTAINED burst and a stray preemption costs a few
// percent of a long window instead of half of a short one.
//
// Usage: bandwidth_d2d [replicates] [warmup_ms] [copies_per_replicate]
//        (defaults 5, 3000, 32; n >= 5)
// Output (stdout, one key per line, parsed by scripts/measure_bandwidth.sh):
//   device_name=<name>
//   bytes=<copy size in bytes>
//   traffic_bytes=<bytes moved through DRAM per replicate = 2 * bytes>
//   warmup_ms=<requested warmup duration>
//   warmup_copies=<copies the warmup actually issued>
//   copies_per_replicate=<copies inside each timed window>
//   replicate=<bytes per second>      (repeated `replicates` times)
// Any failure prints `error=<message>` on stderr and exits non-zero WITHOUT
// printing a partial replicate list: a half-measured bandwidth that a caller
// takes the median of is worse than no bandwidth at all.

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fail(const char *what, cudaError_t err) {
    fprintf(stderr, "error=%s: %s\n", what, cudaGetErrorString(err));
    return 1;
}

int main(int argc, char **argv) {
    int replicates = 5;
    int warmup_ms = 3000;
    if (argc > 1) {
        replicates = atoi(argv[1]);
    }
    int burst = 32;
    if (argc > 2) {
        warmup_ms = atoi(argv[2]);
    }
    if (argc > 3) {
        burst = atoi(argv[3]);
    }
    if (burst < 1) {
        fprintf(stderr, "error=copies_per_replicate must be >= 1 (got %d)\n", burst);
        return 2;
    }
    if (warmup_ms < 0) {
        fprintf(stderr, "error=warmup_ms must be >= 0 (got %d)\n", warmup_ms);
        return 2;
    }
    // n < 5 bounds no variance (§4.3). A caller that asks for fewer is asking
    // for a number that cannot carry a spread, so it is refused here rather
    // than reported with an honest-looking min/max over two points.
    if (replicates < 5) {
        fprintf(stderr, "error=replicates must be >= 5 (got %d)\n", replicates);
        return 2;
    }

    int ndev = 0;
    cudaError_t err = cudaGetDeviceCount(&ndev);
    if (err != cudaSuccess) return fail("cudaGetDeviceCount", err);
    if (ndev < 1) {
        fprintf(stderr, "error=no CUDA device present\n");
        return 2;
    }

    struct cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess) return fail("cudaGetDeviceProperties", err);

    const size_t bytes = (size_t)1 << 30; /* 1 GiB */
    void *src = NULL;
    void *dst = NULL;
    err = cudaMalloc(&src, bytes);
    if (err != cudaSuccess) return fail("cudaMalloc(src)", err);
    err = cudaMalloc(&dst, bytes);
    if (err != cudaSuccess) { cudaFree(src); return fail("cudaMalloc(dst)", err); }

    err = cudaMemset(src, 1, bytes);
    if (err != cudaSuccess) { cudaFree(src); cudaFree(dst); return fail("cudaMemset", err); }

    cudaEvent_t t0, t1;
    err = cudaEventCreate(&t0);
    if (err != cudaSuccess) { cudaFree(src); cudaFree(dst); return fail("cudaEventCreate", err); }
    err = cudaEventCreate(&t1);
    if (err != cudaSuccess) { cudaFree(src); cudaFree(dst); return fail("cudaEventCreate", err); }

    /* Warmup: copy until the device has been busy for warmup_ms, so the
       memory clock has left its idle P-state before anything is timed. Its
       failure is still fatal; a warmup that silently did nothing would put the
       ramp back inside the timed window. */
    double warmed = 0.0;
    int warm_iters = 0;
    while (warmed < (double)warmup_ms) {
        err = cudaEventRecord(t0, 0);
        if (err != cudaSuccess) { cudaFree(src); cudaFree(dst); return fail("warmup eventRecord", err); }
        err = cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice);
        if (err != cudaSuccess) { cudaFree(src); cudaFree(dst); return fail("warmup cudaMemcpy", err); }
        err = cudaEventRecord(t1, 0);
        if (err != cudaSuccess) { cudaFree(src); cudaFree(dst); return fail("warmup eventRecord", err); }
        err = cudaEventSynchronize(t1);
        if (err != cudaSuccess) { cudaFree(src); cudaFree(dst); return fail("warmup sync", err); }
        float wms = 0.0f;
        err = cudaEventElapsedTime(&wms, t0, t1);
        if (err != cudaSuccess) { cudaFree(src); cudaFree(dst); return fail("warmup elapsed", err); }
        warmed += (double)wms;
        warm_iters++;
        /* A zero-cost copy would spin forever; a device that reports no time
           for 1 GiB is broken, not fast. */
        if (warm_iters > 1000000) {
            cudaFree(src); cudaFree(dst);
            fprintf(stderr, "error=warmup did not accumulate time after %d copies\n", warm_iters);
            return 1;
        }
    }

    double *rates = (double *)malloc((size_t)replicates * sizeof(double));
    if (rates == NULL) {
        cudaFree(src); cudaFree(dst);
        fprintf(stderr, "error=out of host memory\n");
        return 1;
    }

    for (int i = 0; i < replicates; i++) {
        err = cudaEventRecord(t0, 0);
        if (err != cudaSuccess) { free(rates); cudaFree(src); cudaFree(dst); return fail("eventRecord", err); }
        for (int b = 0; b < burst; b++) {
            err = cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToDevice);
            if (err != cudaSuccess) { free(rates); cudaFree(src); cudaFree(dst); return fail("cudaMemcpy", err); }
        }
        err = cudaEventRecord(t1, 0);
        if (err != cudaSuccess) { free(rates); cudaFree(src); cudaFree(dst); return fail("eventRecord", err); }
        err = cudaEventSynchronize(t1);
        if (err != cudaSuccess) { free(rates); cudaFree(src); cudaFree(dst); return fail("eventSynchronize", err); }
        float ms = 0.0f;
        err = cudaEventElapsedTime(&ms, t0, t1);
        if (err != cudaSuccess) { free(rates); cudaFree(src); cudaFree(dst); return fail("eventElapsedTime", err); }
        if (!(ms > 0.0f)) {
            free(rates); cudaFree(src); cudaFree(dst);
            fprintf(stderr, "error=non-positive elapsed time %.6f ms\n", (double)ms);
            return 1;
        }
        rates[i] = (2.0 * (double)bytes * (double)burst) / ((double)ms / 1000.0);
    }

    printf("device_name=%s\n", prop.name);
    printf("bytes=%zu\n", bytes);
    printf("traffic_bytes=%zu\n", (size_t)2 * bytes);
    printf("warmup_ms=%d\n", warmup_ms);
    printf("warmup_copies=%d\n", warm_iters);
    printf("copies_per_replicate=%d\n", burst);
    for (int i = 0; i < replicates; i++) {
        printf("replicate=%.0f\n", rates[i]);
    }

    free(rates);
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    cudaFree(src);
    cudaFree(dst);
    return 0;
}
