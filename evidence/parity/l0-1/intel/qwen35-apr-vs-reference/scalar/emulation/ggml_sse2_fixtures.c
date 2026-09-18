// PMAT-3091 scalar: fixtures for the SSE2 vec ops the SCALAR ggml build still runs (scalar/sse2_ops.tsv), produced by
// calling llama.cpp d1d3c3396's OWN exported ggml_vec_silu_f32 / ggml_vec_swiglu_f32 / ggml_vec_soft_max_f32 from the
// scalar libggml-cpu.so (SSE2 body for the leading multiple of 4, libm tail). Inputs: splitmix64 (the Rust test
// regenerates them), special values given as u32 bit patterns. Output: TSV (stdout) + Rust fixture table (argv[1]).
// Build: see build_and_run_sse2_fixtures.sh (gcc -O2 -ffp-contract=off). No threshold: hashes are compared, not judged.
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

void ggml_cpu_init(void);
void ggml_vec_silu_f32(const int n, float *y, const float *x);
void ggml_vec_swiglu_f32(const int n, float *y, const float *x, const float *g);
double ggml_vec_soft_max_f32(const int n, float *y, const float *x, float max);

static const uint32_t SPECIAL[] = {
    0x00000000u, 0x80000000u, 0x00000001u, 0x80000001u, 0x000fffffu, 0x800fffffu, 0x7f800000u, 0xff800000u,
    0x7fc00000u, 0x42b0c28fu, 0x42b0cccdu, 0xc2cff0a4u, 0xc2d00000u, 0x42ae999au, 0x42aeccccu, 0xc2ae999au,
    0xc2aeccccu, 0x42850000u, 0x42860000u, 0xc2850000u, 0xc2860000u, 0x42c80000u, 0xc2c80000u, 0x7149f2cau,
    0xf149f2cau, 0x7f7fffffu, 0xff7fffffu, 0x00800000u, 0x80800000u, 0x3322bcc7u, 0xb322bcc7u, 0x41a00000u,
    0xc1a00000u, 0x3f000000u, 0xbf000000u, 0x3f800000u, 0xbf800000u, 0x3eb17218u, 0x42fe0000u, 0xc2fe0000u,
};
enum { NSPECIAL = sizeof(SPECIAL) / sizeof(SPECIAL[0]) };
// kinds: 0 uniform(-8,8), 1 uniform(-40,40), 2 special table cycled (silu/swiglu only), 3 uniform(-8,8) every 3rd -inf
// (softmax only: the causal mask), 4 uniform(-200,200)
static const int WIDTHS[] = {1, 3, 4, 5, 255, 256, 6144};

static uint64_t st;
static uint64_t next64(void) {
    uint64_t z = (st += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
static float unif(float scale) {
    float u = (float) (next64() >> 40) * (1.0f / 16777216.0f);
    return (u * 2.0f - 1.0f) * scale;
}
static uint64_t fnv(const void *p, size_t n) {
    const unsigned char *b = p;
    uint64_t h = 0xcbf29ce484222325ULL;
    for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 0x100000001b3ULL; }
    return h;
}
static void gen(float *x, int n, int kind) {
    for (int j = 0; j < n; j++) {
        switch (kind) {
        case 0: x[j] = unif(8.0f); break;
        case 1: x[j] = unif(40.0f); break;
        case 2: { uint32_t b = SPECIAL[j % NSPECIAL]; memcpy(&x[j], &b, 4); break; }
        case 3: x[j] = unif(8.0f); if (j % 3 == 2) x[j] = -INFINITY; break;
        default: x[j] = unif(200.0f); break;
        }
    }
}
struct buf { float *x, *g, *y; };

// One (op, kind, n) case: generate, call the scalar ggml C, emit a TSV row and a Rust fixture row.
static void run_case(int op, int kind, int n, int idx, struct buf b, FILE *rs) {
    static const char *OPS[] = {"silu", "swiglu", "soft_max"};
    uint64_t seed = 0x30910000ULL + (uint64_t) idx;
    st = seed;
    gen(b.x, n, kind);
    for (int j = 0; j < n; j++) b.g[j] = unif(2.0f);
    memset(b.y, 0, (size_t) n * 4);
    float max = -INFINITY;
    double sum = 0.0;
    if (op == 0) {
        ggml_vec_silu_f32(n, b.y, b.x);
    } else if (op == 1) {
        ggml_vec_swiglu_f32(n, b.y, b.x, b.g);
    } else {
        for (int j = 0; j < n; j++) max = max > b.x[j] ? max : b.x[j]; // ggml_vec_max_f32 (scalar build)
        sum = ggml_vec_soft_max_f32(n, b.y, b.x, max);
    }
    uint32_t mb; uint64_t sb; memcpy(&mb, &max, 4); memcpy(&sb, &sum, 8);
    uint64_t fi = fnv(b.x, (size_t) n * 4), fg = fnv(b.g, (size_t) n * 4), fo = fnv(b.y, (size_t) n * 4);
    printf("%s\t%d\t%d\t0x%" PRIx64 "\t0x%016" PRIx64 "\t0x%016" PRIx64 "\t0x%016" PRIx64 "\t0x%08" PRIx32 "\t0x%016" PRIx64 "\t",
           OPS[op], kind, n, seed, fi, fg, fo, mb, sb);
    for (int j = 0; j < 4 && j < n; j++) { uint32_t bb; memcpy(&bb, &b.y[j], 4); printf("%s%08" PRIx32, j ? "," : "", bb); }
    printf("\n");
    fprintf(rs, "    Sse2Fixture { op: %d, kind: %d, n: %d, seed: 0x%" PRIx64 ", in_fnv: 0x%016" PRIx64 ", g_fnv: 0x%016" PRIx64
                ", out_fnv: 0x%016" PRIx64 ", max_bits: 0x%08" PRIx32 ", sum_bits: 0x%016" PRIx64 " },\n",
            op, kind, n, seed, fi, fg, fo, mb, sb);
}

// kinds 2 (specials) and 3 (masked -inf) are meaningful for only one side: silu/swiglu vs soft_max.
static int skip_case(int op, int kind) {
    return (op < 2 && kind == 3) || (op == 2 && kind == 2);
}

int main(int argc, char **argv) {
    if (argc < 2) return 2;
    FILE *rs = fopen(argv[1], "w");
    if (!rs) return 3;
    ggml_cpu_init();
    struct buf b = {malloc(6144 * 4), malloc(6144 * 4), malloc(6144 * 4)};
    if (!b.x || !b.g || !b.y) return 4;
    printf("op\tkind\tn\tseed\tin_fnv\tg_fnv\tout_fnv\tmax_bits\tsum_bits\tout_first4_bits\n");
    int idx = 0;
    for (int op = 0; op < 3; op++) {
        for (int kind = 0; kind < 5; kind++) {
            if (skip_case(op, kind)) continue;
            for (size_t w = 0; w < sizeof(WIDTHS) / sizeof(WIDTHS[0]); w++, idx++) {
                run_case(op, kind, WIDTHS[w], idx, b, rs);
            }
        }
    }
    fclose(rs);
    return 0;
}
