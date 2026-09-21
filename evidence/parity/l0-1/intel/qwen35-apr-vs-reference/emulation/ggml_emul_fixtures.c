// PMAT-3091 ggml vec_dot emulation: fixture generator that calls llama.cpp d1d3c3396's OWN ggml C.
//
// Links build/bin/libggml-cpu.so + libggml-base.so of the intel llama.cpp tree (GGML_NATIVE=ON, AVX-512 host),
// so every expected value below is produced by the exact ggml objects llama-side measurements ran.
// For each case: a deterministic input row x (splitmix64; the Rust port regenerates the same bytes) and
// deterministic weight blocks (random bytes with sane fp16 scales), then
//   quantize_row_q8_K_ref / quantize_row_q8_K (x86 = ref) / quantize_row_q8_K_generic
//   ggml_vec_dot_{q4_K,q5_K,q6_K}_q8_K_generic and the SIMD ggml_vec_dot_{..}_q8_K
//   quantize_row_q8_0 (x86 AVX2) / quantize_row_q8_0_ref / _generic, ggml_vec_dot_q8_0_q8_0(_generic)
//   ggml_quantize_mat_q8_K_4x8 (the REPACK q4_K_8x8 activation quantizer) de-interleaved vs the ref quants.
// Output: a TSV (stdout) and a Rust fixture table (argv[1]). fixtures.bin (argv[2]) holds every input row and
// block for audit. No threshold anywhere: equalities are reported, not judged.
// Build: see build_and_run.sh (gcc -O2 -ffp-contract=off so the harness's own float generation is plain IEEE).
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void ggml_cpu_init(void);
void quantize_row_q8_K_ref(const float *x, void *y, int64_t k);
void quantize_row_q8_K(const float *x, void *y, int64_t k);
void quantize_row_q8_K_generic(const float *x, void *y, int64_t k);
void quantize_row_q8_0(const float *x, void *y, int64_t k);
void quantize_row_q8_0_ref(const float *x, void *y, int64_t k);
void quantize_row_q8_0_generic(const float *x, void *y, int64_t k);
void ggml_quantize_mat_q8_K_4x8(const float *x, void *y, int64_t k);
typedef void vdot_t(int n, float *s, size_t bs, const void *vx, size_t bx, const void *vy, size_t by, int nrc);
vdot_t ggml_vec_dot_q4_K_q8_K, ggml_vec_dot_q4_K_q8_K_generic;
vdot_t ggml_vec_dot_q5_K_q8_K, ggml_vec_dot_q5_K_q8_K_generic;
vdot_t ggml_vec_dot_q6_K_q8_K, ggml_vec_dot_q6_K_q8_K_generic;
vdot_t ggml_vec_dot_q8_0_q8_0, ggml_vec_dot_q8_0_q8_0_generic;

enum { QK = 256, B_Q8K = 292, B_Q4K = 144, B_Q5K = 176, B_Q6K = 210, B_Q80 = 34, B_Q8KX4 = 1168 };
enum kind { UNIFORM, ZERO, MIXED, TIES, SIGNS, OUTLIERS };
struct fcase { int n; enum kind kind; float scale; const char *what; };
static const struct fcase CASES[] = {
    {256, UNIFORM, 1.0f, "uniform(-1,1)"},
    {1024, UNIFORM, 3.0f, "uniform(-3,3)"},
    {2048, UNIFORM, 0.05f, "uniform(-0.05,0.05)"},
    {3584, UNIFORM, 1.0f, "uniform(-1,1), 14 blocks (ffn_down width)"},
    {512, ZERO, 0.0f, "all zero"},
    {256, UNIFORM, 1e30f, "extreme magnitude uniform(-1e30,1e30)"},
    {768, MIXED, 0.0f, "block0 zero, block1 x1e30, block2 x1e-30"},
    {256, TIES, 0.0f, "x0=127, xj=(j%254)-126.5: exact .5 ties at iscale=-1"},
    {256, SIGNS, 0.0f, "uniform(-1,1) with x5=-2.5, x9=+2.5 (|max| tie, negative first)"},
    {1024, OUTLIERS, 0.0f, "uniform(-1,1), every 17th x40"},
};

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
static uint64_t fnv(const void *vp, size_t n) {
    const uint8_t *p = vp;
    uint64_t h = 0xcbf29ce484222325ULL;
    for (size_t i = 0; i < n; i++) { h ^= p[i]; h *= 0x100000001b3ULL; }
    return h;
}
static uint16_t f16gen(void) {
    uint64_t r = next64();
    uint16_t e = (uint16_t) (3 + (r >> 60) % 11);
    uint16_t m = (uint16_t) ((r >> 20) & 0x3FF);
    return (uint16_t) ((e << 10) | m);
}
static void fill(uint8_t *p, size_t n) { for (size_t i = 0; i < n; i++) p[i] = (uint8_t) (next64() >> 56); }

// One input element; consumes the generator exactly as the Rust mirror does (only the kinds that draw).
static float input_value(const struct fcase *c, int j) {
    switch (c->kind) {
        case UNIFORM: return unif(c->scale);
        case ZERO: return 0.0f;
        case MIXED: return j < QK ? 0.0f : unif(j < 2 * QK ? 1e30f : 1e-30f);
        case TIES: return j == 0 ? 127.0f : (float) (j % 254) - 126.5f;
        case SIGNS: return unif(1.0f);
        case OUTLIERS: { float v = unif(1.0f); return j % 17 == 0 ? v * 40.0f : v; }
    }
    return 0.0f;
}
static void gen_input(const struct fcase *c, float *x) {
    for (int j = 0; j < c->n; j++) x[j] = input_value(c, j);
    if (c->kind == SIGNS) { x[5] = -2.5f; x[9] = 2.5f; }
}
// K-quant weight blocks: random bytes, then sane fp16 d (and dmin) at d_off (and 2).
static void gen_k(uint8_t *w, int nb, int bsz, int d_off, int has_dmin) {
    for (int b = 0; b < nb; b++) {
        uint8_t *p = w + b * bsz;
        fill(p, bsz);
        uint16_t d = f16gen();
        memcpy(p + d_off, &d, 2);
        if (has_dmin) { uint16_t dm = f16gen(); memcpy(p + 2, &dm, 2); }
    }
}
static void gen_q80(uint8_t *w, int nb) {
    for (int b = 0; b < nb; b++) {
        uint8_t *p = w + b * B_Q80;
        fill(p, B_Q80);
        uint16_t d = f16gen();
        memcpy(p, &d, 2);
        for (int j = 2; j < B_Q80; j++) if (p[j] == 0x80) p[j] = 0x81;  // ggml weights never hold -128
    }
}
static uint32_t bits(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static float dot(vdot_t *f, int n, const void *w, const void *y) { float s = 0; f(n, &s, 0, w, 0, y, 0, 1); return s; }

// REPACK activation quantizer vs ref: de-interleave block_q8_Kx4 (4 identical rows) and count differences.
static void repack_check(const float *x, int n, const uint8_t *ref, int *qs_diff, int *d_diff, int *neg_eq) {
    int nb = n / QK;
    float *rows = malloc(4 * (size_t) n * sizeof(float));
    uint8_t *y = calloc((size_t) nb, B_Q8KX4);
    for (int r = 0; r < 4; r++) memcpy(rows + r * n, x, (size_t) n * sizeof(float));
    ggml_quantize_mat_q8_K_4x8(rows, y, n);
    *qs_diff = 0; *d_diff = 0; *neg_eq = 1;
    for (int i = 0; i < nb; i++) {
        const uint8_t *yb = y + i * B_Q8KX4, *rb = ref + i * B_Q8K;
        float dref; memcpy(&dref, rb, 4);
        int blk_neg = 0;
        for (int r = 0; r < 4; r++) {
            float dr; memcpy(&dr, yb + 4 * r, 4);
            if (memcmp(yb + 4 * r, rb, 4) != 0) { (*d_diff)++; blk_neg = 1; if (dr != -dref) *neg_eq = 0; }
        }
        for (int j = 0; j < QK * 4; j++) {
            int src_id = (j % 32) / 8, src_off = (j / 32) * 8 + (j % 8);
            (void) src_id;  // all four rows are the same input
            int8_t a = (int8_t) yb[16 + j], b = (int8_t) rb[4 + src_off];
            if (a != b) (*qs_diff)++;
            if (blk_neg ? a != -b : a != b) *neg_eq = 0;
        }
    }
    free(rows); free(y);
}

int main(int argc, char **argv) {
    if (argc != 3) { fprintf(stderr, "usage: %s <fixtures.rs.txt> <fixtures.bin>\n", argv[0]); return 2; }
    ggml_cpu_init();
    FILE *rs = fopen(argv[1], "w"), *bin = fopen(argv[2], "wb");
    if (!rs || !bin) { perror("open"); return 2; }
    printf("case\tn\twhat\tinput_fnv\tq8k_ref_fnv\tq8k_cpu_eq_ref\tq8k_generic_eq_ref"
           "\tq4k_w_fnv\tq4k_dot_generic\tq4k_dot_simd\tq5k_w_fnv\tq5k_dot_generic\tq5k_dot_simd"
           "\tq6k_w_fnv\tq6k_dot_generic\tq6k_dot_simd\tq80_w_fnv\tq80_cpu_fnv\tq80_ref_eq_cpu\tq80_generic_eq_cpu"
           "\tq80_dot_generic\tq80_dot_simd\trepack_q8kx4_qs_diff\trepack_q8kx4_d_diff\trepack_value_equal_ref\n");
    int ncase = (int) (sizeof(CASES) / sizeof(CASES[0]));
    for (int ci = 0; ci < ncase; ci++) {
        const struct fcase *c = &CASES[ci];
        int n = c->n, nb = n / QK, nb0 = n / 32;
        st = 0x3091000000000000ULL + (uint64_t) ci;
        float *x = malloc((size_t) n * sizeof(float));
        gen_input(c, x);
        uint8_t *q8k = calloc((size_t) nb, B_Q8K), *q8k_cpu = calloc((size_t) nb, B_Q8K), *q8k_gen = calloc((size_t) nb, B_Q8K);
        quantize_row_q8_K_ref(x, q8k, n);
        quantize_row_q8_K(x, q8k_cpu, n);
        quantize_row_q8_K_generic(x, q8k_gen, n);
        uint8_t *w4 = malloc((size_t) nb * B_Q4K), *w5 = malloc((size_t) nb * B_Q5K), *w6 = malloc((size_t) nb * B_Q6K), *w80 = malloc((size_t) nb0 * B_Q80);
        gen_k(w4, nb, B_Q4K, 0, 1);
        gen_k(w5, nb, B_Q5K, 0, 1);
        gen_k(w6, nb, B_Q6K, 208, 0);
        gen_q80(w80, nb0);
        uint8_t *q80 = calloc((size_t) nb0, B_Q80), *q80_ref = calloc((size_t) nb0, B_Q80), *q80_gen = calloc((size_t) nb0, B_Q80);
        quantize_row_q8_0(x, q80, n);
        quantize_row_q8_0_ref(x, q80_ref, n);
        quantize_row_q8_0_generic(x, q80_gen, n);
        float d4g = dot(ggml_vec_dot_q4_K_q8_K_generic, n, w4, q8k), d4s = dot(ggml_vec_dot_q4_K_q8_K, n, w4, q8k);
        float d5g = dot(ggml_vec_dot_q5_K_q8_K_generic, n, w5, q8k), d5s = dot(ggml_vec_dot_q5_K_q8_K, n, w5, q8k);
        float d6g = dot(ggml_vec_dot_q6_K_q8_K_generic, n, w6, q8k), d6s = dot(ggml_vec_dot_q6_K_q8_K, n, w6, q8k);
        float d8g = dot(ggml_vec_dot_q8_0_q8_0_generic, n, w80, q80), d8s = dot(ggml_vec_dot_q8_0_q8_0, n, w80, q80);
        int rq, rd, rneg;
        repack_check(x, n, q8k, &rq, &rd, &rneg);
        uint64_t hx = fnv(x, (size_t) n * 4), hq = fnv(q8k, (size_t) nb * B_Q8K);
        uint64_t h4 = fnv(w4, (size_t) nb * B_Q4K), h5 = fnv(w5, (size_t) nb * B_Q5K), h6 = fnv(w6, (size_t) nb * B_Q6K);
        uint64_t h80w = fnv(w80, (size_t) nb0 * B_Q80), h80q = fnv(q80, (size_t) nb0 * B_Q80);
        printf("%d\t%d\t%s\t%016" PRIx64 "\t%016" PRIx64 "\t%d\t%d\t%016" PRIx64 "\t%08x\t%08x\t%016" PRIx64 "\t%08x\t%08x"
               "\t%016" PRIx64 "\t%08x\t%08x\t%016" PRIx64 "\t%016" PRIx64 "\t%d\t%d\t%08x\t%08x\t%d\t%d\t%d\n",
               ci, n, c->what, hx, hq, memcmp(q8k, q8k_cpu, (size_t) nb * B_Q8K) == 0, memcmp(q8k, q8k_gen, (size_t) nb * B_Q8K) == 0,
               h4, bits(d4g), bits(d4s), h5, bits(d5g), bits(d5s), h6, bits(d6g), bits(d6s), h80w, h80q,
               memcmp(q80, q80_ref, (size_t) nb0 * B_Q80) == 0, memcmp(q80, q80_gen, (size_t) nb0 * B_Q80) == 0,
               bits(d8g), bits(d8s), rq, rd, rneg);
        fprintf(rs, "    Fixture { case: %d, n: %d, input_fnv: 0x%016" PRIx64 ", q8k_fnv: 0x%016" PRIx64
                    ", q4k: (0x%016" PRIx64 ", 0x%08x), q5k: (0x%016" PRIx64 ", 0x%08x), q6k: (0x%016" PRIx64 ", 0x%08x)"
                    ", q80: (0x%016" PRIx64 ", 0x%016" PRIx64 ", 0x%08x) },\n",
                ci, n, hx, hq, h4, bits(d4g), h5, bits(d5g), h6, bits(d6g), h80w, h80q, bits(d8g));
        fwrite(&n, 4, 1, bin); fwrite(x, 4, (size_t) n, bin); fwrite(q8k, B_Q8K, (size_t) nb, bin);
        fwrite(w4, B_Q4K, (size_t) nb, bin); fwrite(w5, B_Q5K, (size_t) nb, bin); fwrite(w6, B_Q6K, (size_t) nb, bin);
        fwrite(w80, B_Q80, (size_t) nb0, bin); fwrite(q80, B_Q80, (size_t) nb0, bin);
        free(x); free(q8k); free(q8k_cpu); free(q8k_gen); free(w4); free(w5); free(w6); free(w80); free(q80); free(q80_ref); free(q80_gen);
    }
    if (fclose(rs) != 0 || fclose(bin) != 0) { perror("close"); return 2; }
    return 0;
}
