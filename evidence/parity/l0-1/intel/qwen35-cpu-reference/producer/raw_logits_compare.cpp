// raw_logits_compare — the falsifier for apr_raw_logits (PMAT-3303).
//
// Compares an APRRAWLG file against chunk 0 of a `llama-perplexity --save-all-logits`
// file. Chunk 0 must be the same tokens. llama-perplexity saves positions
// first = n_ctx/2 .. n_ctx-2, one row each: float scale, float min_log_prob, then
// n_vocab uint16 q (padded to an even count). Decoding: log_prob = min_log_prob + scale*q
// for q > 0; q == 0 means "at or below the clamp max_logit-16".
//
// Two checks per saved row:
//  (A) exact: re-encode our raw logits with the encoder copied VERBATIM from
//      tools/perplexity/perplexity.cpp@d1d3c3396 (nearest_int + log_softmax, l.72-107)
//      and count differing uint16 words (0 iff the logits reproduce perplexity's bit for bit
//      up to the encoder).
//  (B) tolerance: decode the file and compare with log_softmax(raw) computed in double.
//      q > 0:  |decoded - ours| <= tol_row, tol_row = 0.51*scale + 1e-5
//      q == 0: ours <= min_log_prob + tol_row  (clamped entry must really be at/below clamp)
// Usage: raw_logits_compare RAW.bin PERPLEXITY.kld
// Exit: 0 agree, 1 disagree, 2 I/O or format error.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <vector>

// ---- verbatim from tools/perplexity/perplexity.cpp @ d1d3c3396 -------------------------
static inline int nearest_int(float fval) {
    //assert(fval <= 4194303.f);
    float val = fval + 12582912.f;
    int i; memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}

static double log_softmax(int n_vocab, const float * logits, uint16_t * log_prob, int tok) {
    float max_logit = logits[0];
    float min_logit = logits[0];
    for (int i = 1; i < n_vocab; ++i) {
        max_logit = std::max(max_logit, logits[i]);
        min_logit = std::min(min_logit, logits[i]);
    }
    min_logit = std::max(min_logit, max_logit - 16);
    double sum_exp = 0.0;
    for (int i = 0; i < n_vocab; ++i) {
        sum_exp += expf(logits[i] - max_logit);
    }
    const float log_sum_exp = log(sum_exp);
    const float min_log_prob = min_logit - max_logit - log_sum_exp;
    const float scale = (max_logit - min_logit)/65535.f;
    float * d = (float *)log_prob;
    d[0] = scale;
    d[1] = min_log_prob;
    log_prob += 4;
    if (scale) {
        const float inv_scale = 1/scale;
        for (int i = 0; i < n_vocab; ++i) {
            log_prob[i] = logits[i] > min_logit ? nearest_int(inv_scale*(logits[i] - min_logit)) : 0;
        }
    } else {
        std::memset(log_prob, 0, n_vocab*sizeof(uint16_t));
    }
    return max_logit + log_sum_exp - logits[tok];
}
// ------------------------------------------------------------------------------------------

static bool read_all(const char * path, std::vector<char> & buf) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    buf.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    return true;
}

int main(int argc, char ** argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s RAW.bin PERPLEXITY.kld\n", argv[0]);
        return 2;
    }
    std::vector<char> raw, kld;
    if (!read_all(argv[1], raw) || !read_all(argv[2], kld)) {
        fprintf(stderr, "cannot read inputs\n");
        return 2;
    }
    if (raw.size() < 20 || std::memcmp(raw.data(), "APRRAWLG", 8) != 0) { fprintf(stderr, "bad raw magic\n"); return 2; }
    if (kld.size() < 20 || std::memcmp(kld.data(), "_logits_", 8) != 0) { fprintf(stderr, "bad kld magic\n"); return 2; }
    int32_t n_pos, rv, kn_ctx, kn_vocab, kn_chunk; uint32_t ver;
    std::memcpy(&ver, raw.data() + 8, 4); std::memcpy(&n_pos, raw.data() + 12, 4); std::memcpy(&rv, raw.data() + 16, 4);
    std::memcpy(&kn_ctx, kld.data() + 8, 4); std::memcpy(&kn_vocab, kld.data() + 12, 4); std::memcpy(&kn_chunk, kld.data() + 16, 4);
    const size_t raw_hdr = 20 + 4 * (size_t) n_pos;
    if (ver != 1 || raw.size() != raw_hdr + 4 * (size_t) n_pos * rv) { fprintf(stderr, "raw size/version mismatch\n"); return 2; }
    if (kn_vocab != rv || kn_ctx != n_pos) {
        fprintf(stderr, "shape mismatch: raw n_pos=%d n_vocab=%d, kld n_ctx=%d n_vocab=%d\n", n_pos, rv, kn_ctx, kn_vocab);
        return 1;
    }
    const int n_vocab = rv;
    const int nv = 2 * ((n_vocab + 1) / 2) + 4;
    const int first = n_pos / 2;
    const int n_rows = n_pos - 1 - first;
    const size_t k_tok = 20, k_data = 20 + 4 * (size_t) kn_chunk * kn_ctx;
    if (kld.size() != k_data + (size_t) kn_chunk * n_rows * nv * 2) { fprintf(stderr, "kld size mismatch\n"); return 2; }

    const int32_t * rtok = (const int32_t *) (void *) (raw.data() + 20);
    const int32_t * ktok = (const int32_t *) (void *) (kld.data() + k_tok);
    int tok_mismatch = 0;
    for (int k = 0; k < n_pos; ++k) tok_mismatch += rtok[k] != ktok[k];
    printf("tokens: n_pos=%d mismatches_vs_kld_chunk0=%d\n", n_pos, tok_mismatch);
    if (tok_mismatch) return 1;

    const float * rl = (const float *) (void *) (raw.data() + raw_hdr);
    std::vector<uint16_t> enc(nv);
    long long word_mismatch = 0, rows_identical = 0, n_cmp = 0, n_clamped = 0, viol_q = 0, viol_clamp = 0;
    double max_err = 0, max_err_over_scale = 0, max_tol = 0, max_clamp_excess = -1e30, nll = 0;
    int max_err_pos = -1, max_err_tok = -1;
    for (int r = 0; r < n_rows; ++r) {
        const int pos = first + r;
        const float * row = rl + (size_t) pos * n_vocab;
        const uint16_t * file = (const uint16_t *) (void *) (kld.data() + k_data + (size_t) r * nv * 2);

        std::fill(enc.begin(), enc.end(), 0);
        log_softmax(n_vocab, row, enc.data(), ktok[pos + 1]);
        long long wm = 0;
        for (int j = 0; j < nv; ++j) wm += enc[j] != file[j];
        word_mismatch += wm; rows_identical += wm == 0;

        double mx = row[0];
        for (int j = 1; j < n_vocab; ++j) mx = std::max(mx, (double) row[j]);
        double se = 0;
        for (int j = 0; j < n_vocab; ++j) se += std::exp((double) row[j] - mx);
        const double lse = std::log(se);
        float scale, minlp;
        std::memcpy(&scale, file, 4); std::memcpy(&minlp, file + 2, 4);
        const double tol = 0.51 * (double) scale + 1e-5;
        max_tol = std::max(max_tol, tol);
        nll -= (double) row[ktok[pos + 1]] - mx - lse;
        for (int j = 0; j < n_vocab; ++j) {
            const double ours = (double) row[j] - mx - lse;
            const uint16_t q = file[4 + j];
            if (q > 0) {
                const double dec = (double) minlp + (double) scale * q;
                const double e = std::fabs(dec - ours);
                ++n_cmp;
                if (e > max_err) { max_err = e; max_err_pos = pos; max_err_tok = j; }
                if (scale > 0) max_err_over_scale = std::max(max_err_over_scale, e / scale);
                viol_q += e > tol;
            } else {
                ++n_clamped;
                const double ex = ours - (double) minlp;
                max_clamp_excess = std::max(max_clamp_excess, ex);
                viol_clamp += ex > tol;
            }
        }
    }
    const long long total_words = (long long) n_rows * nv;
    printf("rows_compared=%d positions=%d..%d n_vocab=%d nv=%d\n", n_rows, first, n_pos - 2, n_vocab, nv);
    printf("A_reencode: uint16_words_differing=%lld of %lld, rows_byte_identical=%lld of %d\n",
           word_mismatch, total_words, rows_identical, n_rows);
    printf("B_decode: entries_q_gt_0=%lld max_abs_err=%.9e (pos=%d tok=%d) max_err_over_scale=%.6f max_tol=%.9e violations=%lld\n",
           n_cmp, max_err, max_err_pos, max_err_tok, max_err_over_scale, max_tol, viol_q);
    printf("B_clamped: entries_q_eq_0=%lld max(ours-min_log_prob)=%.9e violations=%lld\n", n_clamped, max_clamp_excess, viol_clamp);
    printf("chunk0_ppl_from_raw=%.4f (llama-perplexity printed [1]42.2513)\n", std::exp(nll / n_rows));
    const bool agree = viol_q == 0 && viol_clamp == 0;
    printf("verdict=%s\n", agree ? "AGREE" : "DISAGREE");
    return agree ? 0 : 1;
}
