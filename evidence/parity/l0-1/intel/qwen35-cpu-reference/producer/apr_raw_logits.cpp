// apr_raw_logits — write float32 logits for EVERY prompt position (PMAT-3303).
//
// Built against llama.cpp d1d3c3396 (libllama + libllama-common). Context setup goes
// through common_params_parse(..., LLAMA_EXAMPLE_PERPLEXITY) and common_init_from_params,
// the same path llama-perplexity takes, so `-m -f -c -b -t -ngl` mean exactly what they
// mean there. Differences from llama-perplexity, all deliberate:
//   * one context = the whole tokenized prompt (no 2*n_ctx requirement, no chunking);
//   * batch.logits = 1 for every position (perplexity sets it only for pos >= n_ctx/2);
//   * raw float32 logits are written, not uint16-compressed log-softmax.
//
// Extra flags (stripped before common parsing):
//   --raw-out PATH   (required)
//   --per-token      decode ONE token per llama_decode call (batch of 1, logits=1), in order,
//                    on the same context/KV, instead of one batched decode of the whole prompt.
//                    Same output format. Isolates the batched-vs-per-token kernel confound.
//
// Output (little-endian, no timestamps, so identical logits => identical bytes):
//   char[8]  "APRRAWLG"
//   uint32   version = 1
//   int32    n_pos
//   int32    n_vocab
//   int32    token_ids[n_pos]
//   float32  logits[n_pos * n_vocab]   row i = logits after position i
#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <clocale>
#include <fstream>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    std::string raw_out;
    bool per_token = false;
    std::vector<char *> args;
    for (int i = 0; i < argc; ++i) {
        if (std::strcmp(argv[i], "--per-token") == 0) {
            per_token = true;
            continue;
        }
        if (std::strcmp(argv[i], "--raw-out") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "apr_raw_logits: --raw-out needs a PATH\n");
                return 2;
            }
            raw_out = argv[++i];
            continue;
        }
        args.push_back(argv[i]);
    }
    if (raw_out.empty()) {
        fprintf(stderr, "apr_raw_logits: --raw-out PATH is required\n");
        return 2;
    }

    common_params params;
    params.n_ctx  = 512;    // llama-perplexity's default
    params.escape = false;  // llama-perplexity sets this too

    common_init();
    if (!common_params_parse((int) args.size(), args.data(), params, LLAMA_EXAMPLE_PERPLEXITY)) {
        return 2;
    }
    // llama-perplexity: n_parallel = max(1, n_batch / n_ctx). This tool supports n_seq = 1 only.
    const int32_t n_ctx = params.n_ctx;
    if (n_ctx <= 0 || params.n_batch / n_ctx > 1) {
        fprintf(stderr, "apr_raw_logits: need n_ctx > 0 and n_batch/n_ctx <= 1 (n_seq = 1); got n_ctx=%d n_batch=%d\n",
                n_ctx, params.n_batch);
        return 2;
    }
    params.n_parallel = 1;
    params.n_batch = std::min(params.n_batch, params.n_ctx);

    llama_backend_init();
    llama_numa_init(params.numa);

    auto llama_init = common_init_from_params(params);
    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();
    if (model == nullptr || ctx == nullptr) {
        fprintf(stderr, "apr_raw_logits: failed to load model or create context\n");
        return 3;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);

    // Same call llama-perplexity makes.
    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, true);
    const int32_t n_pos = (int32_t) tokens.size();
    if (n_pos == 0 || n_pos > n_ctx || n_pos > params.n_batch) {
        fprintf(stderr, "apr_raw_logits: prompt has %d tokens; need 1..min(n_ctx=%d, n_batch=%d)\n",
                n_pos, n_ctx, params.n_batch);
        return 4;
    }

    // llama-perplexity overwrites token 0 with BOS when the vocab adds one; mirror it.
    const bool add_bos = llama_vocab_get_add_bos(vocab);
    std::vector<llama_token> ids(tokens.begin(), tokens.end());
    if (add_bos) {
        ids[0] = llama_vocab_bos(vocab);
    }
    std::vector<float> logits((size_t) n_pos * (size_t) n_vocab);

    llama_memory_clear(llama_get_memory(ctx), true);
    const int32_t n_batch_tokens = per_token ? 1 : n_pos;
    llama_batch batch = llama_batch_init(n_batch_tokens, 0, 1);
    for (int32_t start = 0; start < n_pos; start += n_batch_tokens) {
        for (int32_t j = 0; j < n_batch_tokens; ++j) {
            batch.token[j]     = ids[start + j];
            batch.pos[j]       = start + j;
            batch.n_seq_id[j]  = 1;
            batch.seq_id[j][0] = 0;
            batch.logits[j]    = 1;
        }
        batch.n_tokens = n_batch_tokens;
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "apr_raw_logits: llama_decode failed at position %d\n", start);
            llama_batch_free(batch);
            return 5;
        }
        llama_synchronize(ctx);
        for (int32_t j = 0; j < n_batch_tokens; ++j) {
            const float * row = llama_get_logits_ith(ctx, j);
            if (row == nullptr) {
                fprintf(stderr, "apr_raw_logits: no logits for position %d\n", start + j);
                llama_batch_free(batch);
                return 7;
            }
            std::memcpy(logits.data() + (size_t) (start + j) * (size_t) n_vocab, row, (size_t) n_vocab * sizeof(float));
        }
    }

    const std::string tmp = raw_out + ".partial";
    std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
    if (!out) {
        fprintf(stderr, "apr_raw_logits: cannot open %s\n", tmp.c_str());
        llama_batch_free(batch);
        return 6;
    }
    const uint32_t version = 1;
    out.write("APRRAWLG", 8);
    out.write((const char *) &version, sizeof(version));
    out.write((const char *) &n_pos, sizeof(n_pos));
    out.write((const char *) &n_vocab, sizeof(n_vocab));
    for (int32_t k = 0; k < n_pos; ++k) {
        const int32_t id = ids[k];
        out.write((const char *) &id, sizeof(id));
    }
    out.write((const char *) logits.data(), (std::streamsize) logits.size() * sizeof(float));
    out.close();
    if (!out || std::rename(tmp.c_str(), raw_out.c_str()) != 0) {
        fprintf(stderr, "apr_raw_logits: write/rename to %s failed\n", raw_out.c_str());
        llama_batch_free(batch);
        return 6;
    }

    printf("apr_raw_logits: n_pos=%d n_vocab=%d add_bos=%d mode=%s out=%s\n", n_pos, n_vocab, add_bos ? 1 : 0,
           per_token ? "per-token" : "batched", raw_out.c_str());
    printf("apr_raw_logits: token_ids=");
    for (int32_t k = 0; k < n_pos; ++k) {
        printf("%s%d", k ? "," : "", (int) ids[k]);
    }
    printf("\n");

    llama_batch_free(batch);
    llama_backend_free();
    return 0;
}
