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
//   --kv-type f16|f32        (PMAT-3091 kvconfig) sets common_params.cache_type_k AND cache_type_v, which
//                            common_context_params_to_llama copies to llama_context_params.type_k/type_v
//                            (common.cpp:1751-1752 at d1d3c3396). Absent = common's default (F16) or -ctk/-ctv.
//   --flash-attn on|off|auto (PMAT-3091 kvconfig) sets common_params.flash_attn_type ->
//                            llama_context_params.flash_attn_type (common.cpp:1742), enum llama_flash_attn_type
//                            (llama.h:190-193: AUTO=-1, DISABLED=0, ENABLED=1). Stripped here, so common's own
//                            -fa/--flash-attn never sees this spelling. Absent = common's default (AUTO).
//   Neither flag touches params when absent, so a no-flag run is the pre-kvconfig producer.
//   --raw-out PATH   (required)
//   --per-token      decode ONE token per llama_decode call (batch of 1, logits=1), in order,
//                    on the same context/KV, instead of one batched decode of the whole prompt.
//                    Same output format. Isolates the batched-vs-per-token kernel confound.
//   --dump-tensors DIR       (PMAT-3091 layerwise) install a ggml_backend_sched eval callback
//                            (llama_context_params.cb_eval / cb_eval_user_data, the mechanism of
//                            examples/eval-callback at d1d3c3396). Requires --per-token, so each
//                            decode step is exactly one position. Writes:
//                              DIR/tensor_names.tsv  every graph node name the callback is ASKED about,
//                                                    first-seen order, with op, type, ne (all positions);
//                              DIR/pos<P>/<name>.f32 raw little-endian float32 data for matching tensors
//                                                    at the dumped positions (ggml_backend_tensor_get);
//                              DIR/manifest.tsv      name, layer, pos, shape, n_bytes, file.
//                            sha256 of each .f32 is appended by the driver script (sha256sum).
//   --dump-positions LIST    comma-separated 0-based positions to dump (required with --dump-tensors).
//   --dump-regex RE          ECMAScript regex, full match, selecting tensors to dump. Default:
//                            model\.input_embed|inp_embd|l_out-[0-9]+|result_norm|result_output
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
#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <clocale>
#include <fstream>
#include <algorithm>
#include <map>
#include <regex>
#include <set>
#include <sstream>
#include <sys/stat.h>
#include <string>
#include <vector>

// ---- PMAT-3091 layerwise dump -------------------------------------------------------------
struct dump_state {
    std::string          dir;
    std::set<int32_t>    positions;
    std::regex           select;
    int32_t              cur_pos = -1;
    std::vector<std::string>            names_order;   // first-seen order
    std::map<std::string, std::string>  names_desc;    // name -> "op\ttype\tne"
    std::ostringstream   manifest;
    int                  n_written = 0;
    int                  n_errors  = 0;
};

static std::string ne_str(const ggml_tensor * t) {
    std::string r;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        r += std::to_string(t->ne[i]);
        if (i + 1 < GGML_MAX_DIMS) r += "x";
    }
    return r;
}

// layer index parsed from a "<base>-<il>" graph name; -1 when there is none
static int layer_of(const std::string & name) {
    const size_t dash = name.rfind('-');
    if (dash == std::string::npos || dash + 1 >= name.size()) return -1;
    for (size_t i = dash + 1; i < name.size(); ++i) {
        if (name[i] < '0' || name[i] > '9') return -1;
    }
    return std::atoi(name.c_str() + dash + 1);
}

static bool dump_cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * st = (dump_state *) user_data;
    const std::string name = t->name;
    if (ask) {
        if (st->names_desc.find(name) == st->names_desc.end()) {
            st->names_order.push_back(name);
            st->names_desc[name] = std::string(ggml_op_desc(t)) + "\t" + ggml_type_name(t->type) + "\t" + ne_str(t);
        }
        return st->positions.count(st->cur_pos) > 0 && std::regex_match(name, st->select);
    }
    // ask == false: data is computed and available
    if (!(st->positions.count(st->cur_pos) > 0 && std::regex_match(name, st->select))) {
        return true;
    }
    if (t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t)) {
        fprintf(stderr, "apr_raw_logits: dump: %s is %s contiguous=%d, not dumped\n", name.c_str(),
                ggml_type_name(t->type), (int) ggml_is_contiguous(t));
        st->n_errors++;
        return true;
    }
    const size_t n_bytes = ggml_nbytes(t);
    std::vector<float> buf(n_bytes / sizeof(float));
    ggml_backend_tensor_get(t, buf.data(), 0, n_bytes);
    const std::string pdir = st->dir + "/pos" + std::to_string(st->cur_pos);
    mkdir(pdir.c_str(), 0755);
    const std::string file = "pos" + std::to_string(st->cur_pos) + "/" + name + ".f32";
    std::ofstream f(st->dir + "/" + file, std::ios::binary | std::ios::trunc);
    f.write((const char *) buf.data(), (std::streamsize) n_bytes);
    f.close();
    if (!f) {
        fprintf(stderr, "apr_raw_logits: dump: write %s failed\n", file.c_str());
        st->n_errors++;
        return true;
    }
    st->manifest << name << "\t" << layer_of(name) << "\t" << st->cur_pos << "\t" << ne_str(t) << "\t" << n_bytes
                 << "\t" << file << "\n";
    st->n_written++;
    return true;
}
// -------------------------------------------------------------------------------------------

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    std::string raw_out;
    bool per_token = false;
    std::string dump_dir;
    std::string dump_positions;
    std::string dump_regex = "model\\.input_embed|inp_embd|l_out-[0-9]+|result_norm|result_output";
    std::string kv_type;
    std::string flash_attn;
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
        if (std::strcmp(argv[i], "--kv-type") == 0 || std::strcmp(argv[i], "--flash-attn") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "apr_raw_logits: %s needs a value\n", argv[i]);
                return 2;
            }
            std::string & dst = argv[i][2] == 'k' ? kv_type : flash_attn;
            dst = argv[++i];
            continue;
        }
        if (std::strcmp(argv[i], "--dump-tensors") == 0 || std::strcmp(argv[i], "--dump-positions") == 0 ||
            std::strcmp(argv[i], "--dump-regex") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "apr_raw_logits: %s needs a value\n", argv[i]);
                return 2;
            }
            std::string & dst = argv[i][7] == 't' ? dump_dir : (argv[i][7] == 'p' ? dump_positions : dump_regex);
            dst = argv[++i];
            continue;
        }
        args.push_back(argv[i]);
    }
    if (raw_out.empty()) {
        fprintf(stderr, "apr_raw_logits: --raw-out PATH is required\n");
        return 2;
    }
    dump_state dump;
    if (!dump_dir.empty()) {
        if (!per_token || dump_positions.empty()) {
            fprintf(stderr, "apr_raw_logits: --dump-tensors needs --per-token and --dump-positions\n");
            return 2;
        }
        dump.dir = dump_dir;
        mkdir(dump_dir.c_str(), 0755);
        std::stringstream ss(dump_positions);
        for (std::string tok; std::getline(ss, tok, ',');) {
            if (!tok.empty()) dump.positions.insert(std::atoi(tok.c_str()));
        }
        dump.select = std::regex(dump_regex, std::regex::ECMAScript);
    }

    common_params params;
    params.n_ctx  = 512;    // llama-perplexity's default
    params.escape = false;  // llama-perplexity sets this too

    common_init();
    if (!common_params_parse((int) args.size(), args.data(), params, LLAMA_EXAMPLE_PERPLEXITY)) {
        return 2;
    }
    if (!kv_type.empty()) {
        ggml_type t;
        if (kv_type == "f16") {
            t = GGML_TYPE_F16;
        } else if (kv_type == "f32") {
            t = GGML_TYPE_F32;
        } else {
            fprintf(stderr, "apr_raw_logits: --kv-type must be f16 or f32, got '%s'\n", kv_type.c_str());
            return 2;
        }
        params.cache_type_k = t;
        params.cache_type_v = t;
    }
    if (!flash_attn.empty()) {
        if (flash_attn == "on") {
            params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        } else if (flash_attn == "off") {
            params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        } else if (flash_attn == "auto") {
            params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
        } else {
            fprintf(stderr, "apr_raw_logits: --flash-attn must be on, off or auto, got '%s'\n", flash_attn.c_str());
            return 2;
        }
    }
    if (!kv_type.empty() || !flash_attn.empty()) {
        printf("apr_raw_logits: kvconfig cache_type_k=%s cache_type_v=%s flash_attn_type=%s\n",
               ggml_type_name(params.cache_type_k), ggml_type_name(params.cache_type_v),
               llama_flash_attn_type_name(params.flash_attn_type));
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

    if (!dump_dir.empty()) {
        params.cb_eval           = dump_cb_eval;
        params.cb_eval_user_data = &dump;
        params.warmup            = false;  // as examples/eval-callback: no warmup graph through the callback
    }
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
        dump.cur_pos = per_token ? start : -1;
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

    if (!dump_dir.empty()) {
        std::ofstream nf(dump_dir + "/tensor_names.tsv", std::ios::trunc);
        nf << "name\top\ttype\tne_at_first_sight\n";
        for (const auto & n : dump.names_order) {
            nf << n << "\t" << dump.names_desc[n] << "\n";
        }
        std::ofstream mf(dump_dir + "/manifest.tsv", std::ios::trunc);
        mf << "name\tlayer\tpos\tshape\tn_bytes\tfile\n" << dump.manifest.str();
        if (!nf || !mf) {
            fprintf(stderr, "apr_raw_logits: dump: cannot write tensor_names.tsv / manifest.tsv\n");
            return 6;
        }
        printf("apr_raw_logits: dump dir=%s names=%zu written=%d errors=%d\n", dump_dir.c_str(),
               dump.names_order.size(), dump.n_written, dump.n_errors);
        if (dump.n_errors != 0) {
            return 8;
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
