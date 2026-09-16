# Qwen3.5-0.8B Q4_K_M: raw float32 logits for all 78 prompt positions (intel, llama.cpp d1d3c3396), PMAT-3303

**tree:** aprender `0b6e2ffaa` (branch `PMAT-3303-qwen35-raw-logits`, cut from `PMAT-3303-qwen35-cpu-reference`) · llama.cpp `d1d3c3396` (`d1d3c3396aa13a5f239109a822666c4870490ad5`, `git status --porcelain --untracked-files=no` = 0 lines, before and after the build)

This replaces [`REFERENCE.md`](REFERENCE.md)'s `--save-all-logits` file as the basis the parity gate can use. That file
has only 38 of 78 positions, and it holds uint16-compressed log-softmax, not logits. The orchestrator decided (§8) to
**keep the 78-token corpus prompt**. The prompt was not lengthened.
**No threshold is set here.** `min_cosine` stays `[U]` and fail-closed (B5 undecided).

## Producer: new source (no existing target does this)

Targets checked at `d1d3c3396`:
- `examples/debug` (`llama-debug --save-logits`) decodes with `llama_batch_get_one`, which requests only the last output, and writes `llama_get_logits_ith(ctx, tokens.size()-1)`. That is one position.
- `examples/simple` and `examples/eval-callback` print tensors or generate; neither writes per-position logits.
- `tools/perplexity` writes compressed log-softmax for `n_ctx/2..n_ctx-2`.

None of them emits full float logits for every position, so a new program was written.

| field | value |
|---|---|
| source | [`producer/apr_raw_logits.cpp`](producer/apr_raw_logits.cpp) |
| falsifier source | [`producer/raw_logits_compare.cpp`](producer/raw_logits_compare.cpp). The encoder (`nearest_int` and `log_softmax(..., uint16_t*)`) is copied verbatim from `tools/perplexity/perplexity.cpp` l.72-107 |
| build | [`producer/build.sh`](producer/build.sh). It refuses a llama.cpp tree that is not `d1d3c3396` |
| driver | [`producer/run_n5.sh`](producer/run_n5.sh); full output in [`producer/run_n5-transcript.txt`](producer/run_n5-transcript.txt) |
| how it matches llama-perplexity | the same `common_params_parse(..., LLAMA_EXAMPLE_PERPLEXITY)`, `params.escape=false`, `common_init_from_params`, `common_tokenize(ctx, prompt, true)`, the BOS-overwrite rule, and `llama_memory_clear` before decode |
| how it differs from llama-perplexity | one context = the whole prompt. `batch.logits=1` for **every** position (perplexity sets it only for `pos >= n_ctx/2`). One `llama_decode`. Raw float32 rows are written |
| binary location (intel) | `~/src/llama.cpp-d1d3c3396/apr-raw-logits/`, a new untracked directory. `~/src/llama.cpp` was not touched |
| `apr_raw_logits` sha256 | `f3cca8cc39ac3a6c3ae8b6ab47699a49c77c210b4a203abebe30f5a56671474b` |
| `raw_logits_compare` sha256 | `c6f6f23e147b70f52745118f8bce31f275e6761aa2a23e18cecf122088614272` |
| library resolution | `ldd apr_raw_logits` resolves `libllama-common`, `libllama`, `libggml`, `libggml-base` and `libggml-cpu` all to `~/src/llama.cpp-d1d3c3396/build/bin/` (rpath) |

Build command, as `build.sh` runs it (with `L=~/src/llama.cpp-d1d3c3396`, the build tree described in REFERENCE.md: Release, GGML_NATIVE, CPU-only, shared libs):

```bash
c++ -O2 -std=c++17 -Wall -Wextra -I"$L/include" -I"$L/common" -I"$L/ggml/include" \
  apr_raw_logits.cpp -o "$L/apr-raw-logits/apr_raw_logits" \
  -L"$L/build/bin" -lllama-common -lllama -lggml -lggml-base -Wl,-rpath,"$L/build/bin"
c++ -O2 -std=c++17 -Wall -Wextra raw_logits_compare.cpp -o "$L/apr-raw-logits/raw_logits_compare"
```

Compiler: `c++ (Ubuntu 11.4.0-1ubuntu1~22.04.3) 11.4.0`. `build.sh` runs under `set -euo pipefail` and printed both sha256s, so both compiles succeeded.

## Output format (little-endian, no timestamps)

```
char[8] "APRRAWLG" | uint32 version=1 | int32 n_pos | int32 n_vocab | int32 token_ids[n_pos] | float32 logits[n_pos*n_vocab]
```

Row `i` holds the logits after position `i`, predicting token `i+1`. The size is `20 + 4·n_pos + 4·n_pos·n_vocab` = `20 + 312 + 77475840` = **77476172** bytes, which matches the `stat` on every run.

## Command (identical for runs 1..5), host, load

```bash
timeout 600 ~/src/llama.cpp-d1d3c3396/apr-raw-logits/apr_raw_logits \
  -m ~/models/Qwen3.5-0.8B-Q4_K_M.gguf -f /tmp/ref3303/prompt.txt -ngl 0 -t 8 -c 78 -b 78 \
  --raw-out ~/parity-ref/qwen35-0.8b-q4km-d1d3c3396/raw-run$i.bin < /dev/null > raw-run$i.log 2>&1
```

| field | value | source |
|---|---|---|
| host | `intel` (`hostname` = `mac-server`), Xeon W-3245, 32 logical CPUs | transcript |
| threads | `-t 8` (`llama threadpool init, n_threads = 8` in every run log) | `raw-run$i.log` |
| load | 1-min **96.54** at run 1 start → 79.59 at run 5 end (5-min 96.61 → 92.74). That is well above the 45–60 the brief assumed, from CI fleet runner load. 113.27 when the build started. One job at a time | `cut -d' ' -f1-3 /proc/loadavg` per run |
| window | 2026-09-15T17:06:15Z → 17:06:39Z, serial | `date -u` per run |
| model | `~/models/Qwen3.5-0.8B-Q4_K_M.gguf`, sha256 `bd258782e35f7f458f8aced1adc053e6e92e89bc735ba3be89d38a06121dc517` | `sha256sum` (transcript) |
| prompt file | `/tmp/ref3303/prompt.txt`, 445 bytes, sha256 `e6796dc1e57917920a2021896d74a35ae3f63bd12cb8dc1f63f2e2b178ba0eaa`. This is `$PROMPT` from `scripts/check_model_parity.sh:24`, as in REFERENCE.md. `-f` strips one trailing `\n` and the file has none | `sha256sum` (transcript) |

## Shape

| measurement | value | how |
|---|---|---|
| n_pos | **78** | producer stdout `n_pos=78`; file header |
| n_vocab | **248320** | producer stdout; equals the kld header `n_vocab` |
| add_bos | 0 (vocab adds none; no position spent) | producer stdout |
| **min_positions: 64** | **satisfiable against this file: 78 ≥ 64** | — |

Prompt token ids (78): identical to `tokens[0:78]` of `run1.kld` (the comparator checks this: `mismatches_vs_kld_chunk0=0`)

```
760,3841,13477,37550,33075,888,279,15217,5388,1345,279,12433,21250,34227,36165,23497,883,31108,4649,5638,11,4098,23122,11,15167,47410,11,9966,1452,6326,61369,11,321,279,26984,314,638,37309,290,42903,3808,2943,94408,17882,26,1396,13901,557,47193,11,1396,1898,557,21197,11,321,279,1830,9202,440,264,50802,314,50614,39663,22188,7123,421,995,310,1440,1518,279,4722,1362,381,35883,13
```

## Determinism, n = 5 (intel, CPU, `-t 8`, `-c 78 -b 78`)

| run | start_utc | rc | load at start | bytes | sha256 |
|---|---|---|---|---|---|
| 1 | 17:06:15Z | 0 | 96.54 96.61 70.65 | 77476172 | `1d5eaf129545aa6087cd6c49792f9774b61caa874fccefdd79e6164afa15bb93` |
| 2 | 17:06:20Z | 0 | 91.37 95.54 70.44 | 77476172 | `1d5eaf129545aa6087cd6c49792f9774b61caa874fccefdd79e6164afa15bb93` |
| 3 | 17:06:25Z | 0 | 91.37 95.54 70.44 | 77476172 | `1d5eaf129545aa6087cd6c49792f9774b61caa874fccefdd79e6164afa15bb93` |
| 4 | 17:06:29Z | 0 | 87.18 94.60 70.27 | 77476172 | `1d5eaf129545aa6087cd6c49792f9774b61caa874fccefdd79e6164afa15bb93` |
| 5 | 17:06:34Z | 0 | 83.56 93.73 70.12 | 77476172 | `1d5eaf129545aa6087cd6c49792f9774b61caa874fccefdd79e6164afa15bb93` |

**Byte-identical: 1 distinct sha256 over 5 runs**, under a load of 80–97. The verdict covers only this configuration.

## Durable output (NOT in git: 77.5 MB)

`intel:~/parity-ref/qwen35-0.8b-q4km-d1d3c3396/qwen35-0.8b-q4km-d1d3c3396.rawlogits.bin`, **77476172** bytes,
sha256 `1d5eaf129545aa6087cd6c49792f9774b61caa874fccefdd79e6164afa15bb93` (checked again after the move). It is run 1, renamed. Runs 2–5 were
byte-identical copies and were deleted. The same directory keeps `raw-run{1..5}.log` and `run_n5.transcript`.
It lives on `/` (`/dev/nvme0n1p2`), not `/tmp`. Re-derive it with the command above and check the sha256.

## Falsifier: against llama-perplexity's own saved log-softmax

`raw_logits_compare qwen35-0.8b-q4km-d1d3c3396.rawlogits.bin /tmp/ref3303/run1.kld` (intel; the `.kld` is REFERENCE.md's run 1, sha256 `609e0481…54ef`).
Rows compared: chunk 0, positions **39..76** (38 rows × 248320). Chunk 0 is `tok(prompt)` exactly (REFERENCE.md).

### Encoding (read from `tools/perplexity/perplexity.cpp@d1d3c3396` l.72-107 and l.144-173)

Each row is `nv = 2*((n_vocab+1)/2)+4 = 248324` uint16 words:
- word 0..1: `float scale = (max_logit - min_logit)/65535`
- word 2..3: `float min_log_prob = min_logit - max_logit - log_sum_exp`
- then n_vocab words of `q`

Here `min_logit` is clamped up to `max_logit - 16`, and `log_sum_exp = (float) log(Σ expf(l - max))`.
- Encoding: `q = nearest_int((l - min_logit)/scale)` if `l > min_logit`, else `0`.
- Decoding: `log_prob = min_log_prob + scale·q` for `q > 0`. `q = 0` only means "at or below the clamp".

### Tolerance, and how it was derived

- Rounding to the nearest int gives `|err| ≤ scale/2`. Since `scale ≤ 16/65535`, that bound is ≤ 1.2207e-4.
- `min_log_prob` and `log_sum_exp` are float32 values of magnitude ≲ 40, and `lse` is summed from `expf`. Their representation and rounding add ~1e-6 to 1e-5 in absolute terms. The float32 product `inv_scale·(l-min)` is accurate to ~6e-8 relative, which is < 0.004 of a step at `q ≤ 65535`.
- Per-row tolerance used: **`tol = 0.51·scale + 1e-5`**. It is 1.345e-4 at the largest scale observed.
- Entries with `q > 0`: need `|decoded − log_softmax_double(raw)| ≤ tol`.
- Entries with `q = 0`: need `log_softmax_double(raw) ≤ min_log_prob + tol`. A clamped entry has to really sit at or below the clamp. Its decoded value is a floor, not a value, so it is not compared as one.

### Result

| check | result |
|---|---|
| (A) re-encode the raw logits with the verbatim encoder, compare every uint16 word | **0 of 9,436,312 words differ; 38/38 rows byte-identical.** The raw logits reproduce what llama-perplexity encoded, bit for bit up to the encoder |
| (B) decode, `q > 0` (830,268 entries) | **max abs err 1.234967e-04** (pos 73, token 7926), max err/scale 0.5058, tolerance max 1.345136e-04, **0 violations** |
| (B) clamped, `q = 0` (8,605,892 entries) | max(ours − min_log_prob) = 1.198389e-04, **0 violations** |
| chunk-0 PPL recomputed from raw logits (38 positions) | **42.2513**, the same as llama-perplexity's printed `[1]42.2513` |
| verdict | **AGREE** (`raw_logits_compare` rc 0) |

The max error/scale of 0.5058 is just above the pure-rounding 0.5. The excess is ≈ 0.0058·scale ≈ 1.4e-6, which is the float32 `min_log_prob` term. It is covered by the `+1e-5`. Check (A) is the stronger statement: at the 38 overlapping positions, the producer's logits are exactly the logits llama-perplexity computed, even though perplexity requested 39 outputs and this producer requests 78.

## Open items

| what | status | command that would measure it |
|---|---|---|
| positions 0..38 and 77 are not covered by the falsifier (perplexity never saved them) | `[U]`: same decode call, same rows buffer, no independent check | for a position k: `llama-debug -m <model> -p "<detokenized tok[0:k+1]>" --save-logits -t 8` (it saves the last position only). Compare with row k at tolerance, n ≥ 3 positions spread over 0..38 and 77 |
| determinism at other `-t` (1, 16, 32) | `[U]` | `run_n5.sh` with `-t 1` / `-t 16`, compare sha256 with `1d5eaf12…bb93` |
| cross-host agreement (lambda / gx10 CPU at `d1d3c3396`) | `[U]` | `build.sh` + `run_n5.sh` on that host, same model sha256, then `cmp` against the intel file |
| an apr-side reader of `APRRAWLG` for the parity gate | `[U]`: not in this phase's scope | — |
| `min_cosine` for Qwen3.5 | `[U]`, fail-closed: B5 undecided; **not set, by design** | — |
