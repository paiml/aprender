# Qwen3.5-0.8B Q4_K_M — llama.cpp CPU reference (intel), PMAT-3303

**tree:** aprender `03e301475` (branch `PMAT-3303-qwen35-cpu-reference`, cut from `PMAT-3329-llama-pin-bump`) · llama.cpp `d1d3c3396` (`d1d3c3396aa13a5f239109a822666c4870490ad5`, `git status --porcelain --untracked-files=no` = 0 lines)

This is the reference that `apr` (#3091) will later be measured against. **It sets no threshold.**
The `min_cosine` row stays `[U]` and fail-closed: the known-bad half of the basis (B5) is undecided.

## Host

| field | value | command |
|---|---|---|
| host | `intel` (`hostname` = `mac-server`), Intel Xeon W-3245 @ 3.20GHz, 32 logical CPUs | `hostname; lscpu; nproc` |
| load during the runs | 1-min 55.57 at run 1 start → 60.81 at run 5 end (5-min 49.16 → 50.86). CI fleet host under runner load | `cut -d' ' -f1-3 /proc/loadavg` before and after each run |
| threads | `-t 8` (`system_info: n_threads = 8 (n_threads_batch = 8) / 32`) | run log |
| window | 2026-09-15T16:55:52Z → 16:56:16Z, 5 runs serially, one job at a time | `date -u` per run |

## Comparator build

| field | value |
|---|---|
| source | `~/src/llama.cpp-d1d3c3396` at `d1d3c3396` ("ci: build MUSA for only 1 arch (#28944)") |
| CMake | `CMAKE_BUILD_TYPE=Release`, `GGML_NATIVE=ON`, `GGML_CUDA=OFF`, `GGML_BLAS=OFF`, `GGML_OPENMP=ON`, `GGML_LLAMAFILE=ON`, `BUILD_SHARED_LIBS=ON`, compiler `/usr/bin/c++` (from `build/CMakeCache.txt`) |
| CPU features reported | `SSE3 SSSE3 AVX AVX2 F16C FMA BMI2 AVX512 AVX512_VNNI LLAMAFILE OPENMP REPACK` (run log `system_info`) |
| `llama-perplexity` sha256 | `0222a898db4e1dda9e19c6e76bbb591ff19dd8dcec1f86fa5851bd9e54e16efd` |
| `libllama-perplexity-impl.so` sha256 | `aa31191a2692879a92c6fb5661e75301fd1fa1b42edff8d0864b334cfaa30bd1` |
| library resolution | `ldd` resolves every `libllama*`/`libggml*` to `~/src/llama.cpp-d1d3c3396/build/bin/`, never to `~/src/llama.cpp` (transcript) |
| host mutation | one target built: `cmake --build . --target llama-tokenize -j 4` in `~/src/llama.cpp-d1d3c3396/build` (rc 0) |

## Model

`~/models/Qwen3.5-0.8B-Q4_K_M.gguf`, 532517120 bytes,
sha256 `bd258782e35f7f458f8aced1adc053e6e92e89bc735ba3be89d38a06121dc517` (`sha256sum`, intel).
n_vocab = **248320** (derived from the saved-file size below; the tool prints no vocab line at this verbosity).

## Prompt, token count and min_positions

The prompt is `$PROMPT` from `scripts/check_model_parity.sh:24`, written byte-for-byte with
`printf '%s'` (445 bytes, no trailing newline, sha256 `e6796dc1e57917920a2021896d74a35ae3f63bd12cb8dc1f63f2e2b178ba0eaa`).

| measurement | value | command (intel) |
|---|---|---|
| tokens under the qwen35 tokenizer | **78** | `llama-tokenize -m <model> -f prompt.txt --ids --log-disable` |
| tokens with `--no-bos` | 78. The vocab adds no BOS, so no position is spent on one | same + `--no-bos` |
| min_positions: 64 against the **prompt** (the apr side: 78 prompt positions) | **satisfiable**: 78 ≥ 64 | — |
| min_positions: 64 against **this reference file** | **NOT satisfiable**. See the next section | source reading + file-size arithmetic |

## What llama-perplexity requires, and what it saves

Read from `tools/perplexity/perplexity.cpp` at `d1d3c3396`:

1. `perplexity()` refuses any input with `tokens.size() < 2*n_ctx` (l.480). `n_chunk = tokens.size() / n_ctx`.
2. `--save-all-logits` does **not** write raw logits. For each chunk it writes `process_logits` output:
   a **uint16-compressed log-softmax** row of `nv = 2*((n_vocab+1)/2)+4` entries (l.144-173).
3. It writes that only for positions `first = n_ctx/2` … `n_ctx-2`, i.e. **`n_ctx - 1 - n_ctx/2` positions per chunk** (l.542, l.619-621).
   Every chunk starts from a fresh context.
4. `main` sets `n_parallel = max(1, n_batch / n_ctx)` and `n_ctx = n_parallel * n_ctx` (l.2035-2038). With the default
   `-b 2048` and `-c 78` that would be 26 parallel sequences in one batch, so `-b 78` is passed to force `n_seq=1`.
   The log confirms: `calculating perplexity over 2 chunks, n_ctx=78, batch_size=78, n_seq=1`.

Consequence: with the prompt as one 78-token context, the file carries **38 positions**
(logits at positions 39..76, predicting tokens 40..77), compressed. 38 < 64. A single chunk yields
≥ 64 saved positions only at `n_ctx ≥ 130`, which needs a prompt of ≥ 130 tokens. **That is a
corpus decision and it was not taken here. The prompt was not padded.** See Open items.

Arithmetic check on the saved file: header `8 ("_logits_") + 4 (n_ctx) + 4 (n_vocab) + 4 (n_chunk) + 156×4 (tokens) = 644`;
`37745892 − 644 = 37745248 = 2 chunks × 38 positions × 248324 (nv) × 2 bytes` ⟹ `nv = 248324`, `n_vocab = 248320`,
38 positions per chunk. The size matches the reading exactly.

## Input construction (recorded, not silent)

`tokens ≥ 2·n_ctx` with `n_ctx = 78` needs ≥ 156 tokens, so the input is the prompt twice:

| candidate | tokens | first 78 == tok(prompt) | verdict |
|---|---|---|---|
| `P P` (plain concatenation) | 155 | **no**: index 77 `13` (".") becomes `11235` because `.` merges with `The` | rejected: chunk 0 would not be the prompt, and 155 < 156 |
| `P " " P` | 156 | yes; the second copy is **not** tok(P) (`" The"` ≠ `"The"`) | rejected |
| **`P "\n" P`** | **157** | **yes**; the last 78 tokens == tok(P); separator = token `198` | **used** |

Input file: [`input-prompt-nl-prompt.txt`](input-prompt-nl-prompt.txt), 891 bytes (445 + 1 + 445), sha256 `14294256f2ef509c0759d1b4042f57fad77ea88a595b0226f7e487de806d1672`.
With `-c 78` the tool reads 2 chunks and the 157th token is dropped:

- **chunk 0 = tokens[0:78] = tok(prompt) exactly. This is the reference.** Saved positions 39..76.
- chunk 1 = tokens[78:156] = `[198]` + tok(prompt)[0:77]. It is a by-product, not the prompt, and must not be compared as if it were.

## Command (identical for runs 1..5)

```bash
timeout 600 ~/src/llama.cpp-d1d3c3396/build/bin/llama-perplexity \
  -m ~/models/Qwen3.5-0.8B-Q4_K_M.gguf -f /tmp/ref3303/cand-nl.txt \
  -ngl 0 -t 8 -c 78 -b 78 --save-all-logits /tmp/ref3303/run$i.kld < /dev/null > /tmp/ref3303/run$i.log 2>&1
```

The full driver output is [`run5-transcript.txt`](run5-transcript.txt) (driver script `/tmp/ref3303/run5.sh` on intel).

## Results, n = 5 (intel, CPU, `-t 8`)

| run | start_utc | rc | load (1/5/15) at start | PPL (final) | per-chunk `[1]`,`[2]` | saved file bytes | sha256 |
|---|---|---|---|---|---|---|---|
| 1 | 16:55:52Z | 0 | 55.57 49.16 46.81 | 42.2235 ± 14.11843 | 42.2513, 42.2235 | 37745892 | `609e048112a1d328cd9cbc766431886cd85adb2f470bab607b9df24ef76d54ef` |
| 2 | 16:55:57Z | 0 | 56.41 49.44 46.91 | 42.2235 ± 14.11843 | (in log) | 37745892 | `609e048112a1d328cd9cbc766431886cd85adb2f470bab607b9df24ef76d54ef` |
| 3 | 16:56:02Z | 0 | 56.21 49.51 46.95 | 42.2235 ± 14.11843 | (in log) | 37745892 | `609e048112a1d328cd9cbc766431886cd85adb2f470bab607b9df24ef76d54ef` |
| 4 | 16:56:07Z | 0 | 57.88 49.97 47.11 | 42.2235 ± 14.11843 | (in log) | 37745892 | `609e048112a1d328cd9cbc766431886cd85adb2f470bab607b9df24ef76d54ef` |
| 5 | 16:56:11Z | 0 | 59.49 50.43 47.28 | 42.2235 ± 14.11843 | (in log) | 37745892 | `609e048112a1d328cd9cbc766431886cd85adb2f470bab607b9df24ef76d54ef` |

`[1]` is the running PPL after chunk 0 alone (38 scored positions: **42.2513**); `[2]` is the running PPL over both chunks.

## Determinism verdict

**DETERMINISTIC at this configuration: 1 distinct sha256 over 5 runs.** The whole saved file
(tokens + compressed log-probs for both chunks) is byte-identical, and so is the PPL, under a
load of 55–61 on the same host. As in [`../../lambda/n5/DETERMINISM.md`](../../lambda/n5/DETERMINISM.md),
the five copies carry no information the one sha256 does not, so none is committed.

The verdict covers **this** configuration: intel, `-t 8`, `-b 78` (`n_seq=1`), this binary and this model. It does not cover other thread counts, other batch
sizes, `n_seq>1` or another host. Those are `[U]` below.

## Logits file location (NOT in git: 37.7 MB > 1 MB)

`intel:/tmp/ref3303/run1.kld` (runs 2..5 are byte-identical copies at `run2.kld`..`run5.kld`),
37745892 bytes, sha256 `609e048112a1d328cd9cbc766431886cd85adb2f470bab607b9df24ef76d54ef`.
**`/tmp` on intel is not durable**, so it can be lost on a reboot. Re-derive it with the command above and check the sha256.

## LEDGER

**No row was added to `evidence/parity/LEDGER.md`.** Its header defines a row as one *cell run*: a
(host, workload, model, quantization, commit, interleaved) tuple *driven across its bands* (PP-9).
A single-lane reference-logit capture has no bands, no subject lane, no interleaving and no
throughput, so it is not a cell and spends no PP-9 key. That file is also outside this ticket's scope.

## Open items

| what | status | command that would measure / decide it |
|---|---|---|
| a reference with ≥ 64 saved positions in one context | `[U]`: blocked on a corpus decision (a prompt of ≥ 130 tokens, or a different logit producer). The 78-token prompt yields 38 | decide the corpus, then `llama-perplexity ... -c <≥130> -b <same> --save-all-logits` over ≥ 2·n_ctx tokens |
| raw (uncompressed) logits rather than uint16 log-softmax | `[U]`: `--save-all-logits` cannot emit them. Whether cosine similarity on dequantized log-probs is an acceptable basis is a design question | a raw-logit producer (e.g. a `llama-eval-callback`-class dump), not built here |
| determinism across `-t` (e.g. 1, 16, 32) | `[U]` | same command with `-t 1` / `-t 16`, n=5 each, compare sha256 |
| determinism at `n_seq>1` (default `-b 2048`) | `[U]` | drop `-b 78`, n=5, compare sha256 with `609e0481…` |
| cross-host agreement (lambda/gx10 CPU) | `[U]` | same command, same model sha256, on another host's `d1d3c3396` CPU build |
| `min_cosine` threshold for Qwen3.5 | `[U]`, fail-closed: B5 (known-bad half) undecided; **not set here by design** | — |
