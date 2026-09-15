# Qwen3.5-0.8B Q4_K_M: llama.cpp per-token vs batched prefill, and apr #3114 against both (PMAT-3303)

**Tree:** worktree `a47b84fe5` (producer with `--per-token`) · llama.cpp `d1d3c3396` · model `Qwen3.5-0.8B-Q4_K_M.gguf`
sha256 `bd258782e35f7f458f8aced1adc053e6e92e89bc735ba3be89d38a06121dc517`
**Host:** intel (`mac-server`, Xeon W-3245, 32 threads, kernel 6.8.0-138), CPU only (`-ngl 0`), `-t 8`,
one job at a time. Load averages sat between 39 and 42 for the whole run (CI fleet host), measured 2026-09-15 19:20–19:21Z.

No threshold is set here. The Qwen3.5 row stays **[U]**, fail-closed (B5 undecided).

## The confound

The #3091 reference came from ONE batched `llama_decode` of all 78 tokens. apr #3114 processes one token at a time.
`apr_raw_logits --per-token` now decodes one token per `llama_decode` call (batch of 1, `logits=1`), in order,
on the same context and memory. It writes the identical APRRAWLG format. See `producer/apr_raw_logits.cpp`.

## Commands (host: intel)

```
bash producer/build.sh                 # rebuild into ~/src/llama.cpp-d1d3c3396/apr-raw-logits (rc=0, no warnings)
bash producer/run_per_token.sh         # full transcript: producer/run_per_token-transcript.txt (script rc=0)
```
Each producer run is `timeout 900 apr_raw_logits -m MODEL -f /tmp/ref3303/prompt.txt -ngl 0 -t 8 -c 78 -b 78 [--per-token] --raw-out F < /dev/null`.
The comparator is `compare_raw_logits.py`, committed on branch PMAT-3091-apr-vs-reference (`af0425fa3`), sha256 `f05cd87a…d4ce`.
The intel copy has the same bytes. It is called as `compare_raw_logits.py <ref.bin> <sub.bin> --json`.

Inputs (sha256, intel):

| artifact | sha256 |
|---|---|
| producer binary (rebuilt) | `5ef2b60ac1761efcd0584cfdcf67363e2c7ebaf1b08a6bcb19cf9d933af403e3` |
| producer binary before the rebuild (kept at `~/parity-ref/qwen35-0.8b-q4km-d1d3c3396-per-token/apr_raw_logits.pre-per-token`) | `f3cca8cc39ac3a6c3ae8b6ab47699a49c77c210b4a203abebe30f5a56671474b` |
| prompt `/tmp/ref3303/prompt.txt` (78 tokens, add_bos=0) | `e6796dc1e57917920a2021896d74a35ae3f63bd12cb8dc1f63f2e2b178ba0eaa` |
| batched reference | `1d5eaf129545aa6087cd6c49792f9774b61caa874fccefdd79e6164afa15bb93` |
| apr #3114 subject run1 (run2 has the same sha) | `f6f792649014dcef6eec1afaf96852a293cc95c6cfceb1c88d62e16ee36c0252` |

The token ids printed by every run are identical to the batched reference's run1 log. The comparator also refuses any pair whose header ids differ, and it accepted all three pairs.

## Determinism and regression (intel, n as stated)

| run | n | sha256 | note |
|---|---|---|---|
| batched (rebuilt binary) | 1 | `1d5eaf12…bb93` | `cmp` to committed reference rc=0: the rewrite did not change batched output |
| per-token | 3 | `2401c11005e08bfac946a1de57af84338cc64a0d8e9a95d0941a6023b81bebc4` ×3 | run1 vs run2 and run1 vs run3 `cmp` rc=0: **byte-identical** |

Outputs: `~/parity-ref/qwen35-0.8b-q4km-d1d3c3396-per-token/` on intel.

## Comparisons (intel, n=1 compare per pair; inputs are deterministic as above)

| | pair (ref → sub) | byte-identical | min cosine (pos) | mean cosine | below 0.98 | argmax mismatches | max abs diff (pos) |
|---|---|---|---|---|---|---|---|
| **A** | per-token llama → batched llama | no (`cmp` rc=1) | 0.998197 (34) | 0.999243 | 0 | **2, 21, 36, 77** | 1.168524 (2) |
| **B** | per-token llama → apr #3114 | no | 0.994599 (4) | 0.998995 | 0 | **21, 28, 73, 77** | 1.434902 (2) |
| **C** | batched llama → apr #3114 | no | 0.995905 (4) | 0.998989 | 0 | **2, 28, 36, 73** | 1.342222 (2) |

TSV and JSON sha256 (intel): A `4ea0bcc5…1102`/`fa65a138…d8f6`, B `fea8ed26…a865`/`bbe9c397…6119`,
C `40c81bd7…1d15`/`3b58b5d7…c4c6`.

Rows at every flip position (columns follow the comparator. "apr" means the sub, which in A is batched llama):

| pos | A: ref gap / sub rank of ref argmax | B: ref gap / apr rank / apr gap | C: ref gap / apr rank / apr gap |
|---|---|---|---|
| 2  | 0.075659 / 3 | 0.075659 / **1** / 0 | 0.432076 / 2 / 0.197060 |
| 21 | 0.036892 / 2 | 0.036892 / 2 / 0.083242 | 0.065741 / 1 / 0 |
| 28 | 0.314670 / 1 (match) | 0.314670 / 2 / 0.216549 | 0.632557 / 2 / 0.216549 |
| 36 | 0.010532 / 2 | 0.010532 / **1** / 0 | 0.013315 / 2 / 0.075449 |
| 73 | 0.245474 / 1 (match) | 0.245474 / 2 / 0.119488 | 0.109070 / 2 / 0.119488 |
| 77 | 0.024683 / 2 | 0.024683 / 2 / 0.093266 | 0.018402 / 1 / 0 |

## Verdict on the four #3091 flips (2, 28, 36, 73)

- **Vanish: 2, 36.** llama.cpp flips against itself at these positions (A). apr matches the per-token reference.
  At 2, the batched reference's 0.43 gap was a batch-path value; per-token llama's own top-2 gap there is 0.076.
- **Persist: 28, 73.** Both llama modes agree on the argmax (A matches). apr holds llama's argmax at rank 2 in
  (B), with apr gaps 0.216549 and 0.119488. The per-token reference's own top-2 gaps are 0.314670 and 0.245474.
  These two are not batch-path artifacts. They remain candidate apr deviations, still [U] on cause.
- **New: 21, 77.** These are llama-internal near-ties (A flips; per-token gaps 0.036892 and 0.024683). apr agrees with
  the batched reference there and not with the per-token one.

llama.cpp's own batched-vs-per-token spread (A: min cosine 0.998197, max |diff| 1.168524, 4 flips) is about the same size as
apr vs either llama mode. On this prompt, the argmax at a near-tie below ~0.08 does not discriminate between
implementations. Positions 28 and 73 are the only flips that survive both reference modes.

## Remaining confounds

- **Threads.** Both llama modes used `-t 8`. apr ran with its default thread pool (632% CPU in `run1.log`). The
  thread count's effect on apr logits was not measured: [U] (`qwen35_raw_logits` under
  `RAYON_NUM_THREADS=8` vs default, n≥2, `cmp`).
- **Mechanism of A.** The kernel path that separates a 78-token ubatch from a 1-token one (recurrent/linear-attention
  step vs chunked, matmul kernel choice) was not traced: [U] (`GGML_SCHED`/op dump per mode).
- **Host load.** The runs were under load averages of about 40. Determinism held (n=3), but load-dependent kernel selection was not varied: [U].
- **One prompt.** All of this is a single 78-token prompt. Vary it before naming a cause for 28 and 73 (Verification Discipline §6): [U].
- **Batched regression n=1** on the rebuilt binary. It matched the n=5 committed reference byte for byte.
