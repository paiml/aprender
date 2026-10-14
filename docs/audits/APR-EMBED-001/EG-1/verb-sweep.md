# EG-1 verb sweep: the 14 verbs on 6 of 6 files (2026-10-09)

**Result: 84 / 84 PASS** on the second run (`eg1-sweep2`), 14 verbs × 6 files. The first run scored 75 / 84. Its
9 failures were 6 wrong `list` expectations and 3 `validate` failures from a real defect (BF16 tensors had no
reader). The defect is fixed in 2a42c3d99a and the two expectations are corrected. Nothing else changed between
the runs.

| Item | Run 1 (`eg1-sweep`) | Run 2 (`eg1-sweep2`) |
|------|--------------------|----------------------|
| apr | release build @ 85ac2d193b (EG-1 slice 5), `apr 0.70.2 (85ac2d193b)` | release build @ 2a42c3d99a (BF16 reader), `apr 0.70.2 (2a42c3d99a)` |
| sha256 | `432fe29ad38db69f400b90cbae7b88df52ca17c495bbf6bc0d180061468998aa` | `50049e707fd8ac5d224e9629b8f5a2187c7d319eca3e8d2f6feaf858c3db3e6d` |
| `cargo test -p apr-cli --lib flow_embedding` | 10 passed, 0 failed | 10 passed, 0 failed |
| Window (UTC) | 2026-10-09 19:58 → 20:34 | 2026-10-09 21:54 → 22:18 |
| Cells | 75 / 84 | 84 / 84 |

**Files:** `hf://unsloth/embeddinggemma-2-GGUF`: BF16, F16, Q8_0, UD-Q4_K_XL, UD-Q5_K_XL, UD-Q6_K_XL. Each file's
sha256 was checked against a SHA256SUMS before the cells started, and each `pull` cell re-checks the pulled bytes.
All six have 413 tensors.

**How a cell is scored.** Every verb runs under a timeout with stdin closed. A cell passes only on exit code 0
**and** a positive check of its output. Neither one alone is enough. `pull`, `list` and `rm` run in a sandboxed
`HOME` and `XDG_CACHE_HOME` per file and fetch the real file from the hub.

| Verb | Check (beyond rc 0) |
|------|---------------------|
| pull | the cached file's sha256 equals SHA256SUMS |
| list | lists the pulled file by its cache path (run 2; see below) |
| rm | no `.gguf` left in the sandbox, and `list` no longer shows that cache path (run 2; see below) |
| tokenize | `encode -p "hello world"`: `roundtrip: true`, no greedy fallback |
| capability | `--json` lists the 8 embedding ops: BidirectionalMask, SlidingWindowAttn, GatedGeluMlp, PerLayerInput, LayerOutputScale, MeanPool, DenseProjection, L2Normalize |
| inspect | architecture `gemma-embedding2`, `attention.causal` false |
| debug | `GGUF gemma-embedding2`, health OK |
| validate | `VALID: N tensors checked, 0 contract violations` |
| lint | lint passed, 0 errors |
| tensors | tensor table naming `token_embd.weight` or `blk.0.attn_q.weight` |
| diff | `Model Diff` against the BF16 file (F16 for the BF16 row) |
| hex | `Format: GGUF v3` |
| tree | root line `<file> (N tensors` |
| flow | `(embedding, gemma-embedding2`, `Pool: mean`, `L2 normalise`, `OUTPUT  embedding [batch, 768]`; no LM head, logits, next_token or decoder-only text; `--json` has `"kind": "embedding"` |

## Run 1: 75 / 84

| Failed cells | Cause | Class |
|--------------|-------|-------|
| `validate` on BF16 (219 violations), Q8_0 (1), UD-Q6_K_XL (1) | `INVALID: 413 tensors, N contract violations`. Each violation is a BF16 tensor (ggml type 30) that aprender-core could not read: "tensor data could not be read … is BF16 (ggml type 30): aprender-core has no dequantizer". Q8_0 and UD-Q6_K_XL keep one BF16 tensor, `per_layer_model_proj.weight`. UD-Q4_K_XL and UD-Q5_K_XL have none and were VALID | **defect**, fixed |
| `list` on all 6 | The check looked for the file by name. An `hf://` single-file pull writes no manifest (GH-352), so `apr list` shows the file under its cache path, `<cache>/pacha/models/<hash16>.gguf`, as an orphan. The file **was** listed, so the expectation was wrong | **check**, corrected |

The `rm` cells passed in run 1, but that pass was vacuous. Their "list no longer shows it" half searched for the
file name, which `list` never printed even before `rm`.

### The fix (2a42c3d99a, Refs #4900)

BF16 is the top 16 bits of an F32, so it widens to F32 **bit-exactly**: `f32::from_bits(u32::from(bits) << 16)`.
`get_tensor_f32` now reads ggml type 30 next to F16 (type 1), through one bounds-checked two-byte widener. Both
whole-model F32 loads that go through it now accept a BF16 tensor: `get_all_tensors_f32`, which is the GH-375
fallback of `apr import` and the path `apr convert` uses. Nothing is approximated. That is the difference from the
#3947 IQ decoders, which stay "inspection only".

New tests in `reader_tests_ggml_sizes.rs`:

- `test_4900_get_tensor_f32_widens_bf16_bit_exactly`: 8 values, including -0.0, `f32::MIN_POSITIVE`, a
  non-trivial mantissa and +∞, are equal `to_bits()`. The fixture first asserts that its low 16 bits are zero, so
  the test cannot pass by rounding.
- `test_4900_f16_keeps_its_own_widening`: the F16 arm is unchanged. 0x3C00 is 1.0 in F16 but 2^-7 in BF16, so
  sharing the arm cannot silently swap the decoders.
- `test_4900_short_bf16_data_is_refused`: a truncated BF16 tensor is an error naming the tensor ("exceeds file
  size"), never a short read.
- `test_4900_whole_model_f32_load_widens_bf16_exactly`: `get_all_tensors_f32` returns the same bits.

`cargo test -p aprender-core --lib -- test_4900 test_3601 test_3656 test_3947` ran on a tree whose `aprender-core`
is that of 2a42c3d99a: 16 passed, 0 failed, 14294 filtered out. The four `test_4900_*` tests are among the 16.
`cargo clippy -p aprender-core --lib -- -D warnings` is clean on the same tree. Under `--profile test`, clippy
reported one finding in the BF16 test (`verbose_bit_mask`, fixed in de03bbb402: the check is now
`trailing_zeros() >= 16`) and 19 in files this branch does not touch.

### The corrected checks

`list` and `rm` now take the cache path from the pull log (`  Path: <path>`). `list` must show that path, and after
`rm` the path must be gone. The `rm` check also fails if no path was read, so it can no longer pass vacuously. The
corrected checks were first replayed on run 1's logs: on all 6 files, `list` showed the pulled path and the
`list` taken after `rm` did not. Only the two predicates changed. The binary-independent parts of the script, the
files and the other 12 checks are the same as in run 1.

## Run 2: 84 / 84

All 84 cells passed, 14 on each of the 6 files. The 84 rows (verb, file, exit code, verdict, check) are in
`verb-sweep-cells.tsv`, next to this file.

- `validate` printed `VALID: 413 tensors checked, 0 contract violations` on all 6 files. That includes BF16, Q8_0
  and UD-Q6_K_XL, the three that failed in run 1, so every BF16 tensor in them was read.
- `list` showed the pulled cache path on all 6 files. After `rm`, that path was gone on all 6. The `rm` check now
  needs a path read from the pull log, so these passes are not vacuous.
- The other 11 verbs passed on all 6 files, as they did in run 1.

## Findings (not fixed here)

1. **`apr list` after an `hf://` single-file pull shows a hash name.** The pull writes no manifest, so the model
   is listed as an orphan `<hash16>.gguf` under its cache path, never by its repo or file name. This affects every
   architecture, not only this one. It needs its own ticket.
2. **`apr pull` suggests `apr run` / `apr serve` for an embedding model.** Its closing hint is the same for every
   kind. For `ModelKind::Embedding` the useful next command is `apr embed`.
3. **`apr inspect` prints a bool array as `[len=0]`.** `gemma-embedding2.attention.sliding_window_pattern` is a
   24-entry bool array in all 6 files (5 local, then 1 global, repeated 4 times; read straight from the GGUF header
   with a one-off reader). The metadata table shows `[len=0]`, so the per-layer attention type that I-9 reads from
   the file cannot be seen with `inspect`. This is a display defect only. It does not change what the file holds.
