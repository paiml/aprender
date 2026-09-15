# Qwen3.5-0.8B Q4_K_M: apr with llama.cpp's CPU matmul arithmetic emulated (PMAT-3091)

**Question.** Is most of the apr-vs-llama.cpp gap caused by the REFERENCE quantizing the activation before each quantized dot product (`vec_dot_type`), rather than by an apr defect? The test: make apr compute the way ggml does, then check whether apr converges to llama.
**No threshold is set anywhere. The Qwen3.5 parity row stays [U], fail-closed.**

**Result in brief:**
- **Mechanism: confirmed at the kernel.** With the switch ON, apr's layer-0 `linear_attn_out-0` against llama falls from rel L2 0.034–0.042 to **≤ 0.000002** at all 6 measured positions. Every observed point through `l_out-2` sits at ≤ 0.000002 at p4 pos 0–1 and orig pos 4.
  At layer 0 each apr kernel's departure from float64 `dequant(W)·x` now equals llama's to 6 digits, for Q4_K `z`, Q5_K `ssm_out` and the Q4_K+Q6_K FFN (§6).
- **"Most of the gap": not supported.** ON removes this fraction of the Frobenius logits gap:

  | prompt | gap removed |
  |---|---|
  | orig | **15.7%** |
  | p1 | **0.8%** |
  | p2 | **4.3%** |
  | p3 | **10.2%** |
  | p4 | **32.3%** |

  Most of the gap remains on every prompt.
- **Residual.** It enters at the first full-attention layer (3), in the mixer, while that layer's input still matches to 1e-6. At p4 pos 0 it is reproduced by rounding apr's attention output to f16: llama's KV cache is `K (f16), V (f16)` with Flash Attention enabled, and 2044 of 2048 elements become bit-equal.
  The largest remaining step at p4 pos 1 is `l_out-21` (layer 21 gated DeltaNet FFN: Q4_K gate/up, Q6_K down), +0.012802.

## Tree

| side | tree |
|---|---|
| apr code | worktree `obs-3091`, branch `PMAT-3091-layer-observer` @ **`2ae1971606724c1988722d6cc0146b02dff0e352`** (`ggml_vecdot_emul.rs` sha256 `79deb812a650c277583cc990092cc2c68d2996d6e48eecdde024ce5f0b96392b`), on the observer `0543266ee`, stacked on #3114 **`31448f6c3`**. It is a local measurement branch and is never PR'd. The binary was built with only the uncommitted example `emulation/qwen35_layer_obs.rs` (sha256 `f1e274c6…1dc6b0cb`) added. `git status` shows no other change (`run_emulation.transcript` header). Release binary `qwen35_layer_obs` sha256 `e56bacdb0ba9fc9a5c32213671dccf130a64fb3bf2fae6f45b68bb6577c167e3` |
| reference | llama.cpp **`d1d3c3396`**, intel `~/src/llama.cpp-d1d3c3396`: `build/bin/libggml-cpu.so.0.24.0` `6d8708c4…08c13`, `libggml-base.so.0.24.0` `bddf9ff4…41463`, producer and dumps as in `LAYERWISE.md` §5 (not rerun) |
| model | `Qwen3.5-0.8B-Q4_K_M.gguf` sha256 `bd258782e35f7f458f8aced1adc053e6e92e89bc735ba3be89d38a06121dc517` |
| llama logits refs (per-token) | orig `2401c110…c4bc`, p1 `67dcc9e7…f10a`, p2 `a0f87b14…b760`, p3 `df47beff…6ea5`, p4 `0eca077c…6e1e`, copied intel → lambda `/tmp/emul3091/ref/`, and the hashes were re-read after the copy (`compare_emulation.transcript`) |

## Hosts and load

- **intel** (`mac-server`, Xeon W-3245, AVX-512, glibc 2.35), in `~/parity-ref/emulation-3091/` and `/tmp/emul3091/` only:
  - fixture harness at 21:05:57Z, load 36.66/38.98/53.16; rerun at 21:06:33Z, load 30.28/37.01/51.90 (`build_and_run.transcript`);
  - one verbose producer probe (`-v`, 1-token prompt, `-t 8`) at 21:01:33Z, load 26.59/40.03/58.34.
- **lambda** (48 threads), isolated `CARGO_TARGET_DIR=/mnt/nvme-raid0/targets/obs-3091`:
  - build and tests at load 1.26;
  - apr runs 21:17:35Z–21:22:45Z, load 1.12 → 40.77;
  - comparators 21:24:29Z–21:26:54Z, load 58.96 → 38.65.

  Every result is a deterministic function of hashed inputs. The two ON repeats were `cmp` rc 0.

## 1. What llama.cpp d1d3c3396 actually computes on intel (read from source, then proved by a run log)

| weight (count) | CPU `vec_dot_type` (`ggml-cpu.c`) | path taken on intel (AVX2/AVX-512, `GGML_NATIVE=ON`, `REPACK=ON`, `LLAMAFILE=ON`) |
|---|---|---|
| Q4_K (98 tensors: attn_gate, attn_q/k/output, 2 attn_v, ffn_gate/up, 12 ffn_down) | Q8_K (`:314`) | **REPACK** `q4_K_8x8` (`repack.cpp:4605-4609`, needs `avx2` and `ne[1] % 8 == 0`): `ggml_quantize_mat_q8_K_4x8` + `gemv_q4_K_8x8_q8_K`. The probe log names all 98 tensors `repack tensor … with q4_K_8x8` and shows `CPU_REPACK model buffer size = 160.88 MiB` (`llama_repack_probe.excerpt.log`) |
| Q5_K (18 attn_qkv, 18 ssm_out) | Q8_K (`:324`) | `ggml_compute_forward_mul_mat` (`:1323-1352`): `from_float` = `quantize_row_q8_K`, which on x86 is `quantize_row_q8_K_ref` (`arch/x86/quants.c:505`). Then `ggml_vec_dot_q5_K_q8_K` AVX2 (`arch/x86/quants.c:2216`). Q5_K has no x86 repack (`repack.cpp:4649`: NEON only) |
| Q6_K (12 ffn_down, 4 attn_v, token_embd as lm_head) | Q8_K (`:330`) | same, `ggml_vec_dot_q6_K_q8_K` AVX2 (`:2426`). No x86 repack (`repack.cpp:4660`). `token_embd` is also refused by `CPU_REPACK` (probe log) |
| Q8_0 (18 ssm_alpha, 18 ssm_beta) | Q8_0 (`:275`) | `llamafile_sgemm` returns false for `n < 2` (`sgemm.cpp:3819`, per-token n = 1). The Q8_0 x86 repack is NEON-only (`repack.cpp:4704`). So `quantize_row_q8_0` AVX2 (`arch/x86/quants.c:302`) + `ggml_vec_dot_q8_0_q8_0` AVX2 (`:1308`) |
| F32 (norms, ssm_a, conv1d, dt_bias) | F32 | not quantized; unchanged by the switch |

- IQ panels (`iqp.cpp`) need `src1->ne[1] >= GGML_IQP_MIN_BATCH` (8), so they are not taken at n = 1.
- Reference sources: `quantize_row_q8_K_ref` is `ggml-quants.c:2768`. The generic dots are `ggml-cpu/quants.c:696` (Q4_K), `:771` (Q5_K), `:851` (Q6_K) and `:451` (Q8_0).

**SIMD vs generic.** llama runs the AVX2 dots and the AVX2 repack path. Their integer sums are those of the generic code, but their float accumulation order differs.
- ggml has no bit-exact test for this. `tests/test-quantize-fns.cpp` checks tolerances only: `MAX_DOT_PRODUCT_ERROR = 0.02`, `MAX_QUANTIZATION_REFERENCE_ERROR = 0.0001`.
- Measured with ggml's own objects on the 10 fixtures (`fixtures.tsv`):

  | dot | SIMD == generic bit-exact | max rel diff |
  |---|---|---|
  | Q4_K | 3/10 | 1.8e-5 |
  | Q5_K | 2/10 | 2.2e-7 |
  | Q6_K | 3/10 | 1.3e-6 |
  | Q8_0 | 5/10 | 4.2e-7 |

- The repack activation quantizer (`ggml_quantize_mat_q8_K_4x8`, AVX2) produced the same bytes as `quantize_row_q8_K_ref` on 9 of 10 fixtures. On case 8 (a ±|max| tie) it produced exact negations of d and qs, so the values are equal on 10/10 (`repack_value_equal_ref`).
- `gemv_q4_K_8x8` itself was not ported. Its integer terms are those of `vec_dot_q4_K_q8_K` (source); its float order is not.

## 2. The port and its bit-exact proof

`crates/aprender-serve/src/quantize/ggml_vecdot_emul.rs` ports:
- `quantize_row_q8_K_ref`, including `nearest_int`'s float-bits trick;
- the x86 AVX2 `quantize_row_q8_0` (round half to even, `id = 127/max`, `_cvtss_sh` fp16), because it differs from `_ref` on exact ties (case 7: `q80_ref_eq_cpu = 0`);
- `ggml_vec_dot_{q4_K,q5_K,q6_K}_q8_K_generic` and `ggml_vec_dot_q8_0_q8_0_generic`.

It uses no `unwrap()`.

**FMA.** The shipped generic dots were built with `-march=native`, and gcc contracted exactly these statements (objdump in `build_and_run.transcript`: 2/2/1/1 FMA instructions):
- `sums[l] += d*aux32[l]` → `vfmadd231ps`
- `sumf -= dmin*sumi` → `vfnmadd231ss`
- Q8_0 `sumf += sumi*(dx*dy)` → `vfmadd231ss`

The port uses `mul_add` at exactly those sites.

**Fixtures** are generated by calling ggml's C directly. `emulation/ggml_emul_fixtures.c` (sha256 `80518776…77b29aaf`) is linked against intel's `libggml-cpu.so`/`libggml-base.so` by `emulation/build_and_run.sh`. There are 10 cases:

| case | n | input |
|---|---|---|
| 0 | 256 | uniform ±1 |
| 1 | 1024 | uniform ±3 |
| 2 | 2048 | uniform ±0.05 |
| 3 | 3584 | uniform ±1, 14 blocks |
| 4 | 512 | **all zero** |
| 5 | 256 | **extreme magnitude** ±1e30 |
| 6 | 768 | one zero, one ×1e30, one ×1e-30 block |
| 7 | 256 | exact .5 ties |
| 8 | 256 | a ±\|max\| tie |
| 9 | 1024 | outliers |

- Every n is a multiple of 256.
- Inputs and weight blocks come from splitmix64. The weights are random bytes with sane fp16 scales; Q8_0 bytes never hold −128.
- Outputs: `fixtures.tsv` `d10f6ca1…f0479e78`, `fixtures.rs.txt` `54d88cd6…02bb73ff` (pasted verbatim into the test), `fixtures.bin` `c36cea69…351345ec` (every input row and block).

Unit tests (`cargo test -p aprender-serve --lib ggml_vecdot_emul`, 7 tests):
- `fixture_generator_reproduces_the_c_harness_bytes`: the input and weight FNV hashes equal the C harness's.
- `quantize_row_q8_k_is_bit_exact_vs_ggml_c`: Q8_K block bytes equal on 10/10.
- `vec_dot_k_quants_are_bit_exact_vs_ggml_c_generic`: Q4_K, Q5_K and Q6_K `f32` bits equal on 10/10, including NaN-free extreme and zero rows.
- `q8_0_quantize_and_dot_are_bit_exact_vs_ggml_c`: block bytes and dot bits equal on 10/10. That includes cases 5/6, where ggml's dot is NaN `0xffc00000` because the fp16 scale overflows.
- Row-vs-matvec consistency, bad-shape refusal, and switch parsing.
- **TDD:** RED first. The stubs compiled, and 5 of 7 failed, all except the generator and bad-shape tests (`unit_red.log`). GREEN: 7/7 (`unit_green.log`).

## 3. The switch

- `APR_EMULATE_GGML_VECDOT` is read once per process through a `OnceLock`:
  - unset, empty or `0` = off;
  - `1` or `all` = Q4_K, Q5_K, Q6_K and Q8_0;
  - otherwise a comma list of those names. Any other value is an error, never ignored.
- In the Qwen3.5 forward, all 16 `fused_matmul_into` call sites now call `Qwen35Model::qmatmul`. With the switch off, `qmatmul` is `fused_matmul_into`. With it on, a selected qtype runs `emulated_matvec`: quantize x once to `vec_dot_type`, then one ggml dot per row.
- Per-qtype counters are printed by the harness at exit.
- `cargo test -p aprender-serve --lib qwen35`: 22 passed, the same 22 as before.

## 4. Invariance proof: switch OFF (before any comparison)

`run_emulation.transcript`:

| prompt | switch-OFF noop logits sha256 | subject (recorded before) | equal |
|---|---|---|---|
| orig (78) | `f6f792649014dcef6eec1afaf96852a293cc95c6cfceb1c88d62e16ee36c0252` | `apr-intel-run1.bin` f6f79264…0252 | **true** |
| p4 (82) | `92f1b54d504a53f758a56911c4779ce1900207bd89c78d58597e558ebc6d3ee9` | `p4-apr.bin` 92f1b54d…3ee9 | **true** |
| p1 (83) | `b9815d9b…b7f8` | `VARIATION.md` p1 apr | true |
| p2 (117) | `d5ffa2b7…a7fa` | `VARIATION.md` p2 apr | true |
| p3 (108) | `b2a984d0…3e91` | `VARIATION.md` p3 apr | true |

- The OFF observer dumps' logits `cmp` equal the OFF noop logits (rc 0).
- `diff -rq` of the OFF dumps against the `0543266ee` dumps (`/tmp/obs3091/apr/obs-p{4,0}`) reports only `Only in …: manifest.sha256.tsv`, a file the old run added afterwards. No `.f32` or `manifest.tsv` differs.
- `compare_layerwise.py` over the OFF dumps reproduces the committed `layerwise/sublayer_p4.tsv` and `sublayer_orig.tsv` **byte for byte** (`cmp` rc 0).
- Every OFF counter is `[0, 0, 0, 0]`.

## 5. Mechanism-engaged proof: switch ON

- **Counters at exit**, `emulated_matmuls[Q4_K,Q5_K,Q6_K,Q8_0]`. For orig (78 positions) they are `[7644, 2808, 1326, 2808]`:
  - Q4_K: 78 × 98;
  - Q5_K: 78 × 36;
  - Q6_K: 78 × 17 (12 ffn_down, 4 attn_v, lm_head);
  - Q8_0: 78 × 36.

  That is exactly the GGUF's tensor count per qtype.
  - p4 `[8036, 2952, 1394, 2952]` = 82 × the same.
  - Each `only-<qtype>` run counts only its own qtype.
- **Outputs change:**
  - ON logits differ from OFF on all 5 prompts (`cmp` rc 1);
  - `linear_attn_out-0` and `final_output-0` ON differ from OFF at p4 pos 0–3 (`cmp` rc 1);
  - the ON repeats for orig/p4 are byte-identical (`485341ae…9a53`, `6ccada2b…1f54`).
- **Kernel level (§6, `cmp/kernel_isolation_on.tsv`):** at layer 0, where both engines' inputs are equal, apr's departure from float64 `dequant(W)·x` equals llama's to 6 digits at all 6 positions. Example, p4 pos 1: `linear_attn_out-0` apr 0.041157 = llama 0.041157, `z-0` 0.005277 = 0.005277, `ffn_out-0` 0.018996 = 0.018996. OFF, apr was at 0.000001 on `ssm_out`.

## Commands

```bash
# intel
scp ggml_emul_fixtures.c build_and_run.sh intel:parity-ref/emulation-3091/
bash ~/parity-ref/emulation-3091/build_and_run.sh      # rc 0 -> fixtures.{tsv,rs.txt,bin}, FMA objdump
timeout 300 apr_raw_logits -m MODEL -f tiny.txt -ngl 0 -t 8 -c 8 -b 8 --per-token -v --raw-out /tmp/emul3091/tiny.bin < /dev/null   # probe, rc 0
# lambda (cargo is a zsh function: called by absolute path)
CARGO_TARGET_DIR=/mnt/nvme-raid0/targets/obs-3091 CARGO_BUILD_JOBS=8 nice -n 10 cargo test -p aprender-serve --lib ggml_vecdot_emul   # RED 5 failed, then GREEN 7 passed
CARGO_TARGET_DIR=... cargo test -p aprender-serve --lib qwen35                                                                     # 22 passed
CARGO_TARGET_DIR=... cargo build --release -p aprender-serve --example qwen35_layer_obs                                           # rc 0
bash emulation/run_emulation.sh     > run_emulation.transcript       # OFF invariance gate (exit 3 if orig/p4 differ), ON, dumps, per-qtype, repeats
bash emulation/compare_emulation.sh > compare_emulation.transcript   # compare_raw_logits.py, logits_gap.py, compare_layerwise.py, layer_steps.py, kernel_isolation.py (24 layers)
python3 emulation/tables.py /tmp/emul3091/cmp > cmp/tables.md
python3 emulation/kv_f16_check.py /tmp/obs3091/llama/sub-p4 runs/on-dump-p4 runs/off-dump-p4 0,1,2,3 3,7,11,15,19,23 > cmp/kv_f16_check.tsv
# gate (on 2ae197160): cargo test --lib qwen35 && cargo test --lib ggml_vecdot_emul && cargo fmt --all -- --check   -> rc 0
```

Script sha256:

| script | sha256 |
|---|---|
| `run_emulation.sh` | `8d3b4698…e79f63fa` |
| `compare_emulation.sh` | `adebf9e3…a085d4fd` |
| `logits_gap.py` | `ea30daa6…eee7f42e` |
| `tables.py` | `ffefe534…26e44fd4` |
| `kv_f16_check.py` | `64919ed3…b74f6eb5` |
| `build_and_run.sh` | `10760e4a…61f0dbeb` |

The committed comparators are unchanged: `compare_raw_logits.py` `f05cd87a…d4ce`, plus `layerwise/compare_layerwise.py`, `layer_steps.py` and `kernel_isolation.py` as in `LAYERWISE.md`.
- Every run's logits sha256 is in `runs/logits.sha256.txt` (36 files).
- The ON/OFF dump manifests with per-file sha256 are `runs/{on,off}-dump-{p4,orig}.manifest.sha256.tsv`.

## 6. Measurements, ON vs OFF

All tables below are generated by `tables.py` (`cmp/tables.md`) from `cmp/`. "rel L2" = ‖apr − llama‖ / ‖llama‖.


### 6a. Logits against llama per-token (every variant; `only-X` = the switch emulates qtype X alone)

| prompt | variant | min cos (pos) | mean cos | median cos | n < 0.98 | argmax mismatches | max abs diff |
|---|---|---|---|---|---|---|---|
| orig | off | 0.994599 (4) | 0.998995 | 0.999282 | 0 | 4 | 1.434902 |
| orig | on | 0.998157 (46) | 0.999266 | 0.999319 | 0 | 3 | 1.129599 |
| orig | only-Q4_K | 0.994714 (4) | 0.999027 | 0.999240 | 0 | 5 | 1.567030 |
| orig | only-Q5_K | 0.998137 (43) | 0.999228 | 0.999304 | 0 | 4 | 1.197620 |
| orig | only-Q6_K | 0.993866 (4) | 0.998863 | 0.999109 | 0 | 6 | 1.398344 |
| orig | only-Q8_0 | 0.993910 (4) | 0.998990 | 0.999234 | 0 | 6 | 1.686095 |
| p1 | off | 0.997448 (0) | 0.999509 | 0.999580 | 0 | 4 | 1.452764 |
| p1 | on | 0.998868 (0) | 0.999528 | 0.999565 | 0 | 8 | 0.752477 |
| p1 | only-Q4_K | 0.997532 (0) | 0.999516 | 0.999567 | 0 | 2 | 1.362767 |
| p1 | only-Q5_K | 0.997644 (0) | 0.999550 | 0.999606 | 0 | 5 | 1.246348 |
| p1 | only-Q6_K | 0.998839 (3) | 0.999491 | 0.999522 | 0 | 6 | 0.838742 |
| p1 | only-Q8_0 | 0.997648 (0) | 0.999523 | 0.999589 | 0 | 6 | 1.186209 |
| p2 | off | 0.995370 (44) | 0.998371 | 0.998489 | 0 | 2 | 1.923379 |
| p2 | on | 0.993047 (64) | 0.998519 | 0.998728 | 0 | 2 | 2.209243 |
| p2 | only-Q4_K | 0.995590 (0) | 0.998374 | 0.998498 | 0 | 2 | 1.732664 |
| p2 | only-Q5_K | 0.995430 (101) | 0.998519 | 0.998662 | 0 | 1 | 1.678028 |
| p2 | only-Q6_K | 0.994483 (44) | 0.998241 | 0.998446 | 0 | 4 | 1.694886 |
| p2 | only-Q8_0 | 0.995935 (0) | 0.998393 | 0.998574 | 0 | 4 | 1.839685 |
| p3 | off | 0.995687 (0) | 0.999133 | 0.999353 | 0 | 5 | 1.376676 |
| p3 | on | 0.997929 (76) | 0.999302 | 0.999432 | 0 | 5 | 0.810490 |
| p3 | only-Q4_K | 0.995202 (0) | 0.999149 | 0.999360 | 0 | 5 | 1.376700 |
| p3 | only-Q5_K | 0.995787 (0) | 0.999239 | 0.999406 | 0 | 6 | 1.290376 |
| p3 | only-Q6_K | 0.996394 (84) | 0.999114 | 0.999332 | 0 | 4 | 1.133504 |
| p3 | only-Q8_0 | 0.995721 (0) | 0.999181 | 0.999352 | 0 | 8 | 1.389621 |
| p4 | off | 0.961309 (1) | 0.994771 | 0.998415 | 4 | 6 | 4.114042 |
| p4 | on | 0.984332 (74) | 0.997562 | 0.998841 | 0 | 3 | 2.094245 |
| p4 | only-Q4_K | 0.962798 (1) | 0.994403 | 0.998479 | 5 | 7 | 4.179953 |
| p4 | only-Q5_K | 0.981708 (1) | 0.997245 | 0.998814 | 0 | 4 | 2.708869 |
| p4 | only-Q6_K | 0.965610 (1) | 0.994567 | 0.998435 | 5 | 5 | 3.715054 |
| p4 | only-Q8_0 | 0.963463 (1) | 0.995244 | 0.998427 | 3 | 4 | 4.415126 |

### 6b. Fraction of the logits rel-L2 gap removed (per-pos rel L2 at p4 pos 0-3 and orig 4/28 in the last column)

| prompt | variant | Frobenius rel L2 | mean per-pos rel L2 | median per-pos rel L2 | gap removed (Frob) | gap removed (mean) | per-pos rel L2 |
|---|---|---|---|---|---|---|---|
| orig | off | 0.044677 | 0.044208 | 0.038956 | 0.000000 | 0.000000 | 4:0.110592,28:0.053922 |
| orig | on | 0.037667 | 0.038570 | 0.038220 | 0.156905 | 0.127530 | 4:0.034940,28:0.039182 |
| orig | only-Q4_K | 0.044002 | 0.043575 | 0.039162 | 0.015104 | 0.014307 | 4:0.109141,28:0.049522 |
| orig | only-Q5_K | 0.039541 | 0.039885 | 0.037907 | 0.114956 | 0.097788 | 4:0.051791,28:0.038735 |
| orig | only-Q6_K | 0.047046 | 0.046958 | 0.043582 | -0.053034 | -0.062208 | 4:0.115912,28:0.040155 |
| orig | only-Q8_0 | 0.044773 | 0.044168 | 0.039884 | -0.002138 | 0.000888 | 4:0.117700,28:0.051956 |
| p1 | off | 0.031619 | 0.032024 | 0.030915 | 0.000000 | 0.000000 |  |
| p1 | on | 0.031376 | 0.031908 | 0.030769 | 0.007667 | 0.003636 |  |
| p1 | only-Q4_K | 0.031503 | 0.031847 | 0.031170 | 0.003672 | 0.005532 |  |
| p1 | only-Q5_K | 0.030536 | 0.030865 | 0.030002 | 0.034257 | 0.036203 |  |
| p1 | only-Q6_K | 0.032629 | 0.033019 | 0.031514 | -0.031959 | -0.031047 |  |
| p1 | only-Q8_0 | 0.031037 | 0.031441 | 0.030820 | 0.018405 | 0.018233 |  |
| p2 | off | 0.056340 | 0.056694 | 0.055777 | 0.000000 | 0.000000 |  |
| p2 | on | 0.053926 | 0.054011 | 0.051421 | 0.042838 | 0.047322 |  |
| p2 | only-Q4_K | 0.056158 | 0.056669 | 0.055679 | 0.003214 | 0.000432 |  |
| p2 | only-Q5_K | 0.053682 | 0.054190 | 0.052874 | 0.047173 | 0.044172 |  |
| p2 | only-Q6_K | 0.058082 | 0.058912 | 0.056718 | -0.030930 | -0.039123 |  |
| p2 | only-Q8_0 | 0.056046 | 0.056408 | 0.054889 | 0.005216 | 0.005043 |  |
| p3 | off | 0.040229 | 0.041500 | 0.037755 | 0.000000 | 0.000000 |  |
| p3 | on | 0.036117 | 0.038183 | 0.035224 | 0.102196 | 0.079944 |  |
| p3 | only-Q4_K | 0.039936 | 0.041255 | 0.037513 | 0.007284 | 0.005925 |  |
| p3 | only-Q5_K | 0.037573 | 0.039211 | 0.036686 | 0.066011 | 0.055163 |  |
| p3 | only-Q6_K | 0.040582 | 0.042333 | 0.037876 | -0.008775 | -0.020061 |  |
| p3 | only-Q8_0 | 0.038813 | 0.040608 | 0.038213 | 0.035183 | 0.021515 |  |
| p4 | off | 0.092010 | 0.087886 | 0.057325 | 0.000000 | 0.000000 | 0:0.097638,1:0.278233,2:0.266478,3:0.214359 |
| p4 | on | 0.062323 | 0.063440 | 0.051131 | 0.322653 | 0.278165 | 0:0.063340,1:0.135822,2:0.080967,3:0.089452 |
| p4 | only-Q4_K | 0.095411 | 0.091085 | 0.056606 | -0.036963 | -0.036396 | 0:0.095852,1:0.271728,2:0.286035,3:0.224230 |
| p4 | only-Q5_K | 0.066919 | 0.066504 | 0.051743 | 0.272693 | 0.243292 | 0:0.100068,1:0.190801,2:0.172990,3:0.128448 |
| p4 | only-Q6_K | 0.093754 | 0.089490 | 0.057238 | -0.018953 | -0.018247 | 0:0.065101,1:0.264946,2:0.250511,3:0.282877 |
| p4 | only-Q8_0 | 0.087600 | 0.084928 | 0.059150 | 0.047925 | 0.033657 | 0:0.095340,1:0.269571,2:0.244933,3:0.213508 |

### 6c. Layer 0 against llama (rel L2)

| label pos | final_output-0 OFF | final_output-0 ON | linear_attn_out-0 OFF | linear_attn_out-0 ON | attn_residual-0 OFF | attn_residual-0 ON | l_out-0 OFF | l_out-0 ON |
|---|---|---|---|---|---|---|---|---|
| p4 0 | 0.003361 | 0.000000 | 0.038959 | 0.000001 | 0.037412 | 0.000001 | 0.036982 | 0.000001 |
| p4 1 | 0.003381 | 0.000000 | 0.041934 | 0.000002 | 0.039963 | 0.000002 | 0.053324 | 0.000001 |
| p4 2 | 0.003577 | 0.000000 | 0.042371 | 0.000001 | 0.042473 | 0.000001 | 0.046291 | 0.000001 |
| p4 3 | 0.002697 | 0.000000 | 0.036260 | 0.000001 | 0.036257 | 0.000001 | 0.044281 | 0.000235 |
| orig 4 | 0.002259 | 0.000000 | 0.034282 | 0.000000 | 0.033858 | 0.000000 | 0.037471 | 0.000000 |
| orig 28 | 0.004327 | 0.000000 | 0.039255 | 0.000001 | 0.039555 | 0.000001 | 0.048359 | 0.000001 |

### 6d. Full residual-stream curve at p4 pos 1, ON vs OFF

| point (p4 pos 1) | rel L2 OFF | rel L2 ON | cos OFF | cos ON |
|---|---|---|---|---|
| model.input_embed | 0.000000 | 0.000000 | 1.000000 | 1.000000 |
| attn_residual-0 | 0.039963 | 0.000002 | 0.999204 | 1.000000 |
| l_out-0 | 0.053324 | 0.000001 | 0.998590 | 1.000000 |
| attn_residual-1 | 0.053683 | 0.000001 | 0.998566 | 1.000000 |
| l_out-1 | 0.064590 | 0.000001 | 0.997977 | 1.000000 |
| attn_residual-2 | 0.062996 | 0.000001 | 0.998023 | 1.000000 |
| l_out-2 | 0.066006 | 0.000001 | 0.997825 | 1.000000 |
| attn_residual-3 | 0.071860 | 0.002536 | 0.997422 | 0.999997 |
| l_out-3 | 0.080963 | 0.010622 | 0.996731 | 0.999945 |
| attn_residual-4 | 0.082368 | 0.012662 | 0.996613 | 0.999920 |
| l_out-4 | 0.083895 | 0.018335 | 0.996478 | 0.999833 |
| attn_residual-5 | 0.084387 | 0.022775 | 0.996434 | 0.999741 |
| l_out-5 | 0.093088 | 0.026534 | 0.995661 | 0.999648 |
| attn_residual-6 | 0.091241 | 0.027400 | 0.995829 | 0.999625 |
| l_out-6 | 0.093633 | 0.029526 | 0.995659 | 0.999564 |
| attn_residual-7 | 0.103630 | 0.034916 | 0.994632 | 0.999391 |
| l_out-7 | 0.112255 | 0.036633 | 0.993705 | 0.999330 |
| attn_residual-8 | 0.111697 | 0.037680 | 0.993784 | 0.999291 |
| l_out-8 | 0.120305 | 0.042099 | 0.992741 | 0.999115 |
| attn_residual-9 | 0.122609 | 0.043904 | 0.992457 | 0.999038 |
| l_out-9 | 0.136550 | 0.045557 | 0.990639 | 0.998962 |
| attn_residual-10 | 0.133372 | 0.043796 | 0.991092 | 0.999041 |
| l_out-10 | 0.139653 | 0.045411 | 0.990222 | 0.998974 |
| attn_residual-11 | 0.142580 | 0.048231 | 0.989794 | 0.998842 |
| l_out-11 | 0.157639 | 0.051318 | 0.987689 | 0.998689 |
| attn_residual-12 | 0.156778 | 0.052131 | 0.987840 | 0.998647 |
| l_out-12 | 0.177771 | 0.053885 | 0.984729 | 0.998550 |
| attn_residual-13 | 0.177192 | 0.054038 | 0.984775 | 0.998543 |
| l_out-13 | 0.187669 | 0.058313 | 0.983093 | 0.998303 |
| attn_residual-14 | 0.185727 | 0.059172 | 0.983354 | 0.998254 |
| l_out-14 | 0.180173 | 0.060812 | 0.984252 | 0.998161 |
| attn_residual-15 | 0.182158 | 0.063824 | 0.983820 | 0.997977 |
| l_out-15 | 0.174635 | 0.060845 | 0.985146 | 0.998158 |
| attn_residual-16 | 0.157479 | 0.059396 | 0.987684 | 0.998235 |
| l_out-16 | 0.176957 | 0.068734 | 0.984385 | 0.997643 |
| attn_residual-17 | 0.172843 | 0.069882 | 0.985101 | 0.997563 |
| l_out-17 | 0.189268 | 0.077512 | 0.982032 | 0.996993 |
| attn_residual-18 | 0.191530 | 0.083136 | 0.981559 | 0.996554 |
| l_out-18 | 0.203555 | 0.089004 | 0.979126 | 0.996033 |
| attn_residual-19 | 0.211931 | 0.095469 | 0.977403 | 0.995434 |
| l_out-19 | 0.230613 | 0.103971 | 0.973098 | 0.994581 |
| attn_residual-20 | 0.248115 | 0.112716 | 0.969102 | 0.993652 |
| l_out-20 | 0.262045 | 0.123665 | 0.965589 | 0.992345 |
| attn_residual-21 | 0.284240 | 0.128373 | 0.959212 | 0.991778 |
| l_out-21 | 0.314435 | 0.141175 | 0.950480 | 0.990048 |
| attn_residual-22 | 0.309310 | 0.139836 | 0.952225 | 0.990282 |
| l_out-22 | 0.299569 | 0.138729 | 0.954867 | 0.990392 |
| attn_residual-23 | 0.319942 | 0.149764 | 0.948530 | 0.988953 |
| l_out-23 | 0.332800 | 0.151325 | 0.944144 | 0.988811 |
| result_norm | 0.331704 | 0.150906 | 0.944845 | 0.988633 |
| result_output | 0.278233 | 0.135822 | 0.961309 | 0.990794 |

### 6e. Kernel isolation: each engine's output vs float64 `dequant(W)` applied to its OWN input, all 24 layers x 6 positions

| mode | output | qtypes | engine | rows | rel L2 vs float64 dequant min | max |
|---|---|---|---|---|---|---|
| off | attn_output | Q4_K | apr | 36 | 0.015207 | 0.061895 |
| off | attn_output | Q4_K | llama | 36 | 0.015518 | 0.062566 |
| off | ffn_out | Q4_K,Q4_K,Q4_K | apr | 72 | 0.016175 | 0.032559 |
| off | ffn_out | Q4_K,Q4_K,Q4_K | llama | 72 | 0.016414 | 0.031686 |
| off | ffn_out | Q4_K,Q4_K,Q6_K | apr | 72 | 0.000983 | 0.027979 |
| off | ffn_out | Q4_K,Q4_K,Q6_K | llama | 72 | 0.009533 | 0.032245 |
| off | linear_attn_out | Q5_K | apr | 108 | 0.000000 | 0.000001 |
| off | linear_attn_out | Q5_K | llama | 108 | 0.016144 | 0.063286 |
| off | z | Q4_K | apr | 108 | 0.003710 | 0.014706 |
| off | z | Q4_K | llama | 108 | 0.003750 | 0.013164 |
| on | attn_output | Q4_K | apr | 36 | 0.015837 | 0.060762 |
| on | attn_output | Q4_K | llama | 36 | 0.015518 | 0.062566 |
| on | ffn_out | Q4_K,Q4_K,Q4_K | apr | 72 | 0.016837 | 0.032023 |
| on | ffn_out | Q4_K,Q4_K,Q4_K | llama | 72 | 0.016414 | 0.031686 |
| on | ffn_out | Q4_K,Q4_K,Q6_K | apr | 72 | 0.010410 | 0.030860 |
| on | ffn_out | Q4_K,Q4_K,Q6_K | llama | 72 | 0.009533 | 0.032245 |
| on | linear_attn_out | Q5_K | apr | 108 | 0.016974 | 0.062262 |
| on | linear_attn_out | Q5_K | llama | 108 | 0.016144 | 0.063286 |
| on | z | Q4_K | apr | 108 | 0.004016 | 0.013918 |
| on | z | Q4_K | llama | 108 | 0.003750 | 0.013164 |

**Layer-0 `linear_attn_out-0` collapse (the brief's test).**
- OFF: 0.034282–0.042371 at p4 pos 0–3 and orig 4/28.
- ON: 0.000000–0.000002.

This is below the OFF pre-`ssm_out` level (`final_output-0` OFF 0.002259–0.004327), because ON makes `final_output-0` itself 0.000000: the Q5_K `attn_qkv` feeding it is emulated too.

## 7. Residual with the switch ON (observations, not verdicts)

1. **First departure** (`cmp/sublayer_*_on.tsv`, layers 1–3):
   - **At p4 pos 0–1 and orig pos 4, every point through `attn_norm-3` is ≤ 0.000001** (orig 4: `conv_output_silu-2` is 0.001939, and `l_out-2` reaches 0.014620). At p4 pos 0/1, the first point above 1e-6 is `attn_pregate-3` (0.000212 / 0.000346), in **layer 3, full attention, mixer**. Across the Q4_K `attn_output` matmul it becomes `attn_output-3` 0.002276 / 0.007835.
   - At p4 pos 2–3 and orig 28, points in layers 1–2 (DeltaNet `a_softplus`, `gate`, `ffn_out`) already depart by 7e-5 to 0.018. At p4 pos 3, `l_out-0` is 0.000235.
   - Where exactly that departure starts is **[U]** (§8).
2. **The f16 KV cache.**
   - The probe log shows `llama_kv_cache: … K (f16), V (f16)` and `resolve_fused_ops: Flash Attention enabled`.
   - At p4 pos 0 a single key has softmax weight 1, so `attn_pregate-3` is V_0 read from the cache. Rounding apr's ON `attn_pregate-3` to f16 lowers rel L2 from **0.000212 to 0.000020**, and makes **2044 of 2048** elements bit-equal to llama's (`cmp/kv_f16_check.tsv`).
   - At pos 1 (two keys, a real softmax) rounding the output does not reproduce llama (0.000346 → 0.000405). That test needs f16 on K and V before attention.
3. **Largest remaining step** (`cmp/layer_steps_on.tsv`):

   | label pos | largest ON step | step | OFF largest step there |
   |---|---|---|---|
   | p4 1 | **`l_out-21`, layer 21 gated DeltaNet, FFN (Q4_K gate/up, Q6_K down)** | 0.012802 | `attn_residual-0` (0.039963) |
   | p4 0 | `attn_residual-15` (layer 15 full attention, mixer) | 0.039400 | the same point (0.085038) |
   | p4 2 | `attn_residual-23` (layer 23 full attention, mixer) | 0.008998 | `attn_residual-0` |
   | p4 3 | `l_out-2` (layer 2 DeltaNet, FFN) | 0.012007 | `attn_residual-0` |
   | orig 4 | `l_out-2` | 0.012193 | `attn_residual-0` |
   | orig 28 | `l_out-2` | 0.012575 | `attn_residual-0` |

   - At `l_out-21`, p4 pos 1, each engine's FFN kernel departs from float64 on its own input by a similar amount: apr 0.020865, llama 0.019409 (`kernel_isolation_on.tsv`).
   - The p4-pos-1 excess over the contrast positions is now largest at `l_out-16` (+0.008714), `l_out-20`, `attn_residual-18`, `l_out-17` and `l_out-21`. All are DeltaNet sublayers in layers 16–21.
4. **Per-qtype attribution** (the §6 gap table):
   - Q5_K alone removes the most on every prompt: p4 27.3%, orig 11.5%, p3 6.6%, p2 4.7%, p1 3.4%.
   - Q6_K alone **increases** the gap on every prompt (−0.9% to −5.3%).
   - Q4_K alone is −3.7% to +1.5%, and Q8_0 alone −0.2% to +4.8%.
   - The parts do not add up to the whole.
5. Not every ON logits summary improves:
   - p2's minimum cosine falls from 0.995370 to 0.993047;
   - p1's argmax mismatches rise from 4 to 8, and p1's gap removed is 0.8%;
   - p4's n < 0.98 falls from 4 to 0, and its argmax mismatches from 6 to 3.

## 8. Verdict (only what the data supports)

- **The hypothesis's MECHANISM holds.**
  - llama's CPU matmul departs from `dequant(W)·x` by quantizing the activation. apr made to compute the same way reproduces llama at every observed point through layer 2 (≤ 2e-6), with byte-level fidelity of the ported primitives against ggml's own C.
  - The OFF "largest step" at layer-0 `ssm_out` was the reference's arithmetic, not an apr defect.
- **The hypothesis's MAGNITUDE ("most of the gap") is killed on this data.**
  - Emulating every quantized matmul removes 0.8%–32.3% of the Frobenius logits gap on 5 of 5 prompts, and the majority remains on each.
  - The remainder is first observed in the first full-attention layer's mixer. At position 0 it is consistent with llama's f16 KV cache.
- The Qwen3.5 parity row stays **[U]**. No threshold is set.

## Remaining unknowns, each with the command that would measure it

- **[U] Whether llama's f16 K/V cache (plus the Flash Attention path) accounts for the ON residual at positions ≥ 1.**
  Intel: `timeout 900 apr_raw_logits -m MODEL -f prompt-4.txt -ngl 0 -t 8 -c 82 -b 82 --per-token -ctk f32 -ctv f32 [--dump-tensors … as run_sublayer.sh] --raw-out p4-kvf32.bin < /dev/null`, and the same with `-fa off`. Then `compare_raw_logits.py p4-kvf32.bin runs/on-p4.bin` and `compare_layerwise.py` against `on-dump-p4`.
- **[U] Which layer-0/1/2 sub-point first departs at p4 pos 2–3 and orig pos 28 with the switch ON.** (Candidates are the DeltaNet recurrent state and the Q8_0 `ssm_alpha`/`ssm_beta` dots, whose AVX2 float order differs from the ported generic by ≤ 4.2e-7 relative on fixtures.)
  Run `awk -F'\t' '$5<=2' cmp/sublayer_p4_on.tsv` over every layer-0 point at pos 3, then `kernel_isolation.py` extended with `ssm_alpha`/`ssm_beta` (Q8_0).
- **[U] The effect of the AVX2 float assembly (vec_dot SIMD, repack `gemv_q4_K_8x8`) that the generic port does not reproduce.** Port `ggml_gemv_q4_K_8x8_q8_K` and the AVX2 dots' accumulation order, add them to `ggml_emul_fixtures.c` (both symbols are exported), and rerun `run_emulation.sh`.
- **[U] ON logits on intel.** apr ran on lambda. OFF logits are byte-identical across hosts (`MEASUREMENT.md`), but ON was not run on intel. Copy the binary, run `APR_EMULATE_GGML_VECDOT=1 qwen35_layer_obs noop MODEL prompt-4.ids on-p4-intel.bin`, then `cmp` it with `runs` `6ccada2b…1f54`.
- **[U] Carried over:** llama's batched-vs-per-token noise floor per sub-layer; markup content vs position for the p4 early-position excess (`LAYERWISE.md`, `VARIATION.md`).

## Files (all under `emulation/`)

| file | what |
|---|---|
| `ggml_emul_fixtures.c`, `build_and_run.sh`, `build_and_run.transcript` | C harness linked against ggml's `.so`; fixture outputs; FMA objdump counts |
| `fixtures.tsv`, `fixtures.rs.txt`, `fixtures.bin` | 10 fixture cases: FNV hashes, generic and SIMD dot bits, Q8_0 ref/cpu equality, repack quantizer equality; the Rust table; raw rows and blocks |
| `llama_repack_probe.excerpt.log` | probe log lines: 98 `repack tensor … q4_K_8x8`, buffer sizes, `K (f16)`/`V (f16)`, Flash Attention |
| `qwen35_layer_obs.rs` | the uncommitted example as built (prints the switch and counters) |
| `run_emulation.sh`, `run_emulation.transcript`, `runs/*.log`, `runs/logits.sha256.txt`, `runs/*.manifest.sha256.tsv` | apr runs: invariance, ON, per-qtype, repeats; dump manifests |
| `compare_emulation.sh`, `compare_emulation.transcript`, `logits_gap.py`, `tables.py`, `kv_f16_check.py` | comparisons |
| `cmp/logits-<prompt>-<variant>.{tsv,json}`, `cmp/gap-<prompt>.tsv`, `cmp/sublayer_{p4,orig}_{off,on}.tsv` (+ selfcheck), `cmp/layer_steps_{off,on}.tsv`, `cmp/kernel_isolation_{off,on}.tsv`, `cmp/kv_f16_check.tsv`, `cmp/tables.md` | every number above |
| `unit_red.log`, `unit_green.log`, `qwen35_lib.log` | TDD RED/GREEN, qwen35 lib tests |
