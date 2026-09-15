# PMAT-3091 — SCALAR llama.cpp reference vs apr scalar emulation (Qwen3.5-0.8B Q4_K_M, CPU)

**Question.** Is there ANY algorithmic difference between apr's Qwen3.5 forward and ggml's, or is the whole parity gap kernel
selection and float order? **Hypothesis under test:** with llama built SCALAR (no SIMD, no repack) in config C, and apr emulating
that exact scalar arithmetic, the two engines agree to ~1e-6 at every layer and at the logits.

## Tree
- evidence: layer-3091 on top of `e92e6f166` (branch PMAT-3091-layerwise); code: obs-3091 `543da1188` (branch PMAT-3091-layer-observer,
  local measurement branch, stacked on #3114 `31448f6c3`). Uncommitted examples used from obs-3091: `qwen35_layer_obs.rs`
  (as in EMULATION.md) and `scalar/scalar_isolate.rs` (copied here, sha256 of the release binary `556adc04…4d14`).
- apr subject binary `qwen35_layer_obs` sha256 `e57e3053…0da` (lambda, CARGO_TARGET_DIR=/mnt/nvme-raid0/targets/obs-3091).
- llama.cpp `d1d3c3396` (clean tree), SCALAR build `~/parity-ref/llama-scalar-d1d3c3396` (intel), cmake cache (`scalar/cmake_cache_flags.txt`):
  `CMAKE_BUILD_TYPE=Release BUILD_SHARED_LIBS=ON GGML_NATIVE=OFF GGML_SSE42=OFF GGML_AVX=OFF GGML_AVX2=OFF GGML_AVX_VNNI=OFF GGML_AVX512=OFF
  GGML_AVX512_VBMI=OFF GGML_AVX512_VNNI=OFF GGML_AVX512_BF16=OFF GGML_AMX_TILE/INT8/BF16=OFF GGML_BMI2=OFF GGML_FMA=OFF GGML_F16C=OFF
  GGML_CPU_REPACK=OFF GGML_LLAMAFILE=OFF GGML_OPENMP=ON` (option names read from `ggml/CMakeLists.txt:123-170,197,247` and
  `ggml/src/ggml-cpu/CMakeLists.txt:306-345,582`; `INS_ENB` defaults the ISA options ON when NATIVE is OFF, so each is set OFF).
  libggml-cpu sha256 `5f087688…99ea`, libggml-base `bddf9ff4…1463`, libllama `5cdec1e9…2b68`, producer `f7443c64…dc45`
  (same `apr_raw_logits.cpp` as kvconfig, cmp rc=0).
- model `Qwen3.5-0.8B-Q4_K_M.gguf` sha256 `bd258782e35f7f458f8aced1adc053e6e92e89bc735ba3be89d38a06121dc517`.
- hosts: llama on intel (`mac-server`, 32 cores, -t 8, load 27-80 — the shared build box), apr on lambda (`noah-Lambda-Vector`, 48 cores, load 3-36).
  2026-09-15 22:33-22:47 UTC (intel clock).

## Commands (all scripts in `scalar/`, transcripts beside them)
1. `bash scalar/build_scalar.sh` (intel) → `build_scalar.transcript`, `build.log`, `cmake_cache_flags.txt`.
2. `bash scalar/build_and_run_scalar_fixtures.sh` (intel): the UNCHANGED `emulation/ggml_emul_fixtures.c` re-linked against the scalar libs
   → `fixtures_scalar.{rs.txt,tsv}`, objdump counts, compile flags.
3. `bash scalar/run_scalar_ref.sh` (intel): config C (`--kv-type f32 --flash-attn off`) per-token, orig ×2, p4, sub-layer dumps
   (p4 pos 0-3, orig pos 4/28, the KVCONFIG regex), p1-p3 → `run_scalar_ref.transcript`, `runs.sha256.txt`, `SC-sub-*.manifest.sha256.tsv`.
4. `gcc system_info.c` against each libllama (intel) → `system_info.txt`.
5. `cargo test -p aprender-serve --lib ggml_vecdot_emul` RED → `unit_red.log`, GREEN → `unit_green.log`.
6. `bash scalar/run_apr_scalar.sh` (lambda): switch-OFF invariance, `APR_EMULATE_GGML_VECDOT=scalar` subjects orig/p4/p1-p3 (+orig n=2),
   dumps, then the committed comparators (`compare_raw_logits.py`, `layerwise/compare_layerwise.py`, `layerwise/layer_steps.py`) → `cmp/`.
7. `python3 scalar/walk_points.py` (full-precision walk of every dumped point, callback order) → `cmp/walk_{p4,orig}.tsv`;
   `python3 scalar/curve_summary.py` → `cmp/curve_summary.txt`.
8. `python3 scalar/isolate_matmul.py <scalar_isolate>` → `cmp/isolate_matmul.tsv`; `python3 scalar/isolate_silu.py <scalar_isolate>` → `cmp/isolate_silu.tsv`.

## Engaged proofs (scalar llama)
- llama's own `llama_print_system_info()` (`system_info.txt`): scalar `CPU : OPENMP = 1 |`; native `CPU : SSE3 = 1 | SSSE3 = 1 | AVX = 1 | AVX2 = 1 |
  F16C = 1 | FMA = 1 | BMI2 = 1 | AVX512 = 1 | AVX512_VNNI = 1 | LLAMAFILE = 1 | OPENMP = 1 | REPACK = 1 |`.
- repack: 0 `repack` lines in every scalar run log (with `-v`); the native `-v` default run (kvconfig `A-v-orig.log`) has 103.
- objdump: libggml-cpu 144981 instructions, ymm=0, zmm=0, FMA=0, VEX `v*` mnemonics=0; same zero counts for libggml-base and libllama.
- fp-contract: ggml-cpu sources compile with `-O3 -fPIC -fopenmp -std=gnu11|gnu++17` and no `-m` flag (compile_commands.json). gcc 11.4 defaults
  to `-ffp-contract=fast` in GNU mode, but without `-mfma` it has no FMA instruction to contract into: the generic dots contain 0 FMA
  instructions (`ggml_vec_dot_{q4_K,q5_K,q6_K}_q8_K_generic`, `ggml_vec_dot_q8_0_q8_0_generic`). So the scalar build is NOT contracted.
- `quantize_row_q8_0` in the scalar build is 3 instructions (a tail call to `quantize_row_q8_0_ref`, `arch/x86/quants.c:92-95`), and
  `quantize_row_q8_0_ref` calls `roundf@plt`. Fixture column `q80_ref_eq_cpu` = 1 for all 10 cases.
- **Not scalar, and no cmake option can make it so:** x86-64 baseline defines `__SSE2__` (`gcc -dM -E`: SSE2 1, SSE3 0). `ggml_vec_silu_f32`
  (`ggml-cpu/vec.cpp:390-393`) and `ggml_vec_swiglu_f32` (`vec.cpp:427-430`) take their `__SSE2__` branch in the scalar build
  (66 `*ps` instructions in `ggml_vec_silu_f32`), i.e. `ggml_v_silu`/`ggml_v_expf` (`vec.h:1278-1315`), an approximate exp.
- Scalar C is deterministic: `SC2-orig` cmp `SC-orig` rc=0 (`e22120ae…fdb5`). It differs from the native-build C: `SC-{orig,p4}` vs kvconfig `C-{orig,p4}` cmp rc=1.
  Each dump run's logits cmp its logits run (rc=0); the comparator self-check `callback result_output == per-token bin row: True` at every dumped position.

## Scalar emulation variant (obs-3091 `543da1188`, `crates/aprender-serve/src/quantize/ggml_vecdot_emul.rs`)
`APR_EMULATE_GGML_VECDOT=scalar` = every ported qtype with the scalar build's arithmetic: the generic dots without `mul_add`, the reference
`quantize_row_q8_0_ref` (roundf, `id = 1/d`, scalar `ggml_compute_fp32_to_fp16` ported bit for bit), and the plain Q4_K dot (no repack).
`=1` keeps its functions and dispatch, unchanged.
- RED (`unit_red.log`): with the scalar entry points stubbed to the native arithmetic, 3 of 11 tests fail. The q8_0 dot for case 0 is 0xc11dca90
  vs scalar C 0xc11dca91, the q4_K dot for case 1 is 0x426a3b0e vs 0x426a3b08, and the `scalar` switch value is refused.
- GREEN (`unit_green.log`, `gate1.log`): 11/11. Scalar fixtures, 10 cases: Q8_K quantize, Q4_K/Q5_K/Q6_K dots, Q8_0 ref quantize + dot all bit-exact.
  The scalar fixtures share every input/weight/Q8_K hash with the native ones and differ in 12 dot bit patterns plus the case-7 Q8_0 blocks.
- In-engine bit-exactness (`cmp/isolate_matmul.tsv`): for all 18 DeltaNet layers × 6 dumped positions, apr's scalar kernel on **llama's own dumped input**
  reproduces llama's output: attn_norm→z (Q4_K attn_gate) and final_output→linear_attn_out (Q5_K ssm_out), **331776/331776 elements bit-equal**.
  On apr's own input it reproduces apr's dump, 331776/331776.

## Invariance (switch OFF)
`off-orig` sha256 = `f6f792649014dcef…c0252` (true), `off-p4` = `92f1b54d504a53f7…d3ee9` (true); mask printed `scalar: false`, emulated matmuls `[0,0,0,0]`.
Switch ON: mask `scalar: true`, counts orig `[7644, 2808, 1326, 2808]`. `scalar2-orig` cmp `scalar-orig` rc=0. `scalar-{orig,p4}` differ from the `=1` subjects (cmp rc=1).

## Result: apr(scalar) vs llama-scalar-C
**Logits** (`cmp/logits-*-scalar-vs-SC.json`; no threshold):

| prompt | positions | min cos | at pos | mean cos | argmax mismatches | max |Δ| |
|---|---|---|---|---|---|---|
| orig | 78 | 0.997685 | 30 | 0.999249 | 2 (pos 2, 77) | 1.297 |
| p4 | 82 | 0.991322 | 1 | 0.997874 | 3 | 2.041 |
| p1 | 83 | 0.998382 | 3 | 0.999527 | 4 | 0.892 |
| p2 | 117 | 0.996157 | 100 | 0.998574 | 0 | 1.393 |
| p3 | 108 | 0.997008 | 61 | 0.999345 | 4 | 0.825 |

**Sub-layer curves** (`cmp/walk_*.tsv`, full precision; `cmp/sublayer_*_scalar_vs_SC.tsv` from the committed comparator):
max rel L2 over all 2250 points = **0.1727 at p4 pos 1 `linear_attn_out-21`** (orig: 0.0988 at pos 28 `linear_attn_out-8`). The engines do NOT
agree to ~1e-6. First point above 1e-5 per position (`cmp/curve_summary.txt`): p4 pos0 `conv_output_silu-8` (1.17e-3), pos1 `ffn_out-2` (3.4e-4),
pos2 `attn_pregate-3` (9.6e-4), pos3 `ffn_out-0` (4.5e-4); orig pos4 `state_predelta-1` (4.9e-4, state carried from earlier tokens), pos28 `a_softplus-1` (7.3e-5).

**First departing point:** p4 pos 0 (token 27), layer 0 (linear_gated_deltanet), `conv_output_silu-0`: 5636/6144 elements bit-equal, rel L2 4.54e-8.
Everything before it is bit-equal (`model.input_embed`, `attn_norm-0`), and so are the downstream `a_softplus-0`, `gate-0`, `beta_sigmoid-0` and `z-0`.

**Narrowed to one op: silu** (`cmp/isolate_silu.tsv`). At pos 0 the conv state is zero, so conv output = qkv·w[3] in both engines
(apr `causal_conv1d`, `forward_qwen35.rs:85-117`: `sum = 0; sum += state*w; sum += x*w[3]`; ggml `ggml_compute_forward_ssm_conv_f32`,
`ops.cpp:9744-9750`: `sumf += s[i0]*c[i0]` over the window [state×3, x]: same operations, same order), and the qkv matmul is bit-exact (above).
Pushing each engine's `attn_norm` through that and then silu, for all 18 DeltaNet layers at p4 pos 0:
- ggml's SSE2 `ggml_v_silu` (ported literally, `vec.h:1278` `ggml_v_expf` + `vec.h:1309` `ggml_v_silu`, no FMA ⇒ `MADD128 = add(mul)`) on llama's input
  reproduces llama's `conv_output_silu`: **110592/110592 bit-equal**.
- apr's silu (`forward_qwen35.rs:912-914`, `*x = *x / (1.0 + (-*x).exp())`, libm `expf`) on apr's input reproduces apr's dump: **110592/110592**.
- Cross: apr's silu on llama's input matches llama for only 96648/110592 (layer 0: 5636, exactly the walk's count).
- What each computes: llama `x / (1 + ggml_v_expf(0 - x))`, where `ggml_v_expf` is a vectorised approximation (`n = round(x/ln2)`, a degree-5 polynomial in
  the remainder `b`, scaled by `2^n` through bit tricks). apr: `x / (1 + expf(-x))` with glibc `expf`. Same formula; different exp arithmetic.
- Values (layer 0, same raw input, first 4): llama `0.0151475985, -0.000458114344, 0.00156917621, 0.422838897`;
  apr silu `0.0151475985, -0.000458114344, 0.00156917621, 0.422838867`.

**Why ulps become 1e-2** (`cmp/isolate_matmul.tsv`): apr and llama run the identical quantized matmul, but it quantizes its input to Q8_K
first, and a sub-ulp-scale input change flips codes. Measured amplification rel_l2(out)/rel_l2(in) on the bit-identical kernel:
p4 pos0 layer 8 attn_norm→z **1224×** (5.9e-7 → 7.2e-4); mean 7.0 over the 210 pairs with a nonzero input difference.

## Verdict
The hypothesis as stated is **falsified**: apr(scalar) and llama-scalar-C do not agree to ~1e-6 (max rel L2 0.17; logits min cos 0.9913, argmax
mismatches at 13 of 468 positions). But the reason is not an algorithmic difference. The "scalar" ggml build is not fully scalar: SSE2 is the x86-64
baseline, and ggml's silu/swiglu use their SSE2 `ggml_v_expf` approximation. The first departing point is exactly that op: each engine's silu is
bit-exact on its own inputs. Every quantized matmul is bit-identical in-engine (331776/331776), and the Q8_K activation quantization turns the
silu's ulp differences into 1e-3 steps (up to 1224× in one matmul). Up to and including the first departure, no measured op differs algorithmically.
Ops downstream of it (see [U]) were measured only with inputs that already differ, so this pass makes no claim that they are free of algorithmic
differences. The parity row stays [U].

## Remaining [U]
- [U] Whether ANY op after the first departure differs algorithmically (rms_norm, per-head l2_norm, softplus/sigmoid, delta rule `new_state`,
  gated RMSNorm `final_output`, full-attention q/k norm + RoPE + f32 softmax, swiglu). Next: extend `scalar/isolate_*.py` so apr's op runs on llama's own dumped
  input for each (`python3 scalar/isolate_<op>.py <scalar_isolate>`), or add a `ggml_v_silu` port to the forward under `=scalar` and re-run
  `bash scalar/run_apr_scalar.sh && python3 scalar/walk_points.py p4 /tmp/scalar3091/ref/SC-sub-p4 /tmp/scalar3091/runs/scalar-dump-p4` to find the next first departure.
- [U] FFN `ffn_out` jumps (p4 pos1 layer 2, pos3 layer 0) are presumed to be the same silu via `ggml_vec_swiglu_f32` (`vec.cpp:427-430`) plus Q8_K amplification;
  not isolated (`attn_post_norm → ffn_out` needs the ffn gate/up/down weights through `scalar_isolate`).
- [U] A llama build with SSE2 silu disabled (e.g. `-DCMAKE_C_FLAGS=-mno-sse2` is not an x86-64 ABI option; untested). Command to try: none settled.
- [U] Sub-layer dumps for p1-p3 (logits only here): `run_scalar_ref.sh` section 2 with p1-p3 positions.
- [U] Parity row (no threshold set).
