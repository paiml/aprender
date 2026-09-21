# PMAT-3091 — SCALAR llama.cpp reference vs apr scalar emulation (Qwen3.5-0.8B Q4_K_M, CPU)

**Question.** Is there ANY algorithmic difference between apr's Qwen3.5 forward and ggml's, or is the whole parity gap kernel
selection and float order? **Hypothesis under test:** with llama built SCALAR (no SIMD, no repack) in config C, and apr emulating
that exact scalar arithmetic, the two engines agree to ~1e-6 at every layer and at the logits.

## Tree
- evidence: layer-3091 on top of `e92e6f166` (branch PMAT-3091-layerwise); code: obs-3091 `543da1188` (branch PMAT-3091-layer-observer,
  local measurement branch, stacked on #3114 `31448f6c3`). Examples used from obs-3091 (untracked during iterations 0-6, committed at `60ac62327`): `qwen35_layer_obs.rs`
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
1. `bash scalar/build_scalar.sh` (intel) → `build_scalar.transcript`, `build.transcript`, `cmake_cache_flags.txt`.
2. `bash scalar/build_and_run_scalar_fixtures.sh` (intel): the UNCHANGED `emulation/ggml_emul_fixtures.c` re-linked against the scalar libs
   → `fixtures_scalar.{rs.txt,tsv}`, objdump counts, compile flags.
3. `bash scalar/run_scalar_ref.sh` (intel): config C (`--kv-type f32 --flash-attn off`) per-token, orig ×2, p4, sub-layer dumps
   (p4 pos 0-3, orig pos 4/28, the KVCONFIG regex), p1-p3 → `run_scalar_ref.transcript`, `runs.sha256.txt`, `SC-sub-*.manifest.sha256.tsv`.
4. `gcc system_info.c` against each libllama (intel) → `system_info.txt`.
5. `cargo test -p aprender-serve --lib ggml_vecdot_emul` RED → `unit_red.transcript`, GREEN → `unit_green.transcript`.
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
- RED (`unit_red.transcript`): with the scalar entry points stubbed to the native arithmetic, 3 of 11 tests fail. The q8_0 dot for case 0 is 0xc11dca90
  vs scalar C 0xc11dca91, the q4_K dot for case 1 is 0x426a3b0e vs 0x426a3b08, and the `scalar` switch value is refused.
- GREEN (`unit_green.transcript`, `gate1.transcript`): 11/11. Scalar fixtures, 10 cases: Q8_K quantize, Q4_K/Q5_K/Q6_K dots, Q8_0 ref quantize + dot all bit-exact.
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

## Iterations (lane B slot 3, 2026-09-15 23:05-23:20 UTC; scripts `scalar/run_iteration.sh`, outputs `scalar/iter/<label>/`)

Each iteration ports ONE departing op into the `=scalar` path, re-checks the switch-OFF invariance and `=1`, then re-walks
every dumped point (`scalar/walk_points.py`) to find the next first departure. Code: obs-3091 `b4c736668` (port + fixtures),
`b84e694eb` (silu wiring), `100bd3482` (norms + delta rule), `7a451e3a1` (swiglu). No threshold is judged anywhere.

| # | op ported | bit-exact fixtures | first departure after | class | max rel L2 after (p4 / orig) | argmax mismatches after (orig / p4) |
|---|---|---|---|---|---|---|
| 0 | (baseline, `543da1188`) | 10 (quantized dots) | p4 pos0 `conv_output_silu-0` | float-approx | 0.1727 / 0.0988 | 2 of 78 / 3 of 82 |
| 1 | `ggml_vec_silu_f32` SSE2 (`ggml_v_expf`+`ggml_v_silu` lanes, libm tail) at `conv_output_silu` and `build_norm_gated` | 84 of 84 (RED 53/84 with libm stubs) | p4 pos0 `q_conv_predelta-0` (648/2048) | float-order | 0.1495 / 0.0925 | 2 of 78 / 2 of 82 |
| 2 | `build_gdn_l2_norm` + gated RMSNorm in ggml order (rms_norm double sum, `mean=(float)(sum/n)`, then `*1/sqrtf(n)`) | (same 84; norms proven in-engine) | p4 pos0 `attn_output-0` (545/2048) | float-order | 0.1354 / 0.0975 | 4 of 78 / 2 of 82 |
| 3 | delta rule dots through `ggml_vec_dot_f32`'s double accumulator | (same 84) | p4 pos0 `ffn_out-0` (410/1024) | float-order | 0.1007 / 0.0971 | 1 of 78 / 3 of 82 |
| 4 | `ggml_vec_swiglu_f32` SSE2 at both FFN sites | 84 of 84 | p4 pos0 `attn_norm-2` (28/1024) | float-order | 0.1036 / 0.0965 | 3 of 78 / 1 of 82 |

Invariance held at EVERY iteration: `off-orig` = `f6f79264…c0252`, `off-p4` = `92f1b54d…d3ee9` (sha_equal=true ×4 runs), and each
`=1` subject was byte-identical to the committed `on-{orig,p4}` (cmp rc=0 ×4). `scalar/emulation/ggml_sse2_fixtures.c` calls the
SCALAR libggml-cpu's exported `ggml_vec_silu_f32`/`ggml_vec_swiglu_f32`/`ggml_vec_soft_max_f32` over widths 1, 3, 4, 5, 255, 256,
6144 × 5 input kinds (±0, subnormals, ±inf, NaN, ±FLT_MAX, the 126/192 branch edges, masked -inf, |x| up to 200): 84 rows, all
bit-exact, and 51 of the 84 differ when the SAME source is linked against the NATIVE build (so the fixtures discriminate).

### Classification of every departure met
- `conv_output_silu` — **float-approx**: ggml's SSE2 `ggml_v_expf` polynomial vs libm `expf` (same formula `x/(1+exp(-x))`).
- `q_conv_predelta`/`k_conv_predelta` — **float-order**: llama `ggml_scale(ggml_rms_norm(x, eps/n), 1/sqrtf(n))` (`models.h:14`),
  a DOUBLE sum of `x*x`, `mean=(float)(sum/n)`, `1/sqrtf(mean+eps/n)`, then `*1/sqrtf(n)` (`ops.cpp:3957-3981`, `4737`);
  apr `x/sqrt(sum_f32 + eps)` (`forward_qwen35.rs` `l2_norm_per_head`). Algebraically identical.
- `attn_output` (delta rule) — **float-order**: both `S^T k` and `S^T q` go through `ggml_vec_dot_f32`, whose scalar branch
  accumulates `(ggml_float)(x[i]*y[i])` in a double and narrows once (`vec.cpp:128-136`, called at `ops.cpp:11018,11031`);
  apr accumulated in f32. Every other step of the kernel (`ops.cpp:10895-11045`) is the same op in the same order as apr's.
- `ffn_out` — **float-approx**: the same SSE2 silu inside `ggml_vec_swiglu_f32` (`vec.cpp:427-430`).
- `attn_norm-2` (open) — the remaining RMSNorm: apr `crate::gguf::ops::rms_norm_into` sums in f32, ggml in double. Same class as #2.

**No ALGORITHMIC departure was found.** Every op reached was the same mathematics in a different float arithmetic, and each
port moved the first departure strictly later in the callback order (`conv_output_silu-0` → `q_conv_predelta-0` → `attn_output-0`
→ `ffn_out-0` → `attn_norm-2`, i.e. past ALL of layer 0 and layer 1 at p4 pos 0).

### Iterations 5-6 (scope widened to `crates/aprender-serve/src/gguf/ops.rs`, switch-gated, §8 decision)

New dumps live under `/mnt/nvme-raid0/parity-tmp/scalar3091-it5plus/` (lambda `/` was at 96-99%); the llama refs stay at
`/tmp/scalar3091/ref`. `scalar/run_iteration.sh` takes `RUNROOT` for this.

| # | op ported | bit-exact fixtures | first departure after | class | max rel L2 after (p4 / orig) | argmax mismatches after (orig / p4) |
|---|---|---|---|---|---|---|
| 5 | `rms_norm_into_ggml` at `attn_norm`, `attn_post_norm`, final `norm` (double row sum, `mean=(float)(sum/n)`, `x*scale*w`) | in-engine (all of p4 pos0 bit-equal) | p4 pos1 `attn_pregate-3` (first FULL-ATTENTION layer) | float-order | 0.1082 / 0.0710 | 2 of 78 / 1 of 82 |
| 6 | full-attention path in ggml order: per-head q/k RMSNorm double sum, UNSCALED KQ dot via `ggml_vec_dot_f32`, `ggml_soft_max_ext` (kq_scale multiply, 256-padded `n_kv`, SSE2 exp lanes, double sum, `(float)(1/sum)` scale), double-accumulated KQV | 84 of 84 (the soft_max rows now exercised in-engine) | **none** | — | **0.000000 / 0.000000** | **0 of 78 / 0 of 82** |

Iteration 5 made EVERY dumped point of p4 pos 0 bit-equal (all 18 DeltaNet layers and both FFN sites). Iteration 6 closed the
full-attention layers, and with it the whole model.

## RESULT: apr(scalar) is BIT-IDENTICAL to llama-scalar-C

- Walk: `1500 of 1500` points bit-equal on p4 (pos 0-3) and `750 of 750` on orig (pos 4, 28); `max_rel_l2 = 0.000000e+00`,
  `first_not_bit_equal = None` on both (`scalar/iter/it6-fullattn/walk_{p4,orig}.summary.txt`).
- Logits: `cmp $R/SC-orig.bin <apr>` and `cmp $R/SC-p4.bin <apr>` both **rc=0** — the full per-token logit streams are
  BYTE-IDENTICAL (sha256 `e22120ae…` orig, `b40d6649…` p4), not merely close. 0 argmax mismatches at 78 + 82 positions,
  min cosine 1.000000.
- Invariance held at every one of the six iterations: `off-orig` = `f6f79264…c0252`, `off-p4` = `92f1b54d…d3ee9`, and each
  `=1` subject byte-identical to the committed `on-{orig,p4}`.
- The measured binary is the committed code — but NOT as this line first claimed; see the verification pass below,
  which replaces the interrupted re-run with two clean-tree reproductions (`it6-recommit`, `it6-final`).

### Amplification floor: is ~1e-6 reachable while llama quantizes activations to Q8_K?

**Yes — 0 is reachable, and the floor is far below 1e-6.** `scalar/ulp_amplification.py` feeds ONE Q8_K-quantized matmul the
same f32 activation row (llama's own `attn_norm-0` at p4 pos 0), then the same row with ONE element moved by exactly 1 ulp
(`scalar_isolate` job `ulpamp`), for Q5_K `blk.0.attn_qkv` (1024→6144) and Q4_K `blk.0.attn_gate` (1024→2048), 9 element
positions each (`cmp/ulp_amplification.tsv`):

| perturbed element | in rel L2 | out rel L2 (Q5_K / Q4_K) | amplification |
|---|---|---|---|
| 7 of 9 (ordinary elements, and ±0) | 1.5e-9 … 3.0e-9 | **0.0 / 0.0** — output BIT-IDENTICAL | 0 |
| element 723 (`-4.7527`, the max-\|x\| element, which sets the Q8_K block scale) | 1.2e-8 | 1.14e-7 / 1.50e-7 | 9.45× / 12.41× |

So Q8_K quantization is mostly a *snap-back*: a 1-ulp input change usually produces the identical output, because the code
lands in the same bucket. It amplifies only when the perturbed element is the block's scale-setter, and even then by ~10×,
i.e. 1 ulp in → ~1.5e-7 out. The earlier 1224× figure was amplification of an ALREADY-1e-6-scale difference, not a floor.
The residual we were chasing was never the arithmetic's own noise — it was the five ordered-arithmetic differences above,
and removing them removed the gap entirely.

### Stop condition: (a) reached — exceeded

apr(scalar) equals llama-scalar-C at all 2250 points to rel L2 **0.0** (bit-identical, better than the ≤1e-6 the condition
asks for) and the logits argmax match at every position on BOTH orig and p4 (in fact the logit bytes are identical).
No ALGORITHMIC departure was found anywhere: every one of the five departures met was the same mathematics in a different
float arithmetic, and each was fixed by reproducing ggml's arithmetic, never by changing what is computed.

### Verification pass (lane B slot 3 continuation, 2026-09-16 04:40-04:52 UTC, lambda)

The iteration-6 result above was produced by a binary built from a DIRTY tree — its transcript reads
`obs_code=391ce6826 dirty_tracked=2` — and the re-run that was to prove it against the commit was KILLED partway:
the run dir `/mnt/nvme-raid0/parity-tmp/scalar3091-it5plus/it6-fullattn/` still holds a 0-byte `off-p4.log` beside
an `off-orig.bin` written seven minutes after the rest. So the claim "iteration 6 was re-run" was ahead of its
evidence. Two clean-tree reproductions now stand in its place (`scalar/iter/it6-{recommit,final}/`):

| run | obs_code | dirty | subject sha256 | walk p4 / orig | logits `cmp` vs SC | invariance OFF |
|---|---|---|---|---|---|---|
| `it6-recommit` | `60ac62327` | 0 | `f7b4a7d6…eec88` | 1500/1500 and 750/750 bit-equal, max rel L2 `0.000000e+00` | rc=0 both | `f6f79264…`, `92f1b54d…`, `=1` cmp rc=0 |
| `it6-final` | `b59433f7d` | 0 | `f5807796…469af` | same | rc=0 both | same |

`it6-recommit`'s binary hashes EXACTLY as the `it6-fullattn` binary (`f7b4a7d6…eec88`), so the dirty tree at 04:30
held the same bytes the commit later captured, and its `walk_{p4,orig}.tsv` are sha256-identical to
`it6-fullattn`'s (`e6df8e4e…`, `af2890a6…`): the measurement reproduces bit for bit, twice, from a clean tree.
Row-level re-check of the walk (not just its summary line): every one of the 1500 p4 rows and 750 orig rows has
`bit_equal == n`; every one of the 78 + 82 logit rows has `argmax_match=True` at cosine 1.000000.

**A gate was RED and had not been re-run.** `cargo test -p aprender-serve --lib qwen35` last passed at 01:18 — a log
that predates iterations 5 and 6 — and FAILS at `77778d20d` (21 passed, 1 failed). Iteration 6 added the
`Qcur_normed` / `Kcur_normed` observation points to the full-attention path without declaring them in
`QWEN35_OBS_ATTENTION_POINTS`, from which `observer_emits_llama_names_in_graph_order_with_exact_residuals` derives
its expected sequence. Fixed in `b59433f7d` by declaring them, with the note that llama computes those per-head
q/k RMSNorm nodes but leaves them UNNAMED (`node_30` / `node_33`, `RMS_NORM f32 128x16`, `layerwise/llama_tensor_names.tsv`),
so they have no llama counterpart and are apr-side diagnostic points. They never entered any comparison:
`walk_points.py` iterates LLAMA's manifest, and the extra apr rows are exactly 48 per prompt
(2 points x 6 full-attention layers x 4 positions; apr's p4 manifest has 1548 rows against the walk's 1500).
22 passed after the fix. The measured numbers are unchanged by it — `it6-final` reproduces `it6-recommit` exactly.

Gate on `b59433f7d` (logs `scalar/gate_*.transcript`): `cargo test -p aprender-serve --lib ggml_vecdot_emul` 12 passed
rc=0 · `cargo test -p aprender-serve --lib qwen35` 22 passed rc=0 · `cargo fmt --all -- --check` rc=0 ·
`bash scripts/check_llama_pin.sh --self-test` rc=0 (PASS, the pin discriminates).

The amplification floor was reproduced independently: `scalar_isolate` rebuilt from the now-committed source
(sha256 `83c16a83…9371c`), `python3 scalar/ulp_amplification.py` re-run, and `diff` against
`cmp/ulp_amplification.tsv` rc=0 — the same 18 rows, the same two amplifying elements (9.450x / 12.412x).
The `ulpamp` job perturbs by `f32::from_bits(x.to_bits() + 1)`, i.e. exactly one ulp, read from the committed source.

**Run logs are committed as `.transcript`.** `.gitignore:38` is `*.log`, so NO `.log` file in this evidence tree is
tracked — every `*.log` citation was unreachable from the repo. The cited run logs are therefore committed as
byte-identical `.transcript` copies beside them (`scalar/{build,unit_red,unit_green,gate1-4,gate_*}.transcript`).
Two citations remain deliberate scratch paths, NOT repo paths: the interrupted run's `off-p4.log` under
`/mnt/nvme-raid0/parity-tmp/…` and kvconfig's `A-v-orig.log` (an intel scratch file, labelled as one there).
The sibling evidence docs of this directory (EMULATION.md, KVCONFIG.md) were given the same treatment in the
PMAT-3303 landing pass: their `.log` citations now name byte-identical `.transcript` copies (`cmp` rc 0).

### Remaining [U] (each with its command)
- ~~[U] RMSNorm sum order~~ — CLOSED by iteration 5.
- ~~[U] softplus/sigmoid, full-attention q/k norm + RoPE + f32 softmax measured only with already-differing inputs~~ — CLOSED:
  iteration 6 made every one of them bit-equal in-engine. ggml's softplus is `(x > 20.0f) ? x : logf(1.0f + expf(x))`
  (`ggml-impl.h:107`, `unary-ops.cpp:80`) and its sigmoid `1.f/(1.f + expf(-x))` — character for character apr's, threshold
  included; RoPE needed no change (apr's iterative theta already matches `ggml_rope_multi`).
- ~~[U] Whether ~1e-6 is reachable with Q8_K activation quantization between layers~~ — CLOSED by the amplification floor above.
- [U] Sub-layer dumps for p1-p3 (logits only): `run_scalar_ref.sh` section 2 with p1-p3 positions.
- [U] Parity row (no threshold set).
