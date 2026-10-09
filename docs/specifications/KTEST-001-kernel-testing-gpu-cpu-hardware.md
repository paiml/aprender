# KTEST-001: Kernel provability and hardware testing (CPU SIMD · CUDA · ROCm/HIP · WGPU/Vulkan · Metal/Apple Silicon)

**Status:** v1.1 draft · 2026-09-28 (v1.1: peer-review fixes, §12) · **Operator:** Noah Gift · **Home (proposed):** `paiml/aprender` → `docs/specifications/KTEST-001-kernel-testing-gpu-cpu-hardware.md`
**Trains:**
- 0.71 "Verbs Are Fast": V5, the kernel registry (KREG-001 #4539);
- 0.73 "Runs Everywhere": E1 and E5;
- replaces the release gate #3715 v1 as **#3715 v2**.

**Proofs:** `KernelTesting.lean` (11 theorems, Lean 4.22 core, 0 `sorry`, sha256 `3fa84410…40ff7`).

**Marks:**
- `[V]` verified;
- `[C]` carried from a cited source;
- `[A]` assumed;
- `[U]` unknown, resolved in KTEST-00;
- `[D]` operator decision.

---

## ELI5 (30 seconds)

**Today:** to trust a release we test-drive every **car**: every model × host × verb × context. That's 1,008 test cells and about 6.7 GPU hours, and it all repeats on every commit.

**This spec:** test every **engine part** (kernel) once per chip, with maths that says exactly how wrong a part is allowed to be. A car is then approved *automatically* when all its parts are approved on that chip, plus one short test drive.

**Why it's better:**
- about 10× cheaper before any reuse (K4);
- nothing is re-tested unless the part itself changed;
- it catches the bugs end-to-end tests miss: one bad tile, a wrong subgroup size, a missing GPU architecture.

**It covers every kind of hardware we run:**
- CPU SIMD: x86 AVX2/AVX-512, ARM NEON/SVE;
- NVIDIA CUDA: sm_89, sm_121;
- AMD: through Vulkan/WGPU, and ROCm/HIP only on officially supported cards;
- Apple Silicon: Metal and NEON.

**Target:** release kernel gate ≤ 30 min, 0 unvalidated cells, 0 unregistered kernel dispatches.

---

## §0 Operating assumptions (state line 1 of the first message)

1. **Prove the part; derive the product.** Model readiness is *derived* by SHACL from kernel receipts plus one end-to-end smoke test per model × device. It is never inferred from a model's name or a backend flag (G5.1 in APR-QUALITY: "never infer hardware from a feature's name").
2. **Tolerances come from error analysis, never from taste.** Every kernel has an *error model*, a formula with a citation. A kernel with no error model is unregistered, and so refused (V5).
3. **The oracle is independent.** The reference is a scalar f64 CPU implementation of the *mathematical* definition. Never the optimized CPU backend: an optimized kernel doesn't judge another optimized kernel (the "producer is never the gate" rule, FLOW-002 §0.3).
4. **Quantization error is separated out.** Dequantization is **bit-exact** against pinned llama.cpp fixtures (PP-QUANT D5). Kernel tolerances are then measured against the *dequantized* weights, so they bound kernel error only, not quantization error.
5. **A backend is claimed only with receipts.** No receipt means the backend is unsupported, and it refuses cleanly. `HSA_OVERRIDE_GFX_VERSION` and similar "make an unsupported card pretend" knobs are never used in CI.
6. **Receipts bind to an input set, not a release sha.** A kernel receipt stays valid while its input set is unchanged: kernel source, codegen, toolchain, driver and device (FLOW-003 Def 14). The release sha binds only the per-model smoke test.
7. **No gate waivers.** A red cell blocks. `[U]` rows are resolved in KTEST-00 before any row claims coverage.

---

## §1 Ground truth: fleet × backend matrix

Source of truth: `paiml/infra` `machines/*/forjar.yaml`. Values below are `[C]` from APR-QUALITY-001 §6, stack-architecture.yaml and review-experiment-protocol §1; `[U]` rows are resolved in KTEST-00.

| Host | CPU ISA | GPU / backend | Role for kernels |
|---|---|---|---|
| `intel` | Xeon W-3245 (Cascade Lake): AVX2, AVX-512 F/BW/VL, VNNI `[C]` | **dual AMD GPU via Vulkan/WGPU** `[C]`; models `[U]`; ROCm eligibility `[U]` | CPU reference host (f64 oracle); AVX-512 + AVX2 + scalar paths; WGPU-Vulkan on AMD |
| `gx10` | Grace, aarch64 Neoverse V2 (Armv9.0-A): NEON + **SVE2, 128-bit vectors** `[C]` from the vendor architecture; confirm with `/proc/cpuinfo` flags in KTEST-00 | **NVIDIA GB10, CUDA 13.0, sm_121**, 120 GB unified `[C]` | aarch64 CPU paths; the only Blackwell CUDA; unified memory |
| `lambda-labs` | x86 `[U]` | **RTX 4090, CUDA sm_89** `[C]` | the only sm_89 source; **not a CI runner** (retired 2026-05-10; don't revive) `[C]`. Receipts come from scheduled, declared jobs only |
| `yoga` | x86 `[U]` | CUDA (arch `[U]`) `[C]` | CUDA-x86 build and tests |
| `mini` | Apple M4: NEON, dotprod, i8mm `[C]`; **SME `[U]`**: resolve from `sysctl hw.optional.arm` (FEAT_SME/FEAT_SME2). A reviewer's claim that macOS blocks SME in user space conflicts with public reports of user-space SME on M4, so the device query decides | **Metal via WGPU**, 16 GB unified (the binding constraint) `[C]` | Apple Silicon GPU + NEON |
| `framework16` | AMD Ryzen, model `[U]`. If it's Zen 4 or Zen 5, AVX-512 (F, VNNI, BF16) is present `[C]`; confirm with `lscpu` | Radeon iGPU/dGPU `[U]`. Integrated GPUs and mobile Navi 33 aren't on the ROCm 7.2 official list, so **WGPU/Vulkan only** unless KTEST-00 finds the gfx id on the matrix | AMD CPU paths; an extra Vulkan device if present |
| `jetson` | aarch64 | CUDA (Orin) | repair-or-drop pending `[C]`; **out of scope** until the decision |

**External facts (researched 2026-09-28):**

- **WGPU 30.0.1 feature support** `[C]` (docs.rs):

  | Feature | Supported on |
  |---|---|
  | `SHADER_F16` | Vulkan, Metal, DX12, WebGPU |
  | `SUBGROUP` | Vulkan, DX12, Metal (native only) |
  | `SUBGROUP_BARRIER` | Vulkan, Metal |
  | `SHADER_INT64` | Vulkan, DX12 (DXC), Metal (MSL 2.3+) |
  | `EXPERIMENTAL_COOPERATIVE_MATRIX` | Metal, Vulkan; **8×8 f32 only** |
  | `SHADER_FLOAT32_ATOMIC` | Metal 3.0+, Vulkan with `VK_EXT_shader_atomic_float` |

- **WGSL arithmetic rules** `[V]`, from the spec source (`gpuweb/wgsl/index.bs` §"Accuracy of Concrete Floating Point Expressions"):
  - `+ − ×` are correctly rounded; `fma` inherits from `x*y + z`;
  - `/` is **2.5 ULP** for |y| ∈ [2⁻¹²⁶, 2¹²⁶] (f32) or [2⁻¹⁴, 2¹⁴] (f16);
  - `exp`, `exp2`: **3 + 2·|x| ULP** (f32), **1 + 2·|x| ULP** (f16);
  - `log`, `log2`: **absolute 2⁻²¹ on [0.5, 2.0], else 3 ULP** (f32); absolute 2⁻⁷ on [0.5, 2.0] (f16);
  - `inverseSqrt`: **2 ULP**; `sqrt` **inherits from `1.0 / inverseSqrt(x)`**, so it is *not* correctly rounded;
  - `tanh`: the worse of absolute 1e-5 or the error inherited from sinh/cosh; `pow` inherits from `exp2(y·log2(x))`;
  - operation inputs and outputs **may be flushed to zero**, and implementations **may reassociate and fuse** operations (§"Reassociation and Fusion").
- **ROCm 7.2** (2026-01-21) `[C]`:
  - official consumer support includes the **RX 7700 series (RDNA3)** and RDNA4 cards such as the **RX 9060 XT LP**;
  - **integrated GPUs (780M/890M) are not listed**;
  - the current matrix is ROCm 10.0 (the compatibility page), and the fleet's cards are checked against it in KTEST-00.
- **llama.cpp `test-backend-ops`** `[C]`, the reference practice in our field:
  - compares every backend against CPU op by op, with an NMSE metric `Σ(a−b)²/Σa²`;
  - default max NMSE is **1e-7**, with overrides of **5e-4** for `mul_mat`, `mul_mat_id` and `out_prod`, **1e-6** for `soft_max` and `cpy`, and **0** for `argmax`;
  - a NaN anywhere fails; Inf must match in sign.
  - **We adopt its structure and keep NMSE as a secondary signal,** but replace global NMSE with per-element error bounds (§3.2), because a global NMSE can hide one wrong tile among a million correct outputs.

---

## §2 Architecture: seven layers of evidence

| Layer | Question | Technique | Where it runs | Output |
|---|---|---|---|---|
| **L0 Algorithm** | Is the maths right? | Lean proofs of the *specification*: the Q4_K block formula, the online-softmax rescaling identity (flash attention equals softmax), the RoPE rotation, and that RMSNorm is scale-invariant | CI (x86, gx10 cache) | `pv` contract + Lean theorem |
| **L1 Indexing** | Can it read or write out of bounds? Can two threads write the same address? | index maps written as pure Rust `const fn`s shared by the kernel generator and the verifier; **Kani** bounded proofs; Lean for the affine case (K3) | CPU, any host | proof receipt |
| **L2 Numerics** | Is every output within its proven error bound? | a differential test against the f64 oracle with the **per-element γ bound** (§3.2) | every device | kernel receipt (Pass/Fail + margin) |
| **L3 Shape classes** | Did we exercise every control path? | shape-class partition per tiling (K2): tails, alignment, device limits, 2³¹ overflow | every device | class coverage 100% |
| **L4 Properties** | Does it obey laws that need no oracle? | metamorphic tests: linearity, batch-permutation invariance, softmax shift invariance, causal masking, RoPE norm preservation | every device | property receipt |
| **L5 Runtime sanitizers** | Races, uninitialized reads, barrier divergence? | CUDA `compute-sanitizer` (memcheck, racecheck, initcheck, synccheck); Vulkan validation layers; Metal API validation; ASan/UBSan and Miri (scalar CPU) | nightly, small shapes | sanitizer receipt |
| **L6 Composition** | Do the parts add up to a correct model? | SHACL derivation (§5) + **one smoke test per model × device** on the release sha: top-1/top-k agreement and logit cosine vs the oracle / pinned llama.cpp | per release | derived cell + smoke receipt |

**Why all seven:**
- L2 alone misses paths it never runs (so L3 exists).
- L2 needs an oracle, and L4 doesn't.
- L2 and L3 cover numbers and paths, but races can be nondeterministic (so L1 and L5).
- Per-kernel bounds don't compose into model error, because layers can amplify error (so L6).

---

## §3 Numerics: tolerances from error analysis

### 3.1 Error models (registry field `error_model`)

| Id | Kernel family | Bound per output element | Source |
|---|---|---|---|
| `EM-DOT` | dot product, GEMV, GEMM (any summation order) | \|ŷᵢ − yᵢ\| ≤ **γ_K** · (\|A\|·\|B\|)ᵢ + η_FTZ | Higham, *Accuracy and Stability*, Thm 3.1 / §3.1; γ_K = K·u/(1 − K·u), where u is the **accumulator's** rounding unit |
| `EM-DEQ` | dequantization (Q4_K, Q6_K, Q8_0, …) | **bit-exact** (0 ULP) against pinned llama.cpp fixtures | PP-QUANT D5 |
| `EM-ELEM` | element-wise (add, mul, SiLU, GELU, exp-based) | ≤ the **backend's published builtin bound** composed through the op (WGSL f32: exp **3 + 2·\|x\| ULP**, log 2⁻²¹ abs / 3 ULP, `/` 2.5 ULP, sqrt inherited from 1/inverseSqrt), + 1 ULP for the final rounding. Bounds that depend on \|x\| are evaluated per element. | WGSL spec (`[V]`, table above); CUDA/Metal math-library docs `[U]` per function |
| `EM-RED` | reductions (sum, max, RMSNorm) | EM-DOT with K = the reduction length (max: exact) | Higham |
| `EM-SMX` | softmax / online softmax | max-subtraction makes it stable; bound = EM-ELEM(exp) + EM-RED(sum), relative per row | L0 proof that the online rescaling identity is exact over ℝ |
| `EM-ATT` | attention (QKᵀ, softmax, ·V; flash variants) | the composition of EM-DOT, EM-SMX and EM-DOT, per row | derived; checked against L0 |
| `EM-ROPE` | rotary embedding | EM-ELEM(sin, cos) + 2 ULP; plus the property \|rot(x)\| = \|x\| within the bound | L4 |
| `EM-NONDET` | atomics-based reductions | the same bound, but **not bit-deterministic**. Declared in the registry, and never used where determinism is claimed | — |

### 3.2 The per-element test (replaces global NMSE as the gate)

For each output element *i*, compute the **margin** m_i = |ŷ_i − y_i| / B_i, where B_i is the bound from the kernel's error model, evaluated in f64 from |A| and |B|.

- **Pass iff max_i m_i ≤ 1**, and there are no NaNs, and the Inf set and signs match.
- The receipt records max m_i, the p99.9 of m_i and the NMSE (llama.cpp-comparable), per shape class.
- **The margin trend is a quality metric.** A kernel that drifts from m = 0.02 to m = 0.6 across releases is flagged amber before it fails.

**Theorem K1a (the bound doesn't depend on summation order).** The recursive-summation bound |ê| ≤ γ_{n−1} Σ|x_i| holds for **every** summation order and bracketing, because each term takes part in at most n − 1 roundings. A fused multiply-add only removes roundings.
- So one tolerance is valid across CPU SIMD lanes, CUDA warp trees, WGSL reassociation (which the spec explicitly permits) and Metal. **One spec, every backend.** ∎ (Higham Lemma 3.1 and the proof of Thm 3.1.)

**Theorem K1b (when the bound is vacuous; [Lean] `f16_acc_k4096_vacuous`, `bf16_acc_k256_vacuous`, `f32_needed_for_4096`).** γ_K is finite only when K·u < 1, i.e. K < 2^p for a p-bit significand.
- f16 accumulation (p = 11) gives **no guarantee at K = 4096**.
- bf16 accumulation (p = 8) gives none at **K = 256**.
- f32 (p = 24) is meaningful up to K = 2²⁰ (γ ≤ 1/15).
- **So:** every GEMM/GEMV/attention kernel with K ≥ 2¹¹ **must accumulate in f32**, whatever its storage type. Tensor-core paths (fp16/bf16 inputs) must declare an f32 accumulator. TF32 mode must be explicit and carried as its own `dtype_acc`. ∎

**Flush-to-zero term η_FTZ.** WGSL, and Metal/CUDA fast-math modes, may flush subnormals to zero. Each flushed operation adds absolute error below the smallest normal number (f32: 2⁻¹²⁶, negligible; **f16: 2⁻¹⁴ ≈ 6.1e-5, not negligible**). So η_FTZ = K · min_normal(dtype_acc) is included in B_i, and f16 storage paths get an explicit "subnormal-heavy" shape class.
- **Worked f16 example ([Lean] `f16_ftz_drift_quarter`):** at K = 4096, η_FTZ = 4096 · 2⁻¹⁴ = **0.25 absolute**. That is one more reason, besides K1b, that f16 accumulation is refused for long reductions.

### 3.3 Input generation
- Uniform and normal inputs, plus **adversarial** ones:
  - large dynamic range (1e-4 … 1e4);
  - cancellation-heavy (x, −x pairs);
  - all-equal rows (softmax);
  - near-overflow logits (softmax max-subtraction);
  - subnormals (FTZ);
  - NaN and Inf propagation.
- Quantized blocks come from real model tensors (pinned sha) **and** synthetic extremes: absmax at the scale limit, all-zero blocks, minimum scale.
- Seeds are recorded, so every failure is reproducible from its receipt.

---

## §4 Coverage: shape classes and paths

**Theorem K2 ([Lean] `classes_covered`, `zero_class`, `classes4_distinct`).** A loop over d elements in tiles of T runs ⌊d/T⌋ full tiles and a tail of d mod T. Its control flow depends only on the pair (⌊d/T⌋ = 0?, d mod T = 0?).
- The representatives {T−1, T, T+1} (plus 2T for multi-tile) cover every reachable class.
- (true, true) happens only for d = 0.
- Nested tilings and vector widths V give the product of classes per dimension. ∎
- **v1.1 refinement (`cls4`):** the model is (min(⌊d/T⌋, 2), d mod T = 0?), which separates *one* full tile from *two or more*. Accumulator-carry and unroll bugs only show up with ≥ 2 tiles. `classes4_distinct` proves that {T−1, T, T+1, 2T} land in **four different** classes, so 2T is a required representative, not a duplicate of T.

**Standard class set per dimension**, from the registry's `tile` and `vec` fields:

| Class | Values | Catches |
|---|---|---|
| empty | 0 | empty launches, division by 0 |
| sub-vector | 1, V−1 | scalar-tail loops |
| vector edge | V, V+1 | SIMD remainder masks |
| tile edge | T−1, T, T+1, 2T | tile tails, off-by-one (the #749 bug class) |
| misaligned | offset/stride not a multiple of the alignment | alignment assumptions (NEON/AVX loads, 16-byte WGSL) |
| device limit | workgroup/grid maxima (for example CUDA grid y ≤ 65,535; `maxComputeWorkgroupStorageSize`; `maxStorageBufferBindingSize`) | silent clamping, split-dispatch bugs |
| 2³¹ | element index ≥ 2³¹ (large KV caches) | i32 index overflow |
| production | the real model shapes (hidden, head_dim, vocab, GQA ratios) | the shapes users actually hit |

**Subgroup-size classes (GPU):** a kernel that uses subgroup operations is tested at **every subgroup size the device reports**. NVIDIA reports 32; AMD RDNA reports 32 or 64 (wave32/wave64); Apple reports 32; Intel iGPUs report 8, 16 or 32 `[C]`, from general knowledge, and KTEST-00 records each device's `subgroup_min_size`/`max_size`.
- **A hard-coded 32 is the classic cross-vendor bug**, and it's falsifier F-3.

**Dispatch-path classes (CPU):** every ISA path the binary contains (scalar, AVX2, AVX-512/VNNI, NEON, dotprod/i8mm, SVE) is **forced** on every host that supports it, through a dispatch override (`APR_FORCE_ISA=…`). So `intel` tests the scalar, AVX2 and AVX-512 paths, and `gx10` and `mini` test the NEON family.
- An ISA path no host in the fleet can run is **unregistered, so refused at dispatch**. Shipping untestable code paths is banned.

---

## §5 Ontology: registry, receipts and derivation (SHACL)

### 5.1 Classes (extends KREG-001 / the PP-QUANT TRAITS table #3430)

```
:Kernel      key, op, dtype_in, dtype_acc, dtype_out, quant, layout, tile, vec,
             backend ∈ {cpu-scalar, cpu-avx2, cpu-avx512, cpu-neon, cpu-sve, cuda, hip, wgpu-vulkan, wgpu-metal},
             requires (features: SHADER_F16, SUBGROUP, sm_XX, …), determinism ∈ {bitwise, bounded},
             error_model (EM-*), source_hash, codegen_hash
:Device      host, vendor, model, driver, toolchain (CUDA/ROCm/naga/Metal versions), subgroup sizes,
             limits (from wgpu Limits / cudaDeviceProp / hipDeviceProp), arch (sm_89, sm_121, gfx1101, apple-m4, …)
:ShapeClass  kernel, dim, class (§4)
:Receipt     kernel, device, shape_class, layer (L1–L5), verdict, max_margin, p999_margin, nmse,
             input_set_hash = H(source_hash, codegen_hash, toolchain, driver, device), seed, run_id
:Model       sha256, arch, quant; kernel_set (from the dispatch trace, APR-OBS OBS-09)
:SmokeReceipt model, device, apr_sha (= release sha), top1_agree, logit_cosine, verdict
```

### 5.2 Shapes (these replace #3715 v1's per-cell shape)

- **S-REG:** every dispatched kernel key in any trace is a registered `:Kernel` (`sh:minCount 1`). This is V5 and E5: 0 unregistered dispatches.
- **S-EM:** every `:Kernel` has exactly one `error_model` (`sh:minCount 1; sh:maxCount 1`).
- **S-COV:** for every (`:Kernel` k, `:Device` d) where d satisfies k.requires, and for every `:ShapeClass` of k, there is a `:Receipt` with verdict Pass for L2, L3 and L4, and an `input_set_hash` equal to the current one (`sh:minCount 1`). A missing, stale or failing receipt is a violation that names the cell.
- **S-DERIVE** (a SHACL-AF rule): `:Model` m is **ready** on `:Device` d iff
  1. every k ∈ m.kernel_set has S-COV satisfied on d, **and**
  2. a `:SmokeReceipt`(m, d, release sha) is Pass.
- **S-SAN:** every kernel carries an L5 sanitizer receipt no older than 7 days per backend family.

**Theorem K5 (conditional soundness of the derivation; proof sketch).** Suppose that:
- (i) the dispatch trace is complete: every kernel launch is logged with its key and shape (OBS-09, falsified by F-7);
- (ii) every launch's shape falls in a covered class (§4, K2);
- (iii) no aliasing between buffers, proven by L1 and checked by L5.

Then every kernel invocation in m's forward pass on d is within its bound B. The per-kernel bounds say nothing about how error compounds across layers, so condition 2 of S-DERIVE (the smoke test) is **necessary**, not optional. ∎ Stated honestly: K5 bounds *kernel* error, and the smoke test bounds *model* behaviour. Neither replaces the other.

### 5.3 Reuse (why the gate gets fast)

A kernel receipt is valid while its `input_set_hash` is unchanged. A release re-runs only:
- kernels whose source or codegen changed;
- devices whose driver or toolchain changed;
- the per-model smoke tests (which bind to the release sha).

**Theorem K4 ([Lean] `kernel_receipts_cheaper`, with `[A]` inputs).**
- **Kernel receipts:** 60 kernels × 8 device-backends × 10 shape classes × 0.5 s = **40 min** of device time, split across hosts.
- **Model cells:** 1,008 × 24 s (the measured 6.7 h) = **6.7 h**.
- So kernel receipts are ≈10× cheaper *before* reuse. With ≥ 80% reuse between rc and FINAL, the kernel part drops to ≤ 8 min, plus the smoke tests (8 models × 5 devices × ~30 s ≈ 20 min, in parallel across hosts). ∎

---

## §6 Per-backend rules

### 6.1 CPU SIMD (x86 AVX2/AVX-512/VNNI; ARM NEON/dotprod/i8mm/SVE)
- Every ISA path is forced and tested on every host that has it (§4). The scalar path is always present and always tested; it's also the L2 fallback.
- **L1:** Kani proves the scalar reference's index maps and the SIMD remainder masks for bounded sizes. Miri runs the scalar and portable paths (Miri doesn't cover most `core::arch` intrinsics `[C]`, so those are covered by L2, L3 and ASan).
- Denormal and FTZ behaviour on CPU: MXCSR FTZ/DAZ state is pinned and recorded in the receipt (a thread-pool worker that inherits FTZ is a known heisenbug class).

### 6.1.1 KTEST-04 amendment: one registry row per ISA path (2026-09-30, kreg)

**Finding.** Every CPU registry row today is `arch: any`, `isa_features: none`, and its intel receipt was taken in a native process. On a row whose source has a SIMD branch, the receipt therefore measured the AVX2/AVX-512 path while claiming the portable one. The scalar path was never receipted; this is STOP S-2 (a backend claimed without receipts).

| Row | Paths in the source (the features its dispatch checks) |
|---|---|
| `cpu.matvec.q4_0` | `avx512vnni+avx512bw`; `avx2`; scalar |
| `cpu.matvec.q8_0` | `avx2+fma`; scalar |
| `cpu.matvec.q4_k` | `avx512f+avx512vnni`; `avx2+fma`; scalar |
| `cpu.matvec.q6_k` | `avx2+fma`; scalar |
| `cpu.matvec.f16` / `bf16` | `avx2+fma+f16c` / `avx2+fma`; scalar |
| `cpu.rmsnorm.f32` | trueno backend (AVX2/AVX-512); scalar |
| `cpu.rope.*` | `avx512f`; `avx2+fma` (NEOX only); scalar |
| attention ops | `avx2+fma`; scalar |

Every other CPU row (q5_k, iq\*, q2_k/q3_k/q4_1/q5_0/q5_1, layernorm, gelu, argmax) is scalar only, so its existing receipt is already correct.

**Decision.** §4 and §5.1 name one kernel per ISA path (`cpu-scalar`, `cpu-avx2`, `cpu-avx512`, …), and the KREG registry already expresses that as `backend: cpu` + `arch` + `isa_features`. The registry's `admit_for` picks the most specific row the target satisfies, and `APR_FORCE_ISA` narrows `Target::host()`. So:
1. Each SIMD path gets its own row, `cpu.<op>.<qtype>.<isa>`, with `arch: x86_64` and `isa_features` equal to exactly the features its dispatch checks (the table above). Because the registry key includes the feature set, two paths cannot share a row.
2. The base `any`/`none` row becomes the scalar path. Its receipt is re-emitted under `APR_FORCE_ISA=scalar`.
3. The emitter refuses to write a receipt unless `admit_for(Target::host())` is the row being receipted, and it records `isa_forced` in the header. Together these stop a native run from being stamped on the scalar row.
4. The committed-receipt hold test checks each row in a child process forced to that row's ceiling, because the ceiling is fixed once per process.
5. **Every CPU row is receipted.** FALSIFY-KREG-008 refuses a `cpu` row whose tolerance is `unmeasured`, with no allowance count, so a per-ISA row lands only in the same commit as its receipt.
6. **F-11.** A fleet ISA table, verified from `/proc/cpuinfo`, lists what each host can run. intel (Xeon W-3245) has `fma sse4_1 f16c avx2 avx512f avx512bw avx512_vnni`, so it covers every x86 row. gx10 and mini are `[U]` until KTEST-00. A row whose `isa_features` no host in the table satisfies is refused.
7. The MXCSR FTZ/DAZ state goes into the receipt header next to `isa_forced` (§6.1).

The mechanism (`APR_FORCE_ISA`, every serve dispatch site gated, and trueno backend selection capped) is on branch `la-71/ktest-04-force-isa`. Steps 1–7 land after that branch's intel receipts.

### 6.2 NVIDIA CUDA (sm_89 lambda, sm_121 gx10, yoga `[U]`)
- **Native code for every arch, no PTX JIT:** the fatbin must contain SASS for every arch in the support matrix, checked with `cuobjdump --list-elf`. A separate L2 run with `CUDA_FORCE_PTX_JIT=1` proves the PTX path is also correct. Missing `sm_121` SASS is the "correct but silently slow" bug class (APR-QUALITY §6.1) → falsifier F-4.
- Tensor-core and TF32 modes are explicit registry dtypes (K1b). `compute-sanitizer` runs its four tools nightly on the small-shape classes (L5).
- Unified memory (GB10) is its own `:Device` capability. Kernels that assume discrete-GPU copies are tested there separately.

### 6.3 AMD: WGPU/Vulkan first, ROCm/HIP only where supported
- **Primary AMD path: WGPU → Vulkan** on intel's dual AMD GPUs and framework16's Radeon `[U]`. This is the 0.73 E1 cell `intel-wgpu`.
- **HIP is claimed only for GPUs on the ROCm compatibility matrix** (KTEST-00 records each card's gfx id against it). Integrated GPUs (780M/890M) aren't listed in ROCm 7.2 `[C]`, so for those it's WGPU only.
- **Wave64 vs wave32 subgroup classes** are mandatory for every AMD device that exposes both (§4).

### 6.4 WGPU (Vulkan / Metal / DX12)
- A kernel's `requires` lists wgpu features. On a device without a feature, the kernel isn't eligible: S-COV doesn't require it there, and dispatch picks the fallback kernel, which must itself be registered.
- **The same WGSL is tested separately per backend,** because naga translates to SPIR-V for Vulkan and to MSL for Metal: different compilers, different bugs.
- `EXPERIMENTAL_COOPERATIVE_MATRIX` (8×8 f32 only) `[C]` isn't used by release kernels until it leaves experimental status and has receipts on ≥ 2 vendors.
- Tolerances include WGSL's permitted slack: division 2.5 ULP, exp/log 1 ULP, FTZ, and reassociation, which K1a covers.

### 6.5 Apple Silicon (M4: Metal GPU + NEON/SME CPU)
- Metal is reached through WGPU (naga → MSL). **Fast math is never implicit:** the pipeline compile options are recorded in the receipt. Softmax, RMSNorm and exp-heavy kernels run the near-overflow and subnormal classes to catch fast-math reassociation of the max-subtraction → falsifier F-6.
- The 16 GB unified memory is recorded as a device limit. The 2³¹ and large-KV classes run on the largest size that fits, and the rest are **declared not applicable on this device**, never skipped silently.
- **The Neural Engine is out of scope:** there's no public API for programmable kernels, so it stays out of the claimed hardware.

---

## §7 Falsifiers (each row: planted defect → the gate must turn RED; anti-vacuity arm → it stays GREEN)

| Id | Planted defect | Must go RED | Anti-vacuity |
|---|---|---|---|
| F-1 | off-by-one in a tile tail (read d+1) | L3 class T+1 (L2 margin > 1) and L1 Kani OOB | correct tail → Pass |
| F-2 | fp16 accumulator in a K=4096 GEMM | L2 margin > 1 on the cancellation class; K1b refusal at registration | f32 accumulator → Pass |
| F-3 | subgroup size hard-coded to 32 | L2 RED on the wave64 (AMD) or size-16 (Intel) class | the device-reported size → Pass |
| F-4 | `sm_121` missing from the fatbin | the SASS check RED on gx10 | full arch list → Pass |
| F-5 | a barrier removed in a shared-memory reduction | L5 racecheck RED; L2 flaky across 20 repeats | barrier present → 20/20 identical |
| F-6 | softmax without max-subtraction under fast math | the near-overflow class gives Inf → RED | max-subtraction → Pass |
| F-7 | a kernel dispatched without being registered | S-REG violation naming the key | registered → no violation |
| F-8 | a kernel source changed, old receipt kept | S-COV stale `input_set_hash` → every model using it goes RED (derived) | re-run → Pass |
| F-9 | a model using an unvalidated kernel on a device | S-DERIVE RED for that (model, device) only | other models unaffected |
| F-10 | dequant changed by one rounding step | EM-DEQ bit-exact RED | pinned fixtures → Pass |
| F-11 | an ISA path compiled in that no host can run | registration refused (unregistered ISA) | path removed → Pass |
| F-12 | a global NMSE pass hiding one bad tile (1 wrong tile in 10⁶ outputs) | per-element margin RED **while NMSE passes** (shows why §3.2 exists) | — |

---

## §8 Rows (the cop mints one `pmat` ticket per row; the 0.71 L1 slot owns them)

| Row | Train | What | Done when | Est |
|---|---|---|---|---|
| **KTEST-00** | 0.71 | Genchi genbutsu: resolve every `[U]` in §1 from `forjar.yaml` + device queries (wgpu adapter info/limits/features, `nvidia-smi`, `rocminfo`, `lscpu`/`sysctl`) | a fleet device table with sources; ROCm eligibility per card | 0.5 d |
| **KTEST-01** | 0.71 | Registry fields (§5.1) in KREG-001: `error_model`, `requires`, `determinism`, `input_set_hash` | S-REG + S-EM green on main; F-7 armed | 1 d |
| **KTEST-02** | 0.71 | The f64 scalar oracle crate plus the per-element margin harness (§3.2); NMSE reported alongside | F-12 RED while NMSE passes | 1.5 d |
| **KTEST-03** | 0.71 | Shape-class generator from tile/vec/limits (§4) + adversarial inputs (§3.3) | F-1 and F-6 RED | 1 d |
| **KTEST-04** | 0.71 | CPU ISA forcing (`APR_FORCE_ISA`) + receipts on intel, gx10, mini | every compiled ISA path has receipts; F-11 armed | 1 d |
| **KTEST-05** | 0.71 | CUDA: SASS-per-arch check + PTX-JIT arm + compute-sanitizer nightly | F-4 and F-5 RED | 1 d |
| **KTEST-06** | 0.71 | L1 index-map `const fn`s + Kani harnesses; Lean lemmas (K3) promoted to `contracts/` | F-1 caught by Kani | 1.5 d |
| **KTEST-07** | 0.71 | SHACL S-COV, S-DERIVE, S-SAN + receipt reuse by `input_set_hash` | F-8 and F-9 armed; derived cells match the #3715 v1 results on the same sha | 1.5 d |
| **KTEST-08** | 0.71 | **#3715 v2 cutover:** the release gate uses S-DERIVE + smoke tests | gate ≤ 30 min on a real rc; 0 unvalidated cells | 0.5 d |
| **KTEST-09** | 0.73 | WGPU-Vulkan (AMD) and WGPU-Metal (M4) receipts, including subgroup classes | F-3 RED; E1/E5 kernel coverage 100% | 2 d |
| **KTEST-10** | 0.73 | HIP receipts on ROCm-supported cards only (if KTEST-00 finds any) | a receipt per supported gfx id, or a recorded "none eligible" | 1 d |
| **KTEST-11** | 0.73 | L0 Lean specs: online-softmax identity, RoPE norm, Q4_K block formula | theorems in `contracts/` with `#print axioms` clean | 2 d |
| **KTEST-12** | 0.73+ | Margin-trend dashboard (APR-OBS) + amber rule | drift from m to ≥ 3·m flagged before it fails | 0.5 d |

---

## §9 Metrics and targets (Toyota: measure, then ratchet)

| Metric | Now | Target | By |
|---|---|---|---|
| Release kernel-gate time | 6.7 h (#3715 v1, cells) | **≤ 30 min** | 0.71 (KTEST-08) |
| Receipt reuse between rc and FINAL | 0% (sha-bound) | **≥ 80%** | 0.71 |
| Unregistered kernel dispatches | `[U]` | **0** | 0.71 (V5) |
| Kernels without an error model | `[U]` | **0** | 0.71 |
| Shape-class coverage per (kernel, device) | `[U]` | **100%** | 0.71 CPU/CUDA · 0.73 WGPU/HIP |
| Compiled ISA/arch paths without receipts | `[U]` | **0** | 0.71 |
| Backends claimed without receipts | `[U]` | **0** | always |
| Worst-case margin max mᵢ (release kernels) | `[U]` | ≤ 0.5 (headroom), amber > 0.5 | 0.72 |
| Falsifiers armed (§7) | 0/12 | **12/12** | 0.73 |

---

## §10 STOP conditions (stop that row, report, don't work around)

- **S-1** A tolerance would be set without an `error_model` citation.
- **S-2** A backend would be claimed from a feature flag or a name, without receipts.
- **S-3** An unsupported-hardware override (`HSA_OVERRIDE_GFX_VERSION`, forcing a feature the adapter doesn't report) would be used in CI.
- **S-4** A receipt would be reused across a changed `input_set_hash`.
- **S-5** The model smoke test would be dropped because kernels are green (it contradicts K5).
- **S-6** A red kernel cell would be waived. Gates are never waived.

---

## Appendix A: Lean theorems (`KernelTesting.lean`, Lean 4.22.0 core, 0 `sorry`)

| Theorem | Backs | Axioms |
|---|---|---|
| `store_index_injective` | K3: stride-slice stores are race-free | propext, Quot.sound |
| `store_index_in_bounds` | K3: those stores stay in bounds | propext, Quot.sound |
| `f16_acc_k4096_vacuous` | K1b: f16 accumulation has no bound at K = 4096 | none |
| `bf16_acc_k256_vacuous` | K1b: bf16 accumulation has none at K = 256 | none |
| `f32_acc_k2p20_ok`, `f32_needed_for_4096` | K1b: f32 is required, and sufficient up to 2²⁰ | none |
| `classes_covered` | K2: {T−1, T, T+1} cover the tiled-loop classes | propext, Quot.sound |
| `zero_class` | K2: the (true, true) class is only d = 0 | propext, Quot.sound |
| `kernel_receipts_cheaper` | K4: ≈10× cost ratio (`[A]` inputs) | none |
| `classes4_distinct` | K2 v1.1: {T−1, T, T+1, 2T} → 4 distinct classes | propext, Quot.sound |
| `f16_ftz_drift_quarter` | §3.2: f16 FTZ drift at K = 4096 is 0.25 | none |

**Not in Lean, and why:**
- **K1a** (order-independence of the γ bound) needs real analysis (Mathlib); it's cited from Higham, with the proof idea in §3.2.
- **K5** is conditional on the trace, coverage and aliasing assumptions; a proof sketch is in §5.2.
- KTEST-11 moves the L0 specs to Mathlib on gx10.

## Sources
- [llama.cpp `test-backend-ops.cpp`](https://huggingface.co/spaces/Steven10429/apply_lora_and_quantize/blob/main/llama.cpp/tests/test-backend-ops.cpp): NMSE definition and thresholds (1e-7 default; 5e-4 mul_mat; 1e-6 soft_max; NaN/Inf rules)
- [llama.cpp testing infrastructure (DeepWiki)](https://deepwiki.com/ggml-org/llama.cpp/9.2-testing-infrastructure)
- [wgpu `Features` (docs.rs, wgpu 30.0.1)](https://docs.rs/wgpu/latest/wgpu/struct.Features.html): SHADER_F16, SUBGROUP, INT64, cooperative matrix, float atomics
- [W3C WGSL specification](https://www.w3.org/TR/WGSL/): floating-point accuracy, FTZ, reassociation
- [Phoronix: ROCm 7.2 released](https://www.phoronix.com/news/AMD-ROCm-7.2-Released) · [ROCm compatibility matrix](https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html) · [TheRock SUPPORTED_GPUS](https://github.com/ROCm/TheRock/blob/main/SUPPORTED_GPUS.md)
- [GPUVerify (Betts et al., OOPSLA 2012)](https://nchong.github.io/papers/oopsla12.pdf): race and barrier-divergence verification (the model for L1/L5)
- [Castaldo, Whaley & Chronopoulos, "Reducing floating point error in dot product" (SIAM SISC)](https://epubs.siam.org/doi/10.1137/070679946): blocked summation bounds
- N. J. Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., SIAM 2002: Lemma 3.1, Thm 3.1 (γ_n)
- Internal: APR-QUALITY-001 §6 (fleet), PP-QUANT-001 D1–D5, APR-LOOKAHEAD-001 §2a (V5, E1–E6), FLOW-002 §0, FLOW-003 Def 14, #3715

## §12 Review log
- **v1.1 (peer review, another agent), checked against the compiler and the spec source:**
  - **Accepted:**
    - WGSL builtin accuracies were wrong in v1.0. Corrected from `gpuweb/wgsl/index.bs`: exp 3 + 2·|x| ULP, log 2⁻²¹ abs / 3 ULP, sqrt inherited (not correctly rounded). *Lesson: v1.0 took these from a summarizing fetch of a truncated page, so the primary source is now cited.*
    - The shape-class model now separates 1 full tile from ≥ 2 (`cls4`, `classes4_distinct`).
    - The f16 FTZ drift of 0.25 is stated explicitly (`f16_ftz_drift_quarter`).
    - Rewrites use explicit arguments.
    - gx10 SVE2 (Neoverse V2) and framework16 AVX-512 (Zen 4/5) are marked `[C]`, pending device queries.
  - **Rejected, with compiler evidence:**
    - "`obtain`/`rcases` aren't in core Lean": they are, and the file compiles on Lean 4.22.0 core on x86 and aarch64.
    - "Replace the `zero_class` proof with `omega`": **omega fails** on it, because it can't handle division by a variable. The reviewer's own run never got Lean installed (a DNS failure), so neither claim was ever compiled.
    - "`simp` leaves `d / T = 0`, so `rcases` fails": simp actually rewrites it to `T = 0 ∨ d < T`, so the case split is valid.
  - **Kept as `[U]`:** M4 SME availability is decided by `sysctl`, not by either reviewer.
