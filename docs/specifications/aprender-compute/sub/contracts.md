# Sub-spec: Provable-Contract-First Design

**Parent:** [trueno-spec.md](../trueno-spec.md) Sections 2, 17

---

## 1. Contract-First Principle

No kernel ships without a provable contract. The contract is the specification; the Rust code is the implementation. This is non-negotiable.

**Workflow (inverse of typical development):**

1. **Write contract YAML** in `contracts/` — equations, domain, codomain, invariants
2. **Define FALSIFY tests** — what would disprove correctness
3. **Define proof obligations** — backend equivalence, numerical stability, performance bounds
4. **Register binding** in `../provable-contracts/contracts/trueno/binding.yaml`
5. **Generate scaffold** — `pv generate contracts/my-kernel-v1.yaml`
6. **Implement** — fill scaffold with real logic
7. **Prove** — FALSIFY tests + Kani harnesses + `pv lint`

If step 7 fails, the contract is wrong or the implementation is wrong — either way, you know.

## 2. Escape-Proof Six-Stage Pipeline

**Principle:** It must be *impossible* to ship code that violates a contract. Not "difficult." Not "caught in CI." Impossible — the compiler refuses to produce a binary.

Each stage gates the next. Skip one → compile error.

```
A. Equation (YAML)           — mathematical ground truth
   ↓ must exist
B. Lean 4 Proof              — theorem with no sorry
   ↓ must discharge all obligations
C. YAML Contract Validation  — pv lint Gates 1-7 pass
   ↓ schema valid, internally consistent
D. build.rs Codegen          — generates debug_assert!() from pre/postconditions
   ↓ sets CONTRACT_* env vars from binding.yaml
E. #[contract] Proc Macro    — checks CONTRACT_* env var, inserts assertions
   ↓ missing binding = compile_error!
F. Test Execution            — cargo test runs falsification tests
   ↓ failure blocks merge
```

### Escape Analysis

| Attempted escape | What catches it |
|-----------------|-----------------|
| Delete contract YAML | build.rs: CONTRACT_* env var missing → compile_error! |
| Remove lean_theorem | build.rs: theorem field required → compile_error! |
| Leave `sorry` in Lean | build.rs: sorry detection → compile_error! |
| Remove `#[contract]` | pv lint Gate 4: test refs don't match → CI blocks |
| Remove test function | pv lint Gate 4: test missing → CI blocks |
| Skip pv lint | CI workflow requires pv lint pass |
| Ship without tests | CB-1201: PV Lint FAIL → pmat comply NON-COMPLIANT |
| Weaken postcondition | Lean proof no longer matches → sorry required → compile_error! |

## 3. Binding Registry (`binding.yaml`)

Every contract equation maps to a Rust function via `../provable-contracts/contracts/trueno/binding.yaml`.

```yaml
version: 1.0.0
target_crate: trueno
bindings:
  - contract: softmax-kernel-v1.yaml
    equation: softmax
    module_path: trueno
    function: softmax
    status: implemented           # implemented | pending | not_implemented
    notes: Auto-inferred by pv infer

  - contract: gemm-backward-tiled-v1.yaml
    equation: backward_a_gemm
    module_path: trueno_gpu::kernels::backward::gemm
    function: GemmBackwardAKernel::tiled
    signature: 'fn tiled(tile_size: u32) -> Self'
    status: implemented
    notes: grad_A = grad_C @ B^T; tiled with shared memory

critical_path:
  - softmax
  - matmul
  - silu
  - gelu
  - execute_matmul
  - PipelineCache
  - GemmBackward
  - arithmetic_intensity
```

### Status Values

| Status | Meaning | Build effect |
|--------|---------|-------------|
| `implemented` | Function exists and conforms | Build passes |
| `pending` | Planned but not yet wired | Build passes (treated as gap warning) |
| `not_implemented` | Required but missing | **BUILD FAILS** (AllImplemented policy) |

### Current Binding Stats

38 total bindings, 17 implemented, 21 pending. See Section 9 for the gap closure plan.

## 4. build.rs AllImplemented Policy

`build.rs` reads binding.yaml, sets `CONTRACT_*` env vars, and enforces the AllImplemented policy:

```
binding.yaml → parse → deduplicate → set CONTRACT_* env vars
                                    → count implemented/pending/gaps
                                    → if any not_implemented: panic!()
```

**Env var naming:** `CONTRACT_{CONTRACT_STEM}_{EQUATION}={status}`

Example: `CONTRACT_SOFTMAX_KERNEL_V1_SOFTMAX=implemented`

These env vars are consumed at compile time by the `#[contract]` proc macro. If the env var is missing or has wrong status, the macro emits `compile_error!`.

**Graceful degradation:** When binding.yaml is not found (CI, crates.io builds), build.rs sets `CONTRACT_BINDING_SOURCE=none` and continues without enforcement.

## 5. `#[contract]` Proc Macro

The proc macro on implementing functions does three things:

1. **Verifies** `CONTRACT_*` env var exists at compile time
2. **Inserts** precondition assertions at function entry
3. **Inserts** postcondition assertions before return

```rust
#[contract("softmax-kernel-v1", equation = "softmax")]
pub fn softmax_1d(logits: &[f32]) -> Vec<f32> {
    // Macro verifies CONTRACT_SOFTMAX_KERNEL_V1_SOFTMAX=implemented
    // Macro inserts: check_softmax_pre(logits) at entry
    // Macro inserts: check_softmax_post(&ret, logits) before return
    // ...
}
```

Assertions use `debug_assert!()` — **zero cost in release builds**.

## 6. Contract-Trait Enforcement (Compiler-Verified)

build.rs binding checks verify the *name* exists. Traits verify the function *actually exists with the correct signature*.

```
YAML Contract           →    Generated Trait           ←    Consumer Impl
softmax-kernel-v1.yaml  →    SoftmaxKernelV1 trait     ←    impl SoftmaxKernelV1
  equations:                   fn softmax(...)               for Trueno { ... }
    softmax:                   fn log_softmax(...)
    log_softmax:
```

If the YAML changes, every consumer gets a **compile error** until they update. No build.rs scanning. No name matching. The Rust compiler does it.

Generate traits: `pv scaffold --trait contracts/softmax-kernel-v1.yaml`

## 7. `pv lint` Quality Gates

`pv lint` is the single-command quality gate. 7 gates, any failure exits non-zero:

| Gate | What it checks |
|------|---------------|
| Gate 1 (validate) | YAML parses, schema valid |
| Gate 2 (audit) | Internal consistency: obligations match harnesses |
| Gate 3 (score) | Completeness metrics above threshold |
| Gate 4 (verify) | All `test:` references resolve to `fn test_*` in source |
| Gate 5 (enforce) | Equations have preconditions/postconditions |
| Gate 6 (enforcement-level) | Contracts meet minimum enforcement level |
| Gate 7 (reverse-coverage) | Binding registry coverage (with `--binding`) |

```bash
pv lint contracts/                                  # run all gates
pv lint contracts/ --min-score 0.60                 # set score threshold
pv lint contracts/ --binding binding.yaml           # check binding coverage
pv lint contracts/ -f sarif                         # SARIF output for IDE
pv lint contracts/ --diff main                      # only lint changed contracts
pv lint contracts/ --suggest                        # auto-fix suggestions
```

## 8. Verification Ladder

Trueno's verification depth across levels:

| Level | Method | Tool | Trueno status |
|-------|--------|------|--------------|
| L5 | Theorem proving | Lean 4 | 53 theorems, 22 domains, 0 sorry, all eqs covered |
| L4 | Bounded model checking | Kani | YAML-defined harnesses, not yet in CI |
| L3 | Property-based testing | proptest | Active: commutativity, associativity |
| L2 | Falsification tests | `#[test]` | Active: FALSIFY tests in all contracts |
| L1 | Type system | rustc + traits | Active: trait enforcement, AllImplemented |
| L0 | Code review | Pre-commit hooks | Active: pv lint, pmat comply |

**L0 through L2 enforce on every `cargo build` + `cargo test`.** L3 enforces on debug builds. L4/L5 defined in YAML but not yet in CI.

### L5 Coverage Detail

53 Lean 4 theorems across 22 domains, 0 sorry, `lake build` passes (4731 jobs).

**All 37 contract equations have matching Lean theorems.** Coverage by domain:

| Domain | Theorems | Key properties |
|--------|----------|----------------|
| Softmax | 5 | partition-of-unity, non-neg, bounded, shift-invariance, monotonicity |
| Image | 9 | rgb coeff sum, hsv achromatic, histogram mass, conv2d size, canny thresholds, sobel non-neg, morphology anti-extensive, resize lerp bounded, components bounded |
| FFT | 5 | Parseval energy, Bluestein chirp-z, 2D/3D separability, batch independence |
| Sparse | 5 | SpMV linearity, BSR linearity, SELL additivity, SpGEMM assoc, SpMM distrib |
| Elementwise | 3 | add commutativity, mul_scalar associativity, ReLU non-negativity |
| Sigmoid | 3 | bounded, symmetry, SiLU zero |
| BLAS | 3 | SYRK symmetry, TRMM associativity, TRSM identity |
| MatMul | 2 | associativity, identity |
| LayerNorm | 2 | denominator positive, shift invariance |
| RMSNorm | 2 | denominator positive, scale invariance |
| CrossEntropy | 2 | log-softmax bound, non-negativity |
| Rand | 2 | Philox determinism, ThreeFry determinism |
| + 10 more | 10 | Cholesky SPD, LU existence, QR orthogonality, SVD non-neg, Quantization roundtrip, Transpose involution, GEMV correctness, AdamW weight decay, Einsum assoc+identity, GPU dim-independence |

## 9. KAIZEN Workflow (Performance Contracts)

KAIZEN tickets follow the contract-first pattern:

```
1. Create KAIZEN ticket          → pmat work start KAIZEN-NNN
2. Write contract YAML           → performance obligation + threshold
3. Implement in trueno           → commit references KAIZEN-NNN
4. Measure improvement           → baseline vs result in commit message
5. Contract serves as gate       → future regressions trigger alert
```

| Ticket | Contract | Result |
|--------|----------|--------|
| KAIZEN-040 | Matrix::transpose BLIS AVX2 | BLIS delegation |
| KAIZEN-041 | vecmat BLIS gemv | Eliminated 3x temp allocs |
| KAIZEN-049 | GPU L2 norm reduction | GPU-side norm |
| KAIZEN-052 | In-place fused xent | Eliminated 77.8MB/step |

## 10. Contract Schema Reference

```yaml
kernel: kernel-name
version: 1
status: draft | implemented | verified

equations:
  equation_name:
    formula: "y_i = f(x_i) for all i in [0, n)"
    domain: "R^n"
    codomain: "R^n"
    preconditions:
      - "!input.is_empty()"
      - "input.iter().all(|x| x.is_finite())"
    postconditions:
      - "ret.len() == input.len()"
      - "(ret.iter().sum::<f32>() - 1.0).abs() < 1e-6"
    invariants:
      - "numerical_stability: |y_simd - y_scalar| < 1e-5"
    lean_theorem: "ProvableContracts.Theorems.Softmax.PartitionOfUnity"

proof_obligations:
  - type: invariant | equivalence | bound | precondition | postcondition
    property: "backend_equivalence"
    tolerance: 1e-5
    backends: [scalar, sse2, avx2, avx512, neon, wasm, cuda, wgpu]

falsification_tests:
  - id: FALSIFY-XX-001
    property: "SIMD output matches scalar reference"
    assertion: "max_abs_error < 1e-5"
    prediction: PASS
    test_sizes: [1, 7, 8, 15, 16, 1000, 100000]

kani_harnesses:
  - name: verify_no_overflow
    bounded: true
    max_input_size: 16

performance:
  baseline: scalar
  measured: { avx2: "4.3x", cuda: "50x (1M elements)" }
  regression_threshold: "5%"

qa_gate:
  - "coverage >= 90%"
  - "mutation_kill_rate >= 80%"
  - "all FALSIFY tests PASS"
```

## 11. Current Contract Inventory (28 local + 38 bindings)

**Local contracts** (`contracts/*.yaml`): 28 files covering core kernels, BLAS, FFT, image, sparse, solvers, random, GPU.

**Binding registry** (`../provable-contracts/contracts/trueno/binding.yaml`): 38 entries mapping equations to trueno functions.

| Domain | Local contracts | Binding entries |
|--------|----------------|-----------------|
| Activations | elementwise | relu, sigmoid, gelu, silu |
| Softmax/CE | softmax | softmax, log_softmax |
| Matmul | gemv, blas-level3, trsm | matmul, matmul_dispatch |
| GPU backward | dimension-independent | backward_a/b_gemm, arithmetic_intensity |
| GPU infra | — | pipeline_cache, pcie_overhead, culink_skip |
| Profiling | — | wall_coverage, sync_verification, brick_ordering |
| Roofline | — | bandwidth_ceiling, throughput_bound |
| Misc | — | embedding_lookup, tensor_count, precision, recall |

## 12. CLI Quick Reference

```bash
# Contract authoring
pv generate contracts/my-kernel-v1.yaml      # scaffold from contract
pv scaffold --trait contracts/my.yaml        # generate trait file

# Validation
pv lint contracts/                           # 4-gate quality check
pv lint contracts/ -f sarif                  # SARIF for IDE
pv verify-bindings                           # check binding registry

# Analysis
pv score                                     # PVScore metric
pv status                                    # contract summary
pv audit                                     # traceability audit
pv coverage                                  # cross-contract obligation coverage

# Formal methods
cargo kani                                   # bounded model checking
pv lean-status                               # Lean proof status

# KAIZEN
pv kaizen                                    # KAIZEN ticket management
pmat work list                               # all work items
```
