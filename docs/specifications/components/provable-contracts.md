# Provable Contracts — Design Foundation

Status: proposed
Date: 2026-04-09

**Version**: 1.0.0
**Status**: Active
**Parent**: [aprender-spec.md](../aprender-spec.md) §2
**Crate**: `provable-contracts` (lib + CLI `pv`)

---

## 1. Overview

Provable contracts is the foundational design methodology for the entire
sovereign Rust AI stack. Every kernel, tensor operation, and backend dispatch
begins as a YAML contract — a mathematical specification with equations,
proof obligations, and falsification tests. Code is generated from contracts,
not the reverse. The compiler refuses to build code without a valid contract.

**Tagline**: Papers → Math → Contracts → Code → Proofs.

### Why Contracts First

Traditional ML kernel development:
```
Paper → Developer's intuition → Code → Tests → Ship
```

Contract-first development:
```
Paper (arXiv)
  → Equations (canonical math in YAML)
    → Contract (proof obligations + falsification tests)
      → Scaffolded traits (auto-generated)
        → Implementation (scalar, SIMD, PTX, WGSL)
          → Kani bounded model checking
            → Lean 4 unbounded proof
```

Every link is a versioned artifact. The chain is auditable from paper to
proof. Deleting any link breaks the build.

---

## 2. Contract Specification Format (YAML)

### 2.1 Required Sections

```yaml
metadata:
  version: "1.0.0"
  created: "2026-03-31"
  author: "PAIML Engineering"
  description: "Softmax kernel with numerical stability"
  references:
    - "Bridle (1990). Training Stochastic Model Recognition."
  enforcement_level: "standard"   # basic | standard | strict | proven
  locked_level: "standard"        # Floor — cannot weaken without unlock

equations:
  softmax:
    formula: "σ(x)_i = exp(x_i - max(x)) / Σ_j exp(x_j - max(x))"
    domain: "x ∈ ℝ^n, n ≥ 1"
    codomain: "σ(x) ∈ (0,1)^n"
    invariants:
      - "Σ σ(x)_i = 1.0 (normalization)"
      - "σ(x)_i > 0 for all i (positivity)"
    preconditions: ["x.len() > 0", "x.iter().all(|v| v.is_finite())"]
    postconditions: ["result.len() == x.len()"]

proof_obligations:
  - type: invariant
    property: "Output sums to 1"
    formal: "|Σ σ(x)_i - 1.0| < ε"
    tolerance: 1.0e-06
  - type: equivalence
    property: "SIMD matches scalar"
    formal: "max_ulp_error(simd(x), scalar(x)) <= 2"

falsification_tests:
  - id: "FALSIFY-SM-001"
    rule: "Normalization"
    prediction: "sum(softmax(x)) ≈ 1.0 for random x"
    test: "proptest with 10000 vectors, dim 1..128"
    if_fails: "Missing max-subtraction trick"

kani_harnesses:
  - id: "KANI-SM-001"
    obligation: "SM-INV-001"
    property: "Softmax sums to 1.0 for small vectors"
    bound: 8
    strategy: "stub_float"
    harness: "verify_softmax_normalization"
```

### 2.2 Proof Obligation Types (26 Total)

**Property types (19)**:
invariant, equivalence, bound, monotonicity, idempotency, linearity,
symmetry, associativity, conservation, ordering, completeness, soundness,
involution, determinism, roundtrip, state_machine, classification,
independence, termination

**Eiffel Design-by-Contract types (7)**:
precondition, postcondition, frame, loop_invariant, loop_variant,
old_state, subcontract

### 2.3 Enforcement Levels

| Level | Requirements |
|-------|-------------|
| `basic` | Valid YAML schema |
| `standard` | + falsification tests + Kani harnesses |
| `strict` | + all bindings implemented across consumer crates |
| `proven` | + Lean 4 theorems with 0 `sorry` |

Enforcement level is locked — once set, it cannot be weakened without an
explicit unlock. This prevents regression from `proven` to `basic`.

---

## 3. The Verification Ladder

Five levels, each strictly stronger than the last:

| Level | Guarantee | Mechanism | Cost |
|-------|-----------|-----------|------|
| **L1** | True by construction | Rust type system (newtypes, const generics) | Zero runtime |
| **L2** | True for edge cases | Falsification tests (Popperian) | Milliseconds |
| **L3** | True for ~10K random inputs | Property-based tests (proptest) | Seconds |
| **L4** | True for ALL inputs ≤ N | Kani bounded model checking (SAT/SMT) | Minutes |
| **L5** | True for ALL inputs | Lean 4 theorem proving | Hours (one-time) |

### Stack Coverage

| Metric | Count |
|--------|-------|
| YAML contracts | 171 |
| Binding registries | 26 repos |
| Real bindings | 660 (260 with module_path) |
| `#[contract]` annotated functions | 38 |
| Kani harnesses | 985 |
| Lean 4 theorems proved | 64 (0 sorry) |

---

## 4. Escape-Proof Enforcement

Six stages make it **impossible** to ship code violating its contract:

### Stage 1: YAML Contract Exists
No contract → no scaffolding → nothing to implement.

### Stage 2: Lean 4 Proof (for `proven` level)
Lean's kernel checks the proof. `sorry` = build failure in CI.

### Stage 3: `pv lint` (7 Gates)
Schema validation, audit trail, scoring, Kani verification, enforcement
level check, reverse coverage (unbound functions).

### Stage 4: build.rs Codegen
Reads contract YAML → sets environment variables:
```
CONTRACT_SOFTMAX_KERNEL_V1_SOFTMAX=implemented
CONTRACT_SOFTMAX_KERNEL_V1_SOFTMAX_PRE_COUNT=2
CONTRACT_SOFTMAX_KERNEL_V1_SOFTMAX_PRE_0=!x.is_empty()
CONTRACT_SOFTMAX_KERNEL_V1_SOFTMAX_POST_0=ret.len() == x.len()
```

### Stage 5: `#[contract]` Proc Macro
Reads env vars at **compile time** via `option_env!()`:
```rust
#[contract("softmax-kernel-v1", equation = "softmax")]
pub fn softmax(x: &[f32]) -> Vec<f32> {
    // Preconditions injected as debug_assert!()
    // Postconditions checked on return value
    // Missing contract YAML → compile_error!()
}
```

Delete the YAML → `option_env!()` returns `None` → `compile_error!()`.
Not a CI check. Not a test failure. A **compiler error**.

### Stage 6: Test Execution
Kani harnesses + falsification tests + property tests run in CI.
Failure blocks merge.

**Zero runtime cost**: all assertions are `debug_assert!()` — expand to
nothing in release builds.

---

## 5. Binding Registries

Each consumer crate has a `binding.yaml` mapping contract equations to
implementations:

```yaml
version: 1.0.0
target_crate: aprender
bindings:
  - contract: softmax-kernel-v1.yaml
    equation: softmax
    module_path: aprender::nn::functional::softmax
    function: softmax
    status: implemented
```

### Integration Across the Stack

| Crate | Bindings | Enforcement |
|-------|----------|-------------|
| aprender | 38 `#[contract]` + build.rs | Compile-time |
| trueno | YAML contracts | CI gates |
| realizar | Quantization + shape contracts | CI + probar |
| entrenar | Optimization contracts | CI gates |
| forjar | 4 `#[contract]` | Compile-time |
| 21 more repos | YAML bindings | Documentation + CI |

---

## 6. Contracts for Compute Backend Equivalence

The most critical use of contracts: proving that all five compute backends
produce equivalent results for the same kernel.

### 6.1 Per-Kernel Equivalence Contract

```yaml
equations:
  rmsnorm:
    formula: "y_i = x_i / sqrt(mean(x²) + ε) * γ_i"

proof_obligations:
  - type: equivalence
    property: "SIMD matches scalar reference"
    formal: "max_ulp_error(simd(x), scalar(x)) <= 2"
    applies_to: simd
  - type: equivalence
    property: "wgpu matches CPU"
    formal: "cosine(wgpu(x), cpu(x)) >= 0.98"
    applies_to: wgpu
  - type: equivalence
    property: "PTX matches CPU"
    formal: "cosine(ptx(x), cpu(x)) >= 0.98"
    applies_to: cuda

simd_dispatch:
  rmsnorm:
    scalar: "rmsnorm_scalar"
    avx2: "rmsnorm_avx2"
    neon: "rmsnorm_neon"
    wgsl: "rmsnorm_shader"
    ptx: "rmsnorm_ptx"
```

### 6.2 Runtime Parity Gate

At model load, the parity gate runs a 1-token forward pass on every
available backend and compares to CPU reference:

```yaml
# contracts/gpu-parity-v2.yaml
equations:
  parity:
    formula: "cosine(gpu_logits, cpu_logits) >= 0.98"

proof_obligations:
  - type: invariant
    property: "At least one backend is correct"
    formal: "∃ b ∈ {wgpu, cuda, nvrtc}: cosine(b, cpu) >= 0.98"
  - type: determinism
    property: "Backend selection is stable"
    formal: "select(model, device) = select(model, device)"
```

### 6.3 Tensor Layout Contract

```yaml
# contracts/tensor-layout-v1.yaml
proof_obligations:
  - type: invariant
    property: "APR tensors are always row-major"
    formal: "∀ t ∈ apr_tensors: layout(t) == RowMajor"
  - type: roundtrip
    property: "GGUF import preserves values"
    formal: "import(export(t)) ≈ t within tolerance"
```

---

## 7. KAIZEN Workflow

Contract-first performance optimization loop:

```
KAIZEN-NNN ticket
  → Write performance contract (YAML with throughput bounds)
    → Implement optimization
      → Measure (contract serves as regression gate)
        → Record improvement in commit message
```

Example: KAIZEN-048 (embed-grad-zero-copy) — contract written, 263x
speedup (145ms → 0.55ms) achieved same day, contract prevents regression.

---

## 8. The `pv` CLI

37 subcommands for contract lifecycle management:

| Command | Purpose |
|---------|---------|
| `pv validate` | Schema + proof obligation validation |
| `pv audit` | Paper → equation → obligation → test traceability |
| `pv score` | 5-dimension quality scoring |
| `pv scaffold` | Generate Rust trait stubs from contract |
| `pv kani` | Generate Kani harness source |
| `pv probar` | Generate property-based test source |
| `pv lint` | Unified 7-gate quality pipeline |
| `pv query` | BM25 semantic search across contracts |
| `pv kaizen` | Fleet-wide enforcement measurement |
| `pv explain` | Chain-of-thought contract narratives |

---

## 9. Key Types

```rust
// Contract — root type, one per YAML file
pub struct Contract {
    pub metadata: Metadata,
    pub equations: BTreeMap<String, Equation>,
    pub proof_obligations: Vec<ProofObligation>,
    pub falsification_tests: Vec<FalsificationTest>,
    pub kani_harnesses: Vec<KaniHarness>,
}

// 26 obligation types
pub enum ObligationType {
    Invariant, Equivalence, Bound, Monotonicity, Precondition,
    Postcondition, Frame, LoopInvariant, /* ... 18 more */
}

// Enforcement levels — cannot weaken once locked
pub enum EnforcementLevel {
    Basic, Standard, Strict, Proven,
}
```

---

## 10. References

1. Meyer (1992) "Applying Design by Contract." IEEE Computer.
2. Chatterjee et al. (2025) "ProofWright: Agentic Formal Verification
   of CUDA." arXiv:2511.12294.
3. Arora et al. (2025) "TensorRight: Automated Verification of Tensor
   Graph Rewrites." arXiv:2511.17838.
4. Rao et al. (2025) "Annotating and Auditing Safety Properties of
   Unsafe Rust." arXiv:2504.21312.
5. Le Blanc & Lam (2025) "Lessons Learned Verifying the Rust Standard
   Library." arXiv:2510.01072.
6. Gond et al. (2026) "LLM-42: Enabling Determinism in LLM Inference."
   arXiv:2601.17768.
7. Zhou et al. (2025) "Linear Layouts: Robust Code Generation of
   Efficient Tensor Computation Using F2." arXiv:2505.23819.
8. Qiu et al. (2024) "Tenspiler: A Verified Lifting-Based Compiler for
   Tensor Operations." arXiv:2404.18249.
