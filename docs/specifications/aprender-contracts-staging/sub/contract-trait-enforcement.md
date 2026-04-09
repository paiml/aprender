# Sub-spec: Contract-Trait Enforcement

Version: 1.0
Status: proposed
Date: 2026-04-09

**Parent:** [pv-spec.md](../pv-spec.md) Section 23

---

## Problem Statement

The current enforcement chain has a structural gap: build.rs checks that
binding.yaml *says* "implemented," but nothing verifies the function
*actually exists* with the *correct signature*. Of 16,977 bindings, only
35 have `#[contract]` proc macro annotations that the compiler can verify.

Build.rs source scanning (the L2 approach) is fragile because:
1. String-matching `pub fn` misses `impl` methods and trait impls
2. 13 copy-pasted inline scanners with no shared code
3. Name-only matching ignores signatures and module paths
4. Build-time I/O on every `cargo build` adds latency
5. Cannot catch type-level mismatches (argument types, return types)

## Chain of Thought

### Step 1: What does "enforced" mean?

Three properties to verify, in order of difficulty:
1. **Existence** — the function exists in the crate's public API
2. **Signature** — it has the expected argument and return types
3. **Behavior** — it satisfies preconditions/postconditions

The Rust compiler already enforces (1) and (2) for anything that passes
type-checking. The question is: how do we make the compiler check that
binding.yaml entries correspond to *real, type-checked* functions?

### Step 2: What do similar systems do?

**SPARK/Ada** (gold standard): Language-level spec/body separation.
The Ada compiler *requires* that every subprogram specification has a
matching body. GNATprove then proves the body satisfies the contract.
Refs: AdaCore SPARK User's Guide §7.4 "Subprogram Contracts."

**Eiffel**: Deferred features (abstract methods) carry contracts.
Any effective (concrete) descendant *must* implement them. The compiler
refuses to instantiate classes with unresolved deferred features.
Refs: Meyer (1988) §11.1; ETH Eiffel Tutorial "Deferred Classes."

**Kani**: `#[kani::proof_for_contract(function_name)]` explicitly binds
a proof harness to a specific function. The function MUST exist — a
typo in the function name is a compile error.
Refs: Kani RFC 0009 "Function Contracts"; VanHattum et al. (2022).

**Prusti**: `#[refine_trait_spec]` on impl blocks adds/refines contracts
for trait methods. The trait itself is the "spec" and the impl is the
"body." If the impl doesn't exist, the trait is incomplete → compile error.
Refs: Astrauskas et al. (2022) "The Prusti Project" (ETH Zurich).

**Creusot**: Trait laws via `#[law]` attribute define algebraic
properties that all implementations must satisfy. The WhyML translation
connects trait contracts to impl bodies via Why3.
Refs: Denis et al. (2022) "Creusot: A Foundry for Rust Verification" ICFEM.

**Batuta (paiml-mcp-agent-toolkit)**: Uses a `ContractValidation`
trait pattern — every contract struct `impl ContractValidation` with a
`validate()` method. Interface-agnostic (CLI/MCP/HTTP share the same
trait). This is the "single source of truth" pattern: define the
contract as a Rust type, then the compiler enforces implementers exist.

### Step 3: What's the Rust-native solution?

The Rust type system already solves this problem: **traits**.

If we generate a trait per contract (from YAML), and consumer crates
`impl` that trait, the Rust compiler verifies:
- The function exists (trait method must be implemented)
- The signature matches (trait method has a fixed signature)
- Visibility is correct (trait impls are public)

This is exactly what `pv scaffold` already does — it generates trait
stubs. The missing step is **making the trait mandatory** (not optional).

### Step 4: Why traits over alternatives?

| Approach | Existence | Signature | No build.rs | One-time |
|----------|-----------|-----------|-------------|----------|
| build.rs scan | Partial | No | No | No (13 copies) |
| `use` import test | Yes | Partial | Yes | Yes |
| `inventory`/`linkme` | Yes | No | Yes | No (runtime dep) |
| rustdoc JSON | Yes | Yes | Yes | No (nightly) |
| **Trait enforcement** | **Yes** | **Yes** | **Yes** | **Yes** |
| `#[contract]` macro | Partial | No | No | No (per-fn) |

Traits win on every dimension. They're:
- **Stable Rust** (no nightly features)
- **Zero runtime cost** (traits are erased at compile time)
- **One-time setup** (generate trait once from YAML, `impl` once)
- **Compiler-enforced** (missing method = compile error)
- **Signature-checked** (wrong arg type = compile error)

### Step 5: The design

```
YAML Contract                    Generated Trait              Consumer Impl
─────────────                    ───────────────              ─────────────
softmax-kernel-v1.yaml    →     SoftmaxKernelV1 trait   ←    impl SoftmaxKernelV1
  equations:                      fn softmax(...)              for Aprender { ... }
    softmax:                      fn log_softmax(...)
    log_softmax:
```

`pv scaffold` generates the trait. Consumer crates add one line:
```rust
use provable_contracts::traits::SoftmaxKernelV1;
impl SoftmaxKernelV1 for MyKernels { ... }
```

If the YAML changes (equation added/removed/renamed), the trait changes,
and **every consumer that `impl`s it gets a compile error** until they
update. No build.rs. No scanning. No name matching. The compiler does it.

---

## Design

### Phase 1: Trait generation (`pv scaffold --trait`) [IMPLEMENTED]

`pv scaffold` already generates traits. Enhance to produce a standalone
trait file per contract:

```rust
// provable-contracts/src/traits/softmax_kernel_v1.rs
// AUTO-GENERATED from contracts/softmax-kernel-v1.yaml
// DO NOT EDIT — regenerate with: pv scaffold --trait contracts/softmax-kernel-v1.yaml

/// Contract trait for softmax-kernel-v1.
///
/// Implementors must provide all equation implementations.
/// Generated from: softmax-kernel-v1.yaml v1.0.0
pub trait SoftmaxKernelV1 {
    /// σ(x)_i = exp(x_i) / Σ_j exp(x_j)
    fn softmax(&self, x: &[f32]) -> Vec<f32>;

    /// log(σ(x)_i) = x_i - log(Σ_j exp(x_j))
    fn log_softmax(&self, x: &[f32]) -> Vec<f32>;
}
```

### Phase 2: Consumer adoption (incremental) [IMPLEMENTED]

**Deployed to ALL 13/13 repos:**

| Repo | Traits | Tests | Status |
|------|--------|-------|--------|
| aprender | 13/13 | 13 | All pass |
| trueno | 13/13 | 19 | All pass |
| entrenar | 13/13 | 13 | All pass |
| pmat | 13/13 | 12 | All pass |
| realizar | 13/13 | 9 | All pass |
| forjar | 13/13 | 9 | All pass |
| presentar | 13/13 | 9 | All pass |
| rmedia | 13/13 | 9 | All pass |
| bashrs | 13/13 | 9 | All pass |
| depyler | 13/13 | 9 | All pass |
| decy | 13/13 | 9 | All pass |
| ruchy | 13/13 | 9 | All pass |
| simular | 13/13 | 9 | All pass |

Total: **138 trait tests** across **13/13 repos**, all passing.
All 13 contracts compiler-enforced in every consumer repo.

Multi-input signatures generated from YAML domains (`split_once` parser).
CI trait-staleness check in `.github/workflows/ci.yml`.
Behavioral assertions: softmax sum-to-1, ReLU non-neg, sigmoid range,
RMS normalization, LayerNorm standardization, CE non-negative loss.

depyler-core required fixing 77 pre-existing compilation errors
(deleted module files from pmat split cleanup, missing type aliases,
duplicate method names, borrow conflicts). Five-whys root cause:
commit `bc8fbafa` deleted critical `impl RustCodeGen` files.

Consumer crates add a dependency on the trait module and implement it:

```rust
// aprender/src/nn/mod.rs
use provable_contracts::traits::SoftmaxKernelV1;

pub struct NNFunctional;

impl SoftmaxKernelV1 for NNFunctional {
    fn softmax(&self, x: &[f32]) -> Vec<f32> { /* real impl */ }
    fn log_softmax(&self, x: &[f32]) -> Vec<f32> { /* real impl */ }
}
```

Missing method → compile error. Wrong signature → compile error.

### Phase 3: CI verification (`pv verify-trait`)

A `pv verify-trait` command checks that every contract with bindings
has a corresponding `impl` in the consumer crate:

```bash
pv verify-trait contracts/softmax-kernel-v1.yaml --crate ../aprender
# Checks: trait SoftmaxKernelV1 is impl'd somewhere in ../aprender/src/
```

### Gradual migration path

1. **Week 1**: Generate traits for Tier 1 contracts (12 foundation kernels)
2. **Week 2**: Implement traits in aprender + trueno (highest coverage)
3. **Month 1**: All Tier 1-2 contracts have trait enforcement
4. **Month 3**: Trait enforcement required for all new contracts

Existing build.rs AllImplemented enforcement remains as L1 (registry
completeness). Traits become L2 (compiler-verified binding). The two
layers are complementary, not competing.

---

## References

### Formal Verification

1. Astrauskas, V. et al. (2022). "The Prusti Project: Formal Verification
   for Rust." ETH Zurich. pm.inf.ethz.ch/publications.
2. Denis, X. et al. (2022). "Creusot: A Foundry for the Deductive
   Verification of Rust Programs." ICFEM 2022.
3. Lattuada, A. et al. (2023). "Verus: Verifying Rust Programs using
   Linear Ghost Types." arXiv:2303.05491.
4. Lehmann, N. & Tanter, E. (2023). "Gradual Liquid Type Inference."
   OOPSLA 2023.
5. Matsushita, Y. et al. (2022). "RustHornBelt: A Semantic Foundation
   for Functional Verification of Rust Programs with Unsafe Code."
   PLDI 2022.
6. "Surveying the Rust Verification Landscape." arXiv:2410.01981 (2024).

### Design by Contract

7. Meyer, B. (1988). *Object-Oriented Software Construction.* Ch. 11.
8. Meyer, B. (2025). "Software engineering as a domain to formalize."
   arXiv:2502.11434.
9. Li, Y. et al. (2025). "Do Large Language Models Respect Contracts?"
   arXiv:2510.12047.

### Rust Language

10. Rust Compiler Team (2024). "MCP 759: Contracts and Invariants."
    github.com/rust-lang/compiler-team/issues/759.
11. Rust Project Goals (2025). "Instrument std with safety contracts."
    rust-lang.github.io/rust-project-goals/2025h1/std-contracts.html.
12. Kani Contributors (2024). "RFC 0009: Function Contracts."
    model-checking.github.io/kani/rfc/rfcs/0009-function-contracts.html.

### Architecture Patterns

13. SPARK/Ada User's Guide (2026). "Subprogram Contracts."
    docs.adacore.com/spark2014-docs.
14. Eiffel Tutorial (2026). "Deferred Classes and Seamless Development."
    eiffel.org/doc/eiffel.
15. PAIML Engineering (2026). "Batuta ContractValidation trait pattern."
    paiml-mcp-agent-toolkit/src/contracts/contract_validation.rs.
