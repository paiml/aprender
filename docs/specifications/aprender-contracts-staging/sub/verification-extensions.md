# provable-contracts — Verification Extension Specification

Version: 1.0
Status: proposed
Date: 2026-04-09

**Orthogonal Approaches, Integration Points, and Implementation Roadmap**

Version 2.0 · March 2026 · Pragmatic AI Labs · [paiml/provable-contracts](https://github.com/paiml/provable-contracts)

---

## Executive Summary

`provable-contracts` implements a seven-phase pipeline (Extract → Specify → Scaffold → Implement → Falsify → Verify → Prove) anchored on YAML kernel contracts, Kani bounded model checking, Lean 4 theorem proving, and probar property-based falsification. The project faithfully captures Bertrand Meyer's Design by Contract methodology and aligns closely with Rust nightly's emerging `#[contracts::requires]` / `#[contracts::ensures]` feature (tracking issue [#128044](https://github.com/rust-lang/rust/issues/128044)).

This specification defines six orthogonal verification approaches that complement, rather than replace, the existing pipeline — each addressing a distinct axis of correctness that bounded model checking alone cannot cover. For each approach, concrete integration points in the codebase are identified alongside a phased implementation roadmap.

---

## Current Architecture

### The Seven-Phase Pipeline

| Phase | Description |
|---|---|
| 1 · Extract | Parse peer-reviewed papers into canonical math (equations, domains, invariants) |
| 2 · Specify | Encode math as YAML kernel contract with proof obligations, falsification predicates, Kani harness definitions |
| 3 · Scaffold | Auto-generate Rust trait stubs and failing test skeletons from contract |
| 4 · Implement | Write scalar reference implementation, then SIMD-accelerated version |
| 5 · Falsify | Popperian falsification via property-based testing (probar + certeza) |
| 6 · Verify | Prove correctness bounds via Kani bounded model checking |
| 7 · Prove | Unbounded proofs via Lean 4 + Mathlib over mathematical reals |

### CLI Surface

| Command | Effect |
|---|---|
| `pv validate` | Parse and validate a YAML kernel contract |
| `pv scaffold` | Generate Rust trait + failing tests |
| `pv kani` | Generate `#[kani::proof]` bounded model harnesses |
| `pv probar` | Generate property-based tests from obligations |
| `pv audit` | Traceability audit: paper ref → equation → obligation → falsification → harness |
| `pv diff` | Compare two contract versions, suggest semver bump |
| `pv coverage` | Cross-contract obligation coverage report |
| `pv generate` | End-to-end codegen (scaffold + kani + probar + book) |
| `pv graph` | Dependency DAG in text / dot / json / mermaid |
| `pv equations` | Display equations in text / latex / ptx / asm |
| `pv score` | 5-dimension contract scoring (spec, falsification, kani, lean, binding) |
| `pv query` | BM25 semantic search across contracts |
| `pv lint` | 5-gate quality pipeline (validate, audit, score, verify, enforce) |
| `pv lean` | Generate Lean 4 definition + theorem files from contracts |
| `pv book` | Generate mdBook pages from contracts |

### Repository Layout

| Path | Contents |
|---|---|
| `contracts/` | YAML kernel contracts across 5 tiers |
| `crates/provable-contracts/` | Core library: schema, scaffold, kani_gen, probar_gen, lean_gen, audit, scoring, query, lint |
| `crates/provable-contracts-cli/` | `pv` binary; CLI command dispatch |
| `book/src/` | mdBook documentation source |
| `lean/` | Lean 4 proof files (via forjar) |
| `docs/specifications/` | This and related specification documents |

---

## Orthogonal Verification Approaches

Each approach occupies a distinct position in the automation–completeness space. They are not alternatives to the current pipeline — they are additive layers addressing bug classes that bounded model checking cannot reach.

---

## 1. Type Invariants

**Orthogonal axis:** Type-system-enforced invariants vs. function-boundary assertions.

Currently, invariants in YAML contracts are enforced at function call boundaries via Kani harnesses. True type invariants are properties the compiler guarantees hold for every value of a type at all stable points — closer to Meyer's original class invariant semantics.

### YAML Schema Extension

Add a top-level `type_invariants` section to the contract schema:

```yaml
type_invariants:
  - name: tensor_validity
    type_name: ValidatedTensor
    predicate: "self.dims.iter().product::<usize>() == self.data.len()"
    description: Data length equals product of dimensions
    check_mode: debug   # debug | always | never
  - name: tensor_non_empty
    type_name: ValidatedTensor
    predicate: "!self.dims.is_empty()"
    description: At least one dimension required
```

### Path A — Rust Nightly `contracts::invariant` (forthcoming)

The GSoC 2025 work added groundwork for type invariants in nightly Rust (tracking issue #128044). Once stabilised, `pv scaffold` generates:

```rust
#![feature(contracts)]

#[core::contracts::invariant(
    self.dims.iter().product::<usize>() == self.data.len()
)]
#[core::contracts::invariant(!self.dims.is_empty())]
pub struct ValidatedTensor {
    pub dims: Vec<usize>,
    pub data: Vec<f32>,
}
```

### Path B — `Invariant` Trait (available now, Kani-compatible)

```rust
pub trait Invariant {
    fn is_valid(&self) -> bool;
}

impl Invariant for ValidatedTensor {
    fn is_valid(&self) -> bool {
        !self.dims.is_empty() &&
        self.dims.iter().product::<usize>() == self.data.len()
    }
}

// Auto-generated preservation harness
#[kani::proof]
fn verify_tensor_invariant_preserved_by_reshape() {
    let t: ValidatedTensor = kani::any();
    kani::assume(t.is_valid());
    let result = t.reshape(kani::any());
    assert!(result.map(|r| r.is_valid()).unwrap_or(true));
}
```

### Integration Points

- **`crates/provable-contracts/src/schema/types.rs`** — Add `TypeInvariant` struct to the `Contract` schema (`name`, `type_name`, `predicate`, `check_mode` fields).
- **`crates/provable-contracts/src/kani_gen.rs`** — Add `generate_invariant_harnesses()` — for each type invariant, emit a `#[kani::proof]` preservation harness for every function touching that type.
- **`crates/provable-contracts/src/scaffold.rs`** — Add `generate_invariant_trait()` — emit the `Invariant` trait impl and, when nightly flag is active, the `#[contracts::invariant(...)]` attribute annotations.
- **`crates/provable-contracts-cli/`** — Add `pv invariants` subcommand. Flags: `--nightly` (use `#[contracts::invariant]`), `--stable` (use `Invariant` trait, default), `--harnesses` (also generate Kani preservation proofs).
- **`crates/provable-contracts/src/audit.rs`** — Extend audit to verify every type with declared invariants has at least one preservation harness covering each function that takes or returns that type.

---

## 2. Theorem Proving (Coq)

**Orthogonal axis:** Full unbounded correctness proofs (Coq) vs. bounded model checking (Kani).

Kani proves properties up to a finite input space. Lean 4 (Phase 7) proves over mathematical reals. Coq adds a second theorem proving axis with `coq-of-rust` for direct Rust-to-Coq translation — proofs reference the actual implementation rather than a parallel specification.

### Tiered Proof Strategy

| Tier | Mechanism |
|---|---|
| Tier 1 (current) | Kani bounded model checking — automated, bounded |
| Tier 2 (current) | Lean 4 theorem proving — unbounded proofs over reals |
| Tier 3 (new) | Coq `admit` stubs — generated by `pv coq`, awaiting manual proof |
| Tier 4 (new) | Coq proved theorems — human-filled proofs, CI-verified with `coqc` |

### YAML Schema Extension

```yaml
coq_spec:
  module: SoftmaxSpec
  imports:
    - "Require Import Reals."
    - "Require Import List."
  definitions:
    - name: softmax_sum_to_one
      statement: |
        Theorem softmax_partition_of_unity : forall (xs : list R),
          xs <> [] ->
          fold_left Rplus (map softmax_fn xs) 0 = 1.
  obligations:
    - links_to: numerical_stability
      coq_lemma: softmax_partition_of_unity
      status: admit   # admit | proved
```

### Generated Coq Output

`pv coq contracts/softmax-kernel-v1.yaml` emits:

```coq
(* Generated from softmax-kernel-v1 v1.0 *)
(* Paper: Attention Is All You Need, Vaswani et al. 2017 *)
Require Import Reals.
Require Import List.

(** Obligation: numerical_stability *)
(** Paper ref: arxiv:1706.03762 eq. 1 *)
Theorem softmax_partition_of_unity : forall (xs : list R),
  xs <> [] ->
  fold_left Rplus (map softmax_fn xs) 0 = 1.
Proof.
  admit. (* replace with proof *)
Qed.
```

### `coq-of-rust` Bridge

Translates Rust source directly to Coq, ensuring proofs reference the actual implementation rather than a parallel specification.

### Integration Points

- **`crates/provable-contracts/src/schema/types.rs`** — Add `CoqSpec` struct: module name, imports, theorem definitions, obligation links, proof status (`admit | proved`).
- **`crates/provable-contracts/src/coq_gen.rs`** — New module. Implement `generate_coq_spec(contract)` — emit `.v` file with imports, equation-derived definitions, and `obligation_to_coq_theorem()` stubs defaulting to `admit.`.
- **`crates/provable-contracts-cli/`** — Add `pv coq` subcommand. Flags: `--output-dir`, `--format` (vernacular | json), `--coq-of-rust` (invoke translation bridge).
- **`crates/provable-contracts/src/audit.rs`** — Extend `pv audit --coq` to report coverage tiers per obligation: `kani-only`, `lean-proved`, `coq-admit`, `coq-proved`.
- **`.github/workflows/ci.yml`** — Add `coq-check` job: `coqc` on every `.v` file in `generated/coq/`. Fail on `coqc` errors. Optionally enforce that `admit` count does not increase on PRs.

---

## 3. Coverage-Guided Fuzzing (cargo-fuzz / libFuzzer)

**Orthogonal axis:** Adversarial coverage-driven exploration vs. structured property checking.

`probar` generates inputs that satisfy preconditions and checks postconditions — mathematical properties. Fuzzing mutates inputs aggressively to find crashes, panics, sanitizer violations, and memory safety issues. These are genuinely different bug classes.

### Auto-Generated Fuzz Targets from Contracts

`pv fuzz contracts/softmax-kernel-v1.yaml` emits to `fuzz/fuzz_targets/softmax.rs`:

```rust
#![no_main]
use libfuzzer_sys::fuzz_target;
use aprender::kernels::softmax;

fuzz_target!(|data: &[u8]| {
    if data.len() < 4 { return; }
    let len = (data[0] as usize % 64) + 1;
    if data.len() < 4 + len * 4 { return; }
    let input: Vec<f32> = data[4..4 + len * 4]
        .chunks(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    // Contract precondition gate — from requires: !input.is_empty() && no NaN
    if input.iter().any(|x| x.is_nan()) { return; }
    // Should never panic — contract guarantees output in [0,1]
    let _ = std::panic::catch_unwind(|| softmax(&input));
});
```

### Integration Points

- **`crates/provable-contracts/src/fuzz_gen.rs`** — New module. Implement `generate_fuzz_target(contract)` — emit a `libfuzzer_sys::fuzz_target!` that gates on contract preconditions and calls the bound function. One target per binding entry.
- **`crates/provable-contracts-cli/`** — Add `pv fuzz` subcommand. Output to `fuzz/fuzz_targets/`. Flags: `--sanitizer` (address | memory | thread), `--max-len`, `--timeout`.
- **`Cargo.toml` (workspace)** — Add optional `fuzz` member to workspace when `pv fuzz` is first invoked. Emit `[dependencies]` on `libfuzzer-sys` and downstream crate into generated `fuzz/Cargo.toml`.

---

## 4. Abstract Interpretation (MIRAI)

**Orthogonal axis:** Sound over-approximation vs. precise bounded checking.

Abstract interpretation proves properties over all possible inputs by abstracting values (intervals, sign, nullability). It is sound — no false negatives — but may report false positives. For ML kernels this catches numerical range violations (softmax output always in [0,1]) provably, without Kani's input-space bound.

### MIRAI Integration for Rust

`pv mirai contracts/softmax-kernel-v1.yaml` generates annotations:

```rust
use mirai_annotations::*;

pub fn softmax(xs: &[f32]) -> Vec<f32> {
    precondition!(!xs.is_empty());
    precondition!(xs.iter().all(|x| !x.is_nan()));
    // ... implementation ...
    postcondition!(result.iter().all(|&x| x >= 0.0 && x <= 1.0));
    postcondition!((result.iter().sum::<f32>() - 1.0).abs() < 1e-5);
    result
}
```

### Integration Points

- **`crates/provable-contracts/src/mirai_gen.rs`** — New module. Translate YAML `proof_obligations` into MIRAI `precondition!` / `postcondition!` / `verify!` annotations. Map numeric range obligations to interval preconditions.
- **`crates/provable-contracts-cli/`** — Add `pv mirai` subcommand. Emit annotated source to `generated/mirai/`. Flag: `--emit-tags` (generate tag structs for taint analysis).
- **`crates/provable-contracts/src/audit.rs`** — Extend audit report with `mirai_annotations_present: bool` per obligation. Track which obligations have both Kani harnesses and MIRAI annotations (belt-and-suspenders coverage).

---

## 5. Refinement Types (Flux for Rust)

**Orthogonal axis:** Type-system-integrated compile-time proof vs. external annotation layer.

Flux brings LiquidHaskell-style refinement types to Rust via nightly. Types carry value predicates: `Vec<f32>[n]` is a vector of exactly `n` elements. The compiler verifies these via SMT at compile time — zero runtime cost, zero manual proof burden for the programmer.

### Flux Annotation Generation from Contracts

`pv flux contracts/tensor-shape-flow.yaml` generates:

```rust
#[flux::refined_by(rows: int, cols: int)]
pub struct Matrix {
    #[flux::field(Vec<f32>[rows * cols])]
    data: Vec<f32>,
    rows: usize,
    cols: usize,
}

#[flux::sig(
    fn(a: &Matrix[@r1, @c1], b: &Matrix[@c1, @c2]) -> Matrix[r1, c2]
)]
pub fn matmul(a: &Matrix, b: &Matrix) -> Matrix {
    // Compiler rejects shape-incompatible calls at compile time
    todo!()
}
```

### Integration Points

- **`crates/provable-contracts/src/flux_gen.rs`** — New module. Map YAML `tensor_shape` contracts to Flux `#[flux::refined_by]` structs and `#[flux::sig]` function signatures. Focus on `tensor-shape-flow`, `validated-tensor`, and `qwen35-shapes` contracts initially.
- **`crates/provable-contracts-cli/`** — Add `pv flux` subcommand. Output generated Flux annotations to `generated/flux/`. Flag: `--verify` (invoke Flux checker inline via `cargo flux`).
- **`crates/provable-contracts/src/audit.rs`** — Add `flux_coverage` field to `AuditReport`. Track which shape-related proof obligations are discharged by Flux at compile time vs. requiring Kani runtime harnesses.

---

## 6. System-Level Model Checking (TLA+ / Alloy)

**Orthogonal axis:** System/protocol-level correctness vs. kernel-level value correctness.

Kani verifies that `softmax` computes the right values. TLA+ verifies that the inference pipeline as a whole maintains liveness, safety, and ordering invariants — that the KV cache is never corrupted across concurrent requests, that attention masking never drops required tokens, that quantization and dequantization are inverse operations at the pipeline level.

### TLA+ Spec Generation from the Contract DAG

`pv tla contracts/qwen35/` generates from the Qwen 3.5 dependency DAG:

```tla
---- MODULE InferencePipeline ----
EXTENDS Naturals, Sequences, TLC

(* Auto-generated from qwen35-e2e-verification.yaml
   Contract DAG: softmax <- attention <- sliding-window-attention
                 silu <- swiglu <- qwen35-hybrid-forward <- e2e *)

VARIABLES kv_cache, active_layers, token_budget

Init == kv_cache = {} /\ active_layers = 0 /\ token_budget = MaxTokens

(* Invariant from kv-cache-equivalence contract *)
KVConsistency == \A k \in DOMAIN kv_cache :
    kv_cache[k].size <= MaxCacheSize

(* Invariant from attention-scaling contract *)
AttentionNormalized == \A layer \in 1..active_layers :
    attention_weights[layer].sum = 1

(* Liveness: pipeline always eventually terminates *)
Termination == <>(active_layers = 0)

Spec == Init /\ [][Next]_vars /\ WF_vars(Next)
INVARIANT KVConsistency /\ AttentionNormalized
PROPERTY Termination
====
```

### Integration Points

- **`crates/provable-contracts/src/tla_gen.rs`** — New module. Translate the contract dependency DAG (from `pv graph` output) into a TLA+ `MODULE`. Map `invariants` to `INVARIANT` declarations, liveness obligations to `PROPERTY` declarations. Focus on the Qwen 3.5 verification DAG as first target.
- **`crates/provable-contracts-cli/`** — Add `pv tla` subcommand. Flags: `--output` (emit `.tla` files), `--check` (invoke TLC model checker if installed), `--alloy` (emit Alloy `.als` instead of TLA+).
- **`crates/provable-contracts/src/graph.rs`** — Extend the existing dependency graph module to export a structured DAG representation consumed by `tla_gen.rs` — nodes are contract names, edges are `depends_on` dependencies, invariants are node annotations.
- **`crates/provable-contracts/src/audit.rs`** — Add `tla_spec_present: bool` to `AuditReport` for contracts with pipeline-level obligations. Flag contracts that have cross-contract dependencies (i.e., appear in `depends_on` chains) but no TLA+ system spec covering them.

---

## Extended Pipeline Architecture

After all six extensions, the full pipeline from YAML contract to proof coverage is:

| Phase | Command / Tool |
|---|---|
| 1 · Extract | `pv validate` — parse paper → canonical math |
| 2 · Specify | YAML contract authoring — equations, obligations, falsification predicates, `type_invariants`, `coq_spec` |
| 3 · Scaffold | `pv scaffold` — Rust traits, tests; `pv invariants` — `Invariant` trait + preservation harnesses |
| 4 · Implement | Human: scalar ref impl, then SIMD impl |
| 5 · Falsify | `pv probar` — property-based (Popperian); `pv fuzz` — adversarial coverage-guided |
| 6 · Verify (bounded) | `pv kani` — bounded model checking; `pv mirai` — abstract interpretation |
| 6.5 · Type-level | `pv flux` — refinement type annotations (compile-time, SMT-discharged) |
| 7 · Prove (full) | `pv lean` — Lean 4 proofs; `pv coq` — Coq theorem stubs → human fills proofs → `coqc` in CI |
| 8 · System-level | `pv tla` — TLA+ protocol specs for pipeline-level invariants |
| Audit | `pv audit` — full traceability: paper → eq → obligation → falsify → kani → lean → flux → coq tier → tla |

---

## Rust Nightly Contracts Alignment

Rust nightly's contracts feature (tracking issue #128044, feature gate `#![feature(contracts)]`) is converging toward the same vocabulary this project already implements in YAML. The table maps each nightly concept to current and planned integration points.

| Nightly Rust concept | provable-contracts equivalent |
|---|---|
| `#[contracts::requires(...)]` | YAML `proof_obligations[].preconditions` → `pv scaffold` generates `#[contracts::requires]` on nightly or Kani `requires` on stable |
| `#[contracts::ensures(...)]` | YAML `proof_obligations[].postconditions` → `pv scaffold` generates `#[contracts::ensures]` or Kani `ensures` |
| `#[contracts::invariant(...)]` | YAML `type_invariants[].predicate` → `pv invariants` Path A (nightly) or `Invariant` trait Path B (stable) |
| Safety vs. correctness split | YAML `check_mode: debug | always | never` maps to the nightly safety vs. correctness contract split |
| No runtime penalty axiom | Respected: contracts compile out unless `--cfg contracts_enabled`; Kani harnesses are `#[cfg(kani)]` gated |
| Compiler interface for tools | `pv audit --nightly` emits the contract AST via rustc's contract intrinsics for external tool consumption |
| Stdlib instrumentation goal | Analogous to `pv audit` chain: every unsafe function has machine-checkable contracts traceable to documentation |

---

## Implementation Roadmap

| Approach | Status | Primary files | Key dependency |
|---|---|---|---|
| Type invariants (`Invariant` trait) | **Current** | `schema/types.rs`, `kani_gen.rs`, `scaffold.rs` | None — stable today |
| `cargo-fuzz` targets from YAML | **Near-term** | `fuzz_gen.rs`, CLI | `cargo-fuzz`, `libfuzzer-sys` |
| Flux refinement types | **Near-term** | `flux_gen.rs`, `audit.rs` | `flux` (nightly), `cargo flux` |
| MIRAI abstract interpretation | **Medium-term** | `mirai_gen.rs`, CLI | `MIRAI`, `mirai_annotations` crate |
| Coq theorem proving | **Medium-term** | `coq_gen.rs`, `audit.rs`, `ci.yml` | `coq-of-rust`, `coqc` |
| TLA+ system model checking | **Long-term** | `tla_gen.rs`, CLI | TLC or Apalache |
| Type invariants (nightly attrs) | **Ambitious** | `scaffold.rs --nightly` flag | Rust nightly #128044 stabilisation |

---

## New CLI Commands Summary

| Command | Description |
|---|---|
| `pv invariants` | Generate `Invariant` trait impl + Kani preservation harnesses from `type_invariants` section |
| `pv fuzz` | Generate libfuzzer fuzz targets gated on contract preconditions |
| `pv flux` | Generate Flux refinement type annotations for shape contracts |
| `pv mirai` | Generate MIRAI `precondition!`/`postcondition!` annotations |
| `pv coq` | Generate Coq `.v` theorem stubs from obligations; optionally invoke `coq-of-rust` |
| `pv tla` | Generate TLA+ `MODULE` from contract dependency DAG |
| `pv audit --coq` | Extended audit: report `kani-only` / `lean-proved` / `coq-admit` / `coq-proved` tiers per obligation |
| `pv audit --flux` | Extended audit: report which shape obligations are Flux-discharged vs. Kani-needed |
| `pv coverage --fuzz` | Report branch coverage from fuzzing alongside probar coverage |

---

## Alignment Scorecard

Scores how well each pipeline state captures Meyer's 2025 research agenda and Rust nightly contracts design axioms.

| Property | Current | After roadmap | Notes |
|---|---|---|---|
| Contracts as primary artifact | Full | Full | YAML contracts drive all codegen |
| Proofs + tests unification | Full | Full | Every obligation has a falsification test |
| Popperian falsification | Full | Full | Phase 5 explicitly named Falsify |
| Paper provenance / traceability | Full | Full | `pv audit` traces paper → proof |
| Type invariants (class invariants) | Partial | Full | Function-level now; type-level after Phase 1 of roadmap |
| Full correctness proofs (not bounded) | Partial | Full | Lean 4 covers reals; Coq adds Rust-level proofs |
| Adversarial / safety-class bug detection | Gap | Full | Fuzzing adds crash / sanitizer coverage |
| Rust nightly contracts compatibility | Full | Full | `requires` / `ensures` / `invariant` all present |
| System-level protocol correctness | Gap | Partial | TLA+ covers pipeline; individual kernels stay Kani |
| Axiom-minimal, theorem-rich structure | Full | Full | YAML schema minimal; harnesses derived |

---

## Key References

- Meyer, Bertrand. *Object-Oriented Software Construction*, 2nd ed. Prentice Hall, 1997.
- Meyer, B. *Software engineering as a domain to formalize.* arXiv preprint, February 2025.
- Meyer, B., et al. *Do AI models help produce verified bug fixes?* arXiv, July 2025.
- Huang, L., Meyer, B., Weber, R. *Loop Unrolling: Formal Definition and Application to Testing.* September 2025.
- Rust tracking issue [#128044](https://github.com/rust-lang/rust/issues/128044) — Contracts feature (`requires` / `ensures` / `invariant`). rust-lang/rust, 2024–2025.
- Rust Project Goal — [Instrument the Rust standard library with safety contracts](https://rust-lang.github.io/rust-project-goals/2025h1/std-contracts.html). 2025H1. Celina G. Val, Michael Tautschnig.
- GSoC 2025 Final Report: Rust safety contracts. github.com/dawidl022.
- [Kani Rust Verifier — Function Contracts](https://model-checking.github.io/kani-verifier-blog/2024/01/29/function-contracts.html). January 2024.
- [coq-of-rust](https://github.com/formal-land/coq-of-rust) — Formal verification of Rust code in Coq.
- [Flux](https://flux-rs.github.io) — Refinement Types for Rust. 2023–2025.
- [MIRAI](https://github.com/facebookexperimental/MIRAI) — Abstract interpreter for Rust.
- Huang et al. *Lessons from Formally Verified Deployed Software Systems.* 2023.
