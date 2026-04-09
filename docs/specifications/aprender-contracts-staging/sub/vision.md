# 1. Vision and Architecture

Version: 1.0
Status: proposed
Date: 2026-04-09

**Papers to Math to Contracts in Code.**

## The Problem

ML kernel implementations derive from research papers, but the derivation
chain is invisible:

```
Paper (LaTeX) -> Developer's head -> Code -> Tests -> Ship
```

The developer's head is an unauditable black box. When a SIMD kernel
produces wrong results six months later, nobody can trace back to which
equation was violated or which paper assumption broke.

## The Solution

Make the derivation chain explicit, auditable, and provable:

```
Paper (arXiv)
  -> Equations (canonical math)
    -> Contract (YAML, machine-parseable)
      -> Trait (Rust scaffold)
        -> Kernel (scalar, SIMD, PTX)
          -> Tests (probar property tests)
            -> Proof (Kani bounded model checking)
              -> Theorem (Lean 4 unbounded proof)
```

Every link is a concrete artifact in version control. The final links —
Kani and Lean 4 — elevate this from "really good testing" to "actual
proof."

## Theoretical Foundations

| Foundation | Source | Application |
|---|---|---|
| Falsificationism | Popper (1959) | Tests designed to refute, not confirm |
| Poka-Yoke | Shingo (1986) | `ValidatedTensor` makes bad states unrepresentable |
| Jidoka | Ohno (1988) | Stop the line on first defect |
| Design by Contract | Meyer (1988) | Preconditions, postconditions, invariants |
| Type-Driven Dev | Brady (2017) | Parse, don't validate (King 2019) |
| Equation-Driven Dev | batuta oracle | Equation -> failing test -> impl -> verify |
| Bounded Model Checking | Kani (AWS 2022) | Exhaustive verification within bounds |
| Theorem Proving | Lean 4 + Mathlib | Unbounded proofs over mathematical reals |

## Architecture

```
provable-contracts/
+-- crates/
|   +-- provable-contracts/         Library crate
|   |   +-- schema/                 YAML parsing + validation
|   |   +-- scaffold/               Trait + test generation
|   |   +-- kani_gen/               Kani harness codegen
|   |   +-- probar_gen/             Property test codegen
|   |   +-- lean_gen/               Lean 4 codegen
|   |   +-- audit/                  Traceability chain
|   |   +-- scoring/                Contract + codebase scoring
|   |   +-- query/                  BM25 contract search index
|   |   +-- binding.rs              Contract -> impl mapping
|   |   +-- diff.rs                 Version diffing
|   |   +-- coverage.rs             Obligation coverage
|   |   +-- graph.rs                Dependency DAG
|   |   +-- proof_status.rs         L1-L5 levels
|   |   +-- kernels/                Reference implementations
|   |   +-- error.rs                Error types
|   +-- provable-contracts-cli/     CLI binary (`pv`)
|   +-- provable-contracts-macros/  Proc macro (#[contract])
+-- contracts/                      YAML contract registry (271 scored contracts)
+-- docs/specifications/            This spec
```

## Scale

| Metric | Value | Verified |
|---|---|---|
| YAML contracts (total files) | 324 | `find contracts/ -name '*.yaml' ! -name 'binding.yaml' \| wc -l` |
| Parseable scored contracts | 271+ | `pv coverage` (excludes kaizen/, legacy/, pipelines/) |
| Equations | 896+ | `pv coverage contracts/` (recursive) |
| Proof obligations | 1241+ | `pv coverage contracts/` (recursive) |
| Falsification tests | 1365+ | `pv coverage contracts/` (recursive) |
| Kani harnesses (YAML-defined) | 1355+ | `pv coverage contracts/` (recursive) |
| **Real bindings (with module_path)** | **958** | `grep -r '^- contract:' contracts/*/binding.yaml` |
| Binding repos with entries | 41 directories | `ls contracts/*/binding.yaml` |
| Kaizen-scanned repos | 40 | `pv kaizen` (requires sibling source directory or `source_dir`) |
| Proof obligation types | 26 (19 property + 7 Eiffel DbC) | schema/types.rs |
| CLI commands | 34 | `pv --help` (includes `pv pipeline`) |
| Repos with build.rs enforcement | 7/40 | aprender, trueno, entrenar, realizar, forjar, ruchy, simular |
| Repos with trait tests | 11/40 | manual audit 2026-03-28 |
| `#[contract]` proc-macro annotations | 18 | forjar: 4, paiml-mcp-agent-toolkit: 11, batuta: 3 |
| Stack LoC governed | ~6.4M Rust | — |

> **v2.2.0 Correction (2026-03-28):** Previous versions inflated binding
> counts with mass-generated entries lacking `module_path`. v2.2.0 stripped
> 28,206 ghost bindings, leaving 660 real bindings that reference actual
> module paths. Of those, ~234 resolve to functions in source code.
> The honest binding rate is **234 verified / 516 equations = 45%** for
> the best-covered repo (aprender), not the 100% previously claimed.
>
> **v2.3.0 Correction (2026-03-28):** `pv coverage` recursion bug fixed —
> previously only scanned top-level contracts/ (131 files), now recurses
> into subdirectories (165 parseable contracts). Totals updated accordingly.
>
> **v2.6.0 Update (2026-04-04):** Fleet expanded from 25→40 repos via
> `source_dir` binding.yaml field. Real binding entries grew to 958 (from
> 660) with 6 new apr-cli bindings and 4 new binding stubs. kaizen now
> resolves name-mismatched repos (pmat → paiml-mcp-agent-toolkit,
> pmcp → rust-mcp-sdk). PV-AUD-003 reduced to 0. pmat infrastructure
> contracts at 0 lint warnings (51/51 equations with pre+postconditions).
> Work-DBC lifecycle state machine falsified and fixed in pmat implementation.
