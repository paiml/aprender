# Integration with PAIML Stack

## Consumer Projects

| Project | Consumes | Role | Integration Level |
|---------|----------|------|-------------------|
| **aprender** | All contracts | ML algorithm layer | Level 3 (compile-time) |
| **entrenar** | Training contracts | Training & optimization | Level 3 (compile-time) |
| **realizar** | Tier 2-3 contracts | GPU inference engine | Level 3 (compile-time) |
| **trueno** | Tier 1 contracts | SIMD kernel implementations | Level 3 (compile-time) |
| **certeza** | QA gates from all contracts | Quality enforcement | Level 1 (binding) |
| **probar** | Proof obligations from all contracts | Property-based testing | Level 2 (wired tests) |
| **Kani** | Proof obligations from all contracts | Bounded model checking | Level 4 (Kani proofs) |
| **pmat** | Contract metadata | Code quality annotations | Level 0 (YAML-only) |

## Compile-Time Enforcement (Level 3)

All four primary consumer crates enforce contract bindings at compile
time via `build.rs`. Each crate's build script reads its binding
registry from `../provable-contracts/contracts/<crate>/binding.yaml`
and emits `CONTRACT_*` environment variables consumed by the
`#[contract]` proc macro.

| Crate | Policy | Bindings | Coverage |
|-------|--------|----------|----------|
| **aprender** | AllImplemented | 301 | 100% |
| **entrenar** | WarnOnGaps | 96 | 84% |
| **realizar** | WarnOnGaps | 23 | 100% |
| **trueno** | AllImplemented | 22 | 100% |

**AllImplemented** — Build fails if any binding has status
`not_implemented` or `partial`. Used by crates with complete coverage.

**WarnOnGaps** — Emits `cargo:warning` for gaps but does not fail
the build. Used by crates with known gaps tracked via GitHub issues.

### Environment Variable Convention

**Binding status** (existing):
```
CONTRACT_<CONTRACT_STEM>_<EQUATION>=<status>
```
Example: `CONTRACT_SOFTMAX_KERNEL_V1_SOFTMAX=implemented`

**Precondition/postcondition assertions** (new, Phase 7):
```
CONTRACT_<STEM>_<EQ>_PRE_COUNT=2
CONTRACT_<STEM>_<EQ>_PRE_0=!logits.is_empty()
CONTRACT_<STEM>_<EQ>_PRE_1=logits.iter().all(|v| v.is_finite())
CONTRACT_<STEM>_<EQ>_POST_COUNT=1
CONTRACT_<STEM>_<EQ>_POST_0=ret.len() == logits.len()
```

**Invariant assertions** (v0.2.2+):
```
CONTRACT_<STEM>_<EQ>_INV_COUNT=2
CONTRACT_<STEM>_<EQ>_INV_0=result >= 0.0
CONTRACT_<STEM>_<EQ>_INV_1=result.is_finite()
```

The `#[contract]` proc macro reads these at compile time and injects
`debug_assert!()` calls. Variable names in YAML preconditions MUST match
the function parameter names in source code.

### Escape-Proof Pipeline (Phase 7)

```
contracts/*.yaml         →  build.rs reads PRE/POST   →  #[contract] proc macro
    (single source          (sets env vars at             (reads env vars,
     of truth)               build time)                   injects debug_assert)
        ↓                       ↓                              ↓
  Lean theorem ref       cargo:rustc-env=...           Zero cost in release
  in lean_theorem:        propagates to rustc            (debug_assert! only)
```

**Change YAML** → assertions change automatically at next build.
**Remove YAML** → `compile_error!()` (env var missing).
**Remove `#[contract]`** → `pmat comply` FAILS (CB-1203).

### Build Dependencies

Each downstream crate adds to `Cargo.toml`:

```toml
[dependencies]
provable-contracts-macros = { path = "../provable-contracts/crates/provable-contracts-macros" }

[build-dependencies]
serde = { version = "1", features = ["derive"] }
serde_yaml_ng = "0.10"
```

### Example: Wiring a Function

```yaml
# contracts/softmax-kernel-v1.yaml
equations:
  softmax:
    lean_theorem: "ProvableContracts.Theorems.Softmax.PartitionOfUnity"
    preconditions:
      - "!logits.is_empty()"
      - "logits.iter().all(|v| v.is_finite())"
    postconditions:
      - "ret.len() == logits.len()"
      - "ret.iter().all(|&v| v >= 0.0)"
```

```rust
use provable_contracts_macros::contract;

#[contract("softmax-kernel-v1", equation = "softmax")]
pub fn softmax_1d_alloc(logits: &[f32]) -> Vec<f32> {
    // Preconditions injected automatically from YAML:
    //   debug_assert!(!logits.is_empty(), "Contract [softmax] Pre-condition violated: ...");
    //   debug_assert!(logits.iter().all(|v| v.is_finite()), "...");
    // ... implementation ...
    // Postconditions checked on return value:
    //   debug_assert!(ret.len() == logits.len(), "...");
}
```

### Demos

```bash
# Trueno: full pipeline demo
cd trueno && cargo run --example contract_pipeline_demo

# Provable-contracts: lint, codegen, extraction
cd provable-contracts
cargo run --example lint
cargo run --example codegen
cargo run --example extract_pytorch
```

## Batuta Integration

Batuta orchestrates the pipeline, delegating to provable-contracts for
Phases 2-6:

```bash
# Phase 1: Extract equations from paper context
batuta oracle "softmax numerical stability" --arxiv --arxiv-live

# Phase 2: Validate contract (delegates to pv validate)
pv validate contracts/softmax-kernel-v1.yaml

# Phase 3: Generate scaffold (delegates to pv scaffold)
pv scaffold contracts/softmax-kernel-v1.yaml --output src/softmax/

# Phase 5: Run falsification suite (Level 3)
batuta falsify --contract contracts/softmax-kernel-v1.yaml

# Phase 6: Run Kani proof harnesses (Level 4)
pv verify contracts/softmax-kernel-v1.yaml
# or directly:
cargo kani --harness verify_softmax_normalization
cargo kani --harness verify_softmax_simd_parity --solver kissat

# Full status: which obligations are proven?
pv status contracts/
```

## Library Integration (Rust API)

Consumer crates (aprender, entrenar, realizar, trueno) add
provable-contracts as a dev-dependency for contract-driven testing:

```toml
[dev-dependencies]
provable-contracts = "0.1"
```

```rust
use provable_contracts::schema::parse_contract;
use provable_contracts::audit::audit_contract;

#[test]
fn test_contract_compliance() {
    let contract = parse_contract(
        Path::new("contracts/softmax-kernel-v1.yaml")
    ).unwrap();
    let report = audit_contract(&contract, Path::new("src/"));
    assert!(report.all_obligations_covered(),
        "Uncovered obligations: {:?}", report.gaps());
}
```

## EDD Recipe Integration

The `quality-edd` recipe from batuta's cookbook maps directly:

```
EDD Cycle              Provable Contracts Phase
─────────              ──────────────────
Equation        →      Phase 1 (Extract)
Failing Test    →      Phase 3 (Scaffold)
Implementation  →      Phase 4 (Implement)
Verification    →      Phase 5 (Falsify — probar, Level 3)
Falsification   →      Phase 5 (Falsify — introduce bugs, verify detection)
Proof           →      Phase 6 (Verify — Kani, Level 4)
```
