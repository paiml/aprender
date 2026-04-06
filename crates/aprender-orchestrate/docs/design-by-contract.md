# Design by Contract in Batuta

Batuta enforces Design by Contract (DbC) across the PAIML stack through three
mechanisms: contract gap analysis, stack version drift detection, and
cross-repo dependency validation.

## Contract Gap Analysis (BH-26)

The bug hunter's `contracts` module (`src/bug_hunter/contracts.rs`) analyzes
`provable-contracts` YAML binding registries to surface verification gaps.

### How It Works

1. **Discovery** -- `discover_contracts_dir()` locates the contracts directory.
   It checks an explicit path first, then auto-discovers
   `../provable-contracts/contracts/` relative to the project root.

2. **Binding gap analysis** -- `analyze_contract_gaps()` reads every
   `binding.yaml` registry and produces `BH-CONTRACT-NNNN` findings for
   bindings whose `status` field is `not_implemented` (severity High,
   suspiciousness 0.8) or `partial` (severity Medium, suspiciousness 0.6).
   Bindings with status `implemented` are silently accepted.

3. **Unbound contract detection** -- Any contract YAML file that is not
   referenced by any `binding.yaml` registry produces a Medium-severity
   finding indicating the contract has no implementation binding.

4. **Proof obligation coverage** -- For each contract YAML, the ratio of
   `falsification_tests` to `proof_obligations` is computed. If fewer than
   50% of proof obligations have corresponding falsification tests, a
   Low-severity finding is emitted.

### Binding Status Semantics

| Status             | Severity | Suspiciousness | Meaning                              |
|--------------------|----------|----------------|--------------------------------------|
| `implemented`      | --       | --             | Contract fully satisfied; no finding  |
| `partial`          | Medium   | 0.6            | Binding exists but is incomplete      |
| `not_implemented`  | High     | 0.8            | No implementation exists              |
| (unbound)          | Medium   | 0.5            | Contract YAML has no binding at all   |

### Source of Truth

- Contract YAML files live in `provable-contracts/contracts/`
- Each target crate has a `binding.yaml` under a subdirectory (e.g.,
  `contracts/aprender/binding.yaml`)
- The contract spec follows the schema defined by `provable-contracts`

## Stack Version Drift Detection

The `stack::drift` module (`src/stack/drift.rs`) detects when PAIML stack
crates depend on outdated versions of sibling crates.

- `DriftChecker` queries crates.io for the latest version of each dependency
  and compares it against the version requirement in Cargo.toml
- Drift is classified as Major, Minor, or Patch severity
- Run via: `batuta stack drift`

## Cross-Repo Dependency Validation

The `stack::graph` module (`src/stack/graph.rs`) builds a dependency graph of
the entire PAIML stack and provides:

- **Cycle detection** -- `has_cycles()` verifies the dependency graph is a DAG
- **Topological ordering** -- `topological_order()` returns the correct publish
  order (e.g., trueno before aprender before realizar)
- **Release ordering** -- `release_order_for(crate_name)` returns the minimal
  set of crates that must be released first
- **Conflict detection** -- `detect_conflicts()` finds version conflicts
  across the stack

## Running the Tests

```bash
# Contract gap analysis unit tests
cargo test --lib -- contracts --features native

# Stack graph and drift tests
cargo test --lib -- stack --features native
```

## CLI Usage

```bash
# Auto-discover contracts and analyze gaps
batuta bug-hunter analyze . --contracts-auto

# Explicit contracts path
batuta bug-hunter analyze . --contracts /path/to/provable-contracts/contracts

# Filter by suspiciousness (only not_implemented at 0.8+)
batuta bug-hunter analyze . --contracts-auto --min-suspiciousness 0.7

# Stack drift detection
batuta stack drift

# Dependency graph and release ordering
batuta stack check
batuta stack release --all --bump minor --dry-run
```

## Cross-References

- `src/bug_hunter/contracts.rs` -- Contract gap analysis implementation
- `src/bug_hunter/mod.rs` -- `run_contract_gap_phase()` integration point
- `src/stack/drift.rs` -- Version drift checker
- `src/stack/graph.rs` -- Dependency graph with topological sort
- `src/stack/types.rs` -- `CrateInfo`, `DependencyInfo` domain types
- `provable-contracts/contracts/` -- Contract YAML source of truth
