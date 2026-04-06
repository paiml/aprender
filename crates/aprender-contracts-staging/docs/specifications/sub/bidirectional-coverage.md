# Sub-spec: Bidirectional Contract Coverage

**Parent:** [pv-spec.md](../pv-spec.md) Section 15.7

> **Implementation Status (2026-04-03):**
> - ✅ `pv coverage --reverse` — scans pub fns, diffs against binding.yaml
> - ✅ Gate 7: reverse-coverage lint gate (50% threshold, hardcoded)
> - ✅ `pv infer` — suggests bindings and new contracts for unmatched fns
> - ✅ Auto-exemption for trivial functions (200+ patterns)
> - ❌ `#[must_contract]` crate attribute — NOT IMPLEMENTED
> - ❌ `pv annotate` auto-insert command — NOT IMPLEMENTED
> - ⚠️ `reverse_coverage` in CodebaseScore hardcoded to 0.0
> - ⚠️ Metric includes exempt fns: `(bound + exempt) / total` not `bound / total`

---

## Problem: Whack-a-Mole Enforcement

Current enforcement is **unidirectional**: binding → implementation.

```
binding.yaml says softmax → aprender::nn::functional::softmax
pv lint checks: does that function exist?  ✓
```

But nobody checks the **reverse**: when a developer adds `pub fn avg_pool`
to aprender, nothing tells them to create a contract or binding. New code
escapes the contract system silently.

This is analogous to:
- Unit tests without coverage: you don't know what's untested
- Types without the borrow checker: you don't know what's unsafe

### Five-Whys

1. **Why do new functions escape?** No reverse check exists
2. **Why no reverse check?** `pv lint` only reads binding.yaml forward
3. **Why only forward?** Original design assumed contracts come first (contract-first workflow)
4. **Why doesn't contract-first work?** Developers often implement first, contract later
5. **Root cause:** Missing bidirectional coverage — no tool infers "this function needs a contract"

---

## Solution: Three Inference Mechanisms

### Mechanism 1: `pv coverage --reverse` (Static API Diff)

Scan a crate's public API, diff against binding.yaml, report unbound functions.

```bash
pv coverage --reverse ../aprender contracts/aprender/binding.yaml
```

Output:
```
Unbound public functions (23/324):
  aprender::nn::conv::Conv2d::forward      — no binding
  aprender::nn::pool::MaxPool2d::forward    — no binding
  aprender::optim::SGD::step               — no binding

Auto-matched (suggest bindings):
  aprender::nn::conv::Conv2d::forward → conv1d-kernel-v1.yaml/conv1d  [95%]
  aprender::nn::norm::BatchNorm::forward → batchnorm-kernel-v1.yaml   [92%]

Contract coverage: 301/324 (93%)
```

**Implementation:**

1. Parse crate's `pub fn` signatures via `syn` or `cargo doc --json`
2. Parse binding.yaml for all `function:` entries
3. Diff: functions in crate but not in binding = unbound
4. Match unbound functions against existing contract equations by name similarity
5. Report coverage percentage and suggestions

**Integration points:**
- `crates/provable-contracts/src/coverage.rs` — add `reverse_coverage()`
- `crates/provable-contracts-cli/src/commands/coverage.rs` — add `--reverse` flag
- `pv lint` Gate 7 — warn when reverse coverage < threshold

### Mechanism 2: `#[must_contract]` Crate Attribute

Compile-time enforcement via a crate-level attribute that warns when public
functions lack `#[contract]` annotations.

```rust
// lib.rs
#![cfg_attr(debug_assertions, warn(provable_contracts::must_contract))]

// Any pub fn without #[contract] or #[contract(exempt)] triggers:
// warning: `Conv2d::forward` has no #[contract] annotation
//   help: add #[contract("conv1d-kernel-v1", equation = "conv1d")]
//   help: or #[contract(exempt = "internal helper, not a kernel")]
```

**Implementation:**
- Extend `provable-contracts-macros` with a `must_contract` lint
- Walk all `pub fn` items in the crate
- Check each for `#[contract(...)]` or `#[contract(exempt)]`
- Emit `warning` for undecorated functions

**Exemption patterns:**
```rust
#[contract(exempt = "test helper")]        // Explicit exemption with reason
#[contract(exempt)]                         // Blanket exemption
mod internal { ... }                        // Private modules auto-exempt
```

### Mechanism 3: `pv infer` (Semantic Matching)

Automatically suggest contracts for unbound functions using:
- Function name → equation name similarity (Levenshtein / BM25)
- Parameter types → domain inference (`&[f32]` → numerical kernel)
- Module path → tier inference (`nn::` → Tier 1-2, `optim::` → Tier 4)
- Return type → codomain inference

```bash
pv infer ../aprender

Inferred bindings (high confidence):
  Conv2d::forward(input: &Tensor) → conv1d-kernel-v1/conv1d       [98%]
  BatchNorm::forward(...)         → batchnorm-kernel-v1/batchnorm  [95%]

Missing contracts (no matching contract exists):
  MaxPool2d::forward  → suggest: maxpool-kernel-v1.yaml  [CREATE]
  AvgPool2d::forward  → suggest: avgpool-kernel-v1.yaml  [CREATE]
```

**Implementation:**
- `crates/provable-contracts/src/infer.rs` — new module
- Uses existing `query::ContractIndex` BM25 engine for matching
- Outputs suggested binding.yaml entries and contract YAML stubs

---

## Enforcement Ladder

| Level | Mechanism | When | Cost |
|---|---|---|---|
| L0 | Manual | Developer remembers | Free (unreliable) |
| L1 | `pv coverage --reverse` | CI check | Low (static analysis) |
| L2 | `pv lint` Gate 7 | CI gate | Low (blocks merge) |
| L3 | `#[must_contract]` | Compile time | Medium (proc macro) |
| L4 | `pv infer` | Contract authoring | Medium (BM25 matching) |
| L5 | `pv annotate` | Auto-patch source | High (source rewriting) |

Recommended rollout: L1 → L2 → L4 → L3 → L5.

---

## Metrics

| Metric | Definition | Target |
|---|---|---|
| Forward coverage | bindings with status=implemented / total bindings | 100% |
| Reverse coverage | pub fns with bindings / total pub fns | > 80% |
| Macro coverage | pub fns with `#[contract]` / pub fns with bindings | > 50% |
| Inference accuracy | auto-suggested bindings accepted / total suggested | > 70% |

---

## CLI Commands

| Command | Description |
|---|---|
| `pv coverage --reverse <crate> <binding>` | Report unbound public functions |
| `pv infer <crate>` | Suggest contracts + bindings for unbound functions |
| `pv annotate <crate> <binding>` | Auto-insert `#[contract]` annotations |
| `pv lint` Gate 7 | Fail if reverse coverage below threshold |

---

## Integration with Existing Gates

```
Gate 1: validate     — YAML schema valid
Gate 2: audit        — traceability chain complete
Gate 3: score        — quality grade ≥ threshold
Gate 4: verify       — test references exist
Gate 5: enforce      — pre/post/lean in YAML
Gate 6: build-rs     — consumer crates have build.rs + #[contract]
Gate 7: reverse-cov  — unbound pub fns below threshold (NEW)
```
