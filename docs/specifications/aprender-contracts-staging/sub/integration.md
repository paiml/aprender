# Sub-spec: Stack Integration

Version: 1.0
Status: proposed
Date: 2026-04-09

**Parent:** [pv-spec.md](../pv-spec.md) Section 11

---

## 1. Consumer Projects

### aprender (Primary Consumer)

**Relationship:** Library dependency + proc macro annotations.

```toml
# aprender/Cargo.toml
[dependencies]
provable-contracts-macros = "0.1"

[dev-dependencies]
provable-contracts = "0.1"
```

**Integration points:**
- 38 `#[contract(...)]` annotations across 22 files
- `build.rs` reads `binding.yaml`, sets `CONTRACT_*` env vars
- `AllImplemented` policy: unbound equations fail the build
- `ALLOWED_GAPS` whitelist for known unimplemented contracts

**Code example:**
```rust
#[provable_contracts_macros::contract("rmsnorm-kernel-v1", equation = "rmsnorm")]
pub fn rms_norm(input: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    // Contract enforcement: build.rs verifies this binding exists
}
```

**Makefile integration:**
```makefile
PV_DIR := ../provable-contracts
# pv validate, pv audit run as part of `make lint`
```

### trueno (SIMD/CUDA Kernels)

**Relationship:** YAML contract consumer only (no Cargo dependency).
Implements Tier 1-2 kernel contracts via KAIZEN ticket references.

KAIZEN tickets flow: contract in provable-contracts -> kernel in trueno.

| Contract | trueno Implementation |
|---|---|
| C-XENT-002 | fused cross-entropy loss kernel |
| C-XENT-003 | in-place fused cross-entropy |
| KAIZEN-049 | GPU L2 norm reduction |
| KAIZEN-050 | fused cross-entropy + softmax backward |
| KAIZEN-052 | in-place xent (eliminated grad_ptr) |

### entrenar (Training Pipeline)

**Relationship:** YAML contract consumer only (no Cargo dependency).
KAIZEN performance contracts drive optimization.

The KAIZEN contract-first workflow is most active here:

```
1. Profile reveals bottleneck (e.g., embed_bwd 145ms)
2. Write contract (embed-grad-zero-copy-v1.yaml)
3. Implement optimization (zero-copy scatter-add)
4. Measure (embed_bwd 0.55ms, 263x speedup)
```

68 KAIZEN tickets across entrenar, 34 with provable-contracts refs.

### bashrs (SSC Pipeline)

**Relationship:** YAML contract consumer only (no Cargo dependency).
Encoder and classifier contracts.

4 SSC encoder contracts (PMAT-064), plus PV-003/PV-004 contract
bindings for encoder infrastructure.

---

## 2. The KAIZEN Workflow

### Process

KAIZEN (continuous improvement) tickets follow a contract-first pattern:

```
KAIZEN-NNN created in pmat work
  |
  v
Write contract YAML in provable-contracts
  - Define performance obligation
  - Set measurable threshold
  - Reference profiling baseline
  |
  v
Implement in target project (entrenar/trueno)
  - Commit references KAIZEN-NNN
  |
  v
Measure improvement
  - Record baseline vs result in commit message
  |
  v
Contract serves as regression gate
  - Future changes that violate obligation trigger alert
```

### Evidence: KAIZEN Velocity

| Ticket | Contract | Project | Measurable Result |
|---|---|---|---|
| KAIZEN-040 | Matrix::transpose BLIS AVX2 | trueno | BLIS delegation |
| KAIZEN-041 | vecmat BLIS gemv | trueno | Eliminated 3x rows temp allocs |
| KAIZEN-043 | LoRA grad zero buffer reuse | entrenar | Zero-alloc grad accumulation |
| KAIZEN-044 | Host buffer pre-allocation | entrenar | D2H pre-alloc |
| KAIZEN-045 | Backward scratch pre-alloc | entrenar | Scratch reuse |
| KAIZEN-048 | embed-grad-zero-copy | entrenar | **145ms -> 0.55ms (263x)** |
| KAIZEN-049 | GPU L2 norm reduction | trueno | GPU-side norm |
| KAIZEN-050 | Fused GPU cross-entropy | trueno | Fused kernel |
| KAIZEN-052 | In-place fused xent | trueno | **Eliminated 77.8MB/step** |
| KAIZEN-058 | Lazy PTX generation | entrenar | Skip cached modules |
| KAIZEN-064 | Fused causal LM xent | entrenar | GPU causal loss |
| KAIZEN-066 | GPU RMSNorm forward | entrenar | **Eliminated D2H round-trip** |
| KAIZEN-068 | GPU-resident embed | entrenar | Eliminated ~1.45GB H2D/step |

### Timeline Correlation

Provable-contracts KAIZEN commits and entrenar implementations happen
on the same day — same-day contract-to-code:

```
2026-03-03: 15 contracts written in provable-contracts
2026-03-04: 25 contracts + 31 KAIZEN implementations in entrenar
```

---

## 3. Batuta Integration

Batuta orchestrates the pipeline:

```bash
# Phase 1: Extract from papers
batuta oracle "softmax numerical stability" --arxiv --arxiv-live

# Phase 2-3: Validate + scaffold
pv validate contracts/softmax-kernel-v1.yaml
pv scaffold contracts/softmax-kernel-v1.yaml --output src/softmax/

# Phase 5: Falsify
batuta falsify --contract contracts/softmax-kernel-v1.yaml

# Phase 6: Verify (run Kani directly — no `pv verify` command yet)
cargo kani --harness verify_softmax_normalization
```

---

## 4. certeza Integration

Every contract defines a QA gate consumed by certeza:

```yaml
qa_gate:
  id: "F-SOFTMAX-001"
  name: "Softmax Kernel Contract"
  checks:
    - "All normalization tests pass"
    - "Shift invariance holds"
    - "SIMD matches scalar"
  pass_criteria: "All falsification tests pass"
  falsification: "Introduce off-by-one -- gate must catch"
```

certeza validates: test coverage >= 95%, mutation coverage >= 80%,
all QA gates pass.

---

## 5. pmat Integration

pmat uses contract metadata for code quality annotations:

```bash
pmat query "softmax" --include-source
# Shows TDG grade + contract binding status
```

Contract scores from `pv score` can be ingested by pmat for enriched
quality reporting.

---

## 6. Proc Macro Enforcement Chain

```
binding.yaml
  |
  v (read by)
build.rs (in aprender)
  |
  v (sets)
CONTRACT_SOFTMAX_KERNEL_V1_SOFTMAX=implemented
  |
  v (read by)
#[contract("softmax-kernel-v1", equation = "softmax")]
  |
  v (compile-time)
option_env!("CONTRACT_SOFTMAX_KERNEL_V1_SOFTMAX")
  |
  v (if missing + AllImplemented + not in ALLOWED_GAPS)
BUILD FAILURE
```

### Enforcement Levels

| Context | Enforcement |
|---|---|
| Local build (sibling repo present) | build.rs reads binding.yaml, sets env vars |
| crates.io build (no sibling) | option_env! returns None, graceful degrade |
| CI with AllImplemented policy | Missing binding = hard build failure |
| ALLOWED_GAPS whitelist | Known gaps exempt from failure |

---

## 7. Cross-Project Statistics

| Metric | Value |
|---|---|
| Total stack Rust LoC governed | ~900K |
| provable-contracts LoC | ~37K (1.7% of stack) |
| KAIZEN tickets with contracts | 32+ |
| Contract annotations in aprender | 38 across 22 files |
| Binding entries | 301 (295 implemented = 98.0%) |
| Same-day contract-to-code ratio | >80% of KAIZEN tickets |
