# Stack Management Specification

> Parent: [batuta-spec.md](../batuta-spec.md)
> Sources: batuta-stack-spec, stack-quality-matrix-spec, stack-tree-view, batuta-stack-0.1-100-point-qa-checklist, score-a-plus-spec, book-score-spec

---

## 1. Overview

Batuta Stack Orchestration manages dependencies, coordinates releases, and enforces quality across all PAIML Rust ecosystem crates. It was motivated by the December 2025 incident where `entrenar v0.2.2` shipped with broken path dependencies to crates.io.

### Problem Solved

- Broken published crates from path dependencies
- Arrow version conflicts across crates (53.x vs 54.x)
- Manual coordination failures with no automated checks
- No unified quality enforcement across 20+ components

---

## 2. Dependency Graph

### PAIML Stack Components

| Layer | Crate | Latest | Purpose |
|-------|-------|--------|---------|
| Compute | trueno | 0.14.x | SIMD/GPU primitives |
| Compute | trueno-db | 0.3.x | GPU-first analytics DB |
| Compute | trueno-graph | 0.1.x | Graph database |
| Compute | trueno-rag | 0.1.x | RAG pipeline (BM25+vector) |
| Compression | trueno-zram-core | 0.3.x | SIMD compression |
| Distribution | repartir | 2.0.x | Distributed compute |
| ML | aprender | 0.24.x | ML algorithms, APR v2 |
| Training | entrenar | 0.5.x | Autograd, LoRA/QLoRA |
| Inference | realizar | 0.5.x | GGUF/APR inference |
| Speech | whisper-apr | 0.1.x | Whisper ASR |
| Data | alimentar | 0.2.x | Parquet/Arrow loading |
| Registry | pacha | 0.1.x | Model registry (Ed25519) |
| Tracing | renacer | 0.7.x | Syscall tracer |

### Inter-Dependencies

```
whisper-apr --> trueno, aprender, realizar
realizar ----> trueno, aprender, alimentar, pacha
aprender ----> trueno, alimentar, entrenar
entrenar ----> trueno, aprender, trueno-db, trueno-rag
repartir ----> trueno, trueno-db (checkpoint), wgpu (gpu)
```

---

## 3. Command Specifications

### 3.1 `batuta stack check` -- Dependency Health

Analyzes dependency graphs across all PAIML projects:
- Detects path dependencies that should be crates.io versions
- Identifies version conflicts before build failures
- Validates dependency alignment across the stack

```bash
batuta stack check
# Output: dependency health matrix with warnings/errors
```

### 3.2 `batuta stack release` -- Coordinated Release

Coordinates releases in topological order:
- Validates quality gates (lint, coverage) before each release
- Automates Cargo.toml updates for downstream dependencies
- Publishes crates in correct dependency order

### 3.3 `batuta stack status` -- Dashboard

TUI dashboard showing health of all stack components.

### 3.4 `batuta stack versions` -- Version Check

```bash
batuta stack versions              # Check latest crates.io versions
batuta stack versions --format json  # JSON output for tooling
```

### 3.5 `batuta stack quality` -- Quality Matrix

```bash
batuta stack quality --verify

# Output:
# Component          Rust    Repo    README   Hero     Status
# trueno            107/114  98/110   20/20    ok       A+
# aprender          109/114  96/110   19/20    ok       A+
# entrenar           89/114  82/110   15/20    --       B+
```

### 3.6 `batuta stack publish-status` -- Publish Readiness

O(1) cached check for which crates need publishing:

```bash
batuta stack publish-status         # Warm cache: <100ms
batuta stack publish-status --force  # Cold cache: ~7s
```

Cache invalidation triggers: Cargo.toml change, git HEAD move, crates.io TTL (15 min).

### 3.7 `batuta stack gate` -- CI Quality Gate

Enforces quality thresholds before merge:
- TDG grade >= B
- Test coverage >= 90%
- Zero clippy warnings
- No prohibited dependencies (e.g., cargo-tarpaulin)

---

## 4. Quality Matrix

### Scoring Dimensions

| Dimension | Tool | A+ Threshold |
|-----------|------|-------------|
| Rust Project Score | `pmat rust-project-score` | 105-114/114 |
| Repository Score | `pmat repo-score` | 95-110/110 |
| README Score | `pmat repo-score` (Category A) | 18-20/20 |
| Hero Image | `batuta stack quality` | Present & Valid |

### Quality Gates

| Gate | Enforcement Point | Threshold |
|------|-------------------|-----------|
| Formatting | Pre-commit | `cargo fmt --check` passes |
| Linting | Pre-commit | `cargo clippy -- -D warnings` |
| Unit tests | Pre-commit | All pass in < 5s |
| Full tests | Pre-push | All pass in < 5 min |
| Coverage | CI | >= 90% (95% preferred) |
| Mutation score | CI | >= 80% |
| TDG Score | Release gate | >= A grade (85/100) |

---

## 5. Stack Tree View

Visual hierarchical display of stack components:

```bash
batuta stack tree [OPTIONS]

Options:
  --format <FORMAT>    ascii | json | dot [default: ascii]
  --depth <N>          Max depth [default: unlimited]
  --filter <LAYER>     core | ml | inference | orchestration | distributed
  --health             Include health status indicators
```

### Data Model

```rust
pub struct StackTree {
    pub layers: Vec<StackLayer>,
}

pub struct StackLayer {
    pub name: String,
    pub components: Vec<StackComponent>,
}

pub struct StackComponent {
    pub name: String,
    pub version: String,
    pub health: HealthStatus,
    pub dependencies: Vec<String>,
}
```

---

## 6. 100-Point QA Checklist

### QA Tiers

| Tier | Command | Scope | Points |
|------|---------|-------|--------|
| `make qa-local` | batuta internal quality | 50 |
| `make qa-stack` | Full stack validation (CI) | 50 |

### Inspection Sections (100 Points Total)

| Section | Domain | Points | Key Checks |
|---------|--------|--------|------------|
| I | Foundation (Trueno) | 20 | SIMD detection, GPU fallback, matrix accuracy, Miri, WASM, thread safety |
| II | ML/Inference (Aprender/Realizar) | 20 | Regression exactness, GGUF loading, tokenization parity, inference latency, quantization |
| III | Transpilation (Decy/Depyler/Bashrs) | 20 | Ownership inference, type conversion, shell safety, semantic equivalence |
| IV | Orchestration (Batuta) | 20 | Pipeline phases, PMAT integration, stack health, error handling |
| V | Stack Integration | 20 | Cross-crate data flow, version compatibility, CI health, deployment |

### Toyota Way Principles in QA

| Principle | Application |
|-----------|-------------|
| **Genchi Genbutsu** | Every check requires specific command execution and verifiable result |
| **Jidoka** | Pipeline halts on first failed check |
| **Poka-Yoke** | Automated checks prevent human error in release process |
| **Kaizen** | Checklist refined after each release cycle |

---

## 7. Release Coordination

### Topological Release Order

Crates must be released bottom-up through the dependency graph:

1. trueno (no PAIML deps)
2. trueno-db, trueno-graph, trueno-rag (depend on trueno)
3. aprender, alimentar (depend on trueno)
4. entrenar (depends on trueno, aprender)
5. realizar (depends on trueno, aprender, alimentar)
6. repartir (depends on trueno)
7. whisper-apr (depends on trueno, aprender, realizar)
8. batuta (depends on all)

### Pre-Release Validation

```bash
# For each crate in topological order:
cargo fmt --check
cargo clippy -- -D warnings
cargo test
cargo publish --dry-run
# If all pass: cargo publish
```

### Toyota Way Analysis

| Principle | Application |
|-----------|-------------|
| **Muda** | Eliminate wasted releases (broken crates, version conflicts) |
| **Jidoka** | Stop release pipeline on first quality failure |
| **Just-in-Time** | Publish only when downstream crate needs the update |
| **Kaizen** | Post-release retrospective improves next cycle |
| **Heijunka** | Level the release cadence (no big-bang multi-crate releases) |
