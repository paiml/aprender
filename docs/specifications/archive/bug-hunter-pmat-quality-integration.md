# Bug-Hunter PMAT Quality Integration Specification

**Status:** Implemented
**Version:** 1.0
**Authors:** PAIML Engineering
**Date:** 2025-05-20
**Document ID:** BATUTA-BH-PMAT-001
**Refs:** BH-21, BH-22, BH-23, BH-24, BH-25

---

## 1. Executive Summary

This specification defines the integration of PMAT function-level quality metrics into the bug-hunter subsystem. Five capabilities are added: quality-weighted suspiciousness scoring (BH-21), smart target scoping (BH-22), SATD-enriched findings (BH-23), regression risk scoring (BH-24), and spec claim quality gates (BH-25).

The integration enables bug-hunter to leverage PMAT's TDG (Technical Debt Grade) scores, cyclomatic complexity, and SATD (Self-Admitted Technical Debt) counts to improve fault localization accuracy and provide actionable quality context alongside bug findings.

---

## 2. Problem Statement

Bug-hunter's fault localization uses a 4-channel SBFL (Spectrum-Based Fault Localization) system combining spectrum, mutation, static analysis, and semantic signals. This approach treats all code locations equally regardless of their quality characteristics.

Research shows that code with higher technical debt and complexity is statistically more likely to contain defects (Zazworka et al., 2011). By incorporating PMAT quality metrics as a 5th channel, bug-hunter can:

1. Boost suspiciousness of low-quality code (more likely buggy)
2. Reduce suspiciousness of high-quality code (less likely buggy)
3. Surface self-admitted technical debt as potential bug locations
4. Quantify regression risk for each finding
5. Gate spec verification claims against implementation quality

---

## 3. Architecture

### 3.1 Module Structure

```
src/bug_hunter/
├── mod.rs              # hunt(), analyze_common_patterns(), hunt_with_spec()
├── types.rs            # Finding, EvidenceKind, ChannelWeights, HuntConfig
├── localization.rs     # ScoredLocation, 5-channel combine
├── pmat_quality.rs     # NEW: PMAT quality integration
└── ...
```

### 3.2 Data Flow

```
CLI flags (--pmat-quality, --quality-weight, --pmat-scope, --pmat-query)
         │
         ▼
    HuntConfig
         │
         ├──[BH-22]──► scope_targets_by_quality() ──► filtered target list
         │
         ▼
    hunt() / analyze_common_patterns()
         │
         ├──[BH-21]──► build_quality_index() ──► PmatQualityIndex
         │                  │
         │                  ├──► apply_quality_weights()  ──► adjusted suspiciousness
         │                  └──► apply_regression_risk()  ──► regression_risk field
         │
         ├──[BH-23]──► generate_satd_findings() ──► BH-SATD-XXXX findings
         │
         └──[BH-25]──► hunt_with_spec() quality gates ──► BH-QGATE findings
```

### 3.3 Cross-Crate Design

The `bug_hunter` module lives in the library crate (`lib.rs`) while `cli::oracle::pmat_query` lives in the binary crate (`main.rs`). To avoid cross-crate dependencies:

- `pmat_quality.rs` defines a local `PmatFunctionInfo` struct
- PMAT is invoked directly via `tools::run_tool("pmat", ...)` (available in lib crate)
- JSON output is parsed locally with serde

---

## 4. Feature Specifications

### 4.1 BH-21: Quality-Weighted Suspiciousness

**CLI:** `--pmat-quality` `--quality-weight <f64>`

Adjusts finding suspiciousness based on TDG score:

```
adjusted = base * (1 + weight * (0.5 - tdg / 100))
```

- TDG = 50 (C grade): no adjustment (centered)
- TDG > 50 (A/B grade): suspiciousness reduced
- TDG < 50 (D/F grade): suspiciousness boosted
- Result clamped to [0.0, 1.0]

Default weight: 0.5

Evidence is attached as `EvidenceKind::QualityMetrics` with description including TDG score, grade, complexity, and SATD count.

### 4.2 BH-22: Smart Target Scoping

**CLI:** `--pmat-scope` `--pmat-query <query>`

When `--pmat-scope` is enabled, `scope_targets_by_quality()` queries PMAT for functions matching the query and returns file paths sorted by worst quality first. This focuses analysis on the most debt-laden code.

### 4.3 BH-23: SATD-Enriched Findings

**CLI:** `--pmat-quality` (implied when PMAT available and `pmat_satd` config is true)

When PMAT is available during `analyze_common_patterns()`:
1. Generates `BH-SATD-XXXX` findings from PMAT's SATD data
2. Skips redundant regex-based TODO/FIXME/HACK/XXX pattern scanning
3. Retains unwrap/unsafe/transmute/panic patterns (not covered by SATD)

SATD finding severity scales with count: >3 = High, >1 = Medium, else Low.

### 4.4 BH-24: Regression Risk Scoring

**CLI:** `--pmat-quality` (automatic when quality index available)

Computes a regression risk score for each finding:

```
risk = 0.5 * (1 - tdg/100) + 0.3 * min(complexity/50, 1) + 0.2 * min(satd/5, 1)
```

- TDG weight: 50% (dominant signal)
- Complexity weight: 30% (normalized to 50)
- SATD weight: 20% (normalized to 5)
- Result clamped to [0.0, 1.0]

Stored in `Finding.regression_risk` and displayed in text/markdown output.

### 4.5 BH-25: Spec Claim Quality Gates

**CLI:** `--pmat-quality` with `bug-hunter spec --spec <file>`

During `hunt_with_spec()`, after mapping findings to spec claims:
1. Queries PMAT for functions implementing each claim
2. If any implementing function has grade D/F or complexity > 20:
   - Generates a `BH-QGATE` finding with High severity
   - Downgrades claim status from Verified to Warning
   - Evidence includes the failing function's quality metrics

---

## 5. Type Changes

### 5.1 ChannelWeights (5-Channel)

| Channel | Default Weight | Purpose |
|---------|---------------|---------|
| spectrum | 0.30 | SBFL spectrum coverage |
| mutation | 0.25 | Mutation testing signals |
| static_analysis | 0.20 | Static analyzer findings |
| semantic | 0.15 | Semantic similarity |
| quality | 0.10 | PMAT quality metrics |

`combine(spectrum, mutation, static_a, semantic, quality)` computes weighted sum after normalization.

### 5.2 Finding Extensions

- `regression_risk: Option<f64>` — BH-24 regression risk (0.0 to 1.0)
- `with_regression_risk(risk: f64)` — builder method

### 5.3 EvidenceKind Extension

- `QualityMetrics` variant — attached by `apply_quality_weights()`

### 5.4 HuntConfig Extensions

| Field | Type | Default | Feature |
|-------|------|---------|---------|
| `use_pmat_quality` | `bool` | `false` | BH-21 |
| `quality_weight` | `f64` | `0.5` | BH-21 |
| `pmat_scope` | `bool` | `false` | BH-22 |
| `pmat_satd` | `bool` | `true` | BH-23 |
| `pmat_query` | `Option<String>` | `None` | BH-22 |

---

## 6. PmatQualityIndex

```rust
pub type PmatQualityIndex = HashMap<PathBuf, Vec<PmatFunctionInfo>>;
```

Built by `build_quality_index()` which invokes `pmat query --format json` and groups results by file path. Functions within each file are sorted by `start_line` for efficient span lookup.

`lookup_quality(index, file, line)` finds the function containing the given line (span match) or falls back to the nearest function by `start_line`.

---

## 7. Testing

18 unit tests in `src/bug_hunter/pmat_quality.rs::tests`:

| Test | Coverage |
|------|----------|
| `test_index_from_results_*` | Index construction and grouping |
| `test_lookup_quality_*` | Span match, nearest fallback, missing file |
| `test_quality_adjusted_*` | Formula correctness, clamping, edge cases |
| `test_regression_risk_*` | High/low quality, clamping |
| `test_apply_quality_weights_*` | In-place adjustment + evidence |
| `test_generate_satd_findings_*` | Finding generation, severity scaling |
| `test_pmat_available_*` | Tool detection |

Existing tests updated:
- `test_channel_weights` — 5-channel combine
- `test_scored_location_new` — quality_score field
- `test_handle_analyze_command` — new CLI flags

---

## 8. CLI Usage

```bash
# Quality-weighted analysis
batuta bug-hunter analyze --pmat-quality

# Custom quality weight
batuta bug-hunter analyze --pmat-quality --quality-weight 0.7

# Scoped analysis (worst quality first)
batuta bug-hunter analyze --pmat-scope --pmat-query "error"

# Spec verification with quality gates
batuta bug-hunter spec --spec docs/spec.md --pmat-quality

# Ensemble with quality
batuta bug-hunter ensemble --pmat-quality --quality-weight 0.3
```
