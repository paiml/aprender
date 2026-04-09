# Sub-spec: Lint Quality Gates (Part 2)

Version: 1.0
Status: proposed
Date: 2026-04-09

See also [lint.md](lint.md) (Sections 1-8: Design Goals, SARIF, Diff-Aware, Suppression, Severity, Auto-Fix, PR Annotations, Watch Mode).

---

## 9. Trend / History Tracking

### Motivation

Quality gates answer "is it good enough now?" but not "is it getting
better?" Trend tracking answers the second question by recording
lint results over time and detecting quality drift.

**Reference:** Molnar, A. & Motogna, S. (2024). "Versioned Analysis
of Software Quality Indicators and Self-Admitted Technical Debt."
arXiv:2407.15967 — longitudinal quality metrics and SATD evolution.

### Storage

```
.pv/trend/
+-- 2026-03-07T14:30:00.json    # Timestamped snapshots
+-- 2026-03-06T09:15:00.json
+-- ...
```

Each snapshot:

```json
{
  "timestamp": "2026-03-07T14:30:00Z",
  "commit": "abc123",
  "total_contracts": 165,
  "errors": 0,
  "warnings": 12,
  "mean_score": 0.78,
  "grade_distribution": { "A": 45, "B": 60, "C": 40, "D": 15, "F": 5 },
  "unproven_obligations": 23,
  "binding_gaps": 6,
  "findings_by_rule": { "PV-AUD-001": 8, "PV-AUD-002": 4 }
}
```

### Usage

```bash
# Record current state
pv lint contracts/ --trend

# Show trend (last 30 snapshots)
pv lint contracts/ --trend --show
# Output:
# Date        Score  Errors  Warnings  Grade
# 2026-03-07  0.78   0       12        B
# 2026-03-06  0.76   1       14        B
# 2026-03-01  0.71   3       18        C
# Trend: +0.07 score over 7 days (improving)

# JSON trend for CI dashboards
pv lint contracts/ --trend --show -f json
```

### Drift Detection

If mean score drops >5% from the 7-day rolling average, `--trend`
emits a `PV-TRD-001` warning. This catches gradual quality erosion
that individual per-contract gates miss.

---

## 10. Caching / Incremental Analysis

### Motivation

Full lint of 165 contracts takes ~2s. As the registry grows toward
1000+, incremental analysis via content-addressable caching keeps
lint O(changed) rather than O(total).

**Reference:** Singh, G. et al. (2022). "Interactive Abstract
Interpretation." arXiv:2209.10445 — incremental abstract
interpretation with fixpoint caching.

### Cache Design

```
.pv/cache/
+-- lint/
    +-- <blake3-hash>.json    # Per-contract lint result
```

**Cache key:** BLAKE3 hash of (contract YAML content + binding.yaml
content + rule config). Cache is automatically invalidated when any
input changes.

### Usage

```bash
# Normal lint (uses cache)
pv lint contracts/

# Force full re-lint (bypass cache)
pv lint contracts/ --no-cache

# Show cache stats
pv lint contracts/ --cache-stats
# Output: 165 contracts, 162 cached, 3 re-linted (1.8% work)
```

### Invalidation Rules

| Change | Invalidation |
|---|---|
| Contract YAML modified | That contract |
| binding.yaml modified | All contracts with bindings |
| `.pv.toml` rules changed | All contracts |
| `--no-cache` flag | All contracts |
| Transitive dependency changed | Dependent contracts |

---

## 11. Configuration File (`.pv.toml`)

### Motivation

CLI flags are ephemeral. A checked-in config file makes lint
behavior reproducible across CI, local dev, and team members.

### Location

`pv lint` searches for `.pv.toml` in:
1. Current directory
2. Repository root (git rev-parse --show-toplevel)
3. `$HOME/.config/pv/config.toml` (user default)

CLI flags override config file values.

### Schema

```toml
[lint]
min_score = 0.60
severity = "warning"          # Minimum severity to report
strict = false                # Promote warnings to errors
contracts_dir = "contracts/"
binding = "contracts/aprender/binding.yaml"

[lint.rules]
PV-VAL-001 = "error"
PV-AUD-001 = "warning"
PV-AUD-002 = "info"
PV-SCR-001 = "error"
PV-PRV-001 = "error"

[lint.suppress]
findings = ["SM-INV-001"]
rules = ["PV-AUD-002"]
files = ["contracts/arch-constraints-v1.yaml"]

[lint.diff]
base_ref = "main"             # Default base for --diff

[lint.trend]
enabled = true
retention_days = 90
drift_threshold = 0.05

[lint.cache]
enabled = true
dir = ".pv/cache"

[output]
format = "text"               # text, json, sarif, markdown, github
color = "auto"                # auto, always, never
```

---

## 12. Rule Catalog

All `pv lint` rules follow the pattern `PV-<CATEGORY>-NNN`.

### Validation Rules (PV-VAL)

| Rule | Severity | Description |
|---|---|---|
| PV-VAL-001 | error | Missing required field |
| PV-VAL-002 | error | Missing `metadata.version` |
| PV-VAL-003 | warning | Missing `metadata.created` |
| PV-VAL-004 | error | Invalid cross-reference (obligation -> test) |
| PV-VAL-005 | error | Duplicate obligation/test ID |
| PV-VAL-006 | warning | Unreachable test (no obligation references it) |

### Audit Rules (PV-AUD)

| Rule | Severity | Description |
|---|---|---|
| PV-AUD-001 | warning | Obligation without falsification test |
| PV-AUD-002 | info | Missing paper reference |
| PV-AUD-003 | warning | Obligation ID not referenced by any test |
| PV-AUD-004 | warning | Equation without domain specification |
| PV-AUD-005 | warning | Missing tolerance for numerical obligation |

### Score Rules (PV-SCR)

| Rule | Severity | Description |
|---|---|---|
| PV-SCR-001 | error | Composite score below `--min-score` |
| PV-SCR-002 | warning | Missing binding entry for equation |
| PV-SCR-003 | info | Kani coverage below 50% |
| PV-SCR-004 | info | Lean coverage at 0% |

### Provability Rules (PV-PRV)

| Rule | Severity | Description |
|---|---|---|
| PV-PRV-001 | error | Kernel contract without Kani harnesses |
| PV-PRV-002 | error | Kernel contract without falsification tests |
| PV-PRV-003 | warning | Kani harness count < obligation count |
| PV-PRV-004 | info | No Lean theorems (L5 not attempted) |

### Trend Rules (PV-TRD)

| Rule | Severity | Description |
|---|---|---|
| PV-TRD-001 | warning | Mean score dropped >5% from 7-day rolling avg |
| PV-TRD-002 | info | Error count increased from previous snapshot |

---

## 13. Implementation Plan

### Phase 1: SARIF + Config [Priority: Highest]

- SARIF v2.1.0 output format (`-f sarif`)
- `.pv.toml` config file parsing
- Rule catalog with configurable severity
- GitHub Actions workflow command format (`-f github`)

### Phase 2: Diff-Aware + Caching

- `--diff <base_ref>` with transitive dependent expansion
- BLAKE3 content-addressable lint cache
- `--no-cache` and `--cache-stats`

### Phase 3: Suppression + Baseline

- `--baseline <sarif>` mode
- `--suppress` / `--suppress-rule` / `--suppress-file`
- `.pv/suppressions.yaml` with expiry
- YAML inline `# pv-lint-suppress` comments

### Phase 4: Auto-Fix + Suggestions

- `--suggest` (dry run)
- `--fix` (apply changes)
- SARIF `fix` objects for deterministic repairs

### Phase 5: Trend + Watch

- `--trend` snapshot recording
- `--trend --show` historical display
- `--watch` with inotify/notify debounced re-lint
- PV-TRD drift detection rules

---

## 14. References

### Standards

1. OASIS (2020). *Static Analysis Results Interchange Format (SARIF)
   Version 2.1.0.* docs.oasis-open.org/sarif/sarif/v2.1.0

### Toolchain Integration

2. Feist, J. et al. (2024). "Integrating Static Code Analysis
   Toolchains." arXiv:2403.05986

3. Nachman, L. et al. (2025). "Dealing with SonarQube Cloud:
   Investigating the Integration of a Cloud-Based Static Analysis
   Service." arXiv:2508.18816

### Auto-Fix and Repair

4. Yang, J. et al. (2025). "CodeCureAgent: Automatic Classification
   and Repair of Static Analysis Warnings." arXiv:2509.11787

5. Shestov, A. et al. (2025). "Augmenting LLMs with Static Code
   Analysis for Automated Code Quality Improvements."
   arXiv:2506.10330

### Quality Metrics

6. Molnar, A. & Motogna, S. (2024). "Versioned Analysis of Software
   Quality Indicators and Self-Admitted Technical Debt."
   arXiv:2407.15967

### Incremental Analysis

7. Singh, G. et al. (2022). "Interactive Abstract Interpretation:
   Reanalyzing Whole Programs for Cheap." arXiv:2209.10445

### Formal Contracts

8. Li, Y. et al. (2025). "Do Large Language Models Respect
   Contracts?" arXiv:2510.12047

9. Bruni, R. et al. (2026). "Agent Behavioral Contracts."
   arXiv:2602.22302
