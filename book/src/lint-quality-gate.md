# Lint Quality Gate

The `pv lint` command runs three sequential quality gates across all contracts
in a directory, producing a pass/fail result with detailed findings.

## Gates

| Gate | Purpose | Pass criteria |
|------|---------|---------------|
| **validate** | Schema completeness (SCHEMA-001..013, PROVABILITY-001) | 0 errors |
| **audit** | Traceability chain (paper -> equation -> obligation -> test -> proof) | 0 blocking findings |
| **score** | 5-dimension quality score vs threshold | All contracts >= `--min-score` |

Gates run sequentially: if validation fails, audit and score are skipped.

## Basic Usage

```bash
# Lint all contracts (pass at any score)
pv lint contracts/

# Require minimum score of 0.60
pv lint contracts/ --min-score 0.60

# Include binding registry for binding-dimension scoring
pv lint contracts/ --binding contracts/aprender/binding.yaml
```

## Output Formats

### Text (default)

```bash
pv lint contracts/
```

```
pv lint -- contract quality gate
================================

Gate 1: validate.............. PASS (107 contracts, 0 errors, 9 warnings) [0ms]
Gate 2: audit................. PASS (107 contracts, 0 findings) [0ms]
Gate 3: score................. PASS (107 contracts, min=0.27, mean=0.50, threshold=0.00) [0ms]

Result: PASS (3/3 gates passed) [68ms]
```

### JSON

```bash
pv lint contracts/ --format json
```

Produces a `LintReport` JSON with `passed`, `gates[]`, `findings[]`, and
`total_duration_ms` fields.

### SARIF

```bash
pv lint contracts/ --format sarif
```

Outputs [SARIF v2.1.0](https://docs.oasis-open.org/sarif/sarif/v2.1.0/) for
integration with GitHub Code Scanning, VS Code SARIF Viewer, and other tools.

### GitHub Annotations

```bash
pv lint contracts/ --format github
```

Produces `::warning file=...::message` lines for GitHub Actions workflow
annotations that appear inline on pull requests.

## Suppression

Suppress noisy findings by rule ID, contract stem, or file pattern:

```bash
# Suppress all findings from a specific rule
pv lint contracts/ --suppress-rule PV-AUD-003

# Suppress findings for specific contracts
pv lint contracts/ --suppress arch-constraints-v1,tensor-names-v1

# Suppress by file path pattern
pv lint contracts/ --suppress-file registry
```

Suppressed findings are still counted but shown as "(N suppressed)" and do
not affect the pass/fail result.

## Strict Mode

Promote all warnings to errors:

```bash
pv lint contracts/ --strict
```

In strict mode, any warning-severity finding becomes an error, which can cause
gates to fail that would otherwise pass.

## Severity Filter

Show only findings at or above a threshold:

```bash
pv lint contracts/ --severity error
```

## Rule Overrides

Change the severity of specific rules:

```bash
pv lint contracts/ --rule PV-AUD-003=error --rule PV-SCR-001=warning
```

## Configuration File

Store lint settings in `.pv.toml` or `pv.toml`:

```toml
[lint]
min_score = 0.50
strict = false
severity = "warning"

[lint.suppress]
rules = ["PV-AUD-003"]
files = ["registry"]

[lint.rules]
PV-SCR-001 = "warning"

[output]
format = "text"
```

## Diff-Aware Mode

Only lint contracts that changed since a git ref:

```bash
pv lint contracts/ --diff HEAD~5
pv lint contracts/ --diff main
```

## Content-Addressable Cache

Lint results are cached in `.pv/cache/lint/` using a hash of
(YAML content + rule config). Unchanged contracts are not re-evaluated.

```bash
# Show cache hit/miss statistics
pv lint contracts/ --cache-stats

# Disable cache (force re-evaluation)
pv lint contracts/ --no-cache
```

Example cache output:
```
Cache: 107 total, 105 hits, 2 misses (98% hit rate)
```

## Trend Tracking

Record timestamped quality snapshots in `.pv/trend/` for drift detection:

```bash
# Record a snapshot
pv lint contracts/ --trend

# View trend history
pv lint contracts/ --show-trend
```

Example trend output:
```
Date                 Score  Errors  Warnings  Result
------------------------------------------------------------
2026-03-07T18:35:16  0.50   0       12        PASS
2026-03-07T18:30:34  0.50   0       12        PASS
Trend: +0.000 score over 2 snapshots (stable)
```

When the mean score drops more than 0.05 from the rolling average (7-snapshot
window), a drift warning is emitted.

## Programmatic Usage

```rust
use provable_contracts::lint::{LintConfig, run_lint};
use std::path::Path;

let config = LintConfig::new(Path::new("contracts"), None, 0.50);
let report = run_lint(&config);

println!("Passed: {}", report.passed);
println!("Gates: {}", report.gates.len());
println!("Findings: {}", report.findings.len());
```

See also: `cargo run --example lint -- contracts/`

## CI Integration

### GitHub Actions

```yaml
- name: Contract quality gate
  run: pv lint contracts/ --min-score 0.50 --format sarif > lint.sarif

- name: Upload SARIF
  uses: github/codeql-action/upload-sarif@v3
  with:
    sarif_file: lint.sarif
```

### Pre-commit Hook

```bash
#!/bin/sh
pv lint contracts/ --min-score 0.50 --diff HEAD --strict
```

## Rule Catalog

| Rule ID | Severity | Description |
|---------|----------|-------------|
| PV-VAL-001 | Error | Schema validation error (parse failure or missing section) |
| PV-VAL-004 | Error | Empty equation formula |
| PV-VAL-005 | Error | Empty proof obligation property |
| PV-VAL-006 | Warning | Duplicate formal predicate in proof obligations |
| PV-AUD-001 | Warning | Obligation without falsification test |
| PV-AUD-002 | Info | Missing paper reference |
| PV-AUD-003 | Warning | Equations defined but no proof obligations |
| PV-AUD-004 | Warning | Equation without domain specification |
| PV-AUD-005 | Warning | Missing tolerance for numerical obligation |
| PV-SCR-001 | Error | Contract score below threshold |
| PV-PRV-001 | Error | Kernel contract without Kani harnesses |
| PV-PRV-002 | Error | Kernel contract without falsification tests |
| PV-TRD-001 | Warning | Mean score dropped >5% from rolling average |
