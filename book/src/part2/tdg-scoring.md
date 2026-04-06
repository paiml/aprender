# Technical Debt Grade (TDG)

The Technical Debt Grade is a composite quality score computed by PMAT static analysis. It provides a single letter grade (A through F) that summarizes the migration readiness of the source project.

## Grading Scale

| Grade | Score Range | Meaning |
|-------|-----------|---------|
| A | 85-100 | Excellent -- clean code, low complexity, high coverage |
| B | 70-84 | Good -- minor issues, suitable for automated transpilation |
| C | 55-69 | Fair -- moderate debt, some manual intervention needed |
| D | 40-54 | Poor -- significant debt, plan for refactoring |
| F | 0-39 | Critical -- major rewrite may be more efficient than migration |

## What TDG Measures

TDG is a weighted composite of four dimensions:

1. **Cyclomatic Complexity** -- number of independent paths through functions
2. **Cognitive Complexity** -- how difficult code is for humans to understand
3. **Test Coverage** -- percentage of lines exercised by tests
4. **Code Quality** -- linting violations, dead code, duplication

## How TDG Is Computed

Batuta delegates TDG computation to the PMAT tool:

```bash
# PMAT runs complexity analysis and returns JSON
pmat analyze complexity /path/to/project --format json
```

The `analyze_quality()` function in `src/tools.rs` invokes PMAT and parses the result:

```rust
pub fn analyze_quality(path: &Path) -> Result<String> {
    let args = vec!["analyze", "complexity", &path_str, "--format", "json"];
    run_tool("pmat", &args, None)
}
```

The resulting score is stored as `tdg_score: Option<f64>` in `ProjectAnalysis`.

## CLI Usage

```bash
$ batuta analyze --tdg ./my-python-app

Technical Debt Grade
--------------------
Overall: B (78.3)

  Complexity:  72/100  (12 functions above threshold)
  Coverage:    85/100  (85% line coverage)
  Quality:     81/100  (3 clippy-equivalent warnings)
  Duplication: 75/100  (2 code clones detected)
```

## Migration Priority

TDG scores guide migration order. High-scoring modules are the best candidates for automated transpilation because they have well-defined behavior and test coverage to validate against.

| TDG | Migration Strategy |
|-----|-------------------|
| A-B | Fully automated transpilation via Depyler/Decy/Bashrs |
| C | Automated with manual review of flagged functions |
| D | Partial automation, refactor complex functions first |
| F | Consider rewrite rather than transpilation |

## Pre-commit Integration

Batuta's pre-commit hook enforces complexity thresholds to prevent TDG regression:

```bash
# Pre-commit runs on staged .rs files
pmat analyze complexity --max-cyclomatic 30 --max-cognitive 25
```

Functions exceeding these thresholds block the commit until the complexity is reduced.

---

**Navigate:** [Table of Contents](../SUMMARY.md)
