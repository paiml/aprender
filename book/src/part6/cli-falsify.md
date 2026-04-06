# `batuta falsify`

The `falsify` command runs the Popperian Falsification Checklist - a 108-item quality assurance protocol based on Toyota Production System (TPS) principles and the scientific method.

## Usage

```bash
# Run full checklist on current directory
batuta falsify .

# Run on a specific project
batuta falsify /path/to/project

# Output JSON format
batuta falsify . --json

# Critical checks only (fast mode)
batuta falsify . --critical-only
```

## Overview

The checklist implements Sir Karl Popper's falsification principle: every claim must have explicit rejection criteria. Each of the 108 items is a falsifiable claim about the project's quality.

## Sections

The checklist is organized into 10 sections:

| Section | Items | Focus |
|---------|-------|-------|
| 1. Sovereign Data Governance | 15 | Data residency, privacy, consent |
| 2. ML Technical Debt Prevention | 10 | CACE, entanglement, dead code |
| 3. Hypothesis-Driven Development | 13 | Reproducibility, baselines, statistics |
| 4. Numerical Reproducibility | 15 | IEEE754, cross-platform determinism |
| 5. Performance & Waste Elimination | 15 | PCIe rule, SIMD, latency SLAs |
| 6. Safety & Formal Verification | 10 | Memory safety, fuzzing, Miri |
| 7. Jidoka Automated Gates | 10 | CI/CD circuit breakers |
| 8. Model Cards & Auditability | 10 | Documentation, provenance |
| 9. Cross-Platform & API | 5 | Linux/macOS/Windows, WASM |
| 10. Architectural Invariants | 5 | YAML config, pure Rust testing |

## TPS Grades

Results are graded using Toyota Production System terminology:

| Grade | Score | Meaning |
|-------|-------|---------|
| Toyota Standard | 95-100% | Production ready |
| Kaizen Required | 85-94% | Acceptable with improvements |
| Andon Warning | 70-84% | Issues require attention |
| Stop the Line | <70% | Critical issues block release |

## Severity Levels

Each check has a severity level:

- **Critical**: Blocks release if failed
- **Major**: Requires remediation plan
- **Minor**: Should be documented
- **Info**: Informational only

## Example Output

```
╔═══════════════════════════════════════════════════════════════════╗
║     POPPERIAN FALSIFICATION CHECKLIST - Sovereign AI Protocol    ║
╚═══════════════════════════════════════════════════════════════════╝

Project: .
Evaluated: 2025-12-11T12:00:00+00:00

Grade: ◐ Kaizen Required
Score: 88.9%
Items: 84/108 passed, 0 failed

─── Jidoka Automated Gates ───
  ✓ JA-01 Pre-Commit Hook Enforcement [MAJOR]
  ✓ JA-02 Automated Sovereignty Linting [MAJOR]
  ✓ JA-03 Data Drift Circuit Breaker [MAJOR]
  ...

✅ All critical checks passed - Release allowed
```

## Integration with CI

Add to your CI pipeline:

```yaml
- name: Quality Gate
  run: |
    batuta falsify . --json > falsification-report.json
    # Fail if critical checks fail
    batuta falsify . --critical-only || exit 1
```

## TPS Principles Applied

The checklist embodies Toyota Way principles:

- **Jidoka**: Automated gates stop on quality issues
- **Genchi Genbutsu**: Evidence-based verification
- **Kaizen**: Continuous improvement through feedback
- **Muda**: Waste detection and elimination
- **Poka-Yoke**: Error-proofing through constraints

## Related Commands

- [`batuta stack quality`](./cli-stack.md) - Stack-wide quality metrics
- [`batuta analyze`](./cli-analyze.md) - Project analysis
