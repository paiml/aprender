# Sub-spec: Diagnostic Output

Version: 1.0
Status: proposed
Date: 2026-04-09

**Parent:** [pv-spec.md](../pv-spec.md) Section 22

---

## Motivation

Falsification against 9 reference tools (Kani, SPARK/Ada, Dafny, Clippy,
mypy, ESLint, SonarQube, cargo-deny, OpenSSF Scorecard) revealed 13 gaps
in provable-contracts diagnostic output.

## Gap Analysis

### Tier P0 (high impact, low effort)

| # | Gap | Pattern From | Status |
|---|-----|-------------|--------|
| 1 | **Grouped finding display** | ESLint by-file, Clippy by-lint, cargo-deny by-check | [IMPLEMENTED] |
| 2 | **Color terminal output** | Clippy, mypy, cargo-deny | [IMPLEMENTED] |
| 3 | **`pv lint --explain <rule>`** | Clippy `--explain`, mypy error codes, Rust `--explain E0308` | [IMPLEMENTED] |

### Tier P1 (high impact, medium effort)

| # | Gap | Pattern From | Status |
|---|-----|-------------|--------|
| 4 | **Probe-level score decomposition** | OpenSSF Scorecard 44 probes, SonarQube per-rule | [IMPLEMENTED] |
| 5 | **Source snippets + caret spans** | Clippy, mypy `--pretty`, Dafny IDE | [IMPLEMENTED] |
| 6 | **Per-obligation verification table** | Kani property table, SPARK proved/unproved matrix | [IMPLEMENTED] |

### Tier P2 (medium impact, medium effort)

| # | Gap | Pattern From | Status |
|---|-----|-------------|--------|
| 7 | **Counterexample/evidence data** | Kani concrete-playback, SPARK/Dafny counterexamples | [IMPLEMENTED] |
| 8 | **Remediation effort estimation** | SonarQube time-based debt | [IMPLEMENTED] |
| 9 | **Issue lifecycle (new/pre-existing)** | SonarQube lifecycle, SPARK justified checks | [IMPLEMENTED] |
| 10 | **Structured fix suggestions** | Clippy MachineApplicable, ESLint fixable rules | [IMPLEMENTED] |

### Tier P3 (lower priority)

| # | Gap | Pattern From | Status |
|---|-----|-------------|--------|
| 11 | **Per-contract resource metrics** | Dafny resource units, Kani per-harness timing | [IMPLEMENTED] |
| 12 | **HTML report output** | ESLint HTML formatter, SPARK HTML proof report | [IMPLEMENTED] |
| 13 | **Daemon/LSP mode** | mypy dmypy, Dafny IDE, rust-analyzer | [IMPLEMENTED] |

---

## P0 Implementations

### Grouped Finding Display

Findings grouped by contract file, then by rule within each contract.
Per-contract summary line shows error/warning count.

```
contracts/softmax-kernel-v1.yaml (2 warnings)
  [WARN] PV-ENF-001 — Equation `softmax` has no preconditions
  [WARN] PV-ENF-002 — Equation `softmax` has no lean_theorem

contracts/rmsnorm-kernel-v1.yaml (1 error)
  [ERROR] PV-SCR-001 — Composite score 0.35 below threshold 0.50
```

### Color Terminal Output

ANSI colors for severity-differentiated output:
- Red: errors
- Yellow: warnings
- Cyan: file paths
- Bold: rule IDs
- Green: pass indicators

Controlled by `--color` flag (auto/always/never). Default: auto-detect
terminal capability via `atty` or `is-terminal` crate.

### `pv lint --explain <rule>`

Long-form explanation for any rule ID:

```
$ pv lint --explain PV-ENF-001

PV-ENF-001: Equation without preconditions
Category: enforcement
Default severity: warning

DESCRIPTION:
  Every equation in a kernel contract should have at least one
  precondition (a Rust expression that must hold before the kernel
  executes). Preconditions are compiled to debug_assert!() by build.rs.

WHY IT MATTERS:
  Without preconditions, the contract cannot verify that callers satisfy
  input requirements. This leaves a gap in the verification chain between
  the paper's domain constraints and the implementation.

HOW TO FIX:
  Add preconditions to the equation in your contract YAML:

    equations:
      my_kernel:
        formula: "f(x) = ..."
        preconditions:
          - "!x.is_empty()"
          - "x.iter().all(|v| v.is_finite())"

REFERENCES:
  - Meyer (1988), Object-Oriented Software Construction, Ch. 11
  - docs/specifications/sub/eiffel-dbc.md
```

---

## References

1. Kani Contributors (2022-2026). "Verification Results." model-checking.github.io/kani
2. AdaCore (2026). "GNATprove User's Guide: Viewing Output." docs.adacore.com
3. Dafny Contributors (2024). "Verification Optimization." dafny.org
4. Rust Clippy (2026). "Emitting Lints: Applicability." doc.rust-lang.org/clippy
5. mypy Contributors (2026). "Error Codes." mypy.readthedocs.io
6. ESLint (2026). "Formatters Reference." eslint.org
7. SonarSource (2025). "Quality Gates." docs.sonarsource.com
8. OpenSSF (2024). "Beyond Scores: Probe-Level Results." openssf.org/blog
