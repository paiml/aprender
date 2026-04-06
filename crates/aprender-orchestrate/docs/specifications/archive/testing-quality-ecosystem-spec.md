# Testing & Quality Analysis Ecosystem Specification

> **Version**: 1.0.0
> **Date**: 2026-01-12
> **Status**: Active
> **Spec ID**: BATUTA-TESTING-001

## Executive Summary

The PAIML ecosystem provides three complementary tools for testing and quality analysis, each with distinct but non-overlapping responsibilities:

| Tool | Domain | Primary Function |
|------|--------|------------------|
| **pmat** | Static Analysis | Code quality, TDG, SATD, complexity |
| **oip** | Defect Intelligence | ML defect classification, fault localization |
| **probar** | Runtime Testing | WASM testing, browser automation, visual regression |

**Critical Insight**: These tools are NOT substitutes for each other. A project needs all three for comprehensive quality assurance.

---

## Tool Capability Matrix

### Feature Comparison

| Capability | pmat | oip | probar |
|------------|------|-----|--------|
| **Static Analysis** |
| SATD Detection | ✅ 4-severity, 355+ patterns | ❌ | ❌ |
| Cyclomatic Complexity | ✅ CC metrics | ❌ | ❌ |
| Cognitive Complexity | ✅ | ❌ | ❌ |
| Dead Code Detection | ✅ | ❌ | ❌ |
| Code Duplication | ✅ | ❌ | ❌ |
| TDG Scoring | ✅ A-F grades | ❌ | ❌ |
| **Defect Intelligence** |
| SBFL Fault Localization | ❌ | ✅ Tarantula, Ochiai, DStar | ✅ Basic Tarantula |
| Commit Classification | ❌ | ✅ ML classifier | ❌ |
| Defect Pattern ML | ❌ | ✅ RandomForest, RAG | ❌ |
| Calibrated Predictions | ❌ | ✅ Phase 7 | ❌ |
| Ensemble Models | ❌ | ✅ Phase 6 | ❌ |
| **Runtime Testing** |
| Browser Automation | ❌ | ❌ | ✅ CDP protocol |
| Visual Regression | ❌ | ❌ | ✅ Perceptual diff |
| WASM Coverage | ❌ | ❌ | ✅ Block-level |
| TUI Testing | ❌ | ❌ | ✅ Presentar support |
| Pixel Coverage | ❌ | ❌ | ✅ Heatmaps |
| **Algorithmic Analysis** |
| O(n) Complexity Detection | ❌ | ❌ | ✅ Curve fitting |
| Rc/RefCell Linting | ❌ | ❌ | ✅ AST-based |
| WASM Threading Compliance | ❌ | ❌ | ✅ |

---

## Tool 1: pmat (Static Analysis)

### Purpose

Zero-configuration code quality analysis for 17+ languages. Provides actionable metrics without running code.

### Key Capabilities

```
┌─────────────────────────────────────────────────────────┐
│                    PMAT CAPABILITIES                     │
├─────────────────────────────────────────────────────────┤
│  TDG Scoring          │ A-F grades, 6 weighted metrics  │
│  SATD Detection       │ TODO/FIXME/HACK with severity   │
│  Complexity Analysis  │ Cyclomatic + Cognitive          │
│  Dead Code            │ Unused functions/modules        │
│  Code Duplication     │ Clone detection                 │
│  Security Scan        │ Basic vulnerability patterns    │
│  Documentation        │ Coverage and quality            │
└─────────────────────────────────────────────────────────┘
```

### Example Output

```bash
$ pmat quality-gate

Quality Gate: FAILED
Total violations: 475

  Complexity:      64 violations
  Dead code:       6 violations
  Technical debt:  355 violations (17 critical)
  Code entropy:    41 violations
  Duplicates:      6 violations
```

### When to Use

- Pre-commit quality gates
- CI/CD pipeline checks
- Technical debt assessment
- Code review preparation
- Refactoring prioritization

---

## Tool 2: oip (Organizational Intelligence Plugin)

### Purpose

ML-powered defect pattern analysis and fault localization using git history and coverage data.

### Key Capabilities

```
┌─────────────────────────────────────────────────────────┐
│                    OIP CAPABILITIES                      │
├─────────────────────────────────────────────────────────┤
│  Tarantula SBFL       │ Spectrum-based fault localization│
│  Ochiai/DStar         │ Alternative SBFL formulas        │
│  Commit Classification│ ML labeling of defect types      │
│  Training Extraction  │ Git history → training data      │
│  RAG Enhancement      │ trueno-rag knowledge retrieval   │
│  Ensemble Models      │ Weighted multi-model predictions │
│  Calibrated Output    │ Confidence-calibrated scores     │
└─────────────────────────────────────────────────────────┘
```

### Example Output

```bash
$ oip extract-training-data --repo ../whisper.apr

Training Data Statistics:
  Total examples: 13
  Avg confidence: 0.82

Class Distribution:
  TraitBounds: 3 (23.1%)
  ASTTransform: 3 (23.1%)
  ConfigurationErrors: 3 (23.1%)
  OwnershipBorrow: 2 (15.4%)
  ConcurrencyBugs: 1 (7.7%)
  SecurityVulnerabilities: 1 (7.7%)
```

### Fault Localization

```bash
$ oip localize \
    --passed-coverage passed.lcov \
    --failed-coverage failed.lcov \
    --formula tarantula \
    --top-n 10

🎯 Tarantula Hotspot Report
   Line  | Suspiciousness | Status
   ------|----------------|--------
   142   | 0.950          | 🔴 HIGH
   287   | 0.823          | 🔴 HIGH
   56    | 0.612          | 🟡 MEDIUM
```

### When to Use

- Post-test-failure debugging
- Defect pattern analysis across organization
- Training ML models on historical defects
- Root cause analysis
- Bug triage prioritization

---

## Tool 3: probar (Runtime Testing)

### Purpose

Rust-native testing framework for WASM games and web applications. Browser automation, visual regression, and pixel-level coverage.

### Key Capabilities

```
┌─────────────────────────────────────────────────────────┐
│                   PROBAR CAPABILITIES                    │
├─────────────────────────────────────────────────────────┤
│  CDP Browser Control  │ Chrome DevTools Protocol        │
│  Visual Regression    │ Perceptual diff, mask regions   │
│  WASM Coverage        │ Block/superblock instrumentation│
│  Pixel Coverage       │ Heatmap visualization           │
│  TUI Testing          │ Presentar YAML falsification    │
│  Tarantula SBFL       │ Basic fault localization        │
│  O(n) Detection       │ Empirical complexity curves     │
│  Rc/RefCell Linting   │ AST-based state sync detection  │
└─────────────────────────────────────────────────────────┘
```

### Example: Visual Regression

```rust
use jugar_probar::{VisualRegressionTester, VisualRegressionConfig};

let tester = VisualRegressionTester::new(
    VisualRegressionConfig::default()
        .with_threshold(0.02)
        .with_color_threshold(10)
);

let result = tester.compare_images(&baseline, &current)?;
assert!(result.matches, "Visual regression: {}% diff", result.diff_percentage);
```

### Example: Presentar TUI Testing

```rust
use jugar_probar::{TerminalSnapshot, TerminalAssertion};

let snapshot = TerminalSnapshot::from_string(output, 80, 24);

let assertions = [
    TerminalAssertion::Contains("CPU".into()),
    TerminalAssertion::NotContains("ERROR".into()),
    TerminalAssertion::CharAt { x: 0, y: 0, expected: '┌' },
];

for assertion in &assertions {
    assertion.check(&snapshot)?;
}
```

### When to Use

- WASM game/application testing
- Browser-based UI testing
- Visual regression in CI/CD
- TUI application validation
- Pixel-level coverage analysis

---

## Integration Workflow

### Recommended Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    QUALITY ASSURANCE PIPELINE                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Phase 1: Static Analysis (pmat)                                │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ pmat quality-gate                                        │    │
│  │ → TDG score, SATD, complexity, dead code, duplicates    │    │
│  │ → FAIL if violations > threshold                         │    │
│  └─────────────────────────────────────────────────────────┘    │
│                           ↓                                      │
│  Phase 2: Runtime Testing (probar)                              │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ cargo test + probar coverage                             │    │
│  │ → Unit tests, integration tests                          │    │
│  │ → Visual regression (if UI)                              │    │
│  │ → Generate LCOV coverage                                 │    │
│  └─────────────────────────────────────────────────────────┘    │
│                           ↓                                      │
│  Phase 3: Fault Analysis (oip) [on test failure]                │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ oip localize --passed-coverage --failed-coverage        │    │
│  │ → Tarantula SBFL hotspot report                          │    │
│  │ → Defect pattern classification                          │    │
│  │ → Prioritized debugging targets                          │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Makefile Integration

```makefile
# Combined quality pipeline
.PHONY: quality

quality: static-analysis test fault-analysis

static-analysis:
	@echo "Phase 1: Static Analysis (pmat)"
	pmat quality-gate --strict

test:
	@echo "Phase 2: Runtime Testing (probar)"
	cargo test --all-features
	cargo llvm-cov --lcov --output-path lcov.info

fault-analysis:
	@echo "Phase 3: Fault Analysis (oip) - only on failure"
	@if [ -f failed-tests.lcov ]; then \
		oip localize \
			--passed-coverage lcov.info \
			--failed-coverage failed-tests.lcov \
			--top-n 20; \
	fi
```

---

## Gap Analysis: What Each Tool Cannot Do

### pmat Cannot:

- Run tests or execute code
- Perform fault localization (needs runtime data)
- Train ML models on defects
- Test visual UI/pixels
- Analyze WASM binaries

### oip Cannot:

- Detect SATD (TODO/FIXME)
- Calculate cyclomatic complexity
- Find dead code or duplicates
- Run browser automation
- Generate TDG scores

### probar Cannot:

- Detect SATD patterns
- Analyze code without executing it
- Train defect classification models
- Calculate TDG scores
- Find code duplication

---

## Real-World Analysis Results

### whisper.apr (analyzed 2026-01-12)

| Tool | Finding |
|------|---------|
| **pmat** | 475 violations: 355 SATD (17 critical), 64 complexity, 6 dead code |
| **oip** | 13 defect patterns: TraitBounds 23%, ASTTransform 23%, ConfigErrors 23% |
| **probar** | 435 `unwrap()` calls detected via Rc/RefCell linting |

### interactive.paiml.com (analyzed 2026-01-12)

| Tool | Finding |
|------|---------|
| **pmat** | 561 violations: 439 complexity, 88 duplicates, 21 SATD |
| **oip** | 146 defect patterns: ASTTransform 36%, OwnershipBorrow 30% |
| **probar** | Visual regression capability for 15 WASM demos |

---

## Version Information

| Tool | Current Version | Repository |
|------|-----------------|------------|
| pmat | 2.213.4 | github.com/paiml/paiml-mcp-agent-toolkit |
| oip | 0.3.1 | github.com/paiml/organizational-intelligence-plugin |
| probar | 0.2.x | github.com/paiml/probar (crates.io: jugar-probar) |

---

## Conclusion

**No single tool provides complete quality assurance.** The PAIML ecosystem requires:

1. **pmat** for static analysis before code runs
2. **probar** for runtime testing and visual validation
3. **oip** for post-failure fault localization and defect intelligence

Using all three tools together provides defense-in-depth quality assurance following Toyota Way principles:

- **Jidoka**: pmat catches defects at compile-time
- **Poka-Yoke**: probar's type-safe selectors prevent errors
- **Genchi Genbutsu**: oip's SBFL goes to the source of bugs

---

**Navigate:** [Specifications Index](./README.md) | [Stack Spec](./batuta-stack-spec.md)
