# Quality and Testing Specification

> Parent: [batuta-spec.md](../batuta-spec.md)
> Sources: popperian-falsification-checklist, testing-quality-ecosystem-spec, bug-hunter-pmat-quality-integration

---

## 1. Popperian Falsification Methodology

### Core Philosophy

*"A theory that explains everything, explains nothing."* -- Karl Popper

Every specification claim must be accompanied by a **falsification criterion** -- a concrete, measurable condition that would prove the claim wrong. A claim without a falsification criterion is not engineering; it is marketing.

### Five Pillars of Sovereign AI Verification

| Pillar | Definition | Verification Mechanism |
|--------|-----------|----------------------|
| **AI Capabilities** | Functional ML operations | Numerical parity tests |
| **Data Residency** | Geographic containment | Static analysis + runtime audit |
| **Data Privacy** | PII protection | Formal verification |
| **Legal Controls** | Regulatory compliance | Policy-as-code checks |
| **Security/Resiliency** | Attack resistance | Fuzzing, circuit breakers |

### Toyota Way as Engineering Epistemology

| Principle | Application |
|-----------|-------------|
| **Jidoka** (Autonomation) | CI/CD detects abnormalities and stops automatically before human review |
| **Genchi Genbutsu** (Go and see) | Reviewers inspect distributions, run inference locally, verify data locations |
| **Kaizen** (Continuous improvement) | Each review cycle improves checklists and automation |

### Four Steps of Jidoka in CI

1. **Detect** the abnormality (automated tests, linters, formal proofs)
2. **Stop** the pipeline (circuit breaker)
3. **Fix** the immediate condition (block merge)
4. **Investigate** root cause to prevent recurrence (blameless post-mortem)

### Falsification Checklist Structure

Each specification claim follows this template:

```
CLAIM: [What is asserted]
FALSIFICATION: The claim is falsified if [specific measurable condition]
TEST: [Concrete command or benchmark to execute]
THRESHOLD: [Numeric pass/fail boundary]
```

### ML-Specific Review Concerns

| Risk Category | Falsification Approach |
|--------------|----------------------|
| Entanglement (CACE) | Change one feature, verify others unchanged |
| Hidden feedback loops | Trace data lineage end-to-end |
| Reproducibility | Fixed seeds, deterministic execution |
| Data distribution shift | Statistical tests on input distributions |
| Quantization drift | KL divergence against full-precision baseline |
| Numerical stability | Property-based testing on edge cases |

---

## 2. Testing Ecosystem (pmat / oip / probar)

### Tool Responsibilities

| Tool | Domain | Primary Function |
|------|--------|------------------|
| **pmat** | Static Analysis | Code quality, TDG, SATD, complexity (17+ languages) |
| **oip** | Defect Intelligence | ML defect classification, SBFL fault localization |
| **probar** | Runtime Testing | WASM testing, browser automation, visual regression |

### Capability Matrix

| Capability | pmat | oip | probar |
|------------|------|-----|--------|
| SATD Detection (4-severity) | Yes | -- | -- |
| Cyclomatic/Cognitive Complexity | Yes | -- | -- |
| Dead Code Detection | Yes | -- | -- |
| Code Duplication (MinHash+LSH) | Yes | -- | -- |
| TDG Scoring (A-F grades) | Yes | -- | -- |
| SBFL Fault Localization | -- | Yes (Tarantula/Ochiai/DStar) | Basic |
| Commit Classification | -- | Yes (ML) | -- |
| Defect Pattern ML | -- | Yes (RandomForest, RAG) | -- |
| Browser Automation | -- | -- | Yes (CDP) |
| Visual Regression | -- | -- | Yes (perceptual diff) |
| WASM Coverage | -- | -- | Yes (block-level) |
| O(n) Complexity Detection | -- | -- | Yes (curve fitting) |

### pmat Key Commands

```bash
pmat analyze complexity src/        # Cyclomatic + cognitive complexity
pmat analyze satd src/              # Self-admitted technical debt
pmat analyze duplicates src/        # Code clone detection
pmat rust-project-score .           # Composite quality score (/114)
pmat repo-score .                   # Repository hygiene score (/110)
pmat query "error handling"         # Semantic code search
pmat query --coverage-gaps --limit 30  # Find untested functions
```

### Quality Tiers (Certeza Methodology)

| Tier | Trigger | Target Time | Checks |
|------|---------|-------------|--------|
| **Tier 1** | On save | < 1s | fmt, clippy, check |
| **Tier 2** | Pre-commit | < 5s | test --lib, clippy |
| **Tier 3** | Pre-push | 1-5 min | Full tests |
| **Tier 4** | CI/CD | Full | Release tests + pmat analysis |

---

## 3. Bug-Hunter PMAT Quality Integration

### Problem

Bug-hunter's fault localization treated all code locations equally regardless of quality. Research shows low-quality code is statistically more likely to contain defects (Zazworka et al., 2011).

### Solution: 5-Channel SBFL

Extend the existing 4-channel SBFL with PMAT quality as a 5th channel:

```
CLI flags (--pmat-quality, --quality-weight, --pmat-scope)
         |
         v
    HuntConfig
         |
         +--[BH-22]-> scope_targets_by_quality() -> filtered targets
         |
         v
    hunt() / analyze_common_patterns()
         |
         +--[BH-21]-> build_quality_index() -> PmatQualityIndex
         |
         +--[BH-23]-> enrich_with_satd()     -> SATD annotations
         |
         +--[BH-24]-> regression_risk_score() -> risk scores
         |
         +--[BH-25]-> spec_quality_gate()     -> claim validation
         |
         v
    Final Scored Findings
```

### Five Capabilities

| ID | Capability | Description |
|----|-----------|-------------|
| **BH-21** | Quality-Weighted Suspiciousness | Boost low-TDG code, reduce high-TDG code in fault ranking |
| **BH-22** | Smart Target Scoping | Focus analysis on code below quality thresholds |
| **BH-23** | SATD-Enriched Findings | Surface self-admitted technical debt as bug locations |
| **BH-24** | Regression Risk Scoring | Quantify risk for each finding using quality signals |
| **BH-25** | Spec Claim Quality Gates | Gate verification claims against implementation quality |

### Quality-Weighted Scoring

```rust
pub struct PmatQualityIndex {
    function_grades: HashMap<String, TdgGrade>,
    function_complexity: HashMap<String, u32>,
    function_satd: HashMap<String, Vec<SatdViolation>>,
}

// Suspiciousness adjustment:
// score = base_score * quality_weight(tdg_grade)
// where quality_weight(A) = 0.5, quality_weight(F) = 2.0
```

### CLI Usage

```bash
# Enable PMAT quality integration
batuta bug-hunter hunt --pmat-quality src/

# Scope to low-quality targets only
batuta bug-hunter hunt --pmat-quality --pmat-scope "max-grade:C" src/

# Adjust quality channel weight
batuta bug-hunter hunt --pmat-quality --quality-weight 0.3 src/

# Gate spec claims against quality
batuta bug-hunter hunt-with-spec --pmat-quality spec.md src/
```

---

## 4. Mutation Testing

### Strategy

```bash
# Quick sample (~5 min)
make mutants-fast

# Full suite (~30-60 min)
make mutants

# Specific file
make mutants-file FILE=src/backend.rs
```

### Targets

| Metric | Target |
|--------|--------|
| Mutation kill rate | >= 80% |
| Surviving mutants | Documented with justification |
| Timeout mutants | < 5% of total |

### Integration with Bug-Hunter

Mutation testing results feed into BH-21 quality weighting: functions with high surviving mutant counts receive boosted suspiciousness scores.

---

## 5. Coverage Workflow

```bash
# HTML + LCOV reports
make coverage

# Find coverage gaps ranked by impact
pmat query --coverage-gaps --rank-by impact --limit 20

# Coverage-enriched search
pmat query "error handling" --coverage --limit 10
```

### Coverage Targets

| Scope | Target | Enforcement |
|-------|--------|-------------|
| Overall | >= 95% | CI gate |
| New code | 100% | Pre-merge |
| Critical paths (serve, agent) | >= 98% | Release gate |
| CUDA-gated code | Excluded | Feature flag |
