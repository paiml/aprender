# OIP: Defect Intelligence

> **"OIP (Organizational Intelligence Plugin) provides ML-powered defect pattern analysis and spectrum-based fault localization."**

## Overview

**OIP** analyzes git history and test coverage to identify defect patterns and locate bugs:

- **SBFL Fault Localization**: Tarantula, Ochiai, DStar algorithms
- **Defect Classification**: ML-based commit labeling
- **Training Data Extraction**: Convert git history to ML training data
- **RAG Enhancement**: Knowledge retrieval with trueno-rag
- **Ensemble Models**: Weighted multi-model predictions

## Installation

```bash
# Install from crates.io
cargo install oip

# Verify installation
oip --version
# Output: oip 0.3.1
```

## Basic Usage

### **Training Data Extraction**

Extract defect patterns from git history:

```bash
oip extract-training-data --repo /path/to/project --max-commits 500

# Output:
# Training Data Statistics:
#   Total examples: 146
#   Avg confidence: 0.84
#
# Class Distribution:
#   ASTTransform: 53 (36.3%)
#   OwnershipBorrow: 43 (29.5%)
#   ComprehensionBugs: 12 (8.2%)
#   ...
```

### **Fault Localization**

Find suspicious lines using SBFL:

```bash
oip localize \
    --passed-coverage passed.lcov \
    --failed-coverage failed.lcov \
    --formula tarantula \
    --top-n 10

# Output:
# 🎯 Tarantula Hotspot Report
#    Line  | Suspiciousness | Status
#    ------|----------------|--------
#    142   | 0.950          | 🔴 HIGH
#    287   | 0.823          | 🔴 HIGH
#    56    | 0.612          | 🟡 MEDIUM
```

## SBFL Formulas

OIP supports multiple fault localization formulas:

| Formula | Description | Best For |
|---------|-------------|----------|
| **Tarantula** | Classic SBFL | General use |
| **Ochiai** | Cosine similarity | High precision |
| **DStar2** | D* with power 2 | Balanced |
| **DStar3** | D* with power 3 | Aggressive |

### Suspiciousness Calculation

**Tarantula formula:**

```
suspiciousness = (failed(line) / total_failed) /
                 ((failed(line) / total_failed) + (passed(line) / total_passed))
```

## Defect Pattern Categories

OIP classifies defects into these categories:

| Category | Description | Example |
|----------|-------------|---------|
| **TraitBounds** | Missing or incorrect trait bounds | `T: Clone + Send` |
| **ASTTransform** | Syntax/structure issues | Macro expansion bugs |
| **OwnershipBorrow** | Ownership/lifetime errors | Use after move |
| **ConfigurationErrors** | Config/environment issues | Missing feature flag |
| **ConcurrencyBugs** | Race conditions | Data races |
| **SecurityVulnerabilities** | Security issues | Buffer overflow |
| **TypeErrors** | Type mismatches | Wrong generic |
| **MemorySafety** | Memory bugs | Dangling pointer |

## Advanced Features

### **RAG Enhancement**

Use knowledge retrieval for better localization:

```bash
oip localize \
    --passed-coverage passed.lcov \
    --failed-coverage failed.lcov \
    --rag \
    --knowledge-base bugs.yaml \
    --fusion rrf
```

### **Ensemble Models**

Combine multiple models for higher accuracy:

```bash
oip localize \
    --passed-coverage passed.lcov \
    --failed-coverage failed.lcov \
    --ensemble \
    --ensemble-model trained-model.bin \
    --include-churn
```

### **Calibrated Predictions**

Get confidence-calibrated outputs:

```bash
oip localize \
    --passed-coverage passed.lcov \
    --failed-coverage failed.lcov \
    --calibrated \
    --calibration-model calibration.bin \
    --confidence-threshold 0.7
```

## Integration with Batuta

OIP integrates with Batuta's validation phase:

```bash
# Batuta can invoke OIP for fault analysis
batuta validate --fault-localize
```

## Comparison with pmat

| Capability | pmat | oip |
|------------|------|-----|
| SATD Detection | ✅ | ❌ |
| TDG Scoring | ✅ | ❌ |
| Complexity Analysis | ✅ | ❌ |
| Fault Localization | ❌ | ✅ |
| Defect ML | ❌ | ✅ |
| RAG Enhancement | ❌ | ✅ |

**Key insight**: pmat is for static analysis BEFORE tests run. OIP is for fault analysis AFTER tests fail.

## Command Reference

```bash
oip [COMMAND] [OPTIONS]

COMMANDS:
    analyze                Analyze GitHub organization
    summarize              Summarize analysis report
    review-pr              Review PR with context
    extract-training-data  Extract training data from git
    train-classifier       Train ML classifier
    export                 Export features
    localize               SBFL fault localization

LOCALIZE OPTIONS:
    --passed-coverage <PATH>   LCOV from passing tests
    --failed-coverage <PATH>   LCOV from failing tests
    --formula <FORMULA>        tarantula, ochiai, dstar2, dstar3
    --top-n <N>                Top suspicious lines
    --rag                      Enable RAG enhancement
    --ensemble                 Use ensemble model
    --calibrated               Calibrated predictions
```

## Version

Current version: **0.3.1**

## Next Steps

- **[PMAT: Static Analysis](./pmat.md)**: Pre-test quality checks
- **[Probar: Runtime Testing](./probar.md)**: Test execution and coverage
- **[Phase 4: Validation](../part2/phase4-validation.md)**: Batuta's validation workflow

---

**Navigate:** [Table of Contents](../SUMMARY.md)
