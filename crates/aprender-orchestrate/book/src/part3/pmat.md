# PMAT: Quality Analysis

> **"PMAT (Pragmatic Metrics & Analysis Tool) provides TDG scoring, complexity analysis, and adaptive quality assessment for Batuta workflows."**

## Overview

**PMAT** is Batuta's quality analysis tool that measures code quality and generates actionable roadmaps:

- **TDG (Technical Debt Grade)**: A-F grade for code quality
- **Complexity analysis**: Cyclomatic and cognitive complexity metrics
- **Adaptive analysis**: Muda (waste) elimination through smart analysis
- **Roadmap generation**: Prioritized task lists for improvement
- **Multi-language support**: Python, C, C++, Rust, Shell

## Installation

```bash
# Install from crates.io
cargo install pmat

# Verify installation
pmat --version
# Output: pmat 2.199.0
```

## Basic Usage

### **TDG Scoring**

Calculate Technical Debt Grade for a project:

```bash
# Analyze current directory
pmat tdg .

# Output:
# 📊 Technical Debt Grade (TDG): B
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Complexity:        72/100 (Good)
# Maintainability:   68/100 (Fair)
# Test Coverage:     85/100 (Excellent)
# Documentation:     45/100 (Poor)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Overall Score: 67.5/100 → Grade B
```

### **Complexity Analysis**

Measure code complexity:

```bash
# Analyze complexity (JSON output)
pmat analyze complexity src/ --format json

# Output:
# {
#   "files": [
#     {
#       "path": "src/main.rs",
#       "cyclomatic_complexity": 12,
#       "cognitive_complexity": 8,
#       "lines_of_code": 245
#     }
#   ],
#   "total_complexity": 12,
#   "average_complexity": 3.2
# }
```

### **Language Detection**

Detect languages in a project:

```bash
pmat detect languages /path/to/project

# Output:
# Python:  65% (12,450 lines)
# C:       25% (4,780 lines)
# Shell:   10% (1,920 lines)
```

## Batuta Integration

Batuta uses PMAT for Phase 1 (Analysis):

```bash
# Batuta automatically runs PMAT
batuta analyze /path/to/project

# Internally calls:
pmat tdg /path/to/project
pmat analyze complexity /path/to/project --format json
pmat detect languages /path/to/project
```

**Output integrates into Batuta's analysis phase:**

```
Phase 1: Analysis [████████████████████] 100%
  ✓ Language detection (Python: 65%, C: 25%, Shell: 10%)
  ✓ TDG score: B (67.5/100)
  ✓ Complexity: Medium (avg: 3.2)
  ✓ Recommendations: 5 optimizations identified
```

## TDG Scoring System

### **Grade Scale**

| Grade | Score | Interpretation |
|-------|-------|----------------|
| **A** | 90-100 | Excellent - minimal technical debt |
| **B** | 80-89 | Good - manageable technical debt |
| **C** | 70-79 | Fair - moderate technical debt |
| **D** | 60-69 | Poor - significant technical debt |
| **F** | <60 | Critical - severe technical debt |

### **Components**

TDG is calculated from four weighted metrics:

1. **Complexity (30%)**: Cyclomatic and cognitive complexity
2. **Maintainability (25%)**: Code duplication, naming, structure
3. **Test Coverage (25%)**: Unit test coverage percentage
4. **Documentation (20%)**: Inline comments, API docs, README

**Formula:**

```
TDG = (Complexity × 0.30) + (Maintainability × 0.25) +
      (TestCoverage × 0.25) + (Documentation × 0.20)
```

## Complexity Metrics

### **Cyclomatic Complexity**

Number of independent paths through code:

| Complexity | Rating | Action |
|------------|--------|--------|
| 1-10 | Simple | No action needed |
| 11-20 | Moderate | Consider refactoring |
| 21-50 | Complex | Refactor recommended |
| >50 | Very Complex | Refactor required |

**Example:**

```rust
fn example(x: i32) -> i32 {
    if x > 0 {        // +1
        if x > 10 {   // +1
            x * 2
        } else {      // +1
            x + 1
        }
    } else {
        x - 1
    }
}
// Cyclomatic Complexity: 3
```

### **Cognitive Complexity**

Measures how difficult code is to understand:

- Nested conditions: +1 per level
- Recursion: +1
- Logical operators: +1 per operator
- Goto statements: +5

**Lower is better** - aim for cognitive complexity < 15.

## Adaptive Analysis (Muda Elimination)

PMAT implements **Muda (waste elimination)** by skipping redundant analysis:

### **File Caching**

Skip analysis of unchanged files:

```bash
# First run: analyzes all files
pmat analyze complexity src/

# Second run: only analyzes changed files
pmat analyze complexity src/
# ⏭️  Skipped 42 unchanged files (Muda elimination)
# 📊 Analyzed 3 changed files
```

### **Incremental TDG**

Update TDG score incrementally:

```bash
# Initial full analysis
pmat tdg . --full

# Incremental update (only changed files)
pmat tdg . --incremental
# ⚡ Incremental TDG: B → A (3 files improved)
```

## Roadmap Generation

PMAT generates prioritized improvement roadmaps:

```bash
pmat roadmap generate /path/to/project

# Output:
# 📋 Improvement Roadmap
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Priority 1 (Critical):
#   • Reduce complexity in src/pipeline.rs (CC: 45)
#   • Add tests for src/converter.rs (0% coverage)
#
# Priority 2 (High):
#   • Document public API in src/lib.rs
#   • Refactor src/analyzer.rs (duplicated code)
#
# Priority 3 (Medium):
#   • Improve naming in src/utils.rs
#   • Add examples to README.md
```

## Command-Line Options

```bash
pmat [COMMAND] [OPTIONS]

COMMANDS:
    tdg              Calculate Technical Debt Grade
    analyze          Run specific analysis
    detect           Detect project attributes
    roadmap          Generate improvement roadmap
    work             Workflow management

ANALYZE SUBCOMMANDS:
    complexity       Measure code complexity
    coverage         Analyze test coverage
    duplication      Detect code duplication

DETECT SUBCOMMANDS:
    languages        Detect programming languages
    frameworks       Detect ML frameworks

OPTIONS:
    --format <FORMAT>  Output format: text, json, html [default: text]
    --full             Force full analysis (disable caching)
    --strict           Fail on warnings
    -h, --help         Print help
    -V, --version      Print version
```

## Workflow Management

PMAT integrates with Batuta's workflow:

```bash
# Continue from last task
pmat work continue

# Start specific task
pmat work start BATUTA-008

# List available tasks
pmat work list

# Show workflow status
pmat work status
```

**Example output:**

```
📋 Workflow Status
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Phase 3: ML Library Conversion (60%)

In Progress:
  • BATUTA-008: NumPy → Trueno [████████░░] 80%
  • BATUTA-009: sklearn → Aprender [██████░░░░] 60%

Pending:
  • BATUTA-010: PyTorch → Realizar
  • BATUTA-012: PARF Analysis
```

## Configuration

Configure PMAT via `.pmat.toml`:

```toml
[analysis]
# Skip patterns
skip = [
    "target/",
    "node_modules/",
    "*.pyc"
]

# Complexity thresholds
max_cyclomatic_complexity = 15
max_cognitive_complexity = 20

[tdg]
# Custom weights
complexity_weight = 0.30
maintainability_weight = 0.25
coverage_weight = 0.25
documentation_weight = 0.20

[muda]
# Enable adaptive analysis
enable_caching = true
cache_dir = ".pmat-cache/"
```

## Integration with Make

Add PMAT to Makefile:

```makefile
# Run TDG analysis
tdg:
\t@command -v pmat >/dev/null 2>&1 || { echo "Error: pmat not installed"; exit 1; }
\tpmat tdg src/

# Quality gate (fail if TDG < B)
quality: lint test coverage tdg
\t@echo "✅ All quality gates passed"
```

**Usage:**

```bash
make tdg      # Calculate TDG score
make quality  # Run all quality checks
```

## Version

Current version: **2.199.0**

Check installed version:
```bash
pmat --version
```

Update to latest:
```bash
cargo install pmat --force
```

## Next Steps

- **[Renacer: Syscall Tracing](./renacer.md)**: Runtime validation
- **[TDG Scoring](../part2/tdg-scoring.md)**: Deep dive into TDG calculation
- **[Phase 1: Analysis](../part2/phase1-analysis.md)**: Batuta's analysis workflow

---

**Navigate:** [Table of Contents](../SUMMARY.md)
