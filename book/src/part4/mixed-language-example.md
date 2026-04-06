# Example 4: Mixed-Language Project

This walkthrough demonstrates migrating a project that combines Python, C, and
Shell into a unified Rust codebase using Batuta's multi-transpiler orchestration.

## Scenario

A research lab maintains an image processing toolkit with three components:

- **Python** (`processing/`): OpenCV-based image filters, NumPy matrix ops.
- **C** (`libkernel/`): Custom convolution kernels written for AVX2.
- **Shell** (`scripts/`): Build, test, and benchmark automation.

The components communicate through files and subprocess calls. Builds break
frequently because of Python/C version mismatches and Bash portability issues.

## Source Project Layout

```
image_toolkit/
  processing/
    filters.py          # Python: Gaussian blur, edge detection
    pipeline.py         # Python: orchestration, CLI
    requirements.txt    # opencv-python, numpy, pillow
  libkernel/
    include/kernel.h    # C: public API
    src/convolve.c      # C: AVX2 convolution
    src/resize.c        # C: bilinear interpolation
    Makefile
  scripts/
    build.sh            # Shell: compile C, install Python deps
    benchmark.sh        # Shell: run performance benchmarks
    deploy.sh           # Shell: package and upload
  tests/
    test_filters.py     # Python: pytest suite
    test_kernel.c       # C: CUnit tests
```

## Step 1 -- Analyze All Languages

```bash
batuta analyze --languages --tdg ./image_toolkit
```

```
Languages detected:
  Python  45% (2 files, 580 lines)
  C       35% (3 files, 420 lines)
  Shell   20% (3 files, 240 lines)

ML frameworks: numpy (18 ops), opencv (6 functions)
Unsafe C patterns: 12 raw pointer ops, 4 malloc/free pairs
Shell issues: 3 unquoted variables, 2 missing error checks

Cross-language interfaces:
  Python → C: subprocess call to libkernel.so (filters.py:42)
  Shell → Python: python3 invocation (build.sh:15)
  Shell → C: make invocation (build.sh:8)

TDG Score: D+ (52/100) — cross-language coupling, weak error handling
```

Batuta identifies all three languages, their frameworks, and the interfaces
between them. The cross-language interface map is critical for planning module
boundaries.

## Step 2 -- Prioritized Migration Plan

Batuta generates a migration order based on dependency analysis:

```
Recommended migration order:
  1. Shell scripts → Rust CLI (no dependents)
  2. C library → Rust crate (depended on by Python)
  3. Python processing → Rust (depends on C library)
```

The strategy is bottom-up: migrate leaves first so that each component can be
validated independently before its dependents are converted.

## Step 3 -- Transpile Each Component

```bash
# Phase 1: Shell → Rust CLI
batuta transpile ./scripts --tool bashrs --output ./toolkit_cli

# Phase 2: C → Rust crate
batuta transpile ./libkernel --tool decy --output ./kernel_rs

# Phase 3: Python → Rust (with trueno for NumPy ops)
batuta transpile ./processing --tool depyler --output ./processing_rs
```

Each transpiler handles its source language. Batuta coordinates the three
tools, ensuring that the Rust outputs have compatible module interfaces.

## Step 4 -- Unify Module Boundaries

```bash
batuta optimize ./image_toolkit_rs --unify-modules
```

The optimizer merges the three separate Rust outputs into a single workspace
with shared types. See [Module Boundaries](./mixed-modules.md) for details.

## Step 5 -- Validate

```bash
batuta validate ./image_toolkit_rs --reference ./image_toolkit
```

Batuta runs all original test suites (pytest, CUnit, shell scripts) against
the Rust implementation and compares outputs. Numerical outputs are compared
within floating-point tolerance.

## Result

| Metric           | Mixed (Py/C/Sh) | Unified Rust |
|------------------|------------------|--------------|
| Build time       | 45s              | 8s           |
| Languages        | 3                | 1            |
| Dependency tools | pip, make, bash  | cargo        |
| Portability      | Linux only       | Cross-platform|
| CI config        | 85 lines         | 12 lines     |

## Key Takeaways

- Batuta orchestrates multiple transpilers (depyler, decy, bashrs) in a single
  pipeline, converting each language with its specialized tool.
- Bottom-up migration order (leaves first) minimizes risk at each step.
- Cross-language subprocess calls become direct Rust function calls, eliminating
  serialization overhead and version mismatch bugs.
- The following sub-chapters cover module boundaries, gradual migration, and
  integration testing for mixed-language projects.

---

**Navigate:** [Table of Contents](../SUMMARY.md)
