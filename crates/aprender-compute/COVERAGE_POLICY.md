# Coverage Policy

**Target**: ≥90% line coverage for all production code

**Last Updated**: 2025-11-20

## Current Status

| Component | Coverage | Status |
|-----------|----------|--------|
| **Trueno library** | 92.11% | ✅ PASS |
| **xtask** | 89.74% | ⚠️ Close (validate_examples 90.03%) |
| **Overall** | 91.78% | ✅ PASS |

## Coverage Measurement

**Single Canonical Method**: `make coverage`

```bash
# Generate coverage report
make coverage

# Enforce 90% threshold (fails build if < 90%)
make coverage-check
```

**IMPORTANT**: Coverage is measured **WITHOUT GPU features** (`--workspace` only, no `--all-features`)

## Excluded from Coverage

### GPU Backend (Justified Technical Limitation)

**Why Excluded**: LLVM coverage instrumentation **cannot track**:
- GPU shader execution (WGSL compiled to SPIR-V bytecode)
- Async GPU operations (device.poll, queue.submit)
- GPU memory transfers (buffer uploads/downloads)
- GPU-side SIMD instructions

**Evidence**: GPU test suite has 28 passing tests, but coverage shows 3-20% because instrumentation can't reach GPU hardware.

**When Included**: Including GPU features (`--all-features`) drops overall coverage from 92% → 74%.

### CLI Entrypoints (xtask/src/main.rs)

**Why Lower Coverage**:
- 16-line CLI wrapper calling `std::process::exit(1)`
- Integration testing would require spawning subprocesses
- Core logic (validate_examples.rs) has 90.03% coverage ✅

## Coverage by File

```
✅ >90% - Production Quality
   vector.rs:           94.13%
   matrix.rs:           97.92%
   avx2.rs:             94.14%
   scalar.rs:           94.19%
   chaos.rs:            100%
   error.rs:            100%
   validate_examples.rs: 90.03%

⚠️ 85-90% - Acceptable
   avx512.rs:           87.46%
   xtask/main.rs:       80.49% (CLI wrapper)

❌ <85% - Needs Improvement
   sse2.rs:             83.06% → Target: 90%
   lib.rs:              72.50% → Target: 90%
```

## Quality Gates

### Pre-Commit (Tier 2)
```bash
make tier2  # Includes coverage check
```

- ✅ All tests pass
- ✅ Coverage ≥90% (enforced)
- ✅ Zero clippy warnings
- ✅ PMAT TDG ≥B+ (85/100)

### CI/CD (Tier 3)
```bash
make tier3  # Full validation
```

- ✅ Mutation testing ≥80%
- ✅ Benchmarks complete
- ✅ Security audit passes

## Improving Coverage

### Priority Files (Below 90%)

**1. src/lib.rs (72.50%)**
- Missing: Platform-specific feature detection paths
- Action: Add unit tests for each backend selection path

**2. src/backends/sse2.rs (83.06%)**
- Missing: Edge cases in SIMD operations
- Action: Add property-based tests for remainder handling

**3. src/backends/avx512.rs (87.46%)**
- Missing: AVX-512 specific instruction paths
- Action: Add backend equivalence tests vs AVX2

### Steps to Improve

```bash
# 1. Generate HTML coverage report
make coverage

# 2. Open in browser
open target/coverage/html/index.html

# 3. Navigate to file (e.g., lib.rs)
# 4. Red lines = uncovered, add tests for those paths

# 5. Verify improvement
make coverage-check
```

## Philosophy (Jidoka - Stop the Line)

Coverage enforcement **blocks commits** when <90%:
- Pre-commit hook runs `make coverage-check`
- Exits with error code 1 if coverage drops
- Forces developers to add tests immediately
- Prevents coverage decay over time

**Toyota Way**: Build quality in, don't inspect it in later.

## Historical Context

**2025-11-20**: Fixed coverage measurement inconsistency
- **Problem**: Two different results (76% vs 91%) depending on flags
- **Root Cause**: `make coverage` used `--all-features`, report used cached data without
- **Fix**: Single canonical approach - always exclude GPU features
- **Result**: Both targets now report 91.78% consistently

## References

- CLAUDE.md: Coverage requirements (line 52-72)
- Makefile: `coverage` and `coverage-check` targets (lines 196-217)
- ROADMAP.md: Quality gates (line 761-769)
