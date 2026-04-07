# Sub-spec: Quality Gates & Testing

**Parent:** [trueno-spec.md](../trueno-spec.md) Sections 11, 12, 13

---

## 1. Coverage

**≥90% line coverage is non-negotiable.** Automatically enforced — cannot be bypassed.

**Enforcement layers:**
1. `make coverage-check` exits with error if < 90%
2. CI pipeline rejects PRs below threshold

**Commands:**
```bash
make coverage           # generate report + display total
make coverage-check     # pass/fail check
```

**Rules:**
- ONLY `make coverage` — never `cargo llvm-cov` directly, never `cargo-tarpaulin`
- New code must have 100% coverage
- HTML report: `target/coverage/html/index.html`

| Component | Minimum | Target |
|-----------|---------|--------|
| Public API | 100% | 100% |
| SIMD backends | 90% | 95% |
| GPU backend | 85% | 90% |
| Overall | **90%** | **95%+** |

## 2. Five Test Categories

Every operation requires all five categories:

### 2.1 Unit Tests
- Correctness for normal inputs
- Empty inputs, single element
- Non-aligned sizes (7, 15, 17 — not multiples of lane width)
- Edge cases: NaN, +/-Inf, subnormal, f32::MAX, f32::MIN

### 2.2 Property-Based Tests (proptest)
- Commutativity: `a + b == b + a`
- Associativity: `(a + b) + c == a + (b + c)` (within tolerance)
- Distributivity: `a * (b + c) == (a * b) + (a * c)` (within tolerance)
- Identity: `a + 0 == a`, `a * 1 == a`

### 2.3 Backend Equivalence Tests
- All backends must produce identical results
- Compare: scalar vs SSE2 vs AVX2 vs AVX-512 vs NEON vs WASM vs GPU
- Floating-point tolerance: < 1e-5 for f32

### 2.4 Mutation Testing
- ≥80% mutation kill rate
- Run: `cargo mutants --timeout 120 --minimum-pass-rate 80`
- Tests that don't catch mutations are weak tests

### 2.5 Benchmark Tests
- Every optimization must prove ≥10% speedup vs scalar
- Test sizes: 100, 1K, 10K, 100K, 1M, 10M elements
- ≥100 iterations, CV < 5%
- Saved to `target/criterion/` for regression detection

## 3. Quality Gate Checklist

**Every commit:**
- `cargo clippy --all-features -- -D warnings` (zero warnings)
- `cargo test --all-features` (all pass)
- `make coverage` (≥90%)
- `cargo fmt -- --check` (formatted)
- `pmat analyze tdg --min-grade B+`

**Every PR (additionally):**
- All 5 test categories for new code
- Rustdoc updated for new public API
- Benchmarks prove ≥10% improvement (if optimization)
- Mutation testing ≥80% kill rate
- Contract FALSIFY tests pass
- Integration test if adding backend

**Every release (additionally):**
- Full CI pipeline green
- `pmat repo-score . --min-score 90` (≥90/110)
- Changelog updated (keep-a-changelog format)
- Semver version bump
- Git tag `vX.Y.Z`
