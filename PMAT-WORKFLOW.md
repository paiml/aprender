# PMAT Workflow for Trueno Development

This document guides you through using PMAT (Pragmatic AI Labs Multi-language Agent Toolkit) to scaffold and develop the Trueno project with EXTREME TDD quality standards.

## Quick Start

```bash
# Navigate to project
cd /home/noah/src/trueno

# View available prompts
pmat prompt --list

# Show specific prompt
pmat prompt <name> --format text
```

## Available PMAT Prompts

### 1. **code-coverage** (CRITICAL) - Achieve >85% coverage
```bash
pmat prompt code-coverage --format text
```

**When to use**: Continuously throughout development to maintain >90% coverage target.

**Key Points**:
- Target: >90% coverage (not 85%)
- Use PROPTEST_CASES=100 (not 5) for property tests
- Classify modules FIRST (LOGIC vs UI/CLI, Frontend vs Backend)
- Track ROI (tests-per-percentage), auto-pivot if <0.05%/test for 2 batches
- Commit on 2%+ improvement OR 20+ tests passing OR 90-min time-box
- SKIP UI/CLI modules (low ROI), focus on LOGIC modules

**Module Classification**:
- Frontend (Parser, Lexer) → 95% coverage, property tests (100 cases), fuzz testing
- Backend (Transpiler, Codegen) → 85% coverage, golden file tests
- Runtime (Interpreter, REPL) → 90% coverage, integration tests

### 2. **continue** (HIGH) - Continue next best step
```bash
pmat prompt continue --format text
```

**When to use**: After completing any task, to determine next best action.

**Workflow**:
1. Run `pmat analyze` to identify issues
2. Run `pmat tdg` to find highest debt
3. Prioritize: uncovered code > low-coverage+low-TDG
4. Implement fix using RED-GREEN-REFACTOR
5. Verify all quality gates pass
6. Commit with descriptive message

### 3. **quality-enforcement** (CRITICAL) - Run all quality gates
```bash
pmat prompt quality-enforcement --format text
```

**When to use**: Before EVERY commit.

**Quality Gates** (ALL must pass):
1. Compilation: `cargo build --all-features`
2. Linting: `cargo clippy -- -D warnings` (zero warnings)
3. Formatting: `cargo fmt -- --check`
4. Tests: `make test-fast` (100% passing, <5 min)
5. Coverage: `make coverage` (>85%, <10 min)
6. Mutation: `pmat mutate` (score >80%)
7. Complexity: `pmat analyze` (max <15)
8. TDG: `pmat tdg` (average score >60)
9. Documentation: `pmat validate-docs` (no broken links)
10. README: `pmat validate-readme` (no hallucinations)

**Zero Tolerance**:
- Compilation warnings: 0
- Clippy warnings: 0
- Test failures: 0
- Coverage: Must be >85%
- Broken links: 0

### 4. **debug** (CRITICAL) - Five Whys root cause analysis
```bash
pmat prompt debug --format text
```

**When to use**: When encountering any defect or test failure.

**Process**:
1. Apply Five Whys to find root cause
2. Implement fix using EXTREME TDD
3. STOP THE LINE if defect is due to unimplemented functionality
4. Concept of "pre-existing failure" is irrelevant - FIX IT

### 5. **mutation-testing** (HIGH) - Run mutation testing
```bash
pmat prompt mutation-testing --format text
```

**When to use**: On high-complexity or low-coverage code.

**Target**: >80% mutation kill rate

```bash
# Run mutation testing on specific file
cargo mutants --file src/backend/simd/avx2.rs --timeout 60
```

### 6. **refactor-hotspots** (HIGH) - Refactor high-debt code
```bash
pmat prompt refactor-hotspots --format text
```

**When to use**: When TDG analysis reveals high technical debt.

**Process**:
1. Run `pmat tdg` to identify hotspots
2. Prioritize modules with low coverage + high complexity
3. Refactor using EXTREME TDD (RED-GREEN-REFACTOR)
4. Verify all quality gates pass

### 7. **performance-optimization** (HIGH) - Optimize performance
```bash
pmat prompt performance-optimization --format text
```

**When to use**: After initial implementation, to optimize.

**Constraints**:
- `make coverage` must complete in <10 min
- `make test-fast` must complete in <5 min
- Pre-commit tests must complete in <30 sec

### 8. **documentation** (MEDIUM) - Update documentation
```bash
pmat prompt documentation --format text
```

**When to use**: After completing features or making significant changes.

**Requirements**:
- 100% rustdoc coverage of public API
- Every function has example code that compiles
- Document panics, errors, safety invariants
- Performance characteristics documented
- Run `pmat validate-docs` to verify

### 9. **security-audit** (CRITICAL) - Security analysis
```bash
pmat prompt security-audit --format text
```

**When to use**: Before any release, especially for unsafe code.

**Focus Areas for Trueno**:
- All `unsafe` blocks in SIMD backends
- Memory access patterns (bounds checking)
- GPU buffer management
- WASM sandboxing

### 10. **clean-repo-cruft** (MEDIUM) - Remove temporary files
```bash
pmat prompt clean-repo-cruft --format text
```

**When to use**: Before commits, to keep repo clean.

## PMAT Analysis Commands

### Analyze Code Quality
```bash
# Technical Debt Grading (TDG)
pmat tdg

# Complexity analysis
pmat analyze complexity

# Find Self-Admitted Technical Debt
pmat analyze satd

# Find dead code
pmat analyze dead-code
```

### Repository Health
```bash
# Fast repo score (HEAD only)
pmat repo-score .

# Deep repo score (entire git history)
pmat repo-score . --deep
```

### Mutation Testing
```bash
# Run on entire codebase
pmat mutate --target src/

# Run on specific file
pmat mutate --file src/backend/simd/avx2.rs
```

## PMAT Scaffolding (Not Yet Used)

When ready to scaffold project structure:

```bash
# List available templates
pmat list

# Scaffold Rust CLI project
pmat scaffold project rust
```

## Trueno-Specific Workflow

### Phase 1: Foundation (Weeks 1-2)

1. **Initialize Cargo project** (manual):
   ```bash
   cargo init --lib
   ```

2. **Run quality gates setup**:
   ```bash
   pmat prompt code-coverage --format text
   # Follow instructions to set up coverage infrastructure
   ```

3. **Implement scalar baseline** (EXTREME TDD):
   ```bash
   # Write tests FIRST for Vector<f32>
   # - add(), mul(), dot(), sum(), max()
   # - Property tests (PROPTEST_CASES=100)
   # - Target: >90% coverage
   ```

4. **Verify quality gates**:
   ```bash
   pmat prompt quality-enforcement --format text
   # ALL gates must pass before commit
   ```

5. **Commit**:
   ```bash
   git add .
   git commit -m "feat: scalar baseline Vector<f32> with 95% coverage

Implements:
- Vector::add(), mul(), dot(), sum(), max()
- Property tests (100 cases per test)
- Unit tests for edge cases (empty, single element, non-aligned)

Coverage: 95.2%
TDG: A- (92/100)
Mutation: 82% kill rate

🤖 Generated with Claude Code
Co-Authored-By: Claude <noreply@anthropic.com>"
   ```

### Phase 2: SIMD Backends (Weeks 3-4)

1. **Implement SSE2 backend** (x86_64 baseline):
   ```bash
   pmat prompt code-coverage --format text
   # Module type: Backend (LOGIC)
   # Category: Backend (transpiler/codegen)
   # Target: 85% coverage
   # Technique: Golden file tests (output validation)
   ```

2. **Backend equivalence tests**:
   ```rust
   #[test]
   fn test_backend_equivalence() {
       let scalar = add_f32_scalar(&a, &b);
       let sse2 = unsafe { add_f32_sse2(&a, &b) };
       assert_eq!(scalar, sse2); // Must match exactly
   }
   ```

3. **Benchmark validation**:
   ```bash
   cargo bench -- add_vectors
   # Verify 2-4x speedup vs scalar
   ```

### Continuous Quality Loop

```bash
# 1. Check current state
pmat prompt continue --format text

# 2. Run TDG analysis
pmat tdg

# 3. Add tests to low-coverage modules
pmat prompt code-coverage --format text

# 4. Run quality gates
pmat prompt quality-enforcement --format text

# 5. If all pass, commit
git add . && git commit -m "..."

# 6. Repeat
```

## Key Principles

### Toyota Way Integration

- **Jidoka (Built-in Quality)**: Quality gates enforce >90% coverage, zero warnings
- **Kaizen (Continuous Improvement)**: Every optimization must prove ≥10% speedup
- **Genchi Genbutsu (Go and See)**: FFmpeg case study informs SIMD design
- **Andon Cord**: STOP THE LINE on any defect

### EXTREME TDD Rules

1. **RED-GREEN-REFACTOR**: Write failing test → Make it pass → Refactor
2. **Property tests**: PROPTEST_CASES=100 (not 5!)
3. **Mutation testing**: >80% kill rate
4. **Backend equivalence**: All backends produce identical results
5. **Zero unsafe in public API**: Safety via type system
6. **Coverage targets**:
   - Frontend (Parser/Lexer): 95%
   - Backend (Codegen): 85%
   - Runtime (Interpreter): 90%
   - API/CLI: 80%
   - Overall: >90%

### Time Constraints

- **make coverage**: <10 minutes
- **make test-fast**: <5 minutes
- **Pre-commit**: <30 seconds
- **Time-box**: 90 minutes per module

## Example Session

```bash
# 1. Start session
cd /home/noah/src/trueno
pmat prompt continue --format text

# 2. Check TDG
pmat tdg
# Output: Average TDG: 65/100 (C+)
# Hotspot: src/vector.rs (complexity: 18, coverage: 42%)

# 3. Add tests to hotspot
pmat prompt code-coverage --format text
# Module type: LOGIC (pure functions)
# Category: API/CLI
# Current: 42% coverage
# Target: 80%
# Technique: Property tests (100 cases)

# Write 15 tests...

# 4. Run coverage
make coverage
# Coverage: 42% → 78% (+36%)
# ROI: 36% / 15 tests = 2.4%/test (excellent!)

# 5. Run quality gates
pmat prompt quality-enforcement --format text
cargo build --all-features  # ✅
cargo clippy -- -D warnings  # ✅
cargo fmt -- --check         # ✅
make test-fast               # ✅ (3.2 minutes)
make coverage                # ✅ (78%, 8.1 minutes)
pmat mutate --file src/vector.rs  # ✅ (83% kill rate)
pmat tdg                     # ✅ (Average: 72/100, B-)

# 6. Commit
git add .
git commit -m "test: increase vector.rs coverage to 78% (+36%)

Added 15 property tests (100 cases each):
- Commutative property
- Associative property
- Distributive property
- Identity elements
- Inverse operations

Coverage: 42% → 78%
ROI: 2.4%/test
Mutation score: 83%
TDG: C+ → B- (65 → 72)

🤖 Generated with Claude Code
Co-Authored-By: Claude <noreply@anthropic.com>"

# 7. Continue
pmat prompt continue --format text
```

## Resources

- **PMAT Book**: https://paiml.github.io/pmat-book/
- **Trueno Specification**: `docs/specifications/initial-three-target-SIMD-GPU-WASM-spec.md`
- **CLAUDE.md**: Development guide for Claude Code
- **Toyota Way Code Review**: Specification section 16

## Summary

Use PMAT prompts to guide development with EXTREME TDD quality standards:
- **code-coverage**: Continuously achieve >90% coverage
- **continue**: Determine next best action
- **quality-enforcement**: Run all gates before commit
- **debug**: Root cause analysis with Five Whys

ALL quality gates must pass before ANY commit. Zero tolerance for defects.
