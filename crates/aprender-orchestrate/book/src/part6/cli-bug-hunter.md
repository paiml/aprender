# `batuta bug-hunter`

The `bug-hunter` command provides proactive bug hunting using multiple falsification-driven strategies. It implements Section 11 of the Popperian Falsification Checklist (BH-01 to BH-15).

## Philosophy

> "A theory that explains everything, explains nothing." — Karl Popper

Bug hunting operationalizes falsification: we systematically attempt to break code, not merely verify it works. Each mode represents a different strategy for falsifying the implicit claim "this code is correct."

## Usage

```bash
# LLM-augmented static analysis
batuta bug-hunter analyze .

# SBFL fault localization from coverage data
batuta bug-hunter hunt .

# Mutation-based invariant falsification
batuta bug-hunter falsify .

# Targeted unsafe Rust fuzzing
batuta bug-hunter fuzz .

# Hybrid concolic + SBFL deep analysis
batuta bug-hunter deep-hunt .

# Run all modes and combine results
batuta bug-hunter ensemble .
```

## Modes

### `analyze` - LLM-Augmented Static Analysis (LLIFT Pattern)

Combines traditional static analysis with pattern matching for common defect categories.

```bash
batuta bug-hunter analyze /path/to/project
batuta bug-hunter analyze . --format json
batuta bug-hunter analyze . --min-suspiciousness 0.7
```

### `hunt` - SBFL Without Failing Tests (SBEST Pattern)

Uses Spectrum-Based Fault Localization on coverage data to identify suspicious code regions.

```bash
# Basic hunt with default Ochiai formula
batuta bug-hunter hunt .

# Specify coverage file location
batuta bug-hunter hunt . --coverage ./lcov.info

# Use different SBFL formula
batuta bug-hunter hunt . --formula tarantula
batuta bug-hunter hunt . --formula dstar
```

Coverage file detection searches:
- `./lcov.info` (project root)
- `./target/coverage/lcov.info`
- `./target/llvm-cov/lcov.info`
- `$CARGO_TARGET_DIR/coverage/lcov.info`

### `falsify` - Mutation Testing (FDV Pattern)

Identifies mutation testing targets and weak test coverage.

```bash
batuta bug-hunter falsify .
batuta bug-hunter falsify . --timeout 60
```

### `fuzz` - Targeted Unsafe Fuzzing (FourFuzz Pattern)

Inventories unsafe blocks and identifies fuzzing targets.

```bash
batuta bug-hunter fuzz .
batuta bug-hunter fuzz . --duration 120
```

**Note**: For crates with `#![forbid(unsafe_code)]`, fuzz mode returns `BH-FUZZ-SKIPPED` (Info) instead of `BH-FUZZ-NOTARGETS` (Medium), since there's no unsafe code to fuzz.

### `deep-hunt` - Hybrid Analysis (COTTONTAIL Pattern)

Combines concolic execution analysis with SBFL for complex conditionals.

```bash
batuta bug-hunter deep-hunt .
batuta bug-hunter deep-hunt . --coverage ./lcov.info
```

### `ensemble` - Combined Results

Runs all modes and combines results with weighted scoring.

```bash
batuta bug-hunter ensemble .
batuta bug-hunter ensemble . --min-suspiciousness 0.5
```

## Advanced Features (BH-11 to BH-15)

### Spec-Driven Bug Hunting (BH-11)

Hunt bugs guided by specification files:

```bash
batuta bug-hunter spec . --spec docs/spec.md
batuta bug-hunter spec . --spec docs/spec.md --section "Authentication"
batuta bug-hunter spec . --spec docs/spec.md --update-spec
```

### Ticket-Scoped Hunting (BH-12)

Focus on areas defined by work tickets:

```bash
batuta bug-hunter ticket . --ticket GH-42
batuta bug-hunter ticket . --ticket PERF-001
```

### Cross-Stack Analysis (BH-16)

Scan multiple crates in the Sovereign AI Stack and generate consolidated reports:

```bash
# Scan all default crates (trueno, aprender, realizar, entrenar, repartir)
batuta bug-hunter stack --base /path/to/src

# Scan specific crates
batuta bug-hunter stack --base ~/src --crates trueno,aprender,realizar

# Generate GitHub issue body
batuta bug-hunter stack --base ~/src --issue

# JSON output for CI/CD
batuta bug-hunter stack --base ~/src --format json
```

Example output:
```
╔══════════════════════════════════════════════════════════════════════════╗
║           CROSS-STACK BUG ANALYSIS - SOVEREIGN AI STACK               ║
╚══════════════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────────────┐
│ STACK DEPENDENCY CHAIN: trueno → aprender → realizar → entrenar        │
└─────────────────────────────────────────────────────────────────────────┘

SUMMARY BY CRATE:
┌──────────────┬────────┬──────────┬──────┬────────┬──────┬────────┬──────┬────────┬────────┐
│ Crate        │ Total  │ Critical │ High │ GPU    │ Debt │ Test   │ Mem  │ Ctrct  │ Parity │
├──────────────┼────────┼──────────┼──────┼────────┼──────┼────────┼──────┼────────┼────────┤
│ trueno       │     64 │        0 │   64 │      0 │    4 │      1 │   57 │      0 │      0 │
│ aprender     │    116 │       21 │   95 │      1 │  105 │      1 │    1 │      0 │      0 │
│ realizar     │    373 │       20 │  353 │     33 │   37 │     12 │  242 │      0 │      0 │
│ entrenar     │     57 │        1 │   56 │      0 │   23 │      2 │   22 │      0 │      0 │
│ repartir     │      2 │        0 │    2 │      0 │    0 │      0 │    0 │      0 │      0 │
├──────────────┼────────┼──────────┼──────┼────────┼──────┼────────┼──────┼────────┼────────┤
│ TOTAL        │    612 │       42 │  570 │     34 │  169 │     16 │  322 │      0 │      0 │
└──────────────┴────────┴──────────┴──────┴────────┴──────┴────────┴──────┴────────┴────────┘

CROSS-STACK INTEGRATION RISKS:

  1. GPU Kernel Chain (trueno SIMD → realizar CUDA):
     • 34 GPU kernel bugs detected
     • Impact: Potential performance degradation or kernel failures

  2. Hidden Technical Debt:
     • 169 euphemism patterns (placeholder, stub, etc.)
     • Impact: Incomplete implementations may cause failures

  3. Test Debt:
     • 16 tests ignored or removed
     • Impact: Known bugs not being caught by CI

  4. Contract Verification Gaps:
     • N contract gaps (unbound, partial, missing proofs)
     • Impact: Kernel correctness claims lack formal verification

  5. Model Parity Gaps:
     • N parity gaps (missing oracles, failed claims)
     • Impact: Model conversion pipeline may produce incorrect results
```

## Output Formats

```bash
# Text output (default)
batuta bug-hunter analyze .

# JSON output
batuta bug-hunter analyze . --format json

# Markdown output
batuta bug-hunter analyze . --format markdown
```

## Finding Categories

| Category | Description |
|----------|-------------|
| MemorySafety | Pointer issues, buffer overflows, unsafe blocks |
| LogicErrors | Off-by-one, boundary conditions, unwrap/panic |
| ConcurrencyBugs | Race conditions, deadlocks |
| ConfigurationErrors | Missing configs, wrong settings |
| TypeErrors | Type mismatches, invalid casts |
| GpuKernelBugs | CUDA/PTX kernel issues, dimension limits |
| SilentDegradation | Silent fallbacks that hide failures |
| TestDebt | Skipped/ignored tests indicating known bugs |
| HiddenDebt | Euphemisms hiding tech debt (placeholder, stub, demo) |
| ContractGap | Contract verification gaps (unbound, partial, missing proofs) |
| ModelParityGap | Model parity gaps (missing oracles, failed claims, incomplete ops) |

### GPU/CUDA Kernel Bug Patterns

Bug-hunter detects GPU kernel issues documented in code comments:

| Pattern | Severity | Suspiciousness | Description |
|---------|----------|----------------|-------------|
| `CUDA_ERROR` | Critical | 0.9 | CUDA runtime errors |
| `INVALID_PTX` | Critical | 0.95 | Invalid PTX generation |
| `PTX error` | Critical | 0.9 | PTX compilation errors |
| `kernel fail` | High | 0.8 | Kernel execution failures |
| `cuBLAS fallback` | High | 0.7 | cuBLAS fallback paths |
| `cuDNN fallback` | High | 0.7 | cuDNN fallback paths |
| `hidden_dim >=` | High | 0.7 | Dimension-related GPU bugs |

### Silent Degradation Patterns

Detects code that silently swallows errors or degrades performance:

| Pattern | Severity | Suspiciousness | Description |
|---------|----------|----------------|-------------|
| `.unwrap_or_else(\|_\|` | High | 0.7 | Silent error swallowing |
| `if let Err(_) =` | Medium | 0.5 | Unchecked error handling |
| `Err(_) => {}` | High | 0.75 | Empty error handlers |
| `// fallback` | Medium | 0.5 | Documented fallback paths |
| `// degraded` | High | 0.7 | Documented degradation |

### Test Debt Patterns

Detects skipped or removed tests that indicate known bugs:

| Pattern | Severity | Suspiciousness | Description |
|---------|----------|----------------|-------------|
| `#[ignore]` | High | 0.7 | Ignored tests |
| `// broken` | High | 0.8 | Known broken tests |
| `// fails` | High | 0.75 | Known failing tests |
| `test removed` | Critical | 0.9 | Removed tests |
| `were removed` | Critical | 0.9 | Tests removed from codebase |
| `tests hang` | Critical | 0.9 | Hanging test documentation |
| `hang during` | High | 0.8 | Compilation/runtime hangs |

### Hidden Debt Patterns (Euphemisms)

Detects euphemisms that hide technical debt (addresses [PMAT #149](https://github.com/paiml/paiml-mcp-agent-toolkit/issues/149)):

| Pattern | Severity | Suspiciousness | Description |
|---------|----------|----------------|-------------|
| `placeholder` | High | 0.75 | Placeholder implementations |
| `stub` | High | 0.7 | Stub functions |
| `dummy` | High | 0.7 | Dummy values/objects |
| `not implemented` | Critical | 0.9 | Unimplemented features |
| `unimplemented` | Critical | 0.9 | Unimplemented macro usage |
| `demo only` | High | 0.8 | Demo-only code in production |
| `for demonstration` | High | 0.75 | Demo code |
| `simplified` | Medium | 0.6 | Simplified implementations |
| `temporary` | Medium | 0.6 | Temporary solutions |
| `hardcoded` | Medium | 0.5 | Hardcoded values |
| `workaround` | Medium | 0.6 | Workarounds for issues |
| `quick fix` | High | 0.7 | Quick fixes |
| `bandaid` | High | 0.7 | Band-aid solutions |
| `kludge` | High | 0.75 | Kludge code |
| `tech debt` | High | 0.8 | Acknowledged tech debt |

**Example detection** (from aprender placeholder bug):

```rust
/// This is a placeholder that demonstrates the tracing flow.
fn run_safetensors_generation(...) {
    let placeholder_logits: Vec<f32> = vec![0.0; vocab_size];  // ← HiddenDebt: placeholder
    let token = (last_input.wrapping_add(i as u32)) % (vocab_size as u32);  // garbage output!
}
```

### Contract Verification Gap Patterns (BH-26)

Analyzes [provable-contracts](../part6/cli-stack.md) binding registries and contract YAML files to find verification gaps. Auto-discovers `../provable-contracts/contracts/` or accepts an explicit path.

```bash
# Auto-discover provable-contracts in sibling directory
batuta bug-hunter analyze . --contracts-auto

# Explicit path
batuta bug-hunter analyze . --contracts /path/to/provable-contracts/contracts

# Combined with ensemble
batuta bug-hunter ensemble . --contracts-auto
```

**Checks performed:**

| Check | Finding ID | Severity | Suspiciousness | Description |
|-------|-----------|----------|----------------|-------------|
| Binding `not_implemented` | `BH-CONTRACT-NNNN` | High | 0.8 | Kernel binding has no implementation |
| Binding `partial` | `BH-CONTRACT-NNNN` | Medium | 0.6 | Kernel binding is partially implemented |
| Unbound contract | `BH-CONTRACT-NNNN` | Medium | 0.5 | Contract YAML has no binding reference |
| Low obligation coverage | `BH-CONTRACT-NNNN` | Low | 0.4 | <50% of proof obligations have falsification tests |

### Model Parity Gap Patterns (BH-27)

Analyzes [tiny-model-ground-truth](../part6/cli-stack.md) directory for parity gaps in model conversion testing. Auto-discovers `../tiny-model-ground-truth/` or accepts an explicit path.

```bash
# Auto-discover tiny-model-ground-truth in sibling directory
batuta bug-hunter analyze . --model-parity-auto

# Explicit path
batuta bug-hunter analyze . --model-parity /path/to/tiny-model-ground-truth

# Combined with contract gaps
batuta bug-hunter analyze . --contracts-auto --model-parity-auto
```

**Checks performed:**

| Check | Finding ID | Severity | Suspiciousness | Description |
|-------|-----------|----------|----------------|-------------|
| Missing oracle file | `BH-PARITY-NNNN` | Medium | 0.6 | Oracle output for model/prompt not generated |
| Missing oracle directory | `BH-PARITY-NNNN` | High | 0.8 | No `oracle/` directory found |
| FAIL claim | `BH-PARITY-NNNN` | High | 0.8 | CLAIMS.md contains a failed claim |
| Deferred claim | `BH-PARITY-NNNN` | Low | 0.4 | CLAIMS.md claim is deferred |
| Missing oracle-ops | `BH-PARITY-NNNN` | Low | 0.4 | Oracle-ops directory missing or empty |

**Expected models:** smollm-135m, qwen2-0.5b, gpt2-124m
**Expected prompts:** arithmetic, code, completion, greeting
**Expected ops:** convert, quantize, finetune, merge, prune

### Suspiciousness Filtering

BH-26/27 findings respect `--min-suspiciousness` filtering. For example, `--min-suspiciousness 0.7` will show only `not_implemented` bindings (0.8) and FAIL claims (0.8), filtering out `partial` (0.6), unbound contracts (0.5), and low-severity items (0.4).

```bash
# Only high-suspiciousness contract/parity findings
batuta bug-hunter analyze . --contracts-auto --model-parity-auto --min-suspiciousness 0.7

# Stack-wide with contract/parity flags
batuta bug-hunter stack --contracts-auto --model-parity-auto
```

## Severity Levels

| Severity | Suspiciousness | Action Required |
|----------|----------------|-----------------|
| Critical | 0.9+ | Immediate fix |
| High | 0.7-0.9 | Fix before release |
| Medium | 0.5-0.7 | Review and address |
| Low | 0.3-0.5 | Consider fixing |
| Info | 0.0-0.3 | Informational |

## Example Output

```
Bug Hunter Report
──────────────────────────────────────────────────────────────────────────
Mode: Analyze  Findings: 1952  Duration: 50666ms
scan=50666ms
Severity: 0C 301H 730M 1065L 0I

Category Distribution:
  LogicErrors            ████████████████████ 1611
  MemorySafety           ███ 242
  SilentDegradation      █ 49
  GpuKernelBugs           37
  TestDebt                12

Hotspot Files:
  src/api/tests/part_16.rs ███████████████ 136
  src/api/tests/part_01.rs █████████████ 122
  src/cuda/executor/tests.rs ██████ 55

Findings:
──────────────────────────────────────────────────────────────────────────
[C] BH-PAT-1689 ██████████ 0.95 src/cuda/executor/tests.rs:7562
    Pattern: INVALID_PTX
    // Test removed to avoid CUDA_ERROR_INVALID_PTX
[C] BH-PAT-1686 █████████░ 0.90 src/cuda/executor/tests.rs:6026
    Pattern: were removed
    // were removed because they hang during kernel compilation
[H] BH-PAT-0001 ███████░░░ 0.70 src/api/gpu_handlers.rs:1413
    Pattern: .unwrap_or_else(|_|
    .unwrap_or_else(|_| r#"{"error":"serialization failed"}"#.to_string())
──────────────────────────────────────────────────────────────────────────
```

## Real-World Example: GPU Kernel Bug Detection

Bug-hunter detected critical CUDA kernel issues in the realizar inference runtime:

```bash
$ batuta bug-hunter analyze ../realizar --format json | \
    jq '.findings | map(select(.category == "GpuKernelBugs" or .category == "TestDebt")) |
        sort_by(-.suspiciousness) | .[:5]'
```

| Location | Pattern | Severity | Description |
|----------|---------|----------|-------------|
| `tests.rs:7562` | `INVALID_PTX` | Critical | `fused_qkv_into` test removed |
| `tests.rs:9099` | `INVALID_PTX` | Critical | `fused_gate_up_into` test removed |
| `tests.rs:10629` | `INVALID_PTX` | Critical | `q8_quantize_async` skipped |
| `tests.rs:6026` | `were removed` | Critical | COV-013 tests removed due to hangs |
| `layer.rs:1177` | `PTX error` | Critical | PTX generation error documented |

These findings correlate with the root cause analysis in [apr-model-qa-playbook#5](https://github.com/paiml/apr-model-qa-playbook/issues/5): broken CUDA PTX kernels causing 0.4-0.8 tok/s GPU throughput instead of expected 50+ tok/s.

## New Features (2026)

### Diff Mode

Compare current findings against a baseline to show only new issues:

```bash
# Compare against a git branch
batuta bug-hunter diff --base main

# Compare against a time period (last 7 days)
batuta bug-hunter diff --since 7d

# Save current findings as the new baseline
batuta bug-hunter diff --save-baseline
```

### Trend Tracking

Track tech debt trends over time with snapshots:

```bash
# Show trend over last 12 weeks
batuta bug-hunter trend --weeks 12

# Save a snapshot for trend tracking
batuta bug-hunter trend --snapshot

# JSON output for dashboards
batuta bug-hunter trend --format json
```

### Auto-Triage

Group related findings by root cause (directory + pattern):

```bash
batuta bug-hunter triage

# Output:
# ROOT CAUSE GROUPS:
#   src/api/ + unwrap() → 23 findings
#   src/cuda/ + INVALID_PTX → 5 findings
#   src/model/ + placeholder → 12 findings
```

### Git Blame Integration

Each finding now includes author information:

```
[H] BH-PAT-0014 ████████░░ 0.75 src/oracle/generator.rs:150
    Pattern: placeholder
    // STUB: Test placeholder for {{id}}
    Blame: Noah Gift (b40b402) 2026-02-03
```

### Coverage-Based Hotpath Weighting

Boost suspiciousness for findings in uncovered code paths:

```bash
# Use LCOV coverage data
batuta bug-hunter analyze --coverage lcov.info --coverage-weight 0.7

# Coverage factor:
# - Uncovered (0 hits): +50% boost
# - Low coverage (1-5 hits): +20% boost
# - Medium coverage (6-20 hits): no change
# - High coverage (>20 hits): -30% reduction
```

### PMAT Quality Weighting

Weight findings by code quality metrics:

```bash
batuta bug-hunter analyze --pmat-quality --quality-weight 0.5

# Low-quality code (TDG < 50) gets boosted suspiciousness
# High-quality code (TDG > 50) gets reduced suspiciousness
```

### Allowlist Configuration

Suppress intentional patterns via `.pmat/bug-hunter.toml`:

```toml
[[allow]]
file = "src/optim/*.rs"
pattern = "unimplemented"
reason = "Batch optimizers don't support step()"

[[allow]]
file = "src/test_helpers.rs"
pattern = "*"
reason = "Test helper module"

[[patterns]]
pattern = "PERF-TODO"
category = "PerformanceDebt"
severity = "High"
suspiciousness = 0.8
```

### Multi-Language Support

Bug-hunter now detects patterns in Python, TypeScript, and Go:

**Python patterns:**
| Pattern | Severity | Description |
|---------|----------|-------------|
| `eval(` | Critical | Code injection vulnerability |
| `except:` | High | Bare exception (catches everything) |
| `pickle.loads` | High | Deserialization vulnerability |
| `shell=True` | High | Shell injection risk |
| `raise NotImplementedError` | High | Unimplemented feature |

**TypeScript patterns:**
| Pattern | Severity | Description |
|---------|----------|-------------|
| `any` | Medium | Type safety bypass |
| `as any` | High | Explicit type bypass |
| `@ts-ignore` | High | Type check suppression |
| `innerHTML` | High | XSS vulnerability |
| `it.skip` | High | Skipped test |

**Go patterns:**
| Pattern | Severity | Description |
|---------|----------|-------------|
| `_ = err` | Critical | Ignored error |
| `panic(` | High | Crash on error |
| `exec.Command(` | High | Command injection risk |
| `interface{}` | Medium | Type safety bypass |

```bash
# Scans .rs, .py, .ts, .tsx, .js, .jsx, .go files automatically
batuta bug-hunter analyze /path/to/polyglot/project
```

## Caching & Performance

Bug-hunter uses FNV-1a cache keys with mtime invalidation for fast repeated runs:

| Metric | Cold Cache | Warm Cache | Speedup |
|--------|------------|------------|---------|
| Analysis time | ~50s | ~30ms | 560x |

Cache location: `.pmat/bug-hunter-cache/`

Cache invalidation triggers:
- Source file content changed (mtime check)
- Hunt mode changed
- Configuration changed (targets, min_suspiciousness, contracts/parity flags)

### Parallel Scanning

Bug-hunter uses `std::thread::scope` for parallel file scanning:
- Files are chunked across available CPU cores
- Each thread scans patterns independently
- Results are merged with globally unique `BH-PAT-XXXX` IDs

## Integration with CI

```yaml
- name: Bug Hunter Analysis
  run: |
    batuta bug-hunter ensemble . --format json > findings.json
    # Fail if critical findings exist
    jq -e '[.findings[] | select(.severity == "Critical")] | length == 0' findings.json

- name: GPU Kernel Bug Check
  run: |
    batuta bug-hunter analyze . --format json | \
      jq -e '[.findings[] | select(.category == "GpuKernelBugs")] | length == 0'
```

## Demo

Run the interactive demo to explore all bug-hunter patterns:

```bash
cargo run --example bug_hunter_demo --features native
```

## Related Commands

- [`batuta falsify`](./cli-falsify.md) - Full Popperian Falsification Checklist
- [`batuta analyze`](./cli-analyze.md) - Project analysis
- [`batuta stack quality`](./cli-stack.md) - Stack-wide quality metrics
