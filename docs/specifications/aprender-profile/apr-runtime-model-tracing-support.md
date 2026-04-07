# APR Runtime Model Tracing Support Specification

**Version:** 1.0.0
**Date:** 2025-12-16
**Status:** Draft
**GitHub Issues:** [renacer#22](https://github.com/paiml/renacer/issues/22), [aprender#119](https://github.com/paiml/aprender/issues/119)

---

## Table of Contents

1. [Abstract](#1-abstract)
2. [Design Principles](#2-design-principles)
3. [Validate Subcommand](#3-validate-subcommand)
4. [Golden Trace Format](#4-golden-trace-format)
5. [APR Runtime Integration](#5-apr-runtime-integration)
6. [Model Execution Tracing](#6-model-execution-tracing)
7. [CLI Specification](#7-cli-specification)
8. [Configuration](#8-configuration)
9. [Error Handling](#9-error-handling)
10. [Quality Gates](#10-quality-gates)
11. [Implementation Roadmap](#11-implementation-roadmap)
12. [References](#12-references)
13. [APR 100-Point QA Checklist Integration](#13-apr-100-point-qa-checklist-integration)
14. [SDK Interface](#14-sdk-interface)
15. [Cross-Tool Validation Protocol](#15-cross-tool-validation-protocol)

---

## 1. Abstract

This specification defines the integration between **renacer** (system call tracer) and **APR** (Aprender Portable Representation) runtime for comprehensive model tracing and validation. The goal is to enable:

- **Golden trace comparison**: Validate execution traces against known-good baselines
- **Model execution tracing**: Trace inference operations at the syscall and compute-block level
- **Regression detection**: Automatically detect performance and behavioral regressions
- **Canary validation**: Verify model outputs match expected reference outputs

This specification implements the `validate` subcommand (Issue #22) and aligns with APR-SPEC.md Section 4.10 (Trace Command) and Section 4.4 (Validate Command).

---

## 2. Design Principles

### 2.1 Toyota Way Alignment

| Principle | Application |
|-----------|-------------|
| **Genchi Genbutsu** | Compare actual traces, not abstractions |
| **Jidoka** | Stop on regression detection (Andon principle) |
| **Poka-Yoke** | Error-proof validation with strict schema |
| **Muda** | Eliminate waste by caching golden traces |
| **Kaizen** | Continuous improvement via trace evolution |

### 2.2 Core Requirements

1. **Deterministic Comparison**: Trace comparison must be reproducible across runs
2. **Tolerance-Aware**: Support configurable tolerance for timing variations
3. **Semantic Equivalence**: Focus on behavioral equivalence, not exact byte matching
4. **Incremental Adoption**: Work without APR files (pure syscall tracing)
5. **CI/CD Integration**: Exit codes and machine-readable output for automation

### 2.3 Non-Goals

- Real-time model debugging (use `apr trace --interactive` instead)
- Training-time tracing (inference only)
- Cross-architecture comparison (same platform required)

---

## 3. Validate Subcommand

### 3.1 Overview

The `validate` subcommand compares execution traces against golden baselines, detecting performance and behavioral regressions.

### 3.2 Basic Usage

```bash
# Validate against baseline
renacer validate --baseline golden_traces/ -- cargo test

# Generate new baseline
renacer validate --generate golden_traces/ -- cargo test

# With tolerance for timing variations
renacer validate --baseline golden_traces/ --tolerance 20% -- cargo test

# Validate APR model execution
renacer validate --baseline golden_traces/ --apr-model model.apr -- apr run model.apr --input test.wav
```

### 3.3 Command Syntax

```
renacer validate [OPTIONS] -- <COMMAND>...

OPTIONS:
    --baseline <DIR>          Path to golden trace baseline directory
    --generate <DIR>          Generate new baseline to directory
    --tolerance <PERCENT>     Timing tolerance percentage (default: 10%)
    --strict                  Fail on any deviation (no tolerance)
    --apr-model <FILE>        APR model file for model-aware validation
    --output <FORMAT>         Output format: text, json, junit (default: text)
    --fail-fast               Stop on first regression
    --ignore-timing           Compare behavior only, ignore timing
    --update-baseline         Update baseline with new values on pass

ARGS:
    <COMMAND>...              Command to trace and validate
```

### 3.4 Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Validation passed |
| 1 | Validation failed (regression detected) |
| 2 | Baseline not found |
| 3 | Invalid baseline format |
| 4 | Command execution error |
| 5 | Configuration error |

### 3.5 Validation Modes

#### Behavior Validation (Default)

Compares syscall sequences, arguments, and return values:

```bash
$ renacer validate --baseline golden/ -- ./my_binary

=== Renacer Validation Report ===

Baseline: golden/
Command:  ./my_binary
Mode:     Behavior + Timing

Syscall Comparison:
  Total syscalls:     1,234 (baseline: 1,234) ✓
  Sequence match:     100% ✓
  Argument match:     99.8% ✓ (2 minor deviations)

Timing Comparison (tolerance: 10%):
  open():    1.2ms (baseline: 1.1ms, +9.1%) ✓
  read():    45.3ms (baseline: 42.0ms, +7.9%) ✓
  write():   12.1ms (baseline: 15.0ms, -19.3%) ⚠ REGRESSED

RESULT: FAILED (1 timing regression)
```

#### Strict Mode

Zero tolerance for any deviation:

```bash
$ renacer validate --baseline golden/ --strict -- ./my_binary

FAILED: Strict mode - 2 syscall argument deviations detected
  Line 423: open("/tmp/foo") vs open("/tmp/bar")
  Line 891: read(fd=3, ...) vs read(fd=4, ...)
```

#### Timing-Only Mode

Ignore behavior, focus on performance:

```bash
$ renacer validate --baseline golden/ --behavior-ignore -- ./my_binary

Timing Validation (behavior ignored):
  Total duration:     125.3ms (baseline: 118.2ms, +6.0%) ✓
  P95 syscall:        2.1ms (baseline: 1.9ms, +10.5%) ⚠
  P99 syscall:        8.4ms (baseline: 7.2ms, +16.7%) ✗ REGRESSED

RESULT: FAILED (P99 regression)
```

#### Smart Regression Detection (Optional)

Uses Isolation Forest (see `src/isolation_forest.rs`) to detect anomalies in multi-dimensional metrics (timing, memory, syscall frequency) rather than simple thresholds:

```bash
$ renacer validate --baseline golden/ --detect-anomalies -- ./my_binary

Anomaly Detection (Isolation Forest):
  Score: 0.72 (threshold: 0.60) ✗ ANOMALY DETECTED
  Contributing factors:
    - write() frequency (unexpected spike)
    - memory_usage (gradual increase)
```

---

## 4. Golden Trace Format

### 4.1 Directory Structure

```
golden_traces/
├── manifest.json           # Trace manifest with metadata
├── syscalls.trace          # Compressed syscall trace
├── timing.stats            # Statistical timing data
├── compute_blocks.trace    # Optional: Trueno compute block traces
└── apr_model/              # Optional: APR model-specific data
    ├── tensor_stats.json   # Expected tensor statistics
    ├── canary_outputs.json # Expected model outputs
    └── layer_traces/       # Per-layer activation traces
```

### 4.2 Manifest Format

```json
{
  "version": "1.0.0",
  "renacer_version": "0.1.0",
  "created_at": "2025-12-16T12:00:00Z",
  "platform": {
    "os": "linux",
    "arch": "x86_64",
    "kernel": "6.8.0-87-generic"
  },
  "command": ["cargo", "test"],
  "environment": {
    "RUST_BACKTRACE": "1"
  },
  "apr_model": {
    "path": "model.apr",
    "checksum": "sha256:abc123...",
    "tensor_count": 167,
    "model_type": "whisper"
  },
  "statistics": {
    "total_syscalls": 1234,
    "total_duration_ms": 125.3,
    "unique_syscalls": 42
  },
  "tolerance": {
    "timing_percent": 10.0,
    "syscall_sequence": "exact",
    "argument_match": "fuzzy"
  }
}
```

### 4.3 Syscall Trace Format

Binary format for efficient storage and comparison:

```
┌─────────────────────────────────────────────────────────────┐
│ Header (32 bytes)                                           │
│   Magic: "RNTR" (4 bytes)                                   │
│   Version: u32                                              │
│   Flags: u32 (compressed, has_timing, has_args)             │
│   Entry count: u64                                          │
│   Checksum: u64                                             │
├─────────────────────────────────────────────────────────────┤
│ Syscall Entries (variable)                                  │
│   Each entry:                                               │
│     timestamp_ns: u64                                       │
│     syscall_nr: u32                                         │
│     duration_ns: u64                                        │
│     return_value: i64                                       │
│     arg_count: u8                                           │
│     args: [u64; arg_count]                                  │
│     string_args: [String; ...]  (if has_args flag)          │
├─────────────────────────────────────────────────────────────┤
│ Footer (16 bytes)                                           │
│   Entry checksum: u64                                       │
│   Magic end: "RTNR" (4 bytes)                               │
│   Reserved: u32                                             │
└─────────────────────────────────────────────────────────────┘
```

### 4.4 Timing Statistics Format

```json
{
  "version": "1.0.0",
  "overall": {
    "total_duration_ns": 125300000,
    "syscall_count": 1234
  },
  "by_syscall": {
    "open": {
      "count": 45,
      "total_ns": 54000000,
      "mean_ns": 1200000,
      "std_ns": 150000,
      "min_ns": 800000,
      "max_ns": 2100000,
      "p50_ns": 1150000,
      "p95_ns": 1800000,
      "p99_ns": 2000000
    }
  },
  "by_phase": {
    "initialization": {
      "duration_ns": 25000000,
      "syscalls": ["mmap", "mprotect", "open"]
    },
    "inference": {
      "duration_ns": 95000000,
      "syscalls": ["read", "write", "futex"]
    }
  }
}
```

---

## 5. APR Runtime Integration

### 5.1 Model-Aware Validation

When an APR model file is provided, renacer enables model-specific validation:

```bash
renacer validate --baseline golden/ --apr-model whisper.apr -- apr run whisper.apr --input test.wav
```

### 5.2 Tensor Statistics Validation

Cross-references with APR-SPEC.md Section 10.9 (Expected Tensor Statistics):

```json
{
  "tensor_validation": {
    "encoder.layer_norm.weight": {
      "expected_mean": [0.5, 3.0],
      "expected_std": [0.1, 1.0],
      "actual_mean": 1.48,
      "actual_std": 0.32,
      "status": "pass"
    },
    "decoder.layer_norm.weight": {
      "expected_mean": [0.5, 3.0],
      "expected_std": [0.1, 1.0],
      "actual_mean": 11.1,
      "actual_std": 0.21,
      "status": "FAIL",
      "message": "Mean outside expected range [0.5, 3.0]"
    }
  }
}
```

### 5.3 Canary Output Validation

Validates model outputs match expected reference outputs:

```bash
$ renacer validate --baseline golden/ --apr-model whisper.apr --canary -- apr run whisper.apr --input test.wav

=== Canary Validation ===

Input: test.wav
Expected: "The quick brown fox jumps over the lazy dog"
Actual:   "The quick brown fox jumps over the lazy dog"

Token accuracy:     100%
Character error:    0.0%
Cosine similarity:  1.0000

CANARY: PASSED
```

### 5.4 Layer-by-Layer Tracing

For debugging model regressions, trace activations at each layer:

```bash
renacer validate --baseline golden/ --apr-model model.apr --trace-layers -- apr run model.apr

Layer Trace Comparison:
  encoder.conv1:        ✓ (drift: 0.001)
  encoder.conv2:        ✓ (drift: 0.002)
  encoder.layer_norm:   ✗ (drift: 0.089) ← DIVERGENCE POINT
  decoder.layers.0:     ✗ (drift: 0.234) ← propagated error
```

---

## 6. Model Execution Tracing

### 6.1 Trace Points

Renacer integrates with APR runtime at these trace points:

| Trace Point | Description | Span Name |
|-------------|-------------|-----------|
| Model load | Loading APR file from disk | `apr.model.load` |
| Metadata parse | Parsing JSON metadata | `apr.metadata.parse` |
| Tensor load | Loading tensor data | `apr.tensor.load.{name}` |
| Inference start | Beginning inference | `apr.inference.start` |
| Layer execution | Individual layer forward pass | `apr.layer.{name}` |
| Inference end | Completing inference | `apr.inference.end` |
| Output encode | Encoding output (e.g., tokens) | `apr.output.encode` |

### 6.2 Integration with OTLP Export

Traces can be exported to OpenTelemetry backends:

```bash
renacer validate --baseline golden/ \
  --otlp-endpoint http://localhost:4317 \
  --apr-model model.apr \
  -- apr run model.apr --input test.wav
```

Span hierarchy:

```
renacer.trace (root)
└── apr.model.load
    ├── apr.metadata.parse
    └── apr.tensor.load
        ├── apr.tensor.load.encoder.conv1.weight
        ├── apr.tensor.load.encoder.conv2.weight
        └── ...
└── apr.inference.start
    ├── apr.layer.encoder.conv1
    ├── apr.layer.encoder.conv2
    ├── apr.layer.encoder.layers.0
    │   ├── apr.layer.encoder.layers.0.self_attn
    │   ├── apr.layer.encoder.layers.0.ffn
    │   └── apr.layer.encoder.layers.0.layer_norm
    └── ...
└── apr.inference.end
└── apr.output.encode
```

### 6.3 Compute Block Integration (CPU/GPU/HPU)

Integrates with Trueno compute block tracing (see `trueno-tracing-integration-spec.md`) and Renacer's hardware tracers (`src/gpu_tracer.rs`, `src/hpu.rs`):

```bash
renacer validate --baseline golden/ \
  --trace-compute \
  --trace-gpu \
  -- apr run model.apr --input test.wav

Compute Block Validation:
  calculate_mel_spectrogram (CPU): ✓ (2.3ms vs 2.1ms baseline, +9.5%)
  encoder_forward (GPU):           ✓ (45.2ms vs 42.0ms baseline, +7.6%)
  gemm_kernel_123 (CUDA):          ✓ (120us vs 115us baseline, +4.3%)
  decoder_forward (HPU):           ✗ (89.4ms vs 52.0ms baseline, +71.9%) REGRESSED
```

---

## 7. CLI Specification

### 7.1 New CLI Arguments

Add to `src/cli.rs`:

```rust
/// Validate traces against golden baseline
#[arg(long = "validate-baseline", value_name = "DIR")]
pub validate_baseline: Option<String>,

/// Generate golden baseline to directory
#[arg(long = "validate-generate", value_name = "DIR")]
pub validate_generate: Option<String>,

/// Timing tolerance for validation (default: 10%)
#[arg(long = "validate-tolerance", value_name = "PERCENT", default_value = "10")]
pub validate_tolerance: f32,

/// Strict validation mode (zero tolerance)
#[arg(long = "validate-strict")]
pub validate_strict: bool,

/// APR model file for model-aware validation
#[arg(long = "apr-model", value_name = "FILE")]
pub apr_model: Option<String>,

/// Enable canary output validation
#[arg(long = "validate-canary")]
pub validate_canary: bool,

/// Trace model layers for debugging
#[arg(long = "trace-layers")]
pub trace_layers: bool,

/// Output format for validation results
#[arg(long = "validate-output", value_name = "FORMAT", default_value = "text")]
pub validate_output: ValidationOutputFormat,
```

### 7.2 Output Formats

#### Text (Default)

Human-readable output for terminal use.

#### JSON

Machine-readable for CI/CD integration:

```json
{
  "result": "failed",
  "baseline": "golden_traces/",
  "command": ["apr", "run", "model.apr"],
  "validation": {
    "syscall_match": true,
    "timing_match": false,
    "canary_match": true
  },
  "regressions": [
    {
      "type": "timing",
      "syscall": "futex",
      "baseline_ms": 12.0,
      "actual_ms": 18.5,
      "delta_percent": 54.2
    }
  ],
  "duration_ms": 125.3
}
```

#### JUnit XML

For CI systems (Jenkins, GitHub Actions):

```xml
<?xml version="1.0" encoding="UTF-8"?>
<testsuite name="renacer-validate" tests="3" failures="1" time="0.125">
  <testcase name="syscall_sequence" classname="renacer.validate" time="0.001"/>
  <testcase name="timing_validation" classname="renacer.validate" time="0.001">
    <failure message="Timing regression detected">
      futex: 18.5ms (baseline: 12.0ms, +54.2%)
    </failure>
  </testcase>
  <testcase name="canary_validation" classname="renacer.validate" time="0.123"/>
</testsuite>
```

---

## 8. Configuration

### 8.1 renacer.toml Configuration

```toml
# renacer.toml - Configuration for validation

[golden]
enabled = true
directory = "golden_traces"
tolerance_percent = 20

[golden.timing]
# Per-syscall tolerance overrides
futex = 50       # Futex timing is highly variable
read = 10
write = 10
open = 20

[golden.behavior]
# Argument matching rules
arg_match = "fuzzy"   # exact, fuzzy, ignore
path_normalization = true
fd_normalization = true

[apr]
# APR model validation settings
validate_tensor_stats = true
validate_canary = true
layer_trace = false

[apr.tensor_tolerances]
# Override default tensor stat tolerances
layer_norm_mean_range = [0.5, 3.0]
layer_norm_std_max = 1.0
embedding_mean_max = 0.1
```

### 8.2 Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `RENACER_GOLDEN_DIR` | Default golden trace directory | `./golden_traces` |
| `RENACER_TOLERANCE` | Default timing tolerance | `10` |
| `RENACER_APR_MODEL` | Default APR model path | (none) |
| `RENACER_VALIDATE_OUTPUT` | Default output format | `text` |

---

## 9. Error Handling

### 9.1 Error Categories

| Code | Category | Description |
|------|----------|-------------|
| V001 | BASELINE | Baseline directory not found |
| V002 | BASELINE | Invalid baseline manifest |
| V003 | BASELINE | Baseline version mismatch |
| V004 | COMPARE | Syscall count mismatch |
| V005 | COMPARE | Syscall sequence mismatch |
| V006 | TIMING | Timing regression detected |
| V007 | CANARY | Canary output mismatch |
| V008 | APR | APR model not found |
| V009 | APR | Tensor statistics violation |
| V010 | CONFIG | Invalid configuration |

### 9.2 Error Messages

```bash
$ renacer validate --baseline nonexistent/ -- echo test

ERROR [V001]: Baseline directory not found
  Path: nonexistent/

  Troubleshooting:
    1. Generate a baseline first: renacer validate --generate golden/ -- <command>
    2. Check the path spelling
    3. Ensure the directory contains a valid manifest.json
```

---

## 10. Quality Gates

### 10.1 Test Coverage Requirements

```toml
# .pmat-gates.toml
[renacer-validate]
test_coverage_minimum = 95.0
max_cyclomatic_complexity = 10
mutation_score_minimum = 85.0
```

### 10.2 Required Tests

| Test Category | Count | Description |
|---------------|-------|-------------|
| Unit tests | 25+ | Core validation logic |
| Integration tests | 15+ | End-to-end validation scenarios |
| Property tests | 10+ | Randomized input validation |
| Regression tests | 5+ | Known bug reproductions |
| Ruchy scripts | 3+ | Complex regression scenarios (using `scripts/check_regression.ruchy`) |

### 10.3 Falsifiable Guarantees

| Guarantee | Test Method |
|-----------|-------------|
| Baseline generation creates valid manifest | Parse generated manifest with JSON schema |
| Timing regression detection works | Inject artificial delay, verify detection |
| Canary validation catches mismatches | Modify expected output, verify failure |
| Strict mode has zero tolerance | Single syscall deviation triggers failure |
| JUnit output is valid XML | Parse with XML validator |

---

## 11. Implementation Roadmap

### Phase 1: Core Validate Subcommand (Sprint 50) - COMPLETED

**Goal**: Implement basic `renacer validate` with golden trace comparison.

**Deliverables**:
- [x] `ValidateCommand` struct in `src/cli.rs` (added to `Commands` enum)
- [x] Core validation logic (`src/validate/mod.rs`)
- [x] Config with builder pattern (`src/validate/config.rs`)
- [x] Golden trace generation (`--generate`) (implemented in `src/validate/golden_trace.rs`)
- [x] Basic syscall comparison (`src/validate/comparison.rs`)
- [x] Timing comparison with tolerance (`src/validate/comparison.rs`)
- [x] Error codes V001-V010 per specification (`src/validate/error.rs`)
- [x] Text output format (`src/validate/output.rs`)
- [x] JSON output format (`src/validate/output.rs`)
- [x] JUnit output format (`src/validate/output.rs`)
- [x] 35 tests (32 pass, 3 ignored for ptrace) in `tests/sprint50_validate_tests.rs`

**Acceptance Criteria**:
```bash
# Must pass
renacer validate --generate golden/ -- echo "hello"
renacer validate --baseline golden/ -- echo "hello"

# Must fail (different command)
renacer validate --baseline golden/ -- echo "world"
```
- Core `validate` subcommand works as expected.
- 35 tests for core validation logic (32 passing, 3 ignored for ptrace requirements).

### Phase 2: APR Model Integration (Sprint 51)

**Goal**: Add APR model-aware validation.

**Deliverables**:
- [ ] `--apr-model` flag implementation
- [ ] Tensor statistics validation
- [ ] Canary output validation
- [ ] Layer trace comparison
- [ ] Integration with `aprender::format`
- [ ] 15+ integration tests

**Acceptance Criteria**:
```bash
# Must detect tensor stat violation
renacer validate --baseline golden/ --apr-model bad_model.apr -- apr run bad_model.apr
# Exit code 1, error V009
```

### Phase 3: CI/CD Integration (Sprint 52)

**Goal**: Production-ready CI/CD integration.

**Deliverables**:
- [ ] JSON output format
- [ ] JUnit XML output format
- [ ] GitHub Actions integration example
- [ ] Baseline update workflow
- [ ] Documentation and examples
- [ ] 10+ property tests

**Acceptance Criteria**:
```yaml
# .github/workflows/validate.yml
- name: Validate traces
  run: |
    renacer validate --baseline golden/ --output junit \
      -- cargo test 2>&1 | tee results.xml
- name: Upload results
  uses: actions/upload-artifact@v4
  with:
    name: trace-validation
    path: results.xml
```

### Phase 4: Advanced Features (Sprint 53)

**Goal**: Smart anomaly detection and hardware acceleration support.

**Deliverables**:
- [ ] Isolation Forest integration (`src/isolation_forest.rs`)
- [ ] GPU/HPU compute block tracing
- [ ] `ruchy` script integration
- [ ] Advanced visualization of anomalies

**Acceptance Criteria**:
```bash
# Must detect subtle anomaly
renacer validate --baseline golden/ --detect-anomalies -- ./unstable_binary
```

---

## 12. References

### 12.1 Internal References

- [APR-SPEC.md](https://github.com/paiml/aprender/docs/specifications/APR-SPEC.md) - APR format specification
- [trueno-tracing-integration-spec.md](./trueno-tracing-integration-spec.md) - Compute block tracing
- [golden-thread-open-telemetry-spec.md](./golden-thread-open-telemetry-spec.md) - OTLP integration
- [cli-tui-spec.md](./cli-tui-spec.md) - CLI/TUI Specification

### 12.2 External References

1. Sigelman, B. H., et al. (2010). "Dapper, a Large-Scale Distributed Systems Tracing Infrastructure." Google Technical Report.

2. Mace, J., Roelke, R., & Fonseca, R. (2015). "Pivot Tracing: Dynamic Causal Monitoring for Distributed Systems." ACM SOSP.

3. Sculley, D., et al. (2015). "Hidden technical debt in machine learning systems." Advances in Neural Information Processing Systems.

4. OpenTelemetry Specification v1.0: https://opentelemetry.io/docs/specs/

### 12.3 Related GitHub Issues

- [renacer#22](https://github.com/paiml/renacer/issues/22) - Add 'validate' subcommand for golden trace comparison
- [aprender#119](https://github.com/paiml/aprender/issues/119) - APR v2 format spec for web-scale models
- [aprender#116](https://github.com/paiml/aprender/issues/116) - Add JSON metadata section to APR format

---

## 13. APR 100-Point QA Checklist Integration

### 13.1 Overview

This section defines the **Renacer Verification Matrix** for the APR 100-Point Master Falsification QA Checklist. Each point below represents a falsifiable claim about an APR model or runtime execution that Renacer can automatically verify.

### 13.2 Renacer Verification Matrix

#### Category 1: File Format & Metadata (Points 1-25)

| # | QA Point Description | Falsification Test (Renacer Command) | Failure Condition |
|---|----------------------|--------------------------------------|-------------------|
| 01 | **Magic Header** - File starts with `APR\x01` | `renacer validate --apr-model <file> --check-magic` | Header bytes != `41 50 52 01` |
| 05 | **JSON Manifest** - Valid JSON structure | `renacer validate --apr-model <file> --check-manifest` | JSON parse error or schema violation |
| 08 | **Tensor Count** - Matches manifest count | `renacer validate --apr-model <file> --count-tensors` | Actual tensors found != manifest count |
| 12 | **Checksum** - SHA256 integrity check | `renacer validate --apr-model <file> --verify-checksum` | Computed hash != stored hash |

#### Category 2: Tensor Physics & Statistics (Points 26-50)

| # | QA Point Description | Falsification Test (Renacer Command) | Failure Condition |
|---|----------------------|--------------------------------------|-------------------|
| 26 | **NaN Check** - No NaNs in weights | `renacer validate --apr-model <file> --check-nans` | Any NaN value detected in tensor data |
| 27 | **Inf Check** - No Infinities in weights | `renacer validate --apr-model <file> --check-infs` | Any Inf value detected in tensor data |
| 36 | **Softmax Valid** - Output sums to 1.0 ±1e-6 | `renacer validate --trace-layers --check-softmax` | Sum of softmax output vectors outside [0.999999, 1.000001] |
| 39 | **Token IDs** - Output tokens < vocab size | `renacer validate --validate-canary --vocab-size N` | Any generated token ID >= N |
| 40 | **Audio Range** - Input in [-1.0, 1.0] | `renacer validate --apr-model <file> --check-input-range` | Audio input sample > 1.0 or < -1.0 |
| 43 | **Dead Neurons** - Activations never > 0 | `renacer validate --trace-layers --detect-dead-neurons` | Neuron activation sum across trace == 0.0 |
| 44 | **Exploding Grads** - Activations > 1e6 | `renacer validate --trace-layers --detect-exploding` | Any activation value > 1,000,000.0 |
| 45 | **Repeat Tokens** - Repetition > 5x | `renacer validate --validate-canary --detect-repetition` | 5+ identical consecutive tokens generated |

#### Category 3: Runtime Behavior & Performance (Points 51-75)

| # | QA Point Description | Falsification Test (Renacer Command) | Failure Condition |
|---|----------------------|--------------------------------------|-------------------|
| 51 | **Syscall Determinism** - Identical sequence | `renacer validate --baseline <dir> --strict` | Any deviation in syscall order or arguments |
| 55 | **Memory Leak** - RSS stable after warmup | `renacer validate --detect-anomalies --check-memory` | RSS growth > 10% between inference 10 and 100 |
| 64 | **Trace Output** - `apr trace` produces valid JSON | `renacer validate --output json --verify-output-schema` | Output stdout/stderr not valid JSON schema |
| 72 | **Canary Check** - Output matches reference | `renacer validate --validate-canary` | Generated output string != expected string |
| 74 | **Trace Payload** - Corrupt tensor shows anomaly | `renacer validate --detect-anomalies` | Injection of corrupt tensor fails to trigger anomaly score > 0.8 |
| 75 | **Trace Diff** - Diff identical models = 0 drift | `renacer validate --baseline <self> --tolerance 0` | Any reported drift > 0.0 for identical run |

#### Category 4: Integration & Security (Points 76-100)

| # | QA Point Description | Falsification Test (Renacer Command) | Failure Condition |
|---|----------------------|--------------------------------------|-------------------|
| 82 | **Path Traversal** - No loading outside root | `renacer validate --security-check` | `open()` syscall path contains `../` or `/etc/` |
| 88 | **Network Isolation** - No unexpected sockets | `renacer validate --security-check` | `socket()` or `connect()` syscall detected during offline inference |
| 95 | **OTLP Export** - Spans reach collector | `renacer validate --otlp-endpoint <url> --verify-export` | No spans received by mock collector within timeout |
| 100 | **Golden Trace** - Passes canonical trace | `renacer validate --baseline golden/` | Any regression failure (Exit code != 0) |

### 13.3 Integration with apr-qa

Renacer validation can feed into the automated `apr-qa` tool:

```bash
# Run apr-qa with renacer trace validation
apr-qa verify model.apr --score --trace-backend renacer

# Which internally calls:
renacer validate --baseline golden/ --apr-model model.apr --output json \
  -- apr run model.apr --input test.wav > trace_results.json

# apr-qa incorporates trace_results.json into final score
```

### 13.4 Checklist-Driven Test Generation

Renacer can auto-generate tests based on the 100-point checklist:

```bash
# Generate test cases for physics validation points
renacer generate-tests --checklist physics --apr-model model.apr --output tests/

# Creates:
#   tests/point_36_softmax_validation.rs
#   tests/point_43_dead_neuron_detection.rs
#   tests/point_44_exploding_activation.rs
#   ...
```

---

## 14. SDK Interface

### 14.1 Rust API

```rust
use renacer::{Validator, ValidationConfig, GoldenTrace, AprModelContext};

// Create validator with configuration
let config = ValidationConfig::builder()
    .tolerance_percent(10.0)
    .strict_mode(false)
    .canary_validation(true)
    .layer_tracing(true)
    .build();

let validator = Validator::new(config);

// Load golden baseline
let baseline = GoldenTrace::load("golden_traces/")?;

// Create APR model context for model-aware validation
let apr_context = AprModelContext::from_file("model.apr")?;

// Run validation
let result = validator
    .with_baseline(&baseline)
    .with_apr_model(&apr_context)
    .validate_command(&["apr", "run", "model.apr", "--input", "test.wav"])?;

// Check results
match result.status {
    ValidationStatus::Passed => println!("All checks passed"),
    ValidationStatus::Failed(regressions) => {
        for r in regressions {
            eprintln!("Regression: {:?}", r);
        }
    }
}

// Export results
result.export_json("results.json")?;
result.export_junit("results.xml")?;
```

### 14.2 Inline Validation During Tracing

Renacer supports inline validation callbacks during trace capture:

```rust
use renacer::{Tracer, InlineValidator, TensorExpectation};

let validator = InlineValidator::new()
    .expect_tensor("decoder.layer_norm.weight", TensorExpectation {
        mean_range: (0.5, 3.0),
        std_max: 1.0,
        nan_check: true,
        inf_check: true,
    })
    .on_violation(|tensor, violation| {
        // Called immediately when violation detected
        eprintln!("Tensor {} violated: {:?}", tensor, violation);
        // Return false to abort tracing
        false
    });

let tracer = Tracer::new()
    .with_inline_validator(validator)
    .trace_command(&["apr", "run", "model.apr"])?;
```

### 14.3 Async/Stream API

For long-running inference or CI/CD pipelines:

```rust
use renacer::{AsyncValidator, TraceStream};
use tokio::stream::StreamExt;

let validator = AsyncValidator::new(config);
let mut stream = validator.trace_stream(&["apr", "run", "model.apr"]);

while let Some(event) = stream.next().await {
    match event {
        TraceEvent::Syscall(s) => handle_syscall(s),
        TraceEvent::TensorStat(t) => {
            if t.violates_expectation() {
                stream.abort().await;
            }
        }
        TraceEvent::LayerOutput(l) => handle_layer(l),
        TraceEvent::Complete(result) => return result,
    }
}
```

### 14.4 Python Bindings (Future)

```python
import renacer

# Configure validation
config = renacer.ValidationConfig(
    tolerance_percent=10.0,
    canary_validation=True,
)

# Run validation
result = renacer.validate(
    baseline="golden_traces/",
    apr_model="model.apr",
    command=["apr", "run", "model.apr", "--input", "test.wav"],
    config=config,
)

# Check results
if result.passed:
    print("Validation passed")
else:
    for regression in result.regressions:
        print(f"Regression: {regression}")
```

---

## 15. Cross-Tool Validation Protocol

### 15.1 Protocol Overview

Defines how renacer interacts with APR tools (apr, apr-qa) for end-to-end validation:

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│    User     │───▶│   Renacer   │───▶│  apr run    │───▶│   Output    │
│   Command   │    │  (Tracer)   │    │  (Runtime)  │    │  (Traces)   │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                         │                                      │
                         ▼                                      ▼
                   ┌─────────────┐                        ┌─────────────┐
                   │   apr-qa    │◀───────────────────────│   Golden    │
                   │  (Scoring)  │                        │   Traces    │
                   └─────────────┘                        └─────────────┘
```

### 15.2 Message Format

Tools communicate via JSON messages on stderr (OTLP format compatible):

```json
{
  "renacer_protocol": "1.0",
  "message_type": "tensor_validation",
  "timestamp_ns": 1702732800000000000,
  "payload": {
    "tensor_name": "decoder.layer_norm.weight",
    "expected_mean_range": [0.5, 3.0],
    "actual_mean": 11.1,
    "status": "FAIL",
    "qa_point": 28
  }
}
```

### 15.3 Integration Points

| Tool | Renacer Role | Data Exchange |
|------|--------------|---------------|
| `apr run` | Traces syscalls & timing | Renacer wraps apr run |
| `apr trace` | Compares layer outputs | Consumes apr trace JSON |
| `apr validate` | Augments with syscall data | Provides trace context |
| `apr canary` | Runtime verification | Validates canary outputs |
| `apr-qa` | Provides trace-based scores | Exports JSON/JUnit results |

---

## Appendix A: Example Validation Session

```bash
# Step 1: Generate baseline from known-good run
$ renacer validate --generate golden/ -- apr run whisper.apr --input test.wav
Tracing: apr run whisper.apr --input test.wav
Syscalls captured: 1,234
Duration: 125.3ms
Baseline written to: golden/

# Step 2: Validate new code against baseline
$ renacer validate --baseline golden/ -- apr run whisper.apr --input test.wav

=== Renacer Validation Report ===

Baseline: golden/
Command:  apr run whisper.apr --input test.wav
Tolerance: 10%

Syscall Validation:
  ✓ Sequence match (1,234 syscalls)
  ✓ Argument match (99.8%)

Timing Validation:
  ✓ Total: 127.1ms (baseline: 125.3ms, +1.4%)
  ✓ P95: 2.1ms (baseline: 2.0ms, +5.0%)
  ✓ P99: 8.2ms (baseline: 8.0ms, +2.5%)

APR Model Validation:
  ✓ Tensor statistics within tolerance
  ✓ Canary output matches expected

RESULT: PASSED

# Step 3: After introducing a bug
$ renacer validate --baseline golden/ -- apr run buggy.apr --input test.wav

=== Renacer Validation Report ===

APR Model Validation:
  ✗ Tensor statistics violation
    decoder.layer_norm.weight: mean=11.1 (expected: [0.5, 3.0])
  ✗ Canary output mismatch
    Expected: "The quick brown fox..."
    Actual:   "....................."

RESULT: FAILED (2 regressions)
Exit code: 1
```

---

## Appendix B: Compatibility Matrix

| Renacer Version | APR Version | Features |
|-----------------|-------------|----------|
| 0.1.x | APR v1 | Basic syscall validation |
| 0.2.x | APR v1, v2 | + Tensor stats validation |
| 0.3.x | APR v2 | + Layer tracing, canary validation |
