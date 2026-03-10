# Profile Component Specification

**Parent**: [apr-spec.md](../apr-spec.md) §19
**Status**: Active
**CLI**: `apr profile`, `apr cbtop`, `apr ptx-map`, `apr ptx`
**Implementation**: `crates/apr-cli/src/commands/profile.rs`

---

## 1. Overview

Performance analysis and optimization tools. Roofline analysis, flamegraph
generation, naive implementation detection, energy measurement, and CI
assertion mode.

## 2. CLI Interface

```
apr profile <FILE> \
  [--granular] [--format <human|json|flamegraph>] \
  [--detect-naive] [--energy] [--perf-grade] \
  [--ci] [--assert-throughput <TPS>] [--assert-p99 <MS>]

apr cbtop [--model-path <FILE>] [--headless] [--ci]
```

## 3. Analysis Modes

| Mode | Description |
|------|-------------|
| Roofline | Memory vs compute bound analysis |
| Flamegraph | SVG flamegraph of inference hot paths |
| Naive detection | Identify unoptimized implementations |
| Energy | RAPL-based power measurement |
| Perf grade | Score vs Ollama baseline |
| CI assertions | Exit 1 if below throughput/latency thresholds |

## 4. Baseline Comparisons

```bash
apr profile model.apr --compare other-model.gguf
apr profile model.apr --ollama
apr profile model.apr --baseline-url http://localhost:8083 --baseline-name llama.cpp
```

## 5. ComputeBrick Pipeline Monitor (cbtop)

TUI dashboard for real-time inference profiling with brick scores.

---

## Provable Contracts

### Contract: `profile-v1.yaml`

```yaml
metadata:
  description: "Performance profiling — Roofline, flamegraph, CI assertions"
  depends_on:
    - "roofline-model-v1"

equations:
  roofline_bound:
    formula: "perf = min(peak_compute, peak_bandwidth * operational_intensity)"
    invariants:
      - "Memory-bound: perf limited by bandwidth"
      - "Compute-bound: perf limited by FLOPS"
      - "Operational intensity = FLOPS / bytes_accessed"

  throughput_measurement:
    formula: "tps = tokens / (end_time - start_time - warmup_time)"
    invariants:
      - "Warmup excluded from measurement"
      - "tps > 0 for successful inference"
      - "Multiple iterations averaged"

  naive_detection:
    formula: "naive = measured_gflops < threshold_gflops"
    invariants:
      - "Threshold configurable (default: 10 GFLOPS)"
      - "Detection is per-layer granularity"

  ci_assertion:
    formula: "pass = (tps >= assert_tps) AND (p99 <= assert_p99)"
    invariants:
      - "Exit code 0 on pass, 1 on fail"
      - "Assertions only checked when --ci flag set"

proof_obligations:
  - type: bound
    property: "Throughput positivity"
    formal: "tps > 0 for non-empty inference"
  - type: invariant
    property: "Warmup exclusion"
    formal: "warmup iterations not counted in measurement"
  - type: invariant
    property: "CI exit code contract"
    formal: "assertion failure → exit(1)"
  - type: invariant
    property: "Roofline classification"
    formal: "op_intensity < ridge_point → memory_bound, else compute_bound"

falsification_tests:
  - id: FALSIFY-PROF-001
    rule: "Warmup exclusion"
    prediction: "throughput excludes first N iterations"
    if_fails: "Warmup iterations counted in average"
  - id: FALSIFY-PROF-002
    rule: "CI exit code"
    prediction: "throughput below threshold → exit code 1"
    if_fails: "CI flag not checked or exit code not set"
  - id: FALSIFY-PROF-003
    rule: "Naive detection threshold"
    prediction: "identity kernel (memcpy) flagged as naive"
    if_fails: "Threshold comparison inverted"
  - id: FALSIFY-PROF-004
    rule: "Flamegraph output"
    prediction: "--format flamegraph produces valid SVG"
    if_fails: "SVG generation broken or empty"
  - id: FALSIFY-PROF-005
    rule: "Baseline comparison"
    prediction: "speedup ratio > 0 when baseline is valid"
    if_fails: "Division by zero when baseline is zero"

kani_harnesses:
  - id: KANI-PROF-001
    obligation: "Roofline classification"
    property: "correct bound classification for bounded OI"
    bound: 16
    strategy: stub_float
```

### Binding Requirements

```yaml
  - contract: profile-v1.yaml
    equation: roofline_bound
    module_path: "aprender::profile"
    function: classify_roofline
    status: implemented

  - contract: profile-v1.yaml
    equation: ci_assertion
    module_path: "aprender::profile"
    function: check_ci_assertions
    status: implemented
```
