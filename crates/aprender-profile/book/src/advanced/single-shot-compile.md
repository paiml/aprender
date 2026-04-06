# Single-Shot Compile Tooling

High-level performance and bug detection for transpilers and single-shot compile workflows.

## Overview

Renacer's Single-Shot Compile Tooling provides automated analysis for transpilers and compilers that run once per input file. This hybrid analysis combines **critical path tracing** (performance) with **semantic diff** (bug detection) to provide actionable, high-level insights.

## Key Features

### 🔥 Hotspot Identification
Automatically identifies performance bottlenecks using time-weighted attribution. Instead of showing raw syscall counts, Renacer highlights **where time is actually spent**.

### 🔍 Behavioral Change Detection
Detects unexpected syscall patterns using N-gram sequence mining. Catches subtle bugs like:
- Accidental async runtime initialization
- Telemetry library leaks
- Process control anomalies

### 📊 Statistical Regression Detection
Uses hypothesis testing (t-tests) to detect real performance regressions while filtering noise. No magic "5%" thresholds - adapts to your project's natural variance.

### ✅ Semantic Equivalence Validation
Validates that optimizations preserve observable behavior using state-based comparison.

## Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│ BASELINE TRACE                                              │
│   Golden execution (known-good)                              │
└─────────────────────────────────────────────────────────────┘
                           │
                           │ Compare
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ CURRENT TRACE                                               │
│   New version / optimization                                 │
└─────────────────────────────────────────────────────────────┘
                           │
                           │ Analyze
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ HYBRID ANALYSIS                                             │
│  1. Syscall Clustering (TOML-based, Open-Closed)           │
│  2. Sequence Mining (N-gram grammar detection)              │
│  3. Time-Weighted Attribution (wall-clock hotspots)         │
│  4. Semantic Equivalence (state-based comparison)           │
│  5. Regression Detection (statistical hypothesis testing)   │
└─────────────────────────────────────────────────────────────┘
                           │
                           │ Report
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ ACTIONABLE OUTPUT                                           │
│  • Performance hotspots with explanations                   │
│  • Behavioral anomalies with context                        │
│  • Regression detection with confidence levels              │
│  • Optimization validation (semantic equivalence)           │
└─────────────────────────────────────────────────────────────┘
```

## Target Codebases

This tooling is designed for:

- **Transpilers** (Python→Rust, C→Rust, etc.)
- **Single-shot compilers** (one input file → one output)
- **Build tools** (incremental compilation disabled)
- **Code generators** (template expansion, macros)

## Toyota Way Integration

The implementation follows Toyota Production System principles:

- **Andon** (Stop the Line): Build-time assertions fail CI on regressions
- **Kaizen** (Continuous Improvement): Statistical tracking enables incremental optimization
- **Genchi Genbutsu** (Go and See): Real syscall traces, not synthetic benchmarks
- **Jidoka** (Automation with Human Touch): Automated analysis with actionable explanations
- **Poka-Yoke** (Error-proofing): Statistical tests prevent false positives

## Quick Start

### 1. Collect Baseline Golden Trace

```bash
# Known-good version
renacer trace ./transpiler input.py --output baseline.trace
```

### 2. Collect Current Trace

```bash
# New version to test
renacer trace ./transpiler input.py --output current.trace
```

### 3. Run Hybrid Analysis

```bash
renacer analyze --baseline baseline.trace --current current.trace
```

### Example Output

```text
# Single-Shot Compile Analysis Report

## 1. Performance Summary
- Total Time: 156ms (baseline: 123ms, +26.8%)
- Hotspots: 2 identified
- Anomalies: 1 detected

## 2. Hotspot Analysis (Critical Path Tracer)

### 🔥 Hotspot 1: Memory Allocation (81.2ms, +81% vs baseline)
- mmap: 42 calls (+35 calls, +500%)
- brk: 18 calls (+12 calls, +200%)
- munmap: 35 calls (+28 calls, +400%)

Explanation: Memory allocation dominates execution. This is UNEXPECTED for
transpilers (typical: <20%). Investigation needed.

Recommendation: Profile allocator with --flamegraph

### 🔥 Hotspot 2: Process Control (24.3ms, NEW)
- fork: 24 calls (NEW)
- execve: 24 calls (NEW)
- waitpid: 24 calls (NEW)

Explanation: Process control syscalls detected. This is UNEXPECTED for
transpilers. Possible causes:
- Accidental subprocess spawning
- Shell command execution
- Build system integration

## 3. Behavioral Changes (Semantic Diff)

### Memory Allocation Cluster: +35 calls (+81% time)
Grammar violation: NEW pattern detected
- Baseline: open → read → write → close
- Current:   open → read → **mmap × 35** → write → close

### Process Control Cluster: +24 calls (NEW)
Grammar violation: Process control not expected
- Pattern: fork → execve → waitpid (repeated 24×)

## 4. Verdict
⚠️ REGRESSION DETECTED

Statistical significance: p < 0.001 (99.9% confidence)
- Memory allocation: statistically significant increase
- Process control: new unexpected behavior

## 5. Recommendations
1. Investigate memory allocation spike (81% of runtime)
2. Remove accidental subprocess spawning (24 processes)
3. Run with --flamegraph for allocation profiling
4. Consider memory pooling / arena allocation
```

## Components

- **[Syscall Clustering](./syscall-clustering.md)** - TOML-based configuration for semantic grouping
- **[Sequence Mining](./sequence-mining.md)** - N-gram grammar detection for anomalies
- **[Time-Weighted Attribution](./time-attribution.md)** - Wall-clock hotspot identification
- **[Semantic Equivalence](./semantic-equivalence.md)** - State-based optimization validation
- **[Regression Detection](./regression-detection.md)** - Statistical hypothesis testing

## Peer-Reviewed Foundation

This implementation is based on 19 peer-reviewed papers:

- **Zeller (2002)**: Delta Debugging for noise filtering
- **Heger et al. (2013)**: Statistical regression detection (ICPE)
- **Forrest et al. (1996)**: N-gram anomaly detection (IEEE S&P)
- **Mestel et al. (2022)**: Google-scale profiling (Usenix ATC)
- And 15 more...

See [Single-Shot Compile Tooling Specification](../../docs/specifications/single-shot-compile-tooling-spec.md) for complete citations.

## Implementation Statistics

- **Total Lines**: ~2,400 lines of production code
- **Test Coverage**: 471 passing tests (100% success rate)
- **Zero Defects**: All tests passing, no clippy warnings
- **Dependencies**: Uses aprender/trueno for statistics (no custom implementations)

## Next Steps

1. Learn about [Syscall Clustering](./syscall-clustering.md) configuration
2. Understand [Sequence Mining](./sequence-mining.md) for anomaly detection
3. Use [Time-Weighted Attribution](./time-attribution.md) for performance analysis
4. Validate optimizations with [Semantic Equivalence](./semantic-equivalence.md)
5. Detect regressions with [Statistical Testing](./regression-detection.md)
