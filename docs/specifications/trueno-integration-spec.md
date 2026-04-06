# Trueno Integration Specification for Renacer

**Version:** 1.0
**Date:** 2025-11-17
**Status:** Specification Draft
**Sprint Target:** 19-20 (Enhanced Statistical Analysis)

## Executive Summary

This specification defines enhanced integration of **Trueno** (SIMD/GPU compute library) into **Renacer** (syscall tracer) for high-performance statistical analysis, anomaly detection, and performance modeling. Current integration (v0.3.0) uses only `Vector::sum()` for basic aggregations. This spec expands usage to 20+ Trueno operations across 5 modules.

**Business Value:**
- **Performance**: 3-10x faster statistical computations via SIMD
- **Capabilities**: Advanced analytics (percentiles, anomaly detection, correlation)
- **Sister Project Synergy**: Dogfooding Trueno within PAIML ecosystem
- **Differentiation**: Statistical analysis features unique among tracers

---

## Table of Contents

1. [Current Integration (v0.3.0)](#1-current-integration-v030)
2. [Trueno Capabilities Overview](#2-trueno-capabilities-overview)
3. [Integration Opportunities](#3-integration-opportunities)
4. [Phase 1: Enhanced Statistics (Sprint 19)](#4-phase-1-enhanced-statistics-sprint-19)
5. [Phase 2: Anomaly Detection (Sprint 20)](#5-phase-2-anomaly-detection-sprint-20)
6. [Phase 3: Performance Modeling (Future)](#6-phase-3-performance-modeling-future)
7. [Implementation Guidelines](#7-implementation-guidelines)
8. [Testing Strategy](#8-testing-strategy)
9. [Performance Benchmarks](#9-performance-benchmarks)
10. [Migration Path](#10-migration-path)

---

## 1. Current Integration (v0.3.0)

### 1.1 Existing Usage

**Location:** `src/stats.rs` (lines 55-83)

**Current Implementation:**
```rust
pub fn calculate_totals_with_trueno(&self) -> StatTotals {
    // Extract data into vectors for SIMD processing
    let counts: Vec<f32> = self.stats.values().map(|s| s.count as f32).collect();
    let errors: Vec<f32> = self.stats.values().map(|s| s.errors as f32).collect();
    let times: Vec<f32> = self.stats.values().map(|s| s.total_time_us as f32).collect();

    // Use Trueno for SIMD-accelerated sums
    let total_calls = trueno::Vector::from_slice(&counts).sum().unwrap_or(0.0) as u64;
    let total_errors = trueno::Vector::from_slice(&errors).sum().unwrap_or(0.0) as u64;
    let total_time_us = trueno::Vector::from_slice(&times).sum().unwrap_or(0.0) as u64;

    StatTotals { total_calls, total_errors, total_time_us }
}
```

**Operations Used:**
- `Vector::from_slice()` - Data ingestion
- `Vector::sum()` - SIMD-accelerated summation

**Performance:** ~3.15x faster than scalar sum for reductions

### 1.2 Limitations

1. **Underutilization**: Only 2/40+ Trueno operations used
2. **No Statistical Analysis**: Mean, variance, percentiles not computed
3. **No Anomaly Detection**: No outlier identification
4. **No Correlation**: No relationship analysis between metrics
5. **Basic Output**: No advanced visualizations or insights

---

## 2. Trueno Capabilities Overview

### 2.1 Vector Operations (40+ methods)

| Category | Operations | Use Case in Renacer |
|----------|------------|---------------------|
| **Arithmetic** | `add`, `sub`, `mul`, `div` | Metric normalization |
| **Reductions** | `sum`, `max`, `min`, `argmax`, `argmin` | ✅ Currently used (sum only) |
| **Statistics** | `mean`, `variance`, `stddev`, `covariance`, `correlation` | **Phase 1 target** |
| **Advanced Stats** | `sum_kahan`, `sum_of_squares`, `zscore` | Numerical stability, outlier detection |
| **Normalization** | `minmax_normalize`, `normalize`, `clip` | Data scaling |
| **Activations** | `relu`, `sigmoid`, `softmax`, `gelu` | ML-based modeling (Phase 3) |
| **Norms** | `norm_l1`, `norm_l2`, `norm_linf`, `abs` | Distance metrics |

### 2.2 Matrix Operations (15+ methods)

| Category | Operations | Use Case in Renacer |
|----------|------------|---------------------|
| **Matrix Math** | `matmul`, `transpose`, `identity` | Correlation matrices (Phase 2) |
| **Convolution** | `convolve2d` | Time-series pattern detection |
| **2D Stats** | Per-row/column analysis | Multi-process metrics |

### 2.3 Backend Selection

| Backend | Performance | Availability |
|---------|-------------|--------------|
| **AVX2** | ~3.4x faster (dot), ~1.8x faster (sum) | x86_64 with AVX2+FMA |
| **SSE2** | ~3.15x faster (sum), ~3.4x faster (dot) | x86_64 baseline |
| **NEON** | Similar to SSE2 | ARM/aarch64 |
| **GPU** | 10-50x faster (>10K elements) | Via wgpu (future) |
| **Scalar** | Fallback | All platforms |

**Renacer Benefit:** Auto-selects best backend at runtime via `trueno::select_best_available_backend()`

---

## 3. Integration Opportunities

### 3.1 Module-by-Module Analysis

#### **3.1.1 Statistics Module (`src/stats.rs`)**

**Current:** Basic aggregations (sum)
**Opportunity:** Advanced statistical analysis

**New Operations:**
- `mean()` - Average syscall duration per syscall type
- `variance()` / `stddev()` - Variability in syscall timing
- `max()` / `min()` - Identify slowest/fastest syscalls
- `argmax()` / `argmin()` - Index of slowest syscall
- `sum_kahan()` - Numerically stable summation for large traces
- `clip()` - Outlier rejection (e.g., clip at 99th percentile)

**New Features:**
- Percentile calculations (p50, p75, p90, p95, p99)
- Coefficient of variation (CV = stddev/mean)
- Outlier detection (values >3σ from mean)
- Time distribution histograms

#### **3.1.2 Function Profiler (`src/function_profiler.rs`)**

**Current:** Per-function call counts and total time
**Opportunity:** Statistical timing analysis per function

**New Operations:**
- `mean()` - Average function call duration
- `stddev()` - Timing variability (jitter detection)
- `covariance()` / `correlation()` - Function call pattern analysis
- `zscore()` - Identify anomalous function calls (>3σ)
- `clip()` - Remove timing outliers from statistics

**New Features:**
- Function call timing distribution (mean ± stddev)
- Jitter detection (high stddev → unstable performance)
- Cross-function correlation (A calls B → timing impact)
- Anomaly alerts ("Function X took 10x longer than normal")

#### **3.1.3 Self-Profiling (`src/profiling.rs`)**

**Current:** Category-based timing aggregation
**Opportunity:** Overhead characterization

**New Operations:**
- `mean()` - Average overhead per category
- `stddev()` - Overhead variability
- `minmax_normalize()` - Scale overhead percentages
- `correlation()` - Identify correlated overhead sources

**New Features:**
- Overhead distribution analysis
- Identify overhead outliers (spikes in ptrace time)
- Correlation between categories (DwarfLookup ↔ MemoryRead)

#### **3.1.4 Multi-Process Tracing (`src/tracer.rs`)**

**Current:** Per-process state tracking
**Opportunity:** Cross-process statistical analysis

**New Operations:**
- `Matrix::from_vec()` - 2D process×syscall matrix
- `Matrix::transpose()` - Pivot analysis (syscall×process view)
- `Vector::correlation()` - Inter-process timing correlation
- `Vector::mean()` - Average per-process metrics

**New Features:**
- Process similarity analysis (which processes behave alike?)
- Cross-process syscall correlation
- Multi-process performance matrix (heatmap-ready)

#### **3.1.5 New Module: Anomaly Detection**

**Location:** `src/anomaly.rs` (new file)
**Purpose:** Real-time anomaly detection during tracing

**Trueno Operations:**
- `mean()` / `stddev()` - Baseline statistics
- `zscore()` - Anomaly scoring (z-score > 3 → outlier)
- `clip()` - Outlier rejection for statistics
- `abs()` - Absolute deviations

**Features:**
- Real-time anomaly alerts during tracing
- Configurable thresholds (default: 3σ)
- Anomaly report in summary output
- JSON export of detected anomalies

---

## 4. Phase 1: Enhanced Statistics (Sprint 19)

### 4.1 Goals

**EXTREME TDD Cycle:** RED → GREEN → REFACTOR
**Deliverable:** Enhanced `-c` statistics mode with 10+ new metrics

**Success Criteria:**
- ✅ 20+ new tests (integration + unit)
- ✅ All functions ≤10 cyclomatic complexity
- ✅ Zero clippy warnings
- ✅ 3-5x performance improvement over scalar implementations
- ✅ Backward compatible (existing `-c` output unchanged by default)

### 4.2 Implementation Plan

#### **4.2.1 Enhanced StatsTracker**

**File:** `src/stats.rs`

**New Methods:**
```rust
impl StatsTracker {
    /// Calculate statistical summary using Trueno
    pub fn calculate_statistics_with_trueno(&self) -> StatisticalSummary {
        // Convert syscall durations to vectors
        let durations: Vec<f32> = self.stats.values()
            .flat_map(|s| vec![s.total_time_us as f32 / s.count as f32; s.count as usize])
            .collect();

        let v = Vector::from_slice(&durations);

        StatisticalSummary {
            mean: v.mean().unwrap_or(0.0),
            stddev: v.stddev().unwrap_or(0.0),
            min: v.min().unwrap_or(0.0),
            max: v.max().unwrap_or(0.0),
            median: calculate_percentile(&v, 50.0),
            p95: calculate_percentile(&v, 95.0),
            p99: calculate_percentile(&v, 99.0),
        }
    }

    /// Detect anomalous syscalls (>3σ from mean)
    pub fn detect_anomalies(&self) -> Vec<AnomalyReport> {
        let durations: Vec<f32> = /* ... */;
        let v = Vector::from_slice(&durations);
        let z_scores = v.zscore().unwrap();

        // Identify z-scores > 3.0 (outliers)
        z_scores.as_slice().iter().enumerate()
            .filter(|(_, &z)| z.abs() > 3.0)
            .map(|(idx, z)| AnomalyReport { /* ... */ })
            .collect()
    }
}

#[derive(Debug)]
pub struct StatisticalSummary {
    pub mean: f32,
    pub stddev: f32,
    pub min: f32,
    pub max: f32,
    pub median: f32,
    pub p95: f32,
    pub p99: f32,
}

#[derive(Debug)]
pub struct AnomalyReport {
    pub syscall_name: String,
    pub duration_us: u64,
    pub z_score: f32,
    pub severity: AnomalySeverity,  // Low, Medium, High, Critical
}
```

#### **4.2.2 New CLI Flags**

**File:** `src/cli.rs`

```rust
#[arg(long = "stats-extended", requires = "count")]
pub stats_extended: bool,

#[arg(long = "detect-anomalies")]
pub detect_anomalies: bool,

#[arg(long = "anomaly-threshold", default_value = "3.0")]
pub anomaly_threshold: f32,
```

#### **4.2.3 Enhanced Output**

**Standard `-c` output (unchanged):**
```
% time     seconds  usecs/call     calls    errors syscall
------ ----------- ----------- --------- --------- ----------------
 45.20    0.001200       17.91        67         0 open
 32.10    0.000850       18.89        45         0 read
------ ----------- ----------- --------- --------- ----------------
100.00    0.002650       18.31       145         0 total
```

**Extended `-c --stats-extended` output:**
```
╔════════════════════════════════════════════════════════════════════════════╗
║  Extended Statistical Summary (SIMD-Accelerated via Trueno)               ║
╚════════════════════════════════════════════════════════════════════════════╝

Syscall Duration Statistics:
  Mean:              18.31 µs
  Std Deviation:      5.42 µs (CV: 29.6%)
  Min:                2.10 µs
  Max:              125.40 µs
  Median (P50):      16.50 µs
  P95:               42.30 µs
  P99:               89.10 µs

Anomalies Detected (>3σ):
  ⚠️  1 anomaly detected
  - read() at 125.40 µs (z-score: 19.75, severity: Critical)

Performance Insights:
  - High variability detected (CV > 25%)
  - Consider investigating read() outlier
  - 99% of syscalls complete within 89.10 µs
```

### 4.3 Testing Strategy (Phase 1)

**Integration Tests:** `tests/sprint19_enhanced_stats_tests.rs`

```rust
#[test]
fn test_stats_extended_flag() {
    // Test --stats-extended output includes mean, stddev, percentiles
}

#[test]
fn test_anomaly_detection_basic() {
    // Test --detect-anomalies identifies outliers
}

#[test]
fn test_anomaly_threshold_custom() {
    // Test custom --anomaly-threshold value
}

#[test]
fn test_stats_extended_with_json() {
    // Test JSON output includes statistical summary
}

#[test]
fn test_stats_extended_with_csv() {
    // Test CSV output includes percentile columns
}
```

**Unit Tests:** `src/stats.rs`

```rust
#[test]
fn test_calculate_statistics_with_trueno() {
    // Test statistical summary calculation
}

#[test]
fn test_detect_anomalies_none() {
    // Test no anomalies when all values within 3σ
}

#[test]
fn test_detect_anomalies_found() {
    // Test anomaly detection with outlier
}

#[test]
fn test_percentile_calculation() {
    // Test percentile calculation accuracy
}
```

**Property-Based Tests:** `tests/property_based_comprehensive.rs`

```rust
proptest! {
    #[test]
    fn prop_statistical_summary_never_panics(durations in prop::collection::vec(0u64..1_000_000, 10..1000)) {
        // Test statistical calculations don't panic on any input
    }

    #[test]
    fn prop_mean_bounded_by_min_max(durations in prop::collection::vec(1u64..1000, 10..100)) {
        // Test mean is always between min and max
    }
}
```

---

## 5. Phase 2: Anomaly Detection (Sprint 20)

### 5.1 Goals

**Deliverable:** Real-time anomaly detection during syscall tracing

**Features:**
- Sliding window anomaly detection (last N syscalls)
- Per-syscall type baseline learning
- Real-time alerts during tracing (optional stderr output)
- Anomaly report in summary
- JSON export of anomalies

### 5.2 Implementation Plan

#### **5.2.1 New Module: Anomaly Detector**

**File:** `src/anomaly.rs` (new file)

```rust
use trueno::Vector;
use std::collections::HashMap;

/// Real-time anomaly detector using sliding window statistics
pub struct AnomalyDetector {
    /// Per-syscall baseline statistics
    baselines: HashMap<String, BaselineStats>,
    /// Sliding window size (default: 100 samples per syscall)
    window_size: usize,
    /// Z-score threshold for anomaly (default: 3.0)
    threshold: f32,
}

#[derive(Debug)]
struct BaselineStats {
    /// Recent samples (sliding window)
    samples: Vec<f32>,
    /// Pre-computed mean (updated on each sample)
    mean: f32,
    /// Pre-computed stddev (updated on each sample)
    stddev: f32,
}

impl AnomalyDetector {
    pub fn new(window_size: usize, threshold: f32) -> Self {
        Self {
            baselines: HashMap::new(),
            window_size,
            threshold,
        }
    }

    /// Record a syscall and check for anomaly
    pub fn record_and_check(&mut self, syscall_name: &str, duration_us: u64) -> Option<Anomaly> {
        let baseline = self.baselines.entry(syscall_name.to_string()).or_insert_with(|| {
            BaselineStats {
                samples: Vec::with_capacity(self.window_size),
                mean: 0.0,
                stddev: 0.0,
            }
        });

        // Add sample to sliding window
        baseline.samples.push(duration_us as f32);
        if baseline.samples.len() > self.window_size {
            baseline.samples.remove(0);  // Remove oldest sample
        }

        // Need at least 10 samples for reliable statistics
        if baseline.samples.len() < 10 {
            return None;
        }

        // Update baseline statistics using Trueno
        let v = Vector::from_slice(&baseline.samples);
        baseline.mean = v.mean().unwrap_or(0.0);
        baseline.stddev = v.stddev().unwrap_or(0.0);

        // Calculate z-score for current sample
        let z_score = if baseline.stddev > 0.0 {
            (duration_us as f32 - baseline.mean) / baseline.stddev
        } else {
            0.0
        };

        // Check if anomaly
        if z_score.abs() > self.threshold {
            Some(Anomaly {
                syscall_name: syscall_name.to_string(),
                duration_us,
                z_score,
                baseline_mean: baseline.mean,
                baseline_stddev: baseline.stddev,
                severity: classify_severity(z_score),
            })
        } else {
            None
        }
    }

    /// Get all current baseline statistics
    pub fn get_baselines(&self) -> &HashMap<String, BaselineStats> {
        &self.baselines
    }
}

#[derive(Debug, Clone)]
pub struct Anomaly {
    pub syscall_name: String,
    pub duration_us: u64,
    pub z_score: f32,
    pub baseline_mean: f32,
    pub baseline_stddev: f32,
    pub severity: AnomalySeverity,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnomalySeverity {
    Low,       // 3σ < z < 5σ
    Medium,    // 5σ < z < 10σ
    High,      // 10σ < z < 20σ
    Critical,  // z > 20σ
}

fn classify_severity(z_score: f32) -> AnomalySeverity {
    let abs_z = z_score.abs();
    if abs_z < 5.0 {
        AnomalySeverity::Low
    } else if abs_z < 10.0 {
        AnomalySeverity::Medium
    } else if abs_z < 20.0 {
        AnomalySeverity::High
    } else {
        AnomalySeverity::Critical
    }
}
```

#### **5.2.2 Integration with Tracer**

**File:** `src/tracer.rs`

```rust
// Add to TracerConfig
pub struct TracerConfig {
    // ... existing fields ...
    pub enable_anomaly_detection: bool,
    pub anomaly_threshold: f32,
    pub anomaly_window_size: usize,
}

// In trace_child(), add anomaly detector
fn trace_child(child: Pid, config: TracerConfig) -> Result<i32> {
    let mut anomaly_detector = if config.enable_anomaly_detection {
        Some(AnomalyDetector::new(config.anomaly_window_size, config.anomaly_threshold))
    } else {
        None
    };

    // In syscall processing loop:
    if let Some(detector) = &mut anomaly_detector {
        if let Some(anomaly) = detector.record_and_check(syscall_name, duration_us) {
            eprintln!("⚠️  ANOMALY: {} took {} µs (z={:.2}, expected {}±{} µs)",
                anomaly.syscall_name, anomaly.duration_us, anomaly.z_score,
                anomaly.baseline_mean, anomaly.baseline_stddev);
        }
    }
}
```

#### **5.2.3 CLI Flags**

```rust
#[arg(long = "detect-anomalies")]
pub detect_anomalies: bool,

#[arg(long = "anomaly-threshold", default_value = "3.0")]
pub anomaly_threshold: f32,

#[arg(long = "anomaly-window", default_value = "100")]
pub anomaly_window_size: usize,

#[arg(long = "anomaly-quiet")]
pub anomaly_quiet: bool,  // Suppress real-time alerts, only show summary
```

### 5.3 Example Output

**Real-time alerts (stderr):**
```bash
$ renacer --detect-anomalies -- ./slow-app
openat(AT_FDCWD, "/etc/ld.so.cache", O_RDONLY|O_CLOEXEC) = 3
read(3, buf, 1024) = 512
⚠️  ANOMALY: read took 125400 µs (z=19.75, expected 18±5 µs)
write(1, "result", 6) = 6
...
```

**Summary output:**
```
╔════════════════════════════════════════════════════════════════════════════╗
║  Anomaly Detection Report                                                  ║
╚════════════════════════════════════════════════════════════════════════════╝

Configuration:
  - Window size: 100 samples per syscall
  - Threshold: 3.0σ
  - Total anomalies detected: 3

Detected Anomalies:
  1. read() - 125.40 ms (z-score: 19.75) ⚠️ CRITICAL
     Expected: 18.31 ± 5.42 µs
     Timestamp: [syscall #145]

  2. openat() - 89.20 ms (z-score: 8.42) ⚠️ HIGH
     Expected: 12.10 ± 3.20 µs
     Timestamp: [syscall #89]

  3. write() - 45.30 ms (z-score: 4.12) ⚠️ MEDIUM
     Expected: 8.50 ± 2.10 µs
     Timestamp: [syscall #201]

Recommendations:
  - Investigate read() outlier (20x slower than baseline)
  - Check for I/O contention or blocking operations
```

---

## 6. Phase 3: Performance Modeling (Future)

### 6.1 Goals

**Deliverable:** Predictive performance modeling using ML techniques

**Features:**
- Syscall pattern classification (I/O-bound vs CPU-bound vs mixed)
- Performance prediction based on historical data
- Regression models for expected vs actual performance
- Correlation analysis (which syscalls predict slow behavior?)

### 6.2 Trueno Operations (ML-focused)

| Operation | Use Case |
|-----------|----------|
| `correlation()` | Feature correlation (syscall A → syscall B timing) |
| `covariance()` | Covariance matrix for multi-syscall patterns |
| `Matrix::matmul()` | Linear regression (Xᵀ X)⁻¹ Xᵀ y |
| `sigmoid()` | Logistic classification (slow vs fast) |
| `softmax()` | Multi-class prediction (I/O/CPU/Mixed) |
| `relu()` | Activation function for neural models |
| `convolve2d()` | Time-series pattern detection |

### 6.3 Example: I/O Pattern Classifier

```rust
use trueno::{Vector, Matrix};

pub struct PerformanceClassifier {
    /// Weights learned from training data
    weights: Matrix<f32>,
}

impl PerformanceClassifier {
    /// Predict performance class (I/O-bound, CPU-bound, Mixed)
    pub fn predict(&self, syscall_sequence: &[SyscallEvent]) -> PerformanceClass {
        // Extract features: [read_count, write_count, cpu_intensive_count, ...]
        let features = self.extract_features(syscall_sequence);
        let features_vec = Vector::from_slice(&features);

        // Linear model: logits = weights × features
        let logits = self.weights.matvec(&features_vec).unwrap();

        // Softmax for probabilities
        let probs = logits.softmax().unwrap();

        // Argmax for final prediction
        let class_idx = probs.argmax().unwrap();

        match class_idx {
            0 => PerformanceClass::IOBound,
            1 => PerformanceClass::CPUBound,
            2 => PerformanceClass::Mixed,
            _ => unreachable!(),
        }
    }
}

#[derive(Debug)]
pub enum PerformanceClass {
    IOBound,
    CPUBound,
    Mixed,
}
```

---

## 7. Implementation Guidelines

### 7.1 EXTREME TDD Methodology

**All phases MUST follow RED → GREEN → REFACTOR:**

1. **RED Phase**: Write integration tests that fail
2. **GREEN Phase**: Implement minimal code to pass tests
3. **REFACTOR Phase**: Optimize, extract complexity, add unit tests

**Quality Gates:**
- ✅ All tests pass
- ✅ Cyclomatic complexity ≤10 (all functions)
- ✅ Zero clippy warnings (`-D warnings`)
- ✅ Test coverage ≥90%
- ✅ Mutation score ≥80%
- ✅ PMAT TDG score ≥90

### 7.2 Toyota Way Principles

**Andon Cord:** Stop development if:
- Complexity violation (any function >10)
- Test coverage drops below 90%
- Clippy warnings appear
- Performance regression >10%

**Jidoka:** Built-in quality via:
- Pre-commit hooks (format, clippy, tests, audit)
- Property-based tests (proptest)
- Mutation testing (cargo-mutants)

**Kaizen:** Continuous improvement via:
- Benchmark-driven optimization (≥10% speedup required)
- Iterative refactoring (each sprint)

### 7.3 Error Handling

**All Trueno operations return `Result<T, TruenoError>`:**

```rust
use trueno::{Vector, TruenoError};

// Pattern 1: Fallback to scalar on error
let result = vector.mean().unwrap_or_else(|_| {
    // Fallback to scalar implementation
    vector.as_slice().iter().sum::<f32>() / vector.len() as f32
});

// Pattern 2: Propagate errors
pub fn calculate_stats(&self) -> Result<Stats, TruenoError> {
    let v = Vector::from_slice(&self.data);
    let mean = v.mean()?;  // Propagate TruenoError
    let stddev = v.stddev()?;
    Ok(Stats { mean, stddev })
}

// Pattern 3: Convert to anyhow::Error (Renacer's error type)
use anyhow::Context;

let mean = v.mean()
    .context("Failed to calculate mean with Trueno")?;
```

### 7.4 Performance Guidelines

**Trueno Thresholds:**
- **Sum/Reductions**: Beneficial at 100+ elements (3x speedup)
- **Dot Product**: Beneficial at 100+ elements (3.4x speedup)
- **Matrix Operations**: Beneficial at 64×64+ (SIMD threshold)
- **GPU Dispatch**: Beneficial at 10K+ elements (10-50x speedup)

**Best Practices:**
1. **Batch Operations**: Group syscalls for bulk processing
2. **Avoid Frequent Conversions**: Minimize Vec ↔ Vector conversions
3. **Reuse Vectors**: Pre-allocate Vector objects in hot loops
4. **Profile First**: Measure before optimizing (use `--profile-self`)

**Example: Efficient Batching**
```rust
// ❌ Bad: Convert per-syscall
for syscall in syscalls {
    let v = Vector::from_slice(&[syscall.duration]);
    sum += v.sum().unwrap();  // Overhead > benefit
}

// ✅ Good: Batch convert
let durations: Vec<f32> = syscalls.iter().map(|s| s.duration as f32).collect();
let v = Vector::from_slice(&durations);
let sum = v.sum().unwrap();  // SIMD-accelerated
```

---

## 8. Testing Strategy

### 8.1 Test Pyramid

```
        ┌─────────────┐
        │  Property   │  18 tests (proptest)
        │   Tests     │
        ├─────────────┤
        │ Integration │  30 tests (assert_cmd)
        │   Tests     │
        ├─────────────┤
        │    Unit     │  100+ tests (standard)
        │   Tests     │
        └─────────────┘
```

### 8.2 Integration Test Matrix

| Feature | `-c` | `-c --stats-extended` | `--detect-anomalies` | `--format json` | `--format csv` |
|---------|------|-----------------------|----------------------|-----------------|----------------|
| Basic stats | ✅ | ✅ | ✅ | ✅ | ✅ |
| Extended stats | ❌ | ✅ | ✅ | ✅ | ✅ |
| Anomalies | ❌ | ❌ | ✅ | ✅ | ✅ |
| Percentiles | ❌ | ✅ | ✅ | ✅ | ✅ |

**Total Integration Tests:** 20+ per phase

### 8.3 Unit Test Coverage

**Target: 100% coverage on all new code**

**Per Module:**
- `src/stats.rs`: 30+ tests (existing 17 + 13 new)
- `src/anomaly.rs`: 25+ tests (new module)
- `src/function_profiler.rs`: 15+ tests (8 existing + 7 new)
- `src/profiling.rs`: 10+ tests (existing + 3 new)

**Total New Unit Tests:** 58+

### 8.4 Property-Based Tests

**File:** `tests/property_based_comprehensive.rs`

```rust
proptest! {
    // Statistics never panic
    #[test]
    fn prop_statistics_never_panic(durations in prop::collection::vec(0u64..1_000_000, 10..1000)) {
        let v = Vector::from_slice(&durations.iter().map(|&d| d as f32).collect::<Vec<_>>());
        let _ = v.mean();
        let _ = v.stddev();
        let _ = v.max();
    }

    // Mean is always between min and max
    #[test]
    fn prop_mean_bounded(durations in prop::collection::vec(1u64..10000, 10..100)) {
        let v = Vector::from_slice(&durations.iter().map(|&d| d as f32).collect::<Vec<_>>());
        let mean = v.mean().unwrap();
        let min = v.min().unwrap();
        let max = v.max().unwrap();
        assert!(mean >= min && mean <= max);
    }

    // Z-scores have zero mean (approximately)
    #[test]
    fn prop_zscore_zero_mean(values in prop::collection::vec(-1000.0f32..1000.0, 100..200)) {
        let v = Vector::from_slice(&values);
        let z = v.zscore().unwrap();
        let z_mean = z.mean().unwrap();
        assert!((z_mean).abs() < 0.01);  // Should be close to 0
    }

    // Anomaly detection is deterministic
    #[test]
    fn prop_anomaly_detection_deterministic(durations in prop::collection::vec(1u64..1000, 50..100)) {
        let mut detector1 = AnomalyDetector::new(100, 3.0);
        let mut detector2 = AnomalyDetector::new(100, 3.0);

        for &d in &durations {
            let a1 = detector1.record_and_check("test", d);
            let a2 = detector2.record_and_check("test", d);
            assert_eq!(a1.is_some(), a2.is_some());
        }
    }
}
```

---

## 9. Performance Benchmarks

### 9.1 Benchmark Suite

**Location:** `benches/trueno_integration.rs` (new file)

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use trueno::Vector;

fn bench_statistics_scalar(c: &mut Criterion) {
    let data: Vec<f32> = (0..10000).map(|i| i as f32).collect();

    c.bench_function("statistics_scalar", |b| {
        b.iter(|| {
            let sum: f32 = black_box(&data).iter().sum();
            let mean = sum / data.len() as f32;
            let variance: f32 = data.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32;
            let stddev = variance.sqrt();
            (mean, stddev)
        });
    });
}

fn bench_statistics_trueno(c: &mut Criterion) {
    let data: Vec<f32> = (0..10000).map(|i| i as f32).collect();
    let v = Vector::from_slice(&data);

    c.bench_function("statistics_trueno", |b| {
        b.iter(|| {
            let mean = black_box(&v).mean().unwrap();
            let stddev = black_box(&v).stddev().unwrap();
            (mean, stddev)
        });
    });
}

criterion_group!(benches, bench_statistics_scalar, bench_statistics_trueno);
criterion_main!(benches);
```

**Run:**
```bash
cargo bench --bench trueno_integration
```

### 9.2 Expected Results

**Target Speedups (based on Trueno benchmarks):**

| Operation | Data Size | Scalar Time | Trueno Time | Speedup |
|-----------|-----------|-------------|-------------|---------|
| `sum()` | 10K | ~40 µs | ~12 µs | 3.15x |
| `mean()` | 10K | ~40 µs | ~12 µs | 3.15x |
| `stddev()` | 10K | ~80 µs | ~24 µs | 3.33x |
| `dot()` | 10K | ~100 µs | ~30 µs | 3.4x |
| `correlation()` | 10K | ~200 µs | ~60 µs | 3.33x |

**Acceptable Range:** 2.5x - 4x faster (depending on CPU features)

### 9.3 Regression Prevention

**Pre-commit Hook Addition:**
```bash
# .git/hooks/pre-commit
echo "Running benchmarks (quick check)..."
cargo bench --bench trueno_integration -- --quick

# Fail if regression >10%
if [ $? -ne 0 ]; then
    echo "❌ Performance regression detected!"
    exit 1
fi
```

---

## 10. Migration Path

### 10.1 Backward Compatibility

**Principle:** All existing functionality MUST work unchanged

**Strategy:**
1. **New flags only**: `--stats-extended`, `--detect-anomalies` (opt-in)
2. **Default behavior preserved**: `-c` output unchanged
3. **Fallback mechanisms**: Scalar implementation if Trueno fails
4. **Deprecation policy**: No removals, only additions

### 10.2 Phased Rollout

**Sprint 19 (Phase 1):**
- ✅ Introduce `--stats-extended` flag
- ✅ Add percentile calculations
- ✅ Enhanced summary output
- ❌ No breaking changes to existing flags

**Sprint 20 (Phase 2):**
- ✅ Introduce `--detect-anomalies` flag
- ✅ Real-time anomaly detection
- ✅ Anomaly report in summary
- ❌ No breaking changes to existing flags

**Future Sprints:**
- ✅ Performance modeling (new `--predict` flag)
- ✅ ML-based classification (new `--classify` flag)
- ❌ No breaking changes to existing flags

### 10.3 Documentation Updates

**Files to Update:**
1. `README.md`: Add examples for new flags
2. `CHANGELOG.md`: Document Sprint 19/20 changes
3. `src/cli.rs`: Update help text with new flags
4. `docs/`: Create performance analysis guide

---

## Appendix A: Trueno API Quick Reference

### Vector Operations

```rust
// Construction
let v = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0]);

// Arithmetic
v.add(&other)?      // Element-wise addition
v.sub(&other)?      // Element-wise subtraction
v.mul(&other)?      // Element-wise multiplication
v.div(&other)?      // Element-wise division

// Reductions
v.sum()?            // Sum of all elements (SIMD)
v.max()?            // Maximum element
v.min()?            // Minimum element
v.argmax()?         // Index of maximum
v.argmin()?         // Index of minimum

// Statistics
v.mean()?           // Arithmetic mean (SIMD)
v.variance()?       // Population variance
v.stddev()?         // Standard deviation (SIMD)
v.covariance(&y)?   // Covariance with another vector
v.correlation(&y)?  // Pearson correlation
v.dot(&other)?      // Dot product (SIMD, 3.4x faster)

// Advanced Statistics
v.sum_kahan()?      // Numerically stable sum
v.sum_of_squares()? // Σx²
v.zscore()?         // Z-score normalization

// Normalization
v.minmax_normalize()? // Scale to [0, 1]
v.normalize()?        // L2 normalization
v.clip(min, max)?     // Clamp values

// Norms
v.norm_l1()?        // L1 norm (sum of abs)
v.norm_l2()?        // L2 norm (Euclidean)
v.norm_linf()?      // L∞ norm (max abs)
v.abs()?            // Absolute values

// Activations (for ML)
v.relu()?           // max(0, x)
v.sigmoid()?        // σ(x) = 1/(1+e^-x)
v.softmax()?        // Softmax activation
v.leaky_relu(α)?    // max(αx, x)
v.gelu()?           // Gaussian Error Linear Unit
```

---

## Appendix B: Example: Complete Phase 1 Implementation

### Complete Example: Enhanced Statistics Mode

**Test (RED Phase):**
```rust
// tests/sprint19_enhanced_stats_tests.rs

#[test]
fn test_stats_extended_calculates_percentiles() {
    let tmp_dir = TempDir::new().unwrap();
    let test_program = tmp_dir.path().join("test_app");

    // Create test program with known syscall pattern
    let source = r#"
    #include <unistd.h>
    int main() {
        for (int i = 0; i < 100; i++) {
            write(1, "x", 1);
        }
        return 0;
    }
    "#;
    // ... compile program ...

    let mut cmd = Command::cargo_bin("renacer").unwrap();
    cmd.arg("-c")
        .arg("--stats-extended")
        .arg("--")
        .arg(&test_program);

    cmd.assert()
        .success()
        .stdout(predicate::str::contains("Mean:"))
        .stdout(predicate::str::contains("Std Deviation:"))
        .stdout(predicate::str::contains("Median (P50):"))
        .stdout(predicate::str::contains("P95:"))
        .stdout(predicate::str::contains("P99:"));
}
```

**Implementation (GREEN Phase):**
```rust
// src/stats.rs

impl StatsTracker {
    pub fn print_extended_summary(&self) {
        // ... existing print_summary() code ...

        // NEW: Extended statistics
        if !self.stats.is_empty() {
            let statistical_summary = self.calculate_statistics_with_trueno();

            eprintln!("\n╔════════════════════════════════════════════════════════════════════════════╗");
            eprintln!("║  Extended Statistical Summary (SIMD-Accelerated via Trueno)               ║");
            eprintln!("╚════════════════════════════════════════════════════════════════════════════╝\n");

            eprintln!("Syscall Duration Statistics:");
            eprintln!("  Mean:              {:.2} µs", statistical_summary.mean);
            eprintln!("  Std Deviation:     {:.2} µs (CV: {:.1}%)",
                statistical_summary.stddev,
                (statistical_summary.stddev / statistical_summary.mean) * 100.0);
            eprintln!("  Min:               {:.2} µs", statistical_summary.min);
            eprintln!("  Max:               {:.2} µs", statistical_summary.max);
            eprintln!("  Median (P50):      {:.2} µs", statistical_summary.median);
            eprintln!("  P95:               {:.2} µs", statistical_summary.p95);
            eprintln!("  P99:               {:.2} µs", statistical_summary.p99);
        }
    }

    fn calculate_statistics_with_trueno(&self) -> StatisticalSummary {
        // Expand all syscalls to per-call durations
        let mut all_durations = Vec::new();
        for (_, stats) in &self.stats {
            let avg_duration = if stats.count > 0 {
                stats.total_time_us as f32 / stats.count as f32
            } else {
                0.0
            };
            // Approximate: repeat avg duration for each call
            for _ in 0..stats.count {
                all_durations.push(avg_duration);
            }
        }

        if all_durations.is_empty() {
            return StatisticalSummary::default();
        }

        let v = Vector::from_slice(&all_durations);

        StatisticalSummary {
            mean: v.mean().unwrap_or(0.0),
            stddev: v.stddev().unwrap_or(0.0),
            min: v.min().unwrap_or(0.0),
            max: v.max().unwrap_or(0.0),
            median: calculate_percentile(&v, 50.0),
            p95: calculate_percentile(&v, 95.0),
            p99: calculate_percentile(&v, 99.0),
        }
    }
}

fn calculate_percentile(v: &Vector<f32>, percentile: f32) -> f32 {
    // Simple percentile: sort and index
    let mut sorted = v.as_slice().to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let idx = ((percentile / 100.0) * sorted.len() as f32) as usize;
    sorted.get(idx.min(sorted.len() - 1)).copied().unwrap_or(0.0)
}

#[derive(Debug, Default)]
pub struct StatisticalSummary {
    pub mean: f32,
    pub stddev: f32,
    pub min: f32,
    pub max: f32,
    pub median: f32,
    pub p95: f32,
    pub p99: f32,
}
```

**Refactor (REFACTOR Phase):**
- Extract `calculate_percentile()` into separate module (`src/percentile.rs`)
- Add caching for repeated percentile calculations
- Add unit tests for `StatisticalSummary`
- Verify complexity ≤10 for all functions

---

## Appendix C: References

1. **Trueno Documentation:** `/home/noah/src/trueno/README.md`
2. **Trueno API:** `/home/noah/src/trueno/src/vector.rs`
3. **Renacer Statistics:** `/home/noah/src/renacer/src/stats.rs`
4. **PMAT Workflow:** `https://github.com/paiml/paiml-mcp-agent-toolkit`
5. **EXTREME TDD Methodology:** Renacer CHANGELOG.md Sprint 11-14

---

## Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-11-17 | Claude Code | Initial specification |

**Approval Required:** Product Owner (Noah Gift)
**Next Review:** Post-Sprint 19 Retrospective
**Status:** ✅ Ready for Implementation
