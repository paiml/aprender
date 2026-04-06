# Benchmarking Specification: Common Models & Common Serving

**Version:** 1.1.0
**Status:** Draft
**Authors:** Realizar Development Team
**Date:** 2025-11-27
**Document ID:** SPEC-BENCH-001

---

## Abstract

This specification defines a scientifically rigorous, reproducible benchmarking methodology for evaluating Realizar's ML inference performance against industry-standard serving frameworks. The methodology leverages Alimentar's PyTorch-parity datasets and Trueno's statistical benchmarking patterns to provide fair, transparent, and reproducible performance comparisons. Our goal is to demonstrate that pure Rust ML inference can achieve state-of-the-art performance while maintaining memory safety and zero-copy data handling.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Scope and Objectives](#2-scope-and-objectives)
3. [Reference Frameworks](#3-reference-frameworks)
4. [Dataset Specification](#4-dataset-specification)
5. [Model Zoo](#5-model-zoo)
6. [Benchmarking Methodology](#6-benchmarking-methodology)
7. [Statistical Analysis](#7-statistical-analysis)
8. [Hardware Requirements](#8-hardware-requirements)
9. [Reproducibility Protocol](#9-reproducibility-protocol)
10. [Metrics and Reporting](#10-metrics-and-reporting)
11. [Implementation](#11-implementation)
12. [References](#12-references)
13. [Appendices](#13-appendices)

---

## 1. Introduction

### 1.1 Background

ML inference serving has become a critical infrastructure component for production AI systems. The dominant frameworks—llama.cpp, vLLM, TensorRT-LLM, and ONNX Runtime—each make different trade-offs between performance, flexibility, and implementation complexity [1]. This specification establishes a methodology for fair comparison that accounts for these trade-offs while demonstrating the viability of pure Rust implementations.

### 1.2 Motivation

The ML systems research community has identified several challenges in benchmarking inference systems [2]:

1. **Lack of standardization**: Different frameworks use different metrics, workloads, and measurement methodologies
2. **Cherry-picking**: Results often highlight best-case scenarios rather than representative workloads
3. **Hardware specificity**: Performance varies dramatically across hardware configurations
4. **Reproducibility crisis**: Many published benchmarks cannot be independently verified

This specification addresses these challenges by providing:
- Standardized workloads based on canonical datasets (Alimentar)
- Comprehensive metrics covering latency, throughput, and memory
- Statistical rigor following established benchmarking methodology (Criterion.rs)
- Full reproducibility with deterministic seeding and hardware specifications

### 1.3 Design Principles

Following MLPerf™ Inference benchmarking principles [3]:

| Principle | Implementation |
|-----------|----------------|
| **Transparency** | Open-source benchmarks, methodology, and results |
| **Fairness** | Equivalent configurations and workloads across frameworks |
| **Honesty** | Report both strengths and weaknesses |
| **Context** | Include system specifications and methodology |
| **Reproducibility** | Deterministic seeds, scripts, and instructions |

---

## 2. Scope and Objectives

### 2.1 In Scope

- **Inference latency**: Token generation time across model sizes
- **Throughput**: Tokens per second under various batch sizes
- **Memory efficiency**: Peak and steady-state memory consumption
- **Cold start performance**: Time to first token from process start
- **Quantization impact**: Performance across Q4_0, Q8_0, FP16, FP32

### 2.2 Out of Scope

- Training performance (handled separately)
- Distributed inference across multiple nodes
- Custom hardware accelerators (TPU, Gaudi)
- Model accuracy/quality (assumed equivalent for same weights)

### 2.3 Success Criteria

Realizar aims to demonstrate:

| Metric | Target | Measurement |
|--------|--------|-------------|
| Latency (p50) | ≤100ms for 1B models | Forward pass + sampling |
| Latency (p95) | ≤200ms for 1B models | Under load |
| Throughput | ≥100 tokens/sec | Batch inference |
| Memory overhead | ≤512MB | Runtime excluding model |
| Cold start | ≤500ms | Process start to first token |

---

## 3. Reference Frameworks

### 3.1 Primary Comparisons

| Framework | Version | Configuration | Notes |
|-----------|---------|---------------|-------|
| **llama.cpp** | v3547+ | CPU-only, AVX2 | C++ reference implementation |
| **vLLM** | 0.6+ | CPU-only | PagedAttention, Python |
| **ONNX Runtime** | 1.18+ | CPU-only | Microsoft, C++ with Python bindings |
| **Candle** | 0.6+ | CPU-only | HuggingFace Rust |

### 3.2 Configuration Parity

All frameworks MUST be configured with equivalent settings:

```yaml
common_settings:
  threads: 1  # Single-threaded for fair comparison
  batch_size: [1, 2, 4, 8, 16, 32]
  precision: [fp32, fp16, q8_0, q4_0]
  context_length: [512, 1024, 2048, 4096]
  kv_cache: enabled
  flash_attention: disabled  # For reproducibility
```

### 3.3 Build Configuration

```bash
# llama.cpp
cmake -B build -DLLAMA_NATIVE=OFF -DLLAMA_AVX2=ON -DLLAMA_F16C=ON
cmake --build build --config Release

# vLLM
uv install vllm --no-cache-dir
export VLLM_CPU_ONLY=1

# ONNX Runtime
uv install onnxruntime

# Candle
cargo build --release --features accelerate

# Realizar
cargo build --release --features simd
```

---

## 4. Dataset Specification

### 4.1 Alimentar Integration

Benchmarks use Alimentar's canonical datasets for standardized, reproducible data loading with PyTorch parity [4].

#### 4.1.1 Dataset Loading API

```rust
use alimentar::datasets::{mnist, cifar10, CanonicalDataset};
use alimentar::DataLoader;

// MNIST for embedding benchmarks
let mnist_dataset = mnist()?;
let (train, test) = mnist_dataset.split()?;  // 80/20 split

// DataLoader with deterministic shuffling
let loader = DataLoader::new(train)
    .batch_size(32)
    .shuffle(true)
    .seed(42);  // Reproducibility

for batch in loader {
    // Process batch...
}
```

#### 4.1.2 Canonical Datasets

| Dataset | Size | Features | Use Case |
|---------|------|----------|----------|
| **MNIST** | 70K samples | 784 (28×28) | Embedding layer benchmarks |
| **CIFAR-10** | 60K samples | 3072 (32×32×3) | Vision model benchmarks |
| **CIFAR-100** | 60K samples | 3072, 100 classes | Fine-grained classification |
| **Fashion-MNIST** | 70K samples | 784 | Alternative vision benchmark |
| **Iris** | 150 samples | 4 features | Baseline/smoke tests |

#### 4.1.3 Text Benchmark Corpora

For language model inference, we define standardized prompt sets:

```rust
/// Standard benchmark prompts with varying complexity
pub const BENCHMARK_PROMPTS: &[BenchmarkPrompt] = &[
    BenchmarkPrompt {
        id: "short_qa",
        text: "What is the capital of France?",
        expected_tokens: 5..20,
        category: PromptCategory::ShortAnswer,
    },
    BenchmarkPrompt {
        id: "medium_generation",
        text: "Explain the concept of machine learning in simple terms.",
        expected_tokens: 50..150,
        category: PromptCategory::MediumGeneration,
    },
    BenchmarkPrompt {
        id: "long_context",
        text: include_str!("../data/long_context_prompt.txt"),
        expected_tokens: 200..500,
        category: PromptCategory::LongContext,
    },
    BenchmarkPrompt {
        id: "code_generation",
        text: "Write a Python function to calculate fibonacci numbers.",
        expected_tokens: 30..100,
        category: PromptCategory::CodeGeneration,
    },
];
```

### 4.2 Data Quality Assurance

Following Alimentar's quality checking infrastructure:

```rust
use alimentar::quality::{QualityChecker, QualityReport};

let checker = QualityChecker::new()
    .max_null_ratio(0.0)  // No missing values allowed
    .detect_outliers(true);

let report = checker.check(&dataset)?;
assert!(!report.has_critical_issues(), "Dataset quality check failed");
```

---

## 5. Model Zoo

### 5.1 Reference Models

Standardized models for reproducible comparison:

| Model | Parameters | Format | Source | Use Case |
|-------|------------|--------|--------|----------|
| **TinyLlama-1.1B** | 1.1B | GGUF Q4_0 | HuggingFace | Primary benchmark |
| **Phi-2** | 2.7B | GGUF Q4_0 | Microsoft | Medium model |
| **Llama-2-7B** | 7B | GGUF Q4_0 | Meta | Large model |
| **Mistral-7B** | 7B | GGUF Q4_0 | Mistral AI | Alternative 7B |
| **GPT-2 Small** | 124M | Safetensors | OpenAI | Fast iteration |

### 5.2 Model Configuration

```rust
/// Standard model configurations for benchmarking
pub struct ModelBenchConfig {
    pub name: &'static str,
    pub path: PathBuf,
    pub format: ModelFormat,
    pub quantization: Quantization,
    pub context_length: usize,
    pub expected_memory_mb: usize,
}

pub const BENCHMARK_MODELS: &[ModelBenchConfig] = &[
    ModelBenchConfig {
        name: "tinyllama-1.1b-q4_0",
        format: ModelFormat::GGUF,
        quantization: Quantization::Q4_0,
        context_length: 2048,
        expected_memory_mb: 700,
    },
    // ... additional models
];
```

### 5.3 Model Integrity Verification

```bash
# SHA-256 checksums for reproducibility
sha256sum models/*.gguf > checksums.txt

# Verify before benchmarking
sha256sum -c checksums.txt
```

---

## 6. Benchmarking Methodology

### 6.1 Criterion.rs Framework

Following Trueno's proven statistical benchmarking methodology [5]:

```rust
use criterion::{
    criterion_group, criterion_main, Criterion,
    BenchmarkId, Throughput, BatchSize,
};

fn inference_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("inference");

    // Statistical configuration
    group.sample_size(100);           // 100 samples for statistical rigor
    group.warm_up_time(Duration::from_secs(3));  // Cache warmup
    group.measurement_time(Duration::from_secs(10));

    for batch_size in [1, 2, 4, 8, 16, 32].iter() {
        group.throughput(Throughput::Elements(*batch_size as u64));

        group.bench_with_input(
            BenchmarkId::new("forward_pass", batch_size),
            batch_size,
            |b, &size| {
                b.iter_batched(
                    || setup_inference(size),
                    |input| model.forward(input),
                    BatchSize::SmallInput,
                );
            },
        );
    }

    group.finish();
}
```

### 6.2 Warmup Protocol

Critical for reproducible results [6]:

```rust
/// Warmup protocol to stabilize measurements
pub struct WarmupProtocol {
    /// Number of warmup iterations
    pub iterations: usize,
    /// Minimum warmup duration
    pub min_duration: Duration,
    /// Verify output stability
    pub verify_stability: bool,
}

impl Default for WarmupProtocol {
    fn default() -> Self {
        Self {
            iterations: 10,
            min_duration: Duration::from_secs(3),
            verify_stability: true,
        }
    }
}

pub fn warmup_model(model: &Model, protocol: &WarmupProtocol) -> Result<()> {
    let start = Instant::now();
    let mut outputs = Vec::with_capacity(protocol.iterations);

    for _ in 0..protocol.iterations {
        let output = model.forward(&WARMUP_INPUT)?;
        outputs.push(output);
    }

    // Verify numerical stability
    if protocol.verify_stability {
        verify_output_stability(&outputs)?;
    }

    // Ensure minimum duration
    if start.elapsed() < protocol.min_duration {
        std::thread::sleep(protocol.min_duration - start.elapsed());
    }

    Ok(())
}
```

### 6.3 Measurement Protocol

Following established systems benchmarking practices [7]:

```rust
/// Complete measurement protocol
pub struct MeasurementProtocol {
    // Latency measurements
    pub latency_samples: usize,      // Default: 100
    pub latency_percentiles: Vec<f64>, // [50, 90, 95, 99, 99.9]

    // Throughput measurements
    pub throughput_duration: Duration, // Default: 60s
    pub throughput_ramp_up: Duration,  // Default: 10s

    // Memory measurements
    pub memory_samples: usize,       // Default: 10
    pub memory_interval: Duration,   // Default: 1s
}

pub async fn run_measurement(
    model: &Model,
    protocol: &MeasurementProtocol,
) -> MeasurementResult {
    // Phase 1: Latency measurement
    let latencies = measure_latencies(model, protocol.latency_samples).await;

    // Phase 2: Throughput measurement
    let throughput = measure_throughput(
        model,
        protocol.throughput_duration,
        protocol.throughput_ramp_up,
    ).await;

    // Phase 3: Memory profiling
    let memory = measure_memory(model, protocol.memory_samples).await;

    MeasurementResult {
        latencies,
        throughput,
        memory,
        timestamp: Utc::now(),
    }
}
```

### 6.4 Workload Scenarios

| Scenario | Description | Configuration |
|----------|-------------|---------------|
| **Single Request** | One-at-a-time inference | batch_size=1, concurrency=1 |
| **Batch Processing** | Multiple concurrent requests | batch_size=[1,2,4,8,16,32] |
| **Sustained Load** | Continuous request stream | 60s duration, target QPS |
| **Burst Traffic** | Sudden load spikes | 10x normal load for 5s |
| **Long Context** | Extended context windows | context=[512,1K,2K,4K] |

---

## 7. Statistical Analysis

### 7.1 Descriptive Statistics

Following MLPerf™ reporting standards [3]:

```rust
pub struct LatencyStatistics {
    pub mean: Duration,
    pub std_dev: Duration,
    pub min: Duration,
    pub max: Duration,
    pub p50: Duration,   // Median
    pub p90: Duration,
    pub p95: Duration,
    pub p99: Duration,
    pub p999: Duration,  // Tail latency
    pub samples: usize,
    pub confidence_interval_95: (Duration, Duration),
}

impl LatencyStatistics {
    pub fn from_samples(samples: &[Duration]) -> Self {
        let n = samples.len() as f64;
        let mean = samples.iter().sum::<Duration>() / samples.len() as u32;

        // Standard deviation
        let variance = samples.iter()
            .map(|s| (s.as_nanos() as f64 - mean.as_nanos() as f64).powi(2))
            .sum::<f64>() / (n - 1.0);
        let std_dev = Duration::from_nanos(variance.sqrt() as u64);

        // 95% confidence interval (t-distribution for small samples)
        let t_value = t_distribution_critical_value(0.95, samples.len() - 1);
        let margin = std_dev.as_nanos() as f64 * t_value / n.sqrt();

        Self {
            mean,
            std_dev,
            min: *samples.iter().min().unwrap(),
            max: *samples.iter().max().unwrap(),
            p50: percentile(samples, 0.50),
            p90: percentile(samples, 0.90),
            p95: percentile(samples, 0.95),
            p99: percentile(samples, 0.99),
            p999: percentile(samples, 0.999),
            samples: samples.len(),
            confidence_interval_95: (
                Duration::from_nanos((mean.as_nanos() as f64 - margin) as u64),
                Duration::from_nanos((mean.as_nanos() as f64 + margin) as u64),
            ),
        }
    }
}
```

### 7.2 Outlier Detection

Using Median Absolute Deviation (MAD) method [8]:

```rust
/// Detect outliers using MAD (Median Absolute Deviation)
/// More robust than standard deviation for non-normal distributions
pub fn detect_outliers(samples: &[f64], threshold: f64) -> Vec<usize> {
    let median = percentile_f64(samples, 0.5);

    // Calculate MAD
    let deviations: Vec<f64> = samples.iter()
        .map(|x| (x - median).abs())
        .collect();
    let mad = percentile_f64(&deviations, 0.5);

    // Modified Z-score
    let k = 1.4826; // Constant for normal distribution approximation
    let outliers: Vec<usize> = samples.iter()
        .enumerate()
        .filter(|(_, &x)| {
            let modified_z = (x - median) / (k * mad);
            modified_z.abs() > threshold
        })
        .map(|(i, _)| i)
        .collect();

    outliers
}
```

### 7.3 Regression Detection

Following Trueno's methodology:

```rust
/// Performance regression detection
pub struct RegressionDetector {
    pub baseline: BenchmarkResults,
    pub warning_threshold: f64,   // Default: 0.02 (2%)
    pub failure_threshold: f64,   // Default: 0.05 (5%)
}

impl RegressionDetector {
    pub fn compare(&self, current: &BenchmarkResults) -> RegressionReport {
        let mut regressions = Vec::new();
        let mut warnings = Vec::new();
        let mut improvements = Vec::new();

        for (name, baseline_metric) in &self.baseline.metrics {
            if let Some(current_metric) = current.metrics.get(name) {
                let change = (current_metric.mean - baseline_metric.mean)
                    / baseline_metric.mean;

                if change > self.failure_threshold {
                    regressions.push(Regression {
                        metric: name.clone(),
                        baseline: baseline_metric.mean,
                        current: current_metric.mean,
                        change_percent: change * 100.0,
                    });
                } else if change > self.warning_threshold {
                    warnings.push(/* ... */);
                } else if change < -self.warning_threshold {
                    improvements.push(/* ... */);
                }
            }
        }

        RegressionReport {
            regressions,
            warnings,
            improvements,
            passed: regressions.is_empty(),
        }
    }
}
```

### 7.4 Statistical Significance Testing

Using Welch's t-test for comparing frameworks [9]:

```rust
/// Welch's t-test for unequal variances
pub fn welchs_t_test(
    sample1: &[f64],
    sample2: &[f64],
) -> TTestResult {
    let n1 = sample1.len() as f64;
    let n2 = sample2.len() as f64;

    let mean1 = sample1.iter().sum::<f64>() / n1;
    let mean2 = sample2.iter().sum::<f64>() / n2;

    let var1 = variance(sample1);
    let var2 = variance(sample2);

    let se = ((var1 / n1) + (var2 / n2)).sqrt();
    let t_statistic = (mean1 - mean2) / se;

    // Welch-Satterthwaite degrees of freedom
    let df = ((var1/n1 + var2/n2).powi(2)) /
        ((var1/n1).powi(2)/(n1-1.0) + (var2/n2).powi(2)/(n2-1.0));

    let p_value = 2.0 * (1.0 - t_distribution_cdf(t_statistic.abs(), df));

    TTestResult {
        t_statistic,
        degrees_of_freedom: df,
        p_value,
        significant: p_value < 0.05,
    }
}
```

### 7.5 Log-Transform Analysis for Skewed Distributions

Latency measurements typically follow log-normal or heavy-tailed distributions rather than normal distributions [11]. Applying parametric tests (like t-tests) directly to raw latency data can lead to biased or underpowered results. Following Box et al.'s recommendations, we employ log-transformation for latency metrics.

#### 7.5.1 Distribution Detection

```rust
/// Detect if samples follow a log-normal distribution
/// using the Shapiro-Wilk test on log-transformed data
pub struct DistributionAnalysis {
    pub raw_skewness: f64,
    pub raw_kurtosis: f64,
    pub log_skewness: f64,
    pub log_kurtosis: f64,
    pub recommended_transform: TransformType,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TransformType {
    /// Data is approximately normal, no transform needed
    None,
    /// Log-transform recommended (log-normal distribution)
    Log,
    /// Box-Cox transform for other skewed distributions
    BoxCox(f64),
}

impl DistributionAnalysis {
    pub fn analyze(samples: &[f64]) -> Self {
        let raw_skewness = skewness(samples);
        let raw_kurtosis = kurtosis(samples);

        // Log-transform the data
        let log_samples: Vec<f64> = samples.iter()
            .map(|&x| x.ln())
            .collect();
        let log_skewness = skewness(&log_samples);
        let log_kurtosis = kurtosis(&log_samples);

        // Recommend transform based on skewness reduction
        let recommended_transform = if raw_skewness.abs() < 0.5 {
            TransformType::None
        } else if log_skewness.abs() < raw_skewness.abs() * 0.5 {
            TransformType::Log
        } else {
            // Estimate optimal Box-Cox lambda
            let lambda = estimate_boxcox_lambda(samples);
            TransformType::BoxCox(lambda)
        };

        Self {
            raw_skewness,
            raw_kurtosis,
            log_skewness,
            log_kurtosis,
            recommended_transform,
        }
    }
}

/// Calculate skewness (third standardized moment)
fn skewness(samples: &[f64]) -> f64 {
    let n = samples.len() as f64;
    let mean = samples.iter().sum::<f64>() / n;
    let std_dev = variance(samples).sqrt();

    let m3 = samples.iter()
        .map(|x| ((x - mean) / std_dev).powi(3))
        .sum::<f64>() / n;

    // Adjusted Fisher-Pearson coefficient
    m3 * (n * (n - 1.0)).sqrt() / (n - 2.0)
}
```

#### 7.5.2 Log-Transformed Statistics

```rust
/// Latency statistics computed on log-transformed data
/// with back-transformation for interpretable results
pub struct LogTransformedStatistics {
    /// Geometric mean (exp of mean of logs)
    pub geometric_mean: Duration,
    /// Geometric standard deviation (multiplicative)
    pub geometric_std_dev: f64,
    /// Confidence interval on original scale
    pub confidence_interval_95: (Duration, Duration),
    /// Whether log-transform was applied
    pub transformed: bool,
    /// Original distribution analysis
    pub distribution: DistributionAnalysis,
}

impl LogTransformedStatistics {
    pub fn from_samples(samples: &[Duration]) -> Self {
        let values: Vec<f64> = samples.iter()
            .map(|d| d.as_nanos() as f64)
            .collect();

        let distribution = DistributionAnalysis::analyze(&values);

        match distribution.recommended_transform {
            TransformType::Log | TransformType::BoxCox(_) => {
                // Compute statistics on log scale
                let log_values: Vec<f64> = values.iter()
                    .map(|&x| x.ln())
                    .collect();

                let n = log_values.len() as f64;
                let log_mean = log_values.iter().sum::<f64>() / n;
                let log_var = log_values.iter()
                    .map(|x| (x - log_mean).powi(2))
                    .sum::<f64>() / (n - 1.0);
                let log_std = log_var.sqrt();

                // Back-transform to original scale
                let geometric_mean = Duration::from_nanos(log_mean.exp() as u64);
                let geometric_std_dev = log_std.exp();

                // Confidence interval on log scale, then back-transform
                let t_value = t_distribution_critical_value(0.95, samples.len() - 1);
                let margin = log_std * t_value / n.sqrt();
                let ci_low = Duration::from_nanos((log_mean - margin).exp() as u64);
                let ci_high = Duration::from_nanos((log_mean + margin).exp() as u64);

                Self {
                    geometric_mean,
                    geometric_std_dev,
                    confidence_interval_95: (ci_low, ci_high),
                    transformed: true,
                    distribution,
                }
            }
            TransformType::None => {
                // Fall back to arithmetic statistics
                let mean = values.iter().sum::<f64>() / values.len() as f64;
                let std_dev = variance(&values).sqrt();

                Self {
                    geometric_mean: Duration::from_nanos(mean as u64),
                    geometric_std_dev: 1.0 + (std_dev / mean), // Approximate
                    confidence_interval_95: compute_ci(&values),
                    transformed: false,
                    distribution,
                }
            }
        }
    }
}
```

#### 7.5.3 Log-Transformed Comparison

```rust
/// Compare two frameworks using log-transformed t-test
/// More appropriate for latency data than raw t-test
pub fn log_transformed_t_test(
    sample1: &[f64],
    sample2: &[f64],
) -> TTestResult {
    // Transform both samples
    let log1: Vec<f64> = sample1.iter().map(|x| x.ln()).collect();
    let log2: Vec<f64> = sample2.iter().map(|x| x.ln()).collect();

    // Apply Welch's t-test on log scale
    let result = welchs_t_test(&log1, &log2);

    // The ratio of geometric means
    let log_mean1 = log1.iter().sum::<f64>() / log1.len() as f64;
    let log_mean2 = log2.iter().sum::<f64>() / log2.len() as f64;
    let ratio = (log_mean1 - log_mean2).exp();

    TTestResult {
        t_statistic: result.t_statistic,
        degrees_of_freedom: result.degrees_of_freedom,
        p_value: result.p_value,
        significant: result.p_value < 0.05,
        // Additional field for log-transformed tests
        geometric_mean_ratio: Some(ratio),
    }
}
```

### 7.6 Non-Parametric Tests

When distributions are heavily skewed or multimodal, non-parametric tests provide robust alternatives that make no assumptions about the underlying distribution [11].

#### 7.6.1 Mann-Whitney U Test

The Mann-Whitney U test (also known as Wilcoxon rank-sum test) compares two independent samples without assuming normality:

```rust
/// Mann-Whitney U test for non-parametric comparison
/// Preferred over t-test when:
/// - Distribution is heavily skewed (skewness > 2)
/// - Sample sizes are small (n < 30)
/// - Outliers are present and meaningful
pub struct MannWhitneyResult {
    pub u_statistic: f64,
    pub z_score: f64,
    pub p_value: f64,
    pub significant: bool,
    /// Effect size (rank-biserial correlation)
    pub effect_size: f64,
    /// Interpretation of effect size
    pub effect_interpretation: EffectSize,
}

#[derive(Debug, Clone, Copy)]
pub enum EffectSize {
    Negligible,  // |r| < 0.1
    Small,       // 0.1 <= |r| < 0.3
    Medium,      // 0.3 <= |r| < 0.5
    Large,       // |r| >= 0.5
}

pub fn mann_whitney_u_test(
    sample1: &[f64],
    sample2: &[f64],
) -> MannWhitneyResult {
    let n1 = sample1.len();
    let n2 = sample2.len();

    // Combine and rank all observations
    let mut combined: Vec<(f64, usize)> = sample1.iter()
        .map(|&x| (x, 0))  // Group 0
        .chain(sample2.iter().map(|&x| (x, 1)))  // Group 1
        .collect();

    combined.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    // Assign ranks (handle ties by averaging)
    let ranks = assign_ranks_with_ties(&combined);

    // Sum of ranks for sample 1
    let r1: f64 = ranks.iter()
        .enumerate()
        .filter(|(_, (_, group))| *group == 0)
        .map(|(i, _)| ranks[i].0)
        .sum();

    // Calculate U statistics
    let u1 = r1 - (n1 * (n1 + 1)) as f64 / 2.0;
    let u2 = (n1 * n2) as f64 - u1;
    let u_statistic = u1.min(u2);

    // Normal approximation for large samples
    let mu = (n1 * n2) as f64 / 2.0;
    let sigma = ((n1 * n2 * (n1 + n2 + 1)) as f64 / 12.0).sqrt();
    let z_score = (u_statistic - mu) / sigma;

    // Two-tailed p-value
    let p_value = 2.0 * (1.0 - normal_cdf(z_score.abs()));

    // Effect size: rank-biserial correlation
    let effect_size = 1.0 - (2.0 * u_statistic) / (n1 * n2) as f64;

    let effect_interpretation = match effect_size.abs() {
        r if r < 0.1 => EffectSize::Negligible,
        r if r < 0.3 => EffectSize::Small,
        r if r < 0.5 => EffectSize::Medium,
        _ => EffectSize::Large,
    };

    MannWhitneyResult {
        u_statistic,
        z_score,
        p_value,
        significant: p_value < 0.05,
        effect_size,
        effect_interpretation,
    }
}

/// Assign ranks handling ties by averaging
fn assign_ranks_with_ties(sorted: &[(f64, usize)]) -> Vec<(f64, usize)> {
    let mut ranks = Vec::with_capacity(sorted.len());
    let mut i = 0;

    while i < sorted.len() {
        let value = sorted[i].0;
        let mut j = i;

        // Find all ties
        while j < sorted.len() && sorted[j].0 == value {
            j += 1;
        }

        // Average rank for ties
        let avg_rank = (i + 1..=j).sum::<usize>() as f64 / (j - i) as f64;

        for k in i..j {
            ranks.push((avg_rank, sorted[k].1));
        }

        i = j;
    }

    ranks
}
```

#### 7.6.2 Selecting the Appropriate Test

```rust
/// Automatic test selection based on data characteristics
pub enum StatisticalTest {
    WelchsTTest,
    LogTransformedTTest,
    MannWhitneyU,
}

pub fn select_appropriate_test(
    sample1: &[f64],
    sample2: &[f64],
) -> (StatisticalTest, Box<dyn TestResult>) {
    let dist1 = DistributionAnalysis::analyze(sample1);
    let dist2 = DistributionAnalysis::analyze(sample2);

    let max_skewness = dist1.raw_skewness.abs().max(dist2.raw_skewness.abs());
    let min_n = sample1.len().min(sample2.len());

    // Decision tree per Box et al. (2005) recommendations
    if max_skewness > 2.0 || min_n < 15 {
        // Heavily skewed or small samples: use non-parametric
        let result = mann_whitney_u_test(sample1, sample2);
        (StatisticalTest::MannWhitneyU, Box::new(result))
    } else if max_skewness > 0.5 {
        // Moderately skewed: use log-transform
        let result = log_transformed_t_test(sample1, sample2);
        (StatisticalTest::LogTransformedTTest, Box::new(result))
    } else {
        // Approximately normal: use Welch's t-test
        let result = welchs_t_test(sample1, sample2);
        (StatisticalTest::WelchsTTest, Box::new(result))
    }
}
```

#### 7.6.3 Reporting Guidelines

When reporting benchmark comparisons, include:

| Scenario | Recommended Test | Report |
|----------|-----------------|--------|
| Normal distribution (skew < 0.5) | Welch's t-test | Mean ± SD, 95% CI, p-value |
| Log-normal (0.5 < skew < 2.0) | Log-transformed t-test | Geometric mean, GSD, ratio, p-value |
| Heavy-tailed (skew > 2.0) | Mann-Whitney U | Median, IQR, U-statistic, effect size |
| Small samples (n < 15) | Mann-Whitney U | Median, range, exact p-value |

**Example Report Format:**

```markdown
## Framework Comparison: Realizar vs llama.cpp

**Distribution Analysis:**
- Realizar latency: skewness=1.23 (log-normal)
- llama.cpp latency: skewness=1.45 (log-normal)
- Test selected: Log-transformed t-test

**Results:**
| Metric | Realizar | llama.cpp | Ratio | p-value |
|--------|----------|-----------|-------|---------|
| Geometric Mean | 45.2ms | 52.1ms | 0.87x | 0.003 |
| GSD | 1.15 | 1.18 | - | - |
| p50 | 43.1ms | 49.8ms | 0.87x | - |
| p99 | 78.2ms | 95.3ms | 0.82x | - |

**Conclusion:** Realizar is 1.15x faster (p < 0.01, log-transformed t-test)
```

---

## 8. Hardware Requirements

### 8.1 Reference Hardware Configurations

| Tier | CPU | RAM | Storage | Use Case |
|------|-----|-----|---------|----------|
| **Tier 1: Entry** | 4-core x86_64 | 16GB | SSD | Minimum spec |
| **Tier 2: Standard** | 8-core x86_64, AVX2 | 32GB | NVMe | Default benchmark |
| **Tier 3: High-End** | 16-core x86_64, AVX-512 | 64GB | NVMe | Maximum performance |
| **Tier 4: Server** | 32-core x86_64, AVX-512 | 128GB | NVMe RAID | Production simulation |

### 8.2 SIMD Requirements

| Instruction Set | Required For | Detection |
|-----------------|--------------|-----------|
| SSE2 | Minimum baseline | Always available on x86_64 |
| AVX2 | Default benchmarks | `std::is_x86_feature_detected!("avx2")` |
| AVX-512 | High-performance | `std::is_x86_feature_detected!("avx512f")` |
| NEON | ARM systems | `std::arch::is_aarch64_feature_detected!("neon")` |

### 8.3 System Preparation

```bash
#!/bin/bash
# system_prep.sh - Prepare system for benchmarking

# Disable CPU frequency scaling (requires root)
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Disable turbo boost for consistency
echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo

# Set CPU affinity for benchmark process
taskset -c 0-3 cargo bench

# Disable ASLR for reproducible memory layouts
echo 0 | sudo tee /proc/sys/kernel/randomize_va_space

# Drop filesystem caches
sync && echo 3 | sudo tee /proc/sys/vm/drop_caches
```

### 8.4 Hardware Metadata Collection

```rust
/// Collect hardware metadata for reproducibility
pub struct HardwareMetadata {
    pub cpu_model: String,
    pub cpu_cores: usize,
    pub cpu_threads: usize,
    pub cpu_frequency_mhz: u32,
    pub simd_support: Vec<SimdFeature>,
    pub memory_total_gb: f64,
    pub memory_available_gb: f64,
    pub os_name: String,
    pub os_version: String,
    pub rust_version: String,
    pub commit_hash: String,
}

impl HardwareMetadata {
    pub fn collect() -> Self {
        Self {
            cpu_model: sys_info::cpu_model().unwrap_or_default(),
            cpu_cores: num_cpus::get_physical(),
            cpu_threads: num_cpus::get(),
            cpu_frequency_mhz: sys_info::cpu_speed().unwrap_or(0) as u32,
            simd_support: detect_simd_features(),
            memory_total_gb: sys_info::mem_total().unwrap_or(0) as f64 / 1024.0 / 1024.0,
            memory_available_gb: sys_info::mem_available().unwrap_or(0) as f64 / 1024.0 / 1024.0,
            os_name: std::env::consts::OS.to_string(),
            os_version: sys_info::os_release().unwrap_or_default(),
            rust_version: rustc_version::version().unwrap().to_string(),
            commit_hash: git_hash(),
        }
    }
}
```

---

## 9. Reproducibility Protocol

### 9.1 Deterministic Seeds

All random operations MUST use fixed seeds:

```rust
/// Global seed for all randomness in benchmarks
pub const BENCHMARK_SEED: u64 = 42;

/// Seeded RNG for reproducible shuffling
pub fn benchmark_rng() -> impl Rng {
    StdRng::seed_from_u64(BENCHMARK_SEED)
}

/// DataLoader configuration with fixed seed
let loader = DataLoader::new(dataset)
    .batch_size(32)
    .shuffle(true)
    .seed(BENCHMARK_SEED);  // Alimentar integration
```

### 9.2 Version Locking

```toml
# Cargo.lock MUST be committed for exact reproducibility

# benchmark-requirements.txt
criterion = "0.5.1"
alimentar = "0.1.0"
trueno = "0.4.2"
```

### 9.3 Environment Specification

```yaml
# benchmark-environment.yaml
rust:
  version: "1.82.0"
  edition: "2021"
  profile: "release"
  lto: "fat"
  codegen_units: 1
  opt_level: 3

dependencies:
  alimentar: "0.1.0"
  trueno: "0.4.2"
  criterion: "0.5.1"

system:
  cpu_governor: "performance"
  turbo_boost: disabled
  hyperthreading: enabled
  numa_balancing: disabled
```

### 9.4 Checksum Verification

```rust
/// Verify model and data integrity before benchmarking
pub fn verify_benchmark_integrity() -> Result<()> {
    // Verify model checksums
    for model in BENCHMARK_MODELS {
        let computed = sha256_file(&model.path)?;
        assert_eq!(computed, model.expected_checksum,
            "Model checksum mismatch: {}", model.name);
    }

    // Verify dataset integrity via Alimentar
    let mnist = mnist()?;
    assert_eq!(mnist.len(), 70000, "MNIST size mismatch");
    assert_eq!(mnist.num_features(), 784, "MNIST features mismatch");

    Ok(())
}
```

---

## 10. Metrics and Reporting

### 10.1 Core Metrics

| Metric | Unit | Description | Target |
|--------|------|-------------|--------|
| **TTFT** | ms | Time to First Token | <50ms |
| **TPOT** | ms | Time Per Output Token | <20ms |
| **Throughput** | tokens/s | Tokens generated per second | >100 |
| **Memory Peak** | MB | Maximum memory usage | <model_size × 1.2 |
| **Memory Steady** | MB | Steady-state memory | <model_size × 1.1 |

### 10.2 Derived Metrics

```rust
/// Calculated metrics for comprehensive analysis
pub struct DerivedMetrics {
    /// Tokens per second per watt (efficiency)
    pub tokens_per_watt: f64,

    /// Latency variance coefficient (stability)
    pub cv_latency: f64,

    /// Memory efficiency (tokens generated per MB)
    pub memory_efficiency: f64,

    /// Speedup vs baseline (llama.cpp)
    pub speedup_vs_baseline: f64,
}
```

### 10.3 Report Format

```json
{
  "benchmark_id": "realizar-bench-20251127-001",
  "timestamp": "2025-11-27T10:30:00Z",
  "hardware": {
    "cpu": "AMD Ryzen 9 5950X",
    "cores": 16,
    "memory_gb": 64,
    "simd": ["AVX2", "FMA"]
  },
  "software": {
    "realizar_version": "0.2.1",
    "rust_version": "1.82.0",
    "trueno_version": "0.4.2",
    "alimentar_version": "0.1.0"
  },
  "models": {
    "tinyllama-1.1b-q4_0": {
      "latency": {
        "ttft_ms": { "p50": 23.4, "p95": 28.1, "p99": 35.2 },
        "tpot_ms": { "p50": 12.1, "p95": 15.3, "p99": 18.7 }
      },
      "throughput": {
        "tokens_per_second": { "mean": 82.6, "std": 4.2 }
      },
      "memory": {
        "peak_mb": 720,
        "steady_mb": 695
      }
    }
  },
  "comparisons": {
    "llama_cpp": {
      "speedup": 1.15,
      "p_value": 0.003,
      "significant": true
    }
  }
}
```

### 10.4 Visualization

Generate standardized plots:

```rust
/// Generate benchmark visualization
pub fn generate_plots(results: &BenchmarkResults, output_dir: &Path) -> Result<()> {
    // 1. Latency distribution histogram
    plot_latency_distribution(&results.latencies, output_dir.join("latency_dist.png"))?;

    // 2. Throughput vs batch size
    plot_throughput_scaling(&results.throughput, output_dir.join("throughput_scale.png"))?;

    // 3. Memory timeline
    plot_memory_timeline(&results.memory, output_dir.join("memory_timeline.png"))?;

    // 4. Framework comparison bar chart
    plot_framework_comparison(&results.comparisons, output_dir.join("comparison.png"))?;

    // 5. Percentile ladder (p50, p90, p95, p99, p99.9)
    plot_percentile_ladder(&results.latencies, output_dir.join("percentiles.png"))?;

    Ok(())
}
```

---

## 11. Implementation

### 11.1 Benchmark Runner

```rust
//! benches/common_models_benchmark.rs
//!
//! Scientific benchmarking suite following SPEC-BENCH-001

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use alimentar::datasets::{mnist, cifar10, CanonicalDataset};
use alimentar::DataLoader;
use realizar::{Model, ModelConfig, InferenceConfig};
use std::time::Duration;

const BENCHMARK_SEED: u64 = 42;

fn setup_model(config: &ModelBenchConfig) -> Model {
    Model::load(&config.path, config.quantization).expect("Failed to load model")
}

fn inference_latency(c: &mut Criterion) {
    let mut group = c.benchmark_group("inference_latency");
    group.sample_size(100);
    group.warm_up_time(Duration::from_secs(3));
    group.measurement_time(Duration::from_secs(10));

    for model_config in BENCHMARK_MODELS {
        let model = setup_model(model_config);

        for &seq_len in &[1, 5, 10, 20, 50] {
            group.throughput(Throughput::Elements(seq_len as u64));

            group.bench_with_input(
                BenchmarkId::new(model_config.name, seq_len),
                &seq_len,
                |b, &len| {
                    let input = generate_input(len, BENCHMARK_SEED);
                    b.iter(|| model.forward(&input))
                },
            );
        }
    }

    group.finish();
}

fn throughput_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("throughput_scaling");
    group.sample_size(50);
    group.measurement_time(Duration::from_secs(30));

    for model_config in BENCHMARK_MODELS {
        let model = setup_model(model_config);

        for &batch_size in &[1, 2, 4, 8, 16, 32] {
            group.throughput(Throughput::Elements(batch_size as u64 * 100)); // 100 tokens

            group.bench_with_input(
                BenchmarkId::new(model_config.name, batch_size),
                &batch_size,
                |b, &size| {
                    let batch = generate_batch(size, 100, BENCHMARK_SEED);
                    b.iter(|| model.generate_batch(&batch))
                },
            );
        }
    }

    group.finish();
}

fn dataset_integration(c: &mut Criterion) {
    let mut group = c.benchmark_group("dataset_integration");

    // MNIST embedding benchmark
    let mnist_dataset = mnist().expect("Failed to load MNIST");
    let loader = DataLoader::new(mnist_dataset)
        .batch_size(32)
        .shuffle(true)
        .seed(BENCHMARK_SEED);

    group.bench_function("mnist_batch_processing", |b| {
        b.iter(|| {
            for batch in loader.clone() {
                // Simulate embedding lookup
                std::hint::black_box(batch.num_rows());
            }
        })
    });

    // CIFAR-10 vision benchmark
    let cifar_dataset = cifar10().expect("Failed to load CIFAR-10");
    let loader = DataLoader::new(cifar_dataset)
        .batch_size(16)
        .shuffle(true)
        .seed(BENCHMARK_SEED);

    group.bench_function("cifar10_batch_processing", |b| {
        b.iter(|| {
            for batch in loader.clone() {
                std::hint::black_box(batch.num_rows());
            }
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    inference_latency,
    throughput_scaling,
    dataset_integration,
);
criterion_main!(benches);
```

### 11.2 Cross-Framework Comparison Script

```python
#!/usr/bin/env python3
"""
compare_frameworks.py - Cross-framework benchmark comparison

Usage: python compare_frameworks.py --model tinyllama --output results/
"""

import argparse
import json
import subprocess
import time
import statistics
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class BenchmarkResult:
    framework: str
    model: str
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    throughput_tps: float
    memory_peak_mb: float
    samples: int

def run_realizar_benchmark(model_path: str, n_samples: int = 100) -> BenchmarkResult:
    """Run Realizar benchmark via CLI."""
    cmd = [
        "cargo", "run", "--release", "--bin", "realizar-bench",
        "--", "--model", model_path, "--samples", str(n_samples), "--json"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)

    return BenchmarkResult(
        framework="realizar",
        model=model_path,
        latency_p50_ms=data["latency"]["p50"],
        latency_p95_ms=data["latency"]["p95"],
        latency_p99_ms=data["latency"]["p99"],
        throughput_tps=data["throughput"]["mean"],
        memory_peak_mb=data["memory"]["peak_mb"],
        samples=n_samples,
    )

def run_llama_cpp_benchmark(model_path: str, n_samples: int = 100) -> BenchmarkResult:
    """Run llama.cpp benchmark."""
    latencies = []

    for _ in range(n_samples):
        start = time.perf_counter()
        subprocess.run([
            "llama-cli", "-m", model_path,
            "-p", "Hello, world!",
            "-n", "10", "--log-disable"
        ], capture_output=True, check=True)
        latencies.append((time.perf_counter() - start) * 1000)

    latencies.sort()

    return BenchmarkResult(
        framework="llama_cpp",
        model=model_path,
        latency_p50_ms=latencies[len(latencies) // 2],
        latency_p95_ms=latencies[int(len(latencies) * 0.95)],
        latency_p99_ms=latencies[int(len(latencies) * 0.99)],
        throughput_tps=10000 / statistics.mean(latencies),  # 10 tokens / mean_latency
        memory_peak_mb=0,  # TODO: measure via /proc
        samples=n_samples,
    )

def compare_results(results: List[BenchmarkResult]) -> Dict:
    """Generate comparison report."""
    baseline = next(r for r in results if r.framework == "llama_cpp")
    realizar = next(r for r in results if r.framework == "realizar")

    return {
        "speedup_p50": baseline.latency_p50_ms / realizar.latency_p50_ms,
        "speedup_p95": baseline.latency_p95_ms / realizar.latency_p95_ms,
        "throughput_ratio": realizar.throughput_tps / baseline.throughput_tps,
        "memory_ratio": realizar.memory_peak_mb / baseline.memory_peak_mb
            if baseline.memory_peak_mb > 0 else None,
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", default="results/")
    parser.add_argument("--samples", type=int, default=100)
    args = parser.parse_args()

    results = [
        run_realizar_benchmark(args.model, args.samples),
        run_llama_cpp_benchmark(args.model, args.samples),
    ]

    comparison = compare_results(results)

    output_path = Path(args.output) / "comparison.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump({
            "results": [r.__dict__ for r in results],
            "comparison": comparison,
        }, f, indent=2)

    print(f"Results saved to {output_path}")
    print(f"Speedup (p50): {comparison['speedup_p50']:.2f}x")
```

### 11.3 CI Integration

```yaml
# .github/workflows/benchmark.yml
name: Benchmark CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  benchmark:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Install Rust
        uses: dtolnay/rust-action@stable

      - name: Cache cargo
        uses: Swatinem/rust-cache@v2

      - name: Download benchmark models
        run: |
          mkdir -p models
          wget -q https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_0.gguf -O models/tinyllama-1.1b-q4_0.gguf

      - name: Run benchmarks
        run: cargo bench --bench common_models_benchmark -- --save-baseline pr-${{ github.event.number }}

      - name: Compare with main
        if: github.event_name == 'pull_request'
        run: |
          cargo bench --bench common_models_benchmark -- --baseline main --save-baseline pr-${{ github.event.number }}

      - name: Check for regressions
        run: |
          python scripts/check_regression.py \
            --baseline .criterion/main \
            --current .criterion/pr-${{ github.event.number }} \
            --threshold 0.05

      - name: Upload benchmark results
        uses: actions/upload-artifact@v4
        with:
          name: benchmark-results
          path: target/criterion/
```

---

## 12. References

[1] **Kwon, W., et al.** (2023). "Efficient Memory Management for Large Language Model Serving with PagedAttention." *Proceedings of the 29th Symposium on Operating Systems Principles (SOSP '23)*. ACM. https://doi.org/10.1145/3600006.3613165

[2] **Mattson, P., et al.** (2020). "MLPerf: An Industry Standard Benchmark Suite for Machine Learning Performance." *IEEE Micro*, 40(2), 8-16. https://doi.org/10.1109/MM.2020.2974843

[3] **Reddi, V. J., et al.** (2020). "MLPerf Inference Benchmark." *Proceedings of the 47th Annual International Symposium on Computer Architecture (ISCA '20)*. IEEE. https://doi.org/10.1109/ISCA45697.2020.00045

[4] **Schreiber, J., et al.** (2023). "Apache Arrow: A Cross-Language Development Platform for In-Memory Data." *Proceedings of the VLDB Endowment*, 16(12), 3937-3944. https://doi.org/10.14778/3611540.3611606

[5] **Georges, A., Buytaert, D., & Eeckhout, L.** (2007). "Statistically Rigorous Java Performance Evaluation." *Proceedings of the 22nd Annual ACM SIGPLAN Conference on Object-Oriented Programming, Systems, Languages, and Applications (OOPSLA '07)*. ACM. https://doi.org/10.1145/1297027.1297033

[6] **Mytkowicz, T., Diwan, A., Hauswirth, M., & Sweeney, P. F.** (2009). "Producing Wrong Data Without Doing Anything Obviously Wrong!" *Proceedings of the 14th International Conference on Architectural Support for Programming Languages and Operating Systems (ASPLOS XIV)*. ACM. https://doi.org/10.1145/1508244.1508275

[7] **Ousterhout, J. K.** (2018). "Always Measure One Level Deeper." *Communications of the ACM*, 61(7), 74-83. https://doi.org/10.1145/3213770

[8] **Leys, C., Ley, C., Klein, O., Bernard, P., & Licata, L.** (2013). "Detecting Outliers: Do Not Use Standard Deviation Around the Mean, Use Absolute Deviation Around the Median." *Journal of Experimental Social Psychology*, 49(4), 764-766. https://doi.org/10.1016/j.jesp.2013.03.013

[9] **Welch, B. L.** (1947). "The Generalization of 'Student's' Problem when Several Different Population Variances are Involved." *Biometrika*, 34(1-2), 28-35. https://doi.org/10.1093/biomet/34.1-2.28

[10] **Gerganov, G.** (2023). "GGML: AI at the Edge." *GitHub Repository*. https://github.com/ggerganov/ggml (Accessed: 2025-11-27). *Note: Reference implementation for quantized inference benchmarking.*

[11] **Box, G. E. P., Hunter, J. S., & Hunter, W. G.** (2005). *Statistics for Experimenters: Design, Innovation, and Discovery* (2nd ed.). Wiley-Interscience. ISBN: 978-0471718130. *Note: Foundational text recommending log-transformation and non-parametric tests for skewed data like latency distributions.*

---

## 13. Appendices

### Appendix A: Benchmark Checklist

Before running benchmarks:

- [ ] Verify model checksums match expected values
- [ ] Confirm system preparation script executed
- [ ] Validate CPU governor set to "performance"
- [ ] Check turbo boost is disabled
- [ ] Verify sufficient disk space for results
- [ ] Confirm no other CPU-intensive processes running
- [ ] Validate network disconnected (for isolation)
- [ ] Run warmup benchmarks to verify setup

### Appendix B: Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| High variance | CPU throttling | Check thermal state, disable turbo |
| Inconsistent results | Background processes | Kill unnecessary processes |
| Memory errors | Insufficient RAM | Reduce batch size or model size |
| SIMD not detected | Build flags | Rebuild with `-C target-cpu=native` |

### Appendix C: Model Download Script

```bash
#!/bin/bash
# download_benchmark_models.sh

MODELS_DIR="${MODELS_DIR:-./models}"
mkdir -p "$MODELS_DIR"

# TinyLlama 1.1B Q4_0
wget -c "https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_0.gguf" \
    -O "$MODELS_DIR/tinyllama-1.1b-q4_0.gguf"

# GPT-2 Small (Safetensors)
wget -c "https://huggingface.co/openai-community/gpt2/resolve/main/model.safetensors" \
    -O "$MODELS_DIR/gpt2-small.safetensors"

# Verify checksums
echo "Verifying checksums..."
sha256sum -c checksums.txt

echo "Models downloaded successfully!"
```

### Appendix D: Glossary

| Term | Definition |
|------|------------|
| **TTFT** | Time to First Token - latency from request to first generated token |
| **TPOT** | Time Per Output Token - average latency per subsequent token |
| **TPS** | Tokens Per Second - throughput metric |
| **KV Cache** | Key-Value cache for efficient autoregressive generation |
| **MAD** | Median Absolute Deviation - robust dispersion measure |
| **GGUF** | GPT-Generated Unified Format - quantized model format |
| **Q4_0** | 4-bit quantization with block size 32 |
| **Q8_0** | 8-bit quantization with block size 32 |

---

**Document History**

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.1.0 | 2025-11-27 | Realizar Team | Added §7.5 Log-Transform Analysis, §7.6 Non-Parametric Tests (per Gemini review), Box et al. [11] citation |
| 1.0.0 | 2025-11-27 | Realizar Team | Initial specification |

---

*This specification is maintained in the Realizar repository and follows the project's contribution guidelines.*
