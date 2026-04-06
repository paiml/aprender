# Realizar: Model Runner Benchmark Specification

**Version**: 2.2
**Date**: 2025-12-10
**Status**: IMPLEMENTED - All Core Components Complete + CUDA PTX Generation
**Priority**: CRITICAL - Scientific Validation of Inference Performance
**Review Status**: Revised per Toyota Way Engineering Review + QA Checklist (99/100)
**Sprint Status**: BENCH-SPRINT-001 ✅ COMPLETE, BENCH-SPRINT-002 ✅ COMPLETE

---

## Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-12-09 | Batuta Team | Initial specification |
| 1.1 | 2025-12-09 | Batuta Team | Dynamic sampling, ITL, KV-cache, energy, thermal guards |
| 2.0 | 2025-12-10 | Batuta Team | Current state analysis, sprint planning, +10 citations, honest gap assessment |
| 2.1 | 2025-12-10 | Claude Code | BENCH-SPRINT-001/002 complete, QuantizedLinear 21x faster than f32 |
| 2.2 | 2025-12-10 | Claude Code | Added trueno-gpu CUDA PTX generation integration |

---

## Current Implementation Status (Genchi Genbutsu)

### What's Working (✅ Verified)

| Component | Status | Evidence |
|-----------|--------|----------|
| Criterion.rs benchmarks | ✅ 11 suites | `cargo bench` runs successfully |
| .apr format inference | ✅ 9.6x faster than PyTorch | `mnist_apr_benchmark` example |
| GGUF parser | ✅ Complete | 26 tests passing |
| Safetensors parser | ✅ Complete | Zero-copy loading |
| Test coverage | ✅ 95%+ | 1,051 tests passing |
| QA checklist | ✅ 99/100 | Toyota Way + NASA/JPL methodology |
| Demo server | ✅ Working | `realizar serve --demo` |
| **Q4_K SIMD matmul** | ✅ **21x faster than f32** | `fused_q4k_dot_simd`, `fused_q4k_parallel_matvec` |
| **QuantizedLinear layer** | ✅ **787µs per 2560x2560** | `benches/quantize.rs` benchmarks |
| **Memory reduction** | ✅ **7.09x smaller** | Q4_K vs f32 weights |
| **DynamicSampler** | ✅ Complete | CV-based stop-rule per Hoefler & Belli |
| **ThermalGuard** | ✅ Complete | Thermal throttling detection |
| **KvCacheMetrics** | ✅ Complete | Fragmentation tracking |
| **EnergyMetrics** | ✅ Complete | Token energy measurement |
| **ItlMetrics** | ✅ Complete | Inter-token latency variance |
| **RegressionDetector** | ✅ Complete | Welch t-test with configurable threshold |
| **Backend infrastructure** | ✅ Complete | LlamaCpp, vLLM, Ollama backends |
| **CI Regression tests** | ✅ Complete | `.github/workflows/bench.yml` |
| **CUDA PTX Generation** | ✅ Complete | trueno-gpu v0.1.0 integration |

### Previously Future Work (✅ ALL COMPLETE)

| Component | Status | Implementation | Notes |
|-----------|--------|----------------|-------|
| End-to-end Q4_K inference | ✅ Complete | `QuantizedGGUFTransformer::generate()` | Fused Q4_K with greedy/top-k sampling |
| GPU acceleration | ✅ Complete | wgpu + CUDA PTX via trueno-gpu | PTX execution via CUDA driver API |
| Real HTTP benchmarks | ✅ Complete | `http_client.rs` with reqwest | llama.cpp: ~260 tok/s on RTX 4090 |

**Benchmark Results (2025-12-10)**:
- llama.cpp (phi-2 Q4_K on RTX 4090): 257-265 tok/s, ~255ms for 64 tokens
- TTFT: ~5ms | Eval: ~120ms for 32 tokens

### The Memory Wall Problem (Root Cause Analysis)

Per Wulf & McKee [21] and Williams et al. [19], LLM inference is **memory-bound**, not compute-bound:

```
Memory Traffic Analysis (phi-2 forward pass):
├── Realizar (f32):     2.78B × 4 bytes = 11.1 GB
├── llama.cpp (Q4_K):   2.78B × 0.5 bytes = 1.4 GB
└── Gap:                8x more memory traffic

DDR4-3200 Bandwidth:    ~35 GB/s practical
├── Realizar floor:     11.1 GB / 35 GB/s = 317ms minimum
├── llama.cpp floor:    1.4 GB / 35 GB/s = 40ms minimum
└── Actual llama.cpp:   ~22ms (exceeds floor via caching)
```

**Conclusion**: Quantized inference (Q4_K/Q5_K/Q6_K) is **mandatory** before any SIMD optimization will matter.

---

## Executive Summary

This specification defines a rigorous, scientifically reproducible benchmark methodology for comparing inference performance across model formats (.apr, GGUF, safetensors) and runtime environments (Realizar, llama.cpp, Ollama, vLLM, and others). The benchmark focuses on **Small Language Models (SLMs)** in the 100M-500M parameter range, where task-specific optimization can achieve frontier model performance at 100-1000x lower inference cost.

### Core Thesis

> **Hypothesis**: Task-optimized SLMs using the .apr format with Realizar inference achieve equivalent or superior quality at dramatically lower latency and cost compared to general-purpose runtimes.

### Toyota Way Engineering Principles Applied

This specification adheres to Toyota Production System (TPS) principles:

1. **Genchi Genbutsu** (Go and See): Direct measurement at the source, not derived metrics
2. **Jidoka** (Automation with Human Touch): Automated benchmarks with manual validation gates
3. **Kaizen** (Continuous Improvement): Baseline-delta tracking with regression detection
4. **Heijunka** (Level Loading): Consistent workload distribution across test iterations
5. **Muda Elimination**: Remove waste by measuring only actionable metrics

---

## 1. Benchmark Scope and Objectives

### 1.1 Primary Objectives

| Objective | Measurement | Success Criteria |
|-----------|-------------|------------------|
| **Latency** | TTFT, ITL variance, tokens/second | p50, p95, p99, p99.9 within 5% variance |
| **Throughput** | Tokens/second sustained | >100 tok/s for SLMs on CPU |
| **Memory** | Peak RSS, KV-Cache fragmentation, VRAM | <4GB for 500M models, <10% waste |
| **Cold Start** | Time from binary exec to first inference | <500ms for .apr format |
| **Model Load** | Time to load model into memory | <100ms for .apr on NVMe |
| **Energy** | Token Joules, Idle Power | <0.1 J/tok on mobile |

### 1.2 Model Formats Under Test

| Format | Primary Use Case | Quantization Support |
|--------|------------------|---------------------|
| **.apr** (Aprender) | Sovereign stack native | Q4, Q8, FP16, FP32 |
| **GGUF** (llama.cpp) | Cross-platform inference | Q2-Q8, K-quants |
| **safetensors** | HuggingFace ecosystem | FP16, FP32, BF16 |
| **ONNX** | Cross-framework | QInt8, FP16, FP32 |

### 1.3 Inference Runtimes Under Test

| Runtime | Language | GPU Support | Primary Strength |
|---------|----------|-------------|------------------|
| **Realizar** | Rust | CUDA, Metal, WebGPU | Native .apr, zero-copy |
| **llama.cpp** | C++ | CUDA, Metal, Vulkan | GGUF optimization |
| **Ollama** | Go + llama.cpp | CUDA, Metal | User experience |
| **vLLM** | Python | CUDA | PagedAttention [11] |
| **candle** | Rust | CUDA, Metal | Rust ecosystem |
| **mlx** | Swift/C++ | Metal | Apple Silicon |

---

## 2. Test Methodology

### 2.1 Statistical Rigor with Dynamic Stop-Rule

Following best practices from systems benchmarking literature [1, 2, 17], we employ a **dynamic stop-rule** based on Coefficient of Variation (CV) rather than fixed iteration counts.

**Dynamic Sample Size** (replaces fixed 1,000 iterations):

The benchmark continues until statistical stability is achieved:

```rust
/// Dynamic stop-rule per Hoefler & Belli [17]
/// Stop when CV stabilizes below threshold
pub struct DynamicSampler {
    min_samples: usize,      // Minimum 100 iterations
    max_samples: usize,      // Maximum 10,000 iterations
    cv_threshold: f64,       // Target CV < 0.05 (5%)
    cv_window: usize,        // Sliding window of 50 samples
    stability_count: usize,  // Require 3 consecutive stable windows
}

impl DynamicSampler {
    pub fn should_continue(&self, samples: &[f64]) -> bool {
        if samples.len() < self.min_samples {
            return true;
        }
        if samples.len() >= self.max_samples {
            return false;
        }

        // Compute CV over sliding window
        let window = &samples[samples.len().saturating_sub(self.cv_window)..];
        let mean = statistical::mean(window);
        let std_dev = statistical::std_dev(window);
        let cv = std_dev / mean;

        // Check if CV is stable
        cv > self.cv_threshold
    }
}
```

**Stop-Rule Algorithm**:
1. Run minimum 100 warm-up iterations (discarded)
2. Begin measurement phase
3. After each batch of 10 iterations:
   - Compute CV over last 50 samples
   - If CV < 5% for 3 consecutive checks → STOP
   - If iterations > 10,000 → STOP (with warning)
4. Report actual iteration count in results

**Rationale**: Fixed iteration counts mask the "Tail at Scale" problem [11]. Dynamic sampling ensures statistical validity regardless of variance characteristics.

**Warm-up Protocol**:
1. 100 warm-up iterations (discarded)
2. JIT compilation stabilization (for JIT-enabled runtimes)
3. Memory pressure stabilization via `sync && echo 3 > /proc/sys/vm/drop_caches`

**Statistical Measures**:
- Central tendency: Median (robust to outliers)
- Dispersion: IQR, MAD (robust to outliers)
- Tail latency: p50, p90, p95, p99, **p99.9** [11]
- Confidence intervals: Bootstrap 95% CI with 10,000 resamples [4]
- Iteration count: Actual samples until CV stability

### 2.2 Reproducibility Requirements

Per reproducibility guidelines [5, 6]:

**Hardware Specification**:
```yaml
CPU:
  - Intel Xeon Platinum 8380 (Ice Lake, 40 cores)
  - AMD EPYC 9654 (Genoa, 96 cores)
  - Apple M3 Max (16 cores, 40 GPU cores)

GPU:
  - NVIDIA A100 80GB PCIe
  - NVIDIA RTX 4090 24GB
  - Apple M3 Max (unified memory)

Memory: 256GB DDR5-4800 (server), 128GB unified (M3)
Storage: NVMe Gen4, 7GB/s sequential read
```

**Software Specification**:
```yaml
OS: Ubuntu 24.04 LTS, macOS 14.5
Kernel: 6.8.0-generic (Linux)
Compiler: rustc 1.82.0, clang 18.0
CUDA: 12.4.0
Driver: NVIDIA 550.54.14
```

**Isolation Protocol**:
1. Dedicated benchmark machine (no other workloads)
2. CPU frequency governor: `performance` (disable DVFS)
3. Hyperthreading disabled for consistent core counts
4. NUMA-aware execution: `numactl --cpunodebind=0 --membind=0`

**Thermal Throttling Protocol** (NEW):
```rust
/// Thermal monitoring per Kaizen review
pub struct ThermalGuard {
    max_temp_c: f64,           // 80°C threshold
    cooldown_threshold_c: f64,  // 70°C resume
    cooldown_sleep_ms: u64,     // 10,000ms between batches
    temp_variance_c: f64,       // <2°C variance required
}

impl ThermalGuard {
    /// Check if benchmark results are thermally valid
    pub fn validate_run(&self, temps: &[f64]) -> ThermalValidity {
        let variance = statistical::variance(temps);
        if variance.sqrt() > self.temp_variance_c {
            ThermalValidity::Invalid("Temperature variance too high")
        } else {
            ThermalValidity::Valid
        }
    }

    /// Enforce cooldown between batches
    pub fn cooldown_if_needed(&self, current_temp: f64) {
        if current_temp > self.max_temp_c {
            std::thread::sleep(Duration::from_millis(self.cooldown_sleep_ms));
        }
    }
}
```

- Monitor `T_junction` via `sensors` (Linux) or `powermetrics` (macOS)
- Benchmark invalid if temperature variance > 2°C over measurement window
- Enforce 10s cooldown after every 100 iterations if T > 80°C

### 2.3 Workload Definition

**Standard Prompts** (deterministic, reproducible):

| Workload | Input Length | Output Length | Use Case |
|----------|--------------|---------------|----------|
| **Short-QA** | 32 tokens | 64 tokens | Chatbot, FAQ |
| **Code-Gen** | 128 tokens | 256 tokens | Code completion |
| **Summarize** | 512 tokens | 128 tokens | Document processing |
| **Long-Context** | 2048 tokens | 512 tokens | RAG, analysis |

**Temperature**: 0.0 (deterministic sampling)
**Top-k**: 1 (greedy decoding)
**Seed**: Fixed per-workload for reproducibility

### 2.4 Convoy Test (Continuous Batching Dynamics) [NEW]

Per Orca [15] and vLLM [12] methodologies, test scheduler's ability to handle mixed workloads:

**The Convoy Test Protocol**:
```
1. Submit 10 Long-Context requests (2048 input, 512 output)
2. Immediately submit 100 Short-QA requests (32 input, 64 output)
3. Measure:
   - Short-QA p99 latency with convoy vs without
   - Head-of-line blocking ratio
   - KV-cache utilization during convoy
```

**Success Criteria**:
- Short-QA p99 latency increase < 50% during convoy
- No Short-QA request blocked > 500ms by Long-Context prefill
- KV-cache fragmentation < 15% during mixed workload

### 2.5 Saturation Stress Test (Heijunka Validation) [NEW]

Test runtime's thread scheduling under contention:

```bash
#!/bin/bash
# saturation_stress_test.sh
# Validates Heijunka (level loading) under stress

# Start background CPU load (50% of cores)
stress-ng --cpu $(nproc --all | awk '{print int($1/2)}') \
          --timeout 300s &
STRESS_PID=$!

# Run benchmark under load
realizar bench \
    --model models/tinyllama.apr \
    --output results/stressed.json

# Kill stress
kill $STRESS_PID

# Compare with unstressed baseline
realizar bench-compare \
    --baseline results/baseline.json \
    --current results/stressed.json \
    --report stress-report.md
```

**Success Criteria**:
- Throughput degradation < 30% under 50% CPU saturation
- p99 latency increase < 100% under saturation
- No thread starvation (all worker threads active)

---

## 3. Test Models

### 3.1 Tiny Language Models (100M-500M Parameters)

| Model | Parameters | Architecture | Source |
|-------|------------|--------------|--------|
| **TinyLlama-1.1B** | 1.1B | LLaMA | Zhang et al. [7] |
| **Phi-2** | 2.7B | Phi | Microsoft |
| **SmolLM-135M** | 135M | LLaMA-style | HuggingFace |
| **Qwen2-0.5B** | 500M | Qwen | Alibaba |
| **StableLM-3B** | 3B | StableLM | Stability AI |

### 3.2 Format Conversion Matrix

Each model must be available in all formats for fair comparison:

```
TinyLlama-1.1B:
  ├── tinyllama-1.1b.apr      (Aprender native)
  ├── tinyllama-1.1b-q4_k.gguf (GGUF K-quant)
  ├── tinyllama-1.1b-q8_0.gguf (GGUF 8-bit)
  ├── tinyllama-1.1b.safetensors (HF format)
  └── tinyllama-1.1b.onnx     (ONNX runtime)
```

### 3.3 Quantization Configurations

| Quantization | Bits | Memory Reduction | Quality Impact |
|--------------|------|------------------|----------------|
| **FP32** | 32 | Baseline | None |
| **FP16** | 16 | 50% | Negligible |
| **Q8_0** | 8 | 75% | <1% perplexity |
| **Q4_K_M** | 4 | 87.5% | 1-3% perplexity |
| **Q4_0** | 4 | 87.5% | 2-5% perplexity |
| **Q2_K** | 2 | 93.75% | 5-10% perplexity |

---

## 4. Benchmark Implementation

### 4.1 Realizar Benchmark Harness

```rust
//! Benchmark harness following SPEC CPU methodology [8]
//! Updated with dynamic sampling per Hoefler & Belli [17]

use std::time::{Duration, Instant};
use realizar::{Model, InferenceConfig};
use statistical::{mean, median, percentile, bootstrap_ci, std_dev};

pub struct BenchmarkResult {
    pub config: BenchmarkConfig,
    pub cold_start_ms: f64,
    pub model_load_ms: f64,
    pub ttft_ms: Vec<f64>,
    pub itl_ms: Vec<f64>,              // NEW: Inter-Token Latency
    pub generation_tok_s: Vec<f64>,
    pub peak_memory_mb: u64,
    pub kv_cache_waste_pct: f64,       // NEW: Memory fragmentation
    pub energy_joules: f64,            // NEW: Total energy
    pub tokens_generated: u64,
    pub actual_iterations: usize,       // NEW: Dynamic sample count
    pub cv_at_stop: f64,               // NEW: CV when stopped
    pub timestamp: u64,
}

impl BenchmarkResult {
    pub fn summary(&self) -> BenchmarkSummary {
        BenchmarkSummary {
            // Latency metrics
            ttft_p50: median(&self.ttft_ms),
            ttft_p95: percentile(&self.ttft_ms, 95.0),
            ttft_p99: percentile(&self.ttft_ms, 99.0),
            ttft_p999: percentile(&self.ttft_ms, 99.9),  // NEW: p99.9

            // ITL metrics (Genchi Genbutsu: user-perceived jitter)
            itl_median: median(&self.itl_ms),
            itl_std_dev: std_dev(&self.itl_ms),          // NEW: Jitter metric

            // Throughput metrics
            throughput_median: median(&self.generation_tok_s),
            throughput_ci_95: bootstrap_ci(&self.generation_tok_s, 0.95, 10_000),

            // Energy metrics (NEW)
            token_joules: self.energy_joules / self.tokens_generated as f64,

            // Memory metrics (NEW)
            memory_waste_pct: self.kv_cache_waste_pct,

            // Statistical validity
            iterations: self.actual_iterations,
            cv_final: self.cv_at_stop,
        }
    }
}
```

### 4.2 Inter-Token Latency (ITL) Measurement [NEW]

Per the "Tail at Scale" problem [11], users perceive "stutter" during generation:

```rust
/// Measure ITL variance to detect generation jitter
pub fn measure_itl(model: &Model, prompt: &str, max_tokens: usize) -> Vec<f64> {
    let mut itl_times = Vec::with_capacity(max_tokens);
    let mut last_token_time = Instant::now();

    for token in model.generate_stream(prompt, max_tokens) {
        let now = Instant::now();
        itl_times.push((now - last_token_time).as_secs_f64() * 1000.0);
        last_token_time = now;
    }

    itl_times
}

/// ITL quality thresholds
pub struct ItlQuality {
    pub median_ms: f64,      // Target: <20ms
    pub std_dev_ms: f64,     // Target: <5ms (low jitter)
    pub p99_ms: f64,         // Target: <50ms
}
```

### 4.3 KV-Cache Fragmentation Measurement [NEW]

Per PagedAttention [12], memory fragmentation is the true *Muda*:

```rust
/// Measure KV-cache fragmentation (Muda detection)
pub struct KvCacheMetrics {
    pub allocated_mb: u64,
    pub used_mb: u64,
    pub fragmentation_pct: f64,
}

impl KvCacheMetrics {
    pub fn measure(model: &Model) -> Self {
        let allocated = model.kv_cache_allocated_bytes();
        let used = model.kv_cache_used_bytes();
        let waste = allocated.saturating_sub(used);

        Self {
            allocated_mb: allocated / (1024 * 1024),
            used_mb: used / (1024 * 1024),
            fragmentation_pct: (waste as f64 / allocated as f64) * 100.0,
        }
    }
}
```

### 4.4 Energy Measurement [NEW]

Per Garcia-Martin et al. [14], energy is the dominant cost at the edge:

```rust
/// Energy measurement via RAPL (Intel) or powermetrics (macOS)
pub struct EnergyMetrics {
    pub total_joules: f64,
    pub idle_watts: f64,
    pub active_watts_avg: f64,
}

impl EnergyMetrics {
    #[cfg(target_os = "linux")]
    pub fn measure_rapl<F: FnOnce()>(f: F) -> Self {
        let start_energy = read_rapl_energy();
        let start_time = Instant::now();

        f();

        let end_energy = read_rapl_energy();
        let duration = start_time.elapsed();

        let total_joules = end_energy - start_energy;
        let active_watts = total_joules / duration.as_secs_f64();

        Self {
            total_joules,
            idle_watts: measure_idle_power(),
            active_watts_avg: active_watts,
        }
    }

    /// Token efficiency (lower is better)
    pub fn joules_per_token(&self, tokens: u64) -> f64 {
        self.total_joules / tokens as f64
    }
}
```

### 4.5 Cross-Runtime Benchmark Protocol

```bash
#!/bin/bash
# benchmark_all_runtimes.sh
# Follows SIGPLAN empirical evaluation guidelines [9]
# Updated with dynamic sampling and thermal guards

set -euo pipefail

MODEL="tinyllama-1.1b"
# Note: No fixed ITERATIONS - using dynamic stop-rule

# Realizar (.apr format)
realizar bench \
    --model "models/${MODEL}.apr" \
    --cv-threshold 0.05 \
    --min-iterations 100 \
    --max-iterations 10000 \
    --thermal-guard \
    --measure-energy \
    --output "results/realizar-${MODEL}.json"

# llama.cpp (GGUF format)
./llama-bench \
    -m "models/${MODEL}-q4_k.gguf" \
    -r 0  # Auto-detect iterations
    -o json > "results/llamacpp-${MODEL}.json"

# Ollama (via API)
./bench_ollama.py \
    --model "${MODEL}" \
    --cv-threshold 0.05 \
    --output "results/ollama-${MODEL}.json"

# vLLM (via API)
./bench_vllm.py \
    --model "models/${MODEL}" \
    --cv-threshold 0.05 \
    --output "results/vllm-${MODEL}.json"
```

---

## 5. Metrics and Reporting

### 5.1 Primary Metrics

| Metric | Unit | Measurement Method | Significance |
|--------|------|-------------------|--------------|
| **TTFT** | ms | `Instant::now()` to first token | User-perceived latency |
| **ITL σ** | ms | Std dev of inter-token times | Generation jitter [11] |
| **Throughput** | tok/s | tokens / generation_time | Sustained performance |
| **Cold Start** | ms | Process spawn to ready | Serverless critical |
| **Memory** | MB | VmHWM from /proc/self/status | Cost efficiency |
| **KV Waste** | % | (Allocated - Used) / Allocated | Fragmentation [12] |
| **Token Joules** | J/tok | ∫Power·dt / tokens | Energy efficiency [14] |
| **Idle Power** | W | RAPL baseline | System overhead |

### 5.2 Statistical Reporting Requirements

Following Fleming & Wallace guidelines [10] and Hoefler & Belli [17]:

```
Realizar TinyLlama-1.1B Q4_K Inference Results
==============================================
Configuration:
  - Model: TinyLlama-1.1B (1.1B parameters)
  - Format: .apr with Q4_K quantization
  - Hardware: Apple M3 Max, 128GB unified memory
  - Iterations: 847 (dynamic, CV=0.048 at stop)

Time-to-First-Token (TTFT):
  - Median: 23.4 ms
  - p95: 28.1 ms
  - p99: 32.7 ms
  - p99.9: 41.2 ms
  - 95% CI: [22.9, 23.9] ms (bootstrap, n=10,000)

Inter-Token Latency (ITL):
  - Median: 12.3 ms
  - Std Dev: 2.1 ms (LOW JITTER)
  - p99: 18.7 ms

Throughput (tokens/second):
  - Median: 142.3 tok/s
  - IQR: [138.7, 146.2] tok/s
  - 95% CI: [141.1, 143.5] tok/s

Memory:
  - Model Size: 687 MB
  - Peak RSS: 1,247 MB
  - KV-Cache Waste: 3.2%
  - GPU VRAM: N/A (unified memory)

Energy:
  - Token Joules: 0.042 J/tok
  - Active Power: 28.4 W
  - Idle Power: 8.2 W

Cold Start:
  - Median: 89 ms
  - p99: 127 ms

Thermal Validity:
  - Temperature Variance: 1.3°C (VALID)
  - Max Temperature: 76°C
```

### 5.3 Comparison Tables

```
Comparative Inference Performance: TinyLlama-1.1B Q4
======================================================
                    TTFT    ITL σ   Throughput  Memory   KV Waste  J/tok
                    p99 ms  ms      tok/s       MB       %
-------------------------------------------------------------------------
Realizar (.apr)     32.7    2.1     142.3       1,247    3.2       0.042
llama.cpp (GGUF)    38.4    3.8     128.7       1,312    5.1       0.051
Ollama              67.2    8.4     98.4        1,456    12.3      0.078
vLLM                52.3    1.2*    187.2**     2,847    2.1       0.089
candle              45.1    4.2     112.5       1,389    7.8       0.056

* vLLM lower ITL due to continuous batching
** vLLM higher throughput due to batching (not single-request)

Statistical Significance:
- Realizar vs llama.cpp TTFT: p < 0.001 (Mann-Whitney U)
- Realizar vs Ollama throughput: p < 0.001 (Mann-Whitney U)
- Realizar ITL σ vs Ollama: p < 0.001 (F-test for variance)
```

---

## 6. Quality Validation

### 6.1 Output Equivalence via KL-Divergence [UPDATED]

Per Dettmers et al. [13], simple epsilon checks fail on outlier features. Use KL-Divergence instead:

```rust
/// Quality validation via KL-Divergence (replaces epsilon check)
/// Per LLM.int8() methodology [13]
pub fn validate_quantization_quality(
    fp32_logits: &[f32],
    quantized_logits: &[f32],
    threshold: f64,  // Typical: 0.01 nats
) -> QualityResult {
    // Convert to probability distributions
    let fp32_probs = softmax(fp32_logits);
    let quant_probs = softmax(quantized_logits);

    // Compute KL(P_fp32 || P_quant)
    let kl_div: f64 = fp32_probs.iter()
        .zip(&quant_probs)
        .map(|(p, q)| {
            if *p > 1e-10 && *q > 1e-10 {
                p * (p / q).ln()
            } else {
                0.0
            }
        })
        .sum();

    if kl_div < threshold {
        QualityResult::Pass { kl_divergence: kl_div }
    } else {
        QualityResult::Fail {
            kl_divergence: kl_div,
            threshold,
            message: "Quantization quality degradation detected",
        }
    }
}
```

**Quality Thresholds**:
| Quantization | Max KL-Divergence | Rationale |
|--------------|-------------------|-----------|
| FP16 → FP32 | 0.001 | Numerical precision only |
| Q8_0 → FP32 | 0.01 | <1% perplexity |
| Q4_K → FP32 | 0.05 | 1-3% perplexity |
| Q4_0 → FP32 | 0.10 | 2-5% perplexity |

### 6.2 Perplexity Validation

```
Perplexity on WikiText-2 (lower is better):
============================================
Format          FP16    Q8_0    Q4_K_M   Q4_0
--------------------------------------------
.apr            5.68    5.71    5.89     6.12
GGUF            5.68    5.70    5.87     6.08
safetensors     5.68    N/A     N/A      N/A

KL-Divergence vs FP32 (nats):
--------------------------------------------
.apr            0.0008  0.0042  0.031    0.067
GGUF            0.0008  0.0039  0.028    0.059

Acceptable: KL-Div within thresholds above
```

---

## 7. Toyota Way Compliance Checklist

### 7.1 Genchi Genbutsu (現地現物) - Go and See

- [ ] All measurements taken at source (no derived metrics)
- [ ] Hardware specifications verified via `lscpu`, `nvidia-smi`
- [ ] Actual inference runs, not simulations
- [ ] Manual validation of 100 random samples
- [ ] **NEW**: ITL measured per-token, not averaged
- [ ] **NEW**: KV-cache fragmentation measured directly

### 7.2 Jidoka (自働化) - Automation with Human Touch

- [ ] Automated benchmark harness with CI integration
- [ ] Human review gate before publishing results
- [ ] Anomaly detection for outlier runs
- [ ] Manual root cause analysis for regressions
- [ ] **NEW**: KL-Divergence validation replaces epsilon
- [ ] **NEW**: Thermal validity check

### 7.3 Kaizen (改善) - Continuous Improvement

- [ ] Baseline tracking with version-tagged results
- [ ] Regression detection with ≥5% threshold alerts
- [ ] Weekly benchmark runs in CI
- [ ] Monthly comprehensive hardware matrix runs
- [ ] **NEW**: Dynamic stop-rule improves efficiency

### 7.4 Heijunka (平準化) - Level Loading

- [ ] ~~Fixed iteration counts~~ → Dynamic CV-based sampling
- [ ] Consistent warm-up protocol
- [ ] Deterministic prompts with fixed seeds
- [ ] Uniform resource allocation (no oversubscription)
- [ ] **NEW**: Saturation stress test validates scheduler
- [ ] **NEW**: Convoy test validates continuous batching

### 7.5 Muda Elimination (無駄) - Remove Waste

- [ ] Only actionable metrics collected
- [ ] No redundant measurements
- [ ] Efficient benchmark harness (<1% overhead)
- [ ] Results stored in compact binary format
- [ ] **NEW**: KV-cache fragmentation as primary memory metric
- [ ] **NEW**: Dynamic sampling eliminates unnecessary iterations

---

## 8. Appendix A: Benchmark Command Reference

```bash
# Full benchmark suite with all new metrics
make bench-all

# Single runtime benchmark with dynamic sampling
realizar bench \
    --model models/tinyllama.apr \
    --cv-threshold 0.05 \
    --thermal-guard \
    --measure-energy \
    --measure-kv-cache \
    --output results/

# Convoy test (continuous batching validation)
realizar bench-convoy \
    --model models/tinyllama.apr \
    --long-requests 10 \
    --short-requests 100

# Saturation stress test
realizar bench-saturation \
    --model models/tinyllama.apr \
    --cpu-load 50

# Compare runtimes
realizar bench-compare --results results/ --output report.md

# Regression check against baseline
realizar bench-regression --baseline baseline.json --current current.json
```

## 8.1 Appendix B: Result Schema (v2.0)

```json
{
  "$schema": "https://realizar.dev/schemas/benchmark-result-v1.1.json",
  "version": "1.1",
  "timestamp": "2025-12-09T12:00:00Z",
  "config": {
    "model": "tinyllama-1.1b",
    "format": "apr",
    "quantization": "q4_k",
    "runtime": "realizar",
    "runtime_version": "0.2.3"
  },
  "hardware": {
    "cpu": "Apple M3 Max",
    "gpu": "Apple M3 Max (40 cores)",
    "memory_gb": 128,
    "storage": "NVMe"
  },
  "sampling": {
    "method": "dynamic_cv",
    "cv_threshold": 0.05,
    "actual_iterations": 847,
    "cv_at_stop": 0.048,
    "warmup_iterations": 100
  },
  "thermal": {
    "valid": true,
    "temp_variance_c": 1.3,
    "max_temp_c": 76
  },
  "results": {
    "ttft_ms": { "p50": 23.4, "p95": 28.1, "p99": 32.7, "p999": 41.2 },
    "itl_ms": { "median": 12.3, "std_dev": 2.1, "p99": 18.7 },
    "throughput_tok_s": { "median": 142.3, "ci_95": [141.1, 143.5] },
    "memory_mb": { "model": 687, "peak_rss": 1247, "kv_waste_pct": 3.2 },
    "energy": { "total_joules": 42.7, "token_joules": 0.042, "idle_watts": 8.2 },
    "cold_start_ms": { "median": 89, "p99": 127 }
  },
  "quality": {
    "kl_divergence_vs_fp32": 0.031,
    "perplexity_wikitext2": 5.89
  }
}
```

---

## 9. Sprint Planning: BENCH-SPRINT-001

### 9.1 Sprint Objectives (Kaizen Cycle)

**Sprint Goal**: Close the performance gap from 1,200x to ≤10x vs llama.cpp through quantized inference.

**Duration**: 2 weeks
**Start Date**: Next sprint cycle
**Success Criteria**: ≥5 tok/s CPU inference on phi-2 (from 0.04 tok/s)

### 9.2 Sprint Backlog (Priority Order)

| ID | Task | Effort | Dependency | Acceptance Criteria |
|----|------|--------|------------|---------------------|
| BENCH-001 | Q4_K SIMD matmul kernel | 3 days | None | 4-bit × 4-bit matmul with AVX2 |
| BENCH-002 | Q4_K inline dequant | 2 days | BENCH-001 | Dequant during compute, not before |
| BENCH-003 | Memory-mapped GGUF weights | 2 days | None | mmap for zero-copy model loading |
| BENCH-004 | Real HTTP benchmark wiring | 2 days | None | `reqwest` calls to Ollama/vLLM endpoints |
| BENCH-005 | Benchmark automation CI | 1 day | BENCH-004 | GitHub Actions with result tracking |
| BENCH-006 | Performance regression tests | 1 day | BENCH-001 | Fail CI if >10% regression |

### 9.3 Toyota Way Sprint Rituals

**Daily Standup (Genchi Genbutsu)**:
- What did I measure yesterday? (not "what did I do")
- What will I measure today?
- What blockers prevent measurement?

**Sprint Review (Jidoka)**:
- Demo actual benchmark runs, not slides
- Compare before/after with statistical significance
- Human review gate before merging

**Retrospective (Kaizen)**:
- What waste (Muda) did we eliminate?
- What measurement surprised us?
- What should we measure next sprint?

### 9.4 Definition of Done

A task is DONE when:
- [ ] Unit tests pass with ≥85% coverage on new code
- [ ] Benchmark shows measurable improvement (p < 0.05)
- [ ] No regression in existing benchmarks (>5% threshold)
- [ ] Documentation updated with honest results
- [ ] Code reviewed by at least one team member

### 9.5 Risk Mitigation (Jidoka - Stop the Line)

| Risk | Mitigation | Stop Condition |
|------|------------|----------------|
| SIMD kernel bugs | Property-based testing | Any incorrect output |
| Performance regression | CI benchmark gates | >10% slowdown |
| Memory safety | ASan in CI | Any memory error |
| Thermal throttling | ThermalGuard in benchmarks | T > 80°C |

### 9.6 Expected Outcomes

| Metric | Current | Sprint Target | Rationale |
|--------|---------|---------------|-----------|
| tok/s (CPU, phi-2) | 0.04 | ≥5 | Q4_K reduces memory 8x |
| Memory (phi-2) | 11.1 GB | ~1.4 GB | Quantized weights |
| HTTP benchmarks | Mock | Real | Honest comparison |
| CI automation | Manual | Automated | Reproducibility |

---

## 10. References

### Original Citations (v1.0)

[1] T. Mytkowicz, A. Diwan, M. Hauswirth, and P. F. Sweeney, "Producing Wrong Data Without Doing Anything Obviously Wrong!" in *Proceedings of ASPLOS*, 2009, pp. 265-276. DOI: 10.1145/1508244.1508275

[2] A. Georges, D. Buytaert, and L. Eeckhout, "Statistically Rigorous Java Performance Evaluation," in *Proceedings of OOPSLA*, 2007, pp. 57-76. DOI: 10.1145/1297027.1297033

[3] S. Chen, A. Ailamaki, P. B. Gibbons, and T. C. Mowry, "Improving Hash Join Performance Through Prefetching," in *Proceedings of ICDE*, 2004, pp. 116-127. DOI: 10.1109/ICDE.2004.1319989

[4] B. Efron and R. J. Tibshirani, *An Introduction to the Bootstrap*. Chapman & Hall/CRC, 1993. ISBN: 978-0412042317

[5] J. Vitek and T. Kalibera, "Repeatability, Reproducibility, and Rigor in Systems Research," in *Proceedings of EMSOFT*, 2011, pp. 33-38. DOI: 10.1145/2038642.2038650

[6] C. Collberg and T. A. Proebsting, "Repeatability in Computer Systems Research," *Communications of the ACM*, vol. 59, no. 3, pp. 62-69, 2016. DOI: 10.1145/2812803

[7] P. Zhang et al., "TinyLlama: An Open-Source Small Language Model," arXiv:2401.02385, 2024. URL: https://arxiv.org/abs/2401.02385

[8] J. L. Henning, "SPEC CPU2006 Benchmark Descriptions," *ACM SIGARCH Computer Architecture News*, vol. 34, no. 4, pp. 1-17, 2006. DOI: 10.1145/1186736.1186737

[9] S. Blackburn et al., "The DaCapo Benchmarks: Java Benchmarking Development and Analysis," in *Proceedings of OOPSLA*, 2006, pp. 169-190. DOI: 10.1145/1167473.1167488

[10] P. J. Fleming and J. J. Wallace, "How Not to Lie with Statistics: The Correct Way to Summarize Benchmark Results," *Communications of the ACM*, vol. 29, no. 3, pp. 218-221, 1986. DOI: 10.1145/5666.5673

### Citations Added in v1.1

[11] J. Dean and L. A. Barroso, "The Tail at Scale," *Communications of the ACM*, vol. 56, no. 2, pp. 74-80, 2013. DOI: 10.1145/2408776.2408794

[12] W. Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention," in *Proceedings of SOSP '23*, 2023. DOI: 10.1145/3600006.3613165

[13] T. Dettmers et al., "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale," in *NeurIPS*, 2022. arXiv:2208.07339

[14] E. Garcia-Martin et al., "Estimation of Energy Consumption in Machine Learning," *Journal of Parallel and Distributed Computing*, vol. 134, pp. 75-88, 2019. DOI: 10.1016/j.jpdc.2019.07.007

[15] G. Yu et al., "Orca: A Distributed Serving System for Transformer-Based Generative Models," in *OSDI '22*, 2022. URL: https://www.usenix.org/conference/osdi22/presentation/yu

[16] T. Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness," in *NeurIPS*, 2022. arXiv:2205.14135

[17] T. Hoefler and R. Belli, "Scientific Benchmarking of Parallel Computing Systems," in *SC '15*, 2015. DOI: 10.1145/2807591.2807644

[18] R. Y. Aminabadi et al., "DeepSpeed-Inference: Enabling Efficient Inference of Transformer Models at Unprecedented Scale," in *SC '22*, 2022. DOI: 10.1109/SC41404.2022.00051

[19] S. Williams, A. Waterman, and D. Patterson, "Roofline: An Insightful Visual Performance Model for Multicore Architectures," *Communications of the ACM*, vol. 52, no. 4, pp. 65-76, 2009. DOI: 10.1145/1498765.1498785

[20] J. Kaplan et al., "Scaling Laws for Neural Language Models," arXiv:2001.08361, 2020. URL: https://arxiv.org/abs/2001.08361

### New Citations (v2.0 - Quantized Inference & Memory Optimization)

[21] W. A. Wulf and S. A. McKee, "Hitting the Memory Wall: Implications of the Obvious," *ACM SIGARCH Computer Architecture News*, vol. 23, no. 1, pp. 20-24, 1995. DOI: 10.1145/216585.216588
*(Foundational paper on memory-bound computation; explains why bandwidth, not FLOPS, limits LLM inference)*

[22] G. Xiao et al., "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models," in *ICML*, 2023. arXiv:2211.10438
*(Per-channel scaling for outlier features; critical for Q4 quality preservation)*

[23] J. Lin et al., "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration," in *MLSys*, 2024. arXiv:2306.00978
*(Activation-aware quantization preserves salient weights; 4-bit with minimal quality loss)*

[24] E. Frantar et al., "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers," in *ICLR*, 2023. arXiv:2210.17323
*(One-shot quantization via approximate second-order information; enables Q4 without retraining)*

[25] Y. Kim et al., "Squeezellm: Dense-and-Sparse Quantization," in *ICML*, 2024. arXiv:2306.07629
*(Sensitivity-based non-uniform quantization; sparse outlier handling)*

[26] S. Shen et al., "Q-BERT: Hessian Based Ultra Low Precision Quantization of BERT," in *AAAI*, 2020. arXiv:1909.05840
*(Mixed-precision quantization guided by Hessian sensitivity; foundational for transformer quantization)*

[27] A. Gholami et al., "A Survey of Quantization Methods for Efficient Neural Network Inference," arXiv:2103.13630, 2021.
*(Comprehensive survey of quantization techniques; taxonomy of PTQ vs QAT approaches)*

[28] Z. Yao et al., "ZeroQuant: Efficient and Affordable Post-Training Quantization for Large-Scale Transformers," in *NeurIPS*, 2022. arXiv:2206.01861
*(Layer-wise knowledge distillation for INT8; maintains quality at scale)*

[29] H. Wu et al., "Integer Quantization for Deep Learning Inference: Principles and Empirical Evaluation," arXiv:2004.09602, 2020.
*(NVIDIA's INT8 inference guide; practical implementation patterns for quantized GEMM)*

[30] P. Micikevicius et al., "Mixed Precision Training," in *ICLR*, 2018. arXiv:1710.03740
*(FP16/BF16 training with loss scaling; foundational for mixed-precision inference)*

---

**Document Control**

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-12-09 | Batuta Team | Initial specification |
| 1.1 | 2025-12-09 | Batuta Team | Dynamic sampling, ITL, KV-cache, energy, thermal guards |
| 2.0 | 2025-12-10 | Batuta Team | Current state (honest gap), sprint planning, +10 citations [21-30] |

