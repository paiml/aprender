# Deterministic and Reproducible Benchmarking Specification

**Version:** 1.0.1
**Status:** Draft
**Authors:** Pragmatic AI Labs
**Date:** 2024-12-11

## Abstract

This specification defines a rigorous methodology for deterministic, reproducible benchmarking of ML inference systems using Rust's Criterion.rs framework. It addresses systematic failures observed in ad-hoc benchmarking approaches and establishes standards aligned with Toyota Production System (TPS) principles as codified by Liker [9] and peer-reviewed benchmarking methodology.

---

## 1. Problem Statement

### 1.1 Observed Failures (Root Cause Analysis)

Using the Toyota Way's "5 Whys" methodology [9], we analyzed benchmark failures:

#### Failure 1: Ollama Throughput Reported as 0 tok/s

| Why Level | Analysis |
|-----------|----------|
| Why 1 | `throughput_tps` was 0.0 in benchmark results |
| Why 2 | `tokens_generated` field was 0 |
| Why 3 | Ollama's `eval_count` was not being parsed |
| Why 4 | Model name "phi" did not exist; should be "phi2:2.7b" |
| Why 5 | **No validation layer** between configuration and external system state |

**Root Cause:** Absence of Poka-yoke (error-proofing) mechanisms to validate external system configuration before benchmark execution.

#### Failure 2: Non-Deterministic Results Across Runs

| Why Level | Analysis |
|-----------|----------|
| Why 1 | Benchmark results varied significantly between runs |
| Why 2 | System state was not controlled |
| Why 3 | No warmup standardization; JIT compilation effects |
| Why 4 | Temperature parameter allowed randomness |
| Why 5 | **No determinism protocol** enforcing reproducible conditions |

**Root Cause:** Violation of scientific benchmarking principles requiring controlled, reproducible conditions [1].

#### Failure 3: JSON Parsing Errors with Control Characters

| Why Level | Analysis |
|-----------|----------|
| Why 1 | jq failed to parse llama.cpp response |
| Why 2 | Response contained U+0000-U+001F control characters |
| Why 3 | Generated text included escape sequences |
| Why 4 | No output sanitization layer |
| Why 5 | **Untrusted external output** fed directly to parsers |

**Root Cause:** Missing defensive programming at system boundaries [2].

---

## 2. Toyota Production System Principles Applied

### 2.1 Jidoka (Autonomation with Human Touch)

**Principle:** Build quality in at every step; stop immediately when problems occur.

**Application:**
```rust
/// Benchmark execution with built-in quality checks (Jidoka)
pub struct BenchmarkExecutor {
    /// Pre-flight checks must pass before any measurement
    preflight_checks: Vec<Box<dyn PreflightCheck>>,
    /// Anomaly detection halts execution
    anomaly_detector: AnomalyDetector,
}

impl BenchmarkExecutor {
    pub fn execute(&self) -> Result<BenchmarkResult, BenchmarkError> {
        // Jidoka: Stop the line if any check fails
        for check in &self.preflight_checks {
            check.validate()?;  // Fail fast, no partial results
        }
        // ... proceed only if all checks pass
    }
}
```

### 2.2 Poka-yoke (Error-Proofing)

**Principle:** Design systems that make errors impossible or immediately obvious.

**Application:**
```rust
/// Type-safe model identifier that validates existence at construction
pub struct ValidatedModelId {
    name: String,
    backend: Backend,
}

impl ValidatedModelId {
    /// Poka-yoke: Cannot create invalid model ID
    pub fn new(name: &str, backend: Backend) -> Result<Self, ValidationError> {
        // Verify model exists in backend before accepting
        backend.verify_model_exists(name)?;
        Ok(Self { name: name.to_string(), backend })
    }
}
```

### 2.3 Genchi Genbutsu (Go and See)

**Principle:** Base decisions on firsthand verification, not assumptions.

**Application:**
```rust
/// Health check that verifies actual system state (Genchi Genbutsu)
pub fn verify_server_ready(url: &str) -> Result<ServerState, HealthError> {
    // Don't assume server is ready; verify with actual request
    let response = client.get(&format!("{}/health", url)).send()?;

    // Verify response content, not just status code
    let health: HealthResponse = response.json()?;

    // Verify all subsystems, not just top-level
    if !health.model_loaded {
        return Err(HealthError::ModelNotLoaded);
    }
    if !health.gpu_available && config.requires_gpu {
        return Err(HealthError::GpuNotAvailable);
    }

    Ok(ServerState::Ready(health))
}
```

### 2.4 Kaizen (Continuous Improvement)

**Principle:** Systematic, incremental improvement based on measurement.

**Application:**
- Track benchmark stability metrics over time
- CI/CD integration for regression detection
- Historical trend analysis with statistical significance testing

### 2.5 Heijunka (Leveling)

**Principle:** Smooth out variations for predictable output.

**Application:**
- Warmup iterations to reach steady state
- CPU frequency governor pinning
- Memory pre-allocation to avoid allocation jitter

---

## 3. Statistical Methodology

### 3.1 Coefficient of Variation (CV) Based Stopping

Per Hoefler & Belli [1], dynamic stopping based on CV provides statistically rigorous sample sizes:

```rust
/// CV-based stopping criterion per Hoefler & Belli SC'15
pub struct CvStoppingCriterion {
    /// Minimum samples before CV check (prevents premature stopping)
    min_samples: usize,
    /// Maximum samples (bounded resource usage)
    max_samples: usize,
    /// Target CV threshold (e.g., 0.05 = 5%)
    cv_threshold: f64,
}

impl CvStoppingCriterion {
    pub fn should_stop(&self, samples: &[f64]) -> StopDecision {
        if samples.len() < self.min_samples {
            return StopDecision::Continue;
        }
        if samples.len() >= self.max_samples {
            return StopDecision::Stop(StopReason::MaxSamples);
        }

        let cv = self.calculate_cv(samples);
        if cv < self.cv_threshold {
            StopDecision::Stop(StopReason::CvConverged(cv))
        } else {
            StopDecision::Continue
        }
    }

    fn calculate_cv(&self, samples: &[f64]) -> f64 {
        let n = samples.len() as f64;
        let mean = samples.iter().sum::<f64>() / n;
        let variance = samples.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / (n - 1.0);
        let std_dev = variance.sqrt();
        std_dev / mean
    }
}
```

### 3.2 Warmup Protocol

Per Mytkowicz et al. [3] and Georges et al. [8], warmup is critical for JIT-compiled and cached systems. Georges et al. emphasize that performance evaluation must occur only after the system has reached this steady state to avoid misleading start-up noise and ensure statistical rigor.

```rust
/// Warmup protocol with steady-state detection
pub struct WarmupProtocol {
    /// Fixed warmup iterations
    fixed_iterations: usize,
    /// Optional: detect steady state dynamically
    steady_state_detection: Option<SteadyStateDetector>,
}

impl WarmupProtocol {
    pub fn execute<F: FnMut() -> Duration>(&self, mut workload: F) {
        // Phase 1: Fixed warmup (JIT compilation, cache priming)
        for _ in 0..self.fixed_iterations {
            let _ = workload();
        }

        // Phase 2: Optional steady-state detection
        if let Some(detector) = &self.steady_state_detection {
            let mut recent_times = VecDeque::with_capacity(10);
            while !detector.is_steady_state(&recent_times) {
                recent_times.push_back(workload());
                if recent_times.len() > 10 {
                    recent_times.pop_front();
                }
            }
        }
    }
}
```

### 3.3 Outlier Handling

Per Chen et al. [4], robust statistics handle measurement noise:

```rust
/// Outlier detection using Median Absolute Deviation (MAD)
pub fn detect_outliers_mad(samples: &[f64], k: f64) -> Vec<bool> {
    let median = percentile(samples, 50.0);
    let deviations: Vec<f64> = samples.iter()
        .map(|x| (x - median).abs())
        .collect();
    let mad = percentile(&deviations, 50.0);

    // k=3.0 is typical; corresponds to ~99.7% for normal distribution
    let threshold = k * mad * 1.4826; // 1.4826 scales MAD to std dev

    samples.iter()
        .map(|x| (x - median).abs() > threshold)
        .collect()
}
```

---

## 4. Determinism Requirements

### 4.1 Random Seed Control

Per Fleming & Wallace [5], deterministic benchmarks require seed control:

```rust
/// Deterministic inference configuration
pub struct DeterministicInferenceConfig {
    /// Temperature = 0.0 disables sampling randomness
    temperature: f64,
    /// Fixed seed for any remaining randomness
    seed: u64,
    /// Top-k = 1 forces greedy decoding
    top_k: usize,
}

impl Default for DeterministicInferenceConfig {
    fn default() -> Self {
        Self {
            temperature: 0.0,  // Greedy decoding
            seed: 42,          // Fixed, reproducible seed
            top_k: 1,          // Deterministic token selection
        }
    }
}
```

### 4.2 System State Requirements

Per Curtsinger & Berger [6], system state affects measurements:

| Requirement | Implementation | Rationale |
|-------------|----------------|-----------|
| CPU Frequency | Pin to maximum (performance governor) | Avoid DVFS variation |
| Memory | Pre-allocate, disable swap | Avoid allocation jitter |
| Process Priority | SCHED_FIFO or nice -20 | Minimize preemption |
| Thermal State | Monitor, abort if throttling | Avoid thermal variation |
| Background Processes | Minimize, document | Reduce interference |

### 4.3 Input Determinism

```rust
/// Canonical benchmark inputs (immutable, versioned)
pub mod canonical_inputs {
    /// Version: Changing inputs MUST increment this version
    pub const VERSION: &str = "1.0.0";

    /// Fixed prompt for latency benchmarks
    pub const LATENCY_PROMPT: &str =
        "Explain the concept of machine learning in one sentence.";

    /// Fixed token sequence for throughput benchmarks
    pub const THROUGHPUT_TOKENS: &[u32] = &[1, 2, 3, 4, 5, 6, 7, 8];

    /// Fixed max tokens for generation benchmarks
    pub const MAX_TOKENS: usize = 50;
}
```

---

## 5. Preflight Validation Protocol

### 5.1 Server Availability Checks

```rust
/// Preflight check: Verify server is available and responsive
pub struct ServerAvailabilityCheck {
    url: String,
    timeout: Duration,
    required_model: String,
}

impl PreflightCheck for ServerAvailabilityCheck {
    fn validate(&self) -> Result<(), PreflightError> {
        // 1. TCP connectivity
        let addr = self.url.parse::<SocketAddr>()
            .map_err(|_| PreflightError::InvalidUrl)?;
        TcpStream::connect_timeout(&addr, self.timeout)
            .map_err(|_| PreflightError::ServerUnreachable)?;

        // 2. HTTP health endpoint
        let health_url = format!("{}/health", self.url);
        let response = reqwest::blocking::get(&health_url)
            .map_err(|_| PreflightError::HealthCheckFailed)?;

        // 3. Model availability (Genchi Genbutsu - verify actual state)
        let models = self.list_models()?;
        if !models.contains(&self.required_model) {
            return Err(PreflightError::ModelNotFound {
                requested: self.required_model.clone(),
                available: models,
            });
        }

        Ok(())
    }
}
```

### 5.2 Response Schema Validation

```rust
/// Preflight check: Verify response schema matches expectations
pub struct ResponseSchemaCheck {
    sample_request: Request,
    expected_fields: Vec<String>,
}

impl PreflightCheck for ResponseSchemaCheck {
    fn validate(&self) -> Result<(), PreflightError> {
        let response = self.execute_sample_request()?;

        for field in &self.expected_fields {
            if response.get(field).is_none() {
                return Err(PreflightError::SchemaMismatch {
                    missing_field: field.clone(),
                    actual_response: response,
                });
            }
        }

        // Verify critical fields have expected types
        if let Some(eval_count) = response.get("eval_count") {
            if !eval_count.is_number() {
                return Err(PreflightError::FieldTypeMismatch {
                    field: "eval_count".to_string(),
                    expected: "number",
                    actual: eval_count.to_string(),
                });
            }
        }

        Ok(())
    }
}
```

---

## 6. Benchmark Result Schema

### 6.1 Required Metadata

Per Vitek & Kalibera [7], benchmark results must include complete metadata:

```rust
/// Complete benchmark result with required metadata
#[derive(Serialize, Deserialize)]
pub struct BenchmarkResult {
    // Identification
    pub benchmark_id: Uuid,
    pub timestamp: DateTime<Utc>,
    pub specification_version: String,

    // Environment (reproducibility requirement)
    pub environment: EnvironmentMetadata,

    // Configuration
    pub config: BenchmarkConfig,

    // Results
    pub measurements: Vec<Measurement>,
    pub statistics: Statistics,

    // Quality indicators
    pub quality: QualityMetrics,
}

#[derive(Serialize, Deserialize)]
pub struct EnvironmentMetadata {
    pub os: String,
    pub os_version: String,
    pub cpu_model: String,
    pub cpu_count: usize,
    pub memory_gb: usize,
    pub gpu_model: Option<String>,
    pub gpu_memory_gb: Option<usize>,
    pub rust_version: String,
    pub criterion_version: String,
    pub commit_hash: String,
}

#[derive(Serialize, Deserialize)]
pub struct QualityMetrics {
    pub cv_at_stop: f64,
    pub cv_converged: bool,
    pub outliers_detected: usize,
    pub outliers_excluded: usize,
    pub preflight_checks_passed: Vec<String>,
}
```

---

## 7. Implementation Checklist

### 7.1 Pre-Benchmark Checklist (Poka-yoke)

- [ ] All external servers verified reachable (TCP + HTTP)
- [ ] Required models verified present in each backend
- [ ] Sample request executed and response schema validated
- [ ] CPU frequency governor set to 'performance'
- [ ] Thermal throttling monitor active
- [ ] Memory pre-allocated for measurement storage
- [ ] Random seeds set to deterministic values
- [ ] Temperature set to 0.0 for deterministic output

### 7.2 During-Benchmark Checklist (Jidoka)

- [ ] Warmup protocol completed before measurement
- [ ] CV calculated after each sample
- [ ] Anomaly detection active
- [ ] Thermal throttling check between samples
- [ ] Response validation on each iteration

### 7.3 Post-Benchmark Checklist (Kaizen)

- [ ] Results include complete environment metadata
- [ ] CV convergence documented
- [ ] Outliers identified and documented
- [ ] Results persisted in versioned format
- [ ] Comparison with historical baseline (if available)

---

## 8. References

[1] T. Hoefler and R. Belli, "Scientific Benchmarking of Parallel Computing Systems: Twelve Ways to Tell the Masses When Reporting Performance Results," in *Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis (SC'15)*, Austin, TX, USA, 2015, pp. 1-12. DOI: 10.1145/2807591.2807644

[2] M. Howard and D. LeBlanc, *Writing Secure Code*, 2nd ed. Redmond, WA, USA: Microsoft Press, 2003. ISBN: 978-0735617223

[3] T. Mytkowicz, A. Diwan, M. Hauswirth, and P. F. Sweeney, "Producing Wrong Data Without Doing Anything Obviously Wrong!" in *Proceedings of the 14th International Conference on Architectural Support for Programming Languages and Operating Systems (ASPLOS XIV)*, Washington, DC, USA, 2009, pp. 265-276. DOI: 10.1145/1508244.1508275

[4] J. Chen, S. Bhardwaj, and W. Sun, "A Survey of Deterministic Replay Techniques for Debugging," *ACM Computing Surveys*, vol. 48, no. 2, pp. 1-37, 2015. DOI: 10.1145/2790832

[5] P. J. Fleming and J. J. Wallace, "How Not to Lie with Statistics: The Correct Way to Summarize Benchmark Results," *Communications of the ACM*, vol. 29, no. 3, pp. 218-221, 1986. DOI: 10.1145/5666.5673

[6] C. Curtsinger and E. D. Berger, "Stabilizer: Statistically Sound Performance Evaluation," in *Proceedings of the 18th International Conference on Architectural Support for Programming Languages and Operating Systems (ASPLOS XVIII)*, Houston, TX, USA, 2013, pp. 219-228. DOI: 10.1145/2451116.2451141

[7] J. Vitek and T. Kalibera, "Repeatability, Reproducibility, and Rigor in Systems Research," in *Proceedings of the Ninth ACM International Conference on Embedded Software (EMSOFT '11)*, Taipei, Taiwan, 2011, pp. 33-38. DOI: 10.1145/2038642.2038650

[8] A. Georges, D. Buytaert, and L. Eeckhout, "Statistically Rigorous Java Performance Evaluation," in *Proceedings of the 22nd Annual ACM SIGPLAN Conference on Object-Oriented Programming Systems, Languages and Applications (OOPSLA '07)*, Montreal, Quebec, Canada, 2007, pp. 57-76. DOI: 10.1145/1297027.1297033

[9] J. K. Liker, *The Toyota Way: 14 Management Principles from the World's Greatest Manufacturer*. New York, NY, USA: McGraw-Hill, 2004. ISBN: 978-0071392310

[10] S. Blackburn et al., "Wake Up and Smell the Coffee: Evaluation Methodology for the 21st Century," *Communications of the ACM*, vol. 51, no. 8, pp. 83-89, 2008. DOI: 10.1145/1378704.1378723

---

## Appendix A: Anti-Patterns to Avoid

| Anti-Pattern | Problem | Citation |
|--------------|---------|----------|
| Single-run benchmarks | No statistical validity | [1] |
| Ignoring warmup | JIT/cache effects pollute data | [3], [8] |
| Reporting only mean | Hides distribution shape | [5] |
| Ad-hoc stopping criteria | Arbitrary sample sizes | [1] |
| Undocumented environment | Non-reproducible | [7] |
| Assuming server availability | Silent failures | [2] |
| Trusting external output | Injection/parsing errors | [2] |
| Flawed evaluation metrics | Misleading performance claims | [10] |

---

## Appendix B: Toyota Way Principle Mapping

| TPS Principle | Application in Benchmarking |
|---------------|----------------------------|
| Jidoka | Stop on anomaly detection, fail-fast validation |
| Poka-yoke | Type-safe configurations, preflight checks |
| Genchi Genbutsu | Verify actual system state, don't assume |
| Kaizen | Track metrics over time, CI/CD integration |
| Heijunka | Warmup protocol, steady-state detection |
| Standardization | Canonical inputs, versioned schemas |
| Visual Management | Real-time CV display, progress indicators |

---

## Revision History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.1 | 2024-12-11 | Integrated additional peer-reviewed citations and explicit Toyota Way references |
| 1.0.0 | 2024-12-11 | Initial specification |