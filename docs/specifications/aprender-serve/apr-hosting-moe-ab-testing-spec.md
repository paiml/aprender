# APR Multi-Model Hosting with Hierarchical MOE and A/B Testing

**Version:** 1.0.0
**Status:** Draft
**Date:** 2025-11-27
**Target:** Realizar v0.3.0+
**Dependencies:** aprender v0.11.0, trueno v0.7.3

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Toyota Way Foundation](#2-toyota-way-foundation)
3. [Architecture Overview](#3-architecture-overview)
4. [Multi-Model Registry](#4-multi-model-registry)
5. [Hierarchical MOE](#5-hierarchical-moe)
6. [A/B Testing Framework](#6-ab-testing-framework)
7. [Traffic Routing](#7-traffic-routing)
8. [Observability](#8-observability)
9. [Implementation](#9-implementation)
10. [Scientific Foundation](#10-scientific-foundation)

---

## 1. Executive Summary

### 1.1 Problem Statement

Production ML systems require:
- **Multi-model serving**: 10-100+ models concurrently
- **Intelligent routing**: MOE for automatic model selection
- **Hierarchical ensembles**: MOE of MOEs for complex domains
- **Experimentation**: A/B testing with statistical rigor
- **Zero-downtime deployment**: Canary and blue-green strategies

### 1.2 Solution

```
┌─────────────────────────────────────────────────────────────────┐
│                    REALIZAR HOSTING LAYER                        │
├─────────────────────────────────────────────────────────────────┤
│  Traffic Router → A/B Splitter → MOE Router → Model Registry    │
│       ↓              ↓              ↓              ↓            │
│  [Requests]     [Cohorts]     [Experts]      [APR Models]       │
│                                    ↓                            │
│                          ┌────────────────┐                     │
│                          │ Hierarchical   │                     │
│                          │ MOE of MOEs    │                     │
│                          └────────────────┘                     │
└─────────────────────────────────────────────────────────────────┘
```

### 1.3 Key Differentiators

| Capability | Traditional | Realizar |
|------------|-------------|----------|
| Models hosted | 1-5 | 100+ |
| Routing | Static | Dynamic MOE |
| Ensembles | Flat | Hierarchical (MOE²) |
| A/B testing | External | Native |
| Deployment | Downtime | Zero-downtime |

---

## 2. Toyota Way Foundation

### 2.1 Principle Mapping

This specification applies Toyota Production System (TPS) principles per [1] Liker (2004):

| TPS Principle | Application | Implementation |
|---------------|-------------|----------------|
| **Jidoka** (Built-in Quality) | Model validation at load | CRC32 + Ed25519 signatures |
| **Just-in-Time** | Lazy model loading | Memory-map on first request |
| **Heijunka** (Level Loading) | Traffic smoothing | Rate limiting + load shedding |
| **Kaizen** | Continuous improvement | A/B testing + metrics |
| **Genchi Genbutsu** | Go and see | Real-time observability |
| **Standardized Work** | Consistent APIs | OpenAPI schema validation |
| **Andon** | Stop on defect | Circuit breakers |
| **Poka-Yoke** | Error prevention | Schema validation |

### 2.2 Jidoka: Autonomation with Human Touch

**Stop-the-line triggers** (per [2] Ohno 1988):

```rust
pub enum AndonTrigger {
    ModelChecksumMismatch,      // Stop: corrupted model
    LatencyP99Exceeded(f64),    // Warn: degraded performance
    ErrorRateThreshold(f64),    // Stop: systematic failures
    ExpertImbalance(f64),       // Warn: MOE routing skew
    ABTestSignificance(f64),    // Info: experiment concluded
}

impl AndonTrigger {
    pub fn severity(&self) -> Severity {
        match self {
            Self::ModelChecksumMismatch => Severity::Critical,
            Self::ErrorRateThreshold(_) => Severity::Critical,
            Self::LatencyP99Exceeded(_) => Severity::Warning,
            Self::ExpertImbalance(_) => Severity::Warning,
            Self::ABTestSignificance(_) => Severity::Info,
        }
    }
}
```

### 2.3 Heijunka: Load Leveling

Per [3] Hopp & Spearman (2011), variance reduction improves throughput:

```rust
pub struct HeijunkaController {
    /// Token bucket for rate limiting
    rate_limiter: RateLimiter,
    /// Queue depth triggers load shedding
    max_queue_depth: usize,
    /// Adaptive concurrency per Little's Law
    concurrency_limit: AdaptiveLimiter,
}

impl HeijunkaController {
    /// Shed load when queue exceeds capacity (Muri prevention)
    pub fn admit(&self, request: &Request) -> Result<Permit, Overload> {
        if self.queue_depth() > self.max_queue_depth {
            return Err(Overload::QueueFull);
        }
        self.rate_limiter.acquire()
    }
}
```

---

## 3. Architecture Overview

### 3.1 System Topology

```
                              ┌─────────────────┐
                              │   Load Balancer │
                              └────────┬────────┘
                                       │
                              ┌────────▼────────┐
                              │  Traffic Router │
                              │  (A/B Splitter) │
                              └────────┬────────┘
                                       │
              ┌────────────────────────┼────────────────────────┐
              │                        │                        │
     ┌────────▼────────┐     ┌────────▼────────┐     ┌────────▼────────┐
     │   Cohort A      │     │   Cohort B      │     │   Cohort C      │
     │   (Control)     │     │   (Treatment)   │     │   (Holdout)     │
     └────────┬────────┘     └────────┬────────┘     └────────┬────────┘
              │                        │                        │
     ┌────────▼────────┐     ┌────────▼────────┐     ┌────────▼────────┐
     │   MOE Router    │     │   MOE Router    │     │   MOE Router    │
     │   (Level 1)     │     │   (Level 1)     │     │   (Level 1)     │
     └────────┬────────┘     └────────┬────────┘     └────────┬────────┘
              │                        │                        │
    ┌─────────┼─────────┐    ┌─────────┼─────────┐             │
    │         │         │    │         │         │             │
┌───▼───┐ ┌───▼───┐ ┌───▼───┐ ┌───▼───┐ ┌───▼───┐       ┌───▼───┐
│ MOE-1 │ │ MOE-2 │ │ MOE-3 │ │ MOE-4 │ │ MOE-5 │       │Single │
│(L2)   │ │(L2)   │ │(L2)   │ │(L2)   │ │(L2)   │       │Model  │
└───┬───┘ └───┬───┘ └───┬───┘ └───┬───┘ └───┬───┘       └───────┘
    │         │         │         │         │
  ┌─┴─┐     ┌─┴─┐     ┌─┴─┐     ┌─┴─┐     ┌─┴─┐
  │E│E│     │E│E│     │E│E│     │E│E│     │E│E│
  └─┴─┘     └─┴─┘     └─┴─┘     └─┴─┘     └─┴─┘
   Experts   Experts   Experts   Experts   Experts
```

### 3.2 Component Summary

| Component | Responsibility | Cardinality |
|-----------|---------------|-------------|
| Traffic Router | Request ingress, cohort assignment | 1 |
| A/B Splitter | Experiment cohort allocation | 1 |
| MOE Router L1 | Domain/task routing | 1 per cohort |
| MOE Router L2 | Expert selection within domain | N per L1 |
| Model Registry | APR model storage and loading | 1 (shared) |
| Experts | Individual inference models | M per MOE |

---

## 4. Multi-Model Registry

### 4.1 Registry Design

Per [4] Sculley et al. (2015), ML systems require careful dependency management:

```rust
pub struct ModelRegistry {
    /// Model storage (file system, S3, etc.)
    storage: Arc<dyn ModelStorage>,
    /// In-memory model cache (LRU eviction)
    cache: Arc<RwLock<LruCache<ModelId, Arc<dyn Model>>>>,
    /// Model metadata index
    index: Arc<RwLock<HashMap<ModelId, ModelMetadata>>>,
    /// Maximum models in memory
    max_cached_models: usize,
}

impl ModelRegistry {
    /// Load model with Just-in-Time semantics
    pub async fn get(&self, id: &ModelId) -> Result<Arc<dyn Model>, RegistryError> {
        // Check cache first (hot path)
        if let Some(model) = self.cache.read().await.get(id) {
            return Ok(Arc::clone(model));
        }

        // Load from storage (cold path)
        let model = self.load_from_storage(id).await?;

        // Cache for future requests
        self.cache.write().await.put(id.clone(), Arc::clone(&model));

        Ok(model)
    }

    /// Preload models for latency-sensitive paths
    pub async fn warm(&self, ids: &[ModelId]) -> Result<(), RegistryError> {
        for id in ids {
            let _ = self.get(id).await?;
        }
        Ok(())
    }
}
```

### 4.2 Model Metadata Schema

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub id: ModelId,
    pub version: SemanticVersion,
    pub model_type: ModelType,
    pub created_at: DateTime<Utc>,
    pub checksum: [u8; 32],
    pub signature: Option<Ed25519Signature>,
    pub compression: CompressionType,
    pub size_bytes: u64,
    pub features: ModelFeatures,
    pub tags: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelFeatures {
    pub input_dim: usize,
    pub output_dim: usize,
    pub supports_batch: bool,
    pub quantization: Option<QuantizationType>,
    pub moe_config: Option<MoeConfig>,
}
```

### 4.3 Capacity Planning

| Metric | Target | Rationale |
|--------|--------|-----------|
| Max models in registry | 1000 | Disk-backed, lazy load |
| Max models in memory | 100 | RAM constraint (32GB) |
| Model load latency | <100ms | Memory-mapped APR |
| Cache hit ratio | >95% | LRU with warm-up |

---

## 5. Hierarchical MOE

### 5.1 Motivation

Per [5] Shazeer et al. (2017), sparse MOE scales capacity without proportional compute:

```
Traditional Ensemble:     Hierarchical MOE:
   ┌───┐                     ┌───┐
   │All│                     │L1 │ ← Top-k=2
   │ 8 │                     │MOE│
   │exp│                     └─┬─┘
   └───┘                   ┌──┴──┐
   O(8n)                 ┌─┴─┐ ┌─┴─┐
                         │L2 │ │L2 │ ← Top-k=2 each
                         │MOE│ │MOE│
                         └─┬─┘ └─┬─┘
                          ┌┴┐   ┌┴┐
                          EE    EE   ← 4 experts active
                         O(4n) total
```

**Compute savings:** 8 experts → 4 active = 50% reduction

### 5.2 Hierarchical MOE Architecture

```rust
/// Level 1: Domain/Task Router
pub struct L1MoeRouter {
    /// Routes to domain-specific L2 MOEs
    gating: SoftmaxGating,
    /// L2 MOE instances (domain experts)
    children: Vec<L2MoeRouter>,
    /// Routing configuration
    config: MoeConfig,
}

/// Level 2: Expert Router within Domain
pub struct L2MoeRouter {
    /// Routes to individual model experts
    gating: SoftmaxGating,
    /// Model experts (APR models)
    experts: Vec<Arc<dyn Model>>,
    /// Routing configuration
    config: MoeConfig,
}

/// Unified hierarchical router
pub struct HierarchicalMoe {
    l1: L1MoeRouter,
    /// Total experts across all L2 routers
    total_experts: usize,
    /// Active experts per inference (sparse)
    active_experts: usize,
}

impl HierarchicalMoe {
    pub fn predict(&self, input: &[f32]) -> Result<Vec<f32>, MoeError> {
        // L1: Select top-k domains
        let l1_weights = self.l1.gating.forward(input);
        let l1_top_k = top_k_indices(&l1_weights, self.l1.config.top_k);

        let mut output = vec![0.0; self.output_dim()];
        let mut total_weight = 0.0;

        for &l1_idx in &l1_top_k {
            let l1_weight = l1_weights[l1_idx];
            let l2_router = &self.l1.children[l1_idx];

            // L2: Select top-k experts within domain
            let l2_weights = l2_router.gating.forward(input);
            let l2_top_k = top_k_indices(&l2_weights, l2_router.config.top_k);

            for &l2_idx in &l2_top_k {
                let l2_weight = l2_weights[l2_idx];
                let combined_weight = l1_weight * l2_weight;

                let expert = &l2_router.experts[l2_idx];
                let expert_output = expert.predict(input)?;

                // Weighted combination
                for (i, &v) in expert_output.iter().enumerate() {
                    output[i] += combined_weight * v;
                }
                total_weight += combined_weight;
            }
        }

        // Normalize
        for v in &mut output {
            *v /= total_weight;
        }

        Ok(output)
    }
}
```

### 5.3 Configuration Examples

**Example 1: 10 MOE with 5 experts each = 50 total experts**

```rust
let config = HierarchicalMoeConfig {
    l1_experts: 10,        // 10 domain MOEs
    l1_top_k: 2,           // Select 2 domains per request
    l2_experts: 5,         // 5 experts per domain MOE
    l2_top_k: 2,           // Select 2 experts per domain
    // Active: 2 domains × 2 experts = 4 experts per request
    // Sparsity: 4/50 = 8% compute utilization
};
```

**Example 2: Deep hierarchy for large-scale serving**

```rust
let config = HierarchicalMoeConfig {
    l1_experts: 20,        // 20 task clusters
    l1_top_k: 3,           // 3 active clusters
    l2_experts: 10,        // 10 experts per cluster
    l2_top_k: 2,           // 2 active experts
    // Total: 200 experts, Active: 6 per request
    // Sparsity: 3% compute utilization
};
```

### 5.4 Load Balancing

Per [6] Fedus et al. (2022), expert imbalance degrades performance:

```rust
pub struct LoadBalancer {
    /// Expert usage counts (exponential moving average)
    usage: Vec<AtomicF64>,
    /// Target uniform distribution
    target_usage: f64,
    /// Auxiliary loss weight for training
    aux_loss_weight: f32,
}

impl LoadBalancer {
    /// Compute auxiliary loss to encourage balanced routing
    pub fn auxiliary_loss(&self, router_probs: &[f32]) -> f32 {
        let n = router_probs.len() as f32;
        let mean_prob = 1.0 / n;

        // Coefficient of variation penalty
        let variance: f32 = router_probs.iter()
            .map(|p| (p - mean_prob).powi(2))
            .sum::<f32>() / n;

        self.aux_loss_weight * variance.sqrt() / mean_prob
    }

    /// Update usage statistics
    pub fn record_usage(&self, expert_idx: usize) {
        self.usage[expert_idx].fetch_add(1.0, Ordering::Relaxed);
    }

    /// Check for imbalance (Andon trigger)
    pub fn check_imbalance(&self) -> Option<AndonTrigger> {
        let usages: Vec<f64> = self.usage.iter()
            .map(|u| u.load(Ordering::Relaxed))
            .collect();

        let max = usages.iter().cloned().fold(0.0, f64::max);
        let min = usages.iter().cloned().fold(f64::MAX, f64::min);

        let imbalance = (max - min) / (max + min + 1e-10);

        if imbalance > 0.5 {
            Some(AndonTrigger::ExpertImbalance(imbalance))
        } else {
            None
        }
    }
}
```

---

## 6. A/B Testing Framework

### 6.1 Statistical Foundation

Per [7] Kohavi et al. (2020), online experimentation requires:

1. **Randomization unit**: User/session/request
2. **Sample size**: Power analysis for effect detection
3. **Duration**: Multiple business cycles
4. **Metrics**: Primary (guardrail) + secondary (exploratory)

### 6.2 Experiment Configuration

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Experiment {
    pub id: ExperimentId,
    pub name: String,
    pub status: ExperimentStatus,
    pub allocation: TrafficAllocation,
    pub variants: Vec<Variant>,
    pub metrics: MetricsConfig,
    pub duration: ExperimentDuration,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Variant {
    pub id: VariantId,
    pub name: String,
    pub model_config: ModelConfig,
    pub traffic_weight: f64,  // 0.0 - 1.0
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrafficAllocation {
    /// Percentage of total traffic in experiment
    pub experiment_traffic: f64,
    /// Holdout percentage (no treatment)
    pub holdout_percentage: f64,
    /// Randomization salt for consistent hashing
    pub salt: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsConfig {
    /// Primary metric (statistical significance required)
    pub primary: MetricDefinition,
    /// Guardrail metrics (must not regress)
    pub guardrails: Vec<MetricDefinition>,
    /// Secondary metrics (exploratory)
    pub secondary: Vec<MetricDefinition>,
}
```

### 6.3 Cohort Assignment

Consistent hashing ensures stable assignment per [8] Karger et al. (1997):

```rust
pub struct CohortAssigner {
    experiments: Vec<Experiment>,
    hasher: SipHasher,
}

impl CohortAssigner {
    /// Assign request to cohort (deterministic by user_id)
    pub fn assign(&self, user_id: &str, experiment_id: &ExperimentId) -> Cohort {
        let experiment = self.get_experiment(experiment_id)?;

        // Consistent hash: same user always gets same cohort
        let hash = self.hash(user_id, &experiment.allocation.salt);
        let bucket = (hash % 10000) as f64 / 10000.0;

        // Check if in experiment traffic
        if bucket > experiment.allocation.experiment_traffic {
            return Cohort::NotInExperiment;
        }

        // Check if in holdout
        let holdout_threshold = experiment.allocation.holdout_percentage;
        if bucket < holdout_threshold {
            return Cohort::Holdout;
        }

        // Assign to variant based on traffic weights
        let variant_bucket = (hash >> 32) % 10000;
        let mut cumulative = 0.0;

        for variant in &experiment.variants {
            cumulative += variant.traffic_weight * 10000.0;
            if variant_bucket < cumulative as u64 {
                return Cohort::Variant(variant.id.clone());
            }
        }

        Cohort::Control
    }
}
```

### 6.4 Statistical Analysis

Per [9] Deng et al. (2017), use variance reduction for faster experiments:

```rust
pub struct ExperimentAnalyzer {
    /// Minimum detectable effect size
    mde: f64,
    /// Significance level (typically 0.05)
    alpha: f64,
    /// Statistical power (typically 0.80)
    power: f64,
}

impl ExperimentAnalyzer {
    /// Welch's t-test for unequal variances
    pub fn analyze(&self, control: &[f64], treatment: &[f64]) -> AnalysisResult {
        let n1 = control.len() as f64;
        let n2 = treatment.len() as f64;

        let mean1 = control.iter().sum::<f64>() / n1;
        let mean2 = treatment.iter().sum::<f64>() / n2;

        let var1 = control.iter().map(|x| (x - mean1).powi(2)).sum::<f64>() / (n1 - 1.0);
        let var2 = treatment.iter().map(|x| (x - mean2).powi(2)).sum::<f64>() / (n2 - 1.0);

        let se = (var1 / n1 + var2 / n2).sqrt();
        let t_stat = (mean2 - mean1) / se;

        // Welch-Satterthwaite degrees of freedom
        let df = (var1 / n1 + var2 / n2).powi(2) / (
            (var1 / n1).powi(2) / (n1 - 1.0) +
            (var2 / n2).powi(2) / (n2 - 1.0)
        );

        let p_value = 2.0 * (1.0 - t_distribution_cdf(t_stat.abs(), df));

        AnalysisResult {
            control_mean: mean1,
            treatment_mean: mean2,
            effect_size: (mean2 - mean1) / mean1,
            t_statistic: t_stat,
            p_value,
            significant: p_value < self.alpha,
            confidence_interval: self.confidence_interval(mean2 - mean1, se, df),
        }
    }

    /// Required sample size for desired power
    pub fn required_sample_size(&self, baseline_rate: f64, baseline_variance: f64) -> usize {
        // Two-sample t-test power calculation
        let z_alpha = 1.96;  // 95% confidence
        let z_beta = 0.84;   // 80% power

        let effect = baseline_rate * self.mde;
        let n = 2.0 * baseline_variance * (z_alpha + z_beta).powi(2) / effect.powi(2);

        n.ceil() as usize
    }
}
```

### 6.5 Experiment Lifecycle

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│  Draft   │───▶│  Active  │───▶│ Analyzing│───▶│ Complete │
└──────────┘    └────┬─────┘    └──────────┘    └──────────┘
                     │
                     ▼
               ┌──────────┐
               │  Paused  │ (Andon triggered)
               └──────────┘
```

---

## 7. Traffic Routing

### 7.1 Routing Pipeline

```rust
pub struct TrafficRouter {
    cohort_assigner: CohortAssigner,
    moe_routers: HashMap<VariantId, Arc<HierarchicalMoe>>,
    model_registry: Arc<ModelRegistry>,
    load_balancer: LoadBalancer,
    circuit_breaker: CircuitBreaker,
}

impl TrafficRouter {
    pub async fn route(&self, request: InferenceRequest) -> Result<InferenceResponse, RouterError> {
        // Step 1: Circuit breaker check (Andon)
        self.circuit_breaker.check()?;

        // Step 2: Cohort assignment (A/B)
        let cohort = self.cohort_assigner.assign(
            &request.user_id,
            &request.experiment_id,
        );

        // Step 3: Get appropriate router for cohort
        let router = match &cohort {
            Cohort::Variant(variant_id) => {
                self.moe_routers.get(variant_id)
                    .ok_or(RouterError::VariantNotFound)?
            }
            Cohort::Control => {
                self.moe_routers.get(&VariantId::control())
                    .ok_or(RouterError::ControlNotFound)?
            }
            Cohort::Holdout => {
                // Holdout uses baseline model (no MOE)
                return self.baseline_inference(&request).await;
            }
            Cohort::NotInExperiment => {
                // Default routing
                self.moe_routers.get(&VariantId::default())
                    .ok_or(RouterError::DefaultNotFound)?
            }
        };

        // Step 4: MOE inference
        let start = Instant::now();
        let output = router.predict(&request.input)?;
        let latency = start.elapsed();

        // Step 5: Record metrics
        self.record_metrics(&request, &cohort, latency);

        Ok(InferenceResponse {
            output,
            cohort: cohort.to_string(),
            latency_ms: latency.as_millis() as f64,
            experts_used: router.last_experts_used(),
        })
    }
}
```

### 7.2 Circuit Breaker (Andon Implementation)

Per [10] Nygard (2018), circuit breakers prevent cascade failures:

```rust
pub struct CircuitBreaker {
    state: AtomicU8,
    failure_count: AtomicU64,
    success_count: AtomicU64,
    last_failure: AtomicU64,
    config: CircuitBreakerConfig,
}

#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    /// Failures before opening circuit
    pub failure_threshold: u64,
    /// Successes in half-open to close
    pub success_threshold: u64,
    /// Time before attempting half-open
    pub reset_timeout: Duration,
    /// Error rate threshold
    pub error_rate_threshold: f64,
}

impl CircuitBreaker {
    pub fn check(&self) -> Result<(), CircuitOpen> {
        match self.state.load(Ordering::SeqCst) {
            STATE_CLOSED => Ok(()),
            STATE_OPEN => {
                // Check if timeout elapsed
                let elapsed = self.time_since_last_failure();
                if elapsed > self.config.reset_timeout {
                    self.state.store(STATE_HALF_OPEN, Ordering::SeqCst);
                    Ok(())
                } else {
                    Err(CircuitOpen::new(self.config.reset_timeout - elapsed))
                }
            }
            STATE_HALF_OPEN => Ok(()),  // Allow probe request
            _ => unreachable!(),
        }
    }

    pub fn record_success(&self) {
        self.success_count.fetch_add(1, Ordering::Relaxed);

        if self.state.load(Ordering::SeqCst) == STATE_HALF_OPEN {
            if self.success_count.load(Ordering::Relaxed) >= self.config.success_threshold {
                self.state.store(STATE_CLOSED, Ordering::SeqCst);
                self.failure_count.store(0, Ordering::Relaxed);
            }
        }
    }

    pub fn record_failure(&self) {
        let failures = self.failure_count.fetch_add(1, Ordering::Relaxed) + 1;
        self.last_failure.store(now_millis(), Ordering::Relaxed);

        if failures >= self.config.failure_threshold {
            self.state.store(STATE_OPEN, Ordering::SeqCst);
        }
    }
}
```

---

## 8. Observability

### 8.1 Metrics

```rust
pub struct HostingMetrics {
    // Latency histograms
    pub inference_latency: Histogram,
    pub routing_latency: Histogram,
    pub model_load_latency: Histogram,

    // Counters
    pub requests_total: Counter,
    pub errors_total: Counter,
    pub cache_hits: Counter,
    pub cache_misses: Counter,

    // Gauges
    pub models_loaded: Gauge,
    pub active_experiments: Gauge,
    pub circuit_breaker_state: Gauge,

    // MOE-specific
    pub expert_usage: Vec<Counter>,
    pub expert_imbalance: Gauge,
    pub routing_entropy: Histogram,
}
```

### 8.2 Structured Logging

```rust
#[derive(Serialize)]
pub struct InferenceLog {
    pub request_id: String,
    pub user_id: String,
    pub experiment_id: Option<String>,
    pub cohort: String,
    pub variant_id: Option<String>,
    pub model_ids: Vec<String>,
    pub experts_used: Vec<usize>,
    pub latency_ms: f64,
    pub input_dim: usize,
    pub output_dim: usize,
    pub timestamp: DateTime<Utc>,
}
```

### 8.3 Dashboards

| Dashboard | Metrics | Alert Threshold |
|-----------|---------|-----------------|
| Latency | p50, p95, p99 | p99 > 100ms |
| Errors | Error rate by type | >1% |
| MOE Routing | Expert utilization | Imbalance >50% |
| A/B Tests | Effect size, p-value | Guardrail regression |
| Capacity | Models loaded, memory | >80% utilization |

---

## 9. Implementation

### 9.1 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Inference with automatic routing |
| `/predict/moe` | POST | Explicit MOE inference |
| `/models` | GET | List registered models |
| `/models/{id}` | GET | Model metadata |
| `/models/{id}` | PUT | Register/update model |
| `/experiments` | GET | List experiments |
| `/experiments` | POST | Create experiment |
| `/experiments/{id}` | GET | Experiment details |
| `/experiments/{id}/start` | POST | Start experiment |
| `/experiments/{id}/stop` | POST | Stop experiment |
| `/experiments/{id}/analyze` | GET | Statistical analysis |

### 9.2 Request/Response Schemas

**Inference Request:**
```json
{
  "input": [0.1, 0.2, 0.3],
  "user_id": "user-123",
  "experiment_id": "exp-456",
  "options": {
    "return_routing_info": true,
    "timeout_ms": 100
  }
}
```

**Inference Response:**
```json
{
  "output": [0.85, 0.12, 0.03],
  "cohort": "treatment-a",
  "variant_id": "variant-1",
  "routing": {
    "l1_experts": [2, 5],
    "l1_weights": [0.6, 0.4],
    "l2_experts": [[1, 3], [0, 2]],
    "l2_weights": [[0.7, 0.3], [0.5, 0.5]]
  },
  "latency_ms": 5.2
}
```

### 9.3 Deployment Architecture

```yaml
# Kubernetes deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: realizar-hosting
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: realizar
        image: realizar:0.3.0
        resources:
          requests:
            memory: "8Gi"
            cpu: "4"
          limits:
            memory: "32Gi"
            cpu: "16"
        env:
        - name: MODEL_CACHE_SIZE
          value: "100"
        - name: MAX_CONCURRENT_REQUESTS
          value: "1000"
        volumeMounts:
        - name: models
          mountPath: /models
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: model-storage
```

---

## 10. Scientific Foundation

### 10.1 Annotated Bibliography

#### [1] Liker, J.K. (2004). *The Toyota Way: 14 Management Principles*. McGraw-Hill.

**Relevance:** Foundational text for TPS principles applied throughout this specification.
- Jidoka (autonomation): Model validation, circuit breakers
- Just-in-Time: Lazy model loading
- Kaizen: A/B testing for continuous improvement
- Heijunka: Load leveling via rate limiting

**Applied in:** Sections 2.1, 2.2, 2.3

---

#### [2] Ohno, T. (1988). *Toyota Production System: Beyond Large-Scale Production*. Productivity Press.

**Relevance:** Original TPS documentation from creator. Key insight: "Stop to fix problems" (Andon).

**Applied in:** Circuit breaker design (Section 7.2), AndonTrigger enum (Section 2.2)

---

#### [3] Hopp, W.J. & Spearman, M.L. (2011). *Factory Physics*. 3rd ed. Waveland Press.

**Relevance:** Queueing theory foundation for load balancing. Key results:
- Little's Law: L = λW (queue length = arrival rate × wait time)
- Variance reduction improves throughput
- Utilization near 100% causes exponential wait times

**Applied in:** HeijunkaController (Section 2.3), capacity planning (Section 4.3)

---

#### [4] Sculley, D., et al. (2015). "Hidden Technical Debt in Machine Learning Systems." *NeurIPS*.

**Relevance:** Seminal paper on ML system complexity. Key warnings:
- Entanglement: Changing one model affects others
- Data dependencies harder to track than code
- Configuration debt from hyperparameter management

**Applied in:** Model registry design (Section 4.1), metadata schema (Section 4.2)

---

#### [5] Shazeer, N., et al. (2017). "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer." *ICLR*.

**Relevance:** Foundational MOE paper from Google. Key contributions:
- Sparsely-gated MOE scales to 137B parameters
- Top-k routing reduces compute proportionally
- Load balancing auxiliary loss prevents collapse

**Applied in:** Hierarchical MOE design (Section 5), load balancing (Section 5.4)

---

#### [6] Fedus, W., Zoph, B., & Shazeer, N. (2022). "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity." *JMLR*.

**Relevance:** Simplified MOE routing (k=1). Key insights:
- Expert capacity factor prevents dropped tokens
- Auxiliary load balancing loss critical for training
- Expert parallelism for distributed serving

**Applied in:** Expert imbalance detection (Section 5.4), Andon triggers

---

#### [7] Kohavi, R., Tang, D., & Xu, Y. (2020). *Trustworthy Online Controlled Experiments: A Practical Guide to A/B Testing*. Cambridge University Press.

**Relevance:** Authoritative A/B testing reference from Microsoft/Airbnb. Key principles:
- Randomization unit determines experiment validity
- Multiple testing correction for many metrics
- Guardrail metrics prevent regressions

**Applied in:** Experiment configuration (Section 6.2), metrics config

---

#### [8] Karger, D., et al. (1997). "Consistent Hashing and Random Trees." *STOC*.

**Relevance:** Consistent hashing for stable cohort assignment.
- Same user always in same cohort
- Adding/removing experiments minimal disruption
- Used in CDNs, databases, load balancers

**Applied in:** CohortAssigner (Section 6.3)

---

#### [9] Deng, A., et al. (2017). "Applying the Delta Method in Metric Analytics: A Practical Guide with Novel Ideas." *KDD*.

**Relevance:** Variance reduction for faster A/B tests from Microsoft.
- CUPED (Controlled-experiment Using Pre-Experiment Data)
- Ratio metrics require delta method
- Reduces required sample size 30-50%

**Applied in:** ExperimentAnalyzer (Section 6.4)

---

#### [10] Nygard, M. (2018). *Release It! Design and Deploy Production-Ready Software*. 2nd ed. Pragmatic Bookshelf.

**Relevance:** Stability patterns for production systems. Key patterns:
- Circuit breaker: Prevent cascade failures
- Bulkhead: Isolate failures
- Timeout: Bound latency

**Applied in:** CircuitBreaker (Section 7.2), timeout handling

---

### 10.2 Additional References

| Topic | Reference | Application |
|-------|-----------|-------------|
| Queueing Theory | Kleinrock (1975) *Queueing Systems* | Load balancing math |
| Statistical Power | Cohen (1988) *Statistical Power Analysis* | Sample size calculation |
| Distributed Systems | Kleppmann (2017) *Designing Data-Intensive Applications* | Registry design |

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-11-27 | Claude Code | Initial specification |

---

**END OF SPECIFICATION**
