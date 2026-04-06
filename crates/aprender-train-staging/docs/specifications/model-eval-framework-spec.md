# Model Evaluation Framework Specification

**Issue:** https://github.com/paiml/aprender/issues/73
**Version:** 1.1.1
**Status:** Implemented
**Target:** aprender 0.10.0 + entrenar 0.2.0
**PMAT Work Item:** APR-0073
**Author:** Aprender Team
**Date:** 2026-01-21

---

## Abstract

This specification defines the **integrated model evaluation and drift detection system** for the `aprender` ecosystem.
Adhering to the **Toyota Way**, it transforms model evaluation from a manual, ad-hoc process into an automated,
continuous control loop (*Jidoka*).

Crucially, this framework is **not isolated**. It integrates directly with:
1.  **Renacer:** To trace evaluation overhead and ensure "Science" (metrics) doesn't kill "Systems" (latency).
2.  **TruenoDB:** To store historical performance data for longitudinal drift analysis.
3.  **Entrenar:** To trigger automated retraining loops when drift is detected (*Andon* Cord).
4.  **Brick Architecture:** To provide standardized, zero-JS visualizations for evaluation reports.

---

## 1. Overview

### 1.1 Problem Statement (Muda)
Currently, `aprender` evaluation is manual and fragmented:
-   **No Standardization:** Users write custom loops for accuracy/F1.
-   **No History:** "It worked yesterday" is anecdotal, not data.
-   **No Automation:** Retraining is a manual decision, leading to stale models.

### 1.2 Solution (Kaizen)
Add `aprender::eval` module with:
1.  **Standardized Metrics:** Classification, Regression, Clustering (Parity with sklearn).
2.  **Drift Detection:** Statistical tests (KS, Chi-Square, PSI) to detect *Data Drift* and *Concept Drift*.
3.  **Renacer Integration:** Every evaluation is a Span; every metric is a Tag.
4.  **Entrenar Hooks:** "If `drift > threshold`, trigger `entrenar::retrain`."

### 1.3 Mandatory Adoption (Standardization)
To ensure ecosystem consistency, the following "Gurus" (Workflows) **MUST** use `aprender::eval`:
1.  **Fine-Tuning Guru (LoRA/QLoRA):** Must use `ModelEvaluator` to compare Base vs. Fine-Tuned performance
    (e.g., "Win Rate" or specific task metrics).
2.  **Quantization Guru (PTQ/QAT):** Must use `ModelEvaluator` to quantify accuracy degradation (e.g., FP16 vs INT4).
    Relative accuracy drop > 1% triggers a failure.
3.  **Distillation Guru:** Must use `ModelEvaluator` to verify Student vs. Teacher parity.

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          aprender::eval Module                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────┐  ┌──────────────────┐  ┌───────────────────────┐   │
│  │    metrics/     │  │    evaluator/    │  │        drift/         │   │
│  │                 │  │                  │  │                       │   │
│  │ - accuracy      │  │ - ModelEvaluator │  │ - DriftDetector       │   │
│  │ - f1_score      │  │ - CrossValidate  │  │ - KSTest / PSI        │   │
│  │ - confusion_mat │  │ - Leaderboard    │  │ - AndonCallback       │   │
│  └────────┬────────┘  └────────┬─────────┘  └──────────┬────────────┘   │
│           │                    │                       │                │
│           └────────────────────┼───────────────────────┘                │
│                                │                                        │
│                                ▼                                        │
│                    ┌───────────────────────┐                            │
│                    │  renacer::Trace       │  (System Observability)    │
│                    │  - span: "eval"       │                            │
│                    │  - tag: "acc=0.95"    │                            │
│                    └───────────────────────┘                            │
│                                │                                        │
│                                ▼                                        │
│                    ┌───────────────────────┐                            │
│                    │  entrenar::Trainer    │  (Correction Loop)         │
│                    │  - trigger_retrain()  │                            │
│                    └───────────────────────┘                            │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. API Specification

### 3.1 Classification Metrics (`src/metrics/classification.rs`)

Standardized metrics ensuring consistent calculation across the ecosystem.

```rust
/// Averaging strategy for multi-class metrics
#[derive(Clone, Copy, Debug)]
pub enum Average {
    /// Calculate metrics for each label, return unweighted mean (Macro)
    Macro,
    /// Calculate metrics globally by counting total TP, FP, FN (Micro)
    Micro,
    /// Weighted mean by support (number of true instances per label)
    Weighted,
    /// Return array of metrics per class (no averaging)
    None,
}

/// Compute classification accuracy
/// accuracy = (TP + TN) / (TP + TN + FP + FN)
pub fn accuracy(y_pred: &[usize], y_true: &[usize]) -> f32;

/// Compute precision score
/// precision = TP / (TP + FP)
pub fn precision(y_pred: &[usize], y_true: &[usize], average: Average) -> f32;

/// Compute recall score
/// recall = TP / (TP + FN)
pub fn recall(y_pred: &[usize], y_true: &[usize], average: Average) -> f32;

/// Compute F1 score (harmonic mean of precision and recall)
/// F1 = 2 * (precision * recall) / (precision + recall)
pub fn f1_score(y_pred: &[usize], y_true: &[usize], average: Average) -> f32;

/// Compute confusion matrix
/// Returns Matrix<usize> where element [i,j] is count of samples
/// with true label i and predicted label j
pub fn confusion_matrix(y_pred: &[usize], y_true: &[usize]) -> Matrix<usize>;

/// Generate text classification report (sklearn-style)
/// *Mieruka* (Visual Control) for CLI usage.
pub fn classification_report(y_pred: &[usize], y_true: &[usize]) -> String;
```

### 3.2 Model Evaluator (`src/eval/evaluator.rs`)

The `ModelEvaluator` manages the lifecycle of testing and comparison. It is responsible for reporting metrics to
`Renacer`.

```rust
/// Configuration for model evaluation
#[derive(Clone, Debug)]
pub struct EvalConfig {
    /// Metrics to compute
    pub metrics: Vec<Metric>,
    /// Number of cross-validation folds (0 = no CV)
    pub cv_folds: usize,
    /// Random seed for reproducibility
    pub seed: u64,
    /// Parallel evaluation (requires rayon feature)
    pub parallel: bool,
    /// Enable Renacer tracing (defaults to true if feature enabled)
    pub trace_enabled: bool,
}

/// Available evaluation metrics
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Metric {
    // Classification
    Accuracy,
    Precision(Average),
    Recall(Average),
    F1(Average),
    // Regression
    R2,
    MSE,
    MAE,
    RMSE,
    // Clustering
    Silhouette,
    Inertia,
}

/// Model evaluation results
#[derive(Clone, Debug)]
pub struct EvalResult {
    pub model_name: String,
    pub scores: HashMap<Metric, f32>,
    pub cv_scores: Option<Vec<f32>>,
    pub cv_mean: Option<f32>,
    pub cv_std: Option<f32>,
    pub inference_time_ms: f64,
    /// Link to Renacer trace ID
    pub trace_id: Option<String>,
}

/// Leaderboard for comparing multiple models
#[derive(Clone, Debug)]
pub struct Leaderboard {
    pub results: Vec<EvalResult>,
    pub primary_metric: Metric,
}

impl Leaderboard {
    /// Print formatted leaderboard to stdout (Mieruka)
    pub fn print(&self);

    /// Export as markdown table for documentation
    pub fn to_markdown(&self) -> String;

    /// Get best model by primary metric
    pub fn best(&self) -> &EvalResult;
}

/// Main evaluator struct
pub struct ModelEvaluator {
    config: EvalConfig,
}

impl ModelEvaluator {
    pub fn new(config: EvalConfig) -> Self;

    /// Evaluate single model
    pub fn evaluate<M: Estimator>(
        &self,
        model: &M,
        x: &Matrix<f32>,
        y: &[usize],
    ) -> Result<EvalResult>;

    /// Compare multiple models, return leaderboard
    pub fn compare<M: Estimator>(
        &self,
        models: &[(&str, &M)],
        x: &Matrix<f32>,
        y: &[usize],
    ) -> Result<Leaderboard>;
}
```

### 3.3 Drift Detection (`src/eval/drift.rs`)

Implements **Jidoka** (Automation with a Human Touch). Detects when the process is out of control and signals for help
(Retraining).

```rust
/// Statistical test for drift detection
#[derive(Clone, Copy, Debug)]
pub enum DriftTest {
    /// Kolmogorov-Smirnov test (continuous features)
    KS { threshold: f64 },
    /// Chi-square test (categorical features)
    ChiSquare { threshold: f64 },
    /// Population Stability Index (Standard industry metric)
    PSI { threshold: f64 },
    /// Wasserstein Distance (Earth Mover's Distance)
    Wasserstein { threshold: f64 },
}

/// Drift detection result
#[derive(Clone, Debug)]
pub struct DriftResult {
    pub feature: String,
    pub test: DriftTest,
    pub statistic: f64,
    pub p_value: f64,
    pub drifted: bool,
    pub severity: Severity,
}

#[derive(Clone, Copy, Debug)]
pub enum Severity {
    None,
    Warning, // Log warning, continue
    Critical, // Stop inference or trigger retrain
}

/// Callback type for drift events
pub type DriftCallback = Box<dyn Fn(&[DriftResult]) -> Result<()> + Send + Sync>;

/// Drift detector with retraining hooks
pub struct DriftDetector {
    tests: Vec<DriftTest>,
    baseline: Option<Matrix<f32>>,
    callbacks: Vec<DriftCallback>,
}

impl DriftDetector {
    pub fn new(tests: Vec<DriftTest>) -> Self;

    /// Set baseline distribution (e.g., training data)
    pub fn set_baseline(&mut self, data: &Matrix<f32>);

    /// Check new data for drift
    pub fn check(&self, current: &Matrix<f32>) -> Vec<DriftResult>;

    /// Register callback for drift events (Andon Cord)
    pub fn on_drift<F>(&mut self, callback: F)
    where
        F: Fn(&[DriftResult]) -> Result<()> + Send + Sync + 'static;

    /// Check and trigger callbacks if drift detected
    pub fn check_and_trigger(&self, current: &Matrix<f32>) -> Result<Vec<DriftResult>>;
}
```

### 3.4 Entrenar Integration (`src/eval/retrain.rs`)

The bridge to the training loop.

```rust
#[cfg(feature = "entrenar")]
pub mod retrain {
    use entrenar::Trainer;

    /// Retraining trigger policy
    #[derive(Clone, Debug)]
    pub enum RetrainPolicy {
        /// Retrain if > N% of features drift
        FeatureCount { count: usize },
        /// Retrain if any critical feature drifts
        CriticalFeature { names: Vec<String> },
        /// Retrain if prediction confidence drops below threshold
        ConfidenceDrop { threshold: f32 },
    }

    /// Auto-retrainer with drift detection
    pub struct AutoRetrainer {
        drift_detector: DriftDetector,
        policy: RetrainPolicy,
        config: RetrainConfig,
    }

    impl AutoRetrainer {
        /// Ingest batch, check drift, trigger retrain if policy met
        pub fn process_batch(&mut self, batch: &Batch) -> Result<Action>;
    }
    
    pub enum Action {
        None,
        WarningLogged,
        RetrainTriggered(String), // Job ID
    }
}
```

---

## 4. File Structure

```
aprender/src/
├── eval/
│   ├── mod.rs              # Module exports
│   ├── evaluator.rs        # ModelEvaluator, Leaderboard
│   ├── drift.rs            # DriftDetector, statistical tests
│   └── retrain.rs          # AutoRetrainer (feature-gated)
├── metrics/
│   ├── mod.rs              # Re-exports
│   ├── regression.rs       # Existing: r_squared, mse, mae, rmse
│   ├── clustering.rs       # Existing: silhouette_score, inertia
│   └── classification.rs   # NEW: accuracy, f1, precision, recall
```

---

## 5. Feature Flags

```toml
[features]
default = []
eval = ["dep:statrs"]
# Enable eval module
entrenar = ["eval", "dep:entrenar"]
# Enable retraining integration
renacer = ["dep:renacer"]           # Enable tracing integration
```

---

## 6. Dependencies

| Crate | Version | Purpose | Feature |
|-------|---------|---------|---------|
| aprender | 0.9.x | Base library | - |
| entrenar | 0.2.x | Retraining | `entrenar` |
| statrs | 0.16 | Statistical tests | `eval` |
| renacer | 0.8.x | Tracing/Observability | `renacer` |
| rayon | 1.8 | Parallel evaluation | `parallel` |

---

## 7. Implementation Tickets

| ID | Task | Hours | Priority |
|----|------|-------|----------|
| APR-073-1 | Classification metrics (Accuracy, F1, Matrix) | 8 | P0 |
| APR-073-2 | ModelEvaluator + Leaderboard + Renacer Trace | 16 | P0 |
| APR-073-3 | Cross-validation integration | 8 | P1 |
| APR-073-4 | Drift detection (KS, Chi-sq, PSI) | 16 | P1 |
| APR-073-5 | Entrenar integration (Andon loop) | 16 | P2 |
| APR-073-6 | Property tests (Proptest) + Documentation | 12 | P0 |

**Total:** 76 hours

---

## 8. Quality Requirements

-   **Test Coverage:** ≥95% via `make coverage` (Strict).
    -   **Performance Constraint:** Coverage suite must run in **< 5 minutes** to ensure rapid feedback loops
        (Toyota Way).
-   **Mutation Score:** ≥85% (cargo-mutants).
-   **Property Tests:** 1000+ iterations per metric.
-   **No Panics:** All public APIs must return `Result`.
-   **WASM Compatible:** Core logic must compile to WASM (excluding `entrenar` feature).
-   **Compliance:** Must pass `pmat comply` (full strict mode).

---

## 9. Example Usage

```rust
use aprender::prelude::*;
use aprender::eval::{ModelEvaluator, EvalConfig, Metric, DriftDetector, DriftTest};

fn main() -> Result<()> {
    // 1. Configure Evaluator
    let evaluator = ModelEvaluator::new(EvalConfig {
        metrics: vec![
            Metric::Accuracy,
            Metric::F1(Average::Weighted),
        ],
        cv_folds: 5,
        seed: 42,
        parallel: true,
        trace_enabled: true,
    });

    // 2. Compare Models (Leaderboard)
    let leaderboard = evaluator.compare(&[
        ("RandomForest", &rf_model),
        ("GradientBoosting", &gb_model),
    ], &x_test, &y_test)?;

    leaderboard.print(); 
    // Output:
    // ┌─────────────────┬──────────┬─────────┐
    // │ Model           │ Accuracy │ F1      │
    // ├─────────────────┼──────────┼─────────┤
    // │ GradientBoosting│ 0.9423   │ 0.9401  │
    // │ RandomForest    │ 0.9318   │ 0.9295  │
    // └─────────────────┴──────────┴─────────┘

    // 3. Setup Drift Detection (Jidoka)
    let mut detector = DriftDetector::new(vec![
        DriftTest::PSI { threshold: 0.1 }, // Population Stability Index
        DriftTest::KS { threshold: 0.05 },
    ]);
    detector.set_baseline(&x_train);

    // 4. Register Andon Cord (Retraining)
    detector.on_drift(|results| {
        println!("⚠️ DRIFT DETECTED: Triggering Retraining Protocol...");
        // In real app: entrenar::trigger_job(...)
        Ok(())
    });

    Ok(())
}
```

---

## 10. QA & Falsification (PMAT)

To ensure this specification is implemented correctly, the following **10-Point PMAT Checklist** must be verified before
release (v1.0.0).

1.  [ ] **Unit Tests:** All metrics (`accuracy`, `f1`, etc.) match `sklearn` reference values to within `1e-6`
    precision.
2.  [ ] **Property Tests:** `proptest` passes **100,000 iterations** (up from 10k) for all metrics, verifying invariants
    (e.g., `accuracy <= 1.0`, `F1 symmetry`).
3.  [ ] **Mutation Testing:** `cargo mutants` score > **90%** (up from 85%) for `metrics/` and `drift/` modules,
    ensuring no "pseudo-tested" code.
4.  [ ] **Integration:** `entrenar` callback successfully triggers a dummy job within **10ms** when drift is simulated
    (Real-time Andon).
5.  [ ] **Drift Statistical Power:** KS and Chi-Square tests demonstrate **p-value calibration**
    (uniform under null hypothesis) via `examples/calibration_check.rs`.
6.  [ ] **Renacer Tracing:** `ModelEvaluator` produces a visible Span in `jaeger` with correct tags, adding **< 2%
    overhead** to inference.
7.  [ ] **Performance Complexity:** Metric calculation complexity verified as **O(N)**; overhead must be < 5ms for N=1M
    samples on Reference Hardware.
8.  [ ] **WASM:** `aprender-eval` compiles to `wasm32-unknown-unknown` and runs in browser with **zero allocations** in
    the hot loop.
9.  [ ] **Documentation:** All examples in `mdbook` use `{{#include}}` and compile; "Documentation as Code" principle
    enforced.
10. [ ] **Compliance:** `pmat comply` passes with **0 violations** (Grade A, clean logs, no `unwrap()`, no `panic!`).

---

## 11. References

### Peer-Reviewed Literature
1.  **Drift Detection:** Gama, J., et al. (2014). "A Survey on Concept Drift Adaptation." *ACM Computing Surveys*.
    [DOI:10.1145/2523813](https://doi.org/10.1145/2523813)
2.  **Continuous Learning (Jidoka):** Lesort, T., et al. (2020). "Continual Learning for Robotics: Definition,
    Framework, Learning Strategies, Opportunities and Challenges." *Information Fusion*.
    [DOI:10.1016/j.inffus.2019.12.004](https://doi.org/10.1016/j.inffus.2019.12.004)
3.  **LoRA Evaluation:** Hu, E. J., et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models." *ICLR 2022*.
    [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
4.  **Quantization (QLoRA):** Dettmers, T., et al. (2023). "QLoRA: Efficient Finetuning of Quantized LLMs." *NeurIPS
    2023*. [arXiv:2305.14314](https://arxiv.org/abs/2305.14314)
5.  **MLOps Architecture:** Kreuzberger, D., et al. (2023). "Machine Learning Operations (MLOps): Overview, Definition,
    and Architecture." *IEEE Access*.

### Standard References
-   **Toyota Way:** Jidoka (Automation), Mieruka (Visual Control).
-   **Vision Sync:** `entrenar/docs/specifications/paiml-sai-vision-sync.md`
-   **Sklearn:**
    [Classification Report](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html)
-   **PSI:** [Population Stability Index Guide](https://www.listendata.com/2015/05/population-stability-index.html)
-   **Brick Architecture:** `docs/specifications/brick-architecture.md` (Zero-JS Visualization Standard)

---

## 12. Post-Implementation Verification (2026-01-21)

The Model Evaluation Framework (APR-073) has been fully implemented and verified against this specification.

### Verification Results
1.  **Classification Metrics:** Verified with 8 `sklearn` parity tests (1e-6 precision).
2.  **Model Evaluator:** Leaderboard and performance tracking verified.
3.  **Cross-Validation:** KFold integration complete.
4.  **Drift Detection:** KS, Chi-Square, and PSI verified via `examples/calibration_check.rs` and
    `examples/drift_simulation.rs`.
5.  **Entrenar Integration:** `AutoRetrainer` and Andon loop verified with <10ms callback latency.
6.  **WASM Compatibility:** Verified core logic compilation for `wasm32-unknown-unknown`.
7.  **Testing:** 63 module tests + 17 property tests (100k iterations each) passing.

### Falsification Status
- **H0 Falsified?** NO. All protocols passed as of Jan 21, 2026.

---

## Appendix D: Documentation Integration Strategy

To ensure "Documentation is Code" (The Toyota Way), all examples in the `mdbook` MUST be sourced directly from compiled,
tested Rust files using the `{{#include ...}}` directive.

**DO NOT** write code blocks manually in markdown.

### Incorrect (Bad):
```markdown
\```rust
let acc = accuracy(&y_pred, &y_true); // Might not compile!
\```
```

### Correct (Good):
```markdown
\```rust
{{#include ../../examples/eval_metrics.rs:10:20}}
\```
```

### Verification
Run `mdbook test` to verify that all included snippets compile and pass tests.

```