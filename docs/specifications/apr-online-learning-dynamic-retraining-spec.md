# Online Learning and Dynamic Retraining Infrastructure

**Specification Version**: 1.0.0
**Status**: DRAFT - Awaiting Review
**Author**: Claude Code
**Date**: 2025-12-08
**Dependencies**: ruchy/ORACLE-001, depyler/GH-172

---

## Executive Summary

This specification defines **online learning infrastructure** for aprender to support dynamic retraining in production ML systems. The system enables continuous model improvement through:

1. **Online Learning Traits**: Incremental model updates without full retraining
2. **Drift Detection**: Statistical tests to trigger retraining
3. **Corpus Management**: Automated data collection and deduplication
4. **Curriculum Learning**: Progressive difficulty scheduling
5. **Knowledge Distillation**: Teacher-student model compression
6. **CITL Integration**: Compiler-in-the-loop feedback hooks

**Primary Use Cases**: ruchy Oracle classifier, depyler transpilation model

---

## 1. Online Learning Traits

### 1.1 Core Abstraction

```rust
/// Online learning capability for incremental model updates
///
/// Reference: [Bottou 2010] "Large-Scale Machine Learning with Stochastic
/// Gradient Descent" - Online learning converges to optimal solution with
/// O(1/t) regret bound under convex loss.
pub trait OnlineLearner: Estimator {
    /// Update model with single sample (or mini-batch)
    ///
    /// # Arguments
    /// * `x` - Feature vector(s)
    /// * `y` - Target value(s)
    /// * `learning_rate` - Step size (may be adaptive)
    ///
    /// # Returns
    /// Loss on this sample before update (for monitoring)
    fn partial_fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        learning_rate: Option<f64>,
    ) -> Result<f64>;

    /// Check if model supports warm-starting from checkpoint
    fn supports_warm_start(&self) -> bool { true }

    /// Get current effective learning rate
    fn current_learning_rate(&self) -> f64;

    /// Number of samples seen so far
    fn n_samples_seen(&self) -> u64;
}

/// Passive-Aggressive online learning for classification
///
/// Reference: [Crammer et al. 2006] "Online Passive-Aggressive Algorithms"
/// - Margin-based updates with bounded aggressiveness
/// - Suitable for non-stationary distributions
pub trait PassiveAggressive: OnlineLearner {
    /// Aggressiveness parameter C (higher = more aggressive updates)
    fn aggressiveness(&self) -> f64;

    /// Set aggressiveness for PA-I or PA-II variants
    fn set_aggressiveness(&mut self, c: f64);
}
```

### 1.2 Supported Model Types

| Model | Online Support | Algorithm | Reference |
|-------|---------------|-----------|-----------|
| `LinearRegression` | ✅ | SGD with momentum | [Bottou 2010] |
| `LogisticRegression` | ✅ | SGD + AdaGrad | [Duchi et al. 2011] |
| `RandomForest` | ✅ | Mondrian Forest | [Lakshminarayanan et al. 2014] |
| `NaiveBayes` | ✅ | Sufficient statistics | Native incremental |
| `KMeans` | ✅ | Mini-batch K-Means | [Sculley 2010] |
| `NeuralNetwork` | ✅ | Adam/AdaGrad | [Kingma & Ba 2015] |

### 1.3 Implementation Pattern

```rust
impl OnlineLearner for LogisticRegression {
    fn partial_fit(
        &mut self,
        x: &Matrix,
        y: &Vector,
        learning_rate: Option<f64>,
    ) -> Result<f64> {
        let lr = learning_rate.unwrap_or(self.current_learning_rate());

        // Compute prediction and loss before update
        let pred = self.predict_proba(x)?;
        let loss = cross_entropy_loss(&pred, y);

        // Gradient: (pred - y) * x
        let error = pred.sub(y)?;
        let grad = x.transpose().matmul(&error)?;

        // SGD update with optional momentum
        self.weights = self.weights.sub(&grad.scale(lr))?;
        self.n_samples += x.nrows() as u64;

        // Adaptive learning rate decay (AdaGrad-style)
        self.accum_grad = self.accum_grad.add(&grad.pow(2.0))?;

        Ok(loss)
    }

    fn current_learning_rate(&self) -> f64 {
        // AdaGrad: lr / sqrt(accum + eps)
        self.base_lr / (self.accum_grad.mean().sqrt() + 1e-8)
    }

    fn n_samples_seen(&self) -> u64 {
        self.n_samples
    }
}
```

---

## 2. Drift Detection

### 2.1 Statistical Drift Detectors

```rust
/// Drift detection for triggering model retraining
///
/// Reference: [Gama et al. 2004] "Learning with Drift Detection"
/// - DDM detects concept drift via error rate monitoring
/// - Warning level triggers data collection, drift level triggers retrain
pub trait DriftDetector {
    /// Add new prediction outcome
    fn add_element(&mut self, error: bool);

    /// Check current drift status
    fn detected_change(&self) -> DriftStatus;

    /// Reset detector after handling drift
    fn reset(&mut self);

    /// Get detector statistics
    fn stats(&self) -> DriftStats;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DriftStatus {
    /// No drift detected
    Stable,
    /// Warning level - start collecting data
    Warning,
    /// Drift confirmed - trigger retraining
    Drift,
}

/// DDM (Drift Detection Method) implementation
///
/// Reference: [Gama et al. 2004]
/// Warning: p + s > p_min + 2*s_min
/// Drift: p + s > p_min + 3*s_min
pub struct DDM {
    /// Minimum samples before detection
    min_samples: u64,
    /// Warning threshold (standard deviations)
    warning_level: f64,
    /// Drift threshold (standard deviations)
    drift_level: f64,
    /// Running statistics
    n: u64,
    p: f64,      // error probability
    s: f64,      // standard deviation
    p_min: f64,
    s_min: f64,
}

/// Page-Hinkley Test for gradual drift
///
/// Reference: [Page 1954] "Continuous Inspection Schemes"
/// - Cumulative sum test for mean shift detection
/// - Better for gradual drift than DDM
pub struct PageHinkley {
    /// Minimum magnitude of change to detect
    delta: f64,
    /// Detection threshold
    lambda: f64,
    /// Cumulative sum
    sum: f64,
    /// Running mean
    mean: f64,
    n: u64,
}

/// ADWIN (Adaptive Windowing) for variable-rate drift
///
/// Reference: [Bifet & Gavalda 2007] "Learning from Time-Changing Data
/// with Adaptive Windowing"
/// - Automatically adjusts window size
/// - Detects both abrupt and gradual drift
pub struct ADWIN {
    /// Confidence parameter (smaller = more sensitive)
    delta: f64,
    /// Maximum window size
    max_buckets: usize,
    /// Bucket structure for efficient computation
    buckets: Vec<Bucket>,
}
```

### 2.2 Drift-Triggered Retraining

```rust
/// Automatic retraining orchestrator
pub struct RetrainOrchestrator<M: OnlineLearner, D: DriftDetector> {
    /// Current model
    model: M,
    /// Drift detector
    detector: D,
    /// Data buffer for retraining
    buffer: CorpusBuffer,
    /// Retraining configuration
    config: RetrainConfig,
}

#[derive(Debug, Clone)]
pub struct RetrainConfig {
    /// Minimum samples before retraining
    pub min_samples: usize,
    /// Maximum buffer size
    pub max_buffer_size: usize,
    /// Use curriculum learning during retrain
    pub curriculum_learning: bool,
    /// Distill from previous model
    pub knowledge_distillation: bool,
    /// Save checkpoint after retrain
    pub save_checkpoint: bool,
}

impl<M: OnlineLearner, D: DriftDetector> RetrainOrchestrator<M, D> {
    /// Process new sample and handle drift
    pub fn observe(&mut self, x: &Matrix, y: &Vector, pred: &Vector) -> Result<ObserveResult> {
        // Check prediction correctness
        let errors = compute_errors(y, pred);

        for error in errors {
            self.detector.add_element(error);
        }

        // Buffer data for potential retraining
        self.buffer.add(x.clone(), y.clone());

        match self.detector.detected_change() {
            DriftStatus::Stable => {
                // Incremental update only
                if self.config.incremental_updates {
                    self.model.partial_fit(x, y, None)?;
                }
                Ok(ObserveResult::Stable)
            }
            DriftStatus::Warning => {
                // Start aggressive buffering
                Ok(ObserveResult::Warning)
            }
            DriftStatus::Drift => {
                // Trigger full retraining
                self.retrain()?;
                self.detector.reset();
                Ok(ObserveResult::Retrained)
            }
        }
    }
}
```

---

## 3. Corpus Management

### 3.1 Corpus Buffer

```rust
/// Efficient corpus storage with deduplication
///
/// Reference: [Settles 2009] "Active Learning Literature Survey"
/// - Importance sampling for corpus construction
/// - Hash-based deduplication to avoid redundancy
pub struct CorpusBuffer {
    /// Feature-label pairs
    samples: Vec<(Matrix, Vector)>,
    /// Hash set for deduplication
    seen_hashes: HashSet<u64>,
    /// Maximum buffer size
    max_size: usize,
    /// Eviction policy
    policy: EvictionPolicy,
    /// Sample weights for importance sampling
    weights: Vec<f64>,
}

#[derive(Debug, Clone, Copy)]
pub enum EvictionPolicy {
    /// First-in-first-out
    FIFO,
    /// Remove lowest-weight samples
    ImportanceWeighted,
    /// Reservoir sampling for uniform distribution
    /// Reference: [Vitter 1985] "Random Sampling with a Reservoir"
    Reservoir,
    /// Keep diverse samples (maximize coverage)
    DiversitySampling,
}

impl CorpusBuffer {
    /// Add sample with deduplication
    pub fn add(&mut self, x: Matrix, y: Vector) -> bool {
        let hash = self.compute_hash(&x, &y);

        if self.seen_hashes.contains(&hash) {
            return false; // Duplicate
        }

        if self.samples.len() >= self.max_size {
            self.evict_one();
        }

        self.samples.push((x, y));
        self.seen_hashes.insert(hash);
        true
    }

    /// Export corpus for training
    pub fn to_dataset(&self) -> (Matrix, Vector) {
        // Concatenate all samples
        let x = Matrix::vstack(&self.samples.iter().map(|(x, _)| x).collect::<Vec<_>>())?;
        let y = Vector::concat(&self.samples.iter().map(|(_, y)| y).collect::<Vec<_>>())?;
        (x, y)
    }
}
```

### 3.2 Multi-Source Corpus Merger

```rust
/// Merge multiple data sources with configurable weighting
///
/// Used by ruchy Oracle to combine:
/// - Synthetic data (12,000 samples)
/// - Hand-crafted corpus
/// - Examples corpus
/// - Production corpus
pub struct CorpusMerger {
    sources: Vec<CorpusSource>,
    dedup_strategy: DedupStrategy,
    shuffle_seed: Option<u64>,
}

pub struct CorpusSource {
    /// Source name for provenance
    pub name: String,
    /// Data path or buffer
    pub data: CorpusData,
    /// Weight multiplier (1.0 = normal)
    pub weight: f64,
    /// Priority (higher = prefer in dedup)
    pub priority: u8,
}

impl CorpusMerger {
    /// Merge all sources into unified dataset
    pub fn merge(&self) -> Result<(Matrix, Vector, CorpusProvenance)> {
        let mut all_samples = Vec::new();
        let mut provenance = CorpusProvenance::new();

        for source in &self.sources {
            let (x, y) = source.data.load()?;
            let n = x.nrows();

            // Apply source weight via oversampling
            let effective_n = (n as f64 * source.weight) as usize;
            all_samples.extend(self.sample_with_weight(&x, &y, effective_n)?);

            provenance.add_source(&source.name, n, effective_n);
        }

        // Deduplicate
        let deduped = self.deduplicate(all_samples)?;
        provenance.set_final_size(deduped.len());

        // Shuffle
        let shuffled = self.shuffle(deduped)?;

        Ok((shuffled.0, shuffled.1, provenance))
    }
}
```

---

## 4. Curriculum Learning

### 4.1 Difficulty Scheduler

```rust
/// Curriculum learning: train on easy samples first
///
/// Reference: [Bengio et al. 2009] "Curriculum Learning"
/// - Ordering samples from easy to hard improves convergence
/// - 25% faster training, better generalization
pub trait CurriculumScheduler {
    /// Score sample difficulty (lower = easier)
    fn difficulty(&self, x: &Vector, y: f64) -> f64;

    /// Get next batch according to curriculum
    fn next_batch(&mut self, batch_size: usize) -> (Matrix, Vector);

    /// Current curriculum stage (0.0 = easiest, 1.0 = all data)
    fn stage(&self) -> f64;

    /// Advance to next stage
    fn advance(&mut self);
}

/// Self-paced curriculum learning
///
/// Reference: [Kumar et al. 2010] "Self-Paced Learning for Latent
/// Variable Models"
/// - Automatically determines sample difficulty from loss
/// - Adapts pace based on model performance
pub struct SelfPacedCurriculum {
    /// All training samples with cached difficulties
    samples: Vec<(Vector, f64, f64)>, // (x, y, difficulty)
    /// Current difficulty threshold
    threshold: f64,
    /// Threshold growth rate
    growth_rate: f64,
    /// Model for computing loss-based difficulty
    loss_fn: Box<dyn Fn(&Vector, f64, f64) -> f64>,
}

impl SelfPacedCurriculum {
    /// Update difficulties based on current model predictions
    pub fn update_difficulties(&mut self, model: &impl Estimator) {
        for (x, y, diff) in &mut self.samples {
            let pred = model.predict(&x.to_matrix())?[0];
            *diff = (self.loss_fn)(x, *y, pred);
        }
        // Sort by difficulty
        self.samples.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());
    }
}
```

---

## 5. Knowledge Distillation

### 5.1 Teacher-Student Framework

```rust
/// Knowledge distillation for model compression
///
/// Reference: [Hinton et al. 2015] "Distilling the Knowledge in a
/// Neural Network"
/// - Transfer knowledge from large teacher to small student
/// - Use soft targets (probabilities) not hard labels
pub struct Distiller<T: Estimator, S: OnlineLearner> {
    /// Large, accurate teacher model
    teacher: T,
    /// Small, efficient student model
    student: S,
    /// Temperature for softening probabilities
    temperature: f64,
    /// Weight for distillation loss vs hard label loss
    alpha: f64,
}

impl<T: Estimator, S: OnlineLearner> Distiller<T, S> {
    /// Train student on teacher's soft targets
    ///
    /// Loss = α * KL(student || teacher/T) + (1-α) * CE(student, labels)
    pub fn distill(&mut self, x: &Matrix, y: &Vector) -> Result<f64> {
        // Get teacher's soft predictions
        let teacher_logits = self.teacher.predict_logits(x)?;
        let soft_targets = softmax_temperature(&teacher_logits, self.temperature);

        // Student forward pass
        let student_logits = self.student.predict_logits(x)?;
        let student_soft = softmax_temperature(&student_logits, self.temperature);

        // Combined loss
        let distill_loss = kl_divergence(&student_soft, &soft_targets);
        let hard_loss = cross_entropy(&student_logits, y);
        let total_loss = self.alpha * distill_loss + (1.0 - self.alpha) * hard_loss;

        // Update student
        self.student.backward(total_loss)?;

        Ok(total_loss)
    }
}
```

---

## 6. CITL Integration

### 6.1 Compiler Feedback Hooks

```rust
/// Compiler-in-the-Loop training integration
///
/// Connects to depyler/ruchy compilation pipeline to:
/// 1. Collect compilation outcomes (success/error)
/// 2. Extract error patterns for classifier training
/// 3. Trigger retraining on drift detection
pub trait CITLFeedback {
    /// Called after each compilation attempt
    fn on_compile_result(&mut self, result: &CompileResult);

    /// Get accumulated training samples
    fn drain_samples(&mut self) -> Vec<CITLSample>;

    /// Check if retraining is recommended
    fn should_retrain(&self) -> bool;
}

#[derive(Debug, Clone)]
pub struct CITLSample {
    /// Error message features
    pub error_features: Vector,
    /// Error category label
    pub category: u8,
    /// Confidence in label (for semi-supervised)
    pub confidence: f64,
    /// Source (synthetic, production, etc.)
    pub source: SampleSource,
}

/// CITL collector for Oracle training
pub struct OracleCITLCollector {
    /// Feature extractor
    extractor: ErrorFeatureExtractor,
    /// Drift detector
    drift: DDM,
    /// Sample buffer
    buffer: CorpusBuffer,
    /// Minimum samples for training batch
    min_batch_size: usize,
}

impl CITLFeedback for OracleCITLCollector {
    fn on_compile_result(&mut self, result: &CompileResult) {
        if let Some(error) = &result.error {
            // Extract 73-dimensional feature vector
            let features = self.extractor.extract(error);

            // Classify with current model
            let (category, confidence) = self.classify(&features);

            // Check if prediction was correct (if we have feedback)
            if let Some(actual) = result.actual_category {
                self.drift.add_element(category != actual);
            }

            // Buffer for training
            self.buffer.add(
                features.to_matrix(),
                Vector::from_slice(&[category as f64]),
            );
        }
    }

    fn should_retrain(&self) -> bool {
        self.drift.detected_change() == DriftStatus::Drift
            && self.buffer.len() >= self.min_batch_size
    }
}
```

---

## 7. API Summary

### 7.1 Module Structure

```
src/online/
├── mod.rs           # OnlineLearner trait, core types
├── drift.rs         # DDM, PageHinkley, ADWIN
├── corpus.rs        # CorpusBuffer, CorpusMerger
├── curriculum.rs    # CurriculumScheduler, SelfPaced
├── distillation.rs  # Distiller, temperature scaling
├── citl.rs          # CITLFeedback, OracleCITLCollector
└── orchestrator.rs  # RetrainOrchestrator
```

### 7.2 Integration with Existing Modules

| Existing Module | Integration Point |
|----------------|-------------------|
| `linear_model` | Implement `OnlineLearner` for SGD |
| `classification` | Implement `OnlineLearner` for LogisticRegression |
| `cluster` | Mini-batch K-Means with `partial_fit` |
| `tree` | Mondrian Forest for online trees |
| `metrics/drift` | Extend with DDM, ADWIN |
| `citl` | Add `CITLFeedback` trait |

---

## 8. References

### 8.1 Peer-Reviewed Citations

1. **[Bottou 2010]** Bottou, L. "Large-Scale Machine Learning with Stochastic Gradient Descent." *Proceedings of COMPSTAT'2010*, pp. 177-186. Springer.
   - Foundation for online SGD convergence guarantees

2. **[Crammer et al. 2006]** Crammer, K., Dekel, O., Keshet, J., Shalev-Shwartz, S., & Singer, Y. "Online Passive-Aggressive Algorithms." *Journal of Machine Learning Research*, 7:551-585.
   - Margin-based online learning with theoretical guarantees

3. **[Gama et al. 2004]** Gama, J., Medas, P., Castillo, G., & Rodrigues, P. "Learning with Drift Detection." *Brazilian Symposium on Artificial Intelligence*, pp. 286-295. Springer.
   - DDM algorithm for concept drift detection

4. **[Bifet & Gavalda 2007]** Bifet, A., & Gavalda, R. "Learning from Time-Changing Data with Adaptive Windowing." *SIAM International Conference on Data Mining*, pp. 443-448.
   - ADWIN algorithm for variable-rate drift

5. **[Bengio et al. 2009]** Bengio, Y., Louradour, J., Collobert, R., & Weston, J. "Curriculum Learning." *Proceedings of the 26th International Conference on Machine Learning*, pp. 41-48. ACM.
   - Theoretical foundation for curriculum learning

6. **[Kumar et al. 2010]** Kumar, M.P., Packer, B., & Koller, D. "Self-Paced Learning for Latent Variable Models." *Advances in Neural Information Processing Systems*, 23:1189-1197.
   - Self-paced curriculum with automatic difficulty estimation

7. **[Hinton et al. 2015]** Hinton, G., Vinyals, O., & Dean, J. "Distilling the Knowledge in a Neural Network." *arXiv preprint arXiv:1503.02531*.
   - Knowledge distillation via soft targets

8. **[Sculley 2010]** Sculley, D. "Web-Scale K-Means Clustering." *Proceedings of the 19th International Conference on World Wide Web*, pp. 1177-1178. ACM.
   - Mini-batch K-Means for online clustering

9. **[Duchi et al. 2011]** Duchi, J., Hazan, E., & Singer, Y. "Adaptive Subgradient Methods for Online Learning and Stochastic Optimization." *Journal of Machine Learning Research*, 12:2121-2159.
   - AdaGrad optimizer for adaptive learning rates

10. **[Lakshminarayanan et al. 2014]** Lakshminarayanan, B., Roy, D.M., & Teh, Y.W. "Mondrian Forests: Efficient Online Random Forests." *Advances in Neural Information Processing Systems*, 27:3140-3148.
    - Online random forests with theoretical guarantees

### 8.2 Additional References

- **[Page 1954]** Page, E.S. "Continuous Inspection Schemes." *Biometrika*, 41(1/2):100-115.
- **[Vitter 1985]** Vitter, J.S. "Random Sampling with a Reservoir." *ACM Transactions on Mathematical Software*, 11(1):37-57.
- **[Settles 2009]** Settles, B. "Active Learning Literature Survey." *Computer Sciences Technical Report 1648*, University of Wisconsin-Madison.

---

## 9. Toyota Way Compliance

| Principle | Implementation |
|-----------|---------------|
| **Kaizen** | Continuous model improvement via online learning |
| **Jidoka** | Drift detection stops bad predictions automatically |
| **Just-in-Time** | Retrain only when drift detected, not on schedule |
| **Muda Elimination** | Deduplication avoids redundant training data |
| **Genchi Genbutsu** | CITL feedback from actual compilation results |

---

## 10. Open Questions for Review

1. **Mondrian Forest complexity**: Is online random forest worth the implementation effort vs. simpler ensemble updates?

2. **Drift detector selection**: Should we default to DDM (simpler) or ADWIN (more robust)?

3. **CITL feedback latency**: How to handle async compilation results in streaming scenarios?

4. **Corpus size limits**: What's the right max buffer size for ruchy Oracle (memory vs. accuracy tradeoff)?

5. **Distillation temperature**: Should temperature be fixed or adaptive based on teacher confidence?
