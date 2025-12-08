# Online Learning Theory

Online learning is a machine learning paradigm where models update incrementally as new data arrives, rather than requiring full retraining on the entire dataset. This is essential for streaming applications, real-time systems, and scenarios where data distribution changes over time.

## Core Concepts

### Batch vs Online Learning

**Batch Learning:**
- Train on entire dataset at once
- O(n) memory for n samples
- Requires full retraining for updates
- Suitable for static datasets

**Online Learning:**
- Update model one sample at a time
- O(1) memory per update
- Incremental updates without retraining
- Suitable for streaming data

### The Regret Framework

Online learning is analyzed using *regret*: the difference between the learner's cumulative loss and the best fixed hypothesis in hindsight.

```
Regret_T = Σ_{t=1}^T l(ŷ_t, y_t) - min_h Σ_{t=1}^T l(h(x_t), y_t)
```

A good online algorithm achieves sublinear regret: O(√T) for convex losses.

## Online Gradient Descent

The fundamental online learning algorithm:

```
w_{t+1} = w_t - η_t ∇l(w_t; x_t, y_t)
```

### Learning Rate Schedules

| Schedule | Formula | Use Case |
|----------|---------|----------|
| Constant | η_t = η_0 | Stationary distributions |
| Inverse | η_t = η_0 / t | Convex, bounded gradients |
| Inverse Sqrt | η_t = η_0 / √t | Strongly convex losses |
| AdaGrad | η_{t,i} = η_0 / √(Σ g²_{s,i}) | Sparse features |

### Implementation in Aprender

```rust
use aprender::online::{
    OnlineLearner, OnlineLearnerConfig, OnlineLinearRegression,
    LearningRateDecay,
};

// Configure online learner
let config = OnlineLearnerConfig {
    learning_rate: 0.01,
    decay: LearningRateDecay::InverseSqrt,
    l2_reg: 0.001,
    ..Default::default()
};

let mut model = OnlineLinearRegression::with_config(2, config);

// Incremental updates
for (x, y) in streaming_data {
    let loss = model.partial_fit(&x, &[y], None)?;
    println!("Loss: {:.4}", loss);
}
```

## Concept Drift

Real-world data distributions change over time. **Concept drift** occurs when the relationship P(Y|X) changes, degrading model performance.

### Types of Drift

1. **Sudden Drift**: Abrupt distribution change (e.g., system upgrade)
2. **Gradual Drift**: Slow transition between concepts
3. **Incremental Drift**: Continuous small changes
4. **Recurring Drift**: Cyclic patterns (e.g., seasonality)

### Drift Detection Methods

#### DDM (Drift Detection Method)

Monitors error rate statistics [Gama et al., 2004]:

```rust
use aprender::online::drift::{DDM, DriftDetector, DriftStatus};

let mut ddm = DDM::new();

for prediction_error in errors {
    ddm.add_element(prediction_error);

    match ddm.detected_change() {
        DriftStatus::Drift => println!("Drift detected! Retrain model."),
        DriftStatus::Warning => println!("Warning: potential drift"),
        DriftStatus::Stable => {}
    }
}
```

#### ADWIN (Adaptive Windowing)

Maintains adaptive window size [Bifet & Gavalda, 2007]:

- Automatically adjusts window to recent relevant data
- Detects both sudden and gradual drift
- **Recommended default** for most applications

```rust
use aprender::online::drift::{ADWIN, DriftDetector};

let mut adwin = ADWIN::with_delta(0.002);  // 99.8% confidence

// Add observations
adwin.add_element(true);  // error
adwin.add_element(false); // correct

println!("Window size: {}", adwin.window_size());
println!("Mean error: {:.3}", adwin.mean());
```

## Curriculum Learning

Training on samples ordered by difficulty, from easy to hard [Bengio et al., 2009]:

### Benefits

1. Faster convergence
2. Better generalization
3. Avoids local minima from hard examples early
4. Mimics human learning progression

### Implementation

```rust
use aprender::online::curriculum::{
    LinearCurriculum, CurriculumScheduler,
    FeatureNormScorer, DifficultyScorer,
};

// Linear difficulty progression over 5 stages
let mut curriculum = LinearCurriculum::new(5);

// Score samples by feature norm (larger = harder)
let scorer = FeatureNormScorer::new();

for sample in &samples {
    let difficulty = scorer.score(&sample.features, 0.0);

    // Only train on samples below current threshold
    if difficulty <= curriculum.current_threshold() {
        model.partial_fit(&sample.features, &sample.target, None)?;
    }
}

// Advance to next curriculum stage
curriculum.advance();
```

## Knowledge Distillation

Transfer knowledge from a complex "teacher" model to a simpler "student" model [Hinton et al., 2015].

### Temperature Scaling

Softmax with temperature T reveals "dark knowledge":

```
p_i = exp(z_i/T) / Σ_j exp(z_j/T)
```

- T=1: Standard softmax (hard targets)
- T>1: Softer probability distribution
- **T=3**: Recommended default for distillation

```rust
use aprender::online::distillation::{
    softmax_temperature, DEFAULT_TEMPERATURE,
};

let teacher_logits = vec![1.0, 3.0, 0.5];

// Hard targets (T=1)
let hard = softmax_temperature(&teacher_logits, 1.0);
// [0.111, 0.821, 0.067]

// Soft targets (T=3, default)
let soft = softmax_temperature(&teacher_logits, DEFAULT_TEMPERATURE);
// [0.264, 0.513, 0.223]
```

### Distillation Loss

Combined loss with hard labels and soft targets:

```
L = α * KL(soft_student || soft_teacher) + (1-α) * CE(student, labels)
```

```rust
use aprender::online::distillation::{DistillationConfig, DistillationLoss};

let config = DistillationConfig {
    temperature: 3.0,
    alpha: 0.7,  // 70% distillation, 30% hard labels
    learning_rate: 0.01,
    l2_reg: 0.0,
};

let loss = DistillationLoss::with_config(config);
let distill_loss = loss.compute(&student_logits, &teacher_logits, &hard_labels)?;
```

## Corpus Management

Managing training data in memory-constrained streaming scenarios.

### Eviction Policies

| Policy | Description | Use Case |
|--------|-------------|----------|
| FIFO | Remove oldest samples | Simple, predictable |
| Reservoir | Random sampling, uniform distribution | Statistical sampling |
| Importance | Keep high-loss samples | Hard example mining |
| Diversity | Maximize feature space coverage | Avoid redundancy |

### Sample Deduplication

Hash-based deduplication prevents redundant samples:

```rust
use aprender::online::corpus::{CorpusBuffer, CorpusBufferConfig, EvictionPolicy};

let config = CorpusBufferConfig {
    max_size: 1000,
    policy: EvictionPolicy::Reservoir,
    deduplicate: true,  // Hash-based deduplication
    seed: Some(42),
};

let mut buffer = CorpusBuffer::with_config(config);
```

## RetrainOrchestrator

Automated pipeline combining all components:

```rust
use aprender::online::{
    OnlineLinearRegression,
    orchestrator::OrchestratorBuilder,
};

let model = OnlineLinearRegression::new(n_features);
let mut orchestrator = OrchestratorBuilder::new(model, n_features)
    .min_samples(100)           // Min samples before retraining
    .max_buffer_size(10000)     // Corpus capacity
    .incremental_updates(true)  // Enable partial_fit
    .curriculum_learning(true)  // Easy-to-hard ordering
    .curriculum_stages(5)       // 5 difficulty levels
    .adwin_delta(0.002)         // Drift sensitivity
    .build();

// Process streaming predictions
for (features, target, prediction) in stream {
    match orchestrator.observe(&features, &target, &prediction)? {
        ObserveResult::Stable => {}
        ObserveResult::Warning => println!("Potential drift detected"),
        ObserveResult::Retrained => println!("Model retrained"),
    }
}
```

## Mathematical Foundations

### Convergence Guarantees

For convex loss functions with bounded gradients ||∇l|| ≤ G:

**SGD with η_t = η/√t:**
```
E[Regret_T] ≤ O(√T)
```

**AdaGrad:**
```
Regret_T ≤ O(√T) with adaptive per-coordinate rates
```

### ADWIN Theoretical Properties

ADWIN guarantees [Bifet & Gavalda, 2007]:
1. False positive rate bounded by δ
2. Window contains only data from current distribution
3. Memory: O(log(W)/ε²) where W is window size

## References

1. Gama, J., et al. (2004). "Learning with drift detection." SBIA 2004.
2. Bifet, A., & Gavalda, R. (2007). "Learning from time-changing data with adaptive windowing." SDM 2007.
3. Bengio, Y., et al. (2009). "Curriculum learning." ICML 2009.
4. Hinton, G., et al. (2015). "Distilling the knowledge in a neural network." NIPS 2014 Workshop.
5. Duchi, J., et al. (2011). "Adaptive subgradient methods for online learning." JMLR.
6. Shalev-Shwartz, S. (2012). "Online learning and online convex optimization." Foundations and Trends in ML.
7. Hazan, E. (2016). "Introduction to online convex optimization." Foundations and Trends in Optimization.
8. Lu, J., et al. (2018). "Learning under concept drift: A review." IEEE TKDE.
9. Wang, H., & Abraham, Z. (2015). "Concept drift detection for streaming data." IJCNN 2015.
10. Gomes, H.M., et al. (2017). "A survey on ensemble learning for data stream classification." ACM Computing Surveys.
