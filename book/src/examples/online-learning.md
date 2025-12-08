# Case Study: Online Learning and Dynamic Retraining

This case study demonstrates aprender's online learning infrastructure for streaming data, concept drift detection, and automatic model retraining.

## Overview

Run the complete example:

```bash
cargo run --example online_learning
```

## Part 1: Online Linear Regression

Incremental training on streaming data without storing the full dataset:

```rust
use aprender::online::{
    OnlineLearner, OnlineLearnerConfig, OnlineLinearRegression,
    LearningRateDecay,
};

// Configure with inverse sqrt learning rate decay
let config = OnlineLearnerConfig {
    learning_rate: 0.01,
    decay: LearningRateDecay::InverseSqrt,
    l2_reg: 0.001,
    ..Default::default()
};

let mut model = OnlineLinearRegression::with_config(2, config);

// Simulate streaming data: y = 2*x1 + 3*x2 + 1
let samples = vec![
    (vec![1.0, 0.0], 3.0),   // 2*1 + 3*0 + 1 = 3
    (vec![0.0, 1.0], 4.0),   // 2*0 + 3*1 + 1 = 4
    (vec![1.0, 1.0], 6.0),   // 2*1 + 3*1 + 1 = 6
];

// Train incrementally
for (x, y) in &samples {
    let loss = model.partial_fit(x, &[*y], None)?;
    println!("Loss: {:.4}", loss);
}

// Model state
println!("Weights: {:?}", model.weights());
println!("Bias: {:.4}", model.bias());
println!("Samples seen: {}", model.n_samples_seen());
println!("Current LR: {:.6}", model.current_learning_rate());
```

**Output:**
```
Loss: 9.0000
Loss: 15.7609
Loss: 34.3466
```

## Part 2: Online Logistic Regression

Binary classification with streaming updates:

```rust
use aprender::online::{
    OnlineLearnerConfig, OnlineLogisticRegression, LearningRateDecay,
};

let config = OnlineLearnerConfig {
    learning_rate: 0.5,
    decay: LearningRateDecay::Constant,
    ..Default::default()
};

let mut model = OnlineLogisticRegression::with_config(2, config);

// XOR-like classification
let samples = vec![
    (vec![0.0, 0.0], 0.0),
    (vec![1.0, 1.0], 1.0),
    (vec![0.5, 0.5], 1.0),
    (vec![0.1, 0.1], 0.0),
];

// Train multiple passes
for _ in 0..100 {
    for (x, y) in &samples {
        model.partial_fit(x, &[*y], None)?;
    }
}

// Predict probabilities
for (x, _) in &samples {
    let prob = model.predict_proba_one(x)?;
    let class = if prob > 0.5 { 1 } else { 0 };
    println!("P(y=1) = {:.3}, class = {}", prob, class);
}
```

## Part 3: Drift Detection

### DDM for Sudden Drift

DDM (Drift Detection Method) monitors error rate statistics:

```rust
use aprender::online::drift::{DDM, DriftDetector};

let mut ddm = DDM::new();

// Simulate good predictions
for _ in 0..50 {
    ddm.add_element(false);  // correct prediction
}
println!("Status: {:?}", ddm.detected_change());  // Stable

// Simulate concept drift (many errors)
for _ in 0..50 {
    ddm.add_element(true);  // wrong prediction
}
let stats = ddm.stats();
println!("Status: {:?}", stats.status);      // Drift
println!("Error rate: {:.2}%", stats.error_rate * 100.0);
```

### ADWIN for Gradual/Sudden Drift (Recommended)

ADWIN uses adaptive windowing to detect both types of drift:

```rust
use aprender::online::drift::{ADWIN, DriftDetector};

let mut adwin = ADWIN::with_delta(0.1);  // Sensitivity parameter

// Low error period
for _ in 0..100 {
    adwin.add_element(false);
}
println!("Window size: {}", adwin.window_size());  // 100
println!("Mean error: {:.3}", adwin.mean());       // 0.000

// Concept drift occurs
for _ in 0..100 {
    adwin.add_element(true);
}
println!("Window size: {}", adwin.window_size());  // Adjusted
println!("Mean error: {:.3}", adwin.mean());       // ~0.500
```

### Factory for Easy Creation

```rust
use aprender::online::drift::DriftDetectorFactory;

// Create recommended detector (ADWIN)
let detector = DriftDetectorFactory::recommended();
```

## Part 4: Corpus Management

Memory-efficient sample storage with deduplication:

```rust
use aprender::online::corpus::{
    CorpusBuffer, CorpusBufferConfig, EvictionPolicy,
    Sample, SampleSource,
};

let config = CorpusBufferConfig {
    max_size: 5,
    policy: EvictionPolicy::Reservoir,  // Random sampling
    deduplicate: true,                   // Hash-based dedup
    seed: Some(42),
};

let mut buffer = CorpusBuffer::with_config(config);

// Add samples with source tracking
for i in 0..10 {
    let sample = Sample::with_source(
        vec![i as f64, (i * 2) as f64],
        vec![(i * 3) as f64],
        if i < 5 { SampleSource::Synthetic }
        else { SampleSource::Production },
    );
    let added = buffer.add(sample);
    println!("Sample {}: added={}, size={}", i, added, buffer.len());
}

// Duplicate is rejected
let dup = Sample::new(vec![0.0, 0.0], vec![0.0]);
assert!(!buffer.add(dup));  // false - duplicate

// Export to dataset
let (features, targets, n_samples, n_features) = buffer.to_dataset();
println!("Samples: {}, Features: {}", n_samples, n_features);

// Filter by source
let production = buffer.samples_by_source(&SampleSource::Production);
println!("Production samples: {}", production.len());
```

**Eviction Policies:**

| Policy | Behavior |
|--------|----------|
| `FIFO` | Remove oldest when full |
| `Reservoir` | Random sampling, maintains distribution |
| `ImportanceWeighted` | Keep high-loss samples |
| `DiversitySampling` | Maximize feature coverage |

## Part 5: Curriculum Learning

Progressive training from easy to hard samples:

```rust
use aprender::online::curriculum::{
    LinearCurriculum, CurriculumScheduler,
    FeatureNormScorer, DifficultyScorer,
};

// 5-stage linear curriculum
let mut curriculum = LinearCurriculum::new(5);

println!("Stage | Progress | Threshold | Complete");
for _ in 0..7 {
    println!(
        "{:>5} | {:>7.0}% | {:>9.2} | {:>8}",
        curriculum.stage() as u32,
        curriculum.stage() * 100.0,
        curriculum.current_threshold(),
        curriculum.is_complete()
    );
    curriculum.advance();
}

// Difficulty scoring by feature norm
let scorer = FeatureNormScorer::new();

let samples = vec![
    vec![0.5, 0.5],  // Easy: small norm
    vec![2.0, 2.0],  // Medium
    vec![5.0, 5.0],  // Hard: large norm
];

for sample in &samples {
    let difficulty = scorer.score(sample, 0.0);
    let level = if difficulty < 2.0 { "Easy" }
                else if difficulty < 4.0 { "Medium" }
                else { "Hard" };
    println!("{:?} -> {:.3} ({})", sample, difficulty, level);
}
```

**Output:**
```
Stage | Progress | Threshold | Complete
    0 |       0% |      0.00 |    false
    1 |      20% |      0.20 |    false
    2 |      40% |      0.40 |    false
    3 |      60% |      0.60 |    false
    4 |      80% |      0.80 |    false
    5 |     100% |      1.00 |     true
```

## Part 6: Knowledge Distillation

Transfer knowledge from teacher to student model:

```rust
use aprender::online::distillation::{
    softmax_temperature, DEFAULT_TEMPERATURE,
    DistillationConfig, DistillationLoss,
};

let teacher_logits = vec![1.0, 3.0, 0.5];

// Temperature scaling reveals "dark knowledge"
let hard = softmax_temperature(&teacher_logits, 1.0);
println!("T=1:  [{:.3}, {:.3}, {:.3}]", hard[0], hard[1], hard[2]);

let soft = softmax_temperature(&teacher_logits, DEFAULT_TEMPERATURE);  // T=3
println!("T=3:  [{:.3}, {:.3}, {:.3}]", soft[0], soft[1], soft[2]);

let very_soft = softmax_temperature(&teacher_logits, 10.0);
println!("T=10: [{:.3}, {:.3}, {:.3}]", very_soft[0], very_soft[1], very_soft[2]);

// Distillation loss: combined KL divergence + cross-entropy
let config = DistillationConfig {
    temperature: DEFAULT_TEMPERATURE,
    alpha: 0.7,  // 70% distillation, 30% hard labels
    learning_rate: 0.01,
    l2_reg: 0.0,
};
let loss_fn = DistillationLoss::with_config(config);

let student_logits = vec![0.5, 2.0, 0.8];
let hard_labels = vec![0.0, 1.0, 0.0];

let loss = loss_fn.compute(&student_logits, &teacher_logits, &hard_labels)?;
println!("Distillation loss: {:.4}", loss);
```

**Output:**
```
T=1:  [0.111, 0.821, 0.067]
T=3:  [0.264, 0.513, 0.223]
T=10: [0.315, 0.385, 0.300]
Distillation loss: 0.2272
```

## Part 7: RetrainOrchestrator

Automated pipeline combining all components:

```rust
use aprender::online::{
    OnlineLinearRegression,
    orchestrator::{OrchestratorBuilder, ObserveResult},
};

let model = OnlineLinearRegression::new(2);
let mut orchestrator = OrchestratorBuilder::new(model, 2)
    .min_samples(10)            // Min samples before retrain
    .max_buffer_size(100)       // Corpus capacity
    .incremental_updates(true)  // Use partial_fit
    .curriculum_learning(true)  // Easy-to-hard ordering
    .curriculum_stages(3)       // 3 difficulty levels
    .learning_rate(0.01)
    .adwin_delta(0.1)           // Drift sensitivity
    .build();

println!("Config:");
println!("  Min samples: {}", orchestrator.config().min_samples);
println!("  Max buffer: {}", orchestrator.config().max_buffer_size);

// Process streaming predictions
for i in 0..15 {
    let features = vec![i as f64, (i * 2) as f64];
    let target = if i < 5 { vec![(i * 3) as f64] } else { vec![1.0] };
    let prediction = if i < 5 { vec![(i * 3) as f64] } else { vec![0.0] };

    let result = orchestrator.observe(&features, &target, &prediction)?;

    match result {
        ObserveResult::Stable => {}
        ObserveResult::Warning => println!("Step {}: Warning", i + 1),
        ObserveResult::Retrained => println!("Step {}: Retrained!", i + 1),
    }
}

// Check statistics
let stats = orchestrator.stats();
println!("Samples observed: {}", stats.samples_observed);
println!("Retrain count: {}", stats.retrain_count);
println!("Buffer size: {}", stats.buffer_size);
println!("Drift status: {:?}", stats.drift_status);
```

## Complete Example Output

```
=== Online Learning and Dynamic Retraining ===

--- Part 1: Online Linear Regression ---
Training incrementally on streaming data (y = 2*x1 + 3*x2 + 1)...
Sample       x1       x2          y         Loss
--------------------------------------------------
     1      1.0      0.0        3.0       9.0000
     2      0.0      1.0        4.0      15.7609
     3      1.0      1.0        6.0      34.3466

--- Part 2: Online Logistic Regression ---
Predictions after training:
      x1       x2     P(y=1)        Class
---------------------------------------------
     0.0      0.0      0.031            0
     1.0      1.0      1.000            1

--- Part 3: Drift Detection ---
DDM (for sudden drift):
  After 50 correct: Stable
  After 50 errors: Drift

ADWIN (for gradual/sudden drift - RECOMMENDED):
  Window size: 100
  Mean error: 0.000

--- Part 4: Corpus Management ---
Duplicate sample: added=false
Synthetic: 3, Production: 2

--- Part 5: Curriculum Learning ---
[0.5, 0.5] -> 0.707 (Easy)
[5.0, 5.0] -> 7.071 (Hard)

--- Part 6: Knowledge Distillation ---
Hard targets (T=1): [0.111, 0.821, 0.067]
Soft targets (T=3): [0.264, 0.513, 0.223]
Distillation loss: 0.2272

--- Part 7: RetrainOrchestrator ---
Samples observed: 15
Retrain count: 0
Drift status: Stable

=== Online Learning Complete! ===
```

## Key Takeaways

1. **Use `partial_fit()`** for incremental updates instead of full retraining
2. **ADWIN is the recommended** drift detector for most applications
3. **Temperature T=3** is the default for knowledge distillation
4. **Reservoir sampling** maintains representative samples in bounded memory
5. **Curriculum learning** improves convergence by ordering easy-to-hard
6. **RetrainOrchestrator** combines all components into an automated pipeline

## References

- [Gama et al., 2004] DDM drift detection
- [Bifet & Gavalda, 2007] ADWIN adaptive windowing
- [Bengio et al., 2009] Curriculum learning
- [Hinton et al., 2015] Knowledge distillation
