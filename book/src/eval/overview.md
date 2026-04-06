# Model Evaluation Framework (APR-073)

The Model Evaluation Framework provides standardized metrics, model comparison, and drift detection following **Toyota
Way** principles:

- **Jidoka** (Automation with Human Touch): Automated drift detection that signals when intervention is needed
- **Mieruka** (Visual Control): Clear, structured reports for metrics and comparisons
- **Andon** (Alert System): Callbacks that trigger retraining when drift is detected

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          entrenar::eval Module                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────┐  ┌──────────────────┐  ┌───────────────────────┐   │
│  │  classification │  │    evaluator     │  │        drift          │   │
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
│                    │    retrain Module     │  (Andon Loop)              │
│                    │  - AutoRetrainer      │                            │
│                    │  - RetrainPolicy      │                            │
│                    └───────────────────────┘                            │
└─────────────────────────────────────────────────────────────────────────┘
```

## Quick Start

```rust
use entrenar::eval::{
    ModelEvaluator, EvalConfig, Metric, Average,
    DriftDetector, DriftTest, AutoRetrainer, RetrainConfig,
};

// 1. Evaluate a classifier
let evaluator = ModelEvaluator::new(EvalConfig {
    metrics: vec![
        Metric::Accuracy,
        Metric::F1(Average::Weighted),
    ],
    ..Default::default()
});

let result = evaluator.evaluate_classification("my_model", &predictions, &labels)?;
println!("Accuracy: {:.2}%", result.get_score(Metric::Accuracy).unwrap() * 100.0);

// 2. Set up drift detection
let mut detector = DriftDetector::new(vec![
    DriftTest::KS { threshold: 0.05 },
    DriftTest::PSI { threshold: 0.1 },
]);
detector.set_baseline(&training_data);

// 3. Configure auto-retraining (Andon Loop)
let mut retrainer = AutoRetrainer::new(detector, RetrainConfig::default());
retrainer.on_retrain(|results| {
    println!("Drift detected! Triggering retraining...");
    Ok("job-123".to_string())
});
```

## Features

| Feature | Description |
|---------|-------------|
| Classification Metrics | Accuracy, Precision, Recall, F1 with Macro/Micro/Weighted averaging |
| Confusion Matrix | Full NxN matrix with per-class TP/FP/FN/TN |
| Leaderboard | Compare multiple models, sort by any metric |
| Cross-Validation | K-Fold with shuffle support |
| Drift Detection | KS test, Chi-square test, PSI |
| Auto-Retraining | Policy-based triggers with cooldown and limits |

## Examples

- [Drift Detection Simulation](../examples/drift-simulation.md)
- [P-Value Calibration Check](../examples/calibration-check.md)
