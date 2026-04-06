# ModelEvaluator & Leaderboard

The `ModelEvaluator` provides a standardized interface for evaluating and comparing
models. It supports classification metrics, cross-validation, and leaderboard tracking.

## Quick Start

```rust
use entrenar::eval::{ModelEvaluator, EvalConfig, Metric, Average};

let evaluator = ModelEvaluator::new(EvalConfig {
    metrics: vec![
        Metric::Accuracy,
        Metric::F1(Average::Weighted),
        Metric::F1(Average::Macro),
    ],
    ..Default::default()
});

let result = evaluator.evaluate_classification("my_model", &predictions, &labels)?;
println!("Accuracy: {:.2}%", result.get_score(Metric::Accuracy).unwrap() * 100.0);
```

## Leaderboard

Compare multiple models side-by-side:

```rust
let mut leaderboard = evaluator.leaderboard();

leaderboard.add("baseline_v1", &preds_v1, &labels)?;
leaderboard.add("lora_v2", &preds_v2, &labels)?;
leaderboard.add("lora_v3", &preds_v3, &labels)?;

// Sort by weighted F1
leaderboard.sort_by(Metric::F1(Average::Weighted));
println!("{}", leaderboard.to_table());
```

## Cross-Validation

```rust
use entrenar::eval::CrossValidate;

let cv = CrossValidate::new(5, true); // 5-fold, shuffle
let results = cv.evaluate(&evaluator, &data, &labels)?;
println!("Mean accuracy: {:.2}% ± {:.2}%", results.mean() * 100.0, results.std() * 100.0);
```

## ClassifyEvalReport

For fine-tuned classification models (e.g., Shell Safety Classifier), use
`ClassifyEvalReport` which provides 13 metrics with bootstrap CIs, proper scoring
rules, calibration analysis, and model card generation.

See [Classification Metrics](./classification-metrics.md) for details.
