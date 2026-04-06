# Classification Metrics

The `ClassifyEvalReport` provides 13 metrics across 4 categories for evaluating
classification models. All metrics are computed in a single evaluation pass.

## Accuracy & Agreement

| Metric | Description | Range |
|--------|-------------|-------|
| **Accuracy** | Fraction of correct predictions | [0, 1] |
| **Top-2 Accuracy** | Correct class in top 2 softmax outputs | [0, 1] |
| **Cohen's Kappa** | Chance-corrected agreement | [-1, 1], >0.6 = substantial |
| **MCC** | Matthews Correlation Coefficient | [-1, 1], balanced for skewed classes |

All four include **bootstrap 95% confidence intervals** (1,000 resamples with
deterministic LCG PRNG, seed=42).

## Per-Class Performance

For each class *k*:

| Metric | Formula |
|--------|---------|
| **Precision** | TP_k / (TP_k + FP_k) |
| **Recall** | TP_k / (TP_k + FN_k) |
| **F1** | 2 * P * R / (P + R) |
| **Support** | Number of true instances |

Aggregations: **macro F1** (unweighted average) and **weighted F1** (weighted by support).

## Proper Scoring Rules

| Metric | Formula | Notes |
|--------|---------|-------|
| **Brier Score** | mean(sum((p_k - y_k)^2)) | Multi-class MSE; lower = better |
| **Log Loss** | -mean(log(p_true + epsilon)) | Epsilon=1e-15 clamping; lower = better |

These reward well-calibrated probability estimates, not just correct top-1 predictions.

## Calibration & Confidence

| Metric | Description |
|--------|-------------|
| **ECE** | Expected Calibration Error: weighted |confidence - accuracy| across 10 bins |
| **Mean Confidence** | Average max probability across all predictions |
| **Confidence (correct)** | Average confidence on correctly classified samples |
| **Confidence (wrong)** | Average confidence on misclassified samples |
| **Confidence Gap** | Difference between correct and wrong confidence |

## Baselines

| Baseline | Formula | Purpose |
|----------|---------|---------|
| **Random** | 1/K | Lower bound for any classifier |
| **Majority** | max(class_proportions) | Constant-class predictor |
| **Lift** | model_accuracy / majority | How much better than majority guessing |

## Error Analysis

Top-N most confused class pairs are extracted from off-diagonal confusion matrix entries,
identifying systematic failure modes.

## Usage

```rust
use entrenar::finetune::{evaluate_checkpoint, ClassifyConfig};
use entrenar::transformer::TransformerConfig;

let config = TransformerConfig::qwen2_0_5b();
let classify_config = ClassifyConfig { num_classes: 5, ..Default::default() };

let report = evaluate_checkpoint(
    checkpoint_dir,
    test_data_path,
    &config,
    classify_config,
)?;

// Text report (sklearn-style)
println!("{}", report.to_report());

// JSON
println!("{}", report.to_json());

// HuggingFace model card
let card = report.to_model_card("paiml/shell-safety-classifier", Some("Qwen/Qwen2.5-Coder-0.5B"));
std::fs::write(checkpoint_dir.join("README.md"), card)?;
```

## CLI

```bash
apr eval <checkpoint> --task classify --data test.jsonl --model-size 0.5B --num-classes 5
apr eval <checkpoint> --task classify --data test.jsonl --model-size 0.5B --json
apr eval <checkpoint> --task classify --data test.jsonl --model-size 0.5B --generate-card
```

See also: [Model Evaluation Overview](./overview.md)
