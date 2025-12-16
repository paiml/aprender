# Case Study: Pipeline Verification System

This case study demonstrates aprender's pipeline verification system for ML model debugging, implementing Toyota Way's Jidoka principle: built-in quality with automatic stop on first defect.

## The Problem

When porting ML models between frameworks (PyTorch to Rust, ONNX to native, etc.), subtle numerical differences can cascade through the pipeline:

| Stage | Issue | Symptom |
|-------|-------|---------|
| Preprocessing | Normalization sign flip | Complete output inversion |
| Encoder | Precision loss | Gradual drift in deeper layers |
| Attention | Softmax overflow | NaN propagation |
| Output | Quantization error | Wrong predictions |

**Finding the root cause is like debugging a 10-stage pipeline with a single "wrong output" error message.**

## The Solution: Stage-by-Stage Ground Truth Verification

The `verify` module provides systematic comparison at each pipeline stage:

```rust,ignore
use aprender::verify::{Pipeline, GroundTruth, Tolerance};

let pipeline = Pipeline::builder("whisper-tiny")
    .stage("mel")
        .ground_truth_stats(-0.215, 0.448)  // Expected mean, std
        .tolerance(Tolerance::percent(5.0)) // 5% tolerance
        .build_stage()
    .stage("encoder")
        .ground_truth_stats(0.0, 0.8)
        .tolerance(Tolerance::percent(10.0))
        .build_stage()
    .build()
    .expect("Pipeline definition error");

// Verify outputs against ground truth
let report = pipeline.verify(|stage_name| {
    match stage_name {
        "mel" => Some(GroundTruth::from_stats(-0.210, 0.450)),
        "encoder" => Some(GroundTruth::from_stats(0.01, 0.78)),
        _ => None,
    }
});

assert!(report.all_passed());
```

## Complete Example

Run: `cargo run --example pipeline_verification`

```rust,ignore
{{#include ../../../examples/pipeline_verification.rs}}
```

## Key Features

### 1. Jidoka: Stop-on-First-Failure

By default, verification stops at the first failure (Toyota Way: stop the line when defect is detected):

```rust,ignore
// Default: Jidoka enabled
let pipeline = Pipeline::builder("model")
    .stage("a").ground_truth_stats(0.0, 1.0).tolerance(Tolerance::percent(5.0)).build_stage()
    .stage("b").ground_truth_stats(0.0, 1.0).tolerance(Tolerance::percent(5.0)).build_stage()
    .stage("c").ground_truth_stats(0.0, 1.0).tolerance(Tolerance::percent(5.0)).build_stage()
    .build()?;

// If stage "a" fails, "b" and "c" are skipped
// This prevents cascading failures from obscuring the root cause
```

For full analysis of all stages:

```rust,ignore
let pipeline = Pipeline::builder("full-analysis")
    .stage("a").build_stage()
    .stage("b").build_stage()
    .stage("c").build_stage()
    .continue_on_failure()  // Evaluate ALL stages regardless of failures
    .build()?;
```

### 2. Multiple Tolerance Types

```rust,ignore
// Simple percent tolerance
Tolerance::percent(5.0)

// Separate mean/std thresholds (for high-precision stages)
Tolerance::stats(0.01, 0.02)  // mean <= 0.01, std <= 0.02

// Cosine similarity minimum (for embedding comparisons)
Tolerance::cosine(0.99)  // Require 99% similarity

// KL divergence threshold (for probability distributions)
Tolerance::kl_divergence(0.1)

// Custom multi-criteria tolerance
Tolerance::custom()
    .percent(10.0)
    .mean_delta(0.1)
    .cosine_min(0.95)
    .build()
```

### 3. Ground Truth from Multiple Sources

```rust,ignore
// From known statistics (e.g., from reference implementation docs)
let gt = GroundTruth::from_stats(mean, std);

// From raw data (computed automatically)
let reference_output = vec![0.1, 0.2, 0.3, 0.4, 0.5];
let gt = GroundTruth::from_slice(&reference_output);

// Full statistics available
println!("Mean: {}, Std: {}, Min: {}, Max: {}",
         gt.mean(), gt.std(), gt.min(), gt.max());
```

### 4. Delta Analysis

```rust,ignore
use aprender::verify::Delta;

let our = GroundTruth::from_slice(&our_output);
let reference = GroundTruth::from_slice(&ref_output);
let delta = Delta::compute(&our, &reference);

// Statistical deltas
println!("Mean delta: {:.4}", delta.mean_delta());
println!("Std delta:  {:.4}", delta.std_delta());
println!("Percent:    {:.2}%", delta.percent());

// Sign flip detection (common bug in normalization)
if delta.is_sign_flipped() {
    println!("WARNING: Sign flip detected!");
}

// Vector similarity
if let Some(cos) = delta.cosine() {
    println!("Cosine similarity: {:.4}", cos);
}
```

### 5. Distribution Comparison

```rust,ignore
// Cosine similarity for direction comparison
let cos = Delta::cosine_similarity(&vec_a, &vec_b);

// KL divergence for probability distributions
let kl = Delta::kl_divergence(&probs_a, &probs_b);
```

### 6. Automatic Diagnosis

When a stage fails, the system provides diagnostic hints:

```rust,ignore
if let Some(failure) = report.first_failure() {
    println!("Failed stage: {}", failure.name());

    for diagnosis in failure.diagnose() {
        println!("  - {}", diagnosis);
    }
}
```

Example output:
```
Diagnosis for 'mel_spectrogram' failure:
  - Stage 'mel_spectrogram' failed with delta 89.1%
  - Sign is FLIPPED (positive vs negative)
  - Likely cause: Normalization formula error
  - Check: Log base, subtraction order, sign convention
```

## Real-World Use Case: Whisper Model Porting

```rust,ignore
let whisper = Pipeline::builder("whisper-tiny")
    .stage("mel")
        .ground_truth_stats(-0.215, 0.448)
        .tolerance(Tolerance::percent(5.0))
        .description("Log-mel spectrogram (80 mel bins)")
        .build_stage()
    .stage("encoder_out")
        .ground_truth_stats(0.0, 0.8)
        .tolerance(Tolerance::percent(10.0))
        .description("Encoder final output")
        .build_stage()
    .stage("decoder_logits")
        .ground_truth_stats(0.0, 15.0)
        .tolerance(Tolerance::percent(15.0))
        .description("Decoder output logits")
        .build_stage()
    .stage("probs")
        .ground_truth_stats(0.0001, 0.01)
        .tolerance(Tolerance::percent(20.0))
        .description("Softmax probabilities")
        .build_stage()
    .build()?;

// Run verification against reference implementation
let report = whisper.verify(|stage| {
    get_stage_output_from_our_implementation(stage)
});

if !report.all_passed() {
    eprintln!("Verification failed!");
    eprintln!("{}", report.summary());

    if let Some(first_fail) = report.first_failure() {
        eprintln!("\nFirst failure at: {}", first_fail.name());
        for diag in first_fail.diagnose() {
            eprintln!("  {}", diag);
        }
    }
}
```

## Pipeline Verification in CI/CD

```rust,ignore
#[test]
fn test_model_regression() {
    let pipeline = load_verification_pipeline();
    let report = pipeline.verify(|stage| {
        run_inference_stage(stage)
    });

    assert!(
        report.all_passed(),
        "Model regression detected: {}",
        report.summary()
    );
}
```

## API Reference

### Pipeline Builder

| Method | Description |
|--------|-------------|
| `Pipeline::builder(name)` | Create new pipeline |
| `.stage(name)` | Add a stage |
| `.ground_truth_stats(mean, std)` | Set expected statistics |
| `.ground_truth(gt)` | Set full ground truth |
| `.tolerance(t)` | Set tolerance threshold |
| `.description(desc)` | Add human-readable description |
| `.build_stage()` | Finish stage, return to pipeline |
| `.continue_on_failure()` | Disable Jidoka |
| `.build()` | Build the pipeline |

### Tolerance Types

| Type | Use Case |
|------|----------|
| `Tolerance::percent(n)` | General purpose, % deviation |
| `Tolerance::stats(m, s)` | Precision-critical stages |
| `Tolerance::cosine(min)` | Embedding/vector comparisons |
| `Tolerance::kl_divergence(max)` | Probability distributions |
| `Tolerance::custom()` | Multi-criteria validation |

### Report Methods

| Method | Returns |
|--------|---------|
| `report.all_passed()` | `bool` |
| `report.first_failure()` | `Option<&StageResult>` |
| `report.passed_count()` | `usize` |
| `report.failed_count()` | `usize` |
| `report.skipped_count()` | `usize` |
| `report.summary()` | `String` (colored) |
| `report.results()` | `&[StageResult]` |

## Toyota Way Principles Applied

1. **Jidoka (Built-in Quality)**: Stop-on-first-failure prevents cascading errors
2. **Genchi Genbutsu (Go and See)**: Stage-by-stage inspection reveals actual divergence points
3. **Kaizen (Continuous Improvement)**: CI/CD integration catches regressions early
4. **Visual Management**: Colored output with pass/fail/skip icons

## See Also

- [What is EXTREME TDD?](../methodology/what-is-extreme-tdd.md)
- [Jidoka (Built-in Quality)](../toyota-way/jidoka.md)
- [Case Study: Model Format (.apr)](./model-format.md)
