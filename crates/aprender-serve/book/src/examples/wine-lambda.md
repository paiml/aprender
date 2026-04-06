# Wine Quality Prediction - Lambda Example

A production-ready wine quality rating predictor deployable to AWS Lambda, inspired by [paiml/wine-ratings](https://github.com/paiml/wine-ratings).

## Overview

This example demonstrates end-to-end ML serving on AWS Lambda:

- **Input**: 11 physicochemical wine properties
- **Output**: Quality score (0-10) with category classification
- **Latency**: Sub-millisecond warm inference
- **Throughput**: 200K+ predictions/second
- **Deployment**: ARM64 Graviton2 optimized

## Wine Features

The model accepts 11 standard wine quality features:

| Feature | Description | Typical Range |
|---------|-------------|---------------|
| `fixed_acidity` | Tartaric acid (g/dm³) | 4.0-16.0 |
| `volatile_acidity` | Acetic acid (g/dm³) | 0.1-1.6 |
| `citric_acid` | Citric acid (g/dm³) | 0.0-1.0 |
| `residual_sugar` | Sugar after fermentation (g/dm³) | 0.9-15.0 |
| `chlorides` | Sodium chloride (g/dm³) | 0.01-0.6 |
| `free_sulfur_dioxide` | Free SO₂ (mg/dm³) | 1-72 |
| `total_sulfur_dioxide` | Total SO₂ (mg/dm³) | 6-289 |
| `density` | Density (g/cm³) | 0.99-1.04 |
| `ph` | pH level | 2.7-4.0 |
| `sulphates` | Potassium sulphate (g/dm³) | 0.3-2.0 |
| `alcohol` | Alcohol content (% vol) | 8.0-15.0 |

## Running the Example

```bash
cargo run --example wine_lambda
```

Output:

```
=== Wine Quality Prediction - Lambda Example ===

Wine Quality Predictions:
----------------------------------------------------------------------

Bordeaux Red (Premium)
  Quality Score: 6.26/10 (Average)
  Confidence: 100%
  Latency: 0.002ms
  [COLD START]
  Top factors: alcohol, citric_acid, sulphates

Table Wine (Budget)
  Quality Score: 5.42/10 (Average)
  Confidence: 100%
  Latency: 0.000ms
  Top factors: volatile_acidity, alcohol, sulphates

Napa Cabernet (Premium)
  Quality Score: 6.46/10 (Average)
  Confidence: 100%
  Latency: 0.000ms
  Top factors: alcohol, citric_acid, sulphates

----------------------------------------------------------------------

Batch Prediction (100 random wines):
  Processed: 100 predictions
  Total time: 0.49ms
  Throughput: 206145 predictions/sec
```

## Code Structure

### WineFeatures

```rust
pub struct WineFeatures {
    pub fixed_acidity: f32,
    pub volatile_acidity: f32,
    pub citric_acid: f32,
    pub residual_sugar: f32,
    pub chlorides: f32,
    pub free_sulfur_dioxide: f32,
    pub total_sulfur_dioxide: f32,
    pub density: f32,
    pub ph: f32,
    pub sulphates: f32,
    pub alcohol: f32,
}

impl WineFeatures {
    /// Convert to normalized feature vector
    pub fn normalize(&self) -> Vec<f32>;

    /// Example wines for testing
    pub fn example_bordeaux() -> Self;
    pub fn example_table_wine() -> Self;
    pub fn example_napa_cabernet() -> Self;
}
```

### WinePredictor

```rust
pub struct WinePredictor {
    weights: Vec<f32>,
    bias: f32,
    inference_count: AtomicU64,
}

impl WinePredictor {
    pub fn new() -> Self;
    pub fn predict(&self, features: &WineFeatures) -> WinePrediction;
    pub fn is_cold_start(&self) -> bool;
}
```

### WinePrediction

```rust
pub struct WinePrediction {
    /// Quality score (0-10, typically 3-9)
    pub quality: f32,
    /// Quality category (Poor/Average/Good/Excellent)
    pub category: WineCategory,
    /// Confidence score (0-1)
    pub confidence: f32,
    /// Inference latency in milliseconds
    pub latency_ms: f32,
    /// Feature importance scores
    pub feature_importance: HashMap<String, f32>,
}
```

## Quality Categories

| Category | Score Range | Description |
|----------|-------------|-------------|
| Poor | 0-4 | Below average quality |
| Average | 5-6 | Typical table wine |
| Good | 7 | Above average quality |
| Excellent | 8-10 | Premium quality |

## Model Weights

The predictor uses coefficients derived from wine quality analysis:

```rust
weights: vec![
    -0.05,  // fixed_acidity (slight negative)
    -0.85,  // volatile_acidity (strong negative - vinegar taste)
     0.45,  // citric_acid (positive - freshness)
     0.02,  // residual_sugar (minimal effect)
    -0.15,  // chlorides (negative - salty taste)
     0.08,  // free_sulfur_dioxide (slight positive)
    -0.12,  // total_sulfur_dioxide (slight negative)
    -0.30,  // density (negative - correlates with sugar/alcohol)
    -0.10,  // pH (slight negative)
     0.55,  // sulphates (positive - antimicrobial)
     0.95,  // alcohol (strong positive - body/complexity)
],
bias: 5.5,
```

Key insights:
- **Alcohol** has the strongest positive correlation
- **Volatile acidity** has strong negative impact (vinegar taste)
- **Sulphates** and **citric acid** contribute positively

## Metrics & Monitoring

### WineMetrics

```rust
pub struct WineMetrics {
    pub requests_total: u64,
    pub predictions_by_category: HashMap<String, u64>,
    pub total_latency_ms: f64,
    pub cold_starts: u64,
}

impl WineMetrics {
    pub fn record(&mut self, prediction: &WinePrediction, cold_start: bool);
    pub fn avg_latency_ms(&self) -> f64;
    pub fn to_prometheus(&self) -> String;
}
```

### Prometheus Export

```
# HELP wine_predictions_total Total wine quality predictions
# TYPE wine_predictions_total counter
wine_predictions_total 103

# HELP wine_predictions_by_category Predictions by quality category
# TYPE wine_predictions_by_category counter
wine_predictions_by_category{category="Average"} 103

# HELP wine_latency_avg_ms Average prediction latency
# TYPE wine_latency_avg_ms gauge
wine_latency_avg_ms 0.0002

# HELP wine_cold_starts_total Cold start count
# TYPE wine_cold_starts_total counter
wine_cold_starts_total 1
```

## Drift Detection

Monitor feature distributions for production drift:

```rust
pub struct WineDriftDetector {
    reference_means: Vec<f32>,
    reference_stds: Vec<f32>,
    threshold: f32,  // z-score threshold (default: 3.0)
}

impl WineDriftDetector {
    pub fn check_drift(&self, features: &WineFeatures) -> DriftResult;
}

pub struct DriftResult {
    pub is_drifted: bool,
    pub max_z_score: f32,
    pub drifted_features: Vec<String>,
}
```

Example drift detection:

```rust
let detector = WineDriftDetector::new();
let result = detector.check_drift(&wine_features);

if result.is_drifted {
    println!("DRIFT WARNING: {:?}", result.drifted_features);
    // Trigger retraining pipeline
}
```

## AWS Lambda Deployment

### Build for ARM64 Graviton

```bash
# Install cross-compilation target
rustup target add aarch64-unknown-linux-gnu

# Build release binary
cargo build --release --target aarch64-unknown-linux-gnu --features lambda
```

### Package for Lambda

```bash
# Copy binary as bootstrap
cp target/aarch64-unknown-linux-gnu/release/wine_lambda bootstrap

# Create deployment package
zip wine_lambda.zip bootstrap
```

### Deploy to AWS

```bash
aws lambda create-function \
  --function-name wine-quality-predictor \
  --runtime provided.al2 \
  --architecture arm64 \
  --handler bootstrap \
  --zip-file fileb://wine_lambda.zip \
  --role arn:aws:iam::ACCOUNT:role/lambda-role \
  --memory-size 128 \
  --timeout 30
```

### Lambda Configuration

| Setting | Value | Reason |
|---------|-------|--------|
| Runtime | `provided.al2` | Custom Rust runtime |
| Architecture | `arm64` | Graviton2 (20% cheaper, better perf) |
| Memory | 128 MB | Sufficient for lightweight model |
| Timeout | 30 sec | Allows for cold start |

### Expected Performance

| Metric | Value |
|--------|-------|
| Cold start | < 10ms |
| Warm inference | < 1ms |
| Memory usage | < 50 MB |
| Throughput | 200K+ pred/sec |

## Testing

The example includes 11 unit tests:

```bash
cargo test --example wine_lambda
```

```
running 11 tests
test tests::test_cold_start_detection ... ok
test tests::test_drift_detection_extreme ... ok
test tests::test_drift_detection_normal ... ok
test tests::test_feature_importance ... ok
test tests::test_metrics_recording ... ok
test tests::test_predictor_output_range ... ok
test tests::test_predictor_premium_vs_budget ... ok
test tests::test_prometheus_export ... ok
test tests::test_wine_category_classification ... ok
test tests::test_wine_features_normalize ... ok
test tests::test_wine_features_to_vec ... ok

test result: ok. 11 passed
```

## Integration with Aprender

For production deployment with trained `.apr` models, realizar's `LambdaHandler` provides **real APR model inference** (not mocked):

```rust
use realizar::lambda::{LambdaHandler, LambdaRequest};

// Load trained wine model (must have APRN magic bytes)
const MODEL_BYTES: &[u8] = include_bytes!("wine_model.apr");

fn predict_wine(features: &WineFeatures) -> Result<f32, LambdaError> {
    // Handler validates APRN format and lazily initializes AprModel
    let handler = LambdaHandler::from_bytes(MODEL_BYTES)?;

    let request = LambdaRequest {
        features: features.normalize(),
        model_id: Some("wine_quality_v1".to_string()),
    };

    // Real inference using AprModel::predict() with OnceLock caching
    let response = handler.handle(&request)?;

    // Multi-output models return probabilities
    if let Some(probs) = response.probabilities {
        println!("Class probabilities: {:?}", probs);
    }

    Ok(response.prediction)
}
```

### APR Format Requirements

The `.apr` model file must have:
- **Magic bytes**: `APRN` (0x4150524E)
- **Format version**: 1.0
- **Uncompressed JSON payload** (compression support coming soon)

## Next Steps

1. **Train real model**: Use aprender to train on UCI Wine Quality dataset
2. **Export to .apr**: Save trained model in .apr format
3. **Deploy to Lambda**: Use realizar's Lambda handler
4. **Monitor drift**: Set up alerts for feature distribution shifts
5. **Automate retraining**: Trigger pipeline when drift detected
