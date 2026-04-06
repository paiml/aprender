# Batch Inference

Process multiple predictions in a single Lambda invocation for improved throughput.

## Request Format

```rust
pub struct BatchLambdaRequest {
    /// Multiple instances to predict
    pub instances: Vec<LambdaRequest>,
    /// Optional: Maximum parallel workers (default: CPU count)
    pub max_parallelism: Option<usize>,
}
```

## Response Format

```rust
pub struct BatchLambdaResponse {
    /// Individual predictions
    pub predictions: Vec<LambdaResponse>,
    /// Total batch processing time in milliseconds
    pub total_latency_ms: f64,
    /// Number of successful predictions
    pub success_count: usize,
    /// Number of failed predictions
    pub error_count: usize,
}
```

## Usage

```rust
use realizar::lambda::{LambdaHandler, LambdaRequest, BatchLambdaRequest};

let handler = LambdaHandler::from_bytes(MODEL_BYTES)?;

let batch = BatchLambdaRequest {
    instances: vec![
        LambdaRequest { features: vec![1.0, 2.0], model_id: None },
        LambdaRequest { features: vec![3.0, 4.0], model_id: None },
        LambdaRequest { features: vec![5.0, 6.0], model_id: None },
    ],
    max_parallelism: Some(4),
};

let response = handler.handle_batch(&batch)?;

println!("Processed {} predictions", response.success_count);
println!("Total latency: {}ms", response.total_latency_ms);

for (i, pred) in response.predictions.iter().enumerate() {
    println!("  [{}] prediction={}", i, pred.prediction);
}
```

## Performance Characteristics

From benchmarks (`benches/lambda.rs`):

| Batch Size | Throughput | Notes |
|------------|------------|-------|
| 1 | Baseline | Single prediction overhead |
| 10 | ~10x | Linear scaling |
| 50 | ~50x | Linear scaling |
| 100 | ~100x | Linear scaling |

## Error Handling

Batch processing continues even if individual predictions fail:

```rust
let response = handler.handle_batch(&batch)?;

if response.error_count > 0 {
    eprintln!("Warning: {} predictions failed", response.error_count);
}

// Successful predictions still available
for pred in response.predictions {
    // Process results...
}
```

## When to Use Batch

| Use Case | Recommendation |
|----------|----------------|
| Real-time single prediction | Use `handle()` |
| Bulk scoring | Use `handle_batch()` |
| SQS batch processing | Use `handle_batch()` |
| Streaming predictions | Use `handle()` per event |
