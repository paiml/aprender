# Cold Start Optimization

Cold start is the time from Lambda invocation to first response when no warm container exists. Realizar achieves sub-50ms cold starts through careful optimization.

## Cold Start Anatomy

```
┌─────────────────────────────────────────────────────────────┐
│                    Cold Start Timeline                       │
├──────────┬──────────┬──────────┬──────────┬────────────────┤
│  Init    │  Load    │  Parse   │  First   │   Response     │
│ Runtime  │ Binary   │  Model   │ Inference│                │
├──────────┼──────────┼──────────┼──────────┼────────────────┤
│  ~50ms   │  ~20ms   │  ~10ms   │  ~5ms    │   ~100µs       │
│ (AWS)    │ (embed)  │ (APR)    │ (warm)   │   (send)       │
└──────────┴──────────┴──────────┴──────────┴────────────────┘
```

## Key Techniques

### 1. `include_bytes!()` for Model Embedding

```rust
// Model compiled into binary - no S3 download needed
static MODEL_BYTES: &[u8] = include_bytes!("../models/model.apr");

let handler = LambdaHandler::from_bytes(MODEL_BYTES)?;
```

**Benefits:**
- No network latency for model loading
- Model in memory immediately
- Single binary deployment

### 2. `OnceLock` for Lazy Initialization

```rust
use std::sync::OnceLock;

pub struct LambdaHandler {
    model_bytes: &'static [u8],
    init_time: OnceLock<Instant>,  // Set on first invocation
}
```

The `OnceLock` defers expensive initialization until first use, then caches the result.

### 3. Minimal Dependencies

```toml
[features]
lambda = []  # No HTTP server dependencies!
```

The `lambda` feature excludes:
- `axum` (async runtime overhead)
- `tokio` (large runtime)
- `tower` (middleware stack)

Result: Much smaller binary, faster load.

### 4. Release Profile Optimization

```toml
[profile.release]
opt-level = 3
lto = "fat"           # Link-time optimization
codegen-units = 1     # Single codegen unit
panic = "abort"       # Smaller binary
strip = true          # Remove debug symbols
```

## Measuring Cold Start

### In Handler

```rust
impl LambdaHandler {
    pub fn handle(&self, request: &LambdaRequest) -> Result<LambdaResponse, LambdaError> {
        let start = Instant::now();

        // First invocation sets init_time (cold start)
        let is_cold_start = self.init_time.set(start).is_ok();

        // ... inference ...

        Ok(LambdaResponse {
            cold_start: is_cold_start,
            latency_ms: start.elapsed().as_secs_f64() * 1000.0,
            // ...
        })
    }
}
```

### Cold Start Metrics

```rust
pub struct ColdStartMetrics {
    pub init_duration_ms: f64,
    pub first_inference_ms: f64,
    pub total_cold_start_ms: f64,
}
```

## Benchmark Results

From `benches/lambda.rs`:

| Metric | Target | Achieved |
|--------|--------|----------|
| Handler creation | <50ms | ~100µs |
| First inference | <10ms | ~500ns |
| Total cold start | <50ms | <1ms |

**We exceed targets by 50-100x!**

## Warm vs Cold

| Invocation | First Request | Subsequent |
|------------|---------------|------------|
| Cold | ~85ms (AWS init) + ~1ms (our code) | - |
| Warm | - | ~500ns |

The AWS Lambda runtime initialization (~50-100ms) dominates cold start time. Our code adds <1ms.

## Provisioned Concurrency

For latency-sensitive workloads, use provisioned concurrency:

```yaml
# SAM template
AutoPublishAlias: live
ProvisionedConcurrencyConfig:
  ProvisionedConcurrentExecutions: 10
```

This keeps warm containers ready, eliminating cold starts entirely.

## Best Practices

1. **Keep binary small** - Use `lambda` feature, not `server`
2. **Embed models** - Use `include_bytes!()` for small models
3. **Use ARM64** - Faster startup than x86_64
4. **Profile in production** - CloudWatch provides real metrics
5. **Consider provisioned concurrency** - For consistent latency
