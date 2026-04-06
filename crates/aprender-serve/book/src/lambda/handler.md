# Lambda Handler

The `LambdaHandler` is the core component for AWS Lambda deployment, managing model lifecycle and inference.

## Handler Structure

```rust
pub struct LambdaHandler {
    model_bytes: &'static [u8],
    init_time: OnceLock<Instant>,
    cold_start_metrics: OnceLock<ColdStartMetrics>,
}
```

### Design Decisions

1. **`&'static [u8]`**: Model bytes must be `'static` for `include_bytes!()` embedding
2. **`OnceLock<Instant>`**: Tracks initialization time for cold start detection
3. **`OnceLock<ColdStartMetrics>`**: Captures first-invocation metrics

## Creating a Handler

### From Embedded Bytes (Recommended)

```rust
static MODEL_BYTES: &[u8] = include_bytes!("../models/model.apr");

let handler = LambdaHandler::from_bytes(MODEL_BYTES)?;
```

### Validation

The handler validates `.apr` magic bytes on creation:

```rust
// .apr files start with "APR\0"
if model_bytes.len() >= 4 && &model_bytes[0..4] != b"APR\0" {
    return Err(LambdaError::InvalidMagic { ... });
}
```

## Error Handling

```rust
pub enum LambdaError {
    /// Model bytes were empty
    EmptyModel,

    /// Invalid .apr magic bytes
    InvalidMagic { expected: [u8; 4], actual: [u8; 4] },

    /// Batch request had no instances
    EmptyBatch,

    /// General inference error
    InferenceError(String),
}
```

## Cold Start Detection

The handler automatically detects and reports cold starts:

```rust
impl LambdaHandler {
    pub fn handle(&self, request: &LambdaRequest) -> Result<LambdaResponse, LambdaError> {
        let start = Instant::now();

        // First invocation initializes OnceLock (cold start)
        let is_cold_start = self.init_time.set(Instant::now()).is_ok();

        // ... inference logic ...

        Ok(LambdaResponse {
            cold_start: is_cold_start,
            latency_ms: start.elapsed().as_secs_f64() * 1000.0,
            // ...
        })
    }
}
```

## ARM64 Optimization

The handler detects ARM64/NEON capabilities:

```rust
pub fn is_arm64_optimized() -> bool {
    cfg!(target_arch = "aarch64")
}

pub fn has_neon_simd() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        std::arch::is_aarch64_feature_detected!("neon")
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        false
    }
}
```

When deployed on AWS Graviton:
- **Graviton2**: 64 NEON SIMD units
- **Graviton3**: 2x vector processing width
- **Cost**: ~20% cheaper than x86_64 Lambda
