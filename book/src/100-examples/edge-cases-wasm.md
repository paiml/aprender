# Edge Cases & WASM (Examples 96-100)

This section covers edge cases, error handling, and WASM support.

## Example 96: WASM Build Verification

```bash
# Build for WASM target
cargo build --target wasm32-unknown-unknown \
    --no-default-features --features wasm

# Verify binary size
ls -la target/wasm32-unknown-unknown/release/*.wasm
# Target: <500KB
```

```rust
// WASM-compatible usage
#[cfg(target_arch = "wasm32")]
use alimentar::wasm::{WasmDataset, WasmLoader};

#[cfg(target_arch = "wasm32")]
pub fn load_in_browser(data: &[u8]) -> Result<WasmDataset, JsValue> {
    let dataset = WasmDataset::from_parquet_bytes(data)?;
    Ok(dataset)
}
```

## Example 97: Empty Dataset Handling (Jidoka)

```rust
use alimentar::ArrowDataset;

let result = ArrowDataset::from_parquet("empty.parquet");

// Jidoka: Stop and signal problem
match result {
    Ok(dataset) if dataset.len() == 0 => {
        // Empty but valid - proceed with caution
        println!("Warning: Empty dataset");
    }
    Err(e) => {
        // Error loading - stop the line
        eprintln!("Jidoka: {}", e);
        return Err(e);
    }
    Ok(dataset) => {
        // Normal processing
        process(dataset);
    }
}
```

## Example 98: Corrupt Dataset Handling (Jidoka)

```rust
use alimentar::ArrowDataset;

let result = ArrowDataset::from_parquet("corrupt.parquet");

// Jidoka: Detect and stop on corruption
assert!(result.is_err(), "Corrupt file should return error");

match result {
    Err(e) => {
        eprintln!("Jidoka stop: Corrupt file detected");
        eprintln!("Error: {}", e);
        // Alert human for intervention
    }
    Ok(_) => unreachable!(),
}
```

## Example 99: S3 Backend Integration

```rust
use alimentar::backend::{BackendConfig, S3Config};

// Configure S3 backend
let config = S3Config::builder()
    .bucket("my-bucket")
    .region("us-west-2")
    .endpoint("https://s3.amazonaws.com")
    .build();

let backend = BackendConfig::S3(config).create()?;

// List datasets
let datasets = backend.list("datasets/").await?;

// Load from S3
let data = backend.get("datasets/train.parquet").await?;
```

```bash
# S3 via CLI
AWS_ACCESS_KEY_ID=xxx AWS_SECRET_ACCESS_KEY=yyy \
    alimentar info s3://my-bucket/data.parquet
```

## Example 100: Golden Run (All Features)

```rust
use alimentar::{
    ArrowDataset, DataLoader, QualityChecker, DriftDetector,
    DatasetSplit, Transform, Select,
};

fn golden_run() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Load data
    let dataset = ArrowDataset::from_parquet("data.parquet")?;

    // 2. Quality check
    let checker = QualityChecker::new();
    let quality = checker.check(&dataset)?;
    assert!(quality.quality_score() >= 0.8, "Quality gate failed");

    // 3. Transform
    let select = Select::new(vec!["id".into(), "value".into()]);
    let transformed = dataset.with_transform(Box::new(select));

    // 4. Split
    let split = DatasetSplit::from_ratios(&transformed, 0.8, 0.2, None, Some(42))?;

    // 5. DataLoader
    let loader = DataLoader::new(split.train().clone())
        .batch_size(32)
        .shuffle(true)
        .seed(42);

    // 6. Iterate
    for batch in loader {
        assert!(batch.num_rows() > 0);
    }

    println!("Golden run: PASS");
    Ok(())
}
```

```bash
# Golden run via CLI
cargo test --test example_scenarios test_example_100_golden_run
```

## Error Handling Philosophy

| Principle | Implementation |
|-----------|----------------|
| **Jidoka** | Stop on error, don't propagate bad data |
| **Poka-Yoke** | Type system prevents invalid states |
| **Andon** | Clear error messages with context |
| **Genchi Genbutsu** | Go to the source - include file paths |

## WASM Constraints

| Feature | Native | WASM |
|---------|--------|------|
| Filesystem | Yes | No |
| Threading | Yes | No |
| S3 Backend | Yes | No |
| HTTP Backend | Yes | Limited |
| Memory Backend | Yes | Yes |

## Key Concepts

- **Graceful degradation**: Handle missing features
- **Error types**: Rich, actionable error information
- **WASM portability**: Runs in browser
- **Golden run**: Full integration test
