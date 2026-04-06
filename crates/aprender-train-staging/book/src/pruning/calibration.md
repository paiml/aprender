# Calibration for Pruning

Activation-weighted pruning methods like Wanda and SparseGPT require calibration data to estimate input statistics. This
chapter covers setting up and using calibration.

## Why Calibration?

Magnitude-based pruning only considers weight values:

```
importance(w) = |w|
```

Activation-weighted methods also consider how inputs interact with weights:

```
importance(w_ij) = |w_ij| * sqrt(sum(x_j^2) / n)
```

This captures the actual contribution of each weight given typical inputs.

## Which Methods Need Calibration?

| Method | Requires Calibration | Why |
|--------|---------------------|-----|
| Magnitude | No | Uses only weight values |
| Wanda | Yes | Weights AND Activations |
| SparseGPT | Yes | Hessian approximation |

## CalibrationConfig

Configure calibration data collection:

```rust
use entrenar::prune::CalibrationConfig;

let config = CalibrationConfig::new()
    .with_num_samples(128)        // Number of calibration sequences
    .with_sequence_length(2048)   // Tokens per sequence
    .with_batch_size(1)           // Sequences per batch
    .with_dataset("c4");          // Dataset name

println!("Samples: {}", config.num_samples());
println!("Sequence length: {}", config.sequence_length());
println!("Batch size: {}", config.batch_size());
println!("Dataset: {}", config.dataset());
```

## Parameter Guidelines

### Number of Samples

| Model Size | Recommended Samples |
|------------|---------------------|
| <1B params | 64-128 |
| 1-7B params | 128-256 |
| >7B params | 256-512 |

More samples improve accuracy estimation but increase memory and time.

### Sequence Length

Match your target use case:

```rust
// For chat/instruction models
let chat_config = CalibrationConfig::new()
    .with_sequence_length(2048);

// For code models
let code_config = CalibrationConfig::new()
    .with_sequence_length(4096);

// For document processing
let doc_config = CalibrationConfig::new()
    .with_sequence_length(8192);
```

### Batch Size

Trade-off between memory and efficiency:

```rust
// Memory-constrained (large models)
let low_mem = CalibrationConfig::new()
    .with_batch_size(1);

// Balanced
let balanced = CalibrationConfig::new()
    .with_batch_size(4);

// Fast calibration (small models, lots of VRAM)
let fast = CalibrationConfig::new()
    .with_batch_size(16);
```

### Dataset Selection

Choose data representative of your target domain:

| Dataset | Best For |
|---------|----------|
| c4 | General language modeling |
| wikitext | Wikipedia-style content |
| pile | Diverse text sources |
| code | Programming tasks |
| custom | Domain-specific applications |

```rust
// General purpose
let general = CalibrationConfig::new()
    .with_dataset("c4");

// Code-focused
let code = CalibrationConfig::new()
    .with_dataset("code");
```

## Integration with PruningConfig

Combine calibration with pruning configuration:

```rust
use entrenar::prune::{PruningConfig, PruneMethod, CalibrationConfig};

let pruning_config = PruningConfig::new()
    .with_method(PruneMethod::Wanda);

// Check if calibration is needed
if pruning_config.requires_calibration() {
    let calibration = CalibrationConfig::new()
        .with_num_samples(128)
        .with_sequence_length(2048)
        .with_batch_size(1)
        .with_dataset("c4");

    // Use calibration with pruning...
}
```

## Validation

The calibration config validates parameters:

```rust
// Invalid: zero samples
let bad_config = CalibrationConfig::new()
    .with_num_samples(0);

// Will fail validation...
```

## Memory Estimation

Estimate memory requirements for calibration:

```
Memory ≈ batch_size * seq_length * hidden_dim * 4 bytes * num_layers
```

For a 7B parameter model:
- hidden_dim ≈ 4096
- num_layers ≈ 32

With batch_size=1, seq_length=2048:
```
Memory ≈ 1 * 2048 * 4096 * 4 * 32 ≈ 1 GB per forward pass
```

## Best Practices

1. **Use representative data** - Calibration data should match deployment distribution
2. **Start small** - Begin with 64-128 samples and increase if needed
3. **Monitor activation statistics** - Check for numerical issues (NaN, Inf)
4. **Cache calibration results** - Avoid re-running for same model/data
5. **Batch appropriately** - Balance memory usage and throughput

## Wanda Calibration Details

Wanda computes per-channel activation norms using Welford's online algorithm:

```
For each layer j:
    norm_j = sqrt(sum(x_j^2) / n_samples)
    importance_ij = |w_ij| * norm_j
```

This requires a single forward pass through calibration data, making it efficient compared to Hessian-based methods.

## SparseGPT Calibration Details

SparseGPT requires more compute:

1. Collect input activations for each layer
2. Compute Hessian approximation: H ≈ X^T X
3. Use inverse Hessian for optimal weight updates

This takes longer but produces higher-quality pruned models.

## Troubleshooting

### Out of Memory

```rust
// Reduce batch size
let config = CalibrationConfig::new()
    .with_batch_size(1);

// Or reduce sequence length
let config = CalibrationConfig::new()
    .with_sequence_length(1024);
```

### Slow Calibration

```rust
// Reduce number of samples
let config = CalibrationConfig::new()
    .with_num_samples(64);
```

### Poor Pruning Quality

- Increase number of samples
- Use more representative dataset
- Check for distribution shift between calibration and deployment
