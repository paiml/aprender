# Case Study: Trueno Compute Integration

This chapter demonstrates the integration of trueno 0.8.8+ compute infrastructure with aprender's ML training pipeline.

## Overview

The `aprender::compute` module provides ML-specific wrappers around trueno's simulation testing infrastructure, following Toyota Way principles:

- **Jidoka**: Built-in quality - stop on defect (NaN/Inf detection)
- **Poka-Yoke**: Mistake-proofing via type-safe backend selection
- **Heijunka**: Leveled testing across compute backends

## Features

| Feature | Description | Use Case |
|---------|-------------|----------|
| Backend Selection | Auto CPU/GPU dispatch | Optimize compute for data size |
| Training Guards | NaN/Inf detection | Training stability |
| Divergence Checking | Cross-backend validation | GPU correctness verification |
| Reproducibility | Deterministic seeding | Reproducible experiments |

## Backend Selection (Poka-Yoke)

Automatically select the optimal compute backend based on data size:

```rust
use aprender::compute::{select_backend, should_use_gpu, BackendCategory};

// Auto-select backend
let category = select_backend(data.len(), gpu_available);

match category {
    BackendCategory::SimdOnly => {
        // N < 1,000: Pure SIMD (low overhead)
    }
    BackendCategory::SimdParallel => {
        // 1,000 <= N < 100,000: SIMD + Rayon parallelism
    }
    BackendCategory::Gpu => {
        // N >= 100,000: GPU compute (if available)
    }
}

// Helper functions
if should_use_gpu(data.len()) {
    // Offload to GPU
}
```

### Decision Thresholds (TRUENO-SPEC-012)

| Data Size | Backend | Rationale |
|-----------|---------|-----------|
| N < 1,000 | SIMD Only | Parallelization overhead exceeds benefit |
| 1,000 <= N < 100,000 | SIMD + Parallel | Rayon parallelism beneficial |
| N >= 100,000 | GPU | GPU offload cost amortized |

## Training Guards (Jidoka)

Detect numerical instabilities during training:

```rust
use aprender::compute::TrainingGuard;

let guard = TrainingGuard::new("epoch_1");

// After computing gradients
guard.check_gradients(&gradients)?;

// After weight update
guard.check_weights(&weights)?;

// After loss computation
guard.check_loss(loss)?;
```

### What Gets Detected

| Issue | Cause | Detection |
|-------|-------|-----------|
| NaN values | 0/0, sqrt(-1), log(0) | `check_gradients()`, `check_weights()` |
| Infinity | Overflow, 1/0 | `check_gradients()`, `check_weights()` |
| NaN loss | Gradient explosion | `check_loss()` |
| Infinite loss | Numerical overflow | `check_loss()` |

### Error Handling

```rust
use aprender::compute::TrainingGuard;
use aprender::error::AprenderError;

let guard = TrainingGuard::new("training_step_42");

match guard.check_gradients(&gradients) {
    Ok(()) => {
        // Continue training
    }
    Err(AprenderError::ValidationError { message }) => {
        // Jidoka triggered - stop and investigate
        eprintln!("Training stopped: {}", message);
        // Example: "Jidoka: NaN in gradients at training_step_42:nan"
    }
    Err(e) => {
        // Other error
    }
}
```

## Divergence Checking

Validate that different compute backends produce consistent results:

```rust
use aprender::compute::DivergenceGuard;

// Default ML tolerance (1e-5)
let guard = DivergenceGuard::default_tolerance("cpu_vs_gpu");

// Compare CPU and GPU results
let cpu_result = compute_on_cpu(&input);
let gpu_result = compute_on_gpu(&input);

guard.check(&cpu_result, &gpu_result)?;

// Custom tolerance for specific operations
let relaxed_guard = DivergenceGuard::new(0.01, "approximate_softmax");
relaxed_guard.check(&approx_result, &exact_result)?;
```

### Tolerance Guidelines

| Operation | Recommended Tolerance | Rationale |
|-----------|----------------------|-----------|
| Exact arithmetic | 0.0 | Bit-exact expected |
| FP32 operations | 1e-5 | IEEE 754 precision |
| Mixed precision | 1e-4 | FP16 accumulation |
| Approximate kernels | 1e-2 | Algorithmic differences |

## Reproducible Experiments

Ensure deterministic training with structured seeding:

```rust
use aprender::compute::ExperimentSeed;

// Derive all seeds from master
let seed = ExperimentSeed::from_master(42);

println!("Master: {}", seed.master);
println!("Data shuffle: {}", seed.data_shuffle);
println!("Weight init: {}", seed.weight_init);
println!("Dropout: {}", seed.dropout);

// Use in training
let mut rng_data = StdRng::seed_from_u64(seed.data_shuffle);
let mut rng_weights = StdRng::seed_from_u64(seed.weight_init);
let mut rng_dropout = StdRng::seed_from_u64(seed.dropout);
```

### Seed Derivation

Seeds are derived deterministically using LCG multipliers:

| Seed | Derivation | Use |
|------|------------|-----|
| `master` | Input | Experiment identifier |
| `data_shuffle` | `master * 6364136223846793005` | Dataset shuffling |
| `weight_init` | `master * 1442695040888963407` | Parameter initialization |
| `dropout` | `master * 2685821657736338717` | Dropout/regularization |

## API Reference

### Backend Selection

| Function | Description |
|----------|-------------|
| `select_backend(size, gpu_available)` | Returns recommended `BackendCategory` |
| `should_use_gpu(size)` | Returns `true` if size >= 100,000 |
| `should_use_parallel(size)` | Returns `true` if size >= 1,000 |

### TrainingGuard

| Method | Description |
|--------|-------------|
| `TrainingGuard::new(context)` | Create guard with context string |
| `check_gradients(&[f32])` | Check for NaN/Inf in gradients |
| `check_weights(&[f32])` | Check for NaN/Inf in weights |
| `check_loss(f32)` | Check for NaN/Inf loss value |
| `check_f64(&[f64], kind)` | Check f64 values |

### DivergenceGuard

| Method | Description |
|--------|-------------|
| `DivergenceGuard::new(tolerance, context)` | Create with custom tolerance |
| `DivergenceGuard::default_tolerance(context)` | Create with 1e-5 tolerance |
| `check(&[f32], &[f32])` | Compare two result arrays |

### ExperimentSeed

| Method | Description |
|--------|-------------|
| `ExperimentSeed::from_master(seed)` | Derive all seeds from master |
| `ExperimentSeed::new(...)` | Create with explicit seeds |
| `ExperimentSeed::default()` | Master seed = 42 |

## Running the Example

```bash
cargo run --example trueno_compute_integration
```

Output demonstrates:
1. **Backend Selection**: Auto-dispatch based on data size
2. **Training Guards**: NaN/Inf detection (Jidoka triggered)
3. **Divergence Checking**: Cross-backend tolerance validation
4. **Reproducibility**: Deterministic seed derivation

## Integration with Training Loops

```rust
use aprender::compute::{TrainingGuard, select_backend, ExperimentSeed};

fn train(data: &[f32], epochs: usize) -> Result<Vec<f32>> {
    let seed = ExperimentSeed::from_master(42);
    let backend = select_backend(data.len(), check_gpu_available());

    let mut weights = initialize_weights(seed.weight_init);

    for epoch in 0..epochs {
        let guard = TrainingGuard::new(format!("epoch_{}", epoch));

        // Forward pass
        let output = forward(&weights, data);

        // Backward pass
        let gradients = backward(&output, data);
        guard.check_gradients(&gradients)?;

        // Update weights
        update_weights(&mut weights, &gradients);
        guard.check_weights(&weights)?;

        // Compute loss
        let loss = compute_loss(&output, data);
        guard.check_loss(loss)?;

        println!("Epoch {}: loss = {:.4}", epoch, loss);
    }

    Ok(weights)
}
```

## Toyota Way Principles

| Principle | Implementation |
|-----------|----------------|
| **Jidoka** | `TrainingGuard` stops on NaN/Inf |
| **Poka-Yoke** | Type-safe `BackendCategory` selection |
| **Genchi Genbutsu** | Detailed error context in guards |
| **Heijunka** | Leveled backend thresholds |

## See Also

- [Trueno Ecosystem Integration Spec](../../docs/specifications/include-latest-trueno-features.md)
- [Case Study: Pipeline Verification](./pipeline-verification.md)
- [Case Study: Poka-Yoke Validation](./poka-yoke-validation.md)
- [Toyota Way: Jidoka](../toyota-way/jidoka.md)
