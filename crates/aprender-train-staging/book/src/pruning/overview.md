# Pruning Overview

Entrenar provides a comprehensive pruning module for neural network compression, enabling efficient model deployment
through structured weight removal.

## What is Pruning?

Pruning removes redundant parameters from neural networks to:

- **Reduce model size** - Fewer parameters to store and load
- **Accelerate inference** - Less computation per forward pass
- **Lower energy consumption** - Reduced memory bandwidth and compute
- **Enable edge deployment** - Fit large models on resource-constrained devices

## Module Architecture

```
entrenar::prune
├── PruningConfig          # Configuration for pruning operations
├── PruningSchedule        # When and how to apply pruning
│   ├── OneShot            # Single pruning event
│   ├── Gradual            # Linear interpolation
│   └── Cubic              # Zhu & Gupta (2017) formula
├── PruneMethod            # Importance scoring algorithm
│   ├── Magnitude          # |w| or w^2
│   ├── Wanda              # Activation-weighted
│   └── SparseGPT          # Hessian-based
├── SparsityPatternConfig  # Sparsity structure
│   ├── Unstructured       # Any weight can be pruned
│   ├── NM                 # N non-zeros per M elements
│   └── Block              # Coarse-grained blocks
├── CalibrationConfig      # For activation-weighted methods
└── PruningStage           # Pipeline state machine
```

## Quick Start

```rust
use entrenar::prune::{PruningConfig, PruneMethod, PruningSchedule, SparsityPatternConfig};

// Simple magnitude pruning configuration
let config = PruningConfig::new()
    .with_method(PruneMethod::Magnitude)
    .with_target_sparsity(0.5)
    .with_pattern(SparsityPatternConfig::Unstructured)
    .with_schedule(PruningSchedule::OneShot { step: 0 });

// Validate configuration
config.validate()?;

// Check if calibration is needed
if config.requires_calibration() {
    // Set up calibration for Wanda/SparseGPT
}
```

## Pruning Methods

| Method | Calibration | Speed | Quality | Use Case |
|--------|-------------|-------|---------|----------|
| Magnitude | No | Fast | Good | General purpose |
| Wanda | Yes | Medium | Better | LLM compression |
| SparseGPT | Yes | Slow | Best | Critical accuracy |

### Magnitude Pruning

Removes weights with smallest absolute values. No calibration needed.

```rust
let config = PruningConfig::new()
    .with_method(PruneMethod::Magnitude);

assert!(!config.requires_calibration());
```

### Wanda Pruning

Weights AND Activations - considers both weight magnitude and input activation statistics.

```rust
use entrenar::prune::CalibrationConfig;

let config = PruningConfig::new()
    .with_method(PruneMethod::Wanda);

assert!(config.requires_calibration());

let calibration = CalibrationConfig::new()
    .with_num_samples(128)
    .with_sequence_length(2048)
    .with_dataset("c4");
```

## Sparsity Patterns

### Unstructured

Maximum flexibility - any weight can be pruned independently.

```rust
let pattern = SparsityPatternConfig::Unstructured;
```

### N:M Structured

Hardware-accelerated on NVIDIA Ampere+ GPUs. Common patterns:

```rust
// 2:4 pattern - 50% sparsity
let nm_2_4 = SparsityPatternConfig::nm_2_4();

// 4:8 pattern - 50% sparsity
let nm_4_8 = SparsityPatternConfig::nm_4_8();
```

### Block Sparsity

Prune entire blocks for efficient memory access:

```rust
let block = SparsityPatternConfig::Block {
    block_size: 32
};
```

## Pruning Schedules

### OneShot

Prune to target sparsity in a single step:

```rust
let schedule = PruningSchedule::OneShot { step: 1000 };

// Before step 1000: 0% sparsity
assert_eq!(schedule.sparsity_at_step(999), 0.0);

// At and after step 1000: target sparsity
assert_eq!(schedule.sparsity_at_step(1000), 1.0);
```

### Gradual

Linear interpolation from initial to final sparsity:

```rust
let schedule = PruningSchedule::Gradual {
    start_step: 1000,
    end_step: 5000,
    initial_sparsity: 0.0,
    final_sparsity: 0.5,
    frequency: 500,  // Update every 500 steps
};

// Sparsity increases linearly
assert_eq!(schedule.sparsity_at_step(3000), 0.25);
```

### Cubic (Zhu & Gupta)

Smooth cubic schedule that prunes aggressively early:

```rust
let schedule = PruningSchedule::Cubic {
    start_step: 0,
    end_step: 10000,
    final_sparsity: 0.7,
};

// Formula: s_t = s_f * (1 - (1 - t/T)^3)
```

## Pipeline Stages

The pruning pipeline progresses through stages:

```rust
use entrenar::prune::PruningStage;

let stages = [
    PruningStage::Idle,              // Not started
    PruningStage::Calibrating,       // Collecting activation stats
    PruningStage::ComputingImportance, // Scoring weights
    PruningStage::Pruning,           // Applying masks
    PruningStage::FineTuning,        // Recovering accuracy
    PruningStage::Evaluating,        // Validating quality
    PruningStage::Exporting,         // Saving compressed model
    PruningStage::Complete,          // Done
];

for stage in &stages {
    if stage.is_active() {
        println!("Processing: {}", stage.display_name());
    }
}
```

## Best Practices

1. **Start simple** - Use magnitude pruning first to establish baselines
2. **Gradual for high sparsity** - When targeting >50% sparsity
3. **Fine-tune after pruning** - Critical for accuracy recovery
4. **Match hardware** - Use N:M patterns for GPU acceleration
5. **Validate thoroughly** - Test on representative data

## Related Topics

- [Pruning Schedules](./schedules.md) - Detailed schedule configuration
- [Calibration](./calibration.md) - Setting up activation-weighted methods
- [Pipeline Stages](./pipeline.md) - Managing the pruning workflow
- [Pruning Pipeline Example](../examples/pruning-pipeline.md) - Complete working example
