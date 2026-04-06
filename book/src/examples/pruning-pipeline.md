# Case Study: Pruning Pipeline

This example demonstrates the end-to-end pruning workflow using Entrenar, including schedule configuration, calibration
setup, and pipeline management.

## Running the Example

```bash
cargo run --example pruning_pipeline
```

## Overview

The example showcases:
- Configuring three pruning schedules (OneShot, Gradual, Cubic)
- Setting up magnitude and Wanda pruning methods
- Configuring calibration for activation-weighted methods
- Understanding pipeline stages

## Code Walkthrough

### 1. OneShot Pruning Schedule

Single-step pruning at a specified training step:

```rust
use entrenar::prune::PruningSchedule;

let oneshot = PruningSchedule::OneShot { step: 1000 };

// Sparsity before pruning step
println!("Before: {:.0}%", oneshot.sparsity_at_step(999) * 100.0);  // 0%

// Sparsity at and after pruning step
println!("After: {:.0}%", oneshot.sparsity_at_step(1000) * 100.0);  // 100%
```

### 2. Gradual Pruning Schedule

Linear interpolation from initial to final sparsity:

```rust
let gradual = PruningSchedule::Gradual {
    start_step: 1000,
    end_step: 5000,
    initial_sparsity: 0.0,
    final_sparsity: 0.5,
    frequency: 500,
};

// Sparsity at various steps
for step in [1000, 2000, 3000, 4000, 5000] {
    println!("Step {}: {:.1}%", step, gradual.sparsity_at_step(step) * 100.0);
}
```

Output:
```
Step 1000: 0.0%
Step 2000: 12.5%
Step 3000: 25.0%
Step 4000: 37.5%
Step 5000: 50.0%
```

### 3. Cubic Pruning Schedule (Zhu & Gupta)

Smooth S-curve that prunes aggressively early:

```rust
let cubic = PruningSchedule::Cubic {
    start_step: 0,
    end_step: 10000,
    final_sparsity: 0.7,
};

// Formula: s_t = s_f * (1 - (1 - t/T)^3)
for step in [0, 2500, 5000, 7500, 10000] {
    println!("Step {}: {:.1}%", step, cubic.sparsity_at_step(step) * 100.0);
}
```

Output:
```
Step     0: 0.0%
Step  2500: 48.8%
Step  5000: 61.3%
Step  7500: 68.9%
Step 10000: 70.0%
```

### 4. Magnitude Pruning Configuration

Simple pruning using weight magnitude (no calibration needed):

```rust
use entrenar::prune::{PruningConfig, PruneMethod, SparsityPatternConfig};

let magnitude_config = PruningConfig::new()
    .with_method(PruneMethod::Magnitude)
    .with_target_sparsity(0.5)
    .with_pattern(SparsityPatternConfig::Unstructured)
    .with_schedule(gradual.clone());

println!("Method: {}", magnitude_config.method().display_name());
println!("Requires calibration: {}", magnitude_config.requires_calibration());  // false
```

### 5. Wanda Pruning Configuration

Activation-weighted pruning with N:M structured sparsity:

```rust
let wanda_config = PruningConfig::new()
    .with_method(PruneMethod::Wanda)
    .with_target_sparsity(0.5)
    .with_pattern(SparsityPatternConfig::nm_2_4())
    .with_schedule(PruningSchedule::OneShot { step: 0 });

println!("Method: {}", wanda_config.method().display_name());
println!("Requires calibration: {}", wanda_config.requires_calibration());  // true
```

### 6. Calibration Configuration

Set up calibration data for Wanda/SparseGPT:

```rust
use entrenar::prune::CalibrationConfig;

let calibration_config = CalibrationConfig::new()
    .with_num_samples(128)
    .with_sequence_length(2048)
    .with_batch_size(1)
    .with_dataset("c4");

println!("Samples: {}", calibration_config.num_samples());
println!("Sequence length: {}", calibration_config.sequence_length());
println!("Batch size: {}", calibration_config.batch_size());
println!("Dataset: {}", calibration_config.dataset());
```

### 7. Pipeline Stages

The pruning workflow progresses through defined stages:

```rust
use entrenar::prune::PruningStage;

let stages = [
    PruningStage::Idle,
    PruningStage::Calibrating,
    PruningStage::ComputingImportance,
    PruningStage::Pruning,
    PruningStage::FineTuning,
    PruningStage::Evaluating,
    PruningStage::Exporting,
    PruningStage::Complete,
];

for (i, stage) in stages.iter().enumerate() {
    let status = if stage.is_active() {
        "Active"
    } else if stage.is_terminal() {
        "Terminal"
    } else {
        "Waiting"
    };
    println!("{}. {:20} {}", i + 1, stage.display_name(), status);
}
```

### 8. Configuration Validation

Validate configurations before running:

```rust
match magnitude_config.validate() {
    Ok(()) => println!("Magnitude config: Valid"),
    Err(e) => println!("Magnitude config: Invalid - {}", e),
}

match wanda_config.validate() {
    Ok(()) => println!("Wanda config: Valid"),
    Err(e) => println!("Wanda config: Invalid - {}", e),
}
```

## Expected Output

```
╔══════════════════════════════════════════════════════════════╗
║         Pruning Pipeline with Entrenar                       ║
║         End-to-end model compression workflow                ║
╚══════════════════════════════════════════════════════════════╝

📋 Schedule 1: OneShot Pruning
   Prune at step: 1000
   Sparsity before step 1000: 0%
   Sparsity at step 1000: 100%
   Sparsity after step 1000: 100%

📋 Schedule 2: Gradual Pruning
   Start: step 1000, End: step 5000
   Initial sparsity: 0%, Final sparsity: 50%
   Pruning frequency: every 500 steps
   Sparsity progression:
     Step  1000: 0.0%
     Step  2000: 12.5%
     Step  3000: 25.0%
     Step  4000: 37.5%
     Step  5000: 50.0%

📋 Schedule 3: Cubic Pruning (Zhu & Gupta)
   Formula: s_t = s_f * (1 - (1 - t/T)^3)
   Final sparsity: 70%
   Sparsity progression:
     Step     0: 0.0%
     Step  2500: 48.8%
     Step  5000: 61.3%
     Step  7500: 68.9%
     Step 10000: 70.0%

⚙️  Config 1: Magnitude Pruning (No Calibration)
   Method: Magnitude
   Requires calibration: false
   Target sparsity: 50%
   Pattern: Unstructured

⚙️  Config 2: Wanda Pruning (Requires Calibration)
   Method: Wanda
   Requires calibration: true
   Pattern: 2:4 N:M Sparsity
   Theoretical sparsity: 50%

📊 Calibration Configuration
   Samples: 128
   Sequence length: 2048
   Batch size: 1
   Dataset: c4

🔄 Pipeline Stages
   1. Idle                 Waiting
   2. Calibrating          Active
   3. Computing Importance Active
   4. Pruning              Active
   5. Fine-tuning          Active
   6. Evaluating           Active
   7. Exporting            Active
   8. Complete             Terminal

✓ Validating Configurations
   Magnitude config: Valid
   Wanda config: Valid

╔══════════════════════════════════════════════════════════════╗
║                    Pipeline Summary                          ║
╠══════════════════════════════════════════════════════════════╣
║  Pruning Methods:                                            ║
║    - Magnitude (L1/L2) - No calibration needed               ║
║    - Wanda - Activation-weighted, needs calibration          ║
║    - SparseGPT - Hessian-based, needs calibration            ║
║                                                              ║
║  Sparsity Patterns:                                          ║
║    - Unstructured - Maximum flexibility                      ║
║    - N:M (2:4, 4:8) - Hardware-accelerated on Ampere         ║
║    - Block - Coarse-grained structured                       ║
║                                                              ║
║  Schedules:                                                  ║
║    - OneShot - Single pruning event                          ║
║    - Gradual - Linear interpolation                          ║
║    - Cubic - Zhu & Gupta (2017) formula                      ║
╚══════════════════════════════════════════════════════════════╝
```

## Key Takeaways

1. **Choose the right schedule** - OneShot for post-training, Gradual/Cubic for training-time
2. **Match method to needs** - Magnitude is simple, Wanda/SparseGPT for higher quality
3. **Consider hardware** - Use N:M patterns for GPU acceleration
4. **Validate early** - Catch configuration errors before expensive computation
5. **Monitor stages** - Track pipeline progress for debugging and logging

## Related Documentation

- [Pruning Overview](../pruning/overview.md)
- [Pruning Schedules](../pruning/schedules.md)
- [Calibration](../pruning/calibration.md)
- [Pipeline Stages](../pruning/pipeline.md)
