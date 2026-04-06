# Pruning Pipeline

The pruning pipeline manages the end-to-end workflow from model loading to compressed export. This chapter covers the
pipeline stages and state machine.

## Pipeline Stages

The pruning process follows a defined sequence of stages:

```rust
use entrenar::prune::PruningStage;

let stages = [
    PruningStage::Idle,               // Initial state
    PruningStage::Calibrating,        // Collecting activation statistics
    PruningStage::ComputingImportance,// Scoring weight importance
    PruningStage::Pruning,            // Applying sparsity masks
    PruningStage::FineTuning,         // Recovering accuracy
    PruningStage::Evaluating,         // Validating quality
    PruningStage::Exporting,          // Saving compressed model
    PruningStage::Complete,           // Pipeline finished
];
```

## Stage Details

### Idle

Initial state before pruning begins.

```rust
let stage = PruningStage::Idle;
assert!(!stage.is_active());
assert!(!stage.is_terminal());
```

### Calibrating

Collects activation statistics for Wanda/SparseGPT methods.

```rust
let stage = PruningStage::Calibrating;
assert!(stage.is_active());
println!("{}", stage.display_name());  // "Calibrating"
```

**Activities:**
- Forward pass through calibration data
- Compute per-layer activation norms
- Build Hessian approximations (SparseGPT)

**Skipped when:** Using magnitude pruning (no calibration needed)

### ComputingImportance

Scores each weight's importance using the configured method.

```rust
let stage = PruningStage::ComputingImportance;
assert!(stage.is_active());
```

**Activities:**
- Apply importance formula (|w|, |w|*activation, Hessian-based)
- Compute statistics (min, max, mean, std)
- Validate for numerical stability (NaN, Inf)

### Pruning

Generates and applies sparsity masks to zero out weights.

```rust
let stage = PruningStage::Pruning;
assert!(stage.is_active());
```

**Activities:**
- Sort weights by importance
- Generate masks based on target sparsity and pattern
- Apply masks to model weights
- Verify achieved sparsity matches target

### FineTuning

Recovers accuracy lost during pruning through continued training.

```rust
let stage = PruningStage::FineTuning;
assert!(stage.is_active());
```

**Activities:**
- Continue training with frozen sparsity pattern
- Adjust learning rate (typically lower)
- Monitor loss convergence
- Apply gradient updates only to non-zero weights

**Duration:** Typically 10-20% of original training steps

### Evaluating

Validates pruned model quality against benchmarks.

```rust
let stage = PruningStage::Evaluating;
assert!(stage.is_active());
```

**Activities:**
- Run evaluation benchmarks
- Compare to baseline (unpruned) model
- Check quality gates (max accuracy drop)
- Generate evaluation report

### Exporting

Saves the compressed model in deployment format.

```rust
let stage = PruningStage::Exporting;
assert!(stage.is_active());
```

**Activities:**
- Convert to sparse storage format
- Apply additional compression (quantization)
- Save model weights and configuration
- Verify exported model loads correctly

### Complete

Terminal state indicating successful pipeline completion.

```rust
let stage = PruningStage::Complete;
assert!(!stage.is_active());
assert!(stage.is_terminal());
```

## Stage Properties

Each stage exposes useful properties:

```rust
let stage = PruningStage::FineTuning;

// Display name for UI
println!("Stage: {}", stage.display_name());  // "Fine-tuning"

// Check if actively processing
if stage.is_active() {
    println!("Pipeline is working...");
}

// Check if pipeline is done
if stage.is_terminal() {
    println!("Pipeline complete!");
}
```

## Pipeline Flow Visualization

```
┌─────────┐
│  Idle   │
└────┬────┘
     │
     ▼
┌─────────────┐
│ Calibrating │ (skip if Magnitude)
└──────┬──────┘
       │
       ▼
┌───────────────────┐
│ComputingImportance│
└─────────┬─────────┘
          │
          ▼
    ┌─────────┐
    │ Pruning │
    └────┬────┘
         │
         ▼
   ┌───────────┐
   │FineTuning │
   └─────┬─────┘
         │
         ▼
   ┌───────────┐
   │Evaluating │
   └─────┬─────┘
         │
         ▼
   ┌───────────┐
   │ Exporting │
   └─────┬─────┘
         │
         ▼
   ┌──────────┐
   │ Complete │
   └──────────┘
```

## Monitoring Pipeline Progress

Display pipeline status with stage indicators:

```rust
fn print_pipeline_status(current: PruningStage) {
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

    for stage in &stages {
        let indicator = if *stage == current {
            if stage.is_active() { "🟢" } else { "✅" }
        } else {
            "⚪"
        };
        println!("{} {}", indicator, stage.display_name());
    }
}
```

Output:
```
✅ Idle
✅ Calibrating
🟢 Computing Importance  ← Currently here
⚪ Pruning
⚪ Fine-tuning
⚪ Evaluating
⚪ Exporting
⚪ Complete
```

## Error Handling

Pipeline stages can fail. Handle errors gracefully:

```rust
fn run_stage(stage: PruningStage) -> Result<(), PruneError> {
    match stage {
        PruningStage::Calibrating => {
            // Could fail if out of memory
        }
        PruningStage::ComputingImportance => {
            // Could fail with NaN/Inf weights
        }
        PruningStage::Pruning => {
            // Could fail with invalid pattern
        }
        // ...
    }
    Ok(())
}
```

## Best Practices

1. **Log stage transitions** - Track timing and progress
2. **Checkpoint between stages** - Enable restart on failure
3. **Validate at each stage** - Catch issues early
4. **Monitor memory** - Calibration and importance computation can spike
5. **Set quality gates** - Define acceptable accuracy drop thresholds

## Integration Example

Complete pipeline integration:

```rust
use entrenar::prune::{
    PruningConfig, PruneMethod, PruningSchedule,
    SparsityPatternConfig, CalibrationConfig, PruningStage
};

fn run_pruning_pipeline() {
    // Configure pruning
    let config = PruningConfig::new()
        .with_method(PruneMethod::Wanda)
        .with_target_sparsity(0.5)
        .with_pattern(SparsityPatternConfig::nm_2_4())
        .with_schedule(PruningSchedule::OneShot { step: 0 });

    // Configure calibration
    let calibration = CalibrationConfig::new()
        .with_num_samples(128)
        .with_sequence_length(2048)
        .with_dataset("c4");

    // Validate before starting
    config.validate().expect("Invalid config");

    // Run pipeline stages
    let stages = [
        PruningStage::Calibrating,
        PruningStage::ComputingImportance,
        PruningStage::Pruning,
        PruningStage::FineTuning,
        PruningStage::Evaluating,
        PruningStage::Exporting,
        PruningStage::Complete,
    ];

    for stage in &stages {
        println!("Starting: {}", stage.display_name());
        // Execute stage...
        println!("Completed: {}", stage.display_name());
    }
}
```
