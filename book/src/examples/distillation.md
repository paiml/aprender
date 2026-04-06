# Knowledge Distillation Examples

Examples demonstrating knowledge distillation with temperature-scaled KL divergence.

## Basic Distillation

```rust
use entrenar::distill::DistillationLoss;

// Create distillation loss with temperature=4.0, alpha=0.7
let kd_loss = DistillationLoss::new(4.0, 0.7);

// Compute loss combining soft targets and hard labels
let loss = kd_loss.compute(&student_logits, &teacher_logits, &labels);
```

## Multi-Teacher Ensemble

```rust
use entrenar::distill::EnsembleDistiller;

// Combine knowledge from multiple teachers
let teachers = vec![teacher1, teacher2, teacher3];
let weights = vec![0.5, 0.3, 0.2];

let ensemble = EnsembleDistiller::weighted(&weights);
let soft_targets = ensemble.combine(&teacher_outputs);
```

## Progressive Layer-Wise

```rust
use entrenar::distill::ProgressiveDistiller;

// Match intermediate layer representations
let distiller = ProgressiveDistiller::new()
    .layer_weight(0, 0.1)  // Early layers
    .layer_weight(6, 0.3)  // Middle layers
    .layer_weight(11, 0.6); // Final layers
```

## Running the Example

```bash
cargo run --example distillation
```

## See Also

- [Temperature-Scaled KL Divergence](../distillation/temperature-kl.md)
- [Multi-Teacher Ensemble](../distillation/multi-teacher.md)
- [Progressive Layer-Wise](../distillation/progressive.md)
