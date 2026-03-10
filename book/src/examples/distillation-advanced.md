# Case Study: Advanced Distillation Strategies

**Ticket**: GH-451 | **Contract**: `distill.md §hidden-state, §quant-aware, §online`

## Overview

Three advanced strategies extending the standard KL-divergence distillation:

### 1. Hidden-State Matching (FitNets)

Match intermediate hidden representations between teacher and student via
learned linear projections. Each projection maps teacher dimension → student
dimension using MSE loss.

```rust
use aprender::online::distillation_advanced::{
    HiddenStateConfig, HiddenStateDistiller,
};

let config = HiddenStateConfig {
    teacher_dim: 768,
    student_dim: 256,
    layer_map: vec![(3, 1), (7, 2), (11, 3)],
    projection_lr: 0.001,
    ..Default::default()
};
let mut distiller = HiddenStateDistiller::new(config);

// Compute loss and update projections
let loss = distiller.hidden_loss(&teacher_hiddens, &student_hiddens);
distiller.update_projections(&teacher_hiddens, &student_hiddens);
```

### 2. Quantization-Aware Distillation

Fake quantization during training: weights are quantized → dequantized so the
student learns to be robust to quantization noise. Error diffusion spreads
quantization error to neighbors (Floyd-Steinberg style).

```rust
use aprender::online::distillation_advanced::{
    QuantAwareConfig, QuantAwareDistiller,
};

let distiller = QuantAwareDistiller::new(QuantAwareConfig {
    bits: 4,
    symmetric: true,
    error_diffusion: 0.5,
    poly_degree: 3,
    ..Default::default()
});

let fake_q = distiller.fake_quantize(&weights);
let diffused = distiller.fake_quantize_diffused(&weights);
let error = distiller.quantization_error(&weights);
```

### 3. Online Distillation

Concurrent teacher/student training without precomputing teacher logits.
Optional EMA smoothing reduces variance from stochastic teacher outputs.

```rust
use aprender::online::distillation_advanced::{
    OnlineDistillConfig, OnlineDistiller,
};

let mut distiller = OnlineDistiller::new(OnlineDistillConfig {
    ema_decay: Some(0.999),
    temperature: 3.0,
    alpha: 0.7,
    ..Default::default()
});

let loss = distiller.step(&student_logits, &teacher_logits, &labels)?;
```

## Falsification Tests

| Test | Property |
|------|----------|
| FALSIFY-DISTILL-HS-001 | Hidden-state loss is always non-negative |
| FALSIFY-DISTILL-QA-001 | Fake quantization is idempotent |
| FALSIFY-DISTILL-ONLINE-001 | Online loss is finite with extreme logits |

## Run the Example

```bash
cargo run --example distillation_advanced
```
