# Case Study: Direct Preference Optimization (DPO)

**Ticket**: GH-449 | **Contract**: `finetune.md §dpo`

## Overview

DPO (Rafailov et al. 2023) aligns language models using preference pairs
without training a separate reward model. The policy is optimized directly
to maximize the margin between chosen and rejected responses.

**Loss**: `L = -log σ(β · (log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x)))`

## API

```rust
use aprender::online::dpo::{DpoConfig, DpoLoss, DpoMetrics, PreferencePair};

let loss = DpoLoss::new(DpoConfig {
    beta: 0.1,
    label_smoothing: 0.0,
    use_reference: true,
    ..Default::default()
});

let pair = PreferencePair {
    chosen_logprob: -1.0,
    rejected_logprob: -5.0,
    ref_chosen_logprob: -2.0,
    ref_rejected_logprob: -3.0,
};

let l = loss.compute(&pair);
let (grad_c, grad_r) = loss.gradient(&pair);  // Zero-sum gradients
let metrics = DpoMetrics::from_batch(&loss, &pairs);
```

## Key Properties

- **Zero-sum gradients**: grad_chosen + grad_rejected = 0
- **Non-negative loss**: DPO loss is always >= 0
- **SimPO variant**: Set `use_reference: false` for reference-free training
- **Label smoothing**: Reduces overfitting to noisy preference annotations

## Falsification Tests

| Test | Property |
|------|----------|
| FALSIFY-DPO-001 | Loss is always non-negative |
| FALSIFY-DPO-002 | Gradients are zero-sum |
| FALSIFY-DPO-003 | Correct reward ordering when policy is correct |

## Run the Example

```bash
cargo run --example dpo_preference
```
