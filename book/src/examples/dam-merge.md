# Case Study: Differentiable Adaptive Merging (DAM)

**Ticket**: GH-446
**Module**: `aprender::online::dam`

## Overview

DAM learns per-model merge coefficients that minimize reconstruction loss on a calibration set. Unlike fixed-weight merging, DAM adapts weights per-tensor using Nelder-Mead simplex optimization.

## Key Components

- **`DamConfig`** — Learning rate, iterations, regularization, seed
- **`DamLoss`** — MSE loss, L2 regularization, gradient step
- **`softmax`** — Numerically stable logit-to-probability conversion
- **`optimize_coefficients`** — Nelder-Mead simplex optimizer
- **`DamReport`** — Final loss, convergence status, coefficients

## Run

```bash
cargo run --example dam_merge
```

## Falsification Tests

| ID | Property | Status |
|----|----------|--------|
| FALSIFY-DAM-001 | Softmax sums to 1 | Falsified (holds) |
| FALSIFY-DAM-002 | Optimization reduces loss | Falsified (holds) |
| FALSIFY-DAM-003 | MSE loss is non-negative | Falsified (holds) |
