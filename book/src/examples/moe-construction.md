# Case Study: Mixture of Experts Construction

**Ticket**: GH-445
**Module**: `aprender::online::moe_construction`

## Overview

Constructs MoE architectures from multiple dense models. Each source model contributes expert FFN weights via round-robin assignment, with learned routing to select top-k experts per token.

## Key Components

- **`MoeConfig`** — Expert count, per-token activation, routing method
- **`RoutingMethod`** — TopK, SwitchTransformer, ExpertChoice
- **`RouterInit`** — Random, Uniform, Balanced initialization
- **`plan_moe_construction`** — Round-robin expert assignment planner
- **`compute_gate_weights`** — Router weight initialization (LAYOUT-002 row-major)
- **`compute_expert_load_balance`** — Coefficient of variation metric

## Run

```bash
cargo run --example moe_construction
```

## Falsification Tests

| ID | Property | Status |
|----|----------|--------|
| FALSIFY-MOE-001 | All assignments are valid | Falsified (holds) |
| FALSIFY-MOE-002 | Gate weights have correct dimensions | Falsified (holds) |
| FALSIFY-MOE-003 | Load balance is non-negative | Falsified (holds) |
