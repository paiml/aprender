# Case Study: Reinforcement Learning on Verifiable Rewards (RLVR)

**Ticket**: GH-450
**Module**: `aprender::online::rlvr`

## Overview

RLVR trains language models using binary-verifiable reward signals (math correctness, code pattern matching) instead of learned reward models. Uses REINFORCE policy gradient with KL penalty.

## Key Components

- **`RlvrConfig`** — Learning rate, KL coefficient, reward scale, samples
- **`RlvrLoss`** — Policy gradient, KL penalty, total loss
- **`VerifiableReward`** trait — Binary verification interface
- **`MathReward`** — Verifies numeric answers (`\boxed{}`, `answer is N`, `= N`)
- **`CodeReward`** — Verifies code patterns (`must contain:`, `expected output:`)
- **`RlvrMetrics`** — Batch-level accuracy, KL, loss aggregation

## Run

```bash
cargo run --example rlvr
```

## Falsification Tests

| ID | Property | Status |
|----|----------|--------|
| FALSIFY-RLVR-001 | Policy gradient is finite | Falsified (holds) |
| FALSIFY-RLVR-002 | KL penalty is finite | Falsified (holds) |
| FALSIFY-RLVR-003 | Reward scores in [0, 1] | Falsified (holds) |
