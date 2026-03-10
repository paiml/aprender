# Case Study: Continual Pre-Training (CPT)

**Ticket**: GH-448 | **Contract**: `finetune.md §cpt`

## Overview

Continual Pre-Training adapts a pre-trained language model to a new domain
by continuing causal language modeling on domain-specific corpora. Key
components:

- **LR Schedule**: Linear warmup + cosine decay (standard for CPT)
- **Data Mixing**: Blend domain and general data to prevent catastrophic forgetting
- **Replay Buffer**: Ring buffer of original training examples for experience replay
- **Progress Tracking**: EMA loss, domain/general sample counts

## API

```rust
use aprender::online::cpt::{CptConfig, CptSchedule, DataMixer, ReplayBuffer};

let config = CptConfig {
    learning_rate: 2e-5,
    warmup_steps: 100,
    total_steps: 10_000,
    domain_mix_ratio: 0.7,
    replay_buffer_size: 1000,
    ..Default::default()
};
config.validate()?;

let schedule = CptSchedule::new(&config);
let lr = schedule.lr_at_step(500); // Linear warmup → cosine decay

let mut mixer = DataMixer::new(config.domain_mix_ratio, config.seed);
let batch = mixer.mix_batches(&domain_tokens, &general_tokens, 32);

let mut replay = ReplayBuffer::new(1000);
replay.add(vec![1, 2, 3]);
let samples = replay.sample(4, 42);
```

## Falsification Tests

| Test | Property |
|------|----------|
| FALSIFY-CPT-001 | LR schedule is always non-negative |
| FALSIFY-CPT-002 | Data mixer respects ratio bounds |
| FALSIFY-CPT-003 | Replay buffer never exceeds capacity |

## Run the Example

```bash
cargo run --example continual_pretraining
```
