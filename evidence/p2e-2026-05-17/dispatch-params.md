## P2-E dispatch params

Date: 2026-05-17
Ticket: PMAT-690 P2-E (per evidence/p2c-2026-05-17/findings.md §112-118)

### Hyperparameter changes vs P2-C
- peak LR: 5e-5 → 1.5e-5 (3.3× lower per Hoffmann et al. recommendation for under-provisioned re-runs)
- warmup_steps: 100 → 500 (5× longer; matches §82 P2-A finding that short warmup at low LR oscillates)
- target_val_loss: 3.0 (was 2.2) — gives more headroom before early-stop fires
- seed: 42 (held constant for parity with P2-C)
- num_steps: 5000 (same as P2-C, 1.59× Chinchilla ratio against 0.5B params)
- batch_size: 16 (same)
- seq_length: 512 (same)
- mode: finetune (same)

### Expected outcome
- IF the +0.2 val_loss gap in P2-C was hyperparameter-related, lower LR + longer warmup should produce val_loss < 4.71 (§82's baseline) within 27 epochs.
- IF the gap persists (val_loss ≈ 4.91 again), the binding constraint is NOT hyperparameters and the §84 audit pre-falsification is corroborated.

### Falsifiable prediction
P2-E val_loss best @ epoch 20 < 4.7 → hyperparameters were the binding constraint.
P2-E val_loss best @ epoch 20 ≥ 4.7 → hyperparameters not binding; need to escalate to P2-F (shared val set) or new architecture.

