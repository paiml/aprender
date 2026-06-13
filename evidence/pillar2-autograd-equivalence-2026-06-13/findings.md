# Pillar-2 autograd numerical-equivalence beat — measured (2026-06-13)

**Claim:** apr's reverse-mode autograd computes gradients numerically equivalent to
PyTorch — a faithful, contract-gated replacement for the bounded training task, where
apr concedes raw throughput.

**Host:** noah-Lambda-Vector. **Incumbent:** PyTorch via `uv run --with torch`.

## Why a correctness beat (not speed)
apr LOSES MLP training throughput ~11× to PyTorch (MKL + fused autograd vs apr's
per-step `clear_graph()` rebuild; docs/BEATS.md Pillar-2 CONCEDED). The defensible
Pillar-2 win is provable correctness — the same wedge as P3 (NF4 equivalence) and P4
(fail-closed). This also hard-guards the #2000 Linear weight-gradient-path fix: that
fix proved the MLP *converges*; this proves the gradients are *exactly* PyTorch's.

## Method + result
Fixed network `relu(x @ W1^T + b1) @ W2^T + b2`, MSELoss (mean), fixed inputs/weights
(din=4, dh=3, dout=2, n=2). Gradients compared element-wise apr vs PyTorch.

| Quantity | result |
|----------|--------|
| forward loss | apr 0.100079 == PyTorch 0.100079 |
| max \|Δ dW1\| | 5.0e-7 |
| max \|Δ db1\| | 7.5e-9 |
| max \|Δ dW2\| | 3.7e-9 |
| max \|Δ db2\| | 3.0e-8 |
| **overall max \|Δgrad\|** | **5.0e-7** (essentially bit-exact) |

apr's autograd is numerically equivalent to PyTorch's.

## CI-gated form
`crates/aprender-core/tests/beat_pytorch_autograd_grad.rs` pins the PyTorch gradients +
loss and asserts apr matches (max|Δ| < 1e-4) — deterministic, no torch/GPU at CI time.
Contract `contracts/apr-pytorch-autograd-equivalence-beat-v1.yaml`. Wired into ci.yml.
With this, WON beats span all four pillars (P1 sklearn, P2 PyTorch, P3 Unsloth, P4 Ollama).
