# Equation Model Cards (EMC)

Governing equations for entrenar training components.

## EMC-001: Cross-Entropy Loss

**Equation**: `L = -1/N * sum(log(softmax(logits)[target]))`

**Domain**: logits in R^{B x S x V}, targets in {0..V-1}^{B x S}

**Numerical bounds**: L >= 0, L <= log(V) for uniform distribution

**Implementation**: `src/train/transformer_trainer/cuda_trainer.rs`

**Verification**: `tests/loss_function_accuracy.rs`

## EMC-002: AdamW Optimizer

**Equations**:
- m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
- v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
- m_hat = m_t / (1 - beta1^t)
- v_hat = v_t / (1 - beta2^t)
- theta_t = theta_{t-1} - lr * (m_hat / (sqrt(v_hat) + eps) + wd * theta_{t-1})

**Domain**: g_t in R^d, beta1 in (0,1), beta2 in (0,1), eps > 0

**Numerical bounds**: |m_hat| <= max(|g|) / (1 - beta1), v_hat >= 0

**Implementation**: `src/optim/adamw.rs`, `src/train/transformer_trainer/cuda_trainer.rs`

**Verification**: `tests/optimizer_correctness.rs`

## EMC-003: RMSNorm

**Equation**: `y = x * rsqrt(mean(x^2) + eps) * gamma`

**Domain**: x in R^{B x S x H}, gamma in R^H, eps > 0

**Numerical bounds**: |y| <= |x| * |gamma| / sqrt(eps) (worst case)

**Implementation**: `src/autograd/cuda_forward/rms_norm.rs`

**Verification**: `tests/normalization_correctness.rs`

## EMC-004: Rotary Position Embedding (RoPE)

**Equations**:
- theta_i = base^{-2i/d} for i in 0..d/2
- cos_m = cos(m * theta_i), sin_m = sin(m * theta_i)
- [x_even, x_odd] -> [x_even * cos - x_odd * sin, x_even * sin + x_odd * cos]

**Domain**: m in {0..max_seq_len-1}, x in R^{B x S x H}

**Implementation**: `src/autograd/cuda_forward/rope.rs`

## EMC-005: Scaled Dot-Product Attention

**Equation**: `Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k) + mask) * V`

**Domain**: Q in R^{B x H x S x D}, K in R^{B x H x S x D}, V in R^{B x H x S x D}

**Numerical bounds**: attention weights in [0, 1], sum to 1 per query position

**Implementation**: `src/autograd/cuda_forward/attention.rs`
