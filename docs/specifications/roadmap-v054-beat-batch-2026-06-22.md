# v0.54.0 correctness-beat batch (PMAT-904+) — exhaustive hunt 2026-06-22

Source: `v054-exhaustive-beat-hunt` workflow (3 rounds, 8 lenses, loop-until-dry, 3-vote
adversarial verify; 133 agents). **27 confirmed → ~22 distinct → 11 EV-ranked work-items.**
Excludes all v0.52.0 (PMAT-889..898) + v0.53.0/beat (PMAT-899..903, prepped) obligations.

## Ranked work-items (each 3-vote adversarially verified, RED-on-main confirmed)

| # | EV | Pillar | Item | Files | Obligation |
|---|----|--------|------|-------|------------|
| 1 | 9.6 | P2 autograd | **Norm-backward SWEEP A** — LayerNorm + RMSNorm + BatchNorm1d + GroupNorm grad_fns (HEADLINE: transformers non-fine-tunable today, γ frozen) | normalization/mod.rs, group_norm.rs, grad_fn.rs | OBLIG-{LAYERNORM,RMSNORM,BATCHNORM1D,GROUPNORM}-BACKWARD-GRAD-FLOW |
| 2 | 9.2 | P2 autograd | **Functional-activation SWEEP B** — F::silu + F::gelu + F::leaky_relu + F::log_softmax (gelu/leaky are broken twins — wired ops exist, free F:: rebuild from_vec) | nn/functional.rs, grad_fn.rs | OBLIG-{SILU,FUNCTIONAL-GELU,FUNCTIONAL-LEAKYRELU,LOGSOFTMAX}-BACKWARD-GRAD |
| 3 | 9.0 | P1 scipy.stats | **f32-gamma SWEEP** — t-test/ANOVA/chi-square return NaN p-values for df≳72 (one root cause: raw-space gamma() overflows f32 at z≥36) | hypothesis.rs, beta_continued_fraction.rs | OBLIG-CHISQUARE/HYPOTHESIS-PVALUE-FINITE |
| 4 | 8.4 | P4 fail-closed | **APR-load SWEEP** — reject vocab_size≠embed-rows AND weight-shape≠config-dims | realizar loader_apr_quantized.rs, loading.rs | OBLIG-APR-VOCAB-EMBED-CONSISTENT + OBLIG-APR-WEIGHT-SHAPE-MATCHES-CONFIG |
| 5 | 8.0 | P4 roundtrip | **F16-export RNE** — f32_slice_to_f16_bytes must match half::f16::from_f32 (subnormal flush + truncation; BF16 twin already fixed PMAT-859) | safetensors.rs:283-288 | OBLIG-SAFETENSORS-F16-EXPORT-RNE |
| 6 | 7.8 | P2 loss | **CE label_smoothing** — (1-eps)/C should be eps/C → 3.2× wrong loss | loss.rs:297 | OBLIG-CE-LABEL-SMOOTHING-UNIFORM-MASS |
| 7 | 7.4 | P2 autograd | **Conv/Pool/GRU SWEEP C** — Conv1d/2d + MaxPool/AvgPool/Flatten + GRU gate backward (CNN/RNN untrainable; heavier im2col/col2im) | conv, pooling, gru | OBLIG-CONV/POOL-FLATTEN/GRU-GATE-BACKWARD-GRAD-FLOW |
| 8 | 6.8 | P4 fail-closed | **Special-token ≥ vocab** — eos/bos ≥ vocab_size loads silently → never-stops | realizar config.rs:828-890 | OBLIG-SPECIAL-TOKEN-WITHIN-VOCAB |
| 9 | 6.4 | P1 sklearn | **Weighted-KNN zero-distance** — exact-dup neighbor capped at 1.0 vs sklearn ∞-weight | classification/gaussian_nb.rs:177,276 | OBLIG-KNN-WEIGHTED-ZERO-DISTANCE |
| 10 | 6.0 | P1 sklearn | **Preprocessing SWEEP** — StandardScaler f64-accum + MinMax constant-feature + LogReg balanced-weight | preprocessing/mod.rs, classification/mod.rs | OBLIG-STANDARDSCALER-F64-ACCUM + 2 |
| 11 | 4.6 | P1 metrics | **Metric-eps SWEEP** — log_loss + MAPE finfo eps; avg_precision no-positive→0.0 | metrics/probabilistic.rs, regression.rs | OBLIG-LOGLOSS-EPS-FINFO + 2 |

Dedup applied: F16-export filed twice (one fix); chi-square + t/ANOVA collapse to one gamma root cause.

## Completeness gaps (next hunt waves — still rich, un/under-swept)
1. Conv/Pool/GRU only `is_some()`-probed, not numeric grad-checked; **LSTM/vanilla-RNN/Embedding backward** never probed (likely severed).
2. **Attention/transformer-block** forward (MHA, SDPA, qwen2/bert attention) not swept for severed-graph (masking/scaling/reshape via Tensor::new).
3. f32-gamma overflow likely also NaNs **other distribution CDFs** sharing gamma()/beta_function (Beta/Gamma/F/Student-t, poisson/binomial, KS/Mann-Whitney, Pearson/Spearman p-values).
4. **f32-accum cancellation** beyond StandardScaler: PCA covariance, running_mean/var, online metrics.
5. **Quantized round-trip** fidelity (Q4K/Q5K/Q6K/Q8_0 quantize→dequantize, GGUF→APR transpose LAYOUT-001) — only F16 IEEE swept.
6. More **estimator decision-boundary parity** (SVM, DecisionTree split/tie-break, KMeans++ init, PCA sign).
7. **Pillar-3 serve runtime** barely touched (RoPE theta/scaling, repeat_penalty/top_p/top_k vs llama.cpp, chat-template stop-tokens, quantized GEMV) — most user-facing.
8. Loss reduction parity (KLDiv, NLL reduction modes, BCEWithLogits pos_weight, Huber delta, focal) not re-verified vs PyTorch values.

## Cadence note
Ship rate is queue-limited (~1 PR/hr, trueno-SIGSEGV-flake-throttled). v0.52.0 (8/10) → cut → push
5 prepped (v0.53.0+beat) → THEN v0.54.0. Prep order: gamma/F16/APR-load (clean files, no autograd
rebase) ahead of the grad_fn.rs autograd sweeps (1/2/7) which pile up and must serialize-rebase.
New contracts (autograd-equivalence-beat, per-norm) must be authored + pv-validated IN the fix PR
(single-line falsifier refs) so obligations ship wired, not declared.
