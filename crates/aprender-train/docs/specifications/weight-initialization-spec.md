# Weight Initialization Specification

## Status: CRITICAL BUG — entrenar#309

**Root cause of 16x convergence gap between entrenar and PyTorch (albor canary).**

## 1. Problem Statement

All weight matrices in entrenar's `Transformer::new()` use deterministic
sinusoidal initialization:

```rust
// embedding.rs:24
(0..n).map(|i| ((i as f32 * 0.111).sin() * scale)).collect()

// attention.rs:293 (w_q, w_k, w_v, w_o)
(0..n).map(|i| ((i as f32 * 0.123).sin() * scale)).collect()

// feedforward.rs:37 (w_gate, w_up, w_down)
(0..n).map(|i| ((i as f32 * 0.567).sin() * scale)).collect()
```

This is a placeholder that was never replaced with proper random initialization.
PyTorch/HuggingFace uses `normal(0, 0.02)` per the LLaMA configuration.

## 2. Evidence (Albor PyTorch Canary)

Identical architecture (399M tied LLaMA), same data (codeparrot-clean), same
hyperparameters — only difference is weight initialization:

| Step | PyTorch (normal 0.02) | Entrenar (sinusoidal) | Gap |
|------|----------------------|----------------------|-----|
| 0 | loss=10.60 | loss=10.40 | 2% |
| 1K | val_ppl=50.0 | val_ppl=805 | **16x** |
| 2K | val_ppl=12.1 | val_ppl=793 | **66x** |
| 4K | val_ppl=6.1 | val_ppl=362 | **59x** |

Canary: `scripts/canary_pytorch.py` with `--seed 123`, tied embeddings,
held-out val set. Run via `uv run` with CUDA PyTorch.

## 3. Literature: How LLaMA Initializes Weights

### 3.1 HuggingFace Transformers Implementation

Source: [modeling_llama.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py)

```python
class LlamaPreTrainedModel(PreTrainedModel):
    def _init_weights(self, module):
        std = self.config.initializer_range  # default: 0.02
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
```

All linear layers and embeddings: `N(0, 0.02)`.

### 3.2 Original LLaMA Paper

[Touvron et al. (2023)](https://arxiv.org/abs/2302.13971) — "LLaMA: Open and
Efficient Foundation Language Models." The paper does not explicitly specify
initialization, following the GPT family convention.

### 3.3 GPT-2 Residual Scaling

[Radford et al. (2019)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
— GPT-2 scales residual layer weights by `1/sqrt(2*N)` where N is the number
of layers. HuggingFace implements this for output projections (o_proj, down_proj):

```python
std = self.config.initializer_range / math.sqrt(2 * self.config.num_hidden_layers)
```

### 3.4 Weight Initialization Theory

[He et al. (2015)](https://arxiv.org/abs/1502.01852) — "Delving Deep into
Rectifiers" establishes that proper initialization variance should be
`2/fan_in` for ReLU networks. For transformers with pre-norm (LLaMA), the
simpler `N(0, 0.02)` works because RMSNorm stabilizes activations.

[Glorot & Bengio (2010)](https://proceedings.mlr.press/v9/glorot10a.html) —
Xavier initialization: `Uniform(-sqrt(6/(fan_in+fan_out)), sqrt(6/(fan_in+fan_out)))`.
Appropriate for sigmoid/tanh but not standard for modern transformers.

## 4. Specification: Correct Initialization

### 4.1 Default (Match HuggingFace LLaMA)

For all `nn.Linear` equivalents and `nn.Embedding`:
```
weight ~ N(0, initializer_range)
initializer_range = 0.02
bias = 0 (if present)
```

### 4.2 Optional: Residual Scaling (GPT-2 style)

For output projections in attention and FFN (o_proj, down_proj):
```
std = initializer_range / sqrt(2 * num_layers)
```
This prevents activation variance from growing with depth.

### 4.3 RMSNorm

```
gamma = 1.0 (vector of ones)
```
Already correct in entrenar (`RMSNorm::new` initializes to ones).

### 4.4 Seeded Reproducibility

The `seed` parameter from training config MUST propagate to weight
initialization. Each layer should use a deterministic sub-seed:

```
layer_seed = hash(global_seed, layer_idx, param_name)
```

This ensures:
- Same seed → identical weights → reproducible training
- Different seed → different weights → different initialization landscape

## 5. Files to Modify

| File | Current (sinusoidal) | Target (random normal) |
|------|---------------------|----------------------|
| `transformer/embedding.rs` | `sin(i*0.111)*scale` | `N(0, 0.02)` seeded |
| `transformer/attention.rs` | `sin(i*0.123)*q_scale` | `N(0, 0.02)` seeded |
| `transformer/feedforward.rs` | `sin(i*0.567)*scale` | `N(0, 0.02)` seeded |
| `transformer/feedforward.rs` (encoder) | `sin(i*0.567)*scale` | `N(0, 0.02)` seeded |

## 6. Test Impact

Three tests currently depend on deterministic sinusoidal init:
1. `falsify_e7e_init_deterministic` — update to verify seeded determinism
2. `test_deterministic_training_reproducibility` — update to use same seed
3. `test_falsify_f_conv_006_higher_rank_lower_loss` — LoRA test, may need threshold adjustment

## 7. Provable Contract

See `contracts/weight-initialization-v1.yaml` (C-INIT-001).

## 8. References

- [Touvron et al. (2023)](https://arxiv.org/abs/2302.13971) — LLaMA paper
- [Radford et al. (2019)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — GPT-2 paper (residual scaling)
- [He et al. (2015)](https://arxiv.org/abs/1502.01852) — Kaiming initialization
- [Glorot & Bengio (2010)](https://proceedings.mlr.press/v9/glorot10a.html) — Xavier initialization
- [Zhang & Sennrich (2019)](https://arxiv.org/abs/1910.05895) — RMSNorm
- [HuggingFace LLaMA source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py)
- [open_llama init discussion](https://github.com/openlm-research/open_llama/issues/98)
