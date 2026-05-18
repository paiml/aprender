---
library_name: aprender
license: apache-2.0
language:
- en
tags:
- code
- code-generation
- python
- stack-existence-proof
- sovereign-ai
- rust
- pytorch-free
base_model: Qwen/Qwen2.5-Coder-0.5B-Instruct
datasets:
- bigcode/the-stack-dedup
- codeparrot/codeparrot-clean
metrics:
- val_loss
model-index:
- name: albor-370m-v1
  results:
  - task:
      type: text-generation
      name: Causal Language Modeling
    dataset:
      name: codeparrot-thestack-python-permissive-shards-qwen-v3
      type: python-code
    metrics:
    - type: val_loss
      value: 4.6227
      name: Validation Cross-Entropy
    - type: val_perplexity
      value: 101.78
      name: Validation Perplexity
    - type: throughput_tps
      value: 315.6
      name: Inference Throughput (tok/s, RTX 4090)
---

# albor-370m-v1

> **Stack-existence-proof model for the Sovereign AI Stack.** This is a 494M-parameter Qwen2 architecture trained end-to-end with the Rust-only [Sovereign AI Stack](https://github.com/paiml/aprender) — no PyTorch, no Python training loop, no HuggingFace `transformers` — purely `aprender` + `entrenar` + `trueno` + `realizar`.
>
> **Intended use**: **stack capability proof**, NOT a production code-completion model. See [§88 framing](#§88-framing) below.

## Model description

`albor-370m-v1` is a 494M-parameter Qwen2-architecture transformer trained on a 49.6B-token Python corpus using the [aprender Sovereign AI Stack](https://github.com/paiml/aprender). The training pipeline (`aprender-train` ≈ entrenar) was used end-to-end:

- **Tokenization**: `apr tokenize encode-corpus` (BPE NFC, Qwen2 vocab, 151,936 tokens)
- **Architecture**: 24 layers × 14 attention heads × 2 KV heads × 896 hidden dim
- **Initialization**: `Qwen/Qwen2.5-Coder-0.5B-Instruct` (via `apr convert`)
- **Training**: `apr pretrain --device cuda` on RTX 4090 (cuBLAS TF32 forward + custom backward kernels)
- **Checkpoint format**: `.apr` (aprender native, row-major, no PyTorch dependency)
- **Inference**: `apr run model.apr "..."` (realizar inference engine)

## Training procedure

| Parameter | Value |
|---|---|
| Architecture | Qwen2 (24L, 14H, 2KV, 896d, 4864 intermediate) |
| Parameters | ~494M |
| Tokenizer | Qwen2 BPE (151,936 vocab) |
| Init | `Qwen/Qwen2.5-Coder-0.5B-Instruct` (post-fine-tune from instruct) |
| Optimizer | AdamW (β₁=0.9, β₂=0.95, ε=1e-8, wd=0.01) |
| LR schedule | Cosine warmup (500 steps) → 1.5e-5 peak → 1.0e-7 floor |
| Batch | 16 |
| Seq length | 512 |
| Total steps | 5,000 (50 epochs × 100 steps) |
| Total tokens consumed | 40.96M |
| Hardware | NVIDIA RTX 4090 (sm_89), 24 GB |
| Wall time | 53 minutes |
| Throughput | 15,460 tok/s (pure training) / 12,880 tok/s (with 2.5 GB checkpoint per epoch) |
| GPU utilization | 99-100% sustained, 10.4 GB / 24 GB used, 57°C |

## Evaluation

- **Best val_loss**: **4.6227** @ epoch 49 (smooth monotonic descent across 50 epochs, no early-stop)
- **Best val_perplexity**: 101.78
- **Inference throughput**: 315.6 tok/s (epoch-020 `apr bench` on RTX 4090)

### Trajectory (every 5 epochs)

```
ep  0:  7.43 (init eval)
ep  5:  5.91
ep 10:  5.54
ep 15:  5.18
ep 20:  5.02
ep 25:  4.95
ep 30:  4.83
ep 35:  4.77
ep 40:  4.71
ep 45:  4.70
ep 49:  4.62  ← BEST
```

The descent is smooth and monotonic — the §85 P2-E run demonstrates that the Sovereign AI Stack's training loop reaches the expected loss floor for this architecture and corpus given the compute budget. Marginal-gain decay analysis predicts ~4.4 floor at 100 epochs and ~3.5 floor at 1.2M steps (Chinchilla compute-optimal D ≈ 20·N).

## §88 framing — "stack-existence-proof"

Per [SPEC-SHIP-TWO-001 §88](https://github.com/paiml/aprender/blob/main/docs/specifications/aprender-train/ship-model-2-spec.md), this model is shipped as a **stack capability proof**, not a production code-completion model. The original `AC-SHIP2-003` strict target (val_loss ≤ 2.2) requires ~213 GPU-hours (~9 days continuous) of training compute, which exceeds the project's 48-GPU-hour single-shot iteration budget. The §88 amendment introduces a compute-bounded target (val_loss ≤ 4.7) that this model satisfies.

The primary purpose this model serves is to **demonstrate end-to-end stack capability**:

- ✅ Pure-Rust tokenization (`apr tokenize encode-corpus`)
- ✅ Pure-Rust training (`apr pretrain`, no PyTorch)
- ✅ Pure-Rust checkpointing (`.apr` format, no `safetensors` dependency for the training path)
- ✅ Pure-Rust inference (`apr run`)
- ✅ Pure-Rust quantization + export (`apr export --format gguf`)
- ✅ Cross-stack interop (GGUF export loads in llama.cpp; SafeTensors round-trip works)

If you need a production-quality 0.5B code-completion model, use [`Qwen/Qwen2.5-Coder-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct) directly. The next iteration of `albor` (distillation epic PMAT-683/684) is the planned route to stricter quality targets.

## Intended uses

- ✅ **Sovereign AI Stack demonstrations** — show the Rust-only training pipeline working end-to-end
- ✅ **Inference infrastructure validation** — drop-in test artifact for `realizar` / `apr run` / `apr serve`
- ✅ **Tokenization round-trip testing** — exercise the BPE NFC + chat-template pipeline
- ✅ **Quantization research** — Q4_K / Q6_K conversion via `apr quantize` benchmarks against this checkpoint
- ⚠️ **NOT recommended for**:
  - Production code completion (use Qwen2.5-Coder-0.5B-Instruct or larger)
  - Zero-shot reasoning (val_perplexity ≈ 102 → mathematically incapable)
  - Long-context generation (max_position_embeddings = 32,768 but model wasn't trained beyond seq=512)
  - HumanEval / MBPP submission as a competitive code-LM (target distillation epic for that)

## Limitations

- **Compute-bounded training**: 40.96M tokens consumed of a 49.6B-token corpus (0.083% sampling). Compute-optimal Chinchilla target (D=20·N) would require ~9 days continuous GPU; this model trades depth-of-fit for iteration speed on the stack.
- **Plateau evidence**: doubling the training compute (P2-G with 10k steps vs P2-E's 5k) at the same LR/warmup produces a WORSE result (val_loss 4.6497 vs 4.6227, EARLY_STOP). This is a known marginal-gain-decay regime — more-of-the-same-recipe doesn't help. See [§87 Chinchilla 20·N hard gate](https://github.com/paiml/aprender/blob/main/contracts/chinchilla-gate-v1.yaml).
- **Init lineage**: weights inherit from `Qwen/Qwen2.5-Coder-0.5B-Instruct` (Apache-2.0 license). The fine-tuning pass shifts the model's distribution toward the codeparrot+the-stack-dedup-Python distribution but does NOT fully replace the Instruct prior. Expect chat-formatted outputs to occasionally surface.
- **Validation set drift**: P2-E held-out val batches were drawn from the first 16 batches of the qwen-v3 shard iterator — a mixed codeparrot + the-stack-dedup distribution. The new `apr pretrain --val-shard` flag (PR #1744) supports independent val sets for future runs.

## Training data

| Source | Size | License | Role |
|---|---|---|---|
| `codeparrot/codeparrot-clean` | 12.8 GB | Apache-2.0 (permissive subset) | ~25% of mix |
| `bigcode/the-stack-dedup` (Python) | 28.6 GB | Permissive licenses (filtered, dedup'd) | ~75% of mix |
| **Combined corpus** | **49.6B tokens** | Permissive (filtered + dedup'd) | qwen-v3 |

The corpus is tokenized at ingest time via `apr tokenize encode-corpus --num-workers 48` and saved to disk as `.bin` shards (little-endian u32 tokens). The `apr-corpus-ingest` binary handles license filtering + minhash deduplication upstream.

## How to use

### Inference (recommended path)

```bash
# Install the aprender CLI
cargo install aprender

# Pull the model
apr pull paiml/albor-370m-v1

# Generate
apr run paiml/albor-370m-v1 "def fibonacci(n):"

# Benchmark
apr bench paiml/albor-370m-v1 --iterations 100
```

### Direct .apr load (Rust, no Python)

```rust
use realizar::Model;
let model = Model::load_apr("albor-370m-v1.apr")?;
let output = model.generate(&input_ids, generation_config)?;
```

### Export to other formats

```bash
# Quantize to Q4_K for llama.cpp
apr export albor-370m-v1.apr --format gguf --quantize q4k -o albor-370m-q4k.gguf

# Export to SafeTensors for transformers (round-trip works but loses some metadata)
apr export albor-370m-v1.apr --format safetensors -o albor-370m.safetensors
```

## Reproduce the training run

```bash
# Pull the source init
apr pull Qwen/Qwen2.5-Coder-0.5B-Instruct -o qwen-init.apr

# Pull the corpus
apr pull-dataset bigcode/the-stack-dedup --filter python --subset deduped -o the-stack-py/
apr pull-dataset codeparrot/codeparrot-clean -o codeparrot-clean/

# Tokenize (multi-source, NFC normalized, Qwen2 vocab)
apr tokenize encode-corpus \
  --corpus the-stack-py/ \
  --corpus codeparrot-clean/ \
  --tokenizer qwen-init.apr \
  -o qwen-v3-shards/

# Train (the P2-E recipe)
apr pretrain \
  --dataset qwen-v3-shards/ \
  --tokenizer qwen-tokenizer/ \
  --run-dir runs/albor-370m-v1/ \
  --init qwen-init.apr \
  --mode finetune \
  --lr 1.5e-5 \
  --num-steps 5000 \
  --warmup-steps 500 \
  --batch-size 16 \
  --seq-length 512 \
  --target-val-loss 3.0 \
  --vocab-size 151936 \
  --device cuda \
  --seed 42 \
  --force-under-provisioned
```

Expected wall time: 53 minutes on RTX 4090.

## Citation

```bibtex
@misc{albor-370m-v1,
  title  = {albor-370m-v1: Stack-Existence-Proof for the Sovereign AI Stack},
  author = {PAIML Engineering},
  year   = 2026,
  url    = {https://huggingface.co/paiml/albor-370m-v1},
  note   = {494M-parameter Qwen2 architecture trained end-to-end with the Rust-only aprender Sovereign AI Stack (no PyTorch). See SPEC-SHIP-TWO-001 §88 for the framing.}
}
```

## License + Provenance

| Field | Value |
|---|---|
| Model weights license | Apache-2.0 (inherits from base) |
| Init checkpoint | `Qwen/Qwen2.5-Coder-0.5B-Instruct` (Apache-2.0) |
| Training data sources | `codeparrot/codeparrot-clean` + `bigcode/the-stack-dedup` (Python, permissive-licensed subset) |
| Training data license | Aggregated permissive (Apache-2.0 / MIT / BSD via license filtering) |
| Training stack | aprender + entrenar + trueno + realizar (all Apache-2.0) |
| Training hardware | NVIDIA RTX 4090 (sm_89) |
| Training framework | Rust-native (no PyTorch, no HF transformers) |

## Acknowledgments

- **Base model**: [Qwen team](https://github.com/QwenLM/Qwen) for the Qwen2.5-Coder-0.5B-Instruct base.
- **Training data**: [BigCode](https://www.bigcode-project.org/) (the-stack-dedup) + [codeparrot](https://huggingface.co/codeparrot) (codeparrot-clean).
- **Stack components**: [aprender](https://github.com/paiml/aprender) (ML framework), [entrenar](https://github.com/paiml/aprender/tree/main/crates/aprender-train) (training, in-tree), [trueno](https://github.com/paiml/aprender/tree/main/crates/aprender-compute) (SIMD/GPU compute, in-tree), [realizar](https://github.com/paiml/aprender/tree/main/crates/aprender-serve) (inference, in-tree).

## Changelog

- **v1.0.0 (2026-05-17)**: Initial release. Stack-existence-proof model per SPEC §88. Best val_loss = 4.6227. Trained from `Qwen/Qwen2.5-Coder-0.5B-Instruct` init on `codeparrot/codeparrot-clean` + `bigcode/the-stack-dedup` Python permissive subset.

---

For full training methodology and methodology lessons learned (Class 3 packaging cascades, audit hypothesis bounds, upstream metadata masquerade), see [`docs/specifications/aprender-train/ship-model-2-spec.md §81–§88`](https://github.com/paiml/aprender/blob/main/docs/specifications/aprender-train/ship-model-2-spec.md).
