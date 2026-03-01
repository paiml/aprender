# APR Checkpoint Specification

**Version**: 1.3.0
**Status**: Implemented
**Created**: 2026-03-01
**Last Updated**: 2026-03-01
**Parent Spec**: [APR-SPEC.md](APR-SPEC.md) (v2.1.0)

---

## 1. Abstract

APR is the only model format that spans both training and inference. SafeTensors
stores weights for inference. GGUF stores quantized weights for inference. Neither
stores training state. PyTorch checkpoints store everything but use pickle
(arbitrary code execution risk) and have no alignment, no checksums, no
quantization.

APR checkpoints unify the model lifecycle: **train → checkpoint → resume →
evaluate → deploy** — all in one format, zero format conversion.

---

## 2. Design Principles

1. **Self-contained**: One `.apr` file = one deployable or resumable artifact.
   No sidecar files (no separate `optimizer.pt`, `scheduler.pt`, `rng_state.pth`).
2. **No pickle**: All state is tensors (binary) + metadata (JSON). Zero arbitrary
   code execution risk.
3. **Backward compatible**: Inference-only readers ignore training tensors via
   namespace prefix convention (`__training__.*`).
4. **Canonical ordering**: Tensors are sorted lexicographically by name before
   writing. `save → load → save` produces bit-identical output.
5. **SafeTensors interop**: APR can import/export SafeTensors checkpoints for
   HuggingFace ecosystem compatibility.
6. **Architecture-agnostic**: Metadata uses the HF `config.json` convention.
   Unknown fields pass through as `custom` JSON. Supports dense transformers
   (Qwen2.5), hybrid attention (Qwen3-Next), MoE, and future architectures
   without format changes.

---

## 3. Supported Architectures

APR checkpoints store architecture config as JSON metadata. The metadata schema
mirrors HuggingFace `config.json` so any model architecture is supported without
format changes.

### 3.1 Dense Transformer (Qwen2.5-Coder-0.5B)

Current SSC production model. Standard self-attention + FFN.

```json
{
  "architecture": "qwen2",
  "model_type": "qwen2",
  "hidden_size": 896,
  "num_hidden_layers": 24,
  "num_attention_heads": 14,
  "num_key_value_heads": 2,
  "intermediate_size": 4864,
  "vocab_size": 151936,
  "max_position_embeddings": 32768,
  "rope_theta": 1000000.0,
  "rms_norm_eps": 1e-06
}
```

### 3.2 Hybrid MoE (Qwen3-Coder-Next, 80B/3B active)

Next-generation target. Combines Gated DeltaNet (linear attention) + Gated
Attention + Mixture of Experts. 512 experts, 10 active per token.

```json
{
  "architecture": "qwen3_next",
  "model_type": "qwen3_next",
  "hidden_size": 2048,
  "num_hidden_layers": 48,
  "num_attention_heads": 16,
  "num_key_value_heads": 2,
  "head_dim": 256,
  "intermediate_size": 5120,
  "vocab_size": 151936,
  "max_position_embeddings": 262144,
  "rope_theta": 5000000.0,
  "rms_norm_eps": 1e-06,
  "partial_rotary_factor": 0.25,
  "full_attention_interval": 4,
  "linear_num_key_heads": 16,
  "linear_num_value_heads": 32,
  "linear_key_head_dim": 128,
  "linear_value_head_dim": 128,
  "linear_conv_kernel_dim": 4,
  "num_experts": 512,
  "num_experts_per_tok": 10,
  "moe_intermediate_size": 512,
  "shared_expert_intermediate_size": 512,
  "decoder_sparse_step": 1,
  "norm_topk_prob": true,
  "router_aux_loss_coef": 0.001
}
```

**APR handles this without format changes**: all fields stored in the JSON
metadata section. The `custom` HashMap in `AprV2Metadata` accepts arbitrary
key-value pairs. Tensor names follow HF convention (`model.layers.N.*`).

### 3.3 Architecture-Specific LoRA Targets

Different architectures have different LoRA target modules:

| Architecture | LoRA Targets | Adapters/Layer |
|-------------|-------------|----------------|
| qwen2 | q_proj, v_proj | 2 |
| qwen3_next (attention layers) | q_proj, v_proj | 2 |
| qwen3_next (DeltaNet layers) | q_proj, v_proj (linear) | 2 |
| qwen3_next (MoE experts) | gate_proj, up_proj (per expert) | varies |

The adapter checkpoint stores `target_modules` in metadata. The reader
reconstructs the correct adapter topology from this list + the architecture
config.

---

## 4. Checkpoint Taxonomy

### 4.1 Inference Checkpoint — `.apr`

Full model weights. Sufficient for `realizar` inference.

```
Tensors:
  model.embed_tokens.weight               # [vocab_size, hidden_size]
  model.layers.0.self_attn.q_proj.weight  # [hidden_size, hidden_size]
  model.layers.0.mlp.experts.0.gate_proj  # MoE expert (if applicable)
  ...
  classifier.weight                       # [num_classes, hidden_size]
  classifier.bias                         # [num_classes]

Metadata:
  model_type, architecture config (§3.1 or §3.2)
  chat_template, special_tokens, vocab_size
  quantization (if quantized)
```

For large models (e.g., Qwen3-Coder-Next at 159 GB), APR uses multi-file
sharding via `AprV2Writer::with_sharding()`. Each shard is a valid APR file
with its own header and CRC32.

### 4.2 Adapter Checkpoint — `.adapter.apr`

LoRA fine-tune output. Stores adapter weights + task head.
Base model weights are NOT included (referenced by hash).

```
Tensors:
  lora.0.q_proj.lora_a               # [rank, d_in]
  lora.0.q_proj.lora_b               # [d_out, rank]
  lora.0.v_proj.lora_a               # [rank, d_in]
  lora.0.v_proj.lora_b               # [d_out, rank]
  ... (per layer, per target module)
  classifier.weight                   # [num_classes, hidden_size]
  classifier.bias                     # [num_classes]

Metadata:
  __checkpoint__:
    schema_version: 1
    type: "adapter"
  model_type: "classify_pipeline"
  architecture: "qwen2"              # or "qwen3_next"
  base_model_hash: "sha256:..."      # canonical hash (§10.1)
  base_model_source: "hf://Qwen/Qwen2.5-Coder-0.5B"
  adapter_type: "lora"
  adapter_config:
    rank: 16
    alpha: 16.0
    target_modules: ["q_proj", "v_proj"]
    num_adapters: 48
  task_type: "SEQ_CLS"
  num_classes: 2
  class_labels: ["safe", "unsafe"]
  class_weights: [0.416, 1.584]
  val_loss: 0.42
  val_accuracy: 0.94
  epoch: 2
```

### 4.3 Training Checkpoint — `.ckpt.apr`

Extends adapter checkpoint with optimizer state, scheduler, and RNG.
Enables training resumption.

```
Tensors (model — same as §4.2):
  lora.*.*.lora_a, lora.*.*.lora_b
  classifier.weight, classifier.bias

Tensors (optimizer — AdamW first/second moments):
  __training__.optim.lora.0.q_proj.lora_a.exp_avg      # same shape as param
  __training__.optim.lora.0.q_proj.lora_a.exp_avg_sq   # same shape as param
  ... (per trainable parameter)
  __training__.optim.classifier.weight.exp_avg
  __training__.optim.classifier.weight.exp_avg_sq
  __training__.optim.classifier.bias.exp_avg
  __training__.optim.classifier.bias.exp_avg_sq

Metadata:
  (all fields from §4.2, plus:)
  __checkpoint__:
    schema_version: 1
    type: "training"
  __training__:
    epoch: 1
    step: 176
    global_step: 535
    total_epochs: 3
    steps_per_epoch: 359
    train_loss: 1.065
    train_accuracy: 0.83
    optimizer:
      type: "adamw"
      lr: 0.0001
      betas: [0.9, 0.999]
      eps: 1.0e-8
      weight_decay: 0.01
      step: 535
    scheduler:
      type: "cosine_with_warmup"
      warmup_steps: 107
      total_steps: 1077
      current_step: 535
      current_lr: 0.0000988
    rng_seed: 42
    data:
      train_samples: 14353
      val_samples: 1794
      data_hash: "sha256:..."
    provenance:
      tool: "entrenar v0.5.0"
      started_at: "2026-03-01T10:00:00Z"
      gpu: "NVIDIA GeForce RTX 4090"
```

---

## 5. Tensor Namespace Convention

All training-only tensors use the `__training__` prefix. Inference readers
skip tensors whose name starts with `__training__`.

| Prefix | Purpose | Inference reader | Training reader |
|--------|---------|-----------------|-----------------|
| `model.*` | Base model weights | Read | Read |
| `lora.*` | LoRA adapter weights | Read (merge) | Read |
| `classifier.*` | Task head | Read | Read |
| `__training__.optim.*` | Optimizer moments | **Skip** | Read |
| `__training__.grad_scaler.*` | AMP state | **Skip** | Read |

---

## 6. File Extensions

All three checkpoint types use the APR v2 binary format. The extension is a
human/tooling hint — not a format difference. Same reader parses all three.

| Extension | Type | Contents | Produced by | Consumed by |
|-----------|------|----------|-------------|-------------|
| `.apr` | Base model | All transformer weights | `apr convert` | realizar |
| `.adapter.apr` | Adapter | LoRA + classifier, no optimizer | entrenar | realizar, apr-cli |
| `.ckpt.apr` | Training | LoRA + classifier + optimizer | entrenar | entrenar (resume), apr-cli |

**Detection rule** (when extension is missing or untrusted):
1. Check `__checkpoint__.type` metadata field (authoritative)
2. Fallback: has `__training__` tensors → `.ckpt.apr`
3. Fallback: has `lora.*` tensors but no `__training__` → `.adapter.apr`
4. Fallback: has `model.layers.*` tensors → `.apr`

---

## 7. Tool Responsibilities

The APR checkpoint format is consumed by four external tools and 51+ `apr` CLI
subcommands. This section defines the complete interaction matrix.

### 7.1 External Tools

#### 7.1.1 alimentar (data pipeline)

No `.apr` interaction. Produces JSONL data splits consumed by entrenar.

#### 7.1.2 entrenar (training library)

**Writes**:
- `.ckpt.apr` — every epoch (ephemeral, enables resume)
- `.adapter.apr` — best val_loss and final epoch (permanent, deploy these)

**Reads**:
- `.ckpt.apr` — for `--resume` (restores model + optimizer + scheduler)

```bash
# Train from scratch
apr train apply --data train.jsonl --model-path ./qwen2.5-0.5b --output ./out/

# Resume from checkpoint
apr train apply --resume out/epoch-1.ckpt.apr --data train.jsonl --output ./out/
```

**Output directory**:
```
out/
├── epoch-0.ckpt.apr           # training checkpoint (resumable)
├── epoch-1.ckpt.apr
├── epoch-2.ckpt.apr
├── best.adapter.apr           # best val_loss (deploy this)
├── final.adapter.apr          # last epoch
└── training_log.jsonl         # metrics history
```

#### 7.1.3 realizar (inference engine)

**Reads**: `.apr` (base model) + `.adapter.apr` (fine-tuned adapter)

```bash
# Classify with adapter
realizar classify --adapter best.adapter.apr --model ./qwen2.5-0.5b "rm -rf /"
#   unsafe (0.97)

# Also works with .ckpt.apr — skips __training__.* tensors
realizar classify --adapter epoch-2.ckpt.apr --model ./qwen2.5-0.5b "echo hello"
#   safe (0.99)
```

**Filtered loading**: When opening `.ckpt.apr`, realizar uses
`AprReader::open_filtered(path, |name| !name.starts_with("__training__"))`
to avoid loading optimizer tensors into memory.

**Never writes** `.apr` files.

### 7.2 apr-cli Command-Checkpoint Interaction Matrix

Every `apr` subcommand is classified by which checkpoint types it reads/writes
and whether it must understand the `__training__` tensor namespace.

Legend: R = reads, W = writes, — = no interaction, S = strips, P = preserves

#### 7.2.1 Inference Commands

| Command | `.apr` | `.adapter.apr` | `.ckpt.apr` | `__training__` NS | SafeTensors |
|---------|--------|----------------|-------------|-------------------|-------------|
| `run` | R | R (merge adapter) | R (skip training) | Skip | R |
| `serve` | R | R (merge adapter) | R (skip training) | Skip | R |
| `chat` | R | R (merge adapter) | R (skip training) | Skip | R |

These commands load models for inference. When given `.ckpt.apr`, they filter
out `__training__.*` tensors. When given `.adapter.apr`, they require a
`--model` flag pointing to the base model and merge LoRA weights at load time.

#### 7.2.2 Inspection Commands (read-only)

| Command | `.apr` | `.adapter.apr` | `.ckpt.apr` | `__training__` NS | SafeTensors |
|---------|--------|----------------|-------------|-------------------|-------------|
| `inspect` | R | R | R | Display | R |
| `tensors` | R | R | R | List all | R |
| `debug` | R | R | R | Hex dump | R |
| `hex` | R | R | R | Analyze | R |
| `tree` | R | R | R | Show hierarchy | R |
| `flow` | R | R | R | Map data flow | R |
| `explain` | R | R | R | Explain | R |

```bash
# Inspect auto-detects checkpoint type from __checkpoint__.type metadata
apr inspect best.adapter.apr
#   Type: adapter
#   Architecture: qwen2 (896h, 24L)
#   Base: hf://Qwen/Qwen2.5-Coder-0.5B (sha256:a1b2...)
#   Task: classify (2 classes: safe, unsafe)
#   Tensors: 98 (48 LoRA + 2 classifier)
#   Val accuracy: 94.2%

apr inspect epoch-1.ckpt.apr
#   Type: training_checkpoint
#   Epoch: 1/3, Step: 359/1077
#   Tensors: 294 (98 model + 196 optimizer)
#   Train loss: 1.065

# List tensors with namespace grouping
apr tensors epoch-1.ckpt.apr
#   Model (98 tensors):
#     classifier.weight    [1792]  F32
#     classifier.bias      [2]     F32
#     lora.0.q_proj.lora_a [16, 896]  F32
#     ...
#   Training (196 tensors):
#     __training__.optim.classifier.weight.exp_avg    [1792]  F32
#     __training__.optim.classifier.weight.exp_avg_sq [1792]  F32
#     ...
```

#### 7.2.3 Validation Commands (read-only)

| Command | `.apr` | `.adapter.apr` | `.ckpt.apr` | `__training__` NS | SafeTensors |
|---------|--------|----------------|-------------|-------------------|-------------|
| `validate` | R | R | R | Validate | R |
| `check` | R | R | R | Validate | R |
| `lint` | R | R | R | Lint | — |
| `qa` | R | R | R | Validate | R |
| `canary check` | R | R | R | Test vectors | — |
| `qualify` | R | R | R | Smoke test | R |

**`validate` vs `check`**: `validate` is a 100-point quality scoring rubric
(metadata completeness, tensor statistics, format compliance). `check` is a
10-stage pipeline self-test (load → tokenize → forward → verify output). Both
must handle all three checkpoint types.

```bash
# Validate checkpoint integrity (format-level)
apr validate epoch-1.ckpt.apr
#   ✓ CRC32 header
#   ✓ CRC32 footer
#   ✓ Schema version: 1
#   ✓ 98 model tensors (adapter completeness: F-CKPT-001)
#   ✓ 196 optimizer tensors (training completeness: F-CKPT-002)
#   ✓ base_model_hash present
#   Score: 95/100

# Check pipeline integrity (end-to-end)
apr check best.adapter.apr --model ./qwen2.5-0.5b
#   Stage 1/10: Load model ✓
#   Stage 2/10: Load adapter ✓
#   Stage 3/10: Merge LoRA ✓
#   Stage 4/10: Tokenize ✓
#   ...
#   Stage 10/10: Verify output ✓
```

#### 7.2.4 Evaluation Commands (read-only)

| Command | `.apr` | `.adapter.apr` | `.ckpt.apr` | `__training__` NS | SafeTensors |
|---------|--------|----------------|-------------|-------------------|-------------|
| `eval` | R | R | R | Read epoch/metrics | R |
| `bench` | R | R | R | Skip | R |
| `profile` | R | R | R | Skip | R |
| `parity` | R | — | — | — | — |

`eval` on a `.ckpt.apr` should report the checkpoint's embedded metrics
(epoch, train_loss, val_loss) alongside freshly computed evaluation metrics.
This enables comparing stored vs actual metrics to detect drift.

```bash
# Evaluate adapter on held-out test set
apr eval best.adapter.apr --model ./qwen2.5-0.5b --data test.jsonl
#   Checkpoint metrics:  val_loss=0.42, val_acc=94.2%
#   Evaluated metrics:   test_loss=0.45, test_acc=93.8%

# Evaluate training checkpoint (same interface)
apr eval epoch-1.ckpt.apr --model ./qwen2.5-0.5b --data test.jsonl
#   Stored epoch: 1, train_loss: 1.065
#   Evaluated:    test_loss=0.89, test_acc=86.1%
```

#### 7.2.5 Format Conversion Commands

| Command | Reads | Writes | `__training__` NS | Notes |
|---------|-------|--------|-------------------|-------|
| `import` | SafeTensors, GGUF, `hf://` | `.apr` | — | HF → APR |
| `export` | `.apr`, `.adapter.apr`, `.ckpt.apr` | SafeTensors, GGUF, MLX, ONNX | **Strips** | APR → ecosystem |
| `convert` | `.apr` | `.apr` (optimized) | Preserves | In-format optimization |
| `quantize` | `.apr`, `.adapter.apr` | `.apr` (quantized) | N/A | Weight quantization |
| `compile` | `.apr` | Binary executable | **Strips** | Static binary |

**Command disambiguation** — these five commands have distinct purposes:

- **`import`**: External format → APR. One-way ingest from HuggingFace ecosystem.
- **`export`**: APR → external format. Strips `__training__.*` (inference export).
- **`convert`**: APR → APR. In-place optimization (repack, compress, reorder).
  Also used for `--strip-training` to convert `.ckpt.apr` → `.adapter.apr`.
- **`quantize`**: APR → APR. Specialization of convert for weight quantization
  (Q4K, Q8, INT4, INT8, FP16). Can quantize adapter weights too.
- **`compile`**: APR → standalone binary. Embeds model + runtime into executable.

```bash
# Import HF PEFT directory → APR adapter
apr import ./hf-adapter/ --output best.adapter.apr
# Import sharded base model (40 shards)
apr import ./Qwen3-Coder-Next/ --output qwen3-next.apr

# Export APR adapter → HF PEFT directory (strips __training__)
apr export best.adapter.apr --format safetensors --output ./hf-adapter/

# Convert: strip training state (ckpt → adapter)
apr convert --from epoch-2.ckpt.apr --to final.adapter.apr --strip-training

# Quantize adapter for smaller deployment
apr quantize best.adapter.apr --precision q4k --output best-q4k.adapter.apr
```

#### 7.2.6 Rosetta Commands (universal format bridge)

| Command | Reads | Writes | `__training__` NS | Notes |
|---------|-------|--------|-------------------|-------|
| `rosetta inspect` | Any format | — | Read | Format-agnostic inspection |
| `rosetta convert` | Any format | Any format | Conditional | Universal converter |
| `rosetta chain` | Any format | Any format | Conditional | Multi-step conversion |
| `rosetta verify` | Two files | — | Compare | Round-trip validation |
| `rosetta diff-tensors` | Two files | — | Compare | Tensor-level diff |
| `rosetta fingerprint` | Any format | JSON | Read | Statistical signatures |
| `rosetta validate-stats` | Any format | — | Read | Anomaly detection |

**`convert` vs `rosetta convert`**: `apr convert` is APR→APR optimization.
`rosetta convert` is the universal converter (APR↔SafeTensors↔GGUF). Rosetta
handles the cross-format matrix; `convert` handles in-format optimization.

```bash
# Rosetta: universal conversion
apr rosetta convert best.adapter.apr --to gguf --output best.gguf
apr rosetta convert model.safetensors --to apr --output model.apr

# Rosetta: verify round-trip fidelity
apr rosetta verify best.adapter.apr best-roundtrip.adapter.apr
#   ✓ 98/98 tensors match (max diff: 0.0)

# Rosetta: fingerprint for integrity tracking
apr rosetta fingerprint best.adapter.apr
#   {"format": "apr", "type": "adapter", "tensors": 98, "hash": "sha256:..."}
```

#### 7.2.7 Training Commands

| Command | Reads | Writes | `__training__` NS | Notes |
|---------|-------|--------|-------------------|-------|
| `train plan` | JSONL data | YAML plan | — | Pre-flight only, no GPU |
| `train apply` | Plan + data + model | `.ckpt.apr`, `.adapter.apr` | Creates | Full training loop |
| `finetune` | `.apr` base model | `.adapter.apr` or merged `.apr` | Creates | LoRA/QLoRA training |
| `tune` | `.apr` model config | — | Reads | HPO planning (no GPU) |
| `monitor` | Checkpoint dir | — | Reads metrics | Live training TUI |
| `diagnose` | Checkpoint dir | Analysis report | Reads full state | Five Whys root-cause |

**`train apply` vs `finetune`**: `train apply` executes a validated training
plan (plan/apply pattern). `finetune` is direct LoRA/QLoRA training without
the plan step. Both produce the same output artifacts. `train apply` is the
production path (validated, reproducible); `finetune` is the quick path
(interactive, exploratory).

```bash
# Production path: plan → apply
apr train plan --data train.jsonl --model-size 0.5B --output plan.yaml
apr train apply --plan plan.yaml --model-path ./qwen2.5-0.5b --output ./out/

# Quick path: direct finetune
apr finetune ./qwen2.5-0.5b --data train.jsonl --output ./out/

# Monitor live training (reads training_state.json + .ckpt.apr files)
apr monitor ./out/

# Diagnose failed training (reads .ckpt.apr files + metrics)
apr diagnose ./out/
#   Five Whys Analysis:
#   1. Why did loss plateau? → Learning rate too high after epoch 1
#   2. Why was LR too high? → Warmup fraction too small (0.05)
#   ...
```

#### 7.2.8 Model Operations

| Command | Reads | Writes | `__training__` NS | Notes |
|---------|-------|--------|-------------------|-------|
| `merge` | 2+ `.adapter.apr` | `.apr` (merged) | Strips | Merge adapters into base |
| `prune` | `.apr` | `.apr` (pruned) | Preserves | Structured/unstructured pruning |
| `distill` | Teacher `.apr` | Student `.apr` | Creates | Knowledge distillation |

**Adapter merging**: `apr merge` takes a base model + one or more adapters and
produces a single `.apr` with LoRA weights merged into base weights. Training
state is always stripped (the merged model is inference-ready).

```bash
# Merge adapter into base model (deploy as single .apr)
apr merge ./qwen2.5-0.5b best.adapter.apr --output merged.apr

# Merge multiple adapters (e.g., safety + code-quality)
apr merge ./qwen2.5-0.5b safety.adapter.apr quality.adapter.apr \
    --weights 0.7,0.3 --output multi-task.apr
```

#### 7.2.9 Publishing Commands

| Command | Reads | Writes | `__training__` NS | Notes |
|---------|-------|--------|-------------------|-------|
| `publish` | `.apr`, `.adapter.apr` | HF Hub | Strips for public | Push to HuggingFace |
| `pull` | `hf://` | `.apr` (cached) | From source | Download and cache |
| `list` | Cache dir | — | — | List cached models |
| `rm` | Cache dir | Delete | — | Remove from cache |
| `oracle` | Any format, `hf://` | — | Analyze | Identify model family |
| `compare-hf` | `.apr` | — | Compare | Parity check vs HF source |

```bash
# Publish adapter to HuggingFace (auto-converts to SafeTensors + config)
apr publish best.adapter.apr --repo paiml/ssc-binary-v3
#   Uploading:
#     adapter_model.safetensors (4.3 MB)
#     adapter_config.json
#     config.json
#     tokenizer.json

# Oracle: identify checkpoint type and contract compliance
apr oracle best.adapter.apr
#   Family: Qwen2.5-Coder
#   Size: 0.5B (adapter only: 1.08M trainable params)
#   Type: adapter (LoRA r16, Q/V projections)
#   Contract: F-CKPT-001 ✓ (adapter completeness)
```

#### 7.2.10 Data Pipeline Commands (no checkpoint interaction)

| Command | Reads | Writes | Notes |
|---------|-------|--------|-------|
| `data audit` | JSONL | Report | Dataset quality validation |
| `data split` | JSONL | Train/val/test JSONL | Stratified split |
| `data balance` | JSONL | Resampled JSONL | Class imbalance correction |

These commands operate on training data, not model files.

#### 7.2.11 GPU Analysis Commands (checkpoint-adjacent)

| Command | Reads | Notes |
|---------|-------|-------|
| `ptx` | PTX source | GPU kernel analysis, no model files |
| `ptx-map` | `.gguf`, `.apr` | Layer→kernel mapping |
| `cbtop` | `.gguf`, `.apr` | ComputeBrick pipeline monitor |

#### 7.2.12 Testing Commands

| Command | Reads | Writes | Notes |
|---------|-------|--------|-------|
| `canary create` | `.apr` | Canary test file | Create regression test |
| `canary check` | `.apr` + canary file | — | Verify against canary |
| `qualify` | Any format | — | Cross-command smoke test |
| `probar` | `.apr` | Visual test artifacts | Visual regression export |
| `showcase` | Multiple models | Benchmark report | Demo + baselines |

### 7.3 Checkpoint Type Behavior Summary

Every command that reads `.apr` files must handle all three types gracefully.
The behavior depends on checkpoint type:

| Behavior | `.apr` (base) | `.adapter.apr` | `.ckpt.apr` |
|----------|--------------|----------------|-------------|
| Inference | Load all weights | Require `--model` base, merge LoRA | Filter `__training__.*`, then same as adapter |
| Inspection | Show all tensors | Show adapter tensors + base ref | Show all tensors, grouped by namespace |
| Validation | Standard checks | F-CKPT-001 (adapter completeness) | F-CKPT-001 + F-CKPT-002 (training completeness) |
| Export | All tensors | Adapter tensors only | Strip `__training__.*`, export adapter |
| Merge | Base input | Adapter input | Strip `__training__.*`, use as adapter |

---

## 8. Checkpoint Lifecycle

### 8.1 During Training

```
epoch 0, step 0    → save epoch-0.ckpt.apr
epoch 0, step 359  → save epoch-0.ckpt.apr  [end of epoch]
                     save best.adapter.apr  [if val_loss improved]
epoch 1, step 359  → save epoch-1.ckpt.apr
epoch 2, step 359  → save epoch-2.ckpt.apr  [final]
                     save best.adapter.apr  [if val_loss improved]
                     save final.adapter.apr [always]
```

**Training checkpoints** are ephemeral (enable resume). Keep last N.
**Adapter checkpoints** are permanent (the trained model). Keep best + final.

### 8.2 Resume from Checkpoint

```
1. Open epoch-N.ckpt.apr
2. Verify __checkpoint__.schema_version <= SUPPORTED_VERSION
3. Verify __training__.data.data_hash matches training data file
   (hard error on mismatch; --allow-data-mismatch to override)
4. Load adapter tensors → pipeline.lora_layers, pipeline.classifier
5. Load __training__.optim.* → optimizer.exp_avg, optimizer.exp_avg_sq
6. Read __training__.epoch, __training__.step → resume position
7. Read __training__.scheduler → restore LR schedule position
8. Read __training__.rng_seed → reseed with seed + epoch for shuffle
9. Continue training from step N+1
```

---

## 9. SafeTensors Interoperability

### 9.1 Import: SafeTensors → APR

APR reads HuggingFace checkpoints in two forms:

**Single-file adapter** (typical PEFT LoRA output):
```
adapter_model.safetensors   → LoRA A/B matrices
adapter_config.json         → LoRA hyperparameters + base model reference
config.json                 → model architecture
tokenizer.json              → BPE vocabulary + merges
```

**Sharded base model** (e.g., Qwen3-Coder-Next, 40 shards × 4 GB):
```
model-00001-of-00040.safetensors
model-00002-of-00040.safetensors
...
model-00040-of-00040.safetensors
model.safetensors.index.json      → shard→tensor mapping
config.json                       → architecture config
tokenizer.json                    → BPE vocabulary + merges
```

**Import contract (C-IMPORT-ST-001)**:

- For sharded models: read `model.safetensors.index.json` to discover
  tensor→shard mapping, then load each shard
- Map HF tensor names to APR convention:
  ```
  HF:  base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight
  APR: lora.0.q_proj.lora_a
  ```
- Parse `adapter_config.json` → populate `adapter_config` metadata
- Parse `config.json` → populate architecture metadata (all fields, including
  MoE and hybrid attention fields from §3.2)
- Optionally embed `tokenizer.json` content in metadata
- Write single `.adapter.apr` or sharded `.apr` (for large base models)

### 9.2 Export: APR → SafeTensors

For HuggingFace Hub publishing. Reverse the import mapping.

**Export contract (C-EXPORT-ST-001)**:

- Write `adapter_model.safetensors` with HF-convention tensor names
- Write `adapter_config.json` from APR metadata
- Write `config.json` from APR architecture metadata
- Copy `tokenizer.json` from APR metadata (if embedded) or base model dir
- Skip all `__training__.*` tensors (inference-only export)

### 9.3 Why not just use SafeTensors?

| Requirement | SafeTensors | APR |
|-------------|-------------|-----|
| Optimizer state as tensors | Not designed for it | Yes (`__training__.*`) |
| CRC32 integrity | No | Yes |
| 64-byte alignment | No | Yes |
| Quantized storage (Q4K) | No | Yes |
| Compression (LZ4) | No | Yes |
| Self-contained metadata | String→String only | Unlimited JSON |
| Provenance chain | No | Yes |
| Canonical tensor ordering | No | Yes (sorted by name) |

SafeTensors is a wire format for tensor exchange. APR is a model lifecycle
format.

---

## 10. Provenance Chain

### 10.1 Canonical Model Hash

The `base_model_hash` uniquely identifies a model regardless of storage format
(APR, SafeTensors, GGUF). Defined as:

```
1. Collect all tensor names, sort lexicographically
2. For each tensor in sorted order, concatenate:
   - name as UTF-8 bytes
   - 0x00 separator
   - dtype as single byte (F32=0, F16=1, BF16=2, Q4=3, Q8=4)
   - shape dimensions as little-endian u64 values
   - raw tensor data bytes
3. SHA-256 hash the entire concatenated stream
```

This hash is format-independent: the same model produces the same hash whether
stored as SafeTensors, GGUF, or APR.

### 10.2 Provenance Metadata

Every APR checkpoint records its lineage:

```json
{
  "base_model_hash": "sha256:a1b2c3...",
  "base_model_source": "hf://Qwen/Qwen3-Coder-Next",
  "data_hash": "sha256:d4e5f6...",
  "parent_checkpoint_hash": "sha256:789abc...",
  "provenance": {
    "tool": "entrenar v0.5.0",
    "started_at": "2026-03-01T10:00:00Z",
    "gpu": "NVIDIA GeForce RTX 4090",
    "wall_time_seconds": 2836
  }
}
```

**Verification**: Given a checkpoint, you can verify:
1. Which base model it was fine-tuned from (hash match)
2. Which training data produced it (data hash)
3. Which earlier checkpoint it resumed from (parent hash)
4. Full training environment (tool version, hardware, timing)

---

## 11. Optimizer State Storage

### 11.1 AdamW

For each trainable parameter `P` with name `name`:

| Tensor | Shape | Dtype | Description |
|--------|-------|-------|-------------|
| `__training__.optim.{name}.exp_avg` | same as P | F32 | First moment (m_t) |
| `__training__.optim.{name}.exp_avg_sq` | same as P | F32 | Second moment (v_t) |

The optimizer step counter and hyperparameters (lr, betas, eps, weight_decay)
are stored in metadata, not as tensors.

### 11.2 Size Budget

| Model | Trainable Params | Adapter Size | Training Ckpt | Base Model |
|-------|-----------------|-------------|---------------|------------|
| Qwen2.5-0.5B + LoRA r16 | 1.08M | 4.3 MB | 13 MB | 1 GB |
| Qwen3-Next + LoRA r16 (attn only) | ~6M | 24 MB | 72 MB | 159 GB |
| Qwen3-Next + LoRA r16 (attn+MoE) | ~50M | 200 MB | 600 MB | 159 GB |

Training checkpoints are always 3× adapter size (model + 2× optimizer moments).

### 11.3 Compression

Optimizer moments are highly compressible (many near-zero values early in
training). LZ4 compression typically achieves 2-4x reduction.

Use `AprV2Writer::with_lz4_compression()` for training checkpoints.

---

## 12. File Size Comparison

For SSC binary classifier (Qwen2.5-0.5B + LoRA rank 16, 2-class):

| Format | Files | Total Size | Resume? | Deploy? |
|--------|-------|-----------|---------|---------|
| HF PEFT (SafeTensors) | 7+ files | ~15 MB | Yes (with pickle) | Yes |
| PyTorch checkpoint | 1 file | ~15 MB | Yes (pickle!) | No |
| APR adapter (§4.2) | 1 file | ~4 MB | No | Yes |
| APR training (§4.3) | 1 file | ~13 MB | Yes (no pickle) | Yes* |

*Training checkpoints can be used for inference — readers skip `__training__.*`.

---

## 13. Contracts

Contracts are defined in the provable-contracts YAML at
`provable-contracts/contracts/entrenar/apr-checkpoint-v1.yaml`.
This section mirrors the YAML for spec readability.

### 13.1 Write-Side Contracts (producing checkpoints)

| ID | Name | Severity | Status | Description |
|----|------|----------|--------|-------------|
| F-CKPT-001 | Adapter completeness | P0 | **Done** | APR file contains ALL `lora_a`/`lora_b` tensors + classifier head. `count == 2 + 2 × lora_layers.len()` |
| F-CKPT-002 | Schema version present | P0 | **Done** | `__checkpoint__.schema_version` exists in metadata |
| F-CKPT-003 | No training state in adapter | P1 | Deferred | `.adapter.apr` has zero `__training__.*` tensors |
| F-CKPT-004 | Training state completeness | P0 | **Done** | `.ckpt.apr` has optimizer moments for every trainable param. `count(optim) == 2 × count(trainable)` |
| F-CKPT-007 | Write NaN/Inf check | P0 | **Done** | No tensor contains NaN or Inf values at write time |
| F-CKPT-008 | Write shape validation | P0 | Deferred | Every tensor shape matches model config dimensions |
| F-CKPT-009 | Atomic writes | P0 | Deferred | Write to `.tmp`, fsync, rename. Crash never corrupts existing checkpoint |
| F-CKPT-010 | Dtype consistency | P1 | **Done** | All tensors use F32 unless explicitly quantized. Quantized checkpoints record scheme in metadata |
| F-CKPT-015 | Canonical ordering | P2 | **Done** | Tensors sorted lexicographically by name |
| F-CKPT-017 | Provenance hash | P2 | **Done** | `data_hash`, `base_model_source`, and `provenance` recorded in metadata |

### 13.2 Read-Side Contracts (consuming checkpoints)

| ID | Name | Severity | Status | Description |
|----|------|----------|--------|-------------|
| F-CKPT-005 | Resume equivalence | P0 | **Done** | `resume_from_apr_checkpoint()` restores optimizer state + weights |
| F-CKPT-006 | Data hash verification | P1 | **Done** | Resume rejects mismatched training data. Hard error; `--allow-data-mismatch` overrides |
| F-CKPT-011 | CRC32 verification | P0 | **Done** | Reader verifies CRC32 on open. Corrupt file → hard error (AprV2Reader) |
| F-CKPT-012 | SafeTensors header validation | P0 | N/A | Reader validates SafeTensors header before parsing tensors |
| F-CKPT-013 | Post-load NaN scan | P0 | Deferred | Reader rejects tensors containing NaN/Inf after load |
| F-CKPT-014 | Shape-config validation | P0 | Deferred | Loaded tensor shapes match model architecture config |
| F-CKPT-016 | Backward compatibility | P1 | **Done** | `AprReader::open_filtered()` skips `__training__.*` |
| F-CKPT-018 | SafeTensors round-trip | P2 | Deferred | `import(export(import(dir))) == import(dir)` (bit-identical) |

### 13.3 Command-Specific Contract Requirements

Each `apr` subcommand category (§7.2) must satisfy specific contracts:

| Command Category | Required Contracts |
|-----------------|-------------------|
| Inference (run, serve, chat) | F-CKPT-011, F-CKPT-013, F-CKPT-014, F-CKPT-016 |
| Inspection (inspect, tensors, hex) | F-CKPT-011 |
| Validation (validate, check, qa) | F-CKPT-001, F-CKPT-002, F-CKPT-004, F-CKPT-011, F-CKPT-013 |
| Conversion (import, export, convert) | F-CKPT-011, F-CKPT-012, F-CKPT-015, F-CKPT-018 |
| Training (train apply, finetune) | F-CKPT-001, F-CKPT-002, F-CKPT-004, F-CKPT-007..010 |
| Resume (train apply --resume) | F-CKPT-005, F-CKPT-006, F-CKPT-011, F-CKPT-013, F-CKPT-014 |
| Publishing (publish, export) | F-CKPT-001, F-CKPT-003, F-CKPT-015, F-CKPT-017 |
| Merge (merge) | F-CKPT-001, F-CKPT-011, F-CKPT-013, F-CKPT-014, F-CKPT-017 |

---

## 14. Integration with Plan/Apply and Provable-Contracts

### 14.1 Plan/Apply Lifecycle

The `apr train` command follows a plan/apply pattern (like Terraform):

```
apr train plan              → TrainingPlan (validates data, model, HPO, resources)
                              ↓ verdict: Ready | WarningsPresent | Blocked
apr train apply             → execute_plan(plan, apply_config)
                              ↓ ClassifyTrainer.train() per trial
                              ↓ writes: epoch-N.ckpt.apr + best.adapter.apr
                              ↓ returns: TuneResult (leaderboard of trials)
apr train apply --resume    → loads epoch-N.ckpt.apr, continues from saved state
```

**Plan** touches no GPU. It validates pre-conditions and estimates resources.
The `ResourceEstimate.estimated_checkpoint_mb` field in the plan now reflects
the APR checkpoint size (model + optimizer = 3× adapter size).

**Apply** creates the output directory with APR checkpoints:

```
output_dir/
├── plan.yaml                  # saved plan for reproducibility
├── epoch-0.ckpt.apr           # training checkpoint (APR format)
├── epoch-1.ckpt.apr
├── epoch-2.ckpt.apr
├── best.adapter.apr           # best val_loss (APR format, deploy this)
├── final.adapter.apr          # last epoch
└── training_log.jsonl         # metrics history
```

**Resume** reads a `.ckpt.apr` and restores training state:

```bash
# Resume from specific checkpoint
apr train apply --resume output_dir/epoch-1.ckpt.apr \
                --data train.jsonl --output output_dir/
```

The `ApplyConfig` struct gains a `resume_checkpoint` field:

```rust
pub struct ApplyConfig {
    pub model_path: PathBuf,
    pub data_path: PathBuf,
    pub output_dir: PathBuf,
    pub on_trial_complete: Option<fn(usize, usize, &TrialSummary)>,
    pub resume_checkpoint: Option<PathBuf>,  // NEW: path to .ckpt.apr
}
```

### 14.2 Provable-Contracts Integration

The checkpoint spec maps to one contract with 18 invariants. The authoritative
source is `provable-contracts/contracts/entrenar/apr-checkpoint-v1.yaml`.
See §13 for the full contract table (F-CKPT-001..018).

**Extended contract**: `training-loop-v1.yaml`

F-LOOP-003 ("Checkpoint restorable") is strengthened to require:
- Checkpoint is in APR format (not just any serialization)
- Restored val_loss within 0.05 tolerance (F-CKPT-005)
- Optimizer moments restored (not reinitialized)
- LR schedule position restored (not reset to warmup)

### 14.3 Contract Dependency Chain

```
tokenizer-loading-v1       (F-TOK-001..008)
qwen2-weight-loading-v1    (F-WGT-001..009)
         ↓                        ↓
batch-training-v1          (F-BATCH-001..007)
         ↓
training-loop-v1           (F-LOOP-001..010)
         ↓
apr-checkpoint-v1          (F-CKPT-001..018)  ← 18 invariants
         ↓
cuda-classify-training-v1  (F-CUDA-001..011)
```

The checkpoint contract depends on the training loop contract (checkpoints are
produced by the training loop) and is consumed by the CUDA contract (GPU weights
must survive checkpoint round-trip).

---

## 15. Implementation Phases

### Phase 1: Fix Adapter Checkpoint (P0 — immediate)

Fix `save_apr_checkpoint()` in entrenar to save LoRA weights + rich metadata.
Currently only saves classifier head.

**Files**: `entrenar/src/finetune/classify_trainer.rs`
**Contract**: F-CKPT-001

### Phase 2: Training Checkpoint (P1 — next sprint)

Add optimizer state saving/loading. Enable `--resume` flag.

**Files**:
- `entrenar/src/finetune/classify_trainer.rs` (save/load optimizer moments)
- `entrenar/src/optim/adamw.rs` (expose state_dict / load_state_dict)
- `aprender/crates/apr-cli/src/commands/train.rs` (--resume flag)

**Contracts**: F-CKPT-004, F-CKPT-005, F-CKPT-006

### Phase 3: SafeTensors Import (P2)

Read HF PEFT checkpoints directly. Support both single-file adapters and
sharded base models (via `model.safetensors.index.json`).

**Files**:
- `aprender/src/serialization/safetensors_reader.rs` (extend for adapter layout)
- New: `aprender/src/convert/safetensors_to_apr.rs`

**Contract**: F-CKPT-018

### Phase 4: Provenance Chain (P2)

Canonical model hash (§10.1), data hash, parent checkpoint hash.

**Files**:
- `aprender/src/format/v2/header_impl.rs` (add hash fields to metadata)
- `entrenar/src/finetune/classify_trainer.rs` (compute and embed hashes)

**Contract**: F-CKPT-017

### Phase 5: Canonical Ordering (P2)

Sort tensor index entries by name in `AprV2Writer::write()`.

**Files**: `aprender/src/format/v2/writer.rs`
**Contract**: F-CKPT-015

### Phase 6: Filtered Reader (P3)

Add `AprReader::open_filtered()` to skip `__training__.*` tensors without
loading them into memory.

**Files**: `aprender/src/serialization/apr/mod.rs`
**Contract**: F-CKPT-016 (performance aspect)

---

## 16. Non-Goals

- **Full base model in checkpoint**: Adapter checkpoints reference the base
  model by hash, not by embedding it. A 159 GB base model should not be
  duplicated in every 4 MB checkpoint.
- **Distributed training state**: FSDP/DeepSpeed shard coordination is out of
  scope. APR targets single-GPU and data-parallel training.
- **Pickle compatibility**: APR will never use pickle. This is a hard constraint.
- **GGUF checkpoint format**: GGUF is inference-only by design. No need to
  add training state to GGUF.
- **MoE routing state**: Expert routing is stateless (computed per-token from
  router weights). No routing state needs checkpointing.

---

## 17. Falsification Log

Systematically attempted to break this spec. All confirmed flaws resolved.

### Round 1 (v1.0.0 → v1.1.0)

| ID | Flaw | Severity | Resolution |
|----|------|----------|------------|
| F-1 | Writer doesn't sort tensors → round-trip not bit-identical | Medium | §2.4: canonical ordering, F-CKPT-015 |
| F-2 | `.ckpt.apr` wastes memory at inference | Low | §7.2.1: filtered loading, §15 Phase 6 |
| F-3 | `base_model_hash` undefined for non-APR formats | Medium | §10.1: canonical format-independent hash algorithm |
| F-4 | No checkpoint schema version | Medium | §4.2/4.3: `__checkpoint__.schema_version`, F-CKPT-002 |
| F-5 | Resume doesn't verify data integrity | Medium | §8.2 step 3, F-CKPT-006 |
| F-6 | Tensor name 256-byte limit | N/A | Not a problem (max ~53 chars) |
| F-7 | Dense-only architecture assumption | High | §3: added Qwen3-Next hybrid MoE support |
| F-8 | No sharded SafeTensors import | High | §9.1: added sharded model import via index.json |

### Round 2 — Corruption Prevention (v1.1.0)

Added F-CKPT-007..018 to `apr-checkpoint-v1.yaml` after audit found gaps in
both write-side and read-side integrity.

| ID | Flaw | Severity | Resolution |
|----|------|----------|------------|
| F-9 | No NaN/Inf check at write time | High | F-CKPT-007 |
| F-10 | No shape validation at write time | High | F-CKPT-008 |
| F-11 | Non-atomic writes → crash corruption | High | F-CKPT-009 |
| F-12 | No dtype consistency check | Medium | F-CKPT-010 |
| F-13 | CRC32 not enforced on read | High | F-CKPT-011 |
| F-14 | SafeTensors header not validated | High | F-CKPT-012 |
| F-15 | No post-load NaN scan | High | F-CKPT-013 |
| F-16 | Shape-config mismatch not caught on load | High | F-CKPT-014 |

### Round 3 — Command Coverage (v1.1.0 → v1.2.0)

Audited all 51+ `apr` subcommands. Spec §7 only covered 4 tools (alimentar,
entrenar, apr-cli summary, realizar). 47 subcommands had unspecified checkpoint
behavior.

| ID | Flaw | Severity | Resolution |
|----|------|----------|------------|
| F-17 | §7 only covered 4 tools, missed 47 apr subcommands | High | §7.2: full command matrix (12 categories, 51+ commands) |
| F-18 | `finetune` vs `train apply` undefined | Medium | §7.2.7: finetune = quick path, train apply = plan/apply production path |
| F-19 | `convert` vs `rosetta convert` vs `import`/`export` overlap | Medium | §7.2.5/7.2.6: import=ingest, export=publish, convert=optimize, rosetta=universal |
| F-20 | `validate` vs `check` overlap | Low | §7.2.3: validate=format scoring, check=pipeline self-test |
| F-21 | `merge` behavior with adapters undefined | Medium | §7.2.8: merge LoRA into base, strips training state |
| F-22 | `quantize` on adapters undefined | Medium | §7.2.5: can quantize adapter weights (Q4K) |
| F-23 | `publish` → HF conversion path undefined | Medium | §7.2.9: auto-converts to SafeTensors + config |
| F-24 | `diagnose` on .ckpt.apr undefined | Medium | §7.2.7: Five Whys root-cause analysis on training checkpoints |
| F-25 | `eval` on .ckpt.apr should show stored vs computed metrics | Medium | §7.2.4: reports both embedded and freshly evaluated metrics |
| F-26 | `monitor` checkpoint interaction undefined | Low | §7.2.7: reads training_state.json + .ckpt.apr files |
| F-27 | Contracts not synced between spec §13 and YAML | Medium | §13: complete rewrite synced with apr-checkpoint-v1.yaml (F-CKPT-001..018) |
| F-28 | No per-command contract requirements | Medium | §13.3: command category → required contracts matrix |

### Round 4 — Deep Falsification (v1.2.0)

Systematic adversarial audit of the full 1148-line spec. Fixed 6 spec bugs
and 2 YAML bugs. 24 additional findings logged below; high-severity items
have inline resolutions, medium/low deferred to Open Questions.

| ID | Flaw | Severity | Resolution |
|----|------|----------|------------|
| F-29 | F-CKPT-005/006 categorization mismatch (spec vs YAML) | Low | §13: noted as cross-cutting |
| F-30 | §15 uses C-CKPT prefix, should be F-CKPT | High | **Fixed**: all §15 refs now F-CKPT |
| F-31 | YAML references v1.1.0, spec is v1.2.0 | Medium | **Fixed**: YAML updated to v1.2.0 |
| F-32 | Only 5 falsification tests for 18 contracts | High | Deferred: add FALSIFY-CKPT-006..018 in Phase 2 |
| F-33 | YAML FALSIFY-CKPT-004/005 cross-reference wrong contract IDs | High | **Fixed**: 004→F-CKPT-015, 005→F-CKPT-016 |
| F-34 | Detection rule ambiguous for merged models with lora.* names | Medium | OQ §18.6 |
| F-35 | Zero LoRA layers edge case violates F-CKPT-001 | Medium | OQ §18.7 |
| F-36 | 0-byte file behavior undefined | Medium | F-CKPT-011 precondition: file >= 48 bytes |
| F-37 | Epoch-0 checkpoint overwritten (step 0 vs step 359) | Medium | §8.1: checkpoints written at end-of-epoch only |
| F-38 | Base model deleted but adapter still references it | High | OQ §18.8 |
| F-39 | SUPPORTED_VERSION undefined | High | SUPPORTED_VERSION = 1, hard error if > supported |
| F-40 | Old APR files lack `__checkpoint__` metadata | High | F-CKPT-002 only enforced for schema >= 1; legacy fallback to §6 detection |
| F-41 | CRC32 header vs footer — only footer exists | Medium | F-CKPT-011: verify footer CRC32 only |
| F-42 | `export` doesn't list .ckpt.apr as valid input | Medium | **Fixed**: added .ckpt.apr to export inputs |
| F-43 | `import` claims to create `__training__` NS | Medium | **Fixed**: changed to N/A |
| F-44 | rosetta convert adapter → GGUF undefined | Medium | OQ §18.9 |
| F-45 | `quantize` __training__ column wrong | Low | **Fixed**: changed to N/A |
| F-46 | `merge` should require F-CKPT-017 (provenance) | Medium | Added to §13.3 merge category |
| F-48 | Canonical hash missing n_dims separator | High | OQ §18.10 |
| F-49 | F-CKPT-010 conflicts with quantization | High | **Fixed**: F-CKPT-010 now allows explicit quantization |
| F-50 | SafeTensors name mapping not truly bijective | Medium | OQ §18.11 |
| F-51 | `apr train resume` listed as standalone command | Medium | **Fixed**: changed to `apr train apply --resume` |
| F-53 | Missing contract for `publish` stripping | Medium | OQ §18.12 |
| F-55 | `compile` embeds __training__ (should strip) | Medium | **Fixed**: changed to Strips |
| F-57 | `distill` creates __training__ but has no contract coverage | Medium | Add distill to training contracts in Phase 2 |
| F-60 | `data_hash` — which file exactly, and order-sensitive? | High | OQ §18.13 |

---

## 18. Open Questions

1. **Should training checkpoints use F16 for optimizer moments?** Would halve
   checkpoint size. AdamW moments lose precision at F16 but may be acceptable
   for short fine-tunes (3-5 epochs). Recommendation: F32 default, F16 opt-in
   via `--checkpoint-precision f16`.

2. **Checkpoint rotation policy?** Keep last N training checkpoints to bound
   disk usage. Suggestion: `--keep-checkpoints 3` (default), always keep best.

3. **MoE expert-level LoRA?** For Qwen3-Coder-Next, should LoRA target
   individual MoE experts (512 × rank adapters) or shared expert only? The
   format supports both — this is a training strategy question, not a format
   question.

4. **Adapter quantization UX?** When `apr quantize best.adapter.apr` runs,
   should it quantize LoRA A/B matrices (which are small) or is this only
   useful for merged models? May not be worth the precision loss for ~4 MB
   adapters.

5. **Multi-adapter merge weighting?** `apr merge` with `--weights 0.7,0.3`
   needs a defined algebra. Options: weighted average (TIES), DARE masking,
   or simple linear interpolation. This is an algorithm question, not a format
   question.

6. **Detection rule for merged models (F-34)?** A merged `.apr` may retain
   `lora.*` tensor names from the merge input. The detection rule (§6) would
   misclassify it as `.adapter.apr`. Options: (a) merged models must rename
   `lora.*` tensors, or (b) add a `__checkpoint__.type: "merged"` field.

7. **Zero-LoRA adapter (F-35)?** If only the classifier head is fine-tuned
   (no LoRA), is the result a valid `.adapter.apr`? Proposal: allow
   `adapter_type: "head_only"` in metadata, F-CKPT-001 count becomes just 2.

8. **Deleted base model (F-38)?** When `--model` path doesn't exist but
   `base_model_hash` is set, should inference commands (a) hard error, (b) try
   to download from `base_model_source`, or (c) error with helpful message
   suggesting `apr pull`?

9. **Rosetta convert adapter → GGUF (F-44)?** GGUF has no LoRA concept.
   Options: (a) require `--model` to merge first, (b) reject with error.

10. **Canonical hash n_dims ambiguity (F-48)?** Include `n_dims` as a u8
    between dtype_byte and shape dimensions to prevent collisions.

11. **SafeTensors name mapping bijectivity (F-50)?** HF models use different
    prefixes (`base_model.model.model.layers.*` vs `language_model.model.*`).
    Options: (a) store original HF name in metadata for perfect round-trip,
    or (b) weaken F-CKPT-018 to "equivalent tensor names."

12. **Publish stripping contract (F-53)?** Should `apr publish` on a `.ckpt.apr`
    auto-strip `__training__` tensors? Proposal: yes, always strip before
    upload, add F-CKPT-019.

13. **Data hash scope (F-60)?** When training uses multiple files (train.jsonl +
    val.jsonl), which files are hashed? Proposal: hash a sorted manifest of
    `{filename, size, sha256}` tuples. Line ordering within files should NOT
    affect the hash (sort lines before hashing).
