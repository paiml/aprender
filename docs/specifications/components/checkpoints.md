# APR Checkpoints

Version: 1.0
Status: proposed
Date: 2026-04-09

**Version**: 1.2.0
**Status**: Active
**Parent**: [aprender-spec.md](../aprender-spec.md) §8
**Contracts**: 18/18 implemented

---

## 1. Overview

APR is the only model format spanning both training and inference. SafeTensors
and GGUF store weights for inference only. PyTorch checkpoints use pickle
(arbitrary code execution risk). APR checkpoints unify the lifecycle:

```
train → checkpoint → resume → evaluate → deploy
```

One `.apr` file = one deployable or resumable artifact. No sidecar files.

---

## 2. Design Principles

1. **Self-contained**: No separate optimizer/scheduler/rng state files
2. **No pickle**: All state = tensors (binary) + metadata (JSON)
3. **Backward compatible**: Inference readers ignore `__training__.*` tensors
4. **Canonical ordering**: Lexicographic tensor sort → bit-identical save/load
5. **SafeTensors interop**: Import/export for HuggingFace compatibility
6. **Architecture-agnostic**: HF `config.json` convention in metadata

---

## 3. Checkpoint Taxonomy

| Type | Contains | Use Case |
|------|----------|----------|
| **Inference** | Weights + metadata | `apr run`, `apr serve` |
| **Training** | + optimizer state + LR schedule + RNG + epoch | `apr train --resume` |
| **LoRA** | Adapter weights + base model ref | `apr finetune --merge` |
| **Merged** | Full weights (base + adapter applied) | Deployment |

---

## 4. Tensor Namespace Convention

| Prefix | Purpose | Example |
|--------|---------|---------|
| `model.*` | Model weights | `model.layers.0.self_attn.q_proj.weight` |
| `__training__.optimizer.*` | Optimizer state (m, v) | `__training__.optimizer.m.layers.0...` |
| `__training__.scheduler.*` | LR scheduler state | `__training__.scheduler.last_lr` |
| `__training__.rng.*` | RNG state for reproducibility | `__training__.rng.cpu` |
| `__training__.meta.*` | Training metadata | `__training__.meta.epoch` |
| `lora_a.*`, `lora_b.*` | LoRA adapter matrices | `lora_a.layers.0.q_proj` |

Inference readers skip all `__training__.*` tensors automatically.

---

## 5. Command Matrix

| Command | Read | Write | Checkpoint Type |
|---------|------|-------|----------------|
| `apr run` | Yes | No | Inference |
| `apr serve` | Yes | No | Inference |
| `apr finetune` | Yes | Yes | Training + LoRA |
| `apr train apply` | Yes | Yes | Training |
| `apr train watch` | Yes | Yes | Training (auto-save) |
| `apr merge` | Yes | Yes | Merged |
| `apr quantize` | Yes | Yes | Inference (quantized) |
| `apr prune` | Yes | Yes | Inference (pruned) |
| `apr export` | Yes | Yes | Format-specific |
| `apr import` | No | Yes | Inference |
| `apr validate` | Yes | No | Any (integrity check) |
| `apr inspect` | Yes | No | Any (metadata view) |

---

## 6. Checkpoint Lifecycle

```
┌──────────────────────────────────────────────────┐
│              apr train plan config.yaml           │
│    (estimate VRAM, validate data, dry-run)        │
└──────────────────┬───────────────────────────────┘
                   │
┌──────────────────▼───────────────────────────────┐
│              apr train apply config.yaml          │
│    Epoch 1 → save checkpoint-epoch-001.apr        │
│    Epoch 2 → save checkpoint-epoch-002.apr        │
│    ...                                            │
│    Best → save checkpoint-best.apr                │
└──────────────────┬───────────────────────────────┘
                   │
         ┌─────────┴──────────┐
         │                    │
┌────────▼────────┐  ┌───────▼────────┐
│  apr eval        │  │  apr train     │
│  (perplexity,    │  │  --resume      │
│   classification)│  │  checkpoint.apr│
└────────┬────────┘  └────────────────┘
         │
┌────────▼────────────────────────────────────────┐
│  apr export checkpoint-best.apr --format gguf    │
│  apr serve checkpoint-best.apr                   │
│  apr compile checkpoint-best.apr --target x86_64 │
└─────────────────────────────────────────────────┘
```

---

## 7. Optimizer State Storage

Training checkpoints store Adam/AdamW first and second moments:

```
__training__.optimizer.m.model.layers.0.self_attn.q_proj.weight  # 1st moment
__training__.optimizer.v.model.layers.0.self_attn.q_proj.weight  # 2nd moment
__training__.optimizer.step                                       # Global step
__training__.scheduler.last_lr                                    # Current LR
__training__.meta.epoch                                           # Current epoch
__training__.rng.cpu                                              # CPU RNG state
__training__.rng.cuda                                             # GPU RNG state
```

**Size overhead**: ~2x model size (m + v for each parameter).
A 7B model checkpoint is ~28GB (14GB weights + 14GB optimizer state).

---

## 8. SafeTensors Interop

```bash
# Import HuggingFace checkpoint → APR
apr import hf://org/model -o model.apr

# Export APR → SafeTensors (strips __training__. tensors)
apr export model.apr --format safetensors -o model.safetensors
```

Round-trip is lossy: SafeTensors cannot store optimizer state, so
`import → export` drops training tensors.

---

## 9. Provenance Chain

Each checkpoint records its lineage:

```json
{
  "provenance": {
    "source": "hf://Qwen/Qwen2.5-Coder-0.5B",
    "operations": [
      {"op": "import", "timestamp": "2026-03-01T00:00:00Z"},
      {"op": "finetune", "method": "lora", "rank": 16, "epochs": 3},
      {"op": "merge", "strategy": "weighted", "weights": [0.7, 0.3]}
    ],
    "checksum": "blake3:abc123..."
  }
}
```

---

## 10. Falsification Tests

| ID | Rule | Prediction |
|----|------|-----------|
| F-CK-001 | Save/load roundtrip | Bit-identical tensors after save → load |
| F-CK-002 | Inference ignores training tensors | `apr run` works on training checkpoint |
| F-CK-003 | Resume training | Optimizer state restored, loss continues from last |
| F-CK-004 | Canonical ordering | Two saves of same model produce identical bytes |
| F-CK-005 | Provenance preserved | Operations list grows monotonically |
| F-CK-006 | LoRA merge | `apr finetune --merge` produces valid inference checkpoint |
| F-CK-007 | SafeTensors export | Exported .safetensors loadable by HF transformers |
| F-CK-008 | Large model sharding | >2GB checkpoint splits correctly |

---

## 11. Contracts

18 contracts implemented in `contracts/` covering:
- Tensor namespace separation (`__training__.*` prefix)
- Checkpoint integrity (BLAKE3 checksums)
- Canonical ordering (lexicographic tensor names)
- Optimizer state shape parity (m/v shapes == weight shapes)
- Provenance chain immutability
- SafeTensors roundtrip fidelity
