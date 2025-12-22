# Transfer Learning Theory

Transfer learning leverages knowledge from one task to improve performance on related tasks, dramatically reducing data requirements and training time.

## The Transfer Learning Paradigm

```
Source Domain (Large Data)        Target Domain (Limited Data)
        │                                    │
        ▼                                    ▼
┌───────────────┐                  ┌───────────────┐
│  Pre-train    │                  │  Fine-tune    │
│  on ImageNet  │ ──Transfer──▶    │  on Custom    │
│  (1M images)  │                  │  (1K images)  │
└───────────────┘                  └───────────────┘
```

## Why Transfer Learning Works

### Feature Hierarchy

Neural networks learn hierarchical features:

| Layer | Features | Transferability |
|-------|----------|-----------------|
| Early | Edges, colors, textures | High (universal) |
| Middle | Shapes, parts | Medium |
| Late | Task-specific patterns | Low |

Early layers learn **general features** that apply across domains.

### The Lottery Ticket Hypothesis

Pre-trained networks contain "winning tickets" - subnetworks that train well on new tasks. Transfer learning finds these tickets without expensive search.

## Transfer Strategies

### 1. Feature Extraction (Frozen Base)

```
Pre-trained Model          New Task
┌─────────────────┐       ┌────────┐
│   Base Layers   │──────▶│  New   │──▶ Output
│   (Frozen)      │       │  Head  │
└─────────────────┘       └────────┘
```

- Freeze pre-trained layers
- Only train new classification head
- Best when: Target data is very limited

### 2. Fine-Tuning (Unfrozen Base)

```
Pre-trained Model          New Task
┌─────────────────┐       ┌────────┐
│   Base Layers   │──────▶│  New   │──▶ Output
│   (Trainable)   │       │  Head  │
└─────────────────┘       └────────┘
```

- Train entire network with small learning rate
- Base layers: lr × 0.01-0.1
- Head layers: lr × 1.0
- Best when: Moderate target data available

### 3. Gradual Unfreezing

Progressive unfreezing from top to bottom:

```
Epoch 1: Train head only
Epoch 2: Unfreeze top base layer
Epoch 3: Unfreeze next layer
...
Epoch N: All layers trainable
```

Prevents catastrophic forgetting of pre-trained knowledge.

## Domain Adaptation

When source and target distributions differ:

### Discrepancy-Based Methods

Minimize distribution distance:

```
L = L_task + λ · MMD(source, target)
```

Where MMD = Maximum Mean Discrepancy.

### Adversarial Methods (DANN)

Domain Adversarial Neural Network:

```
Features → Task Classifier (maximize)
    │
    └────▶ Domain Classifier (minimize via gradient reversal)
```

Features become domain-invariant.

## Multi-Task Learning

Learn multiple related tasks simultaneously:

```
       Input
         │
         ▼
    ┌─────────┐
    │ Shared  │
    │ Encoder │
    └────┬────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌──────┐  ┌──────┐
│Task A│  │Task B│
│ Head │  │ Head │
└──────┘  └──────┘
```

Benefits:
- Improved generalization through regularization
- Data efficiency (shared representation)
- Faster training (parallel tasks)

## Low-Rank Adaptation (LoRA)

Efficient fine-tuning for large models:

Instead of updating W directly:

```
W' = W + ΔW
```

Decompose update as low-rank:

```
ΔW = B × A
where B ∈ ℝ^(d×r), A ∈ ℝ^(r×k), r << min(d,k)
```

Parameters: O(r(d+k)) vs O(dk)

Example: GPT-3 (175B params) → LoRA (0.1% trainable)

## Adapter Layers

Insert small trainable modules:

```
Original Layer:  x → [Frozen Transformer] → y

With Adapter:    x → [Frozen Transformer] → [Adapter] → y + x
                                              ↓
                                     Down → ReLU → Up
                                     (d→r)       (r→d)
```

Only adapters train; base model frozen.

## Knowledge Distillation

Transfer knowledge from large to small model:

```
Teacher (Large)        Student (Small)
      │                      │
      ▼                      ▼
   Logits ───────────▶    Logits
      │         KL           │
      │     Divergence       │
      ▼                      ▼
   Labels ──────────────▶  Cross-Entropy
```

Loss:

```
L = α · KL(softmax(t_logits/T), softmax(s_logits/T))
  + (1-α) · CE(s_logits, labels)
```

Temperature T smooths distributions for better transfer.

## Negative Transfer

When transfer hurts performance:

**Causes:**
- Source and target too dissimilar
- Conflicting label spaces
- Domain shift too large

**Mitigation:**
- Measure domain similarity before transfer
- Use regularization to prevent forgetting
- Selective layer transfer

## Best Practices

### 1. Choosing What to Transfer

| Target Data | Source Similarity | Strategy |
|-------------|-------------------|----------|
| Small | High | Feature extraction |
| Small | Low | Careful fine-tuning |
| Large | High | Full fine-tuning |
| Large | Low | Train from scratch |

### 2. Learning Rate Schedule

```
Head:           lr = 1e-3
Upper layers:   lr = 1e-4
Lower layers:   lr = 1e-5
```

Discriminative fine-tuning preserves pre-trained knowledge.

### 3. Data Augmentation

Apply to target domain to increase effective data size:
- Image: rotation, flip, crop, color jitter
- Text: back-translation, synonym replacement
- Audio: time stretch, pitch shift, noise

## Applications

| Domain | Source Task | Target Task |
|--------|-------------|-------------|
| Vision | ImageNet | Medical imaging |
| NLP | Language modeling | Sentiment analysis |
| Speech | ASR pre-training | Voice commands |
| Code | General transpiler | Language-specific |

## References

- Yosinski, J., et al. (2014). "How transferable are features in deep neural networks?" NeurIPS.
- Hu, E. J., et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models." arXiv.
- Houlsby, N., et al. (2019). "Parameter-Efficient Transfer Learning for NLP." ICML.
- Ganin, Y., et al. (2016). "Domain-Adversarial Training of Neural Networks." JMLR.
