# What is Knowledge Distillation?

**Knowledge distillation** trains a smaller "student" model to mimic a larger "teacher" model's behavior.

## The Problem

Large models (7B-70B parameters) perform well but are:
- **Expensive to deploy**: High memory and compute costs
- **Slow inference**: Too slow for latency-sensitive applications
- **Resource-intensive**: Require powerful hardware

**Goal**: Transfer knowledge from large teacher →  smaller student while preserving performance

## The Solution

```
Teacher Model (7B params)  →  Knowledge Transfer  →  Student Model (1B params)
   Accuracy: 92%                                         Accuracy: 89% (vs 82% from scratch)
```

**Key Insight**: Train student on **soft targets** (teacher's probability distributions) rather than hard labels

## How It Works

From `src/distill/loss.rs`:

```rust
use entrenar::distill::DistillationLoss;

// Temperature=3.0, alpha=0.7
let loss_fn = DistillationLoss::new(3.0, 0.7);

// Combine soft targets from teacher + hard labels
let loss = loss_fn.forward(&student_logits, &teacher_logits, &labels);
```

### Distillation Loss Formula

```
L = α * T² * KL(softmax(teacher/T) || softmax(student/T))
  + (1-α) * CrossEntropy(student, labels)
```

Where:
- **T** = Temperature (typically 2.0-5.0)
- **α** = Distillation weight (typically 0.5-0.9)
- **KL** = Kullback-Leibler divergence (measures distribution similarity)

### Temperature Smoothing

**Temperature** softens probability distributions:

```
Logits:     [2.0, 1.0, 0.1]

T=1 (hard): [0.659, 0.242, 0.099]  ← Sharp peaks
T=3 (soft): [0.422, 0.307, 0.271]  ← Smoother distribution
```

**Why soft targets help**: Reveal model's "uncertainty" and inter-class relationships

## Distillation Methods in Entrenar

### 1. Temperature-Scaled KL Divergence

Standard distillation with soft targets:

```rust
let loss_fn = DistillationLoss::new(3.0, 0.7);
```

**From `src/distill/loss.rs`**

### 2. Multi-Teacher Ensemble

Distill from multiple teachers simultaneously:

```rust
use entrenar::distill::EnsembleDistiller;

let distiller = EnsembleDistiller::new(vec![teacher1, teacher2, teacher3]);
let loss = distiller.forward(&student_logits, &teacher_logits_list, &labels);
```

**From `src/distill/ensemble.rs`**

### 3. Progressive Layer-Wise

Layer-by-layer knowledge transfer:

```rust
use entrenar::distill::ProgressiveDistiller;

let distiller = ProgressiveDistiller::new();
distiller.distill_layer(student_layer, teacher_layer)?;
```

**From `src/distill/progressive.rs`**

## Validation

**44 distillation tests** including:
- 13 property-based tests for temperature smoothing
- KL divergence correctness validation
- Multi-teacher ensemble tests
- Progressive distillation tests

## When to Use Distillation

| Scenario | Recommended Method |
|----------|-------------------|
| **Deployment optimization** | Standard KL divergence |
| **Multiple expert models** | Multi-teacher ensemble |
| **Very deep networks** | Progressive layer-wise |
| **Limited training data** | Higher alpha (more distillation weight) |

## Example Results

```
Task: Text classification (SST-2 dataset)

Teacher (BERT-large, 340M params):  Accuracy: 93.2%
Student (BERT-tiny, 14M params):
  - From scratch:                   Accuracy: 84.1%
  - With distillation (T=3, α=0.8): Accuracy: 89.7%  (+5.6% improvement)
```

## Next Steps

- [Temperature-Scaled KL Divergence](./temperature-kl.md)
- [Multi-Teacher Ensemble](./multi-teacher.md)
- [Progressive Layer-Wise](./progressive.md)
- [Examples](../examples/distillation.md)

## References

- **Hinton et al. (2015)**: "Distilling the Knowledge in a Neural Network"
- **Sanh et al. (2019)**: DistilBERT paper
- Implementation in `src/distill/`
