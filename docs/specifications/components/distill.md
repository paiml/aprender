# Distill Component Specification

**Parent**: [apr-spec.md](../apr-spec.md) §6
**Status**: Active
**CLI**: `apr distill`
**Implementation**: `crates/apr-cli/src/commands/distill.rs`
**Library**: `entrenar` distillation module

---

## 1. Overview

Knowledge distillation transfers learned behavior from a large teacher model to
a smaller student model. The student learns to mimic the teacher's output
distribution, achieving better performance than training from scratch.

**Goal**: Full parity with [DistillKit](https://github.com/arcee-ai/DistillKit)
plus sovereign-native features (APR format, offline-only, SIMD kernels).

---

## 2. CLI Interface

```
apr distill <TEACHER> \
  --student <PATH> \
  --data <JSONL> \
  --output <PATH> \
  [--strategy <standard|progressive|ensemble|hidden-state|quant-aware|online>] \
  [--temperature <F>] \
  [--alpha <F>] \
  [--epochs <N>] \
  [--config <YAML>] \
  [--stage <precompute|train>] \
  [--plan]
```

---

## 3. Implemented Strategies

### 3.1 Standard (Logit-Based)

Classic Hinton distillation: minimize KL divergence between teacher and student
softmax outputs, weighted against task-specific loss.

```bash
apr distill teacher.apr --student student.apr \
  --data train.jsonl --strategy standard \
  --temperature 3.0 --alpha 0.7 -o distilled.apr
```

**Loss**: `L = α * KL(softmax(t_logits/T), softmax(s_logits/T)) + (1-α) * L_task`

### 3.2 Progressive

Layer-by-layer progressive distillation with explicit teacher→student layer
mapping. Student learns intermediate representations, not just final output.

```bash
apr distill teacher.apr --student student.apr \
  --strategy progressive --data train.jsonl -o distilled.apr
```

**Config**:
```yaml
distillation:
  progressive:
    layer_mapping: [[0, 0], [4, 1], [8, 2], [12, 3]]
    hidden_weight: 1.0
```

### 3.3 Ensemble

Multiple teachers contribute weighted logits. The student learns from a
committee of experts.

```bash
apr distill teacher1.apr teacher2.apr teacher3.apr \
  --student student.apr --strategy ensemble \
  --weights 0.5,0.3,0.2 --data train.jsonl -o distilled.apr
```

---

## 4. Implemented Features

### 4.1 Two-Stage Pipeline (ALB-011)

Separate teacher logit extraction from student training. Useful when the
teacher is too large to keep in memory during training.

**Stage 1: Precompute** — Extract teacher logits to disk.
```bash
apr distill teacher.apr --stage precompute \
  --data train.jsonl --config distill.yaml
```

**Stage 2: Train** — Train student using cached logits.
```bash
apr distill --stage train --config distill.yaml
```

### 4.2 YAML Config

Full distillation pipeline specification for reproducibility.

```yaml
teacher:
  model_id: "teacher-7b.apr"
  load_in_8bit: true
student:
  model_id: "student-1b.apr"
  load_in_4bit: true
  lora:
    rank: 16
    alpha: 32.0
distillation:
  temperature: 4.0
  alpha: 0.7
  attention_transfer:
    enabled: true
    weight: 0.3
training:
  epochs: 5
  batch_size: 8
  learning_rate: 5e-5
dataset:
  path: "train.jsonl"
output:
  path: "distilled/"
```

### 4.3 Attention Transfer

Transfer attention patterns from teacher to student alongside logit distillation.
Configurable weight relative to main distillation loss.

### 4.4 LoRA on Student

Apply LoRA to the student model during distillation for reduced VRAM usage.

### 4.5 Plan Mode

Estimate VRAM and compute requirements before committing resources.

```bash
apr distill teacher.apr --student student.apr --plan
```

---

## 5. Planned Strategies (Arcee DistillKit Parity)

### 5.1 Hidden State Distillation

Match intermediate hidden state representations between teacher and student.
Uses a learned linear projection to bridge dimensionality differences.

```bash
apr distill teacher.apr --student student.apr \
  --strategy hidden-state \
  --data train.jsonl \
  --hidden-layer-map "0:0,4:1,8:2,12:3" \
  -o distilled.apr
```

**How it works**:
1. Extract hidden states from selected teacher layers
2. Apply learned linear projection: `proj(student_hidden) → teacher_dim`
3. Minimize MSE between projected student and teacher hidden states
4. Combined with logit loss for end-to-end training

**Parameters**:
- `--hidden-layer-map`: Teacher:Student layer index pairs
- `--hidden-weight`: Weight of hidden state loss vs logit loss (default: 1.0)
- `--projection-init`: Initialization for projection layers (default: xavier)

**Advantages over logit-only**:
- Richer learning signal from intermediate representations
- Better knowledge transfer for deeply different architectures
- Can target specific capability layers (e.g., reasoning in middle layers)

**DistillKit finding**: Logit-based consistently outperforms hidden-state across
benchmarks, but hidden-state can be superior for architecturally diverse pairs.

### 5.2 Quantization-Aware Distillation

Distillation that accounts for quantization effects during training. Student
learns to compensate for quantization error.

```bash
apr distill teacher.apr --student student.apr \
  --strategy quant-aware \
  --target-quant q4k \
  --data train.jsonl \
  -o distilled-q4k.apr
```

**How it works**:
1. Polynomial approximation of teacher logit distribution
2. Error-diffusion quantization of residuals between teacher and student
3. Bit-level packing with arbitrary bit widths (1-64 bits)
4. Student trained with simulated quantization noise

**Parameters**:
- `--target-quant`: Target quantization scheme (int4, int8, q4k, q6k)
- `--quant-noise-scale`: Simulated quantization noise amplitude (default: auto)
- `--bit-width`: Custom bit width for error-diffusion (default: matches target)

**Benefits**:
- Higher quality quantized models than post-training quantization
- Student learns to be robust to quantization artifacts
- Can achieve Q4 quality approaching Q8 baseline

### 5.3 Online Distillation

Concurrent teacher inference during student training. No precompute step.
Teacher and student run simultaneously, with teacher logits computed on-the-fly.

```bash
apr distill teacher.apr --student student.apr \
  --strategy online \
  --data train.jsonl \
  --teacher-gpu 0 --student-gpu 1 \
  -o distilled.apr
```

**Parameters**:
- `--teacher-gpu`: GPU for teacher inference (default: 0)
- `--student-gpu`: GPU for student training (default: 1)
- `--teacher-batch-ahead`: Prefetch N batches of teacher logits (default: 2)

**Advantages**:
- No disk space for cached logits
- Can use data augmentation (teacher sees augmented inputs)
- Simpler pipeline (single command vs two-stage)

**Trade-offs**:
- Requires 2× GPU memory (teacher + student simultaneously)
- Slower than offline if teacher is much larger than student

---

## 6. Planned Features

### 6.1 Domain-Specific Distillation

Distill a large general-purpose teacher into a smaller student specialized for
a specific domain (e.g., function calling, code generation, medical QA).

```bash
apr distill teacher.apr --student student.apr \
  --strategy standard \
  --domain function-calling \
  --data function-calling-examples.jsonl \
  -o function-calling-student.apr
```

**Domain presets** auto-configure loss weights and evaluation:
- `function-calling`: Emphasize structured output, tool use accuracy
- `code`: Emphasize pass@1, syntax correctness
- `reasoning`: Emphasize chain-of-thought faithfulness
- `translation`: Emphasize BLEU score, fluency

### 6.2 Multi-Teacher Distillation with Routing

Instead of simple ensemble averaging, learn a router that selects the best
teacher per input. Combines with MoE concepts.

```bash
apr distill --teachers teacher-math.apr teacher-code.apr teacher-lang.apr \
  --student student.apr \
  --strategy routed-ensemble \
  --router-data calibration.jsonl \
  -o distilled.apr
```

### 6.3 Self-Distillation

Distill a model from itself at a higher precision. The teacher is the same
model in FP32/FP16, the student is the quantized version.

```bash
apr distill model-fp16.apr --self-distill \
  --target-quant q4k \
  --data calibration.jsonl \
  -o model-q4k-distilled.apr
```

---

## 7. Architecture Compatibility

### Same Architecture (Easy)

Teacher and student share the same architecture with different sizes.
Layer mapping is straightforward (every N-th layer).

### Different Architecture (Hard)

Teacher and student have different architectures (e.g., LLaMA → Phi).
Requires hidden state distillation with learned projections. Logit-only
distillation still works if vocab matches.

### Vocab Mismatch

If teacher and student have different vocabularies, use tokenizer alignment:
1. Map student vocab → teacher vocab via string matching
2. Project teacher logits to student vocab space
3. Use aligned logits for KL divergence

---

## 8. Testing Requirements

- Unit tests: KL divergence computation, temperature scaling
- Integration test: distill toy teacher → student → inference → assert quality
- Property tests: distillation loss must decrease over epochs
- Two-stage parity: online and offline must produce equivalent results
- Hidden state: verify projection layer dimensions match
- Quant-aware: verify quantized student matches distilled quality
- Mutation testing: >80% mutation score on distillation loss computation

---

## Provable Contracts

### Contract: `distill-core-v1.yaml`

```yaml
metadata:
  description: "Knowledge distillation — KL divergence, temperature scaling, teacher-student"
  references:
    - "Hinton et al. (2015) Distilling the Knowledge in a Neural Network"
  depends_on:
    - "softmax-kernel-v1"
    - "cross-entropy-kernel-v1"

equations:
  kl_distillation:
    formula: "L = α * T² * KL(softmax(t/T), softmax(s/T)) + (1-α) * L_task"
    invariants:
      - "L >= 0 (KL divergence is non-negative)"
      - "T > 0 (temperature must be positive)"
      - "α ∈ [0, 1] (interpolation weight)"

  temperature_scaling:
    formula: "p_i = softmax(logit_i / T)"
    invariants:
      - "T → ∞: uniform distribution"
      - "T → 0: argmax (one-hot)"
      - "T = 1: standard softmax"

  teacher_frozen:
    formula: "∀ param p in teacher: p_after == p_before"
    invariants:
      - "Teacher weights never updated by optimizer"
      - "Teacher in eval mode (no dropout, batchnorm frozen)"

  two_stage_equivalence:
    formula: "offline_logits(batch) == online_logits(batch)"
    invariants:
      - "Precomputed and live teacher logits identical for same input"
      - "Deterministic teacher inference required"

proof_obligations:
  - type: bound
    property: "KL non-negativity"
    formal: "KL(p || q) >= 0 for all valid distributions p, q"
  - type: bound
    property: "Temperature positivity"
    formal: "T > 0 enforced at construction"
  - type: bound
    property: "Alpha bounds"
    formal: "0 <= α <= 1"
  - type: invariant
    property: "Teacher frozen"
    formal: "teacher weights unchanged after distillation"
  - type: equivalence
    property: "Two-stage parity"
    formal: "offline_logits == online_logits for same batch"

falsification_tests:
  - id: FALSIFY-DCORE-001
    rule: "KL non-negative"
    prediction: "KL(teacher, student) >= 0 for random logits"
    if_fails: "KL computation bug (log of negative or division error)"
  - id: FALSIFY-DCORE-002
    rule: "Temperature positivity"
    prediction: "T=0 or T<0 returns validation error"
    if_fails: "Temperature check missing"
  - id: FALSIFY-DCORE-003
    rule: "Alpha bounds"
    prediction: "α outside [0,1] returns validation error"
    if_fails: "Alpha validation missing"
  - id: FALSIFY-DCORE-004
    rule: "Teacher frozen"
    prediction: "teacher weights bitwise identical before/after distill"
    if_fails: "Teacher params included in optimizer"
  - id: FALSIFY-DCORE-005
    rule: "Two-stage parity"
    prediction: "precomputed logits match live inference logits"
    if_fails: "Teacher state differs between precompute and online"

kani_harnesses:
  - id: KANI-DCORE-001
    obligation: "KL non-negativity"
    property: "KL divergence >= 0 for bounded distributions"
    bound: 4
    strategy: stub_float
```

### Binding Requirements

```yaml
  - contract: distill-core-v1.yaml
    equation: kl_distillation
    module_path: "entrenar::distill"
    function: distillation_loss
    status: implemented

  - contract: distill-core-v1.yaml
    equation: temperature_scaling
    module_path: "entrenar::distill"
    function: temperature_softmax
    status: implemented
```
