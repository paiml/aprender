# Fine-Tune Component Specification

**Parent**: [apr-spec.md](../apr-spec.md) §5
**Status**: Active
**CLI**: `apr finetune`
**Implementation**: `crates/apr-cli/src/commands/finetune.rs`
**Library**: `entrenar_lora`, `entrenar::finetune`

---

## 1. Overview

Fine-tuning adapts a pre-trained model to specific domains or tasks. The goal is
to take ANY model (GGUF, SafeTensors, APR) and produce a domain-specialized
variant with minimal compute.

**Goal**: Full parity with Arcee's 4-layer adaptation pipeline (CPT → SFT → DPO
→ RLVR) plus sovereign-native features (local-only, APR format, SIMD kernels).

---

## 2. CLI Interface

```
apr finetune <MODEL> \
  --method <auto|full|lora|qlora|cpt|dpo|rlvr> \
  --data <JSONL> \
  --output <PATH> \
  [--rank <N>] \
  [--vram <GB>] \
  [--epochs <N>] \
  [--learning-rate <F>] \
  [--plan] \
  [--task <classify|generate|cpt|dpo>] \
  [--gpus <INDICES>] \
  [--distributed]
```

---

## 3. Implemented Methods

### 3.1 Auto Selection

Automatically chooses between full, LoRA, and QLoRA based on model size and
available VRAM. Decision tree:
- VRAM ≥ 4× model size → Full
- VRAM ≥ 1.5× model size → LoRA
- Otherwise → QLoRA

### 3.2 Full Fine-Tuning

All parameters are trainable. Highest quality but requires the most VRAM.

```bash
apr finetune model.apr --method full --data train.jsonl -o finetuned.apr
```

### 3.3 LoRA (Low-Rank Adaptation)

Inserts trainable rank-r decomposition matrices (A, B) into attention layers.
Frozen base weights + small trainable adapters.

```bash
apr finetune model.apr --method lora --rank 16 --data train.jsonl -o adapter/
```

**Parameters**:
- `--rank`: LoRA rank (default: auto-selected based on model size)
- `--max-seq-len`: GPU buffer allocation (lower = less VRAM)
- `--oversample`: Balance minority classes

### 3.4 QLoRA

LoRA with NF4-quantized frozen base weights. ~8x VRAM reduction vs full.

```bash
apr finetune model.apr --method qlora --quantize-nf4 --data train.jsonl -o adapter/
```

### 3.5 Adapter Merge

Merge a trained LoRA adapter back into the base model.

```bash
apr finetune model.apr --merge --adapter adapter/ -o merged.apr
```

---

## 4. Implemented Features

### 4.1 Multi-Adapter Training (GPU-SHARE Phase 2)

Train multiple LoRA adapters concurrently on shared GPU via CUDA MPS.

```bash
apr finetune model.apr --method lora \
  --adapters corpus-a.jsonl:checkpoints/adapter-a \
  --adapters corpus-b.jsonl:checkpoints/adapter-b \
  --experimental-mps --gpu-share 50
```

**TOML config** (`adapters.toml`):
```toml
[[adapter]]
data = "corpus-a.jsonl"
checkpoint = "checkpoints/adapter-a"

[[adapter]]
data = "corpus-b.jsonl"
checkpoint = "checkpoints/adapter-b"
```

```bash
apr finetune model.apr --adapters-config adapters.toml
```

### 4.2 Distributed Training

Multi-node training with coordinator/worker architecture.

```bash
# Coordinator
apr finetune model.apr --method lora --data train.jsonl \
  --role coordinator --bind 0.0.0.0:9000 --expect-workers 3

# Workers
apr finetune model.apr --role worker --coordinator intel:9000
```

### 4.3 Multi-GPU Data Parallel

```bash
apr finetune model.apr --method lora --gpus 0,1,2,3 --data train.jsonl
```

### 4.4 Classification Task

```bash
apr finetune model.apr --task classify --num-classes 5 --data train.jsonl
```

### 4.5 Plan Mode

Estimate VRAM, compute time, and optimal configuration without GPU allocation.

```bash
apr finetune model.apr --plan --model-size 7B --vram 24
```

---

## 5. Planned Methods (Arcee Adaptation Pipeline Parity)

### 5.1 CPT — Continual Pre-Training

**Purpose**: Domain knowledge acquisition from raw text corpora. First stage in
the 4-layer adaptation pipeline. Teaches the model domain vocabulary and
concepts without instruction formatting.

**CLI**:
```bash
apr finetune model.apr --method cpt \
  --data domain-corpus.txt \
  --epochs 1 \
  --learning-rate 2e-5 \
  -o cpt-model.apr
```

**Data format**: Plain text (one document per line) or raw text files.
Unlike SFT, no instruction/response pairs needed.

**Key differences from SFT**:
- Uses causal LM objective (next-token prediction) on raw text
- No chat template applied
- Typically lower learning rate (1e-5 to 5e-5)
- Longer training on more data

**Use cases**:
- Legal domain adaptation (train on case law, statutes)
- Medical domain (train on clinical notes, papers)
- Code domain (train on proprietary codebase)

### 5.2 DPO — Direct Preference Optimization

**Purpose**: Alignment stage. Teaches the model to prefer better outputs over
worse ones using human preference data. Replaces RLHF with a simpler,
more stable objective.

**CLI**:
```bash
apr finetune model.apr --method dpo \
  --data preferences.jsonl \
  --dpo-beta 0.1 \
  --epochs 3 \
  -o aligned-model.apr
```

**Data format** (JSONL):
```json
{"prompt": "...", "chosen": "preferred response", "rejected": "worse response"}
```

**Parameters**:
- `--dpo-beta`: KL penalty weight (default: 0.1). Higher = more conservative.
- `--dpo-label-smoothing`: Label smoothing (default: 0.0).
- `--ref-model`: Reference model path (default: use input model as reference).

**Implementation**: DPO loss = -log σ(β * (log π(chosen)/π_ref(chosen) - log π(rejected)/π_ref(rejected)))

### 5.3 RLVR — RL on Verifiable Rewards

**Purpose**: Post-alignment reinforcement learning using verifiable reward
signals (e.g., unit test pass/fail for code, mathematical proof checking,
format compliance).

**CLI**:
```bash
apr finetune model.apr --method rlvr \
  --reward-fn code-test \
  --data problems.jsonl \
  --rlvr-epochs 5 \
  -o rl-model.apr
```

**Reward functions**:
- `code-test`: Execute generated code, reward = test pass rate
- `math-verify`: Verify mathematical answer correctness
- `format-check`: Validate output format compliance
- `custom`: User-provided reward script

**Parameters**:
- `--reward-fn`: Reward function type
- `--reward-script`: Path to custom reward script (for `--reward-fn custom`)
- `--rlvr-kl-coeff`: KL penalty coefficient (default: 0.05)
- `--rlvr-clip-range`: PPO clip range (default: 0.2)

---

## 6. Planned Features

### 6.1 Synthetic Data Generation (EvolKit Parity)

Automatic instruction complexity enhancement using self-play or an auxiliary
model. Generates harder training examples from seed instructions.

**CLI**:
```bash
apr data evolve \
  --seed-data initial-instructions.jsonl \
  --model teacher-model.apr \
  --evolution-rounds 3 \
  --output evolved-data.jsonl
```

**Evolution strategies**:
- Complexity enhancement (add constraints, multi-step reasoning)
- Breadth enhancement (vary domains, contexts)
- Concretization (make abstract instructions specific)

### 6.2 PII Filtering

Automatic detection and removal of PII from training data before fine-tuning.

```bash
apr data filter-pii --input raw-data.jsonl --output clean-data.jsonl
```

**Detected PII types**: email, phone, SSN, credit card, IP address, names
(NER-based).

### 6.3 Domain-Specific Data Filtering

Quality scoring and filtering of training data based on domain relevance.

```bash
apr data filter --input raw.jsonl --domain legal --min-quality 0.7 -o filtered.jsonl
```

### 6.4 Full Pipeline Automation

End-to-end pipeline: `finetune → eval → serve` with automatic quality gates.

```bash
apr pipeline run --config pipeline.yaml
```

**Config**:
```yaml
stages:
  - finetune:
      method: qlora
      data: train.jsonl
      epochs: 3
  - eval:
      dataset: test.jsonl
      threshold: { accuracy: 0.85, ppl: 15.0 }
  - serve:
      port: 8080
      privacy: sovereign
```

---

## 7. 4-Layer Adaptation Pipeline (Arcee Parity Target)

The full adaptation pipeline for taking any base model to production:

```
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
│  CPT    │───►│  SFT    │───►│  DPO    │───►│  RLVR   │
│ (domain │    │ (instru-│    │ (align- │    │ (verify │
│ knowl-  │    │  ction  │    │  ment)  │    │  reward)│
│  edge)  │    │ tuning) │    │         │    │         │
└─────────┘    └─────────┘    └─────────┘    └─────────┘
    raw text    instruction     preference    verifiable
    corpus      pairs (JSONL)   pairs         reward fn
```

Each stage is optional. Common patterns:
- **Quick adapt**: SFT only (LoRA on instructions)
- **Aligned model**: SFT → DPO
- **Domain expert**: CPT → SFT → DPO
- **Verified agent**: CPT → SFT → DPO → RLVR

---

## 8. Data Format Reference

### SFT (current default)
```json
{"instruction": "Summarize this text", "input": "...", "output": "..."}
```

### CPT (plain text)
```
Document text here, one per line.
Another document here.
```

### DPO (preference pairs)
```json
{"prompt": "...", "chosen": "good response", "rejected": "bad response"}
```

### RLVR (problems + verifier)
```json
{"prompt": "Write a function...", "test_cases": ["assert f(1)==1", "assert f(5)==120"]}
```

---

## 9. Testing Requirements

- Unit tests for each method with toy models
- Integration test: LoRA finetune → merge → inference → assert quality
- Property tests: loss must decrease over epochs
- Distributed training: 2-node integration test
- DPO: verify loss converges with synthetic preference data
- RLVR: verify reward signal integration with mock verifier
- Mutation testing: >80% mutation score on training loops

---

## Provable Contracts

### Contract: `finetune-core-v1.yaml`

Extends `classification-finetune-v1.yaml` and `lora-algebra-v1.yaml`.

```yaml
metadata:
  description: "Fine-tuning — LoRA/QLoRA/Full with training loop invariants"
  references:
    - "Hu et al. (2021) LoRA: Low-Rank Adaptation"
    - "Dettmers et al. (2023) QLoRA"
  depends_on:
    - "lora-algebra-v1"
    - "classification-finetune-v1"
    - "cross-entropy-kernel-v1"
    - "adamw-kernel-v1"

equations:
  lora_forward:
    formula: "h = W·x + (B·A)·x * (α/r)"
    invariants:
      - "A ∈ ℝ^{r×d_in}, B ∈ ℝ^{d_out×r}"
      - "α/r scaling applied"
      - "Base weights W frozen during training"

  qlora_nf4:
    formula: "W_nf4 = quantize_nf4(W), h = dequant(W_nf4)·x + (B·A)·x * (α/r)"
    invariants:
      - "NF4 uses 4-bit normal float quantization"
      - "Dequantization on-the-fly during forward"
      - "~8x VRAM reduction vs full fine-tuning"

  training_loss:
    formula: "L = cross_entropy(logits, labels)"
    invariants:
      - "L >= 0"
      - "L decreases over training epochs (on training set)"
      - "Gradient flows through LoRA parameters only (base frozen)"

  adapter_merge:
    formula: "W_merged = W_base + B·A * (α/r)"
    invariants:
      - "Merged model has same architecture as base"
      - "No adapter matrices in merged output"
      - "Inference-equivalent to base + adapter"

proof_obligations:
  - type: invariant
    property: "Base weights frozen"
    formal: "W_base unchanged after training"
  - type: bound
    property: "Loss non-negativity"
    formal: "cross_entropy(logits, labels) >= 0"
  - type: equivalence
    property: "Adapter merge equivalence"
    formal: "forward(merged, x) == forward(base + adapter, x)"
  - type: invariant
    property: "LoRA shape consistency"
    formal: "A.shape[0] == B.shape[1] == rank"
  - type: invariant
    property: "NF4 VRAM reduction"
    formal: "qlora_vram < full_vram / 4"

falsification_tests:
  - id: FALSIFY-FT-001
    rule: "Base frozen"
    prediction: "base model weights identical before and after LoRA training"
    if_fails: "Optimizer includes base params"
  - id: FALSIFY-FT-002
    rule: "Loss non-negative"
    prediction: "training loss >= 0 for all batches"
    if_fails: "Cross-entropy computation bug"
  - id: FALSIFY-FT-003
    rule: "Merge equivalence"
    prediction: "merged model output == base+adapter output for same input"
    if_fails: "Merge formula α/r scaling wrong"
  - id: FALSIFY-FT-004
    rule: "Rank consistency"
    prediction: "A.rows == B.cols == specified rank"
    if_fails: "LoRA matrix initialization wrong shape"
  - id: FALSIFY-FT-005
    rule: "Auto method selection"
    prediction: "auto selects qlora when vram < 1.5× model size"
    if_fails: "VRAM threshold in auto selection wrong"

kani_harnesses:
  - id: KANI-FT-001
    obligation: "Loss non-negativity"
    property: "cross_entropy >= 0 for bounded logits"
    bound: 8
    strategy: stub_float
```

### Binding Requirements

```yaml
  - contract: finetune-core-v1.yaml
    equation: lora_forward
    module_path: "entrenar_lora"
    function: lora_forward
    status: implemented

  - contract: finetune-core-v1.yaml
    equation: adapter_merge
    module_path: "entrenar_lora::MergeEngine"
    function: merge_adapter
    status: implemented

  - contract: finetune-core-v1.yaml
    equation: qlora_nf4
    module_path: "entrenar_lora"
    function: quantize_nf4
    status: implemented
```
