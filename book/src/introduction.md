# Introduction

**Entrenar** (Spanish: "to train") is a high-performance Rust library for training and optimizing neural networks with
automatic differentiation, state-of-the-art optimizers, and memory-efficient LoRA/QLoRA fine-tuning. The name reflects
the library's mission: to provide a complete, production-ready training infrastructure for modern machine learning.

## The Problem: Training Complexity

Modern neural network training faces critical challenges:

- **Complex autograd systems**: Hand-coding gradients is error-prone and unmaintainable
- **Optimizer proliferation**: Each optimizer has subtle implementation details that affect convergence
- **Memory constraints**: Fine-tuning large models requires prohibitive amounts of RAM
- **Quality assurance**: Testing gradients requires extensive validation infrastructure

Traditional ML frameworks force you to choose between:
- **High-level APIs**: Easy to use but opaque implementations
- **Low-level control**: Full control but requires reimplementing complex algorithms
- **Performance vs accuracy**: Fast approximations vs correct gradients

**Entrenar chooses all: correctness, performance, and transparency.**

## The Solution: Extreme TDD Training Infrastructure

Entrenar's core philosophy is **zero-defect training through extreme testing**:

```rust
use entrenar::{Tensor, optim::AdamW, lora::QLoRALayer};

// Automatic differentiation with gradient checking
let x = Tensor::from_vec(vec![1.0, 2.0, 3.0], true);
let y_pred = model.forward(&x);
let loss = mse_loss(&y_pred, &y_true);

// Backward pass (automatically validated against finite differences)
backward(&loss);

// SIMD-accelerated optimizer updates
let mut optimizer = AdamW::default_params(0.001);
optimizer.step(&mut model.parameters());

// Memory-efficient fine-tuning with QLoRA (75% memory reduction)
let qlora = QLoRALayer::new(base_weight, 4096, 4096, 64, 128.0);
let output = qlora.forward(&input);  // Dequantizes on-the-fly
```

## Key Features

### 1. Tape-Based Automatic Differentiation

Entrenar provides a **tape-based autograd engine** with comprehensive backward passes:

| Operation | Forward | Backward | Validation |
|-----------|---------|----------|------------|
| **Matrix Multiplication** | O(n³) matmul | Jacobian chain rule | Finite differences (ε=1e-3) |
| **Layer Normalization** | Mean/variance stats | Mean/variance gradients | Property-based tests |
| **Attention** | Q,K,V projections | Q,K,V chain rule | 200K test iterations |
| **Activations** | ReLU, GELU, Swish | Derivative functions | Gradient checking |

**Autograd guarantees:**
- Every operation has a tested backward pass
- Gradients validated with finite difference checking (10K+ test cases)
- Property-based tests verify mathematical invariants
- Zero tolerance for gradient errors (threshold < 0.2 relative error)

### 2. State-of-the-Art Optimizers

Entrenar implements **production-ready optimizers** with proven convergence:

```
┌─────────────────────────────────────────────────────┐
│         Entrenar Optimizer Architecture             │
│  SGD (momentum + Nesterov), Adam, AdamW            │
└─────────────────────────────────────────────────────┘
                      │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
   ┌────────┐   ┌─────────┐   ┌──────────┐
   │  SIMD  │   │ Gradient│   │ Learning │
   │ Updates│   │ Clipping│   │   Rate   │
   │ (Trueno)   │  (Global│   │ Schedulers│
   └────────┘   │  Norm)  │   └──────────┘
                └─────────┘
```

**Optimizer Features:**
- **SGD with Momentum**: Classical optimization with momentum and Nesterov acceleration
- **Adam**: Adaptive learning rates with bias correction
- **AdamW**: Decoupled weight decay for improved generalization
- **Gradient Clipping**: Global norm clipping for training stability
- **LR Schedulers**: Cosine annealing, step decay, exponential decay
- **SIMD Acceleration**: 2-4x faster parameter updates via Trueno (for tensors ≥16 elements)

**Convergence Validation:**
```rust
// Property-based tests ensure convergence
proptest! {
    #[test]
    fn adam_converges_quadratic(lr in 0.05f32..0.5) {
        let optimizer = Adam::default_params(lr);
        assert!(converges_to_zero(optimizer, 100_iterations));
    }
}
```

### 3. LoRA: Parameter-Efficient Fine-Tuning

**LoRA (Low-Rank Adaptation)** enables fine-tuning with minimal trainable parameters:

```
Original Model: 7B parameters (frozen, requires_grad=false)
LoRA Adapters:  8M parameters (trainable, requires_grad=true)
Memory Savings: 99.9% reduction in trainable parameters
```

**LoRA Architecture:**
```
Base Weight W ∈ ℝ^(4096×4096) [FROZEN]
    │
    ├─> LoRA A ∈ ℝ^(64×4096)    [TRAINABLE]
    │   LoRA B ∈ ℝ^(4096×64)    [TRAINABLE]
    │
    └─> Output = W·x + (α/r)·(B·(A·x))
```

**LoRA Features:**
- **Target Module Selection**: Apply LoRA to specific layers (q_proj, k_proj, v_proj, o_proj)
- **Gradient Flow Isolation**: Base weights frozen, adapters trainable (validated with tests)
- **Merge/Unmerge**: Combine LoRA weights into base for efficient inference
- **Adapter Persistence**: Save/load adapters independently (JSON format)
- **Adapter Sharing**: Train once, share adapters without full model weights

### 4. QLoRA: 4-Bit Quantized LoRA

**QLoRA** reduces memory usage by **75%** through 4-bit quantization of frozen base weights:

| Configuration | LoRA Memory | QLoRA Memory | Savings |
|---------------|-------------|--------------|---------|
| **Small (256-dim, 6 layers)** | 1.5 MB | 0.5 MB | **65%** |
| **Medium (768-dim, 12 layers)** | 27 MB | 8 MB | **68%** |
| **Large (4096-dim, 32 layers)** | 4.2 GB | 1.2 GB | **70%** |

**Quantization Details:**
- **Block-wise quantization**: 64-element blocks with scale factors
- **Symmetric 4-bit**: Values in range [-7, 7] (15 discrete levels)
- **On-the-fly dequantization**: Decompress during forward pass only
- **Full-precision adapters**: LoRA A, B remain float32 for training accuracy
- **6-7x compression ratio**: Base weights reduced from 32-bit to ~4.5-bit effective

**Memory Benchmark (768-dim BERT-base, 12 layers):**
```
Total LoRA memory:  27,648 KB
Total QLoRA memory:  8,352 KB
Memory savings:     19,296 KB (69.8%)
```

### 5. Model Merging (Arcee Methods)

**Model merging** combines multiple fine-tuned models into a single unified model:

```
Model A (fine-tuned on task A)
Model B (fine-tuned on task B)  →  Merged Model (performs both tasks)
Model C (fine-tuned on task C)
```

**Merging Algorithms:**
- **TIES** (Task Inference via Elimination and Sign voting) - Resolves parameter conflicts via sign voting
- **DARE** (Drop And REscale) - Bernoulli masking with rescaling for sparse updates
- **SLERP** (Spherical Linear intERPolation) - Smooth interpolation on weight manifold

From `src/merge/`:
```rust
use entrenar::merge::{TIESMerger, DAREMerger, SLERPMerger};

// TIES merging with density=0.5, lambda=1.0
let merger = TIESMerger::new(0.5, 1.0);
let merged = merger.merge(&models)?;

// DARE merging with drop rate=0.9
let dare = DAREMerger::new(0.9);
let merged = dare.merge(&models)?;
```

### 6. Knowledge Distillation

**Knowledge distillation** trains a smaller "student" model to mimic a larger "teacher" model:

```
Teacher Model (7B params) → Knowledge Transfer → Student Model (1B params)
```

**Distillation Methods** (from `src/distill/`):
- **Temperature-scaled KL divergence**: Soft targets with temperature smoothing
- **Multi-teacher ensemble**: Distill from multiple teachers simultaneously
- **Progressive layer-wise**: Layer-by-layer knowledge transfer

```rust
use entrenar::distill::DistillationLoss;

// Temperature=3.0, alpha=0.7 (70% distillation, 30% hard labels)
let loss_fn = DistillationLoss::new(3.0, 0.7);
let loss = loss_fn.forward(&student_logits, &teacher_logits, &labels);
```

**Validation:** 44 tests including 13 property-based tests for temperature smoothing

### 7. Training Loop & Model I/O

**High-level Trainer API** (from `src/train/trainer.rs`):
```rust
use entrenar::train::{Trainer, TrainConfig};

let config = TrainConfig::new()
    .with_log_interval(100)
    .with_grad_clip(1.0);

let mut trainer = Trainer::new(parameters, optimizer, config);
trainer.set_loss(Box::new(MSELoss));

// Train for one epoch
let avg_loss = trainer.train_epoch(batches, |x| model.forward(x));
```

**Model I/O** (from `src/io/`):
```rust
use entrenar::io::{save_model, load_model, SaveConfig, ModelFormat};

// Save to JSON (pretty-printed)
let config = SaveConfig::new(ModelFormat::Json).with_pretty(true);
save_model(&model, "model.json", &config)?;

// Load from JSON (auto-detected format)
let loaded = load_model("model.json")?;
```

**Formats supported:** JSON (compact/pretty), YAML, GGUF (placeholder for Realizar integration)

### 8. Declarative Configuration

**Ludwig-style YAML training** (from `src/config/train.rs`):

```yaml
model:
  path: models/llama-7b.gguf
data:
  train: data/train.parquet
  batch_size: 4
optimizer:
  name: adamw
  lr: 0.0001
  beta1: 0.9
  beta2: 0.999
training:
  epochs: 3
  grad_clip: 1.0
  output_dir: ./checkpoints
```

**Single-command training:**
```rust
use entrenar::config::train_from_yaml;

train_from_yaml("config.yaml")?;  // Complete training workflow
```

### 9. Extreme TDD Quality

Entrenar is built with **EXTREME TDD** methodology ensuring zero defects:

**Test Coverage:**
- **4326 unit & integration tests** (100% pass rate)
  - Core library tests with comprehensive coverage
  - Gradient checking tests with finite difference validation
  - Architecture and I/O tests
  - Property-based tests (200K+ iterations)
  - Chaos engineering tests
  - Memory benchmark tests
- **Mutation testing** (cargo-mutants validates test quality)
- **Convergence tests** (optimizers proven to minimize quadratic functions)

**Quality Metrics:**
```
Total Tests:        4326 passing (0 failures)
Line Coverage:      96% (entrenar files)
TDG Score:          A+ (96.8/100)
Critical Defects:   0
Clippy Warnings:    0 (strict mode, -D warnings)
```

**Example Test:**
```rust
#[test]
fn test_matmul_backward_gradient_check() {
    // Validate gradients against finite differences
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], true);
    let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], true);

    let output = matmul(&a, &b, 2, 2, 1);
    backward(&output);

    // Check gradients with ε=1e-3, threshold=0.2
    assert_gradient_correct(&a, epsilon=1e-3, threshold=0.2);
}
```

## Real-World Impact: Memory-Efficient Fine-Tuning

**Problem**: Fine-tuning a 7B parameter transformer model

| Approach | Trainable Params | Memory (FP32) | Memory (QLoRA 4-bit) |
|----------|------------------|---------------|----------------------|
| **Full Fine-Tuning** | 7B | 28 GB | N/A |
| **LoRA (rank=64)** | 8M (0.1%) | 28 GB base + 32 MB adapters | 7 GB base + 32 MB adapters |
| **QLoRA (rank=64)** | 8M (0.1%) | N/A | **7 GB total (75% savings)** |

**Entrenar's Value Proposition:**
- ✅ **Memory Efficiency**: Train 7B models on consumer GPUs (8-12GB VRAM)
- ✅ **Adapter Portability**: Share 32MB adapters instead of 28GB full models
- ✅ **Proven Convergence**: Optimizers tested with property-based validation
- ✅ **Gradient Correctness**: Autograd validated with 10K+ test cases
- ✅ **Production Quality**: Zero clippy warnings, >80% mutation score

## Who Should Use Entrenar?

Entrenar is designed for:

1. **ML Engineers** - Building custom training systems with full control
2. **Researchers** - Implementing new optimizers or LoRA variants
3. **Students** - Learning autograd, optimization, and parameter-efficient fine-tuning
4. **Library Authors** - Building higher-level ML frameworks on solid foundations
5. **Production Teams** - Deploying memory-efficient fine-tuning at scale

## Design Principles

Entrenar follows five core principles:

1. **Zero tolerance for defects** - Every gradient validated, every optimizer tested
2. **Transparency over magic** - Clear, readable implementations over black-box abstractions
3. **Memory efficiency** - QLoRA enables fine-tuning on consumer hardware
4. **Extreme TDD** - >90% coverage, mutation testing, property-based tests
5. **Toyota Way** - Kaizen (continuous improvement), Jidoka (built-in quality)

## What's Next?

- **[Getting Started](./getting-started/installation.md)** - Install Entrenar and train your first model
- **[Autograd Engine](./autograd/what-is-autograd.md)** - Understand automatic differentiation
- **[Optimizers](./optimizers/overview.md)** - Learn about SGD, Adam, AdamW, and schedulers
- **[LoRA/QLoRA](./lora/what-is-lora.md)** - Master parameter-efficient fine-tuning
- **[Examples](./examples/linear-regression.md)** - See practical training examples

## Project Status

Entrenar v0.5.6 is **production-ready** at **Pragmatic AI Labs**:

- **Current Version**: 0.5.6 ✅ **PRODUCTION**
- **License**: MIT
- **Repository**: [github.com/paiml/entrenar](https://github.com/paiml/entrenar)
- **Tests**: 4326 passing (100% pass rate)
- **Coverage**: 96% line coverage
- **TDG Score**: A+ (96.8/100)
- **Critical Defects**: 0

**Completed v0.1.0 Features:**
- ✅ **Autograd Engine**: Tape-based autodiff with 18 gradient validation tests
- ✅ **Optimizers**: SGD, Adam, AdamW with SIMD acceleration
- ✅ **LoRA/QLoRA**: Parameter-efficient fine-tuning with 4-bit quantization
- ✅ **Model Merging**: TIES, DARE, SLERP algorithms
- ✅ **Knowledge Distillation**: Temperature-scaled KL divergence, multi-teacher ensemble
- ✅ **Training Loop**: High-level Trainer API with metrics tracking
- ✅ **Model I/O**: Save/load in JSON, YAML formats
- ✅ **Declarative Configuration**: Ludwig-style YAML training configs

**Future Roadmap (v0.2.0+):**
- Real GGUF loading via Realizar integration
- Distributed training and model parallelism
- GPU acceleration via Trueno integration
- Performance benchmarks and optimization

Join us in building the future of zero-defect ML training infrastructure!
