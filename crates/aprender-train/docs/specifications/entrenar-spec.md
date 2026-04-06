# Entrenar: Training & Optimization Specification

**Repo:** https://github.com/paiml/entrenar  
**Stack Deps:** Trueno (compute), Realizar (I/O), Aprender (loss fns)  
**External:** None (100% PAIML)

---

## Core Type System

```rust
// Computational graph node with gradient tape
pub struct Tensor<'g, T> {
    data: trueno::Tensor<'g, T>,
    grad: Option<trueno::Tensor<'g, T>>,
    op: Option<Box<dyn BackwardOp<'g, T>>>,
}

// Backward closure
trait BackwardOp<'g, T>: Send + Sync {
    fn backward(&self, grad_output: &trueno::Tensor<'g, T>) -> Vec<trueno::Tensor<'g, T>>;
}

// Constraint: 'g >= lifetime of all inputs to ensure gradient tape validity
```

---

## Layer 1: Autograd Engine

**Tape-based (eager mode):**
```rust
pub struct Context<'g> {
    tape: Vec<Box<dyn BackwardOp<'g, f32>>>,
    tensors: Vec<Rc<RefCell<Tensor<'g, f32>>>>,
}

impl<'g> Context<'g> {
    pub fn matmul(&mut self, a: &Tensor<'g, f32>, b: &Tensor<'g, f32>) -> Tensor<'g, f32> {
        let out = trueno::Matrix::matmul(&a.data, &b.data);
        let grad_fn = MatmulBackward { a: a.clone(), b: b.clone() };
        Tensor { data: out, grad: None, op: Some(Box::new(grad_fn)) }
    }
    
    pub fn backward(&mut self, loss: &mut Tensor<'g, f32>) {
        loss.grad = Some(trueno::Tensor::ones_like(&loss.data));
        for node in self.tape.iter().rev() {
            node.backward(/* ... */);
        }
    }
}
```

**Ops Required (Trueno extensions):**
- `softmax_backward`: `∂L/∂x = softmax(x) * (∂L/∂y - dot(∂L/∂y, softmax(x)))`
- `layer_norm_backward`: `∂L/∂x = γ/σ * (∂L/∂y - mean(∂L/∂y) - (x-μ)/σ * mean((x-μ) * ∂L/∂y))`
- `relu_backward`: `∂L/∂x = ∂L/∂y * (x > 0)`

---

## Layer 2: Optimizers

```rust
pub trait Optimizer {
    fn step(&mut self, params: &mut [Tensor], grads: &[Tensor]);
    fn zero_grad(&mut self, params: &mut [Tensor]);
}

pub struct Adam {
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    t: u64,
    m: Vec<trueno::Tensor>,  // 1st moment
    v: Vec<trueno::Tensor>,  // 2nd moment
}

impl Optimizer for Adam {
    fn step(&mut self, params: &mut [Tensor], grads: &[Tensor]) {
        self.t += 1;
        let lr_t = self.lr * ((1.0 - self.beta2.powi(self.t as i32)).sqrt() 
                              / (1.0 - self.beta1.powi(self.t as i32)));
        
        for (i, (p, g)) in params.iter_mut().zip(grads).enumerate() {
            // m_t = β1*m_{t-1} + (1-β1)*g
            self.m[i] = self.m[i].scale(self.beta1).add(&g.data.scale(1.0 - self.beta1));
            // v_t = β2*v_{t-1} + (1-β2)*g²
            self.v[i] = self.v[i].scale(self.beta2).add(&g.data.mul(&g.data).scale(1.0 - self.beta2));
            // p_t = p_{t-1} - lr_t * m_t / (√v_t + ε)
            let update = self.m[i].div(&self.v[i].sqrt().add_scalar(self.eps)).scale(lr_t);
            p.data = p.data.sub(&update);
        }
    }
}
```

**Variants:** SGD, AdamW (weight decay), RMSprop.

---

## Layer 3: Quantization

### QAT (Quantization-Aware Training)

```rust
pub struct QATConfig {
    bits: u8,           // 4, 8
    symmetric: bool,    // [-127,127] vs [0,255]
    per_channel: bool,  // Per-row scales
}

pub struct FakeQuantize {
    config: QATConfig,
    scales: Vec<f32>,
    zero_points: Vec<i8>,
}

impl FakeQuantize {
    // Forward: x → quant(x) → dequant → x'
    pub fn forward(&self, x: &trueno::Tensor) -> trueno::Tensor {
        let q = self.quantize(x);
        self.dequantize(&q)
    }
    
    // Backward: Straight-through estimator (STE)
    pub fn backward(&self, grad: &trueno::Tensor) -> trueno::Tensor {
        grad.clone()  // ∂dequant/∂x ≈ I
    }
    
    fn quantize(&self, x: &trueno::Tensor) -> Vec<i8> {
        let scale = (x.max() - x.min()) / ((1 << self.config.bits) - 1) as f32;
        x.as_slice().iter().map(|&v| {
            ((v / scale).round() as i8).clamp(-128, 127)
        }).collect()
    }
}
```

### PTQ (Post-Training Quantization)

```rust
pub fn calibrate(model: &mut Model, data: &[trueno::Tensor]) -> Vec<f32> {
    let mut activations = Vec::new();
    for batch in data {
        let acts = model.forward(batch);
        activations.push(acts);
    }
    
    // Compute per-layer scales (min-max or percentile)
    activations.iter().map(|a| {
        let min = a.min();
        let max = a.max();
        (max - min) / 255.0
    }).collect()
}

pub fn quantize_weights(weights: &trueno::Tensor, bits: u8) -> (Vec<u8>, f32) {
    let scale = weights.max().abs() / ((1 << (bits - 1)) - 1) as f32;
    let quantized = weights.as_slice().iter().map(|&w| {
        ((w / scale).round() as i8 + 128) as u8  // Offset to [0,255]
    }).collect();
    (quantized, scale)
}
```

**Output:** Realizar-compatible GGUF with Q4_0/Q8_0 blocks.

---

## Layer 4: LoRA/QLoRA

```rust
pub struct LoRALayer<'a> {
    base_weight: &'a trueno::Tensor,  // Frozen
    lora_a: trueno::Tensor,            // r×d_in (trainable)
    lora_b: trueno::Tensor,            // d_out×r (trainable)
    rank: usize,                       // r << min(d_in, d_out)
    alpha: f32,                        // Scaling factor
}

impl<'a> LoRALayer<'a> {
    pub fn forward(&self, x: &trueno::Tensor) -> trueno::Tensor {
        // W_full = W_base + (α/r) * B @ A
        let base_out = self.base_weight.matmul(x);
        let lora_out = self.lora_b.matmul(&self.lora_a.matmul(x)).scale(self.alpha / self.rank as f32);
        base_out.add(&lora_out)
    }
    
    pub fn merge(&self) -> trueno::Tensor {
        // W_merged = W_base + (α/r) * B @ A
        let delta = self.lora_b.matmul(&self.lora_a).scale(self.alpha / self.rank as f32);
        self.base_weight.add(&delta)
    }
}

// QLoRA: 4-bit base + LoRA adapters
pub struct QLoRALayer<'a> {
    base_weight_q: &'a [u8],  // Q4_0 frozen
    scale: f32,
    lora_a: trueno::Tensor,
    lora_b: trueno::Tensor,
}

impl<'a> QLoRALayer<'a> {
    pub fn forward(&self, x: &trueno::Tensor) -> trueno::Tensor {
        let w_base = realizar::dequantize_q4_0(self.base_weight_q);  // On-the-fly
        let base_out = trueno::Tensor::from_slice(&w_base).matmul(x);
        let lora_out = self.lora_b.matmul(&self.lora_a.matmul(x));
        base_out.add(&lora_out)
    }
}
```

**Memory:** QLoRA reduces VRAM by 4× (4-bit base vs 16-bit fp16).

---

## Layer 5: Model Merging (Arcee)

```rust
pub enum MergeMethod {
    TIES { density: f32 },      // Task Inference via Elimination and Sign
    DARE { p: f32 },            // Drop And REscale
    SLERP { t: f32 },           // Spherical Linear intERPolation
}

pub fn merge_models(models: &[Model], method: MergeMethod) -> Model {
    match method {
        MergeMethod::TIES { density } => {
            // 1. Trim: τ = top-k% magnitude
            let deltas: Vec<_> = models.iter().map(|m| m.delta_from_base()).collect();
            let trimmed = deltas.iter().map(|d| d.trim(density)).collect();
            // 2. Elect sign: majority vote per parameter
            let signs = elect_signs(&trimmed);
            // 3. Merge: mean of same-sign parameters
            merge_by_sign(&trimmed, &signs)
        }
        MergeMethod::DARE { p } => {
            // Bernoulli(1-p) mask on delta weights, rescale by 1/(1-p)
            let deltas: Vec<_> = models.iter().map(|m| m.delta_from_base()).collect();
            let masked = deltas.iter().map(|d| d.dropout(p).scale(1.0/(1.0-p))).collect();
            average_deltas(&masked)
        }
        MergeMethod::SLERP { t } => {
            // θ = arccos(w0·w1 / |w0||w1|)
            // w = (sin((1-t)θ)/sinθ)*w0 + (sin(tθ)/sinθ)*w1
            let (w0, w1) = (&models[0].weights, &models[1].weights);
            let cos_theta = w0.dot(w1) / (w0.norm() * w1.norm());
            let theta = cos_theta.acos();
            let sin_theta = theta.sin();
            w0.scale((1.0-t)*theta).sin().div(sin_theta)
              .add(&w1.scale(t*theta).sin().div(sin_theta))
        }
    }
}

fn elect_signs(deltas: &[Tensor]) -> Tensor {
    // Per-parameter sign election (majority vote)
    deltas[0].map_indices(|i| {
        let pos = deltas.iter().filter(|d| d[i] > 0.0).count();
        if pos > deltas.len() / 2 { 1.0 } else { -1.0 }
    })
}
```

**Output:** Merged model → Realizar GGUF.

---

## Layer 6: Distillation

```rust
pub struct DistillationLoss {
    temperature: f32,
    alpha: f32,  // Weight for distillation vs task loss
}

impl DistillationLoss {
    pub fn forward(
        &self,
        student_logits: &trueno::Tensor,
        teacher_logits: &trueno::Tensor,
        labels: &[usize],
    ) -> f32 {
        // Soft targets: KL(softmax(teacher/T) || softmax(student/T))
        let s_soft = student_logits.scale(1.0 / self.temperature).softmax();
        let t_soft = teacher_logits.scale(1.0 / self.temperature).softmax();
        let kl_loss = self.kl_divergence(&s_soft, &t_soft) * self.temperature.powi(2);
        
        // Hard targets: Cross-entropy(student, labels)
        let ce_loss = self.cross_entropy(student_logits, labels);
        
        self.alpha * kl_loss + (1.0 - self.alpha) * ce_loss
    }
    
    fn kl_divergence(&self, p: &trueno::Tensor, q: &trueno::Tensor) -> f32 {
        // KL(p||q) = Σ p * log(p/q)
        p.as_slice().iter().zip(q.as_slice()).map(|(&p_i, &q_i)| {
            p_i * (p_i / q_i).ln()
        }).sum()
    }
}
```

---

## Layer 6: Model I/O Integration

```rust
pub struct TrainingConfig {
    model_path: PathBuf,     // Load from Realizar GGUF
    output_path: PathBuf,    // Save to GGUF/Safetensors
    lora_rank: Option<usize>,
    quantize: Option<QATConfig>,
}

pub fn load_model(path: &Path) -> Result<Model> {
    let gguf = realizar::parse_gguf(&std::fs::read(path)?)?;
    // Convert realizar::Model → entrenar::Model with gradient tracking
    Ok(Model::from_realizar(gguf))
}

pub fn save_model(model: &Model, path: &Path, config: &TrainingConfig) -> Result<()> {
    let tensors = if let Some(qat) = &config.quantize {
        model.quantize(qat)  // Q4_0/Q8_0 bit packing
    } else {
        model.weights()
    };
    
    let gguf = realizar::create_gguf(tensors, model.config())?;
    std::fs::write(path, gguf)?;
    Ok(())
}
```

---

## Layer 7: Declarative Config (Ludwig-inspired)

```rust
// YAML → struct deserialization
#[derive(Deserialize)]
pub struct TrainSpec {
    model: ModelRef,
    data: DataConfig,
    optimizer: OptimSpec,
    lora: Option<LoRASpec>,
    quantize: Option<QATConfig>,
    merge: Option<MergeConfig>,
}

#[derive(Deserialize)]
pub struct ModelRef {
    path: PathBuf,      // GGUF base model
    layers: Vec<String>, // Target layers for LoRA
}

#[derive(Deserialize)]
pub struct OptimSpec {
    name: String,       // "adam" | "sgd" | "adamw"
    lr: f32,
    #[serde(flatten)]
    params: serde_json::Value,  // Optimizer-specific (beta1, momentum, etc.)
}

#[derive(Deserialize)]
pub struct DataConfig {
    train: PathBuf,
    val: Option<PathBuf>,
    batch_size: usize,
    auto_infer_types: bool,  // Infer feature types from data
}

// Auto-inference from data schema
pub fn infer_features(data: &Path) -> Vec<Feature> {
    let sample = read_first_batch(data);
    sample.columns().map(|(name, col)| {
        Feature {
            name: name.to_string(),
            dtype: match col.dtype() {
                DataType::Float32 => FeatureType::Numerical,
                DataType::Utf8 => FeatureType::Categorical,
                _ => FeatureType::Binary,
            },
        }
    }).collect()
}

// Single entry point
pub fn train_from_yaml(path: &Path) -> Result<Model> {
    let spec: TrainSpec = serde_yaml::from_reader(File::open(path)?)?;
    
    let model = load_model(&spec.model.path)?;
    if let Some(lora) = spec.lora {
        model.add_lora_layers(lora.rank);
    }
    
    let opt = build_optimizer(&spec.optimizer)?;
    let trainer = Trainer::new(model, opt, CrossEntropyLoss::new());
    
    trainer.train(&spec.data)?;
    
    if let Some(merge) = spec.merge {
        let merged = merge_models(&[model, other_models...], merge.method);
        return Ok(merged);
    }
    
    Ok(trainer.model)
}
```

**Example YAML:**
```yaml
model:
  path: llama-3-7b.gguf
  layers: [q_proj, k_proj, v_proj, o_proj]

data:
  train: train.parquet
  val: val.parquet
  batch_size: 8
  auto_infer_types: true

optimizer:
  name: adam
  lr: 1e-4
  beta1: 0.9
  beta2: 0.999

lora:
  rank: 64
  alpha: 16
  target_modules: [q_proj, v_proj]

quantize:
  bits: 4
  symmetric: true
  per_channel: true

merge:
  method: TIES
  density: 0.2
```

---

## Layer 8: Training Loop

```rust
pub struct Trainer<'g> {
    model: Model<'g>,
    optimizer: Box<dyn Optimizer>,
    loss_fn: Box<dyn LossFn>,
    ctx: Context<'g>,
}

impl<'g> Trainer<'g> {
    pub fn train_step(&mut self, batch: &Batch) -> f32 {
        // Forward
        let logits = self.model.forward(&batch.inputs, &mut self.ctx);
        let loss = self.loss_fn.compute(&logits, &batch.targets);
        
        // Backward
        self.optimizer.zero_grad(&mut self.model.params);
        self.ctx.backward(&mut loss);
        
        // Update
        let grads: Vec<_> = self.model.params.iter().map(|p| p.grad.unwrap()).collect();
        self.optimizer.step(&mut self.model.params, &grads);
        
        loss.data.as_slice()[0]
    }
    
    pub fn train_epoch(&mut self, dataloader: &DataLoader) -> f32 {
        let mut total_loss = 0.0;
        for batch in dataloader {
            total_loss += self.train_step(&batch);
        }
        total_loss / dataloader.len() as f32
    }
}
```

---

## Trueno Extensions Required

**New ops in `trueno/src/ops/`:**

```rust
// trueno/src/ops/backward.rs
pub fn softmax_backward(y: &Tensor, dy: &Tensor) -> Tensor {
    // ∂L/∂x = y ⊙ (∂L/∂y - (y · ∂L/∂y)·1)
    let dot = y.mul(dy).sum();
    dy.sub(&Tensor::filled(dy.len(), dot)).mul(y)
}

pub fn layer_norm_backward(
    x: &Tensor,
    gamma: &Tensor,
    mean: f32,
    var: f32,
    dy: &Tensor,
) -> (Tensor, Tensor, Tensor) {
    let n = x.len() as f32;
    let std_inv = 1.0 / var.sqrt();
    let x_hat = x.sub_scalar(mean).scale(std_inv);
    
    // ∂L/∂γ = Σ(∂L/∂y ⊙ x̂)
    let dgamma = dy.mul(&x_hat).sum();
    // ∂L/∂β = Σ(∂L/∂y)
    let dbeta = dy.sum();
    // ∂L/∂x = γ/σ * (∂L/∂y - mean(∂L/∂y) - x̂·mean(x̂⊙∂L/∂y))
    let dx = dy.sub_scalar(dy.mean())
               .sub(&x_hat.scale(x_hat.mul(dy).mean()))
               .scale(std_inv)
               .mul(gamma);
    
    (dx, Tensor::from_slice(&[dgamma]), Tensor::from_slice(&[dbeta]))
}

pub fn attention_backward(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scores: &Tensor,
    weights: &Tensor,
    dout: &Tensor,
) -> (Tensor, Tensor, Tensor) {
    // ∂L/∂v = weights^T @ ∂L/∂out
    let dv = weights.transpose().matmul(dout);
    // ∂L/∂weights = ∂L/∂out @ v^T
    let dweights = dout.matmul(&v.transpose());
    // ∂L/∂scores = softmax_backward(weights, dweights)
    let dscores = softmax_backward(weights, &dweights);
    // ∂L/∂q = ∂L/∂scores @ k, ∂L/∂k = ∂L/∂scores^T @ q
    let dq = dscores.matmul(k);
    let dk = dscores.transpose().matmul(q);
    
    (dq, dk, dv)
}
```

**Backend dispatch:** Use SIMD for small batches, GPU for large (same 5× rule).

---

## Dependencies

```toml
[dependencies]
trueno = { path = "../trueno" }
realizar = { path = "../realizar" }
aprender = { path = "../aprender" }

# Declarative config
serde = { version = "1", features = ["derive"] }
serde_yaml = "0.9"
serde_json = "1"

[dev-dependencies]
proptest = "1.4"           # Property-based testing (200K+ iterations)
cargo-mutants = "25.3"     # Mutation testing (>80% kill rate)

# Quality gates via PMAT
# https://github.com/paiml/paiml-mcp-agent-toolkit
# Install: cargo install pmat
```

---

## Benchmarks (Target)

| Operation | Size | Target | Backend |
|-----------|------|--------|---------|
| Matmul backward | 512×512 | 3× forward | GPU |
| Adam step | 1M params | <10ms | SIMD |
| Softmax backward | 10K | 2× forward | SIMD |
| Q4_0 quantize | 1GB | <1s | Scalar |
| LoRA merge | 7B, r=64 | <5s | SIMD |

---

## EXTREME TDD Methodology

**Test-First Workflow:**
```rust
// 1. Property test first (gradient check)
#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    
    proptest! {
        #[test]
        fn softmax_backward_gradient_check(x in prop::collection::vec(-10.0f32..10.0, 1..100)) {
            let y = softmax(&x);
            let dy = vec![1.0; x.len()];
            let analytical = softmax_backward(&y, &dy);
            let numerical = finite_diff(|x| softmax(x).sum(), &x, 1e-5);
            prop_assert!((analytical - numerical).abs() < 1e-3);
        }
    }
}

// 2. Implement to pass
pub fn softmax_backward(y: &[f32], dy: &[f32]) -> Vec<f32> { /* ... */ }

// 3. Mutation test (cargo-mutants)
// 4. Refactor
```

**PMAT Integration:**
```bash
# Generate roadmap with 8h/4h/2h estimates
pmat roadmap generate \
    --spec docs/specifications/entrenar-spec.md \
    --output docs/roadmaps/roadmap.yaml

# Track progress per sprint
pmat roadmap update --ticket ENT-001 --status complete --actual-hours 6

# TDG scoring (Toyota Way quality)
pmat analyze tdg src/ --min-score 90

# Complexity analysis
pmat analyze complexity src/ --max-cyclomatic 10 --max-cognitive 15
```

**Quality Gates (CI):**
```yaml
# .github/workflows/quality.yml
- run: cargo test --all-features  # >90% coverage
- run: cargo llvm-cov --lcov > coverage.info
- run: cargo mutants --timeout 300  # >80% kill rate
- run: cargo clippy -- -D warnings
- run: pmat analyze tdg src/ --min-score 90
- run: pmat analyze complexity src/
```

**Property Tests (200K+ iterations):**
```rust
// Gradient correctness
proptest! {
    #[test]
    fn layer_norm_backward_gradient_check(
        x in prop::collection::vec(-5.0f32..5.0, 10..1000),
        gamma in 0.5f32..2.0,
    ) {
        let (mean, var) = compute_stats(&x);
        let y = layer_norm(&x, gamma, mean, var);
        let dy = vec![1.0; x.len()];
        
        let (dx_auto, dgamma_auto, _) = layer_norm_backward(&x, gamma, mean, var, &dy);
        let dx_num = finite_diff(|x| layer_norm(x, gamma, mean, var).sum(), &x, 1e-5);
        let dgamma_num = finite_diff(|g| layer_norm(&x, g, mean, var).sum(), &[gamma], 1e-5)[0];
        
        prop_assert!((dx_auto - dx_num).map(|d| d.abs()).max() < 1e-3);
        prop_assert!((dgamma_auto - dgamma_num).abs() < 1e-3);
    }
}

// Optimizer convergence
proptest! {
    #[test]
    fn adam_quadratic_convergence(
        start in prop::collection::vec(-10.0f32..10.0, 2..10),
    ) {
        let mut params = start.clone();
        let mut opt = Adam::new(0.1, 0.9, 0.999);
        
        for _ in 0..1000 {
            let grad = params.iter().map(|&p| 2.0 * p).collect();  // ∇(x²) = 2x
            opt.step(&mut params, &grad);
        }
        
        prop_assert!(params.iter().all(|&p| p.abs() < 1e-2));  // Converges to 0
    }
}

// Merge method commutativity
proptest! {
    #[test]
    fn ties_merge_order_invariant(
        models in prop::collection::vec(model_strategy(), 2..5),
    ) {
        let merged1 = merge_models(&models, TIES { density: 0.2 });
        let mut shuffled = models.clone();
        shuffled.shuffle();
        let merged2 = merge_models(&shuffled, TIES { density: 0.2 });
        
        prop_assert_eq!(merged1.weights, merged2.weights);
    }
}
```

**Mutation Testing Targets:**
```bash
# Focus on critical paths
cargo mutants --file src/autograd/backward.rs --timeout 300
cargo mutants --file src/optim/adam.rs --timeout 300
cargo mutants --file src/merge/ties.rs --timeout 300

# Per-function granularity
cargo mutants --function softmax_backward --timeout 60
```

**Pre-commit Hook:**
```bash
#!/bin/bash
cargo fmt --check || exit 1
cargo clippy -- -D warnings || exit 1
cargo test --lib || exit 1  # Fast unit tests only
pmat analyze tdg src/ --min-score 90 || exit 1
```

**Metrics:**
- Test coverage: >90% (llvm-cov)
- Mutation kill rate: >80% (cargo-mutants)
- TDG score: >90 (A grade, PMAT)
- Cyclomatic complexity: ≤10
- Cognitive complexity: ≤15
- Gradient error: <1e-3 (property tests)
- Convergence tests: 200K+ iterations (proptest)

---

## Roadmap (PMAT Tracked)

**Total: 824h (103 days @ 8h/day)**

### Phase 1: Autograd (Q1 2025) - 200h
- `ENT-001` Tape-based context + lifetime tracking (32h)
- `ENT-002` Matmul backward (gradient check: 200K iters) (16h)
- `ENT-003` Softmax backward + property tests (24h)
- `ENT-004` Layer norm backward (mean/var gradients) (32h)
- `ENT-005` Attention backward (Q,K,V chain rule) (40h)
- `ENT-006` ReLU/GELU/Swish backward (8h each) (24h)
- `ENT-007` Finite difference validation framework (16h)
- `ENT-008` Mutation testing on backward ops (>80% kill) (8h)

### Phase 2: Optimizers (Q1 2025) - 120h
- `ENT-009` SGD + momentum (16h)
- `ENT-010` Adam (m/v state tracking) (24h)
- `ENT-011` AdamW (decoupled weight decay) (16h)
- `ENT-012` Cosine LR scheduler (8h)
- `ENT-013` Gradient clipping (global norm) (8h)
- `ENT-014` Optimizer convergence property tests (32h)
- `ENT-015` SIMD-accelerated param updates via Trueno (16h)

### Phase 3: LoRA (Q2 2025) - 144h
- `ENT-016` LoRA layer (A, B matrices + merge) (32h)
- `ENT-017` QLoRA (4-bit base + dequant-on-fly) (40h)
- `ENT-018` Target module selection (q/k/v/o_proj) (16h)
- `ENT-019` Adapter save/load (separate from base) (24h)
- `ENT-020` Memory benchmarks (QLoRA vs full FP16) (16h)
- `ENT-021` Gradient flow tests (frozen base + trainable adapters) (16h)

### Phase 4: Quantization (Q2 2025) - 136h
- `ENT-022` Fake quantize (STE backward) (24h)
- `ENT-023` PTQ calibration (min-max, percentile) (32h)
- `ENT-024` Q4_0/Q8_0 bit packing → GGUF (40h)
- `ENT-025` Per-channel vs per-tensor quantization (16h)
- `ENT-026` Quantization error property tests (16h)
- `ENT-027` Accuracy degradation benchmarks (8h)

### Phase 5: Model Merging (Q2 2025) - 96h
- `ENT-028` TIES (trim + sign election + merge) (32h)
- `ENT-029` DARE (dropout + rescale) (24h)
- `ENT-030` SLERP (spherical interp for 2 models) (24h)
- `ENT-031` Merge commutativity property tests (8h)
- `ENT-032` Multi-model ensemble (>2 models) (8h)

### Phase 6: Declarative Config (Q3 2025) - 64h
- `ENT-033` YAML schema + serde deserialization (16h)
- `ENT-034` Auto-feature type inference from data (24h)
- `ENT-035` Config validation (types, paths, ranges) (16h)
- `ENT-036` Single-command training entry point (8h)

### Phase 7: Distillation (Q3 2025) - 64h
- `ENT-037` KD loss (temperature-scaled softmax) (16h)
- `ENT-038` Multi-teacher ensemble distillation (24h)
- `ENT-039` Progressive distillation (layer-wise) (16h)
- `ENT-040` Distillation effectiveness property tests (8h)

**Update tracking:**
```bash
pmat roadmap update --ticket ENT-001 --status in-progress --actual-hours 28
pmat roadmap report --phase 1  # Sprint summary
```

---

## API Examples

**Declarative (Ludwig-style):**
```rust
use entrenar::*;

fn main() -> Result<()> {
    // Single command: YAML → trained model
    let model = train_from_yaml("config.yaml")?;
    save_model(&model, "output.gguf", &config)?;
    Ok(())
}
```

**Programmatic:**
```rust
use entrenar::*;

fn main() -> Result<()> {
    let mut model = load_model("llama-3-7b.gguf")?;
    model.add_lora_layers(64);
    
    let mut opt = Adam::new(1e-4, 0.9, 0.999);
    let mut trainer = Trainer::new(model, opt, CrossEntropyLoss::new());
    
    for epoch in 0..3 {
        let loss = trainer.train_epoch(&dataloader);
        println!("Epoch {}: loss={:.4}", epoch, loss);
    }
    
    trainer.model.merge_lora();
    trainer.model.quantize(QATConfig { bits: 4, symmetric: true, per_channel: true });
    save_model(&trainer.model, "llama-3-7b-finetuned-q4.gguf", &config)?;
    Ok(())
}
```

**Model Merging (Arcee-style):**
```rust
use entrenar::*;

fn main() -> Result<()> {
    let models = vec![
        load_model("llama-3-instruct.gguf")?,
        load_model("llama-3-code.gguf")?,
        load_model("llama-3-math.gguf")?,
    ];
    
    // TIES merge: 20% density
    let merged = merge_models(&models, MergeMethod::TIES { density: 0.2 });
    
    // Or SLERP for 2 models
    let blended = merge_models(&models[0..2], MergeMethod::SLERP { t: 0.5 });
    
    save_model(&merged, "llama-3-merged.gguf", &config)?;
    Ok(())
}
```

---

**Critical Path:** Trueno backward ops → Optimizer → LoRA → QAT → GGUF export.

**Blocker:** Trueno currently inference-only. Requires `ops::backward` module.

**Timeline:** 6 months (Phase 1-4), assuming Trueno backward ops added in parallel.

---

## NF4 QLoRA Training: cuBLAS Integration

### Throughput Comparison

| Backend | Throughput | Status |
|---------|-----------|--------|
| Fused NF4 kernel (trueno-gpu PTX) | 15.5 tok/s | Working (current) |
| cuBLAS GEMM (with NF4 dequant) | 298 tok/s | Parity verified, blocked by Blackwell JIT |

The 19x throughput improvement from cuBLAS is available once the Blackwell JIT bug (trueno#200) is resolved via dimension-independent kernels (trueno#203).

### cuBLAS Integration Requirements

**All backward kernels must be pre-warmed before any GPU work begins.** On Blackwell (sm_121), the NVIDIA JIT compiler crashes (`cuModuleLoadDataEx` returns `CUDA_ERROR_UNKNOWN`) when PTX compilation is attempted during active GPU work. This specifically affects backward kernels because they are compiled on-demand during the training loop.

**Pre-warming protocol**:
1. Enumerate all kernel shapes needed for the model architecture
2. Compile all PTX variants (forward AND backward) before the first training step
3. Cache compiled modules in a shape-indexed map
4. During training, only look up pre-compiled modules -- never trigger JIT

### Blackwell-Specific: `from_ptx_direct`

All kernel compilation on Blackwell must use `from_ptx_direct`, which:
- Loads pre-compiled cubin blobs instead of JIT-compiling PTX
- Avoids `cuModuleLoadDataEx` entirely (the crashing API)
- Works safely during active GPU work

Long-term fix: trueno#203 (dimension-independent kernels with offline cubin compilation via `build.rs` + `nvcc`).

### Provable Contracts

| Contract | File | Tests | Status |
|----------|------|-------|--------|
| NF4 cuBLAS parity | `nf4-cublas-parity-v2.yaml` | FALSIFY-NF4-001..005 | 4/5 PASS |
| Backward cache key | `backward-cache-key-v1.yaml` | FALSIFY-BCK-001..003 | 3/3 PASS |
| NF4 cuBLAS throughput | `nf4-cublas-throughput-v1.yaml` | FALSIFY-THR-001..003 | Blocked (trueno#200) |

### Key Invariants

1. **cuBLAS GEMM parity**: fused NF4 kernel and cuBLAS path must produce identical forward outputs (within FP16 tolerance of 1e-3)
2. **Gradient determinism**: Backward pass must produce identical gradients regardless of backend (fused vs cuBLAS)
3. **No JIT during training**: After pre-warming, zero `cuModuleLoadDataEx` calls may occur during the training loop
4. **Inference unaffected**: `apr run` uses cuBLAS/SIMD directly and is not impacted by the Blackwell JIT bug
