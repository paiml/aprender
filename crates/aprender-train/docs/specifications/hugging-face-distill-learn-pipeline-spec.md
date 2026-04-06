# Hugging Face Distillation & Learning Pipeline Specification

**Version:** 1.0.0
**Status:** Draft
**Author:** PAIML Engineering
**Date:** 2025-11-28

## Abstract

This specification defines a pipeline for downloading arbitrary models from Hugging Face Hub, performing knowledge
distillation from large teacher models to compact student models, and fine-tuning pretrained models on custom datasets.
The pipeline integrates with the PAIML stack (entrenar, aprender, trueno, realizar) to provide a pure-Rust ML training
ecosystem.

## 1. Overview

### 1.1 Goals

1. **Universal Model Download**: Fetch any model from HuggingFace Hub using `HF_TOKEN`
2. **Format Agnostic**: Support SafeTensors, GGUF, PyTorch (.bin), and ONNX formats
3. **Knowledge Distillation**: Train small student models from large teacher models
4. **Fine-Tuning**: Adapt pretrained models to domain-specific tasks
5. **Dataset Integration**: Download and preprocess HuggingFace datasets

### 1.2 Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Hugging Face Hub                                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │ CodeBERT    │  │ StarCoder   │  │ Llama-3     │  │ datasets/   │        │
│  │ .safetensors│  │ .gguf       │  │ .safetensors│  │ .parquet    │        │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘        │
└─────────┼────────────────┼────────────────┼────────────────┼────────────────┘
          │                │                │                │
          ▼                ▼                ▼                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     entrenar::hf_pipeline                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ HfModelFetcher                                                       │   │
│  │  - download_model(repo_id, revision) → ModelArtifact                │   │
│  │  - download_dataset(repo_id) → Dataset                              │   │
│  │  - list_files(repo_id) → Vec<FileInfo>                              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ ModelLoader (format detection + loading)                            │   │
│  │  - SafeTensorsLoader (via safetensors crate)                        │   │
│  │  - GGUFLoader (via realizar)                                        │   │
│  │  - ONNXLoader (optional, via tract)                                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ TeacherModel (inference-only wrapper)                               │   │
│  │  - forward(input) → logits                                          │   │
│  │  - get_hidden_states(input) → Vec<Tensor>                          │   │
│  │  - get_attention_weights(input) → Vec<Tensor>                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                                              │
│              ┌───────────────┴───────────────┐                             │
│              ▼                               ▼                              │
│  ┌─────────────────────────┐    ┌─────────────────────────┐               │
│  │ DistillationTrainer     │    │ FineTuneTrainer         │               │
│  │  - KL divergence loss   │    │  - LoRA/QLoRA adapters  │               │
│  │  - Hidden state match   │    │  - Full fine-tune       │               │
│  │  - Attention transfer   │    │  - Gradient checkpointing│               │
│  └─────────────────────────┘    └─────────────────────────┘               │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ StudentModel (trainable, exported to aprender/realizar formats)     │   │
│  │  - save_safetensors() → compatible with HF ecosystem               │   │
│  │  - save_gguf() → compatible with llama.cpp                         │   │
│  │  - save_apr() → native aprender format                             │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 2. HuggingFace Model Fetcher

### 2.1 API Design

```rust
use entrenar::hf_pipeline::{HfModelFetcher, ModelArtifact, FetchOptions};

// Initialize with token (from env or explicit)
let fetcher = HfModelFetcher::new()?;  // Uses HF_TOKEN env var
let fetcher = HfModelFetcher::with_token("hf_xxx...")?;

// Download any model
let artifact = fetcher.download_model(
    "microsoft/codebert-base",
    FetchOptions::default()
        .revision("main")
        .files(&["model.safetensors", "config.json", "tokenizer.json"])
)?;

// Download dataset
let dataset = fetcher.download_dataset(
    "bigcode/the-stack",
    DatasetOptions::default()
        .split("train")
        .streaming(true)  // For large datasets
)?;
```

### 2.2 Model Artifact Structure

```rust
pub struct ModelArtifact {
    /// Local path to downloaded files
    pub path: PathBuf,
    /// Model configuration (parsed from config.json)
    pub config: ModelConfig,
    /// Tokenizer (if available)
    pub tokenizer: Option<Tokenizer>,
    /// Weight format detected
    pub format: WeightFormat,
    /// Model architecture
    pub architecture: Architecture,
}

pub enum WeightFormat {
    SafeTensors,
    GGUF { quant_type: GGUFQuantType },
    PyTorchBin,
    ONNX,
}

pub enum Architecture {
    BERT { num_layers: usize, hidden_size: usize },
    GPT2 { num_layers: usize, hidden_size: usize },
    Llama { num_layers: usize, hidden_size: usize },
    T5 { encoder_layers: usize, decoder_layers: usize },
    Custom { config: serde_json::Value },
}
```

### 2.3 Authentication

```rust
// Priority order for token resolution:
// 1. Explicit token via with_token()
// 2. HF_TOKEN environment variable
// 3. ~/.huggingface/token file (huggingface-cli login)

impl HfModelFetcher {
    pub fn resolve_token() -> Option<String> {
        std::env::var("HF_TOKEN").ok()
            .or_else(|| {
                dirs::home_dir()
                    .map(|h| h.join(".huggingface/token"))
                    .and_then(|p| std::fs::read_to_string(p).ok())
                    .map(|s| s.trim().to_string())
            })
    }
}
```

## 3. Teacher Model Loading

### 3.1 Format-Agnostic Loading

```rust
pub trait TeacherModel: Send + Sync {
    /// Run forward pass, returning output logits
    fn forward(&self, input: &Tensor) -> Result<Tensor>;

    /// Get intermediate hidden states (for progressive distillation)
    fn hidden_states(&self, input: &Tensor) -> Result<Vec<Tensor>>;

    /// Get attention weights (for attention transfer)
    fn attention_weights(&self, input: &Tensor) -> Result<Vec<Tensor>>;

    /// Model configuration
    fn config(&self) -> &ModelConfig;
}

// Load from artifact
let teacher: Box<dyn TeacherModel> = match artifact.format {
    WeightFormat::SafeTensors => {
        SafeTensorsTeacher::load(&artifact.path)?
    }
    WeightFormat::GGUF { .. } => {
        // Use realizar for GGUF loading
        GGUFTeacher::load(&artifact.path)?
    }
    _ => return Err(Error::UnsupportedFormat),
};
```

### 3.2 SafeTensors Loading (Primary Path)

```rust
use safetensors::SafeTensors;

pub struct SafeTensorsTeacher {
    tensors: HashMap<String, Tensor>,
    config: ModelConfig,
}

impl SafeTensorsTeacher {
    pub fn load(path: &Path) -> Result<Self> {
        let data = std::fs::read(path.join("model.safetensors"))?;
        let tensors = SafeTensors::deserialize(&data)?;

        let config: ModelConfig = serde_json::from_reader(
            File::open(path.join("config.json"))?
        )?;

        Ok(Self { tensors, config })
    }
}
```

### 3.3 GGUF Loading (via realizar - Optional)

```rust
#[cfg(feature = "gguf")]
pub struct GGUFTeacher {
    model: realizar::Model,
    config: ModelConfig,
}

#[cfg(feature = "gguf")]
impl GGUFTeacher {
    pub fn load(path: &Path) -> Result<Self> {
        let model = realizar::Model::load(path)?;
        let config = model.metadata().into();
        Ok(Self { model, config })
    }
}
```

## 4. Knowledge Distillation

### 4.1 Distillation Loss Functions

Based on Hinton et al. (2015) [1], we implement temperature-scaled KL divergence:

```rust
/// Knowledge Distillation Loss
///
/// L_KD = α * KL(softmax(z_s/T) || softmax(z_t/T)) * T² + (1-α) * L_CE(y, z_s)
///
/// References:
/// [1] Hinton, G., Vinyals, O., & Dean, J. (2015). "Distilling the Knowledge
///     in a Neural Network." arXiv:1503.02531
pub struct DistillationLoss {
    /// Temperature for softening probability distributions
    pub temperature: f32,
    /// Weight for distillation loss vs hard label loss
    pub alpha: f32,
}

impl DistillationLoss {
    pub fn forward(
        &self,
        student_logits: &Tensor,
        teacher_logits: &Tensor,
        hard_labels: &[usize],
    ) -> f32 {
        let t = self.temperature;

        // Soft targets from teacher
        let teacher_soft = softmax(&(teacher_logits / t));
        let student_soft = log_softmax(&(student_logits / t));

        // KL divergence (scaled by T²)
        let kl_loss = kl_divergence(&student_soft, &teacher_soft) * t * t;

        // Hard label cross-entropy
        let ce_loss = cross_entropy(student_logits, hard_labels);

        self.alpha * kl_loss + (1.0 - self.alpha) * ce_loss
    }
}
```

### 4.2 Progressive Distillation

Based on Sun et al. (2019) [2], match intermediate representations:

```rust
/// Progressive Knowledge Transfer
///
/// Matches student hidden states to teacher hidden states at selected layers.
///
/// References:
/// [2] Sun, S., Cheng, Y., Gan, Z., & Liu, J. (2019). "Patient Knowledge
///     Distillation for BERT Model Compression." EMNLP 2019.
pub struct ProgressiveDistillation {
    /// Layer mapping: student_layer → teacher_layer
    pub layer_mapping: Vec<(usize, usize)>,
    /// Hidden state projection (if dimensions differ)
    pub projections: Vec<Linear>,
}

impl ProgressiveDistillation {
    pub fn hidden_state_loss(
        &self,
        student_hidden: &[Tensor],
        teacher_hidden: &[Tensor],
    ) -> f32 {
        let mut loss = 0.0;
        for (s_idx, t_idx) in &self.layer_mapping {
            let s_h = &student_hidden[*s_idx];
            let t_h = &teacher_hidden[*t_idx];

            // Project if needed
            let s_projected = self.projections[*s_idx].forward(s_h);

            // MSE loss for hidden state matching
            loss += mse_loss(&s_projected, t_h);
        }
        loss / self.layer_mapping.len() as f32
    }
}
```

### 4.3 Attention Transfer

Based on Zagoruyko & Komodakis (2017) [3]:

```rust
/// Attention Transfer Loss
///
/// Transfers attention maps from teacher to student.
///
/// References:
/// [3] Zagoruyko, S., & Komodakis, N. (2017). "Paying More Attention to
///     Attention: Improving the Performance of CNNs via Attention Transfer."
///     ICLR 2017.
pub fn attention_transfer_loss(
    student_attention: &[Tensor],  // [batch, heads, seq, seq]
    teacher_attention: &[Tensor],
) -> f32 {
    let mut loss = 0.0;
    for (s_attn, t_attn) in student_attention.iter().zip(teacher_attention) {
        // Normalize attention maps
        let s_norm = l2_normalize(s_attn);
        let t_norm = l2_normalize(t_attn);

        // Frobenius norm of difference
        loss += frobenius_norm(&(s_norm - t_norm)).powi(2);
    }
    loss / student_attention.len() as f32
}
```

## 5. Fine-Tuning

### 5.1 LoRA Fine-Tuning

Based on Hu et al. (2021) [4]:

```rust
/// Low-Rank Adaptation for efficient fine-tuning
///
/// W' = W + BA where B ∈ R^{d×r}, A ∈ R^{r×k}, r << min(d,k)
///
/// References:
/// [4] Hu, E. J., Shen, Y., Wallis, P., et al. (2021). "LoRA: Low-Rank
///     Adaptation of Large Language Models." arXiv:2106.09685
pub struct LoRAConfig {
    /// Rank of low-rank matrices
    pub rank: usize,
    /// Scaling factor (alpha/rank)
    pub alpha: f32,
    /// Dropout for LoRA layers
    pub dropout: f32,
    /// Target modules (e.g., ["q_proj", "v_proj"])
    pub target_modules: Vec<String>,
}

pub struct LoRAAdapter {
    pub lora_a: Tensor,  // [rank, in_features]
    pub lora_b: Tensor,  // [out_features, rank]
    pub scaling: f32,
}

impl LoRAAdapter {
    pub fn forward(&self, x: &Tensor, base_output: &Tensor) -> Tensor {
        // LoRA contribution: x @ A^T @ B^T * scaling
        let lora_out = x.matmul(&self.lora_a.t()).matmul(&self.lora_b.t());
        base_output + lora_out * self.scaling
    }
}
```

### 5.2 QLoRA Fine-Tuning

Based on Dettmers et al. (2023) [5]:

```rust
/// Quantized LoRA for memory-efficient fine-tuning
///
/// Base weights in 4-bit NormalFloat, adapters in FP16/BF16
///
/// References:
/// [5] Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023).
///     "QLoRA: Efficient Finetuning of Quantized LLMs." arXiv:2305.14314
pub struct QLoRAConfig {
    pub lora: LoRAConfig,
    /// Quantization config for base model
    pub quant_config: QuantConfig,
    /// Use double quantization
    pub double_quant: bool,
    /// Compute dtype for forward pass
    pub compute_dtype: DType,
}

pub struct QuantConfig {
    pub bits: u8,           // 4 for NF4
    pub group_size: usize,  // 64 typical
    pub quant_type: QuantType,
}

pub enum QuantType {
    NF4,    // NormalFloat 4-bit (optimal for normally distributed weights)
    FP4,    // Float 4-bit
    INT4,   // Integer 4-bit
}
```

## 6. Dataset Integration

### 6.1 HuggingFace Datasets API

```rust
/// Download and process HuggingFace datasets
pub struct HfDatasetFetcher {
    client: HfModelFetcher,
}

impl HfDatasetFetcher {
    /// Download dataset to local cache
    pub fn download(
        &self,
        repo_id: &str,      // e.g., "bigcode/the-stack"
        options: DatasetOptions,
    ) -> Result<Dataset> {
        // Download parquet files
        let files = self.client.list_files(&format!("datasets/{repo_id}"))?
            .into_iter()
            .filter(|f| f.name.ends_with(".parquet"))
            .collect();

        // Stream or download based on size
        if options.streaming {
            Ok(Dataset::Streaming(StreamingDataset::new(files)))
        } else {
            let local_files = self.client.download_files(&files)?;
            Ok(Dataset::InMemory(InMemoryDataset::from_parquet(&local_files)?))
        }
    }
}

pub struct DatasetOptions {
    pub split: String,          // "train", "validation", "test"
    pub streaming: bool,        // For large datasets
    pub num_proc: usize,        // Parallel processing
    pub cache_dir: PathBuf,     // Local cache
}
```

### 6.2 Data Collation for Distillation

```rust
/// Collate function for distillation training
pub struct DistillationCollator {
    pub tokenizer: Tokenizer,
    pub max_length: usize,
    pub teacher: Arc<dyn TeacherModel>,
}

impl DistillationCollator {
    pub fn collate(&self, examples: Vec<Example>) -> DistillationBatch {
        // Tokenize inputs
        let encodings = self.tokenizer.encode_batch(&examples)?;
        let input_ids = stack_tensors(&encodings.input_ids);
        let attention_mask = stack_tensors(&encodings.attention_mask);

        // Get teacher outputs (cached for efficiency)
        let teacher_logits = self.teacher.forward(&input_ids)?;
        let teacher_hidden = self.teacher.hidden_states(&input_ids)?;

        DistillationBatch {
            input_ids,
            attention_mask,
            labels: examples.iter().map(|e| e.label).collect(),
            teacher_logits,
            teacher_hidden,
        }
    }
}
```

## 7. Training Pipeline

### 7.1 Distillation Trainer

```rust
pub struct DistillationTrainer {
    pub student: StudentModel,
    pub teacher: Arc<dyn TeacherModel>,
    pub optimizer: Box<dyn Optimizer>,
    pub loss_fn: DistillationLoss,
    pub progressive: Option<ProgressiveDistillation>,
    pub config: TrainConfig,
    pub callbacks: CallbackManager,
}

impl DistillationTrainer {
    pub fn train(&mut self, dataloader: &DataLoader) -> TrainResult {
        for epoch in 0..self.config.epochs {
            for batch in dataloader {
                // Student forward
                let student_out = self.student.forward(&batch.input_ids)?;

                // Distillation loss
                let mut loss = self.loss_fn.forward(
                    &student_out.logits,
                    &batch.teacher_logits,
                    &batch.labels,
                );

                // Progressive distillation loss
                if let Some(prog) = &self.progressive {
                    let student_hidden = self.student.hidden_states(&batch.input_ids)?;
                    loss += prog.hidden_state_loss(&student_hidden, &batch.teacher_hidden);
                }

                // Backward + optimize
                self.optimizer.zero_grad();
                loss.backward();
                self.optimizer.step();

                // Callbacks (monitoring, checkpointing)
                self.callbacks.on_batch_end(&CallbackContext { loss, epoch, .. })?;
            }
        }

        TrainResult { final_loss: loss, .. }
    }
}
```

### 7.2 YAML Configuration

```yaml
# distill_config.yaml
teacher:
  repo_id: "microsoft/codebert-base"
  revision: "main"
  format: safetensors

student:
  architecture: transformer
  hidden_size: 256
  num_layers: 4
  num_heads: 4

distillation:
  temperature: 4.0
  alpha: 0.7
  progressive:
    enabled: true
    layer_mapping: [[0, 2], [1, 5], [2, 8], [3, 11]]
  attention_transfer:
    enabled: true
    weight: 0.1

dataset:
  repo_id: "bigcode/starcoderdata"
  split: "train"
  streaming: true
  max_length: 512

training:
  epochs: 10
  batch_size: 32
  learning_rate: 5e-5
  warmup_steps: 1000
  gradient_accumulation: 4

output:
  path: "distilled_model"
  formats: [safetensors, apr]
```

## 8. Use Cases

### 8.1 depyler-oracle: Distill CodeBERT → Error Classifier

```rust
// Download CodeBERT as teacher
let fetcher = HfModelFetcher::new()?;
let artifact = fetcher.download_model("microsoft/codebert-base", Default::default())?;
let teacher = SafeTensorsTeacher::load(&artifact.path)?;

// Create small student (12 features → 10 categories)
let student = ClassifierStudent::new(73, 10);  // 73 = depyler feature dim

// Train on error classification data
let trainer = DistillationTrainer::new(student, teacher)
    .with_temperature(4.0)
    .with_alpha(0.5);

let result = trainer.train(&error_dataset)?;

// Export to aprender format
student.save_apr("depyler_oracle_distilled.apr")?;
```

### 8.2 aprender-shell: Distill StarCoder → Command Suggester

```rust
// Download StarCoder (use GGUF for efficiency)
let artifact = fetcher.download_model(
    "TheBloke/starcoder-GGUF",
    FetchOptions::default().files(&["starcoder.Q4_K_M.gguf"])
)?;
let teacher = GGUFTeacher::load(&artifact.path)?;

// Small seq2seq student for command completion
let student = Seq2SeqStudent::new(
    vocab_size: 10000,
    hidden_size: 128,
    num_layers: 2,
);

// Fine-tune with LoRA for memory efficiency
let trainer = FineTuneTrainer::new(student)
    .with_lora(LoRAConfig { rank: 8, alpha: 16.0, .. });

trainer.train(&shell_commands_dataset)?;
```

### 8.3 Language-Specific Oracles

```rust
// Generic pipeline for any language oracle
async fn train_language_oracle(
    language: &str,         // "python", "rust", "go"
    teacher_repo: &str,     // HF model repo
    error_dataset: &str,    // HF dataset repo
) -> Result<Oracle> {
    let fetcher = HfModelFetcher::new()?;

    // Download teacher (e.g., CodeT5 for code understanding)
    let teacher = fetcher.download_and_load(teacher_repo)?;

    // Download error dataset
    let dataset = fetcher.download_dataset(error_dataset)?;

    // Distill to small oracle
    let student = OracleStudent::new(language);
    let trainer = DistillationTrainer::new(student, teacher);

    trainer.train(&dataset)?;

    Ok(student.into_oracle())
}
```

## 9. Integration with PAIML Stack

### 9.1 aprender Integration

```rust
// Use aprender's HfHubClient for authentication
use aprender::hf_hub::HfHubClient;

// Share token resolution
let token = HfHubClient::resolve_token();
let fetcher = HfModelFetcher::with_token(token)?;

// Export distilled models to aprender format
student.save_apr("model.apr")?;

// Push to HuggingFace Hub
let client = HfHubClient::with_token(token)?;
client.push_to_hub("paiml/distilled-oracle", &model_bytes, PushOptions::default())?;
```

### 9.2 realizar Integration (GGUF)

```rust
#[cfg(feature = "gguf")]
use realizar::{Model, Quantization};

// Load GGUF teacher model
let teacher = realizar::Model::load("starcoder.Q4_K_M.gguf")?;

// Export student to GGUF
let quant = Quantization::Q4_K_M;
student.export_gguf("distilled.gguf", quant)?;
```

### 9.3 trueno Integration (Compute)

```rust
use trueno::Tensor;

// Use trueno for SIMD-accelerated operations
impl DistillationLoss {
    pub fn forward_simd(&self, student: &Tensor, teacher: &Tensor) -> f32 {
        // trueno automatically selects SIMD vs scalar
        let diff = student.sub(teacher);
        diff.norm_squared().sqrt()
    }
}
```

## 10. Performance Considerations

### 10.1 Memory Optimization

- **Gradient Checkpointing**: Trade compute for memory [6]
- **Mixed Precision**: BF16/FP16 for forward, FP32 for backward
- **Teacher Caching**: Pre-compute teacher outputs for small datasets
- **Streaming**: Process large datasets without loading into memory

### 10.2 Compute Optimization

- **Data Parallelism**: Distribute batches across GPUs
- **Pipeline Parallelism**: Split teacher model across devices
- **Async Data Loading**: Prefetch batches during compute

## 11. Academic References

1. **Hinton, G., Vinyals, O., & Dean, J.** (2015). "Distilling the Knowledge in a Neural Network." *arXiv:1503.02531*.
   [[link]](https://arxiv.org/abs/1503.02531)
   - Foundation paper for knowledge distillation with temperature scaling

2. **Sun, S., Cheng, Y., Gan, Z., & Liu, J.** (2019). "Patient Knowledge Distillation for BERT Model Compression."
   *EMNLP 2019*. [[link]](https://arxiv.org/abs/1908.09355)
   - Progressive layer-wise distillation for transformers

3. **Zagoruyko, S., & Komodakis, N.** (2017). "Paying More Attention to Attention: Improving the Performance of CNNs via
   Attention Transfer." *ICLR 2017*. [[link]](https://arxiv.org/abs/1612.03928)
   - Attention map transfer between teacher and student

4. **Hu, E. J., Shen, Y., Wallis, P., et al.** (2021). "LoRA: Low-Rank Adaptation of Large Language Models."
   *arXiv:2106.09685*. [[link]](https://arxiv.org/abs/2106.09685)
   - Parameter-efficient fine-tuning with low-rank adapters

5. **Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L.** (2023). "QLoRA: Efficient Finetuning of Quantized
   LLMs." *arXiv:2305.14314*. [[link]](https://arxiv.org/abs/2305.14314)
   - 4-bit quantization with LoRA for memory efficiency

6. **Chen, T., Xu, B., Zhang, C., & Guestrin, C.** (2016). "Training Deep Nets with Sublinear Memory Cost."
   *arXiv:1604.06174*. [[link]](https://arxiv.org/abs/1604.06174)
   - Gradient checkpointing for memory-efficient training

7. **Sanh, V., Debut, L., Chaumond, J., & Wolf, T.** (2019). "DistilBERT, a distilled version of BERT."
   *arXiv:1910.01108*. [[link]](https://arxiv.org/abs/1910.01108)
   - Practical BERT distillation achieving 97% performance at 60% size

8. **Jiao, X., Yin, Y., Shang, L., et al.** (2020). "TinyBERT: Distilling BERT for Natural Language Understanding."
   *EMNLP 2020*. [[link]](https://arxiv.org/abs/1909.10351)
   - Two-stage distillation with embedding and prediction layer matching

9. **Wang, W., Wei, F., Dong, L., et al.** (2020). "MiniLM: Deep Self-Attention Distillation for Task-Agnostic
   Compression of Pre-Trained Transformers." *NeurIPS 2020*. [[link]](https://arxiv.org/abs/2002.10957)
   - Self-attention value relation distillation

10. **Feng, Z., Guo, D., Tang, D., et al.** (2020). "CodeBERT: A Pre-Trained Model for Programming and Natural
    Languages." *EMNLP 2020*. [[link]](https://arxiv.org/abs/2002.08155)
    - Pre-trained model for code understanding (potential teacher model)

## 12. Implementation Phases

### Phase 1: Core Infrastructure (40h)
- [ ] HfModelFetcher with authentication
- [ ] SafeTensors loading
- [ ] Basic TeacherModel trait

### Phase 2: Distillation (32h)
- [ ] DistillationLoss implementation
- [ ] ProgressiveDistillation
- [ ] AttentionTransfer

### Phase 3: Fine-Tuning (24h)
- [ ] LoRA adapter
- [ ] QLoRA with 4-bit quantization
- [ ] Gradient checkpointing

### Phase 4: Dataset Integration (16h)
- [ ] HfDatasetFetcher
- [ ] Streaming support
- [ ] DistillationCollator

### Phase 5: Export & Integration (16h)
- [ ] SafeTensors export
- [ ] GGUF export (via realizar)
- [ ] APR format export
- [ ] HuggingFace Hub push

**Total Estimate: 128 hours**

## 13. Dependencies

```toml
[dependencies]
# HuggingFace ecosystem
hf-hub = "0.3"           # HF Hub API client
safetensors = "0.4"      # SafeTensors format
tokenizers = "0.15"      # Fast tokenizers

# PAIML stack
trueno = "0.7"           # SIMD compute
aprender = "0.12"        # ML models
# realizar = "0.1"       # GGUF I/O (optional)

# Async runtime
tokio = { version = "1", features = ["full"] }

[features]
default = []
gguf = ["realizar"]      # Enable GGUF support
full = ["gguf"]
```
