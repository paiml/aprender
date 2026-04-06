# HuggingFace Distillation Pipeline

This chapter covers entrenar's HuggingFace integration for knowledge distillation,
enabling you to download models from HuggingFace Hub and distill knowledge from
large teacher models into smaller, efficient student models.

## Overview

The HuggingFace pipeline provides:

- **HfModelFetcher**: Authenticated model downloading from HuggingFace Hub
- **SafeTensorsTeacher**: Memory-safe teacher model loading (no pickle)
- **DistillationLoss**: Temperature-scaled KL divergence (Hinton et al. 2015)
- **ProgressiveDistillation**: Hidden state matching (Sun et al. 2019)
- **AttentionTransfer**: Attention map distillation (Zagoruyko & Komodakis 2017)
- **FineTuneConfig**: LoRA/QLoRA fine-tuning configuration
- **Exporter**: Safe model export (SafeTensors, APR, GGUF)

## Quick Start

```rust
use entrenar::hf_pipeline::{
    HfModelFetcher, FetchOptions, SafeTensorsTeacher,
    DistillationTrainer, TrainerConfig, DistillationYamlConfig,
};

// 1. Fetch teacher model from HuggingFace
let fetcher = HfModelFetcher::new()?;
let artifact = fetcher.download_model(
    "microsoft/codebert-base",
    FetchOptions::default()
)?;

// 2. Load as teacher
let teacher = SafeTensorsTeacher::from_path(&artifact.path)?;

// 3. Configure distillation
let config = TrainerConfig::new("microsoft/codebert-base", "student-model")
    .temperature(4.0)
    .alpha(0.7)
    .epochs(10);

// 4. Create trainer and train
let mut trainer = DistillationTrainer::new(config, teacher);
// ... training loop
```

## Components

### HfModelFetcher

Downloads models from HuggingFace Hub with authentication support:

```rust
use entrenar::hf_pipeline::{HfModelFetcher, FetchOptions, WeightFormat};

// Create fetcher (reads HF_TOKEN from environment)
let fetcher = HfModelFetcher::new()?;

// Or with explicit token
let fetcher = HfModelFetcher::with_token("hf_xxxxx")?;

// Download with options
let options = FetchOptions::default()
    .format(WeightFormat::SafeTensors)  // Prefer SafeTensors (secure)
    .cache_dir("~/.cache/entrenar");

let artifact = fetcher.download_model("meta-llama/Llama-2-7b", options)?;

println!("Downloaded to: {:?}", artifact.path);
println!("Format: {:?}", artifact.format);
println!("Architecture: {:?}", artifact.architecture);
```

**Security**: By default, only SafeTensors format is allowed. PyTorch pickle
files are rejected due to arbitrary code execution risk.

### Memory Estimation

Estimate GPU/CPU memory requirements before loading:

```rust
use entrenar::hf_pipeline::{MemoryEstimate, Architecture};

// Estimate for Llama-2-7B
let estimate = MemoryEstimate::from_architecture(Architecture::Llama, 7_000_000_000);

println!("FP32: {:.1} GB", estimate.fp32_bytes as f64 / 1e9);
println!("FP16: {:.1} GB", estimate.fp16_bytes as f64 / 1e9);
println!("INT4: {:.1} GB", estimate.int4_bytes as f64 / 1e9);

// Check if model fits in available memory
if estimate.fits_in(16 * 1024 * 1024 * 1024) {  // 16 GB
    println!("Model fits in 16 GB VRAM");
}
```

### SafeTensorsTeacher

Load teacher models safely without pickle deserialization:

```rust
use entrenar::hf_pipeline::{SafeTensorsTeacher, TeacherModel};

let teacher = SafeTensorsTeacher::from_path("model.safetensors")?;

// Get model info
println!("Layers: {}", teacher.num_layers());
println!("Hidden dim: {}", teacher.hidden_dim());
println!("Memory: {:?}", teacher.memory_estimate());

// Run inference
let logits = teacher.forward(&input_ids)?;
let hidden_states = teacher.hidden_states(&input_ids)?;
let attention_weights = teacher.attention_weights(&input_ids)?;
```

### DistillationLoss

Temperature-scaled knowledge distillation with combined soft and hard targets:

```rust
use entrenar::hf_pipeline::DistillationLoss;
use ndarray::array;

// Create loss function
let loss_fn = DistillationLoss::new()
    .temperature(4.0)   // Soften probability distributions
    .alpha(0.7);        // 70% soft targets, 30% hard targets

// Compute loss
let teacher_logits = array![[10.0, 2.0, 1.0], [1.0, 12.0, 2.0]];
let student_logits = array![[7.0, 3.0, 2.0], [2.0, 8.0, 3.0]];
let labels = vec![0, 1];

let loss = loss_fn.forward(&student_logits, &teacher_logits, &labels);
println!("Distillation loss: {:.4}", loss);
```

**Temperature effect**: Higher temperature produces softer probability
distributions, revealing more "dark knowledge" from the teacher.

### ProgressiveDistillation

Match intermediate hidden states between teacher and student:

```rust
use entrenar::hf_pipeline::ProgressiveDistillation;

// Map student layers to teacher layers
// Student layer 0 -> Teacher layer 3
// Student layer 1 -> Teacher layer 7
// Student layer 2 -> Teacher layer 11
let layer_mapping = vec![(0, 3), (1, 7), (2, 11)];

let progressive = ProgressiveDistillation::new(layer_mapping)
    .weight(0.3);  // Weight for hidden state loss

let hidden_loss = progressive.compute_loss(
    &student_hiddens,
    &teacher_hiddens
);
```

### AttentionTransfer

Transfer attention patterns from teacher to student:

```rust
use entrenar::hf_pipeline::AttentionTransfer;

let attention_transfer = AttentionTransfer::new()
    .weight(0.1);  // Weight for attention loss

let attention_loss = attention_transfer.compute_loss(
    &student_attention,
    &teacher_attention
);
```

### DistillationTrainer

Orchestrates the complete distillation training loop:

```rust
use entrenar::hf_pipeline::{
    DistillationTrainer, TrainerConfig, SafeTensorsTeacher
};

// Configure trainer
let config = TrainerConfig::new("teacher/model", "student/model")
    .temperature(4.0)
    .alpha(0.7)
    .learning_rate(1e-4)
    .epochs(10)
    .batch_size(32)
    .with_progressive(vec![(0, 3), (1, 7), (2, 11)])
    .with_attention_transfer(0.1);

// Create trainer
let teacher = SafeTensorsTeacher::from_path("teacher.safetensors")?;
let mut trainer = DistillationTrainer::new(config, teacher);

// Training loop
for epoch in 0..10 {
    for batch in dataset.batches(32) {
        let loss = trainer.compute_loss(
            &student_logits,
            &batch.input_ids,
            &batch.labels,
            &student_hiddens,
            &student_attention
        );

        trainer.simulate_step(loss);
    }
    trainer.simulate_epoch();

    println!("Epoch {}: loss={:.4}", epoch, trainer.state().avg_loss());
}
```

### Fine-Tuning with LoRA/QLoRA

Configure parameter-efficient fine-tuning:

```rust
use entrenar::hf_pipeline::{FineTuneConfig, FineTuneMethod, MixedPrecision};
use entrenar::lora::LoRAConfig;

// LoRA configuration
let lora_config = LoRAConfig::builder()
    .rank(64)
    .alpha(16.0)
    .target_modules(vec!["q_proj", "k_proj", "v_proj", "o_proj"])
    .build();

// Full fine-tuning
let full_config = FineTuneConfig::builder("model-id")
    .method(FineTuneMethod::Full)
    .learning_rate(1e-5)
    .build();

// LoRA fine-tuning (10x less memory)
let lora_ft_config = FineTuneConfig::builder("model-id")
    .method(FineTuneMethod::LoRA(lora_config.clone()))
    .learning_rate(1e-4)
    .build();

// QLoRA fine-tuning (40x less memory)
let qlora_config = FineTuneConfig::builder("model-id")
    .method(FineTuneMethod::QLoRA {
        lora_config,
        bits: 4
    })
    .learning_rate(1e-4)
    .mixed_precision(MixedPrecision::BF16)
    .build();

// Estimate memory requirements
let requirement = qlora_config.estimate_memory(7_000_000_000);
println!("Training memory: {:.1} GB", requirement.total_bytes as f64 / 1e9);
println!("Memory savings: {:.0}%", requirement.savings_percent);
```

### Dataset Loading

Load and process datasets for distillation:

```rust
use entrenar::hf_pipeline::{
    Dataset, DatasetOptions, HfDatasetFetcher,
    DistillationCollator, TeacherCache, Split
};

// Fetch dataset from HuggingFace
let fetcher = HfDatasetFetcher::new()?;
let dataset = fetcher.fetch("wikitext", DatasetOptions::default()
    .split(Split::Train)
    .streaming(true))?;

// Or load from local parquet
let dataset = fetcher.load_parquet("train.parquet")?;

// Create collator for batching
let collator = DistillationCollator::new(0)  // pad_token_id = 0
    .max_length(512)
    .pad_left(false);

// Batch examples
let batch = collator.collate(&examples);

// Cache teacher outputs for efficiency
let mut cache = TeacherCache::new(1000);  // Cache 1000 examples
cache.cache_logits(0, teacher_logits);
cache.cache_hidden_states(0, teacher_hidden);

println!("Cache hit rate: {:.1}%", cache.hit_rate() * 100.0);
```

### YAML Configuration

Configure the entire pipeline via YAML:

```yaml
teacher:
  model_id: "meta-llama/Llama-2-7b"
  format: safetensors

student:
  model_id: "TinyLlama/TinyLlama-1.1B"

  lora:
    rank: 64
    alpha: 16
    target_modules: [q_proj, k_proj, v_proj, o_proj]
    dropout: 0.1

distillation:
  temperature: 4.0
  alpha: 0.7

  progressive:
    enabled: true
    layer_mapping:
      - [0, 3]
      - [1, 7]
      - [2, 11]
    weight: 0.3

  attention:
    enabled: true
    weight: 0.1

training:
  epochs: 10
  batch_size: 32
  learning_rate: 1e-4
  warmup_steps: 100
  gradient_accumulation: 4

dataset:
  name: "wikitext"
  path: "./data/wikitext"
  split: train
  max_length: 512

output:
  dir: "./distilled-model"
```

Load and use:

```rust
use entrenar::hf_pipeline::DistillationYamlConfig;

let config = DistillationYamlConfig::from_file("distill.yaml")?;
config.validate()?;

let trainer_config = config.to_trainer_config();
```

### Model Export

Export trained models in various formats:

```rust
use entrenar::hf_pipeline::{
    Exporter, ExportFormat, ModelWeights, ModelMetadata
};

// Create weights container
let mut weights = ModelWeights::new();
weights.add_tensor("model.embed", embedding_weights, vec![vocab_size, hidden_dim]);
weights.add_tensor("model.lm_head", lm_head_weights, vec![hidden_dim, vocab_size]);

// Add metadata
weights.set_metadata(ModelMetadata {
    name: "distilled-llama".to_string(),
    architecture: "llama".to_string(),
    hidden_size: 2048,
    num_layers: 12,
    vocab_size: 32000,
    ..Default::default()
});

// Export
let exporter = Exporter::new("./output")
    .format(ExportFormat::SafeTensors)
    .include_metadata(true);

let result = exporter.export(&weights)?;
println!("Exported to: {:?}", result.path);
println!("Size: {}", result.size_human());
```

**Supported formats**:
- **SafeTensors**: Secure, fast loading (recommended)
- **APR**: JSON-based, human-readable
- **GGUF**: llama.cpp compatible

## Complete Example

```rust
use entrenar::hf_pipeline::{
    HfModelFetcher, FetchOptions, SafeTensorsTeacher,
    DistillationTrainer, TrainerConfig, Dataset,
    DistillationCollator, Exporter, ExportFormat,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Fetch teacher model
    let fetcher = HfModelFetcher::new()?;
    let artifact = fetcher.download_model(
        "microsoft/codebert-base",
        FetchOptions::default()
    )?;

    // 2. Load teacher
    let teacher = SafeTensorsTeacher::from_path(&artifact.path)?;
    println!("Teacher loaded: {} layers, {} hidden dim",
             teacher.num_layers(), teacher.hidden_dim());

    // 3. Configure distillation
    let config = TrainerConfig::new("microsoft/codebert-base", "student")
        .temperature(4.0)
        .alpha(0.7)
        .epochs(10)
        .with_progressive(vec![(0, 2), (1, 5), (2, 8), (3, 11)]);

    let mut trainer = DistillationTrainer::new(config, teacher);

    // 4. Load dataset
    let dataset = Dataset::mock(1000, 128);  // Or load real data
    let collator = DistillationCollator::new(0).max_length(128);

    // 5. Training loop
    for epoch in 0..10 {
        for batch in dataset.batches(32) {
            // Forward pass through student (not shown)
            let loss = 0.5;  // Placeholder
            trainer.simulate_step(loss);
        }
        trainer.simulate_epoch();
        println!("Epoch {}: loss={:.4}", epoch, trainer.state().avg_loss());
    }

    // 6. Export distilled model
    let exporter = Exporter::new("./distilled-model")
        .format(ExportFormat::SafeTensors);

    // ... export weights

    println!("Distillation complete!");
    Ok(())
}
```

## Performance Considerations

- **Teacher caching**: Cache teacher outputs to avoid redundant forward passes
- **Gradient accumulation**: Increase effective batch size without more memory
- **QLoRA**: 4-bit quantization reduces memory by ~4x vs LoRA
- **Progressive distillation**: Match subset of layers to reduce computation
- **Streaming datasets**: Process large datasets without loading into memory

## Academic References

1. Hinton, G., Vinyals, O., & Dean, J. (2015). "Distilling the Knowledge in a Neural Network." *NIPS Deep Learning
   Workshop*.

2. Hu, E. J., et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models." *ICLR 2022*.

3. Dettmers, T., et al. (2023). "QLoRA: Efficient Finetuning of Quantized LLMs." *NeurIPS 2023*.

4. Sun, S., et al. (2019). "Patient Knowledge Distillation for BERT Model Compression." *EMNLP 2019*.

5. Zagoruyko, S., & Komodakis, N. (2017). "Paying More Attention to Attention: Improving the Performance of CNNs via
   Attention Transfer." *ICLR 2017*.
