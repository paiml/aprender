//! Example: HuggingFace Distillation Pipeline
//!
//! This example demonstrates the complete HuggingFace distillation pipeline:
//! 1. Model fetching with memory estimation
//! 2. Distillation training with progressive hidden state matching
//! 3. LoRA/QLoRA fine-tuning configuration
//! 4. Model export in safe formats

use entrenar::hf_pipeline::{
    Architecture,
    AttentionTransfer,
    // Dataset
    Dataset,
    DistillationCollator,
    // Distillation
    DistillationLoss,
    // Config
    DistillationYamlConfig,
    ExportFormat,
    // Export
    Exporter,
    // Fine-tuning
    FineTuneConfig,
    // Fetching
    HfModelFetcher,
    // Loading
    MemoryEstimate,
    ModelMetadata,
    ModelWeights,
    ProgressiveDistillation,
    TeacherCache,
    // Training
    TrainerConfig,
    WeightFormat,
};
use entrenar::lora::LoRAConfig;
use ndarray::array;

fn main() {
    println!("=== HuggingFace Distillation Pipeline ===\n");

    // Example 1: Memory Estimation
    println!("1. MEMORY ESTIMATION\n");
    memory_estimation_example();

    // Example 2: Model Fetching
    println!("\n2. MODEL FETCHING\n");
    model_fetching_example();

    // Example 3: Distillation Loss
    println!("\n3. DISTILLATION LOSS\n");
    distillation_loss_example();

    // Example 4: Progressive Distillation
    println!("\n4. PROGRESSIVE DISTILLATION\n");
    progressive_distillation_example();

    // Example 5: Training Configuration
    println!("\n5. TRAINER CONFIGURATION\n");
    trainer_config_example();

    // Example 6: Fine-tuning Methods
    println!("\n6. FINE-TUNING METHODS\n");
    fine_tuning_example();

    // Example 7: Dataset & Collation
    println!("\n7. DATASET & COLLATION\n");
    dataset_example();

    // Example 8: YAML Configuration
    println!("\n8. YAML CONFIGURATION\n");
    yaml_config_example();

    // Example 9: Model Export
    println!("\n9. MODEL EXPORT\n");
    export_example();

    println!("\n=== Examples Complete ===");
}

fn memory_estimation_example() {
    println!("Estimate memory requirements before loading models\n");

    // Llama-2-7B (7 billion params)
    let param_count = 7_000_000_000u64;
    let batch_size = 1;
    let seq_len = 2048;
    let hidden_size = 4096;

    let fp32 = MemoryEstimate::fp32(param_count, batch_size, seq_len, hidden_size);
    let fp16 = MemoryEstimate::fp16(param_count, batch_size, seq_len, hidden_size);
    let int4 = MemoryEstimate::int4(param_count, batch_size, seq_len, hidden_size);

    println!("Llama-2-7B (batch=1, seq=2048):");
    println!("  FP32: {:.1} GB total", fp32.total() as f64 / 1e9);
    println!("  FP16: {:.1} GB total", fp16.total() as f64 / 1e9);
    println!("  INT4: {:.1} GB total", int4.total() as f64 / 1e9);

    // Check if it fits in 16GB
    let fits_16gb = fp16.fits_in(16 * 1024 * 1024 * 1024);
    println!("  Fits in 16GB VRAM (FP16): {fits_16gb}");

    // CodeBERT (smaller model - 125M params)
    println!("\nCodeBERT-base (125M params):");
    let codebert_fp32 = MemoryEstimate::fp32(125_000_000, 8, 512, 768);
    let codebert_fp16 = MemoryEstimate::fp16(125_000_000, 8, 512, 768);
    println!("  FP32: {:.2} GB", codebert_fp32.total() as f64 / 1e9);
    println!("  FP16: {:.2} GB", codebert_fp16.total() as f64 / 1e9);
}

fn model_fetching_example() {
    println!("Fetch models from HuggingFace Hub (mock example)\n");

    // Create fetcher (would use HF_TOKEN from environment in practice)
    let fetcher = HfModelFetcher::new().expect("Failed to create fetcher");

    println!("Fetcher created:");
    println!("  Authenticated: {}", fetcher.is_authenticated());

    println!("\nFetch options:");
    println!("  Format: SafeTensors (secure, no pickle)");
    println!("  Revision: main");

    // Weight format detection
    println!("\nWeight format detection:");
    println!("  'model.safetensors' -> {:?}", WeightFormat::from_filename("model.safetensors"));
    println!("  'model.bin' -> {:?}", WeightFormat::from_filename("model.bin"));
    println!("  'model.gguf' -> {:?}", WeightFormat::from_filename("model.gguf"));

    // Safety check
    println!("\nFormat safety:");
    println!("  SafeTensors: safe = {}", WeightFormat::SafeTensors.is_safe());
    println!("  PyTorchBin: safe = {}", WeightFormat::PyTorchBin.is_safe());

    // Architecture examples
    println!("\nArchitecture examples:");
    let bert = Architecture::BERT { num_layers: 12, hidden_size: 768, num_attention_heads: 12 };
    let llama = Architecture::Llama {
        num_layers: 32,
        hidden_size: 4096,
        num_attention_heads: 32,
        intermediate_size: 11008,
    };
    println!("  BERT: {bert:?}");
    println!("  Llama: {llama:?}");
}

fn distillation_loss_example() {
    println!("Temperature-scaled knowledge distillation (Hinton et al. 2015)\n");

    // Teacher outputs (confident predictions)
    let teacher_logits = array![
        [10.0, 2.0, 1.0], // Confident on class 0
        [1.0, 12.0, 2.0], // Confident on class 1
        [2.0, 1.0, 11.0]  // Confident on class 2
    ];

    // Student outputs (less confident)
    let student_logits = array![[7.0, 3.0, 2.0], [2.0, 8.0, 3.0], [3.0, 2.0, 7.0]];

    let labels = vec![0, 1, 2];

    // Create loss function with temperature=4.0, alpha=0.7
    let loss_fn = DistillationLoss::new(4.0, 0.7);

    let loss = loss_fn.forward(&student_logits, &teacher_logits, &labels);

    println!("Configuration:");
    println!("  Temperature: 4.0 (softens probability distributions)");
    println!("  Alpha: 0.7 (70% soft targets, 30% hard targets)");
    println!("\nDistillation loss: {loss:.4}");

    // Effect of temperature
    println!("\nTemperature effect on loss:");
    for temp in [1.0, 2.0, 4.0, 8.0] {
        let loss_fn_temp = DistillationLoss::new(temp, 0.7);
        let loss_temp = loss_fn_temp.forward(&student_logits, &teacher_logits, &labels);
        println!("  T={temp:.0} -> loss={loss_temp:.4}");
    }

    // Effect of alpha
    println!("\nAlpha (soft target weight) effect:");
    for alpha in [0.0, 0.3, 0.5, 0.7, 1.0] {
        let loss_fn_alpha = DistillationLoss::new(4.0, alpha);
        let loss_alpha = loss_fn_alpha.forward(&student_logits, &teacher_logits, &labels);
        println!("  alpha={alpha:.1} -> loss={loss_alpha:.4}");
    }
}

fn progressive_distillation_example() {
    println!("Progressive layer-wise distillation (Sun et al. 2019)\n");

    // Layer mapping: student layer -> teacher layer
    let mapping = vec![
        (0, 3),  // Student layer 0 matches teacher layer 3
        (1, 7),  // Student layer 1 matches teacher layer 7
        (2, 11), // Student layer 2 matches teacher layer 11
    ];

    let progressive = ProgressiveDistillation::new(mapping.clone());

    println!("Layer mapping:");
    for (s, t) in &mapping {
        println!("  Student {s} -> Teacher {t}");
    }

    // Simulate hidden states (as slices of arrays)
    let student_hidden = vec![
        array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        array![[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]],
        array![[3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
    ];
    let teacher_hidden = vec![
        array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], // Layer 0 (not used)
        array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], // Layer 1 (not used)
        array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], // Layer 2 (not used)
        array![[1.1, 2.1, 3.1], [4.1, 5.1, 6.1]], // Layer 3 -> student 0
        array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], // Layer 4 (not used)
        array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], // Layer 5 (not used)
        array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], // Layer 6 (not used)
        array![[2.2, 3.2, 4.2], [5.2, 6.2, 7.2]], // Layer 7 -> student 1
        array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], // Layer 8 (not used)
        array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], // Layer 9 (not used)
        array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], // Layer 10 (not used)
        array![[3.5, 4.5, 5.5], [6.5, 7.5, 8.5]], // Layer 11 -> student 2
    ];

    let loss = progressive.hidden_state_loss(&student_hidden, &teacher_hidden);
    println!("\nHidden state matching loss: {loss:.4}");

    // Attention transfer
    println!("\nAttention Transfer (Zagoruyko & Komodakis 2017):");
    let attention = AttentionTransfer::new(0.1);
    println!("  Weight: 0.1");
    println!("  Transfers attention patterns from teacher to student");

    let student_attn = vec![array![[0.5, 0.3, 0.2], [0.2, 0.6, 0.2]]];
    let teacher_attn = vec![array![[0.6, 0.2, 0.2], [0.1, 0.7, 0.2]]];
    let attn_loss = attention.loss(&student_attn, &teacher_attn);
    println!("  Attention loss: {attn_loss:.4}");
}

fn trainer_config_example() {
    println!("Configure the distillation trainer\n");

    // Create configuration with builder pattern
    let config = TrainerConfig::new("meta-llama/Llama-2-7b", "TinyLlama/TinyLlama-1.1B")
        .temperature(4.0)
        .alpha(0.7)
        .epochs(10)
        .with_progressive(vec![(0, 3), (1, 7), (2, 11)])
        .with_attention_transfer(0.1);

    println!("TrainerConfig:");
    println!("  Teacher: {}", config.teacher_model);
    println!("  Student: {}", config.student_model);
    println!("  Temperature: {}", config.distillation_loss.temperature);
    println!("  Alpha: {}", config.distillation_loss.alpha);
    println!("  Epochs: {}", config.epochs);
    println!("  Progressive: {}", config.progressive.is_some());
    println!("  Attention transfer: {}", config.attention_transfer.is_some());

    println!("\nNote: DistillationTrainer requires a TeacherModel implementation.");
    println!("Use SafeTensorsTeacher::load(path) to load a real model.");
}

fn fine_tuning_example() {
    println!("Compare fine-tuning methods: Full vs LoRA vs QLoRA\n");

    let params = 7_000_000_000u64; // 7B parameters

    // Full fine-tuning
    let full_config = FineTuneConfig::new("model").full_fine_tune();
    let full_mem = full_config.estimate_memory(params);
    println!("Full Fine-tuning:");
    println!("  Trainable params: 100%");
    println!("  Memory: {:.1} GB", full_mem.total() as f64 / 1e9);

    // LoRA fine-tuning
    let lora_config = LoRAConfig::new(64, 16.0).target_modules(&["q_proj", "v_proj"]);

    let lora_ft = FineTuneConfig::new("model").with_lora(lora_config.clone()).learning_rate(1e-4);

    let lora_mem = lora_ft.estimate_memory(params);
    println!("\nLoRA (rank=64):");
    println!("  Trainable params: ~0.1%");
    println!("  Memory: {:.1} GB", lora_mem.total() as f64 / 1e9);
    println!("  Savings: {:.0}%", lora_mem.savings_vs_full(params) * 100.0);

    // QLoRA fine-tuning
    let qlora_ft = FineTuneConfig::new("model").with_qlora(lora_config, 4).learning_rate(1e-4);

    let qlora_mem = qlora_ft.estimate_memory(params);
    println!("\nQLoRA (4-bit + LoRA):");
    println!("  Base model: 4-bit quantized (frozen)");
    println!("  Adapters: FP16/BF16 (trainable)");
    println!("  Memory: {:.1} GB", qlora_mem.total() as f64 / 1e9);
    println!("  Savings: {:.0}%", qlora_mem.savings_vs_full(params) * 100.0);

    // Memory comparison
    println!("\nMemory Comparison (7B model):");
    println!("  Full:  {:.1} GB (baseline)", full_mem.total() as f64 / 1e9);
    println!(
        "  LoRA:  {:.1} GB ({:.1}x reduction)",
        lora_mem.total() as f64 / 1e9,
        full_mem.total() as f64 / lora_mem.total() as f64
    );
    println!(
        "  QLoRA: {:.1} GB ({:.1}x reduction)",
        qlora_mem.total() as f64 / 1e9,
        full_mem.total() as f64 / qlora_mem.total() as f64
    );
}

fn dataset_example() {
    println!("Load and process datasets for distillation\n");

    // Create mock dataset
    let dataset = Dataset::mock(100, 64); // 100 examples, max 64 tokens
    println!("Dataset:");
    println!("  Examples: {}", dataset.len());

    // Create collator for batching
    let collator = DistillationCollator::new(0) // pad_token_id = 0
        .max_length(128)
        .pad_left(false);

    println!("\nCollator:");
    println!("  Pad token ID: 0");
    println!("  Max length: 128");
    println!("  Padding: right (pad_left=false)");

    // Create batch
    let examples = dataset.examples();
    let batch = collator.collate(&examples[..8]);
    println!("\nBatch:");
    println!("  Size: {}", batch.batch_size());
    println!("  Max sequence length: {}", batch.max_seq_len());

    // Teacher caching
    let mut cache = TeacherCache::new();
    println!("\nTeacher Cache:");

    // Simulate caching
    for i in 0..50 {
        cache.cache_logits(i, array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    }
    println!("  Cached: 50 examples");
    println!("  Get cached: {:?}", cache.get_logits(0).is_some());
}

fn yaml_config_example() {
    println!("Configure distillation via YAML\n");

    let yaml = r#"
teacher:
  model_id: "meta-llama/Llama-2-7b"
  format: safetensors

student:
  model_id: "TinyLlama/TinyLlama-1.1B"
  lora:
    rank: 64
    alpha: 16
    target_modules: [q_proj, k_proj, v_proj, o_proj]

distillation:
  temperature: 4.0
  alpha: 0.7
  progressive:
    enabled: true
    layer_mapping: [[0, 3], [1, 7], [2, 11]]
    weight: 0.3

training:
  epochs: 10
  batch_size: 32
  learning_rate: 0.0001

dataset:
  name: wikitext
  path: ./data/wikitext
  split: train
  max_length: 512

output:
  dir: ./distilled-model
"#;

    println!("Example YAML configuration:");
    println!("{yaml}");

    // Parse and validate
    let config: DistillationYamlConfig = serde_yaml::from_str(yaml).expect("Failed to parse YAML");

    if let Err(e) = config.validate() {
        println!("Validation error: {e}");
    } else {
        println!("Configuration valid!");
    }

    // Convert to trainer config
    match config.to_trainer_config() {
        Ok(trainer_config) => {
            println!("\nParsed configuration:");
            println!("  Teacher: {}", trainer_config.teacher_model);
            println!("  Student: {}", trainer_config.student_model);
            println!("  Temperature: {}", trainer_config.distillation_loss.temperature);
        }
        Err(e) => println!("Error: {e}"),
    }
}

fn export_example() {
    println!("Export trained models in safe formats\n");

    // Create model weights
    let mut weights = ModelWeights::new();
    weights.add_tensor("model.embed_tokens", vec![0.1; 1000], vec![100, 10]);
    weights.add_tensor("model.lm_head", vec![0.2; 1000], vec![10, 100]);

    println!("Model weights:");
    println!("  Tensors: {:?}", weights.tensor_names());
    println!("  Total params: {}", weights.param_count());

    // Add metadata
    let weights = weights.with_metadata(ModelMetadata {
        model_name: Some("distilled-llama".to_string()),
        architecture: Some("llama".to_string()),
        hidden_size: Some(2048),
        num_layers: Some(12),
        num_params: 2000,
        ..Default::default()
    });

    // Export formats
    println!("\nExport formats:");
    for format in [ExportFormat::SafeTensors, ExportFormat::APR, ExportFormat::GGUF] {
        println!("  {format:?}:");
        println!("    Extension: {}", format.extension());
        println!("    Safe: {}", format.is_safe());
    }

    // Create exporter
    let exporter = Exporter::new()
        .output_dir("./output")
        .default_format(ExportFormat::SafeTensors)
        .include_metadata(true);

    println!("\nExporter configured:");
    println!("  Output dir: ./output");
    println!("  Format: SafeTensors");
    println!("  Include metadata: true");

    // In practice, would call:
    // let result = exporter.export_safetensors(&weights, "model")?;
    println!("\n(Export would create: ./output/model.safetensors)");

    // Suppress unused warning
    let _ = exporter;
    let _ = weights;
}
