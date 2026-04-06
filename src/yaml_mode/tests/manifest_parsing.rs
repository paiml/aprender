//! Manifest parsing tests - YAML to struct deserialization

use crate::yaml_mode::*;

#[test]
fn test_parse_minimal_manifest() {
    let yaml = r#"
entrenar: "1.0"
name: "test-experiment"
version: "1.0.0"
"#;
    let manifest: TrainingManifest = serde_yaml::from_str(yaml).expect("operation should succeed");
    assert_eq!(manifest.entrenar, "1.0");
    assert_eq!(manifest.name, "test-experiment");
    assert_eq!(manifest.version, "1.0.0");
}

#[test]
fn test_parse_with_description_and_seed() {
    let yaml = r#"
entrenar: "1.0"
name: "mnist-classifier"
version: "1.0.0"
description: "MNIST digit classification experiment"
seed: 42
"#;
    let manifest: TrainingManifest = serde_yaml::from_str(yaml).expect("operation should succeed");
    assert_eq!(manifest.description, Some("MNIST digit classification experiment".to_string()));
    assert_eq!(manifest.seed, Some(42));
}

#[test]
fn test_parse_data_config_with_source() {
    let yaml = r#"
entrenar: "1.0"
name: "data-test"
version: "1.0.0"

data:
  source: "./data/train.parquet"
  format: "parquet"
"#;
    let manifest: TrainingManifest = serde_yaml::from_str(yaml).expect("operation should succeed");
    let data = manifest.data.expect("operation should succeed");
    assert_eq!(data.source, Some("./data/train.parquet".to_string()));
    assert_eq!(data.format, Some("parquet".to_string()));
}

#[test]
fn test_parse_data_config_with_split() {
    let yaml = r#"
entrenar: "1.0"
name: "split-test"
version: "1.0.0"

data:
  source: "hf://ylecun/mnist"
  split:
    train: 0.8
    val: 0.1
    test: 0.1
    stratify: "label"
    seed: 42
"#;
    let manifest: TrainingManifest = serde_yaml::from_str(yaml).expect("operation should succeed");
    let data = manifest.data.expect("operation should succeed");
    let split = data.split.expect("operation should succeed");
    assert_eq!(split.train, 0.8);
    assert_eq!(split.val, Some(0.1));
    assert_eq!(split.test, Some(0.1));
    assert_eq!(split.stratify, Some("label".to_string()));
    assert_eq!(split.seed, Some(42));
}

#[test]
fn test_parse_data_config_with_loader() {
    let yaml = r#"
entrenar: "1.0"
name: "loader-test"
version: "1.0.0"

data:
  source: "./data/train.parquet"
  loader:
    batch_size: 32
    shuffle: true
    num_workers: 4
    pin_memory: true
    drop_last: false
"#;
    let manifest: TrainingManifest = serde_yaml::from_str(yaml).expect("operation should succeed");
    let loader = manifest.data.expect("load should succeed").loader.expect("load should succeed");
    assert_eq!(loader.batch_size, 32);
    assert!(loader.shuffle);
    assert_eq!(loader.num_workers, Some(4));
    assert_eq!(loader.pin_memory, Some(true));
    assert_eq!(loader.drop_last, Some(false));
}

#[test]
fn test_parse_model_config() {
    let yaml = r#"
entrenar: "1.0"
name: "model-test"
version: "1.0.0"

model:
  source: "hf://meta-llama/Llama-2-7b"
  device: "auto"
  dtype: "float16"
  freeze:
    - "embed_tokens"
    - "layers.0"
"#;
    let manifest: TrainingManifest = serde_yaml::from_str(yaml).expect("operation should succeed");
    let model = manifest.model.expect("operation should succeed");
    assert_eq!(model.source, "hf://meta-llama/Llama-2-7b");
    assert_eq!(model.device, Some("auto".to_string()));
    assert_eq!(model.dtype, Some("float16".to_string()));
    assert_eq!(model.freeze.expect("operation should succeed").len(), 2);
}

#[test]
fn test_parse_optimizer_config() {
    let yaml = r#"
entrenar: "1.0"
name: "optim-test"
version: "1.0.0"

optimizer:
  name: "adamw"
  lr: 0.001
  weight_decay: 0.01
  betas: [0.9, 0.999]
  eps: 1e-8
"#;
    let manifest: TrainingManifest = serde_yaml::from_str(yaml).expect("operation should succeed");
    let optim = manifest.optimizer.expect("operation should succeed");
    assert_eq!(optim.name, "adamw");
    assert_eq!(optim.lr, 0.001);
    assert_eq!(optim.weight_decay, Some(0.01));
    assert_eq!(optim.betas, Some(vec![0.9, 0.999]));
    assert_eq!(optim.eps, Some(1e-8));
}

#[test]
fn test_parse_scheduler_config() {
    let yaml = r#"
entrenar: "1.0"
name: "scheduler-test"
version: "1.0.0"

scheduler:
  name: "cosine_annealing"
  warmup:
    steps: 1000
    start_lr: 1e-7
  T_max: 10000
  eta_min: 1e-6
"#;
    let manifest: TrainingManifest = serde_yaml::from_str(yaml).expect("operation should succeed");
    let sched = manifest.scheduler.expect("operation should succeed");
    assert_eq!(sched.name, "cosine_annealing");
    let warmup = sched.warmup.expect("operation should succeed");
    assert_eq!(warmup.steps, Some(1000));
    assert_eq!(warmup.start_lr, Some(1e-7));
    assert_eq!(sched.t_max, Some(10000));
    assert_eq!(sched.eta_min, Some(1e-6));
}

#[test]
fn test_parse_training_config() {
    let yaml = r#"
entrenar: "1.0"
name: "training-test"
version: "1.0.0"

training:
  epochs: 10
  gradient:
    accumulation_steps: 4
    clip_norm: 1.0
  mixed_precision:
    enabled: true
    dtype: "float16"
"#;
    let manifest: TrainingManifest = serde_yaml::from_str(yaml).expect("operation should succeed");
    let training = manifest.training.expect("operation should succeed");
    assert_eq!(training.epochs, Some(10));
    let grad = training.gradient.expect("operation should succeed");
    assert_eq!(grad.accumulation_steps, Some(4));
    assert_eq!(grad.clip_norm, Some(1.0));
    let mp = training.mixed_precision.expect("operation should succeed");
    assert!(mp.enabled);
    assert_eq!(mp.dtype, Some("float16".to_string()));
}

#[test]
fn test_parse_lora_config() {
    let yaml = r#"
entrenar: "1.0"
name: "lora-test"
version: "1.0.0"

lora:
  enabled: true
  rank: 16
  alpha: 32
  dropout: 0.05
  target_modules:
    - q_proj
    - k_proj
    - v_proj
  bias: "none"
  quantize_base: true
  quantize_bits: 4
  quant_type: "nf4"
"#;
    let manifest: TrainingManifest = serde_yaml::from_str(yaml).expect("operation should succeed");
    let lora = manifest.lora.expect("operation should succeed");
    assert!(lora.enabled);
    assert_eq!(lora.rank, 16);
    assert_eq!(lora.alpha, 32.0);
    assert_eq!(lora.dropout, Some(0.05));
    assert_eq!(lora.target_modules.len(), 3);
    assert_eq!(lora.bias, Some("none".to_string()));
    assert!(lora.quantize_base.expect("operation should succeed"));
    assert_eq!(lora.quantize_bits, Some(4));
    assert_eq!(lora.quant_type, Some("nf4".to_string()));
}

#[test]
fn test_parse_quantize_config() {
    let yaml = r#"
entrenar: "1.0"
name: "quant-test"
version: "1.0.0"

quantize:
  enabled: true
  bits: 4
  scheme: "symmetric"
  granularity: "per_channel"
  group_size: 128
"#;
    let manifest: TrainingManifest = serde_yaml::from_str(yaml).expect("operation should succeed");
    let quant = manifest.quantize.expect("operation should succeed");
    assert!(quant.enabled);
    assert_eq!(quant.bits, 4);
    assert_eq!(quant.scheme, Some("symmetric".to_string()));
    assert_eq!(quant.granularity, Some("per_channel".to_string()));
    assert_eq!(quant.group_size, Some(128));
}

#[test]
fn test_parse_monitoring_config() {
    let yaml = r#"
entrenar: "1.0"
name: "monitor-test"
version: "1.0.0"

monitoring:
  terminal:
    enabled: true
    refresh_rate: 100
    metrics:
      - loss
      - accuracy
  tracking:
    enabled: true
    backend: "trueno-db"
    project: "my-project"
"#;
    let manifest: TrainingManifest = serde_yaml::from_str(yaml).expect("operation should succeed");
    let mon = manifest.monitoring.expect("operation should succeed");
    let term = mon.terminal.expect("operation should succeed");
    assert!(term.enabled);
    assert_eq!(term.refresh_rate, Some(100));
    assert_eq!(term.metrics.expect("operation should succeed").len(), 2);
    let track = mon.tracking.expect("operation should succeed");
    assert!(track.enabled);
    assert_eq!(track.backend, Some("trueno-db".to_string()));
}

#[test]
fn test_parse_output_config() {
    let yaml = r#"
entrenar: "1.0"
name: "output-test"
version: "1.0.0"

output:
  dir: "./experiments/{{ name }}/{{ timestamp }}"
  model:
    format: "safetensors"
    save_optimizer: true
"#;
    let manifest: TrainingManifest = serde_yaml::from_str(yaml).expect("operation should succeed");
    let output = manifest.output.expect("operation should succeed");
    assert_eq!(output.dir, "./experiments/{{ name }}/{{ timestamp }}");
    let model = output.model.expect("operation should succeed");
    assert_eq!(model.format, Some("safetensors".to_string()));
    assert!(model.save_optimizer.expect("save should succeed"));
}

#[test]
fn test_parse_complete_llama_finetune_example() {
    // Complete example from spec section 11
    let yaml = r#"
entrenar: "1.0"
name: "llama2-alpaca-finetune"
version: "1.0.0"
description: "Fine-tune LLaMA-2-7B on Alpaca dataset using QLoRA"
seed: 42

data:
  source: "hf://tatsu-lab/alpaca"
  split:
    train: 0.9
    val: 0.1
    seed: 42
  loader:
    batch_size: 4
    shuffle: true
    num_workers: 4

model:
  source: "hf://meta-llama/Llama-2-7b"
  device: "auto"
  dtype: "float16"

optimizer:
  name: "adamw"
  lr: 0.0002
  betas: [0.9, 0.999]
  weight_decay: 0.01

scheduler:
  name: "cosine_annealing"
  warmup:
    steps: 100
  T_max: 10000
  eta_min: 1e-6

training:
  epochs: 3
  gradient:
    accumulation_steps: 16
    clip_norm: 1.0
  mixed_precision:
    enabled: true
    dtype: "bfloat16"

lora:
  enabled: true
  rank: 64
  alpha: 128
  dropout: 0.05
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
  quantize_base: true
  quantize_bits: 4
  quant_type: "nf4"

monitoring:
  terminal:
    enabled: true
    refresh_rate: 100
  tracking:
    enabled: true
    backend: "trueno-db"
    project: "llama-finetune"

output:
  dir: "./experiments/llama2-alpaca/{{ timestamp }}"
  model:
    format: "safetensors"
"#;
    let manifest: TrainingManifest = serde_yaml::from_str(yaml).expect("operation should succeed");

    // Verify all sections parsed correctly
    assert_eq!(manifest.entrenar, "1.0");
    assert_eq!(manifest.name, "llama2-alpaca-finetune");
    assert_eq!(manifest.seed, Some(42));
    assert!(manifest.data.is_some());
    assert!(manifest.model.is_some());
    assert!(manifest.optimizer.is_some());
    assert!(manifest.scheduler.is_some());
    assert!(manifest.training.is_some());
    assert!(manifest.lora.is_some());
    assert!(manifest.monitoring.is_some());
    assert!(manifest.output.is_some());
}
