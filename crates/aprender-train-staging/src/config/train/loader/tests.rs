use super::*;

#[test]
fn test_create_demo_lm_batches() {
    let batches = create_demo_lm_batches(4, 32).expect("operation should succeed");
    assert_eq!(batches.len(), 4);
    // Each batch should have valid data
    for batch in &batches {
        assert!(batch.has_tokens());
    }
}

#[test]
fn test_create_demo_lm_batches_small() {
    let batches = create_demo_lm_batches(1, 16).expect("operation should succeed");
    assert_eq!(batches.len(), 4);
}

#[test]
fn test_create_demo_lm_batches_large_seq_len() {
    let batches = create_demo_lm_batches(2, 512).expect("operation should succeed");
    assert_eq!(batches.len(), 4);
}

#[test]
fn test_create_lm_batches_from_sequences() {
    let sequences =
        vec![vec![1u32, 2, 3, 4, 5], vec![6u32, 7, 8, 9, 10], vec![11u32, 12, 13, 14, 15]];
    let batches =
        create_lm_batches_from_sequences(&sequences, 2, 32).expect("operation should succeed");
    assert_eq!(batches.len(), 2); // 3 sequences with batch_size 2 = 2 batches
}

#[test]
fn test_create_lm_batches_from_sequences_single_batch() {
    let sequences = vec![vec![1u32, 2, 3], vec![4u32, 5, 6]];
    let batches =
        create_lm_batches_from_sequences(&sequences, 4, 32).expect("operation should succeed");
    assert_eq!(batches.len(), 1);
}

#[test]
fn test_create_lm_batches_from_sequences_empty() {
    let sequences: Vec<Vec<u32>> = vec![];
    let batches =
        create_lm_batches_from_sequences(&sequences, 4, 32).expect("operation should succeed");
    assert!(batches.is_empty());
}

#[test]
fn test_load_pretokenized_json_valid() {
    let examples: Vec<serde_json::Value> = vec![
        serde_json::json!({"input_ids": [1, 2, 3, 4, 5]}),
        serde_json::json!({"input_ids": [6, 7, 8, 9, 10]}),
    ];
    let batches = load_pretokenized_json(&examples, 2, 32).expect("load should succeed");
    assert!(!batches.is_empty());
}

#[test]
fn test_load_pretokenized_json_empty() {
    let examples: Vec<serde_json::Value> = vec![];
    // Falls back to demo batches
    let batches = load_pretokenized_json(&examples, 2, 32).expect("load should succeed");
    assert!(!batches.is_empty()); // Demo batches
}

#[test]
fn test_load_pretokenized_json_no_input_ids() {
    let examples: Vec<serde_json::Value> =
        vec![serde_json::json!({"text": "hello"}), serde_json::json!({"text": "world"})];
    // Falls back to demo batches
    let batches = load_pretokenized_json(&examples, 2, 32).expect("load should succeed");
    assert!(!batches.is_empty());
}

#[test]
fn test_load_lm_batches_from_json_pretokenized() {
    let json = r#"{"examples": [{"input_ids": [1, 2, 3]}, {"input_ids": [4, 5, 6]}]}"#;
    let batches = load_lm_batches_from_json(json, None, 2, 32, None).expect("load should succeed");
    assert!(!batches.is_empty());
}

#[test]
fn test_load_lm_batches_from_json_array_pretokenized() {
    let json = r#"[{"input_ids": [1, 2, 3]}, {"input_ids": [4, 5, 6]}]"#;
    let batches = load_lm_batches_from_json(json, None, 2, 32, None).expect("load should succeed");
    assert!(!batches.is_empty());
}

#[test]
fn test_load_lm_batches_from_json_invalid() {
    let json = "not valid json";
    // Falls back to demo batches
    let batches = load_lm_batches_from_json(json, None, 2, 32, None).expect("load should succeed");
    assert!(!batches.is_empty());
}

#[test]
fn test_load_lm_batches_from_json_empty_examples() {
    let json = r#"{"examples": []}"#;
    // Falls back to demo batches
    let batches = load_lm_batches_from_json(json, None, 2, 32, None).expect("load should succeed");
    assert!(!batches.is_empty());
}

#[test]
fn test_build_transformer_config_defaults() {
    use crate::config::schema::{DataConfig, ModelRef, OptimSpec, TrainingParams};
    use std::collections::HashMap;
    use std::path::PathBuf;

    let spec = TrainSpec {
        model: ModelRef {
            path: PathBuf::from("/nonexistent/model"),
            config: None,
            ..Default::default()
        },
        data: DataConfig {
            train: PathBuf::from("/nonexistent/data.json"),
            batch_size: 4,
            ..Default::default()
        },
        optimizer: OptimSpec { name: "adam".to_string(), lr: 1e-4, params: HashMap::new() },
        training: TrainingParams {
            epochs: 1,
            output_dir: PathBuf::from("/tmp"),
            ..Default::default()
        },
        lora: None,
        quantize: None,
        merge: None,
        publish: None,
    };

    let config = build_transformer_config_from_spec(&spec).expect("config should be valid");
    // Default Qwen2.5-like dimensions
    assert_eq!(config.hidden_size, 896);
    assert_eq!(config.num_attention_heads, 14);
    assert_eq!(config.num_kv_heads, 2);
    assert_eq!(config.intermediate_size, 4864);
}

#[test]
fn test_build_transformer_config_with_architecture_overrides() {
    use crate::config::schema::{
        ArchitectureOverrides, DataConfig, ModelRef, OptimSpec, TrainingParams,
    };
    use std::collections::HashMap;
    use std::path::PathBuf;

    let spec = TrainSpec {
        model: ModelRef {
            path: PathBuf::from("/nonexistent/model"),
            config: None,
            architecture: Some(ArchitectureOverrides {
                hidden_size: Some(1024),
                num_hidden_layers: Some(16),
                num_attention_heads: Some(16),
                num_kv_heads: Some(4),
                intermediate_size: Some(4096),
                vocab_size: Some(50000),
                max_position_embeddings: None,
                rms_norm_eps: Some(1e-5),
                rope_theta: Some(500_000.0),
                use_bias: Some(true),
                head_dim: None,
            }),
            ..Default::default()
        },
        data: DataConfig {
            train: PathBuf::from("/nonexistent/data.json"),
            batch_size: 4,
            ..Default::default()
        },
        optimizer: OptimSpec { name: "adam".to_string(), lr: 1e-4, params: HashMap::new() },
        training: TrainingParams {
            epochs: 1,
            output_dir: PathBuf::from("/tmp"),
            ..Default::default()
        },
        lora: None,
        quantize: None,
        merge: None,
        publish: None,
    };

    let config = build_transformer_config_from_spec(&spec).expect("config should be valid");
    // Overridden fields
    assert_eq!(config.hidden_size, 1024);
    assert_eq!(config.num_hidden_layers, 16);
    assert_eq!(config.num_attention_heads, 16);
    assert_eq!(config.num_kv_heads, 4);
    assert_eq!(config.intermediate_size, 4096);
    assert_eq!(config.vocab_size, 50000);
    assert_eq!(config.rms_norm_eps, 1e-5);
    assert_eq!(config.rope_theta, 500_000.0);
    assert!(config.use_bias);
    // Non-overridden field uses generic default (not Qwen2 demo config)
    assert_eq!(config.max_position_embeddings, 2048);
}

#[test]
fn test_build_transformer_config_partial_overrides() {
    use crate::config::schema::{
        ArchitectureOverrides, DataConfig, ModelRef, OptimSpec, TrainingParams,
    };
    use std::collections::HashMap;
    use std::path::PathBuf;

    let spec = TrainSpec {
        model: ModelRef {
            path: PathBuf::from("/nonexistent/model"),
            config: None,
            architecture: Some(ArchitectureOverrides {
                hidden_size: Some(768),
                vocab_size: Some(32000),
                ..Default::default()
            }),
            ..Default::default()
        },
        data: DataConfig {
            train: PathBuf::from("/nonexistent/data.json"),
            batch_size: 4,
            ..Default::default()
        },
        optimizer: OptimSpec { name: "adam".to_string(), lr: 1e-4, params: HashMap::new() },
        training: TrainingParams {
            epochs: 1,
            output_dir: PathBuf::from("/tmp"),
            ..Default::default()
        },
        lora: None,
        quantize: None,
        merge: None,
        publish: None,
    };

    let config = build_transformer_config_from_spec(&spec).expect("config should be valid");
    // Only these two should be overridden
    assert_eq!(config.hidden_size, 768);
    assert_eq!(config.vocab_size, 32000);
    // Rest keeps demo defaults
    assert_eq!(config.num_attention_heads, QWEN_NUM_ATTENTION_HEADS);
    assert_eq!(config.num_kv_heads, QWEN_NUM_KV_HEADS);
    assert_eq!(config.intermediate_size, QWEN_INTERMEDIATE_SIZE);
}

#[test]
fn test_load_lm_batches_from_parquet_fallback() {
    use std::path::Path;
    let tokenizer = HfTokenizer::qwen2();
    // Non-existent path returns error (ALB-007: real parquet loading, not demo fallback)
    let result =
        load_lm_batches_from_parquet(Path::new("/nonexistent.parquet"), &tokenizer, 4, 32, "text");
    assert!(result.is_err());
}

#[test]
fn test_tokenize_texts_to_batches_empty() {
    let tokenizer = HfTokenizer::qwen2();
    let texts: Vec<String> = vec![];
    // Falls back to demo batches
    let batches =
        tokenize_texts_to_batches(&texts, &tokenizer, 4, 32).expect("operation should succeed");
    assert!(!batches.is_empty());
}

#[test]
fn test_tokenize_texts_to_batches_valid() {
    let tokenizer = HfTokenizer::qwen2();
    let texts = vec!["Hello world".to_string(), "This is a test".to_string()];
    let batches =
        tokenize_texts_to_batches(&texts, &tokenizer, 2, 64).expect("operation should succeed");
    assert!(!batches.is_empty());
}

#[test]
fn test_tokenize_texts_to_batches_single_token() {
    let tokenizer = HfTokenizer::qwen2();
    // Very short text that results in single token gets filtered
    let texts = vec!["a".to_string()];
    let batches =
        tokenize_texts_to_batches(&texts, &tokenizer, 2, 64).expect("operation should succeed");
    // May fall back to demo batches if single token is filtered
    assert!(!batches.is_empty());
}

// =========================================================================
// Format auto-detection tests
// =========================================================================

#[test]
fn test_is_manifest_format_detects_entrenar_key() {
    assert!(is_manifest_format("entrenar: \"1.0\"\nname: test\n"));
    assert!(is_manifest_format("# comment\nentrenar: \"1.0\"\n"));
    assert!(is_manifest_format("entrenar : \"1.0\"\n"));
}

#[test]
fn test_is_manifest_format_rejects_legacy() {
    let legacy = r"
model:
  path: model.gguf
data:
  train: train.parquet
  batch_size: 8
optimizer:
  name: adam
  lr: 0.001
";
    assert!(!is_manifest_format(legacy));
}

#[test]
fn test_load_config_manifest_format() {
    use std::io::Write;
    let manifest_yaml = r#"
entrenar: "1.0"
name: "test-bridge"
version: "1.0.0"

model:
  source: "./models/test.safetensors"

data:
  source: "./data/train.parquet"
  loader:
    batch_size: 16
    shuffle: true

optimizer:
  name: adam
  lr: 0.0001

training:
  epochs: 5
"#;
    let dir = std::env::temp_dir().join("entrenar_bridge_test");
    std::fs::create_dir_all(&dir).expect("operation should succeed");
    let path = dir.join("manifest_test.yaml");
    let mut f = std::fs::File::create(&path).expect("file write should succeed");
    f.write_all(manifest_yaml.as_bytes()).expect("file write should succeed");

    let spec = load_config(&path).expect("load should succeed");
    assert_eq!(spec.model.path, std::path::PathBuf::from("./models/test.safetensors"));
    assert_eq!(spec.data.train, std::path::PathBuf::from("./data/train.parquet"));
    assert_eq!(spec.data.batch_size, 16);
    assert_eq!(spec.optimizer.name, "adam");
    assert!((spec.optimizer.lr - 0.0001).abs() < 1e-6);
    assert_eq!(spec.training.epochs, 5);

    std::fs::remove_file(&path).ok();
}

#[test]
fn test_load_config_legacy_format() {
    use std::io::Write;
    let legacy_yaml = r"
model:
  path: model.gguf
  layers: []

data:
  train: train.parquet
  batch_size: 8

optimizer:
  name: adam
  lr: 0.001
";
    let dir = std::env::temp_dir().join("entrenar_bridge_test");
    std::fs::create_dir_all(&dir).expect("operation should succeed");
    let path = dir.join("legacy_test.yaml");
    let mut f = std::fs::File::create(&path).expect("file write should succeed");
    f.write_all(legacy_yaml.as_bytes()).expect("file write should succeed");

    let spec = load_config(&path).expect("load should succeed");
    assert_eq!(spec.optimizer.name, "adam");
    assert_eq!(spec.data.batch_size, 8);

    std::fs::remove_file(&path).ok();
}

// =========================================================================
// FALSIFY tests — contract violation sweep (C-10/C-11, R-04)
// =========================================================================

#[test]
fn test_falsify_c10_c11_config_with_all_required_fields_succeeds() {
    // C-10/C-11: config.json with all 5 required fields must parse successfully.
    use std::io::Write;
    let config_json = r#"{
            "hidden_size": 768,
            "num_attention_heads": 12,
            "num_hidden_layers": 6,
            "vocab_size": 30000,
            "intermediate_size": 3072,
            "max_position_embeddings": 512,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0,
            "attention_bias": true
        }"#;
    let dir = std::env::temp_dir().join("entrenar_falsify_c10");
    std::fs::create_dir_all(&dir).expect("operation should succeed");
    let config_path = dir.join("config.json");
    let mut f = std::fs::File::create(&config_path).expect("file write should succeed");
    f.write_all(config_json.as_bytes()).expect("file write should succeed");

    let spec = TrainSpec {
        model: crate::config::schema::ModelRef {
            path: PathBuf::from("/nonexistent/model"),
            config: Some(config_path.to_string_lossy().into_owned()),
            ..Default::default()
        },
        data: crate::config::schema::DataConfig {
            train: PathBuf::from("/nonexistent/data.json"),
            batch_size: 4,
            ..Default::default()
        },
        optimizer: crate::config::schema::OptimSpec {
            name: "adam".to_string(),
            lr: 1e-4,
            params: std::collections::HashMap::new(),
        },
        training: crate::config::schema::TrainingParams {
            epochs: 1,
            output_dir: PathBuf::from("/tmp"),
            ..Default::default()
        },
        lora: None,
        quantize: None,
        merge: None,
        publish: None,
    };

    let config = build_transformer_config_from_spec(&spec).expect("config should be valid");
    assert_eq!(config.hidden_size, 768);
    assert_eq!(config.num_attention_heads, 12);
    assert_eq!(config.num_hidden_layers, 6);
    assert_eq!(config.vocab_size, 30000);
    assert_eq!(config.intermediate_size, 3072);
    assert_eq!(config.max_position_embeddings, 512);
    assert!(config.use_bias);

    std::fs::remove_file(&config_path).ok();
}

#[test]
fn test_falsify_c11_missing_hidden_size_errors() {
    // C-11: config.json missing hidden_size must return Err, not silently default.
    use std::io::Write;
    let config_json = r#"{
            "num_attention_heads": 12,
            "num_hidden_layers": 6,
            "vocab_size": 30000,
            "intermediate_size": 3072
        }"#;
    let dir = std::env::temp_dir().join("entrenar_falsify_c11");
    std::fs::create_dir_all(&dir).expect("operation should succeed");
    let config_path = dir.join("config_no_hidden.json");
    let mut f = std::fs::File::create(&config_path).expect("file write should succeed");
    f.write_all(config_json.as_bytes()).expect("file write should succeed");

    let spec = TrainSpec {
        model: crate::config::schema::ModelRef {
            path: PathBuf::from("/nonexistent/model"),
            config: Some(config_path.to_string_lossy().into_owned()),
            ..Default::default()
        },
        data: crate::config::schema::DataConfig {
            train: PathBuf::from("/nonexistent/data.json"),
            batch_size: 4,
            ..Default::default()
        },
        optimizer: crate::config::schema::OptimSpec {
            name: "adam".to_string(),
            lr: 1e-4,
            params: std::collections::HashMap::new(),
        },
        training: crate::config::schema::TrainingParams {
            epochs: 1,
            output_dir: PathBuf::from("/tmp"),
            ..Default::default()
        },
        lora: None,
        quantize: None,
        merge: None,
        publish: None,
    };

    let err = build_transformer_config_from_spec(&spec).unwrap_err();
    assert!(err.to_string().contains("hidden_size"), "Error must mention 'hidden_size': {err}");

    std::fs::remove_file(&config_path).ok();
}

#[test]
fn test_resolve_model_path_local_file() {
    let local_path = Path::new("model.safetensors");
    let resolved = resolve_model_path(local_path).expect("operation should succeed");
    assert_eq!(resolved, PathBuf::from("model.safetensors"));
}

#[test]
fn test_resolve_model_path_local_dir() {
    let local_path = Path::new("./output/model.gguf");
    let resolved = resolve_model_path(local_path).expect("operation should succeed");
    assert_eq!(resolved, PathBuf::from("./output/model.gguf"));
}

#[test]
fn test_resolve_model_path_hf_repo_id() {
    let hf_path = Path::new("Qwen/Qwen2.5-Coder-0.5B");
    let result = resolve_model_path(hf_path);
    // Without hub-publish feature: error with helpful message
    // With hub-publish feature: would attempt download
    #[cfg(not(feature = "hub-publish"))]
    assert!(result.unwrap_err().to_string().contains("hub-publish"));
    #[cfg(feature = "hub-publish")]
    let _ = result; // May succeed or fail depending on network
}

// =========================================================================
// build_train_config tests — optimizer params, LoRA, distributed, mixed precision
// =========================================================================

/// Helper to create a minimal TrainSpec for build_train_config tests
fn minimal_spec() -> TrainSpec {
    use crate::config::schema::{DataConfig, ModelRef, OptimSpec, TrainingParams};
    use std::collections::HashMap;

    TrainSpec {
        model: ModelRef {
            path: PathBuf::from("/nonexistent/model"),
            config: None,
            ..Default::default()
        },
        data: DataConfig {
            train: PathBuf::from("/nonexistent/data.json"),
            batch_size: 4,
            seq_len: Some(256),
            ..Default::default()
        },
        optimizer: OptimSpec { name: "adam".to_string(), lr: 1e-4, params: HashMap::new() },
        training: TrainingParams {
            epochs: 1,
            output_dir: PathBuf::from("/tmp/test_output"),
            warmup_steps: 100,
            ..Default::default()
        },
        lora: None,
        quantize: None,
        merge: None,
        publish: None,
    }
}

/// Helper to create a minimal TransformerConfig
fn minimal_transformer_config() -> TransformerConfig {
    TransformerConfig {
        hidden_size: 64,
        num_attention_heads: 4,
        num_kv_heads: 2,
        intermediate_size: 128,
        num_hidden_layers: 2,
        vocab_size: 1000,
        max_position_embeddings: 512,
        rms_norm_eps: 1e-5,
        rope_theta: 10000.0,
        use_bias: false,
        head_dim_override: None,
        architecture: ModelArchitecture::Decoder,
        hf_architecture: None,
        hf_model_type: None,
        tie_word_embeddings: false,
    }
}

#[test]
fn test_build_train_config_basic_wiring() {
    let spec = minimal_spec();
    let model_config = minimal_transformer_config();
    let config = build_train_config(model_config, &spec);
    assert!((config.lr - 1e-4).abs() < 1e-8);
    assert_eq!(config.warmup_steps, 100);
    assert_eq!(config.max_seq_len, 256);
}

#[test]
fn test_build_train_config_seq_len_default_when_none() {
    let mut spec = minimal_spec();
    spec.data.seq_len = None; // should default to 512
    let model_config = minimal_transformer_config();
    let config = build_train_config(model_config, &spec);
    assert_eq!(config.max_seq_len, 512);
}

#[test]
fn test_build_train_config_grad_clip() {
    let mut spec = minimal_spec();
    spec.training.grad_clip = Some(1.0);
    let model_config = minimal_transformer_config();
    let config = build_train_config(model_config, &spec);
    assert!((config.base.max_grad_norm.expect("grad clip should be set") - 1.0).abs() < 1e-6);
}

#[test]
fn test_build_train_config_optimizer_params_beta2_weight_decay() {
    let mut spec = minimal_spec();
    spec.optimizer.params.insert("beta2".to_string(), serde_json::json!(0.95));
    spec.optimizer.params.insert("weight_decay".to_string(), serde_json::json!(0.01));
    let model_config = minimal_transformer_config();
    let config = build_train_config(model_config, &spec);
    assert!((config.beta2 - 0.95).abs() < 1e-6);
    assert!((config.weight_decay - 0.01).abs() < 1e-6);
}

#[test]
fn test_build_train_config_gradient_accumulation() {
    let mut spec = minimal_spec();
    spec.training.gradient_accumulation = Some(4);
    let model_config = minimal_transformer_config();
    let config = build_train_config(model_config, &spec);
    assert_eq!(config.accumulation_steps, 4);
}

#[test]
fn test_build_train_config_gradient_accumulation_one() {
    let mut spec = minimal_spec();
    spec.training.gradient_accumulation = Some(1);
    let model_config = minimal_transformer_config();
    let config = build_train_config(model_config, &spec);
    assert_eq!(config.accumulation_steps, 1);
}

#[test]
fn test_build_train_config_max_steps() {
    let mut spec = minimal_spec();
    spec.training.max_steps = Some(5000);
    let model_config = minimal_transformer_config();
    let config = build_train_config(model_config, &spec);
    assert_eq!(config.max_steps, Some(5000));
}

#[test]
fn test_build_train_config_mixed_precision_bf16() {
    use crate::autograd::Precision;
    let mut spec = minimal_spec();
    spec.training.mixed_precision = Some("bf16".to_string());
    let model_config = minimal_transformer_config();
    let config = build_train_config(model_config, &spec);
    assert_eq!(config.precision_config.compute_precision, Precision::Bf16);
}

#[test]
fn test_build_train_config_mixed_precision_fp16() {
    use crate::autograd::Precision;
    let mut spec = minimal_spec();
    spec.training.mixed_precision = Some("fp16".to_string());
    let model_config = minimal_transformer_config();
    let config = build_train_config(model_config, &spec);
    assert_eq!(config.precision_config.compute_precision, Precision::Fp16);
}

#[test]
fn test_build_train_config_mixed_precision_fp32() {
    use crate::autograd::Precision;
    let mut spec = minimal_spec();
    spec.training.mixed_precision = Some("fp32".to_string());
    let model_config = minimal_transformer_config();
    let config = build_train_config(model_config, &spec);
    assert_eq!(config.precision_config.compute_precision, Precision::Fp32);
}

#[test]
fn test_build_train_config_mixed_precision_unknown() {
    use crate::autograd::Precision;
    let mut spec = minimal_spec();
    spec.training.mixed_precision = Some("tf32".to_string());
    let model_config = minimal_transformer_config();
    let config = build_train_config(model_config, &spec);
    // Unknown precision falls back to fp32
    assert_eq!(config.precision_config.compute_precision, Precision::Fp32);
}

#[test]
fn test_build_train_config_checkpointing() {
    let mut spec = minimal_spec();
    spec.training.checkpoints = Some(4);
    let model_config = minimal_transformer_config();
    let config = build_train_config(model_config, &spec);
    assert!(config.checkpoint_config.enabled);
}

#[test]
fn test_build_train_config_deterministic_and_seed() {
    let mut spec = minimal_spec();
    spec.training.deterministic = true;
    spec.training.seed = Some(42);
    let model_config = minimal_transformer_config();
    let config = build_train_config(model_config, &spec);
    assert!(config.deterministic);
    assert_eq!(config.seed, 42);
}

#[test]
fn test_build_train_config_profile_interval() {
    let mut spec = minimal_spec();
    spec.training.profile_interval = 50;
    let model_config = minimal_transformer_config();
    let config = build_train_config(model_config, &spec);
    assert_eq!(config.profile_interval, 50);
}

#[test]
fn test_build_train_config_profile_interval_zero_disabled() {
    let mut spec = minimal_spec();
    spec.training.profile_interval = 0;
    let model_config = minimal_transformer_config();
    let config = build_train_config(model_config, &spec);
    // Zero means disabled — should remain at default (0)
    assert_eq!(config.profile_interval, 0);
}

#[test]
fn test_build_train_config_lora() {
    use crate::config::schema::LoRASpec;
    let mut spec = minimal_spec();
    spec.lora = Some(LoRASpec {
        rank: 16,
        alpha: 32.0,
        target_modules: vec!["q_proj".to_string(), "v_proj".to_string()],
        dropout: 0.0,
        lora_plus_ratio: 1.0,
        double_quantize: false,
        quantize_base: false,
    });
    let model_config = minimal_transformer_config();
    let config = build_train_config(model_config, &spec);
    assert_eq!(config.lora_rank, Some(16));
    assert!((config.lora_alpha.expect("lora_alpha should be set") - 32.0).abs() < 1e-6);
    assert_eq!(
        config.lora_target_modules.as_deref(),
        Some(vec!["q_proj".to_string(), "v_proj".to_string()].as_slice())
    );
}

#[test]
fn test_build_train_config_lora_plus_ratio() {
    use crate::config::schema::LoRASpec;
    let mut spec = minimal_spec();
    spec.lora = Some(LoRASpec {
        rank: 8,
        alpha: 16.0,
        target_modules: vec!["q_proj".to_string()],
        dropout: 0.0,
        lora_plus_ratio: 16.0,
        double_quantize: false,
        quantize_base: false,
    });
    let model_config = minimal_transformer_config();
    let config = build_train_config(model_config, &spec);
    assert!((config.lora_plus_ratio - 16.0).abs() < 1e-6);
}

#[test]
fn test_build_train_config_lora_double_quantize() {
    use crate::config::schema::LoRASpec;
    let mut spec = minimal_spec();
    spec.lora = Some(LoRASpec {
        rank: 4,
        alpha: 8.0,
        target_modules: vec!["v_proj".to_string()],
        dropout: 0.0,
        lora_plus_ratio: 1.0,
        double_quantize: true,
        quantize_base: false,
    });
    let model_config = minimal_transformer_config();
    let config = build_train_config(model_config, &spec);
    assert!(config.double_quantize);
}

#[test]
fn test_build_train_config_lora_quantize_base_nf4() {
    use crate::config::schema::LoRASpec;
    let mut spec = minimal_spec();
    spec.lora = Some(LoRASpec {
        rank: 16,
        alpha: 32.0,
        target_modules: vec!["q_proj".to_string(), "v_proj".to_string()],
        dropout: 0.0,
        lora_plus_ratio: 1.0,
        double_quantize: true,
        quantize_base: true,
    });
    let model_config = minimal_transformer_config();
    let config = build_train_config(model_config, &spec);
    assert!(config.quantize_nf4, "quantize_nf4 should be true when lora.quantize_base=true");
    assert!(config.is_nf4());
    assert!(config.is_lora());
    assert_eq!(config.lora_rank, Some(16));
    assert!(config.double_quantize);
}

#[test]
fn test_build_train_config_lora_no_quantize_base() {
    use crate::config::schema::LoRASpec;
    let mut spec = minimal_spec();
    spec.lora = Some(LoRASpec {
        rank: 8,
        alpha: 16.0,
        target_modules: vec!["q_proj".to_string()],
        dropout: 0.0,
        lora_plus_ratio: 1.0,
        double_quantize: false,
        quantize_base: false,
    });
    let model_config = minimal_transformer_config();
    let config = build_train_config(model_config, &spec);
    assert!(!config.quantize_nf4, "quantize_nf4 should be false when lora.quantize_base=false");
    assert!(!config.is_nf4());
    assert!(config.is_lora());
}

#[test]
fn test_build_train_config_distributed_coordinator() {
    use crate::config::schema::DistributedSpec;
    let mut spec = minimal_spec();
    spec.training.distributed = Some(DistributedSpec {
        world_size: 4,
        backend: "cuda".to_string(),
        role: "coordinator".to_string(),
        coordinator_addr: "127.0.0.1:9000".to_string(),
        rank: 0,
        local_rank: 0,
    });
    let model_config = minimal_transformer_config();
    let config = build_train_config(model_config, &spec);
    let dist = config.distributed.expect("distributed config should be set");
    assert_eq!(dist.world_size, 4);
    assert_eq!(dist.rank, 0);
}

#[test]
fn test_build_train_config_distributed_worker() {
    use crate::config::schema::DistributedSpec;
    let mut spec = minimal_spec();
    spec.training.distributed = Some(DistributedSpec {
        world_size: 2,
        backend: "wgpu".to_string(),
        role: "worker".to_string(),
        coordinator_addr: "10.0.0.1:8080".to_string(),
        rank: 1,
        local_rank: 1,
    });
    let model_config = minimal_transformer_config();
    let config = build_train_config(model_config, &spec);
    let dist = config.distributed.expect("distributed config should be set");
    assert_eq!(dist.world_size, 2);
    assert_eq!(dist.rank, 1);
    assert_eq!(dist.local_rank, 1);
}

#[test]
fn test_build_train_config_distributed_auto_backend() {
    use crate::config::schema::DistributedSpec;
    let mut spec = minimal_spec();
    spec.training.distributed = Some(DistributedSpec {
        world_size: 2,
        backend: "auto".to_string(),
        role: "coordinator".to_string(),
        coordinator_addr: "0.0.0.0:9000".to_string(),
        rank: 0,
        local_rank: 0,
    });
    let model_config = minimal_transformer_config();
    let config = build_train_config(model_config, &spec);
    assert!(config.distributed.is_some());
}

#[test]
fn test_build_train_config_distributed_invalid_addr_fallback() {
    use crate::config::schema::DistributedSpec;
    let mut spec = minimal_spec();
    spec.training.distributed = Some(DistributedSpec {
        world_size: 2,
        backend: "auto".to_string(),
        role: "coordinator".to_string(),
        coordinator_addr: "not-a-valid-address".to_string(),
        rank: 0,
        local_rank: 0,
    });
    let model_config = minimal_transformer_config();
    // Should fall back to 0.0.0.0:9000
    let config = build_train_config(model_config, &spec);
    let dist = config.distributed.expect("distributed config should be set");
    assert_eq!(dist.coordinator_addr.port(), 9000);
}

// =========================================================================
// parse_hf_config error paths — C-10/C-11 required fields
// =========================================================================

#[test]
fn test_parse_hf_config_missing_vocab_size() {
    let config = serde_json::json!({
        "hidden_size": 768,
        "num_attention_heads": 12,
        "num_hidden_layers": 6,
        "intermediate_size": 3072
    });
    let err = parse_hf_config(&config).expect_err("should fail without vocab_size");
    assert!(err.to_string().contains("vocab_size"), "Error: {err}");
}

#[test]
fn test_parse_hf_config_missing_intermediate_size() {
    let config = serde_json::json!({
        "hidden_size": 768,
        "num_attention_heads": 12,
        "num_hidden_layers": 6,
        "vocab_size": 30000
    });
    let err = parse_hf_config(&config).expect_err("should fail without intermediate_size");
    assert!(err.to_string().contains("intermediate_size"), "Error: {err}");
}

#[test]
fn test_parse_hf_config_missing_num_attention_heads() {
    let config = serde_json::json!({
        "hidden_size": 768,
        "num_hidden_layers": 6,
        "vocab_size": 30000,
        "intermediate_size": 3072
    });
    let err = parse_hf_config(&config).expect_err("should fail without num_attention_heads");
    assert!(err.to_string().contains("num_attention_heads"), "Error: {err}");
}

#[test]
fn test_parse_hf_config_missing_num_hidden_layers() {
    let config = serde_json::json!({
        "hidden_size": 768,
        "num_attention_heads": 12,
        "vocab_size": 30000,
        "intermediate_size": 3072
    });
    let err = parse_hf_config(&config).expect_err("should fail without num_hidden_layers");
    assert!(err.to_string().contains("num_hidden_layers"), "Error: {err}");
}

#[test]
fn test_parse_hf_config_optional_defaults() {
    // Minimal required fields only — check defaults for optional fields
    let config = serde_json::json!({
        "hidden_size": 768,
        "num_attention_heads": 12,
        "num_hidden_layers": 6,
        "vocab_size": 30000,
        "intermediate_size": 3072
    });
    let tc = parse_hf_config(&config).expect("should parse with required fields only");
    // num_kv_heads defaults to num_attention_heads (MHA)
    assert_eq!(tc.num_kv_heads, 12);
    // max_position_embeddings defaults to 2048
    assert_eq!(tc.max_position_embeddings, 2048);
    // rope_theta defaults to 10000
    assert!((tc.rope_theta - 10000.0).abs() < 1.0);
    // use_bias defaults to false
    assert!(!tc.use_bias);
    // head_dim_override is None by default
    assert!(tc.head_dim_override.is_none());
    // architecture defaults to Decoder
    assert!(matches!(tc.architecture, ModelArchitecture::Decoder));
    assert!(!tc.tie_word_embeddings);
}

#[test]
fn test_parse_hf_config_encoder_architecture_detection() {
    for model_type in &["bert", "roberta", "distilbert", "albert", "electra", "deberta"] {
        let config = serde_json::json!({
            "hidden_size": 768,
            "num_attention_heads": 12,
            "num_hidden_layers": 6,
            "vocab_size": 30000,
            "intermediate_size": 3072,
            "model_type": model_type
        });
        let tc = parse_hf_config(&config).expect("should parse encoder config");
        assert!(
            matches!(tc.architecture, ModelArchitecture::Encoder),
            "model_type '{model_type}' should be Encoder"
        );
    }
}

#[test]
fn test_parse_hf_config_decoder_architecture_for_unknown_type() {
    let config = serde_json::json!({
        "hidden_size": 768,
        "num_attention_heads": 12,
        "num_hidden_layers": 6,
        "vocab_size": 30000,
        "intermediate_size": 3072,
        "model_type": "llama"
    });
    let tc = parse_hf_config(&config).expect("should parse decoder config");
    assert!(matches!(tc.architecture, ModelArchitecture::Decoder));
    assert_eq!(tc.hf_model_type, Some("llama".to_string()));
}

#[test]
fn test_parse_hf_config_preserves_hf_metadata() {
    let config = serde_json::json!({
        "hidden_size": 768,
        "num_attention_heads": 12,
        "num_hidden_layers": 6,
        "vocab_size": 30000,
        "intermediate_size": 3072,
        "architectures": ["Qwen2ForCausalLM"],
        "model_type": "qwen2",
        "tie_word_embeddings": true,
        "num_key_value_heads": 4,
        "head_dim": 64,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        "attention_bias": true
    });
    let tc = parse_hf_config(&config).expect("should parse full config");
    assert_eq!(tc.hf_architecture, Some("Qwen2ForCausalLM".to_string()));
    assert_eq!(tc.hf_model_type, Some("qwen2".to_string()));
    assert!(tc.tie_word_embeddings);
    assert_eq!(tc.num_kv_heads, 4);
    assert_eq!(tc.head_dim_override, Some(64));
    assert!((tc.rms_norm_eps - 1e-6).abs() < 1e-10);
    assert!((tc.rope_theta - 1_000_000.0).abs() < 1.0);
    assert!(tc.use_bias);
}

/// GH-262: parse_hf_config for Qwen3-4B must produce correct q_dim and kv_dim.
#[test]
fn test_parse_hf_config_qwen3_4b_head_dim() {
    let config = serde_json::json!({
        "hidden_size": 2560,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "num_hidden_layers": 36,
        "vocab_size": 151936,
        "intermediate_size": 9728,
        "head_dim": 128,
        "max_position_embeddings": 40960,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        "attention_bias": false
    });
    let tc = parse_hf_config(&config).expect("Qwen3-4B config should parse");
    assert_eq!(tc.hidden_size, 2560);
    assert_eq!(tc.num_attention_heads, 32);
    assert_eq!(tc.num_kv_heads, 8);
    assert_eq!(tc.head_dim_override, Some(128));
    assert_eq!(tc.head_dim(), 128);
    // Key assertion: q_dim != hidden_size for Qwen3-4B
    assert_eq!(tc.q_dim(), 4096); // 32 * 128
    assert_ne!(tc.q_dim(), tc.hidden_size); // 4096 != 2560
                                            // KV dim
    let kv_dim = tc.num_kv_heads * tc.head_dim();
    assert_eq!(kv_dim, 1024); // 8 * 128
    assert!(!tc.use_bias);
}

// =========================================================================
// Helper function tests: should_log, should_save_checkpoint, reached_max_steps
// =========================================================================

#[test]
fn test_should_log_at_interval() {
    // interval=10: should log at iter_idx 0, 9, 19, 29
    assert!(should_log(0, 10)); // always log first
    assert!(should_log(9, 10)); // (9+1) % 10 == 0
    assert!(!should_log(1, 10));
    assert!(!should_log(8, 10));
    assert!(should_log(19, 10));
}

#[test]
fn test_should_log_interval_one() {
    // interval=1: every step should log
    for i in 0..10 {
        assert!(should_log(i, 1));
    }
}

#[test]
fn test_should_save_checkpoint() {
    // save_interval=100, step must be > 0 and multiple of interval
    assert!(!should_save_checkpoint(0, 0, 100)); // step 0 excluded
    assert!(should_save_checkpoint(100, 0, 100)); // step 100, last=0
    assert!(!should_save_checkpoint(100, 100, 100)); // step==last_save
    assert!(should_save_checkpoint(200, 100, 100)); // step 200, last=100
    assert!(!should_save_checkpoint(50, 0, 100)); // not multiple
}

#[test]
fn test_reached_max_steps() {
    assert!(!reached_max_steps(None, 1000)); // no limit
    assert!(!reached_max_steps(Some(1000), 500)); // not reached
    assert!(reached_max_steps(Some(1000), 1000)); // exactly reached
    assert!(reached_max_steps(Some(1000), 1500)); // exceeded
}

// =========================================================================
// push_capped / push_capped_f64 tests
// =========================================================================

#[test]
fn test_push_capped_basic() {
    let mut history = Vec::new();
    push_capped(&mut history, 1.0, 3);
    push_capped(&mut history, 2.0, 3);
    push_capped(&mut history, 3.0, 3);
    assert_eq!(history, vec![1.0, 2.0, 3.0]);
    push_capped(&mut history, 4.0, 3);
    assert_eq!(history, vec![2.0, 3.0, 4.0]); // oldest removed
}

#[test]
fn test_push_capped_f64_basic() {
    let mut window: Vec<f64> = Vec::new();
    for i in 0..5 {
        push_capped_f64(&mut window, f64::from(i), 3);
    }
    assert_eq!(window, vec![2.0, 3.0, 4.0]);
}

// =========================================================================
// shuffled_batch_order tests
// =========================================================================

#[test]
fn test_shuffled_batch_order_sequential() {
    let order = shuffled_batch_order(5, false, 42, 0);
    assert_eq!(order, vec![0, 1, 2, 3, 4]);
}

#[test]
fn test_shuffled_batch_order_shuffled_is_permutation() {
    let order = shuffled_batch_order(10, true, 42, 0);
    assert_eq!(order.len(), 10);
    let mut sorted = order.clone();
    sorted.sort_unstable();
    assert_eq!(sorted, (0..10).collect::<Vec<_>>());
}

#[test]
fn test_shuffled_batch_order_deterministic() {
    let order1 = shuffled_batch_order(10, true, 42, 0);
    let order2 = shuffled_batch_order(10, true, 42, 0);
    assert_eq!(order1, order2, "Same seed+epoch should produce same order");
}

#[test]
fn test_shuffled_batch_order_different_epochs() {
    let order0 = shuffled_batch_order(10, true, 42, 0);
    let order1 = shuffled_batch_order(10, true, 42, 1);
    assert_ne!(order0, order1, "Different epochs should produce different orders");
}

#[test]
fn test_shuffled_batch_order_different_seeds() {
    let order_a = shuffled_batch_order(10, true, 42, 0);
    let order_b = shuffled_batch_order(10, true, 99, 0);
    assert_ne!(order_a, order_b, "Different seeds should produce different orders");
}

// =========================================================================
// checkpoint_path / parse_checkpoint_step tests
// =========================================================================

#[test]
fn test_checkpoint_path() {
    let path = checkpoint_path(Path::new("/output"), 500);
    assert_eq!(path, PathBuf::from("/output/model-step-500.apr"));
}

#[test]
fn test_parse_checkpoint_step_valid() {
    // APR format (primary)
    assert_eq!(parse_checkpoint_step("model-step-100.apr"), Some(100));
    assert_eq!(parse_checkpoint_step("model-step-0.apr"), Some(0));
    assert_eq!(parse_checkpoint_step("model-step-999999.apr"), Some(999_999));
    // Legacy SafeTensors format (backward compat)
    assert_eq!(parse_checkpoint_step("model-step-100.safetensors"), Some(100));
    assert_eq!(parse_checkpoint_step("model-step-0.safetensors"), Some(0));
}

#[test]
fn test_parse_checkpoint_step_invalid() {
    assert_eq!(parse_checkpoint_step("model.safetensors"), None);
    assert_eq!(parse_checkpoint_step("model.apr"), None);
    assert_eq!(parse_checkpoint_step("model-step-.apr"), None);
    assert_eq!(parse_checkpoint_step("model-step-abc.apr"), None);
    assert_eq!(parse_checkpoint_step("other-file.txt"), None);
}

// =========================================================================
// prune_checkpoints tests
// =========================================================================

#[test]
fn test_prune_checkpoints_unlimited() {
    let dir = std::env::temp_dir().join("entrenar_prune_test_unlimited");
    std::fs::create_dir_all(&dir).expect("dir creation should succeed");
    // max_keep=0 means unlimited — nothing should be pruned
    for step in [100, 200, 300] {
        let path = dir.join(format!("model-step-{step}.safetensors"));
        std::fs::write(&path, "test").expect("write should succeed");
    }
    prune_checkpoints(&dir, 0);
    // All files should still exist
    for step in [100, 200, 300] {
        assert!(dir.join(format!("model-step-{step}.safetensors")).exists());
    }
    // Cleanup
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn test_prune_checkpoints_removes_oldest() {
    let dir = std::env::temp_dir().join("entrenar_prune_test_oldest");
    std::fs::create_dir_all(&dir).expect("dir creation should succeed");
    for step in [100, 200, 300, 400, 500] {
        let path = dir.join(format!("model-step-{step}.safetensors"));
        std::fs::write(&path, "test").expect("write should succeed");
    }
    prune_checkpoints(&dir, 2);
    // Only the 2 most recent should remain
    assert!(!dir.join("model-step-100.safetensors").exists());
    assert!(!dir.join("model-step-200.safetensors").exists());
    assert!(!dir.join("model-step-300.safetensors").exists());
    assert!(dir.join("model-step-400.safetensors").exists());
    assert!(dir.join("model-step-500.safetensors").exists());
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn test_prune_checkpoints_no_dir() {
    // Non-existent directory should not panic
    prune_checkpoints(Path::new("/nonexistent_dir_xyz"), 2);
}

// =========================================================================
// verify_checkpoint tests
// =========================================================================

#[test]
fn test_verify_checkpoint_valid_file() {
    let dir = std::env::temp_dir().join("entrenar_verify_test");
    std::fs::create_dir_all(&dir).expect("dir creation should succeed");
    let path = dir.join("test_checkpoint.safetensors");
    std::fs::write(&path, "some model data here").expect("write should succeed");
    // Should not panic — just prints verification message
    verify_checkpoint(&path);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn test_verify_checkpoint_empty_file() {
    let dir = std::env::temp_dir().join("entrenar_verify_empty_test");
    std::fs::create_dir_all(&dir).expect("dir creation should succeed");
    let path = dir.join("empty_checkpoint.safetensors");
    std::fs::write(&path, "").expect("write should succeed");
    // Should not panic — prints warning about empty file
    verify_checkpoint(&path);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn test_verify_checkpoint_missing_file() {
    // Should not panic — prints verification failure
    verify_checkpoint(Path::new("/nonexistent_checkpoint.safetensors"));
}

// =========================================================================
// save_training_state tests
// =========================================================================

#[test]
fn test_save_training_state_creates_file() {
    let dir = std::env::temp_dir().join("entrenar_save_state_test");
    std::fs::create_dir_all(&dir).expect("dir creation should succeed");
    save_training_state(&dir, 100, 2, 50, 42, 1.5);
    let state_path = dir.join("training_state.json");
    assert!(state_path.exists(), "training_state.json should be created");
    let content = std::fs::read_to_string(&state_path).expect("should read training_state.json");
    let parsed: serde_json::Value = serde_json::from_str(&content).expect("should be valid JSON");
    assert_eq!(parsed["step"], 100);
    assert_eq!(parsed["epoch"], 2);
    assert_eq!(parsed["batch_index"], 50);
    assert_eq!(parsed["seed"], 42);
    assert!((parsed["loss_ema"].as_f64().expect("loss_ema") - 1.5).abs() < 1e-10);
    std::fs::remove_dir_all(&dir).ok();
}

// =========================================================================
// zclip_update tests
// =========================================================================

#[test]
fn test_zclip_update_normal_gradient() {
    let mut ema = 1.0;
    let mut ema_sq = 1.0;
    // Normal gradient — no spike expected
    zclip_update(1.1, 10, &mut ema, &mut ema_sq, 0.05, 2.0);
    assert!((ema - (0.05 * 1.1 + 0.95 * 1.0)).abs() < 1e-10);
}

#[test]
fn test_zclip_update_spike_detection() {
    let mut ema = 1.0;
    let mut ema_sq = 1.0;
    // Prime the EMA
    for _ in 0..20 {
        zclip_update(1.0, 0, &mut ema, &mut ema_sq, 0.05, 2.0);
    }
    // Inject a spike — should print warning but not panic
    zclip_update(100.0, 21, &mut ema, &mut ema_sq, 0.05, 2.0);
}

// =========================================================================
// detect_loss_spike tests
// =========================================================================

#[test]
fn test_detect_loss_spike_no_spike() {
    let mut ema = 1.0;
    let mut rollback_count = 0;
    let mut jsonl_file = None;
    // Normal loss — no spike
    detect_loss_spike(1.1, 10, &mut ema, 0.05, 3.0, &mut rollback_count, 3, &mut jsonl_file);
    assert_eq!(rollback_count, 0);
}

#[test]
fn test_detect_loss_spike_with_spike() {
    let mut ema = 1.0;
    let mut rollback_count = 0;
    let mut jsonl_file = None;
    // Loss > 3 * EMA → spike
    detect_loss_spike(5.0, 10, &mut ema, 0.05, 3.0, &mut rollback_count, 3, &mut jsonl_file);
    assert_eq!(rollback_count, 1);
}

#[test]
fn test_detect_loss_spike_max_rollbacks() {
    let mut ema = 1.0;
    let mut rollback_count = 3; // already at max
    let mut jsonl_file = None;
    detect_loss_spike(10.0, 10, &mut ema, 0.05, 3.0, &mut rollback_count, 3, &mut jsonl_file);
    assert_eq!(rollback_count, 3, "Should not increment past max");
}

#[test]
fn test_detect_loss_spike_ema_zero_no_spike() {
    let mut ema = 0.0;
    let mut rollback_count = 0;
    let mut jsonl_file = None;
    // EMA is 0 — condition `*ema > 0.0` fails, no spike
    detect_loss_spike(5.0, 1, &mut ema, 0.05, 3.0, &mut rollback_count, 3, &mut jsonl_file);
    assert_eq!(rollback_count, 0);
}

#[test]
fn test_ent283_cold_start_ema_seeding_prevents_false_rollback() {
    // ENT-283: Simulate the training loop's cold-start EMA seeding.
    // Without seeding, step 2 would see loss=15.5 > 3.0 * EMA(0.775) = 2.325 → false rollback.
    // With seeding, EMA starts at 15.5 so step 2 sees 15.5 < 3.0 * 15.5 → no rollback.
    let mut loss_ema: f64 = 0.0;
    let alpha = 0.05;
    let threshold = 3.0;
    let mut rollback_count = 0;
    let mut jsonl_file = None;

    let batch_loss: f32 = 15.5;

    // Step 1: Seed EMA (mimics the training loop fix)
    if loss_ema == 0.0 {
        loss_ema = f64::from(batch_loss);
    }
    detect_loss_spike(
        batch_loss,
        1,
        &mut loss_ema,
        alpha,
        threshold,
        &mut rollback_count,
        3,
        &mut jsonl_file,
    );
    assert_eq!(rollback_count, 0, "Step 1 should not trigger rollback");

    // Step 2: Same loss — should NOT trigger rollback because EMA is warm
    let batch_loss_2: f32 = 15.5;
    if loss_ema == 0.0 {
        loss_ema = f64::from(batch_loss_2);
    }
    detect_loss_spike(
        batch_loss_2,
        2,
        &mut loss_ema,
        alpha,
        threshold,
        &mut rollback_count,
        3,
        &mut jsonl_file,
    );
    assert_eq!(rollback_count, 0, "Step 2 should not trigger false rollback after EMA seeding");

    // Step 3: Actual spike (10x loss) — SHOULD trigger rollback
    let spike_loss: f32 = 155.0;
    detect_loss_spike(
        spike_loss,
        3,
        &mut loss_ema,
        alpha,
        threshold,
        &mut rollback_count,
        3,
        &mut jsonl_file,
    );
    assert_eq!(rollback_count, 1, "Genuine 10x spike should trigger rollback");
}

#[test]
fn test_ent283_without_seeding_false_rollback_on_step2() {
    // Demonstrates the bug: without EMA seeding, step 2 gets a false rollback.
    let mut loss_ema: f64 = 0.0;
    let alpha = 0.05;
    let threshold = 3.0;
    let mut rollback_count = 0;
    let mut jsonl_file = None;

    // Step 1: EMA=0.0, no rollback (guarded by ema > 0.0 check)
    detect_loss_spike(
        15.5,
        1,
        &mut loss_ema,
        alpha,
        threshold,
        &mut rollback_count,
        3,
        &mut jsonl_file,
    );
    assert_eq!(rollback_count, 0, "Step 1: ema=0 guard prevents rollback");
    // After step 1: EMA = 0.05 * 15.5 + 0.95 * 0.0 = 0.775
    assert!((loss_ema - 0.775).abs() < 1e-9, "EMA should be 0.775 after unseeded step 1");

    // Step 2: 15.5 > 3.0 * 0.775 = 2.325 → FALSE rollback!
    detect_loss_spike(
        15.5,
        2,
        &mut loss_ema,
        alpha,
        threshold,
        &mut rollback_count,
        3,
        &mut jsonl_file,
    );
    assert_eq!(rollback_count, 1, "BUG: unseeded EMA causes false rollback on step 2");
}

// =========================================================================
// write_heartbeat test
// =========================================================================

#[test]
fn test_write_heartbeat_creates_file() {
    let dir = std::env::temp_dir().join("entrenar_heartbeat_test");
    std::fs::create_dir_all(&dir).expect("dir creation should succeed");
    let path = dir.join("heartbeat");
    write_heartbeat(&path, 42);
    assert!(path.exists());
    let content = std::fs::read_to_string(&path).expect("should read heartbeat");
    assert!(content.contains("\t42"), "heartbeat should contain step number");
    std::fs::remove_dir_all(&dir).ok();
}

// =========================================================================
// ScalingLawPredictor tests
// =========================================================================

#[test]
fn test_scaling_law_predictor_insufficient_data() {
    let predictor = ScalingLawPredictor::new();
    assert!(predictor.fit().is_none(), "Need at least 3 points");
    assert!(predictor.predict(1000).is_none());
}

#[test]
fn test_scaling_law_predictor_fit_and_predict() {
    let mut predictor = ScalingLawPredictor::new();
    // Simulate decreasing loss with more tokens: L ≈ 10 - 0.3 * ln(D)
    // Using values that stay positive even at large D
    predictor.record(1000, 10.0 - 0.3 * (1000.0_f64).ln() as f32);
    predictor.record(10000, 10.0 - 0.3 * (10000.0_f64).ln() as f32);
    predictor.record(100000, 10.0 - 0.3 * (100000.0_f64).ln() as f32);

    let (a, b) = predictor.fit().expect("should fit with 3 points");
    assert!(a > 0.0, "intercept should be positive");
    assert!(b > 0.0, "slope should be positive (loss decreasing)");

    // Predict at a nearby token count (not too far to avoid negative loss)
    let prediction = predictor.predict(200_000);
    assert!(prediction.is_some());
    let (pred_loss, pred_ppl, slope) = prediction.expect("prediction should succeed");
    assert!(pred_loss > 0.0, "predicted loss should be positive: {pred_loss}");
    assert!(pred_ppl > 1.0, "predicted perplexity should be > 1: {pred_ppl}");
    assert!(slope > 0.0, "slope should be positive: {slope}");
}

#[test]
fn test_scaling_law_predictor_two_points_insufficient() {
    let mut predictor = ScalingLawPredictor::new();
    predictor.record(1000, 3.0);
    predictor.record(2000, 2.5);
    assert!(predictor.fit().is_none());
}

// =========================================================================
// advance_curriculum tests
// =========================================================================

#[test]
fn test_advance_curriculum_no_stages() {
    let empty: &[crate::config::schema::CurriculumStage] = &[];
    assert_eq!(advance_curriculum(empty, 0, 100), None);
}

#[test]
fn test_advance_curriculum_single_stage_no_until() {
    let stages = vec![crate::config::schema::CurriculumStage {
        data: PathBuf::from("data.jsonl"),
        until_step: None,
    }];
    assert_eq!(advance_curriculum(&stages, 0, 100), None);
}

#[test]
fn test_advance_curriculum_transition_at_boundary() {
    let stages = vec![
        crate::config::schema::CurriculumStage {
            data: PathBuf::from("easy.jsonl"),
            until_step: Some(1000),
        },
        crate::config::schema::CurriculumStage {
            data: PathBuf::from("hard.jsonl"),
            until_step: None,
        },
    ];
    // Before boundary — no transition
    assert_eq!(advance_curriculum(&stages, 0, 999), None);
    // At boundary — transition to stage 1
    assert_eq!(advance_curriculum(&stages, 0, 1000), Some(1));
    // After boundary — still transitions
    assert_eq!(advance_curriculum(&stages, 0, 1500), Some(1));
}

#[test]
fn test_advance_curriculum_already_at_last_stage() {
    let stages = vec![
        crate::config::schema::CurriculumStage {
            data: PathBuf::from("easy.jsonl"),
            until_step: Some(1000),
        },
        crate::config::schema::CurriculumStage {
            data: PathBuf::from("hard.jsonl"),
            until_step: None,
        },
    ];
    // Already at stage 1 (last) — no transition
    assert_eq!(advance_curriculum(&stages, 1, 2000), None);
}

#[test]
fn test_advance_curriculum_beyond_stages() {
    let stages = vec![crate::config::schema::CurriculumStage {
        data: PathBuf::from("data.jsonl"),
        until_step: Some(100),
    }];
    // current=0, until_step=100, step=200 BUT no next stage → None
    assert_eq!(advance_curriculum(&stages, 0, 200), None);
}

// =========================================================================
// config_from_overrides tests
// =========================================================================

#[test]
fn test_config_from_overrides_complete() {
    use crate::config::schema::ArchitectureOverrides;
    let overrides = ArchitectureOverrides {
        hidden_size: Some(512),
        num_hidden_layers: Some(8),
        num_attention_heads: Some(8),
        num_kv_heads: Some(4),
        intermediate_size: Some(2048),
        vocab_size: Some(32000),
        max_position_embeddings: Some(4096),
        rms_norm_eps: Some(1e-6),
        rope_theta: Some(500000.0),
        use_bias: Some(true),
        head_dim: None,
    };
    let config = config_from_overrides(&overrides).expect("should build from complete overrides");
    assert_eq!(config.hidden_size, 512);
    assert_eq!(config.num_hidden_layers, 8);
    assert_eq!(config.num_kv_heads, 4);
    assert_eq!(config.max_position_embeddings, 4096);
    assert!(config.use_bias);
}

#[test]
fn test_config_from_overrides_missing_required_returns_none() {
    use crate::config::schema::ArchitectureOverrides;
    // Missing hidden_size → None
    let overrides = ArchitectureOverrides {
        hidden_size: None,
        num_hidden_layers: Some(8),
        num_attention_heads: Some(8),
        intermediate_size: Some(2048),
        vocab_size: Some(32000),
        ..Default::default()
    };
    assert!(config_from_overrides(&overrides).is_none());
}

#[test]
fn test_config_from_overrides_defaults_for_optional() {
    use crate::config::schema::ArchitectureOverrides;
    let overrides = ArchitectureOverrides {
        hidden_size: Some(512),
        num_hidden_layers: Some(4),
        num_attention_heads: Some(8),
        intermediate_size: Some(1024),
        vocab_size: Some(10000),
        num_kv_heads: None,            // defaults to num_attention_heads
        max_position_embeddings: None, // defaults to 2048
        rms_norm_eps: None,            // defaults to 1e-5
        rope_theta: None,              // defaults to 10000.0
        use_bias: None,                // defaults to false
        head_dim: None,                // defaults to hidden_size / num_heads
    };
    let config = config_from_overrides(&overrides).expect("should build");
    assert_eq!(config.num_kv_heads, 8); // same as num_attention_heads
    assert_eq!(config.max_position_embeddings, 2048);
    assert!((config.rms_norm_eps - 1e-5).abs() < 1e-10);
    assert!((config.rope_theta - 10000.0).abs() < 1.0);
    assert!(!config.use_bias);
}

// =========================================================================
// apply_architecture_overrides tests
// =========================================================================

#[test]
fn test_apply_architecture_overrides_selective() {
    use crate::config::schema::ArchitectureOverrides;
    let mut config = minimal_transformer_config();
    let overrides = ArchitectureOverrides {
        hidden_size: Some(256),
        num_kv_heads: Some(1),
        ..Default::default()
    };
    apply_architecture_overrides(&mut config, &overrides);
    assert_eq!(config.hidden_size, 256);
    assert_eq!(config.num_kv_heads, 1);
    // Other fields unchanged
    assert_eq!(config.num_attention_heads, 4);
    assert_eq!(config.intermediate_size, 128);
}

// =========================================================================
// fallback_demo_config tests
// =========================================================================

#[test]
fn test_fallback_demo_config_values() {
    let config = fallback_demo_config();
    assert_eq!(config.hidden_size, QWEN_HIDDEN_SIZE);
    assert_eq!(config.num_attention_heads, QWEN_NUM_ATTENTION_HEADS);
    assert_eq!(config.num_kv_heads, QWEN_NUM_KV_HEADS);
    assert_eq!(config.intermediate_size, QWEN_INTERMEDIATE_SIZE);
    assert_eq!(config.num_hidden_layers, QWEN_NUM_HIDDEN_LAYERS);
    assert_eq!(config.vocab_size, QWEN_VOCAB_SIZE);
    assert_eq!(config.max_position_embeddings, QWEN_MAX_POSITION_EMBEDDINGS);
}

// =========================================================================
// load_config error paths
// =========================================================================

#[test]
fn test_load_config_file_not_found() {
    let result = load_config("/nonexistent/path/to/config.yaml");
    assert!(result.is_err());
    let err = result.expect_err("should fail");
    assert!(err.to_string().contains("Failed to read config file"), "Error: {err}");
}

#[test]
fn test_load_config_invalid_yaml() {
    use std::io::Write;
    let dir = std::env::temp_dir().join("entrenar_invalid_yaml_test");
    std::fs::create_dir_all(&dir).expect("dir creation should succeed");
    let path = dir.join("invalid.yaml");
    let mut f = std::fs::File::create(&path).expect("file write should succeed");
    f.write_all(b"{{{{not valid yaml: [").expect("write should succeed");

    let result = load_config(&path);
    assert!(result.is_err());
    let err = result.expect_err("should fail on invalid YAML");
    assert!(
        err.to_string().contains("Failed to parse"),
        "Error should mention parse failure: {err}"
    );
    std::fs::remove_file(&path).ok();
}

#[test]
fn test_load_config_invalid_manifest_yaml() {
    use std::io::Write;
    let dir = std::env::temp_dir().join("entrenar_invalid_manifest_test");
    std::fs::create_dir_all(&dir).expect("dir creation should succeed");
    let path = dir.join("bad_manifest.yaml");
    let mut f = std::fs::File::create(&path).expect("file write should succeed");
    // Has entrenar: key but invalid structure
    f.write_all(b"entrenar: \"1.0\"\nbogus_field: [1, 2, 3]\n").expect("write should succeed");

    let result = load_config(&path);
    assert!(result.is_err(), "Should fail on invalid manifest structure");
    std::fs::remove_file(&path).ok();
}

// =========================================================================
// write_jsonl_event / write_jsonl_event_json tests
// =========================================================================

#[test]
fn test_write_jsonl_event_with_none_file() {
    let mut file = None;
    // Should not panic when file is None
    write_jsonl_event(&mut file, "test", 1, 0.5, 0.4);
}

#[test]
fn test_write_jsonl_event_json_with_none_file() {
    let mut file = None;
    let entry = serde_json::json!({"type": "test"});
    write_jsonl_event_json(&mut file, &entry);
}

#[test]
fn test_write_jsonl_event_with_real_file() {
    let dir = std::env::temp_dir().join("entrenar_jsonl_test");
    std::fs::create_dir_all(&dir).expect("dir creation should succeed");
    let path = dir.join("test.jsonl");
    let mut file = Some(
        std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&path)
            .expect("file open should succeed"),
    );
    write_jsonl_event(&mut file, "step", 10, 1.5, 1.4);
    write_jsonl_event_json(&mut file, &serde_json::json!({"type": "eval", "step": 20}));
    drop(file);
    let content = std::fs::read_to_string(&path).expect("should read jsonl");
    let lines: Vec<&str> = content.lines().collect();
    assert_eq!(lines.len(), 2);
    let line0: serde_json::Value = serde_json::from_str(lines[0]).expect("valid json line 0");
    assert_eq!(line0["type"], "step");
    assert_eq!(line0["step"], 10);
    let line1: serde_json::Value = serde_json::from_str(lines[1]).expect("valid json line 1");
    assert_eq!(line1["type"], "eval");
    assert_eq!(line1["step"], 20);
    std::fs::remove_dir_all(&dir).ok();
}

// =========================================================================
// extract_texts_from_array tests
// =========================================================================

#[test]
fn test_extract_texts_from_array_text_column() {
    let array = vec![
        serde_json::json!({"text": "hello"}),
        serde_json::json!({"text": "world"}),
        serde_json::json!({"other": "ignored"}),
    ];
    let texts = extract_texts_from_array(&array, "text");
    assert_eq!(texts, vec!["hello", "world"]);
}

#[test]
fn test_extract_texts_from_array_content_fallback() {
    let array = vec![serde_json::json!({"content": "foo"}), serde_json::json!({"content": "bar"})];
    let texts = extract_texts_from_array(&array, "text"); // primary "text" missing, falls back to "content"
    assert_eq!(texts, vec!["foo", "bar"]);
}

#[test]
fn test_extract_texts_from_array_empty() {
    let array: Vec<serde_json::Value> = vec![];
    let texts = extract_texts_from_array(&array, "text");
    assert!(texts.is_empty());
}

#[test]
fn test_extract_texts_from_array_custom_column() {
    let array = vec![
        serde_json::json!({"code": "fn main() {}"}),
        serde_json::json!({"code": "print('hi')"}),
    ];
    let texts = extract_texts_from_array(&array, "code");
    assert_eq!(texts, vec!["fn main() {}", "print('hi')"]);
}

// =========================================================================
// now_ms test
// =========================================================================

#[test]
fn test_now_ms_returns_reasonable_value() {
    let ms = now_ms();
    // Should be after 2020-01-01 (1577836800000 ms)
    assert!(ms > 1_577_836_800_000, "now_ms should return current time in ms: {ms}");
}

// =========================================================================
// JSONL text loading with tokenizer (try_load_from_jsonl)
// =========================================================================

#[test]
fn test_try_load_from_jsonl_without_tokenizer() {
    let content = r#"{"text": "hello world"}
{"text": "foo bar"}"#;
    let result = try_load_from_jsonl(content, None, 2, 32, "text");
    assert!(result.is_none(), "Without tokenizer, should return None");
}

#[test]
fn test_try_load_from_jsonl_empty_content() {
    let tokenizer = HfTokenizer::qwen2();
    let result = try_load_from_jsonl("", Some(&tokenizer), 2, 32, "text");
    assert!(result.is_none(), "Empty content should return None");
}

#[test]
fn test_try_load_from_jsonl_valid() {
    let tokenizer = HfTokenizer::qwen2();
    let content = r#"{"text": "Hello world, this is a test sentence for tokenization."}
{"text": "Another sentence for testing purposes with more tokens."}"#;
    let result = try_load_from_jsonl(content, Some(&tokenizer), 2, 64, "text");
    assert!(result.is_some());
    let batches = result.expect("should have result").expect("should succeed");
    assert!(!batches.is_empty());
}

// =========================================================================
// is_manifest_format edge cases
// =========================================================================

#[test]
fn test_is_manifest_format_empty_string() {
    assert!(!is_manifest_format(""));
}

#[test]
fn test_is_manifest_format_entrenar_in_value_not_key() {
    // "entrenar" as a value, not a key at line start
    assert!(!is_manifest_format("name: entrenar\n"));
}

#[test]
fn test_is_manifest_format_indented_entrenar_not_detected() {
    // Indented entrenar should not be detected (it's not at line start)
    assert!(!is_manifest_format("  entrenar: \"1.0\"\n"));
}

// =========================================================================
// print_max_steps (smoke test)
// =========================================================================

#[test]
fn test_print_max_steps_some() {
    // Should not panic
    print_max_steps(Some(1000));
}

#[test]
fn test_print_max_steps_none() {
    // Should not panic
    print_max_steps(None);
}

// =========================================================================
// update_noise_scale tests
// =========================================================================

#[test]
fn test_update_noise_scale_insufficient_data() {
    let mut window = Vec::new();
    let mut last_step = usize::MAX;
    let mut file = None;
    // Less than 10 points — should not log
    for i in 0..9 {
        update_noise_scale(1.0, i * 100, &mut window, 100, &mut last_step, &mut file);
    }
    assert_eq!(last_step, usize::MAX, "Should not have logged yet");
}

#[test]
fn test_update_noise_scale_logs_at_interval() {
    let mut window = Vec::new();
    let mut last_step = usize::MAX;
    let mut file = None;
    // Add 10+ points, then hit a step that is multiple of interval
    for i in 1..=10 {
        push_capped_f64(&mut window, 1.0 + 0.01 * f64::from(i), 100);
    }
    update_noise_scale(1.15, 100, &mut window, 100, &mut last_step, &mut file);
    assert_eq!(last_step, 100, "Should have logged at step 100");
}

/// ALB-007: Parquet text column loading via alimentar
#[test]
#[cfg(all(not(target_arch = "wasm32"), feature = "parquet"))]
fn test_load_lm_batches_from_parquet_text_column() {
    use arrow::array::{RecordBatch, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    // Create a temp parquet file with text data
    let dir = tempfile::tempdir().expect("temp dir should succeed");
    let parquet_path = dir.path().join("train.parquet");

    let schema = Arc::new(Schema::new(vec![Field::new("text", DataType::Utf8, false)]));
    let texts = StringArray::from(vec![
        "def hello():\n    print('hello world')",
        "def add(a, b):\n    return a + b",
        "class Foo:\n    def __init__(self):\n        self.x = 1",
        "import os\nprint(os.getcwd())",
    ]);
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(texts)])
        .expect("batch creation should succeed");

    let file = std::fs::File::create(&parquet_path).expect("file create should succeed");
    let mut writer = parquet::arrow::ArrowWriter::try_new(file, schema, None)
        .expect("writer creation should succeed");
    writer.write(&batch).expect("write should succeed");
    writer.close().expect("close should succeed");

    // Load via our new implementation
    let tokenizer = HfTokenizer::qwen2();
    let batches = load_lm_batches_from_parquet(&parquet_path, &tokenizer, 2, 64, "text")
        .expect("parquet loading should succeed");

    assert!(!batches.is_empty());
    // 4 texts with batch_size=2 → at least 2 batches
    assert!(batches.len() >= 2);
    assert!(batches[0].batch_size <= 2);
    assert!(batches[0].seq_len > 0);
}

/// ALB-007: Parquet directory loading (multiple shards)
#[test]
#[cfg(all(not(target_arch = "wasm32"), feature = "parquet"))]
fn test_load_lm_batches_from_parquet_directory() {
    use arrow::array::{RecordBatch, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    let dir = tempfile::tempdir().expect("temp dir should succeed");
    let shard_dir = dir.path().join("shards");
    std::fs::create_dir_all(&shard_dir).expect("dir creation should succeed");

    let schema = Arc::new(Schema::new(vec![Field::new("text", DataType::Utf8, false)]));

    // Write two shard files
    for (i, texts) in
        [vec!["def foo(): pass", "def bar(): return 1"], vec!["class A: pass", "import sys"]]
            .iter()
            .enumerate()
    {
        let arr = StringArray::from(texts.clone());
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(arr)])
            .expect("batch should succeed");
        let path = shard_dir.join(format!("shard_{i:04}.parquet"));
        let file = std::fs::File::create(&path).expect("file should succeed");
        let mut writer = parquet::arrow::ArrowWriter::try_new(file, schema.clone(), None)
            .expect("writer should succeed");
        writer.write(&batch).expect("write should succeed");
        writer.close().expect("close should succeed");
    }

    let tokenizer = HfTokenizer::qwen2();
    let batches = load_lm_batches_from_parquet(&shard_dir, &tokenizer, 2, 64, "text")
        .expect("directory loading should succeed");

    assert!(!batches.is_empty());
    // 4 total texts across 2 shards
    let total_seqs: usize = batches.iter().map(|b| b.batch_size).sum();
    assert_eq!(total_seqs, 4);
}

/// ALB-007: Missing text column returns error
#[test]
#[cfg(all(not(target_arch = "wasm32"), feature = "parquet"))]
fn test_load_lm_batches_from_parquet_missing_column() {
    use arrow::array::{Int32Array, RecordBatch};
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;

    let dir = tempfile::tempdir().expect("temp dir should succeed");
    let path = dir.path().join("numeric.parquet");

    let schema = Arc::new(Schema::new(vec![Field::new("numbers", DataType::Int32, false)]));
    let arr = Int32Array::from(vec![1, 2, 3]);
    let batch =
        RecordBatch::try_new(schema.clone(), vec![Arc::new(arr)]).expect("batch should succeed");

    let file = std::fs::File::create(&path).expect("file should succeed");
    let mut writer =
        parquet::arrow::ArrowWriter::try_new(file, schema, None).expect("writer should succeed");
    writer.write(&batch).expect("write should succeed");
    writer.close().expect("close should succeed");

    let tokenizer = HfTokenizer::qwen2();
    let result = load_lm_batches_from_parquet(&path, &tokenizer, 2, 64, "text");
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(err_msg.contains("No text column found"));
}
