//! Unit tests for configuration validation

use super::error::ValidationError;
use super::validator::validate_config;
use crate::config::schema::*;
use std::collections::HashMap;
use std::path::PathBuf;

fn create_valid_spec() -> TrainSpec {
    TrainSpec {
        model: ModelRef { path: PathBuf::from("model.gguf"), ..Default::default() },
        data: DataConfig {
            train: PathBuf::from("train.parquet"),
            batch_size: 8,
            ..Default::default()
        },
        optimizer: OptimSpec { name: "adam".to_string(), lr: 0.001, params: HashMap::new() },
        lora: None,
        quantize: None,
        merge: None,
        training: TrainingParams::default(),
        publish: None,
    }
}

#[test]
fn test_valid_config() {
    let spec = create_valid_spec();
    assert!(validate_config(&spec).is_ok());
}

#[test]
fn test_invalid_batch_size() {
    let mut spec = create_valid_spec();
    spec.data.batch_size = 0;
    let err = validate_config(&spec).unwrap_err();
    assert!(matches!(err, ValidationError::InvalidBatchSize(0)));
}

#[test]
fn test_invalid_learning_rate() {
    let mut spec = create_valid_spec();
    spec.optimizer.lr = 0.0;
    let err = validate_config(&spec).unwrap_err();
    assert!(matches!(err, ValidationError::InvalidLearningRate(0.0)));

    spec.optimizer.lr = -0.1;
    let err = validate_config(&spec).unwrap_err();
    assert!(matches!(err, ValidationError::InvalidLearningRate(_)));
}

#[test]
fn test_invalid_optimizer() {
    let mut spec = create_valid_spec();
    spec.optimizer.name = "invalid".to_string();
    let err = validate_config(&spec).unwrap_err();
    assert!(matches!(err, ValidationError::InvalidOptimizer(_)));
}

#[test]
fn test_invalid_epochs() {
    let mut spec = create_valid_spec();
    spec.training.epochs = 0;
    let err = validate_config(&spec).unwrap_err();
    assert!(matches!(err, ValidationError::InvalidEpochs(0)));
}

#[test]
fn test_invalid_lora_rank() {
    let mut spec = create_valid_spec();
    spec.lora = Some(LoRASpec {
        rank: 0,
        alpha: 16.0,
        target_modules: vec!["q_proj".to_string()],
        dropout: 0.0,
        lora_plus_ratio: 1.0,
        double_quantize: false,
        quantize_base: false,
    });
    let err = validate_config(&spec).unwrap_err();
    assert!(matches!(err, ValidationError::InvalidLoRARank(0)));
}

#[test]
fn test_invalid_quant_bits() {
    let mut spec = create_valid_spec();
    spec.quantize = Some(QuantSpec { bits: 16, symmetric: true, per_channel: true });
    let err = validate_config(&spec).unwrap_err();
    assert!(matches!(err, ValidationError::InvalidQuantBits(16)));
}

#[test]
fn test_invalid_merge_method() {
    let mut spec = create_valid_spec();
    spec.merge = Some(MergeSpec { method: "invalid".to_string(), params: HashMap::new() });
    let err = validate_config(&spec).unwrap_err();
    assert!(matches!(err, ValidationError::InvalidMergeMethod(_)));
}

#[test]
fn test_invalid_grad_clip() {
    let mut spec = create_valid_spec();
    spec.training.grad_clip = Some(0.0);
    let err = validate_config(&spec).unwrap_err();
    assert!(matches!(err, ValidationError::InvalidGradClip(0.0)));

    spec.training.grad_clip = Some(-1.0);
    let err = validate_config(&spec).unwrap_err();
    assert!(matches!(err, ValidationError::InvalidGradClip(_)));
}

#[test]
fn test_invalid_lr_too_high() {
    let mut spec = create_valid_spec();
    spec.optimizer.lr = 1.5;
    let err = validate_config(&spec).unwrap_err();
    assert!(matches!(err, ValidationError::InvalidLearningRate(_)));
}

#[test]
fn test_invalid_lora_alpha() {
    let mut spec = create_valid_spec();
    spec.lora = Some(LoRASpec {
        rank: 64,
        alpha: 0.0,
        target_modules: vec!["q_proj".to_string()],
        dropout: 0.0,
        lora_plus_ratio: 1.0,
        double_quantize: false,
        quantize_base: false,
    });
    let err = validate_config(&spec).unwrap_err();
    assert!(matches!(err, ValidationError::InvalidLoRAAlpha(_)));
}

#[test]
fn test_invalid_lora_dropout() {
    let mut spec = create_valid_spec();
    spec.lora = Some(LoRASpec {
        rank: 64,
        alpha: 16.0,
        target_modules: vec!["q_proj".to_string()],
        dropout: 1.0,
        lora_plus_ratio: 1.0,
        double_quantize: false,
        quantize_base: false,
    });
    let err = validate_config(&spec).unwrap_err();
    assert!(matches!(err, ValidationError::InvalidLoRADropout(_)));
}

#[test]
fn test_empty_lora_targets() {
    let mut spec = create_valid_spec();
    spec.lora = Some(LoRASpec {
        rank: 64,
        alpha: 16.0,
        target_modules: vec![],
        dropout: 0.0,
        lora_plus_ratio: 1.0,
        double_quantize: false,
        quantize_base: false,
    });
    let err = validate_config(&spec).unwrap_err();
    assert!(matches!(err, ValidationError::EmptyLoRATargets));
}

#[test]
fn test_invalid_lora_rank_too_high() {
    let mut spec = create_valid_spec();
    spec.lora = Some(LoRASpec {
        rank: 2000,
        alpha: 16.0,
        target_modules: vec!["q_proj".to_string()],
        dropout: 0.0,
        lora_plus_ratio: 1.0,
        double_quantize: false,
        quantize_base: false,
    });
    let err = validate_config(&spec).unwrap_err();
    assert!(matches!(err, ValidationError::InvalidLoRARank(_)));
}

#[test]
fn test_invalid_seq_len() {
    let mut spec = create_valid_spec();
    spec.data.seq_len = Some(0);
    let err = validate_config(&spec).unwrap_err();
    assert!(matches!(err, ValidationError::InvalidSeqLen(0)));
}

#[test]
fn test_invalid_lr_scheduler() {
    let mut spec = create_valid_spec();
    spec.training.lr_scheduler = Some("invalid".to_string());
    let err = validate_config(&spec).unwrap_err();
    assert!(matches!(err, ValidationError::InvalidLRScheduler(_)));
}

#[test]
fn test_valid_lr_schedulers() {
    for scheduler in ["cosine", "linear", "constant", "step", "exponential", "one_cycle", "plateau"]
    {
        let mut spec = create_valid_spec();
        spec.training.lr_scheduler = Some(scheduler.to_string());
        assert!(validate_config(&spec).is_ok(), "scheduler '{scheduler}' should be valid");
    }
}

// =========================================================================
// Additional Coverage Tests
// =========================================================================

#[test]
fn test_invalid_save_interval() {
    let mut spec = create_valid_spec();
    spec.training.save_interval = 0;
    let err = validate_config(&spec).unwrap_err();
    assert!(matches!(err, ValidationError::InvalidSaveInterval(0)));
}

#[test]
fn test_valid_optimizers() {
    for opt in ["adam", "adamw", "sgd", "rmsprop", "adagrad", "lamb"] {
        let mut spec = create_valid_spec();
        spec.optimizer.name = opt.to_string();
        assert!(validate_config(&spec).is_ok(), "optimizer '{opt}' should be valid");
    }
}

#[test]
fn test_valid_quant_bits_4_and_8() {
    for bits in [4u8, 8u8] {
        let mut spec = create_valid_spec();
        spec.quantize = Some(QuantSpec { bits, symmetric: true, per_channel: false });
        assert!(validate_config(&spec).is_ok());
    }
}

#[test]
fn test_valid_merge_methods() {
    for method in ["ties", "dare", "slerp"] {
        let mut spec = create_valid_spec();
        spec.merge = Some(MergeSpec { method: method.to_string(), params: HashMap::new() });
        assert!(validate_config(&spec).is_ok());
    }
}

#[test]
fn test_validation_error_display() {
    let e = ValidationError::ModelPathNotFound("model.bin".to_string());
    assert!(e.to_string().contains("Model path does not exist"));

    let e = ValidationError::TrainDataNotFound("train.csv".to_string());
    assert!(e.to_string().contains("Training data path"));

    let e = ValidationError::ValDataNotFound("val.csv".to_string());
    assert!(e.to_string().contains("Validation data path"));

    let e = ValidationError::InvalidLearningRate(0.0);
    assert!(e.to_string().contains("Invalid learning rate"));

    let e = ValidationError::InvalidBatchSize(0);
    assert!(e.to_string().contains("Invalid batch size"));

    let e = ValidationError::InvalidEpochs(0);
    assert!(e.to_string().contains("Invalid epochs"));

    let e = ValidationError::InvalidLoRARank(0);
    assert!(e.to_string().contains("Invalid LoRA rank"));

    let e = ValidationError::InvalidLoRAAlpha(0.0);
    assert!(e.to_string().contains("Invalid LoRA alpha"));

    let e = ValidationError::InvalidLoRADropout(1.0);
    assert!(e.to_string().contains("Invalid LoRA dropout"));

    let e = ValidationError::InvalidQuantBits(3);
    assert!(e.to_string().contains("Invalid quantization bits"));

    let e = ValidationError::InvalidOptimizer("bad".to_string());
    assert!(e.to_string().contains("Invalid optimizer"));

    let e = ValidationError::InvalidMergeMethod("bad".to_string());
    assert!(e.to_string().contains("Invalid merge method"));

    let e = ValidationError::InvalidGradClip(-1.0);
    assert!(e.to_string().contains("Invalid gradient clip"));

    let e = ValidationError::InvalidSeqLen(0);
    assert!(e.to_string().contains("Invalid sequence length"));

    let e = ValidationError::InvalidSaveInterval(0);
    assert!(e.to_string().contains("Invalid save interval"));

    let e = ValidationError::EmptyLoRATargets;
    assert!(e.to_string().contains("cannot be empty"));

    let e = ValidationError::InvalidLRScheduler("bad".to_string());
    assert!(e.to_string().contains("Invalid LR scheduler"));
}

#[test]
fn test_invalid_lora_dropout_negative() {
    let mut spec = create_valid_spec();
    spec.lora = Some(LoRASpec {
        rank: 64,
        alpha: 16.0,
        target_modules: vec!["q_proj".to_string()],
        dropout: -0.1,
        lora_plus_ratio: 1.0,
        double_quantize: false,
        quantize_base: false,
    });
    let err = validate_config(&spec).unwrap_err();
    assert!(matches!(err, ValidationError::InvalidLoRADropout(_)));
}

#[test]
fn test_valid_config_with_all_optional_fields() {
    let mut spec = create_valid_spec();
    spec.data.val = Some(PathBuf::from("val.parquet"));
    spec.data.seq_len = Some(512);
    spec.training.grad_clip = Some(1.0);
    spec.training.lr_scheduler = Some("cosine".to_string());
    spec.lora = Some(LoRASpec {
        rank: 64,
        alpha: 16.0,
        target_modules: vec!["q_proj".to_string(), "v_proj".to_string()],
        dropout: 0.1,
        lora_plus_ratio: 1.0,
        double_quantize: false,
        quantize_base: false,
    });
    spec.quantize = Some(QuantSpec { bits: 4, symmetric: true, per_channel: true });
    spec.merge = Some(MergeSpec { method: "ties".to_string(), params: HashMap::new() });
    assert!(validate_config(&spec).is_ok());
}

#[test]
fn test_valid_publish_config() {
    let mut spec = create_valid_spec();
    spec.publish = Some(PublishSpec {
        repo: "myuser/my-model".to_string(),
        private: false,
        model_card: true,
        merge_adapters: false,
        format: "safetensors".to_string(),
    });
    assert!(validate_config(&spec).is_ok());
}

#[test]
fn test_invalid_publish_repo_no_slash() {
    let mut spec = create_valid_spec();
    spec.publish = Some(PublishSpec {
        repo: "invalid-repo".to_string(),
        private: false,
        model_card: true,
        merge_adapters: false,
        format: "safetensors".to_string(),
    });
    let err = validate_config(&spec).unwrap_err();
    assert!(matches!(err, ValidationError::InvalidPublishRepo(_)));
}

#[test]
fn test_invalid_publish_repo_empty_parts() {
    let mut spec = create_valid_spec();
    spec.publish = Some(PublishSpec {
        repo: "/model".to_string(),
        private: false,
        model_card: true,
        merge_adapters: false,
        format: "safetensors".to_string(),
    });
    let err = validate_config(&spec).unwrap_err();
    assert!(matches!(err, ValidationError::InvalidPublishRepo(_)));
}

#[test]
fn test_invalid_publish_format() {
    let mut spec = create_valid_spec();
    spec.publish = Some(PublishSpec {
        repo: "org/model".to_string(),
        private: false,
        model_card: true,
        merge_adapters: false,
        format: "pytorch".to_string(),
    });
    let err = validate_config(&spec).unwrap_err();
    assert!(matches!(err, ValidationError::InvalidPublishFormat(_)));
}

#[test]
fn test_valid_publish_gguf_format() {
    let mut spec = create_valid_spec();
    spec.publish = Some(PublishSpec {
        repo: "org/model".to_string(),
        private: false,
        model_card: true,
        merge_adapters: false,
        format: "gguf".to_string(),
    });
    assert!(validate_config(&spec).is_ok());
}
