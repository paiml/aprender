//! Property-based tests for configuration validation

use super::error::ValidationError;
use super::validator::validate_config;
use crate::config::schema::*;
use proptest::prelude::*;
use std::collections::HashMap;
use std::path::PathBuf;

fn arb_valid_spec() -> impl Strategy<Value = TrainSpec> {
    (
        1usize..256,                        // batch_size
        1e-6f32..1.0,                       // lr
        1usize..100,                        // epochs
        proptest::option::of(0.1f32..10.0), // grad_clip
    )
        .prop_map(|(batch_size, lr, epochs, grad_clip)| TrainSpec {
            model: ModelRef { path: PathBuf::from("model.gguf"), ..Default::default() },
            data: DataConfig {
                train: PathBuf::from("train.parquet"),
                batch_size,
                ..Default::default()
            },
            optimizer: OptimSpec { name: "adam".to_string(), lr, params: HashMap::new() },
            lora: None,
            quantize: None,
            merge: None,
            training: TrainingParams { epochs, grad_clip, ..Default::default() },
            publish: None,
        })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    #[test]
    fn prop_valid_spec_passes(spec in arb_valid_spec()) {
        prop_assert!(validate_config(&spec).is_ok());
    }

    #[test]
    fn prop_zero_batch_size_fails(spec in arb_valid_spec()) {
        let mut spec = spec;
        spec.data.batch_size = 0;
        prop_assert!(matches!(
            validate_config(&spec),
            Err(ValidationError::InvalidBatchSize(0))
        ));
    }

    #[test]
    fn prop_zero_lr_fails(spec in arb_valid_spec()) {
        let mut spec = spec;
        spec.optimizer.lr = 0.0;
        prop_assert!(matches!(
            validate_config(&spec),
            Err(ValidationError::InvalidLearningRate(_))
        ));
    }

    #[test]
    fn prop_negative_lr_fails(
        spec in arb_valid_spec(),
        neg_lr in -1.0f32..-1e-6
    ) {
        let mut spec = spec;
        spec.optimizer.lr = neg_lr;
        prop_assert!(matches!(
            validate_config(&spec),
            Err(ValidationError::InvalidLearningRate(_))
        ));
    }

    #[test]
    fn prop_lr_above_one_fails(
        spec in arb_valid_spec(),
        high_lr in 1.01f32..10.0
    ) {
        let mut spec = spec;
        spec.optimizer.lr = high_lr;
        prop_assert!(matches!(
            validate_config(&spec),
            Err(ValidationError::InvalidLearningRate(_))
        ));
    }

    #[test]
    fn prop_zero_epochs_fails(spec in arb_valid_spec()) {
        let mut spec = spec;
        spec.training.epochs = 0;
        prop_assert!(matches!(
            validate_config(&spec),
            Err(ValidationError::InvalidEpochs(0))
        ));
    }

    #[test]
    fn prop_valid_lora_passes(
        spec in arb_valid_spec(),
        rank in 1usize..1024,
        alpha in 0.1f32..100.0,
        dropout in 0.0f32..0.99
    ) {
        let mut spec = spec;
        spec.lora = Some(LoRASpec {
            rank,
            alpha,
            target_modules: vec!["q_proj".to_string()],
            dropout,
            lora_plus_ratio: 1.0,
            double_quantize: false,
            quantize_base: false,
        });
        prop_assert!(validate_config(&spec).is_ok());
    }

    #[test]
    fn prop_lora_rank_zero_fails(spec in arb_valid_spec()) {
        let mut spec = spec;
        spec.lora = Some(LoRASpec {
            rank: 0,
            alpha: 16.0,
            target_modules: vec!["q_proj".to_string()],
            dropout: 0.0,
            lora_plus_ratio: 1.0,
            double_quantize: false,
            quantize_base: false,
        });
        prop_assert!(matches!(
            validate_config(&spec),
            Err(ValidationError::InvalidLoRARank(0))
        ));
    }

    #[test]
    fn prop_lora_rank_too_high_fails(
        spec in arb_valid_spec(),
        rank in 1025usize..10000
    ) {
        let mut spec = spec;
        spec.lora = Some(LoRASpec {
            rank,
            alpha: 16.0,
            target_modules: vec!["q_proj".to_string()],
            dropout: 0.0,
            lora_plus_ratio: 1.0,
            double_quantize: false,
            quantize_base: false,
        });
        prop_assert!(matches!(
            validate_config(&spec),
            Err(ValidationError::InvalidLoRARank(_))
        ));
    }

    #[test]
    fn prop_lora_alpha_zero_fails(spec in arb_valid_spec()) {
        let mut spec = spec;
        spec.lora = Some(LoRASpec {
            rank: 64,
            alpha: 0.0,
            target_modules: vec!["q_proj".to_string()],
            dropout: 0.0,
            lora_plus_ratio: 1.0,
            double_quantize: false,
            quantize_base: false,
        });
        prop_assert!(matches!(
            validate_config(&spec),
            Err(ValidationError::InvalidLoRAAlpha(_))
        ));
    }

    #[test]
    fn prop_lora_negative_alpha_fails(
        spec in arb_valid_spec(),
        neg_alpha in -100.0f32..-0.01
    ) {
        let mut spec = spec;
        spec.lora = Some(LoRASpec {
            rank: 64,
            alpha: neg_alpha,
            target_modules: vec!["q_proj".to_string()],
            dropout: 0.0,
            lora_plus_ratio: 1.0,
            double_quantize: false,
            quantize_base: false,
        });
        prop_assert!(matches!(
            validate_config(&spec),
            Err(ValidationError::InvalidLoRAAlpha(_))
        ));
    }

    #[test]
    fn prop_lora_dropout_one_fails(spec in arb_valid_spec()) {
        let mut spec = spec;
        spec.lora = Some(LoRASpec {
            rank: 64,
            alpha: 16.0,
            target_modules: vec!["q_proj".to_string()],
            dropout: 1.0,
            lora_plus_ratio: 1.0,
            double_quantize: false,
            quantize_base: false,
        });
        prop_assert!(matches!(
            validate_config(&spec),
            Err(ValidationError::InvalidLoRADropout(_))
        ));
    }

    #[test]
    fn prop_lora_negative_dropout_fails(
        spec in arb_valid_spec(),
        neg_dropout in -1.0f32..-0.01
    ) {
        let mut spec = spec;
        spec.lora = Some(LoRASpec {
            rank: 64,
            alpha: 16.0,
            target_modules: vec!["q_proj".to_string()],
            dropout: neg_dropout,
            lora_plus_ratio: 1.0,
            double_quantize: false,
            quantize_base: false,
        });
        prop_assert!(matches!(
            validate_config(&spec),
            Err(ValidationError::InvalidLoRADropout(_))
        ));
    }

    #[test]
    fn prop_valid_quant_bits(
        spec in arb_valid_spec(),
        bits in prop_oneof![Just(4u8), Just(8u8)]
    ) {
        let mut spec = spec;
        spec.quantize = Some(QuantSpec {
            bits,
            symmetric: true,
            per_channel: true,
        });
        prop_assert!(validate_config(&spec).is_ok());
    }

    #[test]
    fn prop_invalid_quant_bits_fails(
        spec in arb_valid_spec(),
        bits in 0u8..4
    ) {
        let mut spec = spec;
        spec.quantize = Some(QuantSpec {
            bits,
            symmetric: true,
            per_channel: true,
        });
        prop_assert!(matches!(
            validate_config(&spec),
            Err(ValidationError::InvalidQuantBits(_))
        ));
    }

    #[test]
    fn prop_valid_merge_methods(
        spec in arb_valid_spec(),
        method in prop_oneof!["ties", "dare", "slerp"]
    ) {
        let mut spec = spec;
        spec.merge = Some(MergeSpec {
            method: method.clone(),
            params: HashMap::new(),
        });
        prop_assert!(validate_config(&spec).is_ok());
    }

    #[test]
    fn prop_zero_grad_clip_fails(spec in arb_valid_spec()) {
        let mut spec = spec;
        spec.training.grad_clip = Some(0.0);
        prop_assert!(matches!(
            validate_config(&spec),
            Err(ValidationError::InvalidGradClip(_))
        ));
    }

    #[test]
    fn prop_negative_grad_clip_fails(
        spec in arb_valid_spec(),
        neg_clip in -10.0f32..-0.01
    ) {
        let mut spec = spec;
        spec.training.grad_clip = Some(neg_clip);
        prop_assert!(matches!(
            validate_config(&spec),
            Err(ValidationError::InvalidGradClip(_))
        ));
    }

    #[test]
    fn prop_valid_seq_len(
        spec in arb_valid_spec(),
        seq_len in 1usize..8192
    ) {
        let mut spec = spec;
        spec.data.seq_len = Some(seq_len);
        prop_assert!(validate_config(&spec).is_ok());
    }

    #[test]
    fn prop_zero_seq_len_fails(spec in arb_valid_spec()) {
        let mut spec = spec;
        spec.data.seq_len = Some(0);
        prop_assert!(matches!(
            validate_config(&spec),
            Err(ValidationError::InvalidSeqLen(0))
        ));
    }

    #[test]
    fn prop_valid_lr_schedulers(
        spec in arb_valid_spec(),
        scheduler in prop_oneof!["cosine", "linear", "constant"]
    ) {
        let mut spec = spec;
        spec.training.lr_scheduler = Some(scheduler.clone());
        prop_assert!(validate_config(&spec).is_ok());
    }
}
