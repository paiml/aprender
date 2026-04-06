//! ENT-033: Property tests for YAML schema serialization
//!
//! Tests round-trip serialization, edge cases, and schema robustness.

#[cfg(test)]
mod tests {
    use crate::config::schema::*;
    use crate::config::validate::{validate_config, ValidationError};
    use proptest::prelude::*;
    use std::collections::HashMap;
    use std::path::PathBuf;

    // ============================================================
    // Arbitrary Generators
    // ============================================================

    fn arb_path() -> impl Strategy<Value = PathBuf> {
        prop::string::string_regex("[a-z][a-z0-9_/]{0,20}\\.(gguf|parquet|safetensors)")
            .expect("operation should succeed")
            .prop_map(PathBuf::from)
    }

    fn arb_layer_name() -> impl Strategy<Value = String> {
        prop::string::string_regex("[a-z][a-z0-9_]{0,15}").expect("operation should succeed")
    }

    fn arb_optimizer_name() -> impl Strategy<Value = String> {
        prop_oneof!["adam", "adamw", "sgd"].prop_map(String::from)
    }

    fn arb_model_ref() -> impl Strategy<Value = ModelRef> {
        (arb_path(), proptest::collection::vec(arb_layer_name(), 0..5))
            .prop_map(|(path, layers)| ModelRef { path, layers, ..Default::default() })
    }

    fn arb_data_config() -> impl Strategy<Value = DataConfig> {
        (
            arb_path(),
            proptest::option::of(arb_path()),
            1usize..256,
            any::<bool>(),
            proptest::option::of(64usize..4096),
        )
            .prop_map(|(train, val, batch_size, auto_infer, seq_len)| DataConfig {
                train,
                val,
                batch_size,
                auto_infer_types: auto_infer,
                seq_len,
                ..Default::default()
            })
    }

    fn arb_optim_spec() -> impl Strategy<Value = OptimSpec> {
        (arb_optimizer_name(), 1e-6f32..1.0).prop_map(|(name, lr)| OptimSpec {
            name,
            lr,
            params: HashMap::new(),
        })
    }

    fn arb_lora_spec() -> impl Strategy<Value = LoRASpec> {
        (1usize..128, 1.0f32..64.0, proptest::collection::vec(arb_layer_name(), 1..5), 0.0f32..0.5)
            .prop_map(|(rank, alpha, target_modules, dropout)| LoRASpec {
                rank,
                alpha,
                target_modules,
                dropout,
                lora_plus_ratio: 1.0,
                double_quantize: false,
                quantize_base: false,
            })
    }

    fn arb_quant_spec() -> impl Strategy<Value = QuantSpec> {
        (prop_oneof![Just(4u8), Just(8u8)], any::<bool>(), any::<bool>())
            .prop_map(|(bits, symmetric, per_channel)| QuantSpec { bits, symmetric, per_channel })
    }

    fn arb_merge_spec() -> impl Strategy<Value = MergeSpec> {
        prop_oneof!["ties", "dare", "slerp"]
            .prop_map(|method| MergeSpec { method: method.clone(), params: HashMap::new() })
    }

    fn arb_training_params() -> impl Strategy<Value = TrainingParams> {
        (
            1usize..100,
            proptest::option::of(0.1f32..10.0),
            proptest::option::of(prop_oneof!["cosine", "linear", "constant"]),
            0usize..1000,
            1usize..10,
        )
            .prop_map(|(epochs, grad_clip, lr_scheduler, warmup, save_interval)| {
                TrainingParams {
                    epochs,
                    grad_clip,
                    lr_scheduler,
                    warmup_steps: warmup,
                    save_interval,
                    output_dir: PathBuf::from("./checkpoints"),
                    ..Default::default()
                }
            })
    }

    fn arb_train_spec() -> impl Strategy<Value = TrainSpec> {
        (
            arb_model_ref(),
            arb_data_config(),
            arb_optim_spec(),
            proptest::option::of(arb_lora_spec()),
            proptest::option::of(arb_quant_spec()),
            proptest::option::of(arb_merge_spec()),
            arb_training_params(),
        )
            .prop_map(|(model, data, optimizer, lora, quantize, merge, training)| {
                TrainSpec { model, data, optimizer, lora, quantize, merge, training, publish: None }
            })
    }

    // ============================================================
    // Round-Trip Serialization Tests
    // ============================================================

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(200))]

        #[test]
        fn prop_model_ref_round_trip(model in arb_model_ref()) {
            let yaml = serde_yaml::to_string(&model).expect("operation should succeed");
            let parsed: ModelRef = serde_yaml::from_str(&yaml).expect("parsing should succeed");
            prop_assert_eq!(model.path, parsed.path);
            prop_assert_eq!(model.layers, parsed.layers);
        }

        #[test]
        fn prop_data_config_round_trip(data in arb_data_config()) {
            let yaml = serde_yaml::to_string(&data).expect("operation should succeed");
            let parsed: DataConfig = serde_yaml::from_str(&yaml).expect("parsing should succeed");
            prop_assert_eq!(data.train, parsed.train);
            prop_assert_eq!(data.val, parsed.val);
            prop_assert_eq!(data.batch_size, parsed.batch_size);
            prop_assert_eq!(data.auto_infer_types, parsed.auto_infer_types);
            prop_assert_eq!(data.seq_len, parsed.seq_len);
        }

        #[test]
        fn prop_optim_spec_round_trip(optim in arb_optim_spec()) {
            let yaml = serde_yaml::to_string(&optim).expect("operation should succeed");
            let parsed: OptimSpec = serde_yaml::from_str(&yaml).expect("parsing should succeed");
            prop_assert_eq!(optim.name, parsed.name);
            prop_assert!((optim.lr - parsed.lr).abs() < 1e-6);
        }

        #[test]
        fn prop_lora_spec_round_trip(lora in arb_lora_spec()) {
            let yaml = serde_yaml::to_string(&lora).expect("operation should succeed");
            let parsed: LoRASpec = serde_yaml::from_str(&yaml).expect("parsing should succeed");
            prop_assert_eq!(lora.rank, parsed.rank);
            prop_assert!((lora.alpha - parsed.alpha).abs() < 1e-5);
            prop_assert_eq!(lora.target_modules, parsed.target_modules);
            prop_assert!((lora.dropout - parsed.dropout).abs() < 1e-6);
        }

        #[test]
        fn prop_quant_spec_round_trip(quant in arb_quant_spec()) {
            let yaml = serde_yaml::to_string(&quant).expect("operation should succeed");
            let parsed: QuantSpec = serde_yaml::from_str(&yaml).expect("parsing should succeed");
            prop_assert_eq!(quant.bits, parsed.bits);
            prop_assert_eq!(quant.symmetric, parsed.symmetric);
            prop_assert_eq!(quant.per_channel, parsed.per_channel);
        }

        #[test]
        fn prop_merge_spec_round_trip(merge in arb_merge_spec()) {
            let yaml = serde_yaml::to_string(&merge).expect("operation should succeed");
            let parsed: MergeSpec = serde_yaml::from_str(&yaml).expect("parsing should succeed");
            prop_assert_eq!(merge.method, parsed.method);
        }

        #[test]
        fn prop_training_params_round_trip(params in arb_training_params()) {
            let yaml = serde_yaml::to_string(&params).expect("operation should succeed");
            let parsed: TrainingParams = serde_yaml::from_str(&yaml).expect("parsing should succeed");
            prop_assert_eq!(params.epochs, parsed.epochs);
            prop_assert_eq!(params.warmup_steps, parsed.warmup_steps);
            prop_assert_eq!(params.save_interval, parsed.save_interval);
            match (params.grad_clip, parsed.grad_clip) {
                (Some(a), Some(b)) => prop_assert!((a - b).abs() < 1e-5),
                (None, None) => {},
                _ => prop_assert!(false, "grad_clip mismatch"),
            }
        }

        #[test]
        fn prop_train_spec_round_trip(spec in arb_train_spec()) {
            let yaml = serde_yaml::to_string(&spec).expect("operation should succeed");
            let parsed: TrainSpec = serde_yaml::from_str(&yaml).expect("parsing should succeed");

            // Core fields match
            prop_assert_eq!(spec.model.path, parsed.model.path);
            prop_assert_eq!(spec.data.batch_size, parsed.data.batch_size);
            prop_assert_eq!(spec.optimizer.name, parsed.optimizer.name);
            prop_assert_eq!(spec.training.epochs, parsed.training.epochs);

            // Optional fields match
            prop_assert_eq!(spec.lora.is_some(), parsed.lora.is_some());
            prop_assert_eq!(spec.quantize.is_some(), parsed.quantize.is_some());
            prop_assert_eq!(spec.merge.is_some(), parsed.merge.is_some());
        }

        // ============================================================
        // Validation Tests
        // ============================================================

        #[test]
        fn prop_valid_spec_passes_validation(spec in arb_train_spec()) {
            // Generated specs should always be valid
            let result = validate_config(&spec);
            prop_assert!(result.is_ok(), "Valid spec failed validation: {:?}", result);
        }

        #[test]
        fn prop_zero_batch_size_fails(spec in arb_train_spec()) {
            let mut spec = spec;
            spec.data.batch_size = 0;
            let result = validate_config(&spec);
            prop_assert!(matches!(result, Err(ValidationError::InvalidBatchSize(0))));
        }

        #[test]
        fn prop_zero_lr_fails(spec in arb_train_spec()) {
            let mut spec = spec;
            spec.optimizer.lr = 0.0;
            let result = validate_config(&spec);
            prop_assert!(matches!(result, Err(ValidationError::InvalidLearningRate(_))));
        }

        #[test]
        fn prop_negative_lr_fails(
            spec in arb_train_spec(),
            neg_lr in -1.0f32..-1e-6
        ) {
            let mut spec = spec;
            spec.optimizer.lr = neg_lr;
            let result = validate_config(&spec);
            prop_assert!(matches!(result, Err(ValidationError::InvalidLearningRate(_))));
        }

        #[test]
        fn prop_zero_epochs_fails(spec in arb_train_spec()) {
            let mut spec = spec;
            spec.training.epochs = 0;
            let result = validate_config(&spec);
            prop_assert!(matches!(result, Err(ValidationError::InvalidEpochs(0))));
        }

        #[test]
        fn prop_invalid_optimizer_fails(
            spec in arb_train_spec(),
            bad_name in "[a-z]{5,10}"
        ) {
            // Skip if accidentally generates valid name
            if ["adam", "adamw", "sgd"].contains(&bad_name.as_str()) {
                return Ok(());
            }
            let mut spec = spec;
            spec.optimizer.name = bad_name;
            let result = validate_config(&spec);
            prop_assert!(matches!(result, Err(ValidationError::InvalidOptimizer(_))));
        }

        #[test]
        fn prop_zero_lora_rank_fails(spec in arb_train_spec()) {
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
            let result = validate_config(&spec);
            prop_assert!(matches!(result, Err(ValidationError::InvalidLoRARank(0))));
        }

        #[test]
        fn prop_invalid_quant_bits_fails(
            spec in arb_train_spec(),
            bad_bits in 0u8..=3
        ) {
            let mut spec = spec;
            spec.quantize = Some(QuantSpec {
                bits: bad_bits,
                symmetric: true,
                per_channel: true,
            });
            let result = validate_config(&spec);
            prop_assert!(matches!(result, Err(ValidationError::InvalidQuantBits(_))));
        }

        #[test]
        fn prop_invalid_merge_method_fails(
            spec in arb_train_spec(),
            bad_method in "[a-z]{4,8}"
        ) {
            // Skip if accidentally generates valid name
            if ["ties", "dare", "slerp"].contains(&bad_method.as_str()) {
                return Ok(());
            }
            let mut spec = spec;
            spec.merge = Some(MergeSpec {
                method: bad_method,
                params: HashMap::new(),
            });
            let result = validate_config(&spec);
            prop_assert!(matches!(result, Err(ValidationError::InvalidMergeMethod(_))));
        }

        #[test]
        fn prop_zero_grad_clip_fails(spec in arb_train_spec()) {
            let mut spec = spec;
            spec.training.grad_clip = Some(0.0);
            let result = validate_config(&spec);
            prop_assert!(matches!(result, Err(ValidationError::InvalidGradClip(_))));
        }

        #[test]
        fn prop_negative_grad_clip_fails(
            spec in arb_train_spec(),
            neg_clip in -10.0f32..-0.01
        ) {
            let mut spec = spec;
            spec.training.grad_clip = Some(neg_clip);
            let result = validate_config(&spec);
            prop_assert!(matches!(result, Err(ValidationError::InvalidGradClip(_))));
        }
    }

    // ============================================================
    // JSON Interoperability Tests
    // ============================================================

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn prop_yaml_json_interop(spec in arb_train_spec()) {
            // YAML -> JSON -> back should preserve data
            let yaml = serde_yaml::to_string(&spec).expect("operation should succeed");
            let from_yaml: TrainSpec = serde_yaml::from_str(&yaml).expect("operation should succeed");

            let json = serde_json::to_string(&from_yaml).expect("JSON serialization should succeed");
            let from_json: TrainSpec = serde_json::from_str(&json).expect("JSON deserialization should succeed");

            prop_assert_eq!(spec.model.path, from_json.model.path);
            prop_assert_eq!(spec.data.batch_size, from_json.data.batch_size);
            prop_assert_eq!(spec.optimizer.name, from_json.optimizer.name);
        }
    }

    // ============================================================
    // Edge Case Unit Tests
    // ============================================================

    #[test]
    fn test_empty_layers_serializes() {
        let model = ModelRef { path: PathBuf::from("model.gguf"), ..Default::default() };
        let yaml = serde_yaml::to_string(&model).expect("operation should succeed");
        let parsed: ModelRef = serde_yaml::from_str(&yaml).expect("parsing should succeed");
        assert!(parsed.layers.is_empty());
    }

    #[test]
    fn test_large_batch_size() {
        let data = DataConfig {
            train: PathBuf::from("data.parquet"),
            batch_size: 1_000_000,
            ..Default::default()
        };
        let yaml = serde_yaml::to_string(&data).expect("operation should succeed");
        let parsed: DataConfig = serde_yaml::from_str(&yaml).expect("parsing should succeed");
        assert_eq!(parsed.batch_size, 1_000_000);
    }

    #[test]
    fn test_very_small_lr() {
        let optim = OptimSpec { name: "adam".to_string(), lr: 1e-10, params: HashMap::new() };
        let yaml = serde_yaml::to_string(&optim).expect("operation should succeed");
        let parsed: OptimSpec = serde_yaml::from_str(&yaml).expect("parsing should succeed");
        assert!((parsed.lr - 1e-10).abs() < 1e-15);
    }

    #[test]
    fn test_unicode_in_path() {
        let model = ModelRef {
            path: PathBuf::from("模型/model.gguf"),
            layers: vec!["層".to_string()],
            ..Default::default()
        };
        let yaml = serde_yaml::to_string(&model).expect("operation should succeed");
        let parsed: ModelRef = serde_yaml::from_str(&yaml).expect("parsing should succeed");
        assert_eq!(parsed.path, PathBuf::from("模型/model.gguf"));
    }

    #[test]
    fn test_all_optional_fields_none() {
        let spec = TrainSpec {
            model: ModelRef { path: PathBuf::from("m.gguf"), ..Default::default() },
            data: DataConfig {
                train: PathBuf::from("d.parquet"),
                batch_size: 1,
                ..Default::default()
            },
            optimizer: OptimSpec { name: "sgd".to_string(), lr: 0.01, params: HashMap::new() },
            lora: None,
            quantize: None,
            merge: None,
            training: TrainingParams::default(),
            publish: None,
        };

        let yaml = serde_yaml::to_string(&spec).expect("operation should succeed");
        // Optional None fields should not appear in YAML
        assert!(!yaml.contains("lora:"));
        assert!(!yaml.contains("quantize:"));
        assert!(!yaml.contains("merge:"));
    }

    #[test]
    fn test_optim_params_flattened() {
        let yaml = r"
name: adamw
lr: 0.001
beta1: 0.9
beta2: 0.999
weight_decay: 0.01
";
        let optim: OptimSpec = serde_yaml::from_str(yaml).expect("operation should succeed");
        assert_eq!(optim.name, "adamw");
        assert!(optim.params.contains_key("beta1"));
        assert!(optim.params.contains_key("weight_decay"));
    }

    #[test]
    fn test_merge_params_flattened() {
        let yaml = r"
method: ties
density: 0.2
";
        let merge: MergeSpec = serde_yaml::from_str(yaml).expect("operation should succeed");
        assert_eq!(merge.method, "ties");
        assert!(merge.params.contains_key("density"));
    }
}
