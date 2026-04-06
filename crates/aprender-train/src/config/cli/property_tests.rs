//! Property-based tests for CLI argument parsing

use super::*;
use proptest::prelude::*;

// Strategy for valid config paths
fn config_path_strategy() -> impl Strategy<Value = String> {
    "[a-zA-Z][a-zA-Z0-9_-]{0,20}\\.(yaml|yml)"
}

// Strategy for valid output paths
fn output_path_strategy() -> impl Strategy<Value = String> {
    "[a-zA-Z][a-zA-Z0-9_/-]{0,30}"
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    #[test]
    fn prop_train_command_parses(config in config_path_strategy()) {
        let result = parse_args(["entrenar", "train", &config]);
        prop_assert!(result.is_ok());
        let cli = result.expect("operation should succeed");
        match cli.command {
            Command::Train(args) => {
                prop_assert_eq!(args.config.to_str().expect("config should be valid"), &config);
            }
            _ => prop_assert!(false, "Expected Train command"),
        }
    }

    #[test]
    fn prop_validate_command_parses(config in config_path_strategy()) {
        let result = parse_args(["entrenar", "validate", &config]);
        prop_assert!(result.is_ok());
        let cli = result.expect("operation should succeed");
        match cli.command {
            Command::Validate(args) => {
                prop_assert_eq!(args.config.to_str().expect("config should be valid"), &config);
            }
            _ => prop_assert!(false, "Expected Validate command"),
        }
    }

    #[test]
    fn prop_info_command_parses(config in config_path_strategy()) {
        let result = parse_args(["entrenar", "info", &config]);
        prop_assert!(result.is_ok());
    }

    #[test]
    fn prop_epochs_override_positive(
        config in config_path_strategy(),
        epochs in 1usize..10000
    ) {
        let epochs_str = epochs.to_string();
        let result = parse_args([
            "entrenar", "train", &config,
            "--epochs", &epochs_str,
        ]);
        prop_assert!(result.is_ok());
        let cli = result.expect("operation should succeed");
        match cli.command {
            Command::Train(args) => {
                prop_assert_eq!(args.epochs, Some(epochs));
            }
            _ => prop_assert!(false, "Expected Train command"),
        }
    }

    #[test]
    fn prop_batch_size_override_positive(
        config in config_path_strategy(),
        batch_size in 1usize..1024
    ) {
        let batch_str = batch_size.to_string();
        let result = parse_args([
            "entrenar", "train", &config,
            "--batch-size", &batch_str,
        ]);
        prop_assert!(result.is_ok());
        let cli = result.expect("operation should succeed");
        match cli.command {
            Command::Train(args) => {
                prop_assert_eq!(args.batch_size, Some(batch_size));
            }
            _ => prop_assert!(false, "Expected Train command"),
        }
    }

    #[test]
    fn prop_learning_rate_override(
        config in config_path_strategy(),
        lr in 1e-10f32..1.0
    ) {
        let lr_str = format!("{lr:.10}");
        let result = parse_args([
            "entrenar", "train", &config,
            "--lr", &lr_str,
        ]);
        prop_assert!(result.is_ok());
        let cli = result.expect("operation should succeed");
        match cli.command {
            Command::Train(args) => {
                let parsed_lr = args.lr.expect("parsing should succeed");
                // Allow for float parsing precision
                prop_assert!((parsed_lr - lr).abs() < 1e-6 || (parsed_lr / lr - 1.0).abs() < 1e-4);
            }
            _ => prop_assert!(false, "Expected Train command"),
        }
    }

    #[test]
    fn prop_seed_override(
        config in config_path_strategy(),
        seed in 0u64..u64::MAX
    ) {
        let seed_str = seed.to_string();
        let result = parse_args([
            "entrenar", "train", &config,
            "--seed", &seed_str,
        ]);
        prop_assert!(result.is_ok());
        let cli = result.expect("operation should succeed");
        match cli.command {
            Command::Train(args) => {
                prop_assert_eq!(args.seed, Some(seed));
            }
            _ => prop_assert!(false, "Expected Train command"),
        }
    }

    #[test]
    fn prop_quantize_bits_valid(
        model in output_path_strategy(),
        bits in prop::sample::select(vec![4u8, 8])
    ) {
        let bits_str = bits.to_string();
        let result = parse_args([
            "entrenar", "quantize", &model,
            "--output", "out.gguf",
            "--bits", &bits_str,
        ]);
        prop_assert!(result.is_ok());
        let cli = result.expect("operation should succeed");
        match cli.command {
            Command::Quantize(args) => {
                prop_assert_eq!(args.bits, bits);
            }
            _ => prop_assert!(false, "Expected Quantize command"),
        }
    }

    #[test]
    fn prop_merge_weight_valid(
        weight in 0.0f32..=1.0
    ) {
        let weight_str = format!("{weight:.4}");
        let result = parse_args([
            "entrenar", "merge",
            "model1.gguf", "model2.gguf",
            "--output", "merged.gguf",
            "--method", "slerp",
            "--weight", &weight_str,
        ]);
        prop_assert!(result.is_ok());
        let cli = result.expect("operation should succeed");
        match cli.command {
            Command::Merge(args) => {
                let parsed_weight = args.weight.expect("parsing should succeed");
                prop_assert!((parsed_weight - weight).abs() < 1e-3);
            }
            _ => prop_assert!(false, "Expected Merge command"),
        }
    }

    #[test]
    fn prop_output_format_case_insensitive(
        format in prop::sample::select(vec!["text", "TEXT", "Text", "json", "JSON", "Json", "yaml", "YAML", "Yaml"])
    ) {
        let result = format.parse::<OutputFormat>();
        prop_assert!(result.is_ok());
    }

    #[test]
    fn prop_merge_method_case_insensitive(
        method in prop::sample::select(vec!["ties", "TIES", "dare", "DARE", "slerp", "SLERP", "average", "avg"])
    ) {
        let result = method.parse::<MergeMethod>();
        prop_assert!(result.is_ok());
    }

    #[test]
    fn prop_quant_method_case_insensitive(
        method in prop::sample::select(vec!["symmetric", "SYMMETRIC", "sym", "SYM", "asymmetric", "asym"])
    ) {
        let result = method.parse::<QuantMethod>();
        prop_assert!(result.is_ok());
    }

    #[test]
    fn prop_verbose_quiet_exclusive(config in config_path_strategy()) {
        // Can't have both verbose and quiet
        let cli_v = parse_args(["entrenar", "-v", "train", &config]).expect("parsing should succeed");
        let cli_q = parse_args(["entrenar", "-q", "train", &config]).expect("parsing should succeed");

        prop_assert!(cli_v.verbose && !cli_v.quiet);
        prop_assert!(!cli_q.verbose && cli_q.quiet);
    }

    #[test]
    fn prop_multiple_models_merge(
        model_count in 2usize..=5
    ) {
        let mut args: Vec<String> = vec!["entrenar".to_string(), "merge".to_string()];
        let models: Vec<String> = (0..model_count).map(|i| format!("model{i}.gguf")).collect();
        for m in &models {
            args.push(m.clone());
        }
        args.push("--output".to_string());
        args.push("merged.gguf".to_string());

        let result = parse_args(&args);
        prop_assert!(result.is_ok());
        let cli = result.expect("operation should succeed");
        match cli.command {
            Command::Merge(merge_args) => {
                prop_assert_eq!(merge_args.models.len(), model_count);
            }
            _ => prop_assert!(false, "Expected Merge command"),
        }
    }
}
