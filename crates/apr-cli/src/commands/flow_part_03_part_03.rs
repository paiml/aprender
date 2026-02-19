
    #[test]
    fn test_flow_component_from_str_mixed_case_ffn() {
        // Data-driven: all FFN aliases map to FlowComponent::Ffn
        let ffn_aliases = ["FFN", "MLP", "FEEDFORWARD", "Mlp", "FeedForward"];
        for alias in &ffn_aliases {
            assert_eq!(
                alias.parse::<FlowComponent>().expect("parse"),
                FlowComponent::Ffn,
                "{alias} should parse as Ffn"
            );
        }
    }

    #[test]
    fn test_flow_component_from_str_error_message() {
        let err = "banana".parse::<FlowComponent>().unwrap_err();
        assert_eq!(err, "Unknown component: banana");
    }

    #[test]
    fn test_flow_component_from_str_error_whitespace() {
        for bad in [" full", "full ", " "] {
            assert!(bad.parse::<FlowComponent>().is_err(), "{bad:?} should fail");
        }
    }

    #[test]
    fn test_flow_component_from_str_error_partial() {
        for bad in ["ful", "encode", "decode", "atten"] {
            assert!(bad.parse::<FlowComponent>().is_err(), "{bad:?} should fail");
        }
    }

    #[test]
    fn test_flow_component_copy() {
        // FlowComponent derives Copy
        let a = FlowComponent::Full;
        let b = a; // Copy, not move
        assert_eq!(a, b); // a is still accessible
    }

    #[test]
    fn test_flow_component_all_variants_distinct() {
        let variants = [
            FlowComponent::Full,
            FlowComponent::Encoder,
            FlowComponent::Decoder,
            FlowComponent::SelfAttention,
            FlowComponent::CrossAttention,
            FlowComponent::Ffn,
        ];
        // Every pair of distinct variants should be unequal
        for i in 0..variants.len() {
            for j in (i + 1)..variants.len() {
                assert_ne!(
                    variants[i], variants[j],
                    "{:?} should not equal {:?}",
                    variants[i], variants[j]
                );
            }
        }
    }

    // ========================================================================
    // detect_architecture Additional Coverage (data-driven)
    // ========================================================================

    /// Data-driven architecture detection: (tensor_names, expected_result)
    const ARCH_DETECTION_CASES: &[(&[&str], &str)] = &[
        // Has both encoder and decoder, but NO cross_attn -> unknown
        (&["encoder.layers.0.self_attn.q_proj.weight",
          "decoder.layers.0.self_attn.q_proj.weight"], "unknown"),
        // Has cross_attn but no encoder/decoder prefix -> decoder-only (transformer)
        (&["model.layers.0.cross_attn.q_proj.weight"], "decoder-only (transformer)"),
        // encoder_attn keyword -> encoder-decoder
        (&["encoder.layers.0.weight",
          "decoder.layers.0.weight",
          "decoder.layers.0.encoder_attn.q_proj.weight"], "encoder-decoder (Whisper/T5)"),
        // Single encoder tensor -> encoder-only
        (&["encoder.conv1.weight"], "encoder-only (BERT)"),
        // Single decoder tensor -> decoder-only
        (&["decoder.embed_tokens.weight"], "decoder-only (GPT/LLaMA)"),
        // LLaMA-style (model.layers.* + lm_head)
        (&["model.embed_tokens.weight",
          "model.layers.0.self_attn.q_proj.weight",
          "lm_head.weight"], "decoder-only (LLaMA/Qwen2)"),
        // GGUF-style (blk.* + output.weight)
        (&["token_embd.weight",
          "blk.0.attn_q.weight",
          "blk.0.ffn_gate.weight",
          "output.weight"], "decoder-only (LLaMA/Qwen2)"),
        // "encoder" as substring, not prefix -> unknown
        (&["some_encoder_layer.weight"], "unknown"),
        // "decoder" as substring, not prefix -> unknown
        (&["pre_decoder.layers.0.weight"], "unknown"),
    ];

    #[test]
    fn test_detect_architecture_additional_cases() {
        for (tensor_strs, expected) in ARCH_DETECTION_CASES {
            let names: Vec<String> = tensor_strs.iter().map(|s| s.to_string()).collect();
            assert_eq!(
                detect_architecture(&names), *expected,
                "Failed for tensors: {tensor_strs:?}"
            );
        }
    }

    // ========================================================================
    // compute_stats Additional Coverage (data-driven)
    // ========================================================================

    /// Helper: assert compute_stats results within tolerance
    fn assert_stats(data: &[f32], exp_min: f32, exp_max: f32, exp_mean: f32, exp_std: f32, tol: f32) {
        let (min, max, mean, std) = compute_stats(data);
        assert!((min - exp_min).abs() < tol, "min: {min} != {exp_min}");
        assert!((max - exp_max).abs() < tol, "max: {max} != {exp_max}");
        assert!((mean - exp_mean).abs() < tol, "mean: {mean} != {exp_mean}");
        assert!((std - exp_std).abs() < tol, "std: {std} != {exp_std}");
    }

    #[test]
    fn test_compute_stats_two_values() {
        assert_stats(&[3.0, 7.0], 3.0, 7.0, 5.0, 2.0, 0.001);
    }

    #[test]
    fn test_compute_stats_large_values() {
        assert_stats(&[1e6, 2e6, 3e6], 1e6, 3e6, 2e6, 816496.6, 1.0);
    }

    #[test]
    fn test_compute_stats_very_small_values() {
        let data = [1e-7, 2e-7, 3e-7];
        let (min, max, mean, _std) = compute_stats(&data);
        assert!((min - 1e-7).abs() < 1e-10);
        assert!((max - 3e-7).abs() < 1e-10);
        assert!((mean - 2e-7).abs() < 1e-10);
    }

    #[test]
    fn test_compute_stats_mixed_sign() {
        let data = [-100.0, 0.0, 100.0];
        let (min, max, mean, _std) = compute_stats(&data);
        assert_eq!(min, -100.0);
        assert_eq!(max, 100.0);
        assert!(mean.abs() < 0.001);
    }

    #[test]
    fn test_compute_stats_single_and_zero_cases() {
        // Data-driven: (input, expected_min, expected_max, expected_mean, expected_std)
        let cases: &[(&[f32], f32, f32, f32, f32)] = &[
            (&[-42.0], -42.0, -42.0, -42.0, 0.0),
            (&[0.0], 0.0, 0.0, 0.0, 0.0),
            (&[0.0, 0.0, 0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0.0),
        ];
        for (data, exp_min, exp_max, exp_mean, exp_std) in cases {
            assert_stats(data, *exp_min, *exp_max, *exp_mean, *exp_std, f32::EPSILON);
        }
    }

    #[test]
    fn test_compute_stats_ascending() {
        let data: Vec<f32> = (1..=100).map(|i| i as f32).collect();
        assert_stats(&data, 1.0, 100.0, 50.5, 28.87, 0.1);
    }

    #[test]
    fn test_compute_stats_descending() {
        let data: Vec<f32> = (1..=100).rev().map(|i| i as f32).collect();
        assert_stats(&data, 1.0, 100.0, 50.5, 28.87, 0.1);
    }

    #[test]
    fn test_compute_stats_typical_weights() {
        // Simulating typical neural network weight distribution
        let data = vec![
            -0.1, 0.05, -0.03, 0.08, -0.07, 0.02, -0.01, 0.04, -0.06, 0.09,
        ];
        let (min, max, mean, std) = compute_stats(&data);
        assert!(min < 0.0);
        assert!(max > 0.0);
        assert!(mean.abs() < 0.05); // near zero mean
        assert!(std > 0.0);
        assert!(std < 0.2); // small spread
    }

    #[test]
    fn test_compute_stats_std_is_non_negative() {
        // Standard deviation must always be >= 0
        let test_cases: &[&[f32]] = &[
            &[1.0, 1.0, 1.0],
            &[-1.0, 1.0],
            &[0.0],
            &[100.0, -100.0],
        ];
        for data in test_cases {
            let (_, _, _, std) = compute_stats(data);
            assert!(std >= 0.0, "std must be non-negative, got {std}");
        }
    }

    #[test]
    fn test_compute_stats_mean_is_between_min_and_max() {
        let data = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let (min, max, mean, _std) = compute_stats(&data);
        assert!(mean >= min, "mean should be >= min");
        assert!(mean <= max, "mean should be <= max");
    }

    // ========================================================================
    // Layer Counting Logic (used in print_encoder_block / print_decoder_block)
    // ========================================================================

    /// Count layers from tensor names matching a given prefix (e.g. "encoder.layers." or "decoder.layers.").
    fn count_layers(tensor_names: &[String], prefix: &str) -> usize {
        tensor_names
            .iter()
            .filter(|n| n.starts_with(prefix))
            .filter_map(|n| {
                n.strip_prefix(prefix)
                    .and_then(|s| s.split('.').next())
                    .and_then(|s| s.parse::<usize>().ok())
            })
            .max()
            .map_or(0, |n| n + 1)
    }

    #[test]
    fn test_encoder_layer_counting() {
        // Data-driven: (tensor_names, expected_layer_count)
        let cases: Vec<(Vec<String>, usize)> = vec![
            // Zero layers (no encoder prefix)
            (vec!["output.weight".to_string()], 0),
            // Single layer (layer 0 only)
            (vec![
                "encoder.layers.0.self_attn.q_proj.weight".to_string(),
                "encoder.layers.0.self_attn.k_proj.weight".to_string(),
            ], 1),
            // Multiple contiguous layers
            (vec![
                "encoder.layers.0.self_attn.weight".to_string(),
                "encoder.layers.1.self_attn.weight".to_string(),
                "encoder.layers.2.self_attn.weight".to_string(),
                "encoder.layers.3.self_attn.weight".to_string(),
            ], 4),
            // Non-contiguous layers (0, 5) -> max=5 -> n_layers=6
            (vec![
                "encoder.layers.0.weight".to_string(),
                "encoder.layers.5.weight".to_string(),
            ], 6),
        ];
        for (names, expected) in &cases {
            assert_eq!(
                count_layers(names, "encoder.layers."), *expected,
                "Failed for: {names:?}"
            );
        }
    }

    #[test]
    fn test_decoder_layer_counting() {
        // Data-driven: (tensor_names, expected_layer_count)
        let cases: Vec<(Vec<String>, usize)> = vec![
            // Zero layers (no decoder prefix)
            (vec!["output.weight".to_string()], 0),
            // Multiple layers with mixed sub-components
            (vec![
                "decoder.layers.0.self_attn.weight".to_string(),
                "decoder.layers.1.encoder_attn.weight".to_string(),
                "decoder.layers.2.ffn.weight".to_string(),
                "decoder.layers.3.self_attn.weight".to_string(),
                "decoder.layers.3.ffn.weight".to_string(),
            ], 4),
        ];
        for (names, expected) in &cases {
            assert_eq!(
                count_layers(names, "decoder.layers."), *expected,
                "Failed for: {names:?}"
            );
        }
    }

    #[test]
    fn test_layer_counting_mixed_encoder_decoder() {
        let tensor_names = vec![
            "encoder.layers.0.weight".to_string(),
            "encoder.layers.1.weight".to_string(),
            "decoder.layers.0.weight".to_string(),
            "decoder.layers.1.weight".to_string(),
            "decoder.layers.2.weight".to_string(),
        ];
        assert_eq!(count_layers(&tensor_names, "encoder.layers."), 2);
        assert_eq!(count_layers(&tensor_names, "decoder.layers."), 3);
    }
