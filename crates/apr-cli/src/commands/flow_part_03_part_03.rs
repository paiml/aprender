
    #[test]
    fn test_flow_component_from_str_mixed_case_ffn() {
        assert_eq!(
            "FFN".parse::<FlowComponent>().expect("parse"),
            FlowComponent::Ffn
        );
        assert_eq!(
            "MLP".parse::<FlowComponent>().expect("parse"),
            FlowComponent::Ffn
        );
        assert_eq!(
            "FEEDFORWARD".parse::<FlowComponent>().expect("parse"),
            FlowComponent::Ffn
        );
        assert_eq!(
            "Mlp".parse::<FlowComponent>().expect("parse"),
            FlowComponent::Ffn
        );
        assert_eq!(
            "FeedForward".parse::<FlowComponent>().expect("parse"),
            FlowComponent::Ffn
        );
    }

    #[test]
    fn test_flow_component_from_str_error_message() {
        let err = "banana".parse::<FlowComponent>().unwrap_err();
        assert_eq!(err, "Unknown component: banana");
    }

    #[test]
    fn test_flow_component_from_str_error_whitespace() {
        assert!(" full".parse::<FlowComponent>().is_err());
        assert!("full ".parse::<FlowComponent>().is_err());
        assert!(" ".parse::<FlowComponent>().is_err());
    }

    #[test]
    fn test_flow_component_from_str_error_partial() {
        assert!("ful".parse::<FlowComponent>().is_err());
        assert!("encode".parse::<FlowComponent>().is_err());
        assert!("decode".parse::<FlowComponent>().is_err());
        assert!("atten".parse::<FlowComponent>().is_err());
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
    // detect_architecture Additional Coverage
    // ========================================================================

    #[test]
    fn test_detect_architecture_encoder_decoder_without_cross_attn() {
        // Has both encoder and decoder, but NO cross_attn
        // This falls through to the else branch
        let names = vec![
            "encoder.layers.0.self_attn.q_proj.weight".to_string(),
            "decoder.layers.0.self_attn.q_proj.weight".to_string(),
        ];
        // No cross_attn -> first condition fails -> encoder_only fails (has_decoder)
        // -> decoder_only fails (has_encoder) -> unknown
        assert_eq!(detect_architecture(&names), "unknown");
    }

    #[test]
    fn test_detect_architecture_only_cross_attn_no_enc_dec() {
        // Has cross_attn mention but neither "encoder" nor "decoder" prefix
        let names = vec!["model.layers.0.cross_attn.q_proj.weight".to_string()];
        // has_encoder=false, has_decoder=false, has_cross_attn=true
        // First condition: false (needs all three)
        // Second: false (no encoder)
        // Third: false (no decoder)
        // PMAT-265: model.layers.* now detected as decoder-only
        assert_eq!(detect_architecture(&names), "decoder-only (transformer)");
    }

    #[test]
    fn test_detect_architecture_encoder_attn_keyword() {
        let names = vec![
            "encoder.layers.0.weight".to_string(),
            "decoder.layers.0.weight".to_string(),
            "decoder.layers.0.encoder_attn.q_proj.weight".to_string(),
        ];
        assert_eq!(detect_architecture(&names), "encoder-decoder (Whisper/T5)");
    }

    #[test]
    fn test_detect_architecture_single_encoder_tensor() {
        let names = vec!["encoder.conv1.weight".to_string()];
        assert_eq!(detect_architecture(&names), "encoder-only (BERT)");
    }

    #[test]
    fn test_detect_architecture_single_decoder_tensor() {
        let names = vec!["decoder.embed_tokens.weight".to_string()];
        assert_eq!(detect_architecture(&names), "decoder-only (GPT/LLaMA)");
    }

    #[test]
    fn test_detect_architecture_llama_style_names() {
        // LLaMA-style models don't use "encoder"/"decoder" prefixes
        let names = vec![
            "model.embed_tokens.weight".to_string(),
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            "lm_head.weight".to_string(),
        ];
        // PMAT-265: LLaMA-style (model.layers.* + lm_head) now detected
        assert_eq!(detect_architecture(&names), "decoder-only (LLaMA/Qwen2)");
    }

    #[test]
    fn test_detect_architecture_gguf_style_names() {
        // GGUF-style: blk.0.attn_q.weight
        let names = vec![
            "token_embd.weight".to_string(),
            "blk.0.attn_q.weight".to_string(),
            "blk.0.ffn_gate.weight".to_string(),
            "output.weight".to_string(),
        ];
        // PMAT-265: GGUF-style (blk.* + output.weight) now detected
        assert_eq!(detect_architecture(&names), "decoder-only (LLaMA/Qwen2)");
    }

    #[test]
    fn test_detect_architecture_encoder_prefix_in_middle() {
        // "encoder" must be a prefix (starts_with), not just a substring
        let names = vec!["some_encoder_layer.weight".to_string()];
        // starts_with("encoder") is false
        assert_eq!(detect_architecture(&names), "unknown");
    }

    #[test]
    fn test_detect_architecture_decoder_prefix_in_middle() {
        let names = vec!["pre_decoder.layers.0.weight".to_string()];
        // starts_with("decoder") is false
        assert_eq!(detect_architecture(&names), "unknown");
    }

    // ========================================================================
    // compute_stats Additional Coverage
    // ========================================================================

    #[test]
    fn test_compute_stats_two_values() {
        let data = [3.0, 7.0];
        let (min, max, mean, std) = compute_stats(&data);
        assert_eq!(min, 3.0);
        assert_eq!(max, 7.0);
        assert_eq!(mean, 5.0);
        assert!((std - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_compute_stats_large_values() {
        let data = [1e6, 2e6, 3e6];
        let (min, max, mean, _std) = compute_stats(&data);
        assert_eq!(min, 1e6);
        assert_eq!(max, 3e6);
        assert!((mean - 2e6).abs() < 1.0);
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
    fn test_compute_stats_single_negative() {
        let data = [-42.0];
        let (min, max, mean, std) = compute_stats(&data);
        assert_eq!(min, -42.0);
        assert_eq!(max, -42.0);
        assert_eq!(mean, -42.0);
        assert_eq!(std, 0.0);
    }

    #[test]
    fn test_compute_stats_single_zero() {
        let data = [0.0];
        let (min, max, mean, std) = compute_stats(&data);
        assert_eq!(min, 0.0);
        assert_eq!(max, 0.0);
        assert_eq!(mean, 0.0);
        assert_eq!(std, 0.0);
    }

    #[test]
    fn test_compute_stats_all_zeros() {
        let data = [0.0, 0.0, 0.0, 0.0, 0.0];
        let (min, max, mean, std) = compute_stats(&data);
        assert_eq!(min, 0.0);
        assert_eq!(max, 0.0);
        assert_eq!(mean, 0.0);
        assert_eq!(std, 0.0);
    }

    #[test]
    fn test_compute_stats_ascending() {
        let data: Vec<f32> = (1..=100).map(|i| i as f32).collect();
        let (min, max, mean, std) = compute_stats(&data);
        assert_eq!(min, 1.0);
        assert_eq!(max, 100.0);
        assert!((mean - 50.5).abs() < 0.01);
        // std of uniform 1..=100 is ~28.87
        assert!((std - 28.87).abs() < 0.1);
    }

    #[test]
    fn test_compute_stats_descending() {
        let data: Vec<f32> = (1..=100).rev().map(|i| i as f32).collect();
        let (min, max, mean, std) = compute_stats(&data);
        assert_eq!(min, 1.0);
        assert_eq!(max, 100.0);
        assert!((mean - 50.5).abs() < 0.01);
        assert!((std - 28.87).abs() < 0.1);
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
        let test_cases: Vec<Vec<f32>> = vec![
            vec![1.0, 1.0, 1.0],
            vec![-1.0, 1.0],
            vec![0.0],
            vec![100.0, -100.0],
        ];
        for data in &test_cases {
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

    #[test]
    fn test_encoder_layer_counting_zero_layers() {
        let tensor_names: Vec<String> = vec!["output.weight".to_string()];
        let n_layers = tensor_names
            .iter()
            .filter(|n| n.starts_with("encoder.layers."))
            .filter_map(|n| {
                n.strip_prefix("encoder.layers.")
                    .and_then(|s| s.split('.').next())
                    .and_then(|s| s.parse::<usize>().ok())
            })
            .max()
            .map_or(0, |n| n + 1);
        assert_eq!(n_layers, 0);
    }

    #[test]
    fn test_encoder_layer_counting_single_layer() {
        let tensor_names = vec![
            "encoder.layers.0.self_attn.q_proj.weight".to_string(),
            "encoder.layers.0.self_attn.k_proj.weight".to_string(),
        ];
        let n_layers = tensor_names
            .iter()
            .filter(|n| n.starts_with("encoder.layers."))
            .filter_map(|n| {
                n.strip_prefix("encoder.layers.")
                    .and_then(|s| s.split('.').next())
                    .and_then(|s| s.parse::<usize>().ok())
            })
            .max()
            .map_or(0, |n| n + 1);
        assert_eq!(n_layers, 1);
    }

    #[test]
    fn test_encoder_layer_counting_multiple_layers() {
        let tensor_names = vec![
            "encoder.layers.0.self_attn.weight".to_string(),
            "encoder.layers.1.self_attn.weight".to_string(),
            "encoder.layers.2.self_attn.weight".to_string(),
            "encoder.layers.3.self_attn.weight".to_string(),
        ];
        let n_layers = tensor_names
            .iter()
            .filter(|n| n.starts_with("encoder.layers."))
            .filter_map(|n| {
                n.strip_prefix("encoder.layers.")
                    .and_then(|s| s.split('.').next())
                    .and_then(|s| s.parse::<usize>().ok())
            })
            .max()
            .map_or(0, |n| n + 1);
        assert_eq!(n_layers, 4);
    }

    #[test]
    fn test_encoder_layer_counting_non_contiguous() {
        // Layers 0, 5 -> max=5, n_layers=6
        let tensor_names = vec![
            "encoder.layers.0.weight".to_string(),
            "encoder.layers.5.weight".to_string(),
        ];
        let n_layers = tensor_names
            .iter()
            .filter(|n| n.starts_with("encoder.layers."))
            .filter_map(|n| {
                n.strip_prefix("encoder.layers.")
                    .and_then(|s| s.split('.').next())
                    .and_then(|s| s.parse::<usize>().ok())
            })
            .max()
            .map_or(0, |n| n + 1);
        assert_eq!(n_layers, 6);
    }

    #[test]
    fn test_decoder_layer_counting_zero_layers() {
        let tensor_names: Vec<String> = vec!["output.weight".to_string()];
        let n_layers = tensor_names
            .iter()
            .filter(|n| n.starts_with("decoder.layers."))
            .filter_map(|n| {
                n.strip_prefix("decoder.layers.")
                    .and_then(|s| s.split('.').next())
                    .and_then(|s| s.parse::<usize>().ok())
            })
            .max()
            .map_or(0, |n| n + 1);
        assert_eq!(n_layers, 0);
    }

    #[test]
    fn test_decoder_layer_counting_multiple_layers() {
        let tensor_names = vec![
            "decoder.layers.0.self_attn.weight".to_string(),
            "decoder.layers.1.encoder_attn.weight".to_string(),
            "decoder.layers.2.ffn.weight".to_string(),
            "decoder.layers.3.self_attn.weight".to_string(),
            "decoder.layers.3.ffn.weight".to_string(),
        ];
        let n_layers = tensor_names
            .iter()
            .filter(|n| n.starts_with("decoder.layers."))
            .filter_map(|n| {
                n.strip_prefix("decoder.layers.")
                    .and_then(|s| s.split('.').next())
                    .and_then(|s| s.parse::<usize>().ok())
            })
            .max()
            .map_or(0, |n| n + 1);
        assert_eq!(n_layers, 4);
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
        let enc_layers = tensor_names
            .iter()
            .filter(|n| n.starts_with("encoder.layers."))
            .filter_map(|n| {
                n.strip_prefix("encoder.layers.")
                    .and_then(|s| s.split('.').next())
                    .and_then(|s| s.parse::<usize>().ok())
            })
            .max()
            .map_or(0, |n| n + 1);
        let dec_layers = tensor_names
            .iter()
            .filter(|n| n.starts_with("decoder.layers."))
            .filter_map(|n| {
                n.strip_prefix("decoder.layers.")
                    .and_then(|s| s.split('.').next())
                    .and_then(|s| s.parse::<usize>().ok())
            })
            .max()
            .map_or(0, |n| n + 1);
        assert_eq!(enc_layers, 2);
        assert_eq!(dec_layers, 3);
    }
