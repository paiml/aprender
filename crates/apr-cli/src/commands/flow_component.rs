
    // ========================================================================
    // FlowComponent Tests
    // ========================================================================

    #[test]
    fn test_flow_component_from_str_full() {
        assert_eq!(
            "full".parse::<FlowComponent>().unwrap(),
            FlowComponent::Full
        );
        assert_eq!("all".parse::<FlowComponent>().unwrap(), FlowComponent::Full);
    }

    #[test]
    fn test_flow_component_from_str_encoder() {
        assert_eq!(
            "encoder".parse::<FlowComponent>().unwrap(),
            FlowComponent::Encoder
        );
        assert_eq!(
            "enc".parse::<FlowComponent>().unwrap(),
            FlowComponent::Encoder
        );
        assert_eq!(
            "ENCODER".parse::<FlowComponent>().unwrap(),
            FlowComponent::Encoder
        );
    }

    #[test]
    fn test_flow_component_from_str_decoder() {
        assert_eq!(
            "decoder".parse::<FlowComponent>().unwrap(),
            FlowComponent::Decoder
        );
        assert_eq!(
            "dec".parse::<FlowComponent>().unwrap(),
            FlowComponent::Decoder
        );
    }

    #[test]
    fn test_flow_component_from_str_self_attention() {
        assert_eq!(
            "self_attn".parse::<FlowComponent>().unwrap(),
            FlowComponent::SelfAttention
        );
        assert_eq!(
            "self-attn".parse::<FlowComponent>().unwrap(),
            FlowComponent::SelfAttention
        );
        assert_eq!(
            "selfattn".parse::<FlowComponent>().unwrap(),
            FlowComponent::SelfAttention
        );
    }

    #[test]
    fn test_flow_component_from_str_cross_attention() {
        assert_eq!(
            "cross_attn".parse::<FlowComponent>().unwrap(),
            FlowComponent::CrossAttention
        );
        assert_eq!(
            "cross-attn".parse::<FlowComponent>().unwrap(),
            FlowComponent::CrossAttention
        );
        assert_eq!(
            "crossattn".parse::<FlowComponent>().unwrap(),
            FlowComponent::CrossAttention
        );
        assert_eq!(
            "encoder_attn".parse::<FlowComponent>().unwrap(),
            FlowComponent::CrossAttention
        );
    }

    #[test]
    fn test_flow_component_from_str_ffn() {
        assert_eq!("ffn".parse::<FlowComponent>().unwrap(), FlowComponent::Ffn);
        assert_eq!("mlp".parse::<FlowComponent>().unwrap(), FlowComponent::Ffn);
        assert_eq!(
            "feedforward".parse::<FlowComponent>().unwrap(),
            FlowComponent::Ffn
        );
    }

    #[test]
    fn test_flow_component_from_str_invalid() {
        assert!("unknown".parse::<FlowComponent>().is_err());
        assert!("invalid".parse::<FlowComponent>().is_err());
        assert!("".parse::<FlowComponent>().is_err());
    }

    #[test]
    fn test_flow_component_debug() {
        // Verify Debug trait is derived
        let comp = FlowComponent::Full;
        let debug = format!("{comp:?}");
        assert!(debug.contains("Full"));
    }

    #[test]
    fn test_flow_component_clone() {
        let comp = FlowComponent::Encoder;
        let cloned = comp.clone();
        assert_eq!(comp, cloned);
    }

    #[test]
    fn test_flow_component_eq() {
        assert_eq!(FlowComponent::Full, FlowComponent::Full);
        assert_ne!(FlowComponent::Full, FlowComponent::Encoder);
    }

    // ========================================================================
    // detect_architecture Tests
    // ========================================================================

    #[test]
    fn test_detect_architecture_encoder_decoder() {
        let names = vec![
            "encoder.layers.0.self_attn.q_proj.weight".to_string(),
            "decoder.layers.0.self_attn.q_proj.weight".to_string(),
            "decoder.layers.0.encoder_attn.q_proj.weight".to_string(),
        ];
        assert_eq!(detect_architecture(&names), "encoder-decoder (Whisper/T5)");
    }

    #[test]
    fn test_detect_architecture_encoder_only() {
        let names = vec![
            "encoder.layers.0.self_attn.q_proj.weight".to_string(),
            "encoder.layers.0.ffn.fc1.weight".to_string(),
        ];
        assert_eq!(detect_architecture(&names), "encoder-only (BERT)");
    }

    #[test]
    fn test_detect_architecture_decoder_only() {
        let names = vec![
            "decoder.layers.0.self_attn.q_proj.weight".to_string(),
            "decoder.layers.0.ffn.fc1.weight".to_string(),
        ];
        assert_eq!(detect_architecture(&names), "decoder-only (GPT/LLaMA)");
    }

    #[test]
    fn test_detect_architecture_unknown() {
        let names = vec!["weight".to_string(), "bias".to_string()];
        assert_eq!(detect_architecture(&names), "unknown");
    }

    #[test]
    fn test_detect_architecture_empty() {
        let names: Vec<String> = vec![];
        assert_eq!(detect_architecture(&names), "unknown");
    }

    #[test]
    fn test_detect_architecture_cross_attn_variant() {
        let names = vec![
            "encoder.layers.0.self_attn.weight".to_string(),
            "decoder.layers.0.cross_attn.weight".to_string(),
        ];
        assert_eq!(detect_architecture(&names), "encoder-decoder (Whisper/T5)");
    }

    // ========================================================================
    // compute_stats Tests
    // ========================================================================

    #[test]
    fn test_compute_stats_empty() {
        let (min, max, mean, std) = compute_stats(&[]);
        assert_eq!(min, 0.0);
        assert_eq!(max, 0.0);
        assert_eq!(mean, 0.0);
        assert_eq!(std, 0.0);
    }

    #[test]
    fn test_compute_stats_single_value() {
        let (min, max, mean, std) = compute_stats(&[5.0]);
        assert_eq!(min, 5.0);
        assert_eq!(max, 5.0);
        assert_eq!(mean, 5.0);
        assert_eq!(std, 0.0);
    }

    #[test]
    fn test_compute_stats_simple_range() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let (min, max, mean, std) = compute_stats(&data);
        assert_eq!(min, 1.0);
        assert_eq!(max, 5.0);
        assert_eq!(mean, 3.0);
        // std for [1,2,3,4,5] is sqrt(2) ≈ 1.414
        assert!((std - 1.4142).abs() < 0.01);
    }

    #[test]
    fn test_compute_stats_negative_values() {
        let data = [-5.0, -2.0, 0.0, 2.0, 5.0];
        let (min, max, mean, _std) = compute_stats(&data);
        assert_eq!(min, -5.0);
        assert_eq!(max, 5.0);
        assert_eq!(mean, 0.0);
    }

    #[test]
    fn test_compute_stats_all_same() {
        let data = [7.0, 7.0, 7.0, 7.0];
        let (min, max, mean, std) = compute_stats(&data);
        assert_eq!(min, 7.0);
        assert_eq!(max, 7.0);
        assert_eq!(mean, 7.0);
        assert_eq!(std, 0.0);
    }

    // ========================================================================
    // Run Command Tests
    // ========================================================================

    #[test]
    fn test_run_file_not_found() {
        let result = run(
            Path::new("/nonexistent/model.apr"),
            None,
            FlowComponent::Full,
            false,
            false,
        );
        assert!(result.is_err());
        match result {
            Err(CliError::FileNotFound(_)) => {}
            _ => panic!("Expected FileNotFound error"),
        }
    }

    #[test]
    fn test_run_invalid_apr_file() {
        let mut file = NamedTempFile::with_suffix(".apr").expect("create temp file");
        file.write_all(b"not a valid apr file").expect("write");

        let result = run(file.path(), None, FlowComponent::Full, false, false);
        // Should fail because it's not a valid APR
        assert!(result.is_err());
    }

    #[test]
    fn test_run_is_directory() {
        let dir = tempdir().expect("create temp dir");
        let result = run(dir.path(), None, FlowComponent::Full, false, false);
        // Should fail because it's a directory
        assert!(result.is_err());
    }

    #[test]
    fn test_run_with_layer_filter() {
        let mut file = NamedTempFile::with_suffix(".apr").expect("create temp file");
        file.write_all(b"not a valid apr file").expect("write");

        let result = run(
            file.path(),
            Some("encoder.layers.0"),
            FlowComponent::Full,
            false,
            false,
        );
        // Should fail (invalid file) but tests the filter path
        assert!(result.is_err());
    }

    #[test]
    fn test_run_with_verbose() {
        let mut file = NamedTempFile::with_suffix(".apr").expect("create temp file");
        file.write_all(b"not a valid apr file").expect("write");

        let result = run(file.path(), None, FlowComponent::Full, true, false);
        // Should fail (invalid file) but tests verbose path
        assert!(result.is_err());
    }

    #[test]
    fn test_run_all_components_on_invalid_file() {
        // Data-driven: each component variant should fail on invalid APR data
        let mut file = NamedTempFile::with_suffix(".apr").expect("create temp file");
        file.write_all(b"not a valid apr file").expect("write");

        let components = [
            FlowComponent::Encoder,
            FlowComponent::Decoder,
            FlowComponent::SelfAttention,
            FlowComponent::CrossAttention,
            FlowComponent::Ffn,
        ];
        for comp in &components {
            let result = run(file.path(), None, *comp, false, false);
            assert!(result.is_err(), "Expected error for component {comp:?}");
        }
    }

    // ========================================================================
    // FlowComponent FromStr Additional Coverage
    // ========================================================================

    #[test]
    fn test_flow_component_from_str_mixed_case_aliases() {
        // Data-driven: (input_alias, expected_variant)
        let cases: &[(&str, FlowComponent)] = &[
            // Full aliases
            ("FULL", FlowComponent::Full),
            ("Full", FlowComponent::Full),
            ("ALL", FlowComponent::Full),
            ("All", FlowComponent::Full),
            // Encoder aliases
            ("ENC", FlowComponent::Encoder),
            ("Enc", FlowComponent::Encoder),
            ("Encoder", FlowComponent::Encoder),
            // Decoder aliases
            ("DEC", FlowComponent::Decoder),
            ("Dec", FlowComponent::Decoder),
            ("Decoder", FlowComponent::Decoder),
            ("DECODER", FlowComponent::Decoder),
            // Self-attention aliases
            ("SELF_ATTN", FlowComponent::SelfAttention),
            ("Self-Attn", FlowComponent::SelfAttention),
            ("SELFATTN", FlowComponent::SelfAttention),
            ("Self_Attn", FlowComponent::SelfAttention),
            // Cross-attention aliases
            ("CROSS_ATTN", FlowComponent::CrossAttention),
            ("CROSS-ATTN", FlowComponent::CrossAttention),
            ("CROSSATTN", FlowComponent::CrossAttention),
            ("ENCODER_ATTN", FlowComponent::CrossAttention),
            ("Encoder_Attn", FlowComponent::CrossAttention),
        ];
        for (input, expected) in cases {
            assert_eq!(
                input.parse::<FlowComponent>().unwrap_or_else(|e| panic!("{input}: {e}")),
                *expected,
                "Alias {input:?} should parse correctly"
            );
        }
    }
