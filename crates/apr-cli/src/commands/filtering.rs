
    // ========================================================================
    // Cross-Attention Layer Filtering Logic
    // ========================================================================

    /// Filter tensor names to find cross-attention q_proj weights,
    /// optionally constrained by a layer filter substring.
    fn filter_cross_attn_q_proj<'a>(
        tensor_names: &'a [String],
        layer_filter: Option<&str>,
    ) -> Vec<&'a String> {
        tensor_names
            .iter()
            .filter(|n| n.contains("encoder_attn") || n.contains("cross_attn"))
            .filter(|n| n.contains("q_proj.weight"))
            .filter(|n| layer_filter.map_or(true, |f| n.contains(f)))
            .collect()
    }

    #[test]
    fn test_cross_attn_layer_detection() {
        // Data-driven: (tensor_names, layer_filter, expected_count)
        let cases: Vec<(Vec<String>, Option<&str>, usize)> = vec![
            // encoder_attn variant: 2 q_proj matches
            (vec![
                "decoder.layers.0.encoder_attn.q_proj.weight".to_string(),
                "decoder.layers.0.encoder_attn.k_proj.weight".to_string(),
                "decoder.layers.1.encoder_attn.q_proj.weight".to_string(),
            ], None, 2),
            // cross_attn variant: 1 q_proj match
            (vec![
                "decoder.layers.0.cross_attn.q_proj.weight".to_string(),
                "decoder.layers.0.cross_attn.k_proj.weight".to_string(),
            ], None, 1),
            // No cross/encoder_attn -> empty
            (vec![
                "decoder.layers.0.self_attn.q_proj.weight".to_string(),
                "decoder.layers.0.ffn.fc1.weight".to_string(),
            ], None, 0),
            // Filter to layers.1 -> 1 match
            (vec![
                "decoder.layers.0.encoder_attn.q_proj.weight".to_string(),
                "decoder.layers.1.encoder_attn.q_proj.weight".to_string(),
                "decoder.layers.2.encoder_attn.q_proj.weight".to_string(),
            ], Some("layers.1"), 1),
            // Filter None -> matches all (2)
            (vec![
                "decoder.layers.0.encoder_attn.q_proj.weight".to_string(),
                "decoder.layers.1.encoder_attn.q_proj.weight".to_string(),
            ], None, 2),
            // Filter layers.99 -> no match
            (vec![
                "decoder.layers.0.encoder_attn.q_proj.weight".to_string(),
                "decoder.layers.1.encoder_attn.q_proj.weight".to_string(),
            ], Some("layers.99"), 0),
        ];
        for (names, filter, expected) in &cases {
            let result = filter_cross_attn_q_proj(names, *filter);
            assert_eq!(
                result.len(), *expected,
                "Failed for filter={filter:?}, names={names:?}"
            );
        }
    }

    #[test]
    fn test_cross_attn_layer_filter_applied_content() {
        let tensor_names = vec![
            "decoder.layers.0.encoder_attn.q_proj.weight".to_string(),
            "decoder.layers.1.encoder_attn.q_proj.weight".to_string(),
            "decoder.layers.2.encoder_attn.q_proj.weight".to_string(),
        ];
        let filtered = filter_cross_attn_q_proj(&tensor_names, Some("layers.1"));
        assert_eq!(filtered.len(), 1);
        assert!(filtered[0].contains("layers.1"));
    }

    // ========================================================================
    // Q weight name prefix stripping (cross-attention flow)
    // ========================================================================

    #[test]
    fn test_strip_q_proj_suffix() {
        let name = "decoder.layers.0.encoder_attn.q_proj.weight";
        let prefix = name.strip_suffix(".q_proj.weight").unwrap_or(name);
        assert_eq!(prefix, "decoder.layers.0.encoder_attn");
    }

    #[test]
    fn test_strip_q_proj_suffix_no_match() {
        let name = "decoder.layers.0.encoder_attn.k_proj.weight";
        let prefix = name.strip_suffix(".q_proj.weight").unwrap_or(name);
        // No match -> returns the full name
        assert_eq!(prefix, name);
    }

    // ========================================================================
    // Conv1 detection in encoder block
    // ========================================================================

    #[test]
    fn test_conv1_detection_present() {
        let tensor_names = vec![
            "encoder.conv1.weight".to_string(),
            "encoder.conv2.weight".to_string(),
            "encoder.positional_embedding".to_string(),
        ];
        let has_conv1 = tensor_names.iter().any(|n| n.contains("conv1"));
        assert!(has_conv1);
    }

    #[test]
    fn test_conv1_detection_absent() {
        let tensor_names = vec![
            "encoder.layers.0.self_attn.weight".to_string(),
            "encoder.layers.0.ffn.weight".to_string(),
        ];
        let has_conv1 = tensor_names.iter().any(|n| n.contains("conv1"));
        assert!(!has_conv1);
    }

    // ========================================================================
    // run() Error Path Tests
    // ========================================================================

    #[test]
    fn test_run_nonexistent_path_specific_error_variant() {
        let result = run(
            Path::new("/tmp/definitely_does_not_exist_xyz123.apr"),
            None,
            FlowComponent::Full,
            false,
            false,
        );
        match result {
            Err(CliError::FileNotFound(p)) => {
                assert_eq!(p, Path::new("/tmp/definitely_does_not_exist_xyz123.apr"));
            }
            other => panic!("Expected FileNotFound, got: {other:?}"),
        }
    }

    #[test]
    fn test_run_empty_apr_file() {
        let file = NamedTempFile::with_suffix(".apr").expect("create temp file");
        // Empty file should fail with InvalidFormat
        let result = run(file.path(), None, FlowComponent::Full, false, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_run_non_apr_extension() {
        let dir = tempdir().expect("create temp dir");
        let path = dir.path().join("model.gguf");
        std::fs::write(&path, b"some data").expect("write");
        // This file exists but flow command requires APR format
        let result = run(&path, None, FlowComponent::Full, false, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_run_all_components_fail_on_invalid() {
        let mut file = NamedTempFile::with_suffix(".apr").expect("create temp file");
        file.write_all(b"invalid").expect("write");

        // All component variants should fail with invalid data
        let components = [
            FlowComponent::Full,
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

    #[test]
    fn test_run_verbose_with_layer_filter() {
        let mut file = NamedTempFile::with_suffix(".apr").expect("create temp file");
        file.write_all(b"not valid apr").expect("write");

        let result = run(
            file.path(),
            Some("decoder.layers.0"),
            FlowComponent::CrossAttention,
            true,
            false,
        );
        assert!(result.is_err());
    }

    // ========================================================================
    // Printing functions (should not panic)
    // ========================================================================

    #[test]
    fn test_print_encoder_block_no_conv1_no_layers() {
        let tensor_names: Vec<String> = vec![];
        // Should not panic with empty tensor names
        print_encoder_block(&tensor_names, false);
    }

    #[test]
    fn test_print_encoder_block_with_conv1_and_layers() {
        let tensor_names = vec![
            "encoder.conv1.weight".to_string(),
            "encoder.conv2.weight".to_string(),
            "encoder.layers.0.self_attn.weight".to_string(),
            "encoder.layers.1.self_attn.weight".to_string(),
        ];
        // Should not panic
        print_encoder_block(&tensor_names, false);
    }

    #[test]
    fn test_print_encoder_block_with_conv1_no_layers() {
        let tensor_names = vec!["encoder.conv1.weight".to_string()];
        // Has conv1 but no "encoder.layers." tensors -> n_layers = 0
        print_encoder_block(&tensor_names, false);
    }

    #[test]
    fn test_print_decoder_block_no_layers() {
        let tensor_names: Vec<String> = vec![];
        print_decoder_block(&tensor_names, false);
    }

    #[test]
    fn test_print_decoder_block_with_layers() {
        let tensor_names = vec![
            "decoder.layers.0.self_attn.weight".to_string(),
            "decoder.layers.1.self_attn.weight".to_string(),
            "decoder.layers.2.self_attn.weight".to_string(),
        ];
        print_decoder_block(&tensor_names, false);
    }

    #[test]
    fn test_print_decoder_block_many_layers() {
        let tensor_names: Vec<String> = (0..32)
            .map(|i| format!("decoder.layers.{i}.self_attn.weight"))
            .collect();
        print_decoder_block(&tensor_names, false);
    }
