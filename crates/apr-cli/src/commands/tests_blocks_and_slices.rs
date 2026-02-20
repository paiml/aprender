
    #[test]
    fn test_print_q6k_blocks_synthetic() {
        let bytes = vec![0u8; Q6K_BLOCK_SIZE * 2];
        print_q6k_blocks(&bytes, 0, 1);
    }

    #[test]
    fn test_print_q8_0_blocks_synthetic() {
        let bytes = vec![0u8; Q8_0_BLOCK_SIZE * 2];
        print_q8_0_blocks(&bytes, 0, 1);
    }

    #[test]
    fn test_print_blocks_exceeds_bounds() {
        // Should print warning, not panic
        let bytes = vec![0u8; 10];
        print_q4k_blocks(&bytes, 0, 1);
    }

    // ========================================================================
    // --slice: parse_slice tests
    // ========================================================================

    #[test]
    fn test_parse_slice_basic() {
        let (start, end) = parse_slice("0:3").expect("valid slice");
        assert_eq!(start, 0);
        assert_eq!(end, 3);
    }

    #[test]
    fn test_parse_slice_range() {
        let (start, end) = parse_slice("5:10").expect("valid slice");
        assert_eq!(start, 5);
        assert_eq!(end, 10);
    }

    #[test]
    fn test_parse_slice_large_range() {
        let (start, end) = parse_slice("100:200").expect("valid slice");
        assert_eq!(start, 100);
        assert_eq!(end, 200);
    }

    #[test]
    fn test_parse_slice_invalid_format_no_colon() {
        assert!(parse_slice("03").is_err());
    }

    #[test]
    fn test_parse_slice_invalid_format_too_many_colons() {
        assert!(parse_slice("0:3:5").is_err());
    }

    #[test]
    fn test_parse_slice_start_equals_end() {
        assert!(parse_slice("3:3").is_err());
    }

    #[test]
    fn test_parse_slice_start_greater_than_end() {
        assert!(parse_slice("5:3").is_err());
    }

    #[test]
    fn test_parse_slice_non_numeric() {
        assert!(parse_slice("a:b").is_err());
    }

    // ========================================================================
    // --slice: SafeTensors slice integration tests
    // ========================================================================

    /// Create a minimal SafeTensors file with a single F32 tensor.
    fn create_safetensors_f32(name: &str, values: &[f32]) -> Vec<u8> {
        let data_bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let header = format!(
            r#"{{"{name}":{{"dtype":"F32","shape":[{len}],"data_offsets":[0,{data_len}]}}}}"#,
            len = values.len(),
            data_len = data_bytes.len(),
        );
        let header_bytes = header.as_bytes();
        let header_len = header_bytes.len() as u64;
        let mut file_bytes = Vec::new();
        file_bytes.extend_from_slice(&header_len.to_le_bytes());
        file_bytes.extend_from_slice(header_bytes);
        file_bytes.extend_from_slice(&data_bytes);
        file_bytes
    }

    #[test]
    fn test_slice_safetensors_json_output() {
        let values = [1.0_f32, 2.0, 3.0, 4.0, 5.0];
        let file_bytes = create_safetensors_f32("test_tensor", &values);

        let mut tmp = NamedTempFile::with_suffix(".safetensors").expect("create temp");
        tmp.write_all(&file_bytes).expect("write");

        let opts = HexOptions {
            file: tmp.path().to_path_buf(),
            tensor: Some("test_tensor".to_string()),
            slice: Some("0:3".to_string()),
            json: true,
            ..HexOptions::default()
        };
        let result = run(&opts);
        assert!(result.is_ok());
    }

    #[test]
    fn test_slice_safetensors_text_output() {
        let values = [10.0_f32, 20.0, 30.0];
        let file_bytes = create_safetensors_f32("weights", &values);

        let mut tmp = NamedTempFile::with_suffix(".safetensors").expect("create temp");
        tmp.write_all(&file_bytes).expect("write");

        let opts = HexOptions {
            file: tmp.path().to_path_buf(),
            tensor: Some("weights".to_string()),
            slice: Some("1:3".to_string()),
            json: false,
            ..HexOptions::default()
        };
        let result = run(&opts);
        assert!(result.is_ok());
    }

    #[test]
    fn test_slice_safetensors_out_of_bounds() {
        let values = [1.0_f32, 2.0, 3.0];
        let file_bytes = create_safetensors_f32("small", &values);

        let mut tmp = NamedTempFile::with_suffix(".safetensors").expect("create temp");
        tmp.write_all(&file_bytes).expect("write");

        let opts = HexOptions {
            file: tmp.path().to_path_buf(),
            tensor: Some("small".to_string()),
            slice: Some("0:10".to_string()),
            json: true,
            ..HexOptions::default()
        };
        let result = run(&opts);
        assert!(result.is_err());
    }

    #[test]
    fn test_slice_safetensors_tensor_not_found() {
        let values = [1.0_f32];
        let file_bytes = create_safetensors_f32("real", &values);

        let mut tmp = NamedTempFile::with_suffix(".safetensors").expect("create temp");
        tmp.write_all(&file_bytes).expect("write");

        let opts = HexOptions {
            file: tmp.path().to_path_buf(),
            tensor: Some("nonexistent".to_string()),
            slice: Some("0:1".to_string()),
            json: true,
            ..HexOptions::default()
        };
        let result = run(&opts);
        assert!(result.is_err());
    }

    #[test]
    fn test_slice_safetensors_correct_values() {
        // Verify that decode_st_slice extracts the right values
        let values = [1.5_f32, 2.5, 3.5, 4.5, 5.5];
        let data_bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let result = decode_st_slice(&data_bytes, "F32", 1, 4).expect("decode");
        assert_eq!(result.len(), 3);
        assert!((result[0] - 2.5).abs() < 1e-6);
        assert!((result[1] - 3.5).abs() < 1e-6);
        assert!((result[2] - 4.5).abs() < 1e-6);
    }

    #[test]
    fn test_slice_decode_f16() {
        // f16 value for 1.0 is 0x3C00
        let data_bytes: Vec<u8> = vec![0x00, 0x3C, 0x00, 0x40, 0x00, 0x44];
        // 0x3C00=1.0, 0x4000=2.0, 0x4400=4.0
        let result = decode_st_slice(&data_bytes, "F16", 0, 3).expect("decode");
        assert_eq!(result.len(), 3);
        assert!((result[0] - 1.0).abs() < 1e-3);
        assert!((result[1] - 2.0).abs() < 1e-3);
        assert!((result[2] - 4.0).abs() < 1e-3);
    }

    #[test]
    fn test_slice_decode_unsupported_dtype() {
        let data_bytes = vec![0u8; 16];
        let result = decode_st_slice(&data_bytes, "I32", 0, 4);
        assert!(result.is_err());
    }

    #[test]
    fn test_hex_options_slice_default_is_none() {
        let opts = HexOptions::default();
        assert!(opts.slice.is_none());
    }
