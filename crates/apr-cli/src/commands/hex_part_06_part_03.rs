
    #[test]
    fn test_print_tensor_anomalies_nan_max() {
        print_tensor_anomalies(0.0, f32::NAN, 0.5, 0.3);
    }

    #[test]
    fn test_print_tensor_anomalies_nan_mean() {
        print_tensor_anomalies(0.0, 1.0, f32::NAN, 0.3);
    }

    #[test]
    fn test_print_tensor_anomalies_all_nan() {
        print_tensor_anomalies(f32::NAN, f32::NAN, f32::NAN, f32::NAN);
    }

    #[test]
    fn test_print_tensor_anomalies_infinite_min() {
        print_tensor_anomalies(f32::NEG_INFINITY, 1.0, 0.5, 0.3);
    }

    #[test]
    fn test_print_tensor_anomalies_infinite_max() {
        print_tensor_anomalies(0.0, f32::INFINITY, 0.5, 0.3);
    }

    #[test]
    fn test_print_tensor_anomalies_low_variance() {
        print_tensor_anomalies(0.0, 1.0, 0.5, 1e-12);
    }

    #[test]
    fn test_print_tensor_anomalies_zero_variance() {
        print_tensor_anomalies(5.0, 5.0, 5.0, 0.0);
    }

    #[test]
    fn test_print_tensor_anomalies_exactly_threshold() {
        print_tensor_anomalies(0.0, 1.0, 0.5, 1e-10);
    }

    #[test]
    fn test_print_tensor_anomalies_above_threshold() {
        print_tensor_anomalies(0.0, 1.0, 0.5, 2e-10);
    }

    #[test]
    fn test_print_tensor_anomalies_nan_and_infinite_together() {
        print_tensor_anomalies(f32::NAN, f32::INFINITY, f32::NAN, 0.0);
    }

    // ========================================================================
    // print_tensor_header tests (preserved)
    // ========================================================================

    fn make_entry(
        name: &str,
        shape: Vec<usize>,
        dtype: TensorDType,
        offset: u64,
        size: u64,
    ) -> TensorIndexEntry {
        TensorIndexEntry::new(name, dtype, shape, offset, size)
    }

    #[test]
    fn test_print_tensor_header_basic() {
        let entry = make_entry(
            "model.layers.0.weight",
            vec![768, 3072],
            TensorDType::F32,
            0,
            (768 * 3072 * 4) as u64,
        );
        print_tensor_header_v2(&entry);
    }

    #[test]
    fn test_print_tensor_header_empty_shape() {
        let entry = make_entry("scalar_param", vec![], TensorDType::F32, 0, 4);
        print_tensor_header_v2(&entry);
    }

    #[test]
    fn test_print_tensor_header_single_dim() {
        let entry = make_entry("bias", vec![512], TensorDType::F32, 1024, 512 * 4);
        print_tensor_header_v2(&entry);
    }

    #[test]
    fn test_print_tensor_header_large_offset() {
        let entry = make_entry(
            "lm_head.weight",
            vec![32000, 4096],
            TensorDType::F16,
            0xFFFF_FFFF,
            32000 * 4096 * 2,
        );
        print_tensor_header_v2(&entry);
    }

    #[test]
    fn test_print_tensor_header_zero_size() {
        let entry = make_entry("empty", vec![0], TensorDType::F32, 0, 0);
        print_tensor_header_v2(&entry);
    }

    #[test]
    fn test_print_tensor_header_3d_shape() {
        let entry = make_entry(
            "conv.weight",
            vec![64, 3, 3],
            TensorDType::F32,
            512,
            (64 * 3 * 3 * 4) as u64,
        );
        print_tensor_header_v2(&entry);
    }

    // ========================================================================
    // print_hex_row tests (preserved)
    // ========================================================================

    #[test]
    fn test_print_hex_row_full_chunk() {
        let vals = [1.0_f32, 2.0, 3.0, 4.0];
        let refs: Vec<&f32> = vals.iter().collect();
        print_hex_row(&refs, 0);
    }

    #[test]
    fn test_print_hex_row_partial_chunk() {
        let vals = [1.0_f32, 2.0];
        let refs: Vec<&f32> = vals.iter().collect();
        print_hex_row(&refs, 16);
    }

    #[test]
    fn test_print_hex_row_single_value() {
        let vals = [42.0_f32];
        let refs: Vec<&f32> = vals.iter().collect();
        print_hex_row(&refs, 0);
    }

    #[test]
    fn test_print_hex_row_three_values() {
        let vals = [0.0_f32, -1.0, f32::MAX];
        let refs: Vec<&f32> = vals.iter().collect();
        print_hex_row(&refs, 48);
    }

    #[test]
    fn test_print_hex_row_special_values() {
        let vals = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.0];
        let refs: Vec<&f32> = vals.iter().collect();
        print_hex_row(&refs, 0);
    }

    #[test]
    fn test_print_hex_row_large_offset() {
        let vals = [1.0_f32];
        let refs: Vec<&f32> = vals.iter().collect();
        print_hex_row(&refs, 0xDEAD_BEEF);
    }

    #[test]
    fn test_print_hex_row_negative_values() {
        let vals = [-0.5_f32, -100.0, -1e-6, -f32::MAX];
        let refs: Vec<&f32> = vals.iter().collect();
        print_hex_row(&refs, 32);
    }

    // ========================================================================
    // print_hex_dump tests (preserved)
    // ========================================================================

    #[test]
    fn test_print_hex_dump_empty_data() {
        print_hex_dump(&[], 100);
    }

    #[test]
    fn test_print_hex_dump_data_smaller_than_limit() {
        let data = [1.0_f32, 2.0, 3.0];
        print_hex_dump(&data, 100);
    }

    #[test]
    fn test_print_hex_dump_data_equal_to_limit() {
        let data = [1.0_f32, 2.0, 3.0, 4.0];
        print_hex_dump(&data, 4);
    }

    #[test]
    fn test_print_hex_dump_data_larger_than_limit() {
        let data: Vec<f32> = (0..100).map(|i| i as f32).collect();
        print_hex_dump(&data, 10);
    }

    #[test]
    fn test_print_hex_dump_limit_zero() {
        let data = [1.0_f32, 2.0, 3.0];
        print_hex_dump(&data, 0);
    }

    #[test]
    fn test_print_hex_dump_single_element() {
        print_hex_dump(&[42.0], 1);
    }

    #[test]
    fn test_print_hex_dump_exactly_one_row() {
        let data = [1.0_f32, 2.0, 3.0, 4.0];
        print_hex_dump(&data, 4);
    }

    #[test]
    fn test_print_hex_dump_two_rows() {
        let data = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        print_hex_dump(&data, 8);
    }

    #[test]
    fn test_print_hex_dump_partial_last_row() {
        let data = [1.0_f32, 2.0, 3.0, 4.0, 5.0];
        print_hex_dump(&data, 5);
    }

    // ========================================================================
    // print_tensor_stats tests (preserved)
    // ========================================================================

    #[test]
    fn test_print_tensor_stats_normal_data() {
        print_tensor_stats(&[1.0_f32, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_print_tensor_stats_empty_data() {
        print_tensor_stats(&[]);
    }

    #[test]
    fn test_print_tensor_stats_single_value() {
        print_tensor_stats(&[42.0]);
    }

    #[test]
    fn test_print_tensor_stats_with_nan() {
        print_tensor_stats(&[1.0, f32::NAN, 3.0]);
    }

    #[test]
    fn test_print_tensor_stats_all_same() {
        let data = [3.14_f32; 100];
        print_tensor_stats(&data);
    }

    // list_tensors_v2 tests require a real AprV2Reader — tested via integration tests

    // ========================================================================
    // Run command tests (updated for HexOptions)
    // ========================================================================

    fn make_opts(file: &Path) -> HexOptions {
        HexOptions {
            file: file.to_path_buf(),
            ..HexOptions::default()
        }
    }

    #[test]
    fn test_run_file_not_found() {
        let opts = make_opts(Path::new("/nonexistent/model.apr"));
        let result = run(&opts);
        assert!(result.is_err());
        match result {
            Err(CliError::FileNotFound(_)) => {}
            _ => panic!("Expected FileNotFound error"),
        }
    }

    #[test]
    fn test_run_invalid_apr_file() {
        let mut file = NamedTempFile::with_suffix(".apr").expect("create temp file");
        file.write_all(b"APRN\x00\x00\x00\x00not valid")
            .expect("write");
        let opts = make_opts(file.path());
        let result = run(&opts);
        assert!(result.is_err());
    }

    #[test]
    fn test_run_unknown_format() {
        let mut file = NamedTempFile::with_suffix(".bin").expect("create temp file");
        file.write_all(b"\x00\x00\x00\x00\x00\x00\x00\x00\x00")
            .expect("write");
        let opts = make_opts(file.path());
        let result = run(&opts);
        assert!(result.is_err());
    }

    #[test]
    fn test_run_with_tensor_filter() {
        let mut file = NamedTempFile::with_suffix(".apr").expect("create temp file");
        file.write_all(b"APRN\x00\x00\x00\x00not valid data")
            .expect("write");
        let mut opts = make_opts(file.path());
        opts.tensor = Some("encoder".to_string());
        let result = run(&opts);
        assert!(result.is_err());
    }

    #[test]
    fn test_run_raw_mode_with_data() {
        let mut file = NamedTempFile::with_suffix(".bin").expect("create temp file");
        // Write GGUF magic so format detection works
        file.write_all(b"GGUF\x03\x00\x00\x00").expect("write");
        file.write_all(&[0u8; 100]).expect("write");
        let mut opts = make_opts(file.path());
        opts.raw = true;
        opts.limit = 32;
        let result = run(&opts);
        assert!(result.is_ok());
    }

    #[test]
    fn test_run_header_mode_gguf() {
        let mut file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        // Write minimal GGUF header
        file.write_all(b"GGUF").expect("write");
        file.write_all(&3u32.to_le_bytes()).expect("write"); // version
        file.write_all(&0u64.to_le_bytes()).expect("write"); // tensor_count
        file.write_all(&0u64.to_le_bytes()).expect("write"); // metadata_kv_count
        let mut opts = make_opts(file.path());
        opts.header = true;
        let result = run(&opts);
        assert!(result.is_ok());
    }

    #[test]
    fn test_run_header_mode_apr() {
        let mut file = NamedTempFile::with_suffix(".apr").expect("create temp file");
        file.write_all(b"APRN\x02\x00\x00\x00\x00\x00\x00\x00")
            .expect("write");
        file.write_all(&[0u8; 20]).expect("write");
        let mut opts = make_opts(file.path());
        opts.header = true;
        let result = run(&opts);
        assert!(result.is_ok());
    }

    #[test]
    fn test_run_entropy_mode() {
        let mut file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        // Write GGUF header + some data
        file.write_all(b"GGUF").expect("write");
        file.write_all(&3u32.to_le_bytes()).expect("write");
        file.write_all(&0u64.to_le_bytes()).expect("write");
        file.write_all(&0u64.to_le_bytes()).expect("write");
        file.write_all(&[0x42u8; 8192]).expect("write");
        let mut opts = make_opts(file.path());
        opts.entropy = true;
        let result = run(&opts);
        assert!(result.is_ok());
    }

    #[test]
    fn test_run_nonexistent_path_returns_file_not_found() {
        let opts = make_opts(Path::new("/tmp/this_does_not_exist_apr_test.apr"));
        let result = run(&opts);
        assert!(result.is_err());
        match result {
            Err(CliError::FileNotFound(p)) => {
                assert_eq!(p, Path::new("/tmp/this_does_not_exist_apr_test.apr"));
            }
            other => panic!("Expected FileNotFound, got {:?}", other),
        }
    }

    #[test]
    fn test_run_empty_file() {
        let file = NamedTempFile::with_suffix(".apr").expect("create temp file");
        let opts = make_opts(file.path());
        let result = run(&opts);
        assert!(result.is_err());
    }

    // ========================================================================
    // Raw hex output tests
    // ========================================================================

    #[test]
    fn test_print_raw_hex_basic() {
        let bytes = [0x41, 0x50, 0x52, 0x4E, 0x00, 0x01, 0x02, 0x03];
        print_raw_hex(&bytes, 0, 8, 16);
    }

    #[test]
    fn test_print_raw_hex_with_offset() {
        let bytes = [0u8; 256];
        print_raw_hex(&bytes, 128, 32, 16);
    }

    #[test]
    fn test_print_raw_hex_width_32() {
        let bytes = [0xAA; 64];
        print_raw_hex(&bytes, 0, 64, 32);
    }

    #[test]
    fn test_print_raw_hex_empty() {
        print_raw_hex(&[], 0, 100, 16);
    }

    #[test]
    fn test_print_raw_hex_offset_past_end() {
        let bytes = [0u8; 10];
        print_raw_hex(&bytes, 100, 10, 16);
    }

    #[test]
    fn test_print_raw_hex_width_zero_uses_default() {
        let bytes = [0x42; 32];
        print_raw_hex(&bytes, 0, 32, 0);
    }

    // ========================================================================
    // format_display_name
    // ========================================================================

    #[test]
    fn test_format_display_name() {
        assert_eq!(format_display_name(FileFormat::Apr), "APR");
        assert_eq!(format_display_name(FileFormat::Gguf), "GGUF");
        assert_eq!(format_display_name(FileFormat::SafeTensors), "SafeTensors");
    }

    // ========================================================================
    // Q4K/Q6K/Q8_0 block print tests (no panic)
    // ========================================================================

    #[test]
    fn test_print_q4k_blocks_synthetic() {
        let mut bytes = vec![0u8; Q4K_BLOCK_SIZE * 2];
        // Set scale bytes to known f16 value (1.0 = 0x3C00)
        bytes[0] = 0x00;
        bytes[1] = 0x3C;
        print_q4k_blocks(&bytes, 0, 1);
    }
