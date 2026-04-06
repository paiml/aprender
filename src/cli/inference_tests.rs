
    // =========================================================================
    // GGUF Artifact Falsification - Real File, Real CLI
    // =========================================================================

    fn create_gguf_artifact() -> NamedTempFile {
        let data = build_executable_pygmy_gguf();
        let mut file = NamedTempFile::with_suffix(".gguf").expect("create temp file");
        file.write_all(&data).expect("write gguf data");
        file.flush().expect("flush");
        file
    }

    #[test]
    fn test_gguf_artifact_cli_inference_text_format() {
        let file = create_gguf_artifact();
        let path = file.path().to_str().unwrap();

        let result = inference::run_gguf_inference(
            path,
            &[], // Empty - forces mmap path
            "Hello",
            5,   // max_tokens
            0.0, // temperature (greedy)
            "text",
            false, // force_gpu
            false, // verbose
            None,  // trace_config
        );

        assert!(
            result.is_ok(),
            "GGUF CLI inference failed: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_gguf_artifact_cli_inference_json_format() {
        let file = create_gguf_artifact();
        let path = file.path().to_str().unwrap();

        let result = inference::run_gguf_inference(
            path,
            &[],
            "Test prompt",
            3,
            0.0,
            "json",
            false,
            false,
            None, // trace_config
        );

        assert!(
            result.is_ok(),
            "GGUF JSON format failed: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_gguf_artifact_cli_inference_verbose() {
        let file = create_gguf_artifact();
        let path = file.path().to_str().unwrap();

        let result = inference::run_gguf_inference(
            path,
            &[],
            "Verbose test",
            2,
            0.0,
            "text",
            false, // force_gpu
            true,  // verbose
            None,  // trace_config
        );

        assert!(
            result.is_ok(),
            "GGUF verbose mode failed: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_gguf_artifact_cli_inference_with_temperature() {
        let file = create_gguf_artifact();
        let path = file.path().to_str().unwrap();

        let result = inference::run_gguf_inference(
            path,
            &[],
            "Temperature test",
            3,
            0.7, // non-zero temperature
            "text",
            false,
            false,
            None, // trace_config
        );

        assert!(
            result.is_ok(),
            "GGUF with temperature failed: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_gguf_artifact_nonexistent_path_error() {
        let result = inference::run_gguf_inference(
            "/nonexistent/path/model.gguf",
            &[],
            "Test",
            5,
            0.0,
            "text",
            false,
            false,
            None, // trace_config
        );

        assert!(result.is_err(), "Should fail for nonexistent file");
    }

    // =========================================================================
    // APR Artifact Falsification - Real File, Real CLI
    // =========================================================================

    fn create_apr_artifact() -> (NamedTempFile, Vec<u8>) {
        let data = build_executable_pygmy_apr();
        let mut file = NamedTempFile::with_suffix(".apr").expect("create temp file");
        file.write_all(&data).expect("write apr data");
        file.flush().expect("flush");
        (file, data)
    }

    #[test]
    fn test_apr_artifact_cli_inference_text_format() {
        let (file, data) = create_apr_artifact();
        let path = file.path().to_str().unwrap();

        let result =
            inference::run_apr_inference(path, &data, "Hello", 5, 0.0, "text", false, false, None);

        assert!(
            result.is_ok(),
            "APR CLI inference failed: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_apr_artifact_cli_inference_json_format() {
        let (file, data) = create_apr_artifact();
        let path = file.path().to_str().unwrap();

        let result = inference::run_apr_inference(
            path,
            &data,
            "JSON test",
            3,
            0.0,
            "json",
            false,
            false,
            None,
        );

        assert!(result.is_ok(), "APR JSON format failed: {:?}", result.err());
    }

    #[test]
    fn test_apr_artifact_cli_inference_verbose() {
        let (file, data) = create_apr_artifact();
        let path = file.path().to_str().unwrap();

        let result = inference::run_apr_inference(
            path, &data, "Verbose", 2, 0.0, "text", false, // force_gpu
            true,  // verbose
            None,  // trace_config
        );

        assert!(
            result.is_ok(),
            "APR verbose mode failed: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_apr_artifact_cli_inference_with_temperature() {
        let (file, data) = create_apr_artifact();
        let path = file.path().to_str().unwrap();

        let result = inference::run_apr_inference(
            path,
            &data,
            "Temp test",
            3,
            0.8,
            "text",
            false,
            false,
            None,
        );

        assert!(
            result.is_ok(),
            "APR with temperature failed: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_apr_artifact_nonexistent_path_error() {
        // With empty data, should fail to parse
        let result = inference::run_apr_inference(
            "/nonexistent/path/model.apr",
            &[],
            "Test",
            5,
            0.0,
            "text",
            false,
            false,
            None, // trace_config
        );

        assert!(result.is_err(), "Should fail for empty data");
    }

    // =========================================================================
    // CLI mod.rs Artifact Falsification - display_model_info, load_*_model
    // =========================================================================

    #[test]
    fn test_display_model_info_gguf_artifact() {
        let file = create_gguf_artifact();
        let path = file.path().to_str().unwrap();
        let data = std::fs::read(file.path()).unwrap();

        let result = crate::cli::display_model_info(path, &data);
        assert!(
            result.is_ok(),
            "display_model_info GGUF failed: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_display_model_info_apr_artifact() {
        let (file, data) = create_apr_artifact();
        let path = file.path().to_str().unwrap();

        let result = crate::cli::display_model_info(path, &data);
        assert!(
            result.is_ok(),
            "display_model_info APR failed: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_load_gguf_model_artifact() {
        let file = create_gguf_artifact();
        let data = std::fs::read(file.path()).unwrap();

        let result = crate::cli::load_gguf_model(&data);
        assert!(result.is_ok(), "load_gguf_model failed: {:?}", result.err());
    }

    #[test]
    fn test_load_apr_model_artifact() {
        let (_file, data) = create_apr_artifact();

        let result = crate::cli::load_apr_model(&data);
        assert!(result.is_ok(), "load_apr_model failed: {:?}", result.err());
    }

    // =========================================================================
    // Edge Cases and Error Paths
    // =========================================================================

    #[test]
    fn test_gguf_artifact_zero_max_tokens() {
        let file = create_gguf_artifact();
        let path = file.path().to_str().unwrap();

        let result =
            inference::run_gguf_inference(path, &[], "Test", 0, 0.0, "text", false, false, None);
        // Should handle gracefully (either Ok with empty output or specific error)
        let _ = result; // Don't assert - just ensure no panic
    }

    #[test]
    fn test_apr_artifact_zero_max_tokens() {
        let (file, data) = create_apr_artifact();
        let path = file.path().to_str().unwrap();

        let result =
            inference::run_apr_inference(path, &data, "Test", 0, 0.0, "text", false, false, None);
        let _ = result;
    }

    #[test]
    fn test_gguf_artifact_empty_prompt() {
        let file = create_gguf_artifact();
        let path = file.path().to_str().unwrap();

        let result =
            inference::run_gguf_inference(path, &[], "", 5, 0.0, "text", false, false, None);
        // Empty prompt should be handled
        let _ = result;
    }

    #[test]
    fn test_apr_artifact_empty_prompt() {
        let (file, data) = create_apr_artifact();
        let path = file.path().to_str().unwrap();

        let result =
            inference::run_apr_inference(path, &data, "", 5, 0.0, "text", false, false, None);
        let _ = result;
    }

    #[test]
    fn test_gguf_artifact_high_temperature() {
        let file = create_gguf_artifact();
        let path = file.path().to_str().unwrap();

        let result = inference::run_gguf_inference(
            path,
            &[],
            "High temp",
            3,
            2.0,
            "text",
            false,
            false,
            None, // trace_config
        );
        assert!(result.is_ok(), "High temperature should work");
    }

    #[test]
    fn test_apr_artifact_debug_mode() {
        let (file, data) = create_apr_artifact();
        let path = file.path().to_str().unwrap();

        let result = inference::run_apr_inference(
            path, &data, "Debug", 2, 0.0, "text", false, true, None, // trace_config
        );
        assert!(result.is_ok(), "Debug mode failed: {:?}", result.err());
    }

    // =========================================================================
    // Directory and Path Edge Cases
    // =========================================================================

    #[test]
    fn test_gguf_artifact_in_subdirectory() {
        let dir = TempDir::new().unwrap();
        let subdir = dir.path().join("models");
        std::fs::create_dir(&subdir).unwrap();

        let file_path = subdir.join("pygmy.gguf");
        let data = build_executable_pygmy_gguf();
        std::fs::write(&file_path, &data).unwrap();

        let result = inference::run_gguf_inference(
            file_path.to_str().unwrap(),
            &[],
            "Subdir test",
            3,
            0.0,
            "text",
            false,
            false,
            None, // trace_config
        );

        assert!(
            result.is_ok(),
            "Subdirectory inference failed: {:?}",
            result.err()
        );
    }

    #[test]
    fn test_apr_artifact_in_subdirectory() {
        let dir = TempDir::new().unwrap();
        let subdir = dir.path().join("models");
        std::fs::create_dir(&subdir).unwrap();

        let file_path = subdir.join("pygmy.apr");
        let data = build_executable_pygmy_apr();
        std::fs::write(&file_path, &data).unwrap();

        let result = inference::run_apr_inference(
            file_path.to_str().unwrap(),
            &data,
            "Subdir test",
            3,
            0.0,
            "text",
            false,
            false,
            None, // trace_config
        );

        assert!(
            result.is_ok(),
            "APR subdirectory inference failed: {:?}",
            result.err()
        );
    }

    // =========================================================================
    // Additional CLI mod.rs coverage - format detection and loading
    // =========================================================================

    #[test]
    fn test_display_model_info_unknown_format() {
        // Unknown format prints "Unknown" but succeeds
        let unknown_data = b"UNKNOWN_FORMAT_DATA_HERE";
        let result = crate::cli::display_model_info("/tmp/unknown.bin", unknown_data);
        // display_model_info handles unknown formats gracefully (prints format info)
        assert!(result.is_ok());
    }

    #[test]
    fn test_load_gguf_model_invalid_data() {
        // Invalid GGUF data should fail
        let invalid = b"NOT_GGUF_DATA";
        let result = crate::cli::load_gguf_model(invalid);
        assert!(result.is_err());
    }

    #[test]
    fn test_load_apr_model_invalid_data() {
        // Invalid APR data should fail
        let invalid = b"NOT_APR_DATA";
        let result = crate::cli::load_apr_model(invalid);
        assert!(result.is_err());
    }
