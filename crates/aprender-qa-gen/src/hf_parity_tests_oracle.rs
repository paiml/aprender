#[test]
fn test_tolerance_atol_vs_rtol() {
    let tol = Tolerance::default();
    // atol = 1e-5, rtol = 1e-4
    // For small expected values, atol dominates
    // bound = 1e-5 + 1e-4 * 1e-6 = 1.0001e-5
    // diff = |1.000001e-6 - 1e-6| = 1e-12 << bound
    assert!(tol.is_close(1.000001e-6, 1e-6));
    // For large expected values, rtol dominates
    // bound = 1e-5 + 1e-4 * 10000 = 1.00001
    // diff = 1, bound = ~1
    assert!(tol.is_close(10001.0, 10000.0));
    // Test that values outside tolerance are detected
    // diff = 0.1, bound = 1e-5 + 1e-4 * 1 = 0.00011
    assert!(!tol.is_close(1.1, 1.0));
}

#[test]
fn test_tensors_close_boundary_mismatch_ratio() {
    // Test exactly at the boundary (1% mismatch)
    let oracle = HfParityOracle::new("/tmp", "test");
    let a: Vec<f32> = vec![1.0; 100];
    let mut b = a.clone();
    // Make exactly 1 element differ (1% of 100)
    b[0] = 100.0;
    // 1% = 0.01, which equals max_mismatch_ratio, should still fail
    // because we use > not >=
    let result = oracle.tensors_close(&a, &b);
    // 1/100 = 0.01, which is NOT > 0.01, so it should pass
    assert!(result.is_ok());
}

#[test]
fn test_tensors_close_just_over_boundary() {
    let oracle = HfParityOracle::new("/tmp", "test");
    let a: Vec<f32> = vec![1.0; 100];
    let mut b = a.clone();
    // Make 2 elements differ (2% of 100)
    b[0] = 100.0;
    b[1] = 100.0;
    // 2% > 1% threshold → should fail
    let result = oracle.tensors_close(&a, &b);
    assert!(matches!(result, Err(TensorDiff::ValueMismatch { .. })));
}

#[test]
fn test_is_close_negative_values() {
    let tol = Tolerance::default();
    assert!(tol.is_close(-1.0, -1.0));
    assert!(tol.is_close(-1.000001, -1.0));
    assert!(!tol.is_close(-2.0, -1.0));
}

#[test]
fn test_value_mismatch_captures_worst_element() {
    let oracle = HfParityOracle::new("/tmp", "test");
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![1.0, 2.0, 100.0, 50.0]; // Index 2 has max diff (97)
    if let Err(TensorDiff::ValueMismatch {
        max_diff_idx,
        max_diff,
        expected_val,
        actual_val,
        ..
    }) = oracle.tensors_close(&a, &b)
    {
        assert_eq!(max_diff_idx, 2);
        assert!((max_diff - 97.0).abs() < 0.001);
        assert!((expected_val - 3.0).abs() < 0.001);
        assert!((actual_val - 100.0).abs() < 0.001);
    } else {
        panic!("Expected ValueMismatch");
    }
}

#[test]
fn test_mean_diff_calculation() {
    let oracle = HfParityOracle::new("/tmp", "test");
    let a = vec![0.0, 0.0, 0.0, 0.0];
    let b = vec![1.0, 1.0, 1.0, 1.0]; // All diff by 1.0
    if let Err(TensorDiff::ValueMismatch { mean_diff, .. }) = oracle.tensors_close(&a, &b) {
        assert!((mean_diff - 1.0).abs() < 0.001);
    } else {
        panic!("Expected ValueMismatch");
    }
}

// ============================================================
// Tolerance Struct Equality Tests
// ============================================================

#[test]
fn test_tolerance_equality() {
    let t1 = Tolerance::default();
    let t2 = Tolerance::default();
    assert_eq!(t1, t2);
}

#[test]
fn test_tolerance_inequality() {
    let t1 = Tolerance::fp32();
    let t2 = Tolerance::int4();
    assert_ne!(t1, t2);
}

#[test]
fn test_tolerance_clone() {
    let t1 = Tolerance::fp16();
    let t2 = t1;
    assert_eq!(t1, t2);
}

// ============================================================
// TensorDiff Equality Tests
// ============================================================

#[test]
fn test_tensor_diff_equality_shape() {
    let d1 = TensorDiff::ShapeMismatch {
        expected: 10,
        actual: 5,
    };
    let d2 = TensorDiff::ShapeMismatch {
        expected: 10,
        actual: 5,
    };
    assert_eq!(d1, d2);
}

#[test]
fn test_tensor_diff_clone() {
    let d1 = TensorDiff::ParseError {
        message: "test".to_string(),
    };
    let d2 = d1.clone();
    assert_eq!(d1, d2);
}

// ============================================================
// SafeTensors File I/O Helper
// ============================================================

/// Create a minimal SafeTensors file with a "logits" tensor from f32 values
fn create_safetensors_file(path: &Path, logits: &[f32], shape: &[usize]) {
    use safetensors::tensor::Dtype;
    use std::borrow::Cow;

    struct TestTensor {
        shape: Vec<usize>,
        data: Vec<u8>,
    }

    impl safetensors::tensor::View for TestTensor {
        fn dtype(&self) -> Dtype {
            Dtype::F32
        }
        fn shape(&self) -> &[usize] {
            &self.shape
        }
        fn data(&self) -> Cow<'_, [u8]> {
            Cow::Borrowed(&self.data)
        }
        fn data_len(&self) -> usize {
            self.data.len()
        }
    }

    let data: Vec<u8> = logits.iter().flat_map(|f| f.to_le_bytes()).collect();
    let tensor = TestTensor {
        shape: shape.to_vec(),
        data,
    };
    let tensors = vec![("logits".to_string(), tensor)];
    let bytes =
        safetensors::tensor::serialize(tensors, None).expect("failed to serialize safetensors");
    std::fs::write(path, bytes).expect("failed to write safetensors file");
}

/// Create a unique temporary directory for a test
fn make_test_dir(test_name: &str) -> PathBuf {
    let dir = std::env::temp_dir()
        .join("apr-qa-gen-tests")
        .join(test_name);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("failed to create test dir");
    dir
}

// ============================================================
// load_golden_from_path Tests (file-based)
// ============================================================

#[test]
fn test_load_golden_from_path_success_no_metadata() {
    let dir = make_test_dir("load_golden_no_meta");
    let logits = vec![1.0f32, 2.0, 3.0, 4.0];
    let st_path = dir.join("test.safetensors");
    create_safetensors_file(&st_path, &logits, &[1, 4]);

    let result = HfParityOracle::load_golden_from_path(&st_path, "test prompt", "abc123");
    assert!(result.is_ok(), "Expected Ok, got: {result:?}");

    let golden = result.expect("already checked");
    assert_eq!(golden.input_hash, "abc123");
    assert_eq!(golden.prompt, "test prompt");
    assert_eq!(golden.logits.len(), 4);
    assert!((golden.logits[0] - 1.0).abs() < 1e-6);
    assert!((golden.logits[3] - 4.0).abs() < 1e-6);
    assert_eq!(golden.shape, vec![1, 4]);
    // No companion JSON, so model_id and transformers_version are empty
    assert!(golden.model_id.is_empty());
    assert!(golden.transformers_version.is_empty());
    assert!(golden.text.is_none());

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_load_golden_from_path_with_metadata_json() {
    let dir = make_test_dir("load_golden_with_meta");
    let logits = vec![0.5f32, 1.5];
    let st_path = dir.join("golden.safetensors");
    create_safetensors_file(&st_path, &logits, &[1, 2]);

    // Write companion metadata JSON
    let meta_path = dir.join("golden.json");
    std::fs::write(
            &meta_path,
            r#"{"model": "test-model-id", "transformers_version": "4.42.0", "generated_text": "hello world"}"#,
        )
        .expect("write meta");

    let result = HfParityOracle::load_golden_from_path(&st_path, "prompt", "hash123");
    assert!(result.is_ok());

    let golden = result.expect("already checked");
    assert_eq!(golden.model_id, "test-model-id");
    assert_eq!(golden.transformers_version, "4.42.0");
    assert_eq!(golden.text, Some("hello world".to_string()));

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_load_golden_from_path_file_not_found() {
    let result =
        HfParityOracle::load_golden_from_path(Path::new("/nonexistent/file.safetensors"), "p", "h");
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Failed to read golden file"));
}

#[test]
fn test_load_golden_from_path_invalid_safetensors() {
    let dir = make_test_dir("load_golden_invalid_st");
    let path = dir.join("bad.safetensors");
    std::fs::write(&path, b"this is not a valid safetensors file").expect("write");

    let result = HfParityOracle::load_golden_from_path(&path, "p", "h");
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Failed to parse SafeTensors"));

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_load_golden_from_path_missing_logits_tensor() {
    use safetensors::tensor::Dtype;
    use std::borrow::Cow;

    struct TestTensor {
        shape: Vec<usize>,
        data: Vec<u8>,
    }
    impl safetensors::tensor::View for TestTensor {
        fn dtype(&self) -> Dtype {
            Dtype::F32
        }
        fn shape(&self) -> &[usize] {
            &self.shape
        }
        fn data(&self) -> Cow<'_, [u8]> {
            Cow::Borrowed(&self.data)
        }
        fn data_len(&self) -> usize {
            self.data.len()
        }
    }

    let dir = make_test_dir("load_golden_no_logits");
    let tensor = TestTensor {
        shape: vec![2],
        data: vec![0u8; 8],
    };
    // Name it "not_logits" so the "logits" lookup fails
    let tensors = vec![("not_logits".to_string(), tensor)];
    let bytes = safetensors::tensor::serialize(tensors, None).expect("serialize");
    let path = dir.join("no_logits.safetensors");
    std::fs::write(&path, bytes).expect("write");

    let result = HfParityOracle::load_golden_from_path(&path, "p", "h");
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Missing 'logits' tensor"));

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_load_golden_from_path_invalid_metadata_json_falls_back() {
    let dir = make_test_dir("load_golden_bad_meta");
    let logits = vec![1.0f32];
    let st_path = dir.join("test.safetensors");
    create_safetensors_file(&st_path, &logits, &[1]);

    // Write an invalid JSON companion
    let meta_path = dir.join("test.json");
    std::fs::write(&meta_path, "not valid json{{{").expect("write");

    // Should still succeed because load_metadata_json error triggers unwrap_or_default
    let result = HfParityOracle::load_golden_from_path(&st_path, "p", "h");
    assert!(result.is_ok());
    let golden = result.expect("already checked");
    assert!(golden.model_id.is_empty());
    assert!(golden.transformers_version.is_empty());
    assert!(golden.text.is_none());

    let _ = std::fs::remove_dir_all(&dir);
}

// ============================================================
// load_golden cache hit Tests
// ============================================================

#[test]
fn test_load_golden_cache_hit() {
    let dir = make_test_dir("load_golden_cache");
    let family = "test-family";
    let prompt = "cached prompt";
    let input_hash = hash_prompt(prompt);

    // Create the directory structure: corpus_path/model_family/
    let family_dir = dir.join(family);
    std::fs::create_dir_all(&family_dir).expect("create family dir");

    // Create the golden safetensors file at the expected path
    let st_path = family_dir.join(format!("{input_hash}.safetensors"));
    create_safetensors_file(&st_path, &[42.0f32], &[1]);

    let mut oracle = HfParityOracle::new(&dir, family);

    // First call loads from file
    let result1 = oracle.load_golden(prompt);
    assert!(result1.is_ok());

    // Manually insert into cache to simulate the cache path
    let golden = result1.expect("already checked");
    oracle.golden_cache.insert(input_hash.clone(), golden);

    // Second call should hit the cache (even if file is deleted)
    std::fs::remove_file(&st_path).expect("remove file");
    let result2 = oracle.load_golden(prompt);
    assert!(result2.is_ok());
    let cached = result2.expect("already checked");
    assert!((cached.logits[0] - 42.0).abs() < 1e-6);

    let _ = std::fs::remove_dir_all(&dir);
}

// ============================================================
// compare_tensor_file Tests
// ============================================================

#[test]
fn test_compare_tensor_file_matching() {
    let dir = make_test_dir("compare_tensor_match");
    let logits = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
    let actual_path = dir.join("actual.safetensors");
    create_safetensors_file(&actual_path, &logits, &[1, 5]);

    let golden = GoldenOutput {
        input_hash: "h".to_string(),
        prompt: "p".to_string(),
        logits: logits.clone(),
        shape: vec![1, 5],
        text: None,
        model_id: String::new(),
        transformers_version: String::new(),
    };

    let oracle = HfParityOracle::new("/tmp", "test");
    let result = oracle.compare_tensor_file(&actual_path, &golden);
    assert!(result.is_ok(), "Matching tensors should pass: {result:?}");

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_compare_tensor_file_mismatched_values() {
    let dir = make_test_dir("compare_tensor_mismatch");
    let actual_logits = vec![100.0f32, 200.0, 300.0, 400.0];
    let actual_path = dir.join("actual.safetensors");
    create_safetensors_file(&actual_path, &actual_logits, &[1, 4]);

    let golden = GoldenOutput {
        input_hash: "h".to_string(),
        prompt: "p".to_string(),
        logits: vec![1.0, 2.0, 3.0, 4.0],
        shape: vec![1, 4],
        text: None,
        model_id: String::new(),
        transformers_version: String::new(),
    };

    let oracle = HfParityOracle::new("/tmp", "test");
    let result = oracle.compare_tensor_file(&actual_path, &golden);
    assert!(matches!(result, Err(TensorDiff::ValueMismatch { .. })));

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_compare_tensor_file_not_found() {
    let golden = GoldenOutput {
        input_hash: "h".to_string(),
        prompt: "p".to_string(),
        logits: vec![1.0],
        shape: vec![1],
        text: None,
        model_id: String::new(),
        transformers_version: String::new(),
    };

    let oracle = HfParityOracle::new("/tmp", "test");
    let result = oracle.compare_tensor_file(Path::new("/nonexistent/file.safetensors"), &golden);
    assert!(matches!(result, Err(TensorDiff::ParseError { .. })));
    if let Err(TensorDiff::ParseError { message }) = result {
        assert!(message.contains("Failed to read actual output"));
    }
}
