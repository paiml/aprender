//! Core unit tests for HuggingFace pipeline
//!
//! Tests for module exports, error handling, and basic functionality.

use crate::hf_pipeline::*;

#[test]
fn test_module_exports() {
    // Verify all public types are accessible
    let _: FetchError = FetchError::MissingToken;
    let _: FetchOptions = FetchOptions::default();
    let _: WeightFormat = WeightFormat::SafeTensors;
}

#[test]
fn test_end_to_end_fetch_error_handling() {
    let temp_dir = std::env::temp_dir().join("hf_e2e_test");
    let _ = std::fs::remove_dir_all(&temp_dir);

    // Create fetcher with invalid token (intentionally)
    let fetcher = HfModelFetcher::with_token("test_token").cache_dir(&temp_dir);

    // Attempt download - should fail with appropriate error
    let result = fetcher.download_model(
        "nonexistent-org-12345/nonexistent-model-67890",
        FetchOptions::new().files(&["model.safetensors", "config.json"]).cache_dir(&temp_dir),
    );

    // Expect error for nonexistent model
    assert!(result.is_err(), "Should fail for nonexistent model");

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
#[ignore] // Requires network access and valid HF token
fn test_end_to_end_real_fetch() {
    let temp_dir = std::env::temp_dir().join("hf_e2e_real_test");
    let _ = std::fs::remove_dir_all(&temp_dir);

    // Create fetcher (uses HF_TOKEN env var if available)
    let fetcher = HfModelFetcher::new().expect("operation should succeed").cache_dir(&temp_dir);

    // Download a small, known model
    let artifact = fetcher
        .download_model(
            "hf-internal-testing/tiny-random-bert",
            FetchOptions::new().files(&["model.safetensors", "config.json"]).cache_dir(&temp_dir),
        )
        .expect("Download should succeed for test model");

    // Verify artifact
    assert_eq!(artifact.format, WeightFormat::SafeTensors);
    assert!(artifact.path.exists());

    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_distillation_pipeline_mock() {
    use ndarray::Array2;

    // Create mock teacher
    let teacher = SafeTensorsTeacher::mock(12, 768);

    // Verify teacher interface
    assert_eq!(teacher.num_layers(), 12);
    assert_eq!(teacher.hidden_size(), 768);

    // Mock forward pass
    let input = Array2::<f32>::zeros((4, 768));
    let output = teacher.forward(&input).expect("Forward should work");
    assert_eq!(output.dim(), (4, 768));

    // Get hidden states
    let hidden = teacher.hidden_states(&input).expect("Hidden states should work");
    assert_eq!(hidden.len(), 12);

    // Memory estimation
    let mem = teacher.estimate_memory(32, 512);
    assert!(mem.total() > 0);
}

#[test]
fn test_full_distillation_loss_flow() {
    use ndarray::Array2;

    // Create loss function
    let loss_fn = DistillationLoss::new(4.0, 0.7);

    // Mock student and teacher outputs
    let student_logits =
        Array2::from_shape_vec((2, 10), (0..20).map(|x| x as f32).collect()).expect("Valid shape");
    let teacher_logits = Array2::from_shape_vec((2, 10), (0..20).map(|x| (x + 1) as f32).collect())
        .expect("Valid shape");
    let targets = vec![5, 3];

    // Compute loss
    let loss = loss_fn.forward(&student_logits, &teacher_logits, &targets);

    // Should be positive and finite
    assert!(loss >= 0.0);
    assert!(loss.is_finite());
}

#[test]
fn test_progressive_distillation_flow() {
    use ndarray::Array2;

    // Setup progressive distillation
    let prog = ProgressiveDistillation::new(vec![(0, 3), (1, 7), (2, 11)]).with_weight(0.5);

    // Mock hidden states (4 student layers, 12 teacher layers)
    let student_hidden: Vec<Array2<f32>> = (0..4).map(|_| Array2::zeros((4, 768))).collect();
    let teacher_hidden: Vec<Array2<f32>> = (0..12).map(|_| Array2::ones((4, 768))).collect();

    // Compute loss
    let loss = prog.hidden_state_loss(&student_hidden, &teacher_hidden);

    // Should be positive (student zeros, teacher ones)
    assert!(loss > 0.0);
    assert!(loss.is_finite());
}

#[test]
fn test_attention_transfer_flow() {
    use ndarray::Array2;

    let at = AttentionTransfer::new(0.1);

    // Mock attention weights
    let student_attn: Vec<Array2<f32>> = (0..4).map(|_| Array2::zeros((16, 16))).collect();
    let teacher_attn: Vec<Array2<f32>> = (0..12).map(|_| Array2::ones((16, 16))).collect();

    // Compute loss
    let loss = at.loss(&student_attn, &teacher_attn);

    // Should be positive
    assert!(loss > 0.0);
}

#[test]
fn test_security_pytorch_rejection() {
    let fetcher = HfModelFetcher::with_token("test");

    // Should reject PyTorch files by default
    let result =
        fetcher.download_model("test/model", FetchOptions::new().files(&["pytorch_model.bin"]));

    assert!(matches!(result, Err(FetchError::PickleSecurityRisk)));
}

#[test]
fn test_error_retryability() {
    let errors = vec![
        (
            FetchError::NetworkTimeout {
                repo: "r".into(),
                elapsed: std::time::Duration::from_secs(30),
            },
            true,
        ),
        (FetchError::RateLimited { retry_after: std::time::Duration::from_secs(60) }, true),
        (FetchError::ModelNotFound { repo: "r".into() }, false),
        (FetchError::MissingToken, false),
    ];

    for (err, expected_retryable) in errors {
        assert_eq!(err.is_retryable(), expected_retryable, "Retryability mismatch for {err:?}");
    }
}

#[test]
fn test_memory_estimation_realistic() {
    // CodeBERT-base: ~110M params
    let codebert_mem = MemoryEstimate::fp16(110_000_000, 32, 512, 768);
    // Should fit in 8GB
    assert!(codebert_mem.fits_in(8 * 1024 * 1024 * 1024));

    // StarCoder-1B: ~1B params
    let starcoder_mem = MemoryEstimate::fp16(1_000_000_000, 8, 2048, 2048);
    // 1B params * 2 bytes = 2GB, plus activations
    assert!(starcoder_mem.weights == 2_000_000_000);
    assert!(starcoder_mem.total() > 1_500_000_000); // At least 1.5GB

    // Llama-7B quantized: ~7B params
    let llama_int4 = MemoryEstimate::int4(7_000_000_000, 1, 4096, 4096);
    // Should fit in 16GB with INT4
    assert!(llama_int4.fits_in(16 * 1024 * 1024 * 1024));
}

#[test]
fn test_teacher_cache() {
    use ndarray::Array2;

    let mut cache = TeacherCache::new();

    // Simulate caching teacher outputs
    for i in 0..100 {
        let logits = Array2::<f32>::zeros((4, 100));
        cache.cache_logits(i, logits);
    }

    // Verify cache stats
    let stats = cache.stats();
    assert_eq!(stats.logits_cached, 100);

    // Test hit rate after accesses
    for i in 0..50 {
        let _ = cache.get_logits(i);
    }
    for i in 100..110 {
        let _ = cache.get_logits(i);
    }

    let stats = cache.stats();
    assert!(stats.hit_rate() > 0.8); // Most should be hits
}

#[test]
fn test_export_flow() {
    // 1. Create mock weights
    let weights = ModelWeights::mock(4, 256);

    // 2. Export to different formats
    let exporter = Exporter::new().output_dir("/tmp/hf_export_test");

    // SafeTensors
    let st_result = exporter.export(&weights, ExportFormat::SafeTensors, "model.safetensors");
    assert!(st_result.is_ok());

    // APR
    let apr_result = exporter.export(&weights, ExportFormat::APR, "model.apr.json");
    assert!(apr_result.is_ok());

    // GGUF
    let gguf_result = exporter.export(&weights, ExportFormat::GGUF, "model.gguf");
    assert!(gguf_result.is_ok());

    // Cleanup
    std::fs::remove_dir_all("/tmp/hf_export_test").ok();
}
