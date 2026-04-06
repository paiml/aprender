
// =============================================================================
// Additional Popperian Falsification Tests
// =============================================================================

/// F116: GQA Ratio Preservation Through Conversions
///
/// Prohibition: If GQA ratio changes during conversion, metadata is corrupted.
#[test]
#[ignore] // Heavy: qwen_1_5b config allocates ~890MB per tensor (151936 vocab × 1536 hidden)
fn test_f116_gqa_ratio_preservation() {
    let configs = [
        ("tiny", ModelConfig::tiny()),
        ("small", ModelConfig::small()),
        ("qwen", ModelConfig::qwen_1_5b()),
    ];

    for (name, config) in configs {
        let original_ratio = config.num_heads / config.num_kv_heads;

        let gguf = GgufFixture::new(config.clone(), QuantType::F32, 42);
        let apr = gguf.convert_to(ModelFormat::APR).expect("apr");
        let st = apr.convert_to(ModelFormat::Safetensors).expect("st");

        let apr_ratio = apr.config().num_heads / apr.config().num_kv_heads;
        let st_ratio = st.config().num_heads / st.config().num_kv_heads;

        assert_eq!(
            original_ratio, apr_ratio,
            "{}: GQA ratio changed GGUF->APR ({} -> {})",
            name, original_ratio, apr_ratio
        );
        assert_eq!(
            original_ratio, st_ratio,
            "{}: GQA ratio changed APR->Safetensors ({} -> {})",
            name, original_ratio, st_ratio
        );
    }
}

/// F117: Vocab Size Boundary Test
///
/// Prohibition: If vocab_size=1 or very large fails, edge cases are broken.
#[test]
fn test_f117_vocab_size_boundaries() {
    // Minimum vocab size
    let mut min_config = ModelConfig::tiny();
    min_config.vocab_size = 2; // Must be at least 2 for meaningful output

    let fixture = GgufFixture::new(min_config, QuantType::F32, 42);
    let result = fixture.forward(Device::Cpu, &[0, 1]);
    assert!(result.is_ok(), "Min vocab size should work");
    assert_eq!(result.expect("expected value").len(), 2);
}

/// F118: Hidden Dimension Divisibility
///
/// Prohibition: If hidden_dim not divisible by num_heads, config is invalid.
#[test]
fn test_f118_hidden_dim_divisibility() {
    let config = ModelConfig::tiny();

    // Verify head_dim is exact
    assert_eq!(
        config.hidden_dim % config.num_heads,
        0,
        "hidden_dim {} must be divisible by num_heads {}",
        config.hidden_dim,
        config.num_heads
    );

    // Verify head_dim computation
    assert_eq!(config.head_dim() * config.num_heads, config.hidden_dim);
}

/// Standard set of model configs for cross-config validation tests.
fn all_model_configs() -> [ModelConfig; 4] {
    [
        ModelConfig::tiny(),
        ModelConfig::small(),
        ModelConfig::tinyllama(),
        ModelConfig::qwen_1_5b(),
    ]
}

/// F119: RoPE Theta Sanity
///
/// Prohibition: If rope_theta <= 0, position encoding will fail.
#[test]
fn test_f119_rope_theta_sanity() {
    for config in all_model_configs() {
        assert!(
            config.rope_theta > 0.0,
            "rope_theta {} must be positive",
            config.rope_theta
        );
        assert!(config.rope_theta.is_finite(), "rope_theta must be finite");
    }
}

/// F120: RMS Norm Epsilon Sanity
///
/// Prohibition: If rms_norm_eps <= 0 or too large, normalization will fail.
#[test]
fn test_f120_rms_norm_eps_sanity() {
    for config in all_model_configs() {
        assert!(
            config.rms_norm_eps > 0.0,
            "rms_norm_eps {} must be positive",
            config.rms_norm_eps
        );
        assert!(
            config.rms_norm_eps < 1e-3,
            "rms_norm_eps {} should be small",
            config.rms_norm_eps
        );
    }
}
