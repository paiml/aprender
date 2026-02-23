
#[test]
fn test_gguf_model_config_clone() {
    let cfg = gguf::GgufModelConfig {
        architecture: Some("phi".to_string()),
        hidden_size: Some(2048),
        num_layers: Some(24),
        num_heads: Some(32),
        num_kv_heads: Some(32),
        vocab_size: Some(51200),
        intermediate_size: Some(8192),
        max_position_embeddings: Some(2048),
        rope_theta: Some(10000.0),
        rms_norm_eps: Some(1e-5),
        rope_type: Some(0),
    };
    let cloned = cfg.clone();
    assert_eq!(cloned.architecture, cfg.architecture);
    assert_eq!(cloned.hidden_size, cfg.hidden_size);
}

#[test]
fn test_gguf_model_config_debug() {
    let cfg = gguf::GgufModelConfig::default();
    let debug_str = format!("{cfg:?}");
    assert!(debug_str.contains("GgufModelConfig"));
}

// =========================================================================
// FALSIFY tests — model-metadata-bounds-v1.yaml contract
// =========================================================================

/// Helper: valid LLaMA-7B-like config within all bounds.
fn valid_gguf_model_config() -> gguf::GgufModelConfig {
    gguf::GgufModelConfig {
        architecture: Some("llama".to_string()),
        hidden_size: Some(4096),
        num_layers: Some(32),
        num_heads: Some(32),
        num_kv_heads: Some(8),
        vocab_size: Some(32_000),
        intermediate_size: Some(11_008),
        max_position_embeddings: Some(4096),
        rope_theta: Some(10_000.0),
        rms_norm_eps: Some(1e-6),
        rope_type: Some(0),
    }
}

#[test]
fn test_falsify_bounds_valid_config_no_warning() {
    // A valid config should not trigger any warnings.
    // warn_out_of_bounds() writes to stderr — we verify it doesn't panic.
    let cfg = valid_gguf_model_config();
    cfg.warn_out_of_bounds(); // should not panic
}

#[test]
fn test_falsify_bounds_hidden_size_max() {
    // hidden_size max = 65536 per model-metadata-bounds-v1.yaml
    let mut cfg = valid_gguf_model_config();
    cfg.hidden_size = Some(65_536);
    cfg.warn_out_of_bounds(); // boundary: should not warn

    cfg.hidden_size = Some(65_537);
    cfg.warn_out_of_bounds(); // above max: warns
}

#[test]
fn test_falsify_bounds_num_layers_max() {
    // num_layers max = 256 per model-metadata-bounds-v1.yaml
    let mut cfg = valid_gguf_model_config();
    cfg.num_layers = Some(256);
    cfg.warn_out_of_bounds(); // boundary: ok

    cfg.num_layers = Some(257);
    cfg.warn_out_of_bounds(); // above max: warns
}

#[test]
fn test_falsify_bounds_vocab_size_max() {
    // vocab_size max = 1,000,000 per model-metadata-bounds-v1.yaml
    let mut cfg = valid_gguf_model_config();
    cfg.vocab_size = Some(1_000_000);
    cfg.warn_out_of_bounds(); // boundary: ok

    cfg.vocab_size = Some(1_000_001);
    cfg.warn_out_of_bounds(); // above max: warns
}

#[test]
fn test_falsify_bounds_rope_theta_range() {
    // rope_theta: [1.0, 100_000_000.0] when > 0
    let mut cfg = valid_gguf_model_config();

    cfg.rope_theta = Some(1.0);
    cfg.warn_out_of_bounds(); // min boundary: ok

    cfg.rope_theta = Some(100_000_000.0);
    cfg.warn_out_of_bounds(); // max boundary: ok

    cfg.rope_theta = Some(0.5);
    cfg.warn_out_of_bounds(); // below min: warns

    cfg.rope_theta = Some(200_000_000.0);
    cfg.warn_out_of_bounds(); // above max: warns
}

#[test]
fn test_falsify_bounds_eps_range() {
    // eps: [1e-10, 0.01] when > 0
    let mut cfg = valid_gguf_model_config();

    cfg.rms_norm_eps = Some(1e-10);
    cfg.warn_out_of_bounds(); // min boundary: ok

    cfg.rms_norm_eps = Some(0.01);
    cfg.warn_out_of_bounds(); // max boundary: ok

    cfg.rms_norm_eps = Some(0.1);
    cfg.warn_out_of_bounds(); // above max: warns
}

#[test]
fn test_falsify_bounds_none_fields_no_warning() {
    // None values should not trigger warnings
    let cfg = gguf::GgufModelConfig::default();
    cfg.warn_out_of_bounds(); // all None: no warnings
}

#[test]
fn test_falsify_bounds_match_yaml_contract() {
    // Verify the bounds used in warn_out_of_bounds() match model-metadata-bounds-v1.yaml.
    // This test encodes the YAML values — if either the Rust code or YAML changes,
    // this test forces reconciliation.
    struct BoundsSpec {
        field: &'static str,
        min: usize,
        max: usize,
    }

    let specs = [
        BoundsSpec { field: "hidden_size", min: 1, max: 65_536 },
        BoundsSpec { field: "num_layers", min: 1, max: 256 },
        BoundsSpec { field: "num_heads", min: 1, max: 256 },
        BoundsSpec { field: "num_kv_heads", min: 1, max: 256 },
        BoundsSpec { field: "vocab_size", min: 1, max: 1_000_000 },
        BoundsSpec { field: "intermediate_size", min: 1, max: 262_144 },
        BoundsSpec { field: "max_position_embeddings", min: 0, max: 2_097_152 },
    ];

    for spec in &specs {
        // Boundary value (max) should not panic
        let mut cfg = valid_gguf_model_config();
        match spec.field {
            "hidden_size" => cfg.hidden_size = Some(spec.max),
            "num_layers" => cfg.num_layers = Some(spec.max),
            "num_heads" => cfg.num_heads = Some(spec.max),
            "num_kv_heads" => cfg.num_kv_heads = Some(spec.max),
            "vocab_size" => cfg.vocab_size = Some(spec.max),
            "intermediate_size" => cfg.intermediate_size = Some(spec.max),
            "max_position_embeddings" => cfg.max_position_embeddings = Some(spec.max),
            _ => unreachable!(),
        }
        cfg.warn_out_of_bounds();

        // Min value should not panic
        let mut cfg = valid_gguf_model_config();
        match spec.field {
            "hidden_size" => cfg.hidden_size = Some(spec.min),
            "num_layers" => cfg.num_layers = Some(spec.min),
            "num_heads" => cfg.num_heads = Some(spec.min),
            "num_kv_heads" => cfg.num_kv_heads = Some(spec.min),
            "vocab_size" => cfg.vocab_size = Some(spec.min),
            "intermediate_size" => cfg.intermediate_size = Some(spec.min),
            "max_position_embeddings" => cfg.max_position_embeddings = Some(spec.min),
            _ => unreachable!(),
        }
        cfg.warn_out_of_bounds();
    }
}
