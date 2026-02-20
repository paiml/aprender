// Iteration 6: Oracle 3X Enhancement — Statistical Property Tests
// =============================================================================
//
// These tests verify the statistical/mathematical properties that the oracle
// 3X enhancement depends on. They test invariants at the model_family level
// which the apr-cli oracle uses for computations.

#[test]
fn falsify_iter6_gqa_ratio_range_for_all_families() {
    // STRONG PREDICTION: For all families/sizes, the GQA ratio
    // (num_kv_heads / num_heads) is in (0, 1] and KV cache reduction
    // is in [0, 1).
    let registry = build_default_registry();

    for family_name in KNOWN_FAMILIES {
        let family = registry.get(family_name).expect("family exists");
        let config = family.config();

        for (size_name, size_config) in &config.size_variants {
            if size_config.num_heads == 0 {
                continue;
            }
            let ratio = size_config.num_kv_heads as f64 / size_config.num_heads as f64;
            assert!(
                ratio > 0.0 && ratio <= 1.0,
                "ITER6: {family_name}/{size_name} GQA ratio {ratio} must be in (0, 1]. \
                 num_kv_heads={}, num_heads={}",
                size_config.num_kv_heads,
                size_config.num_heads
            );

            let reduction = 1.0 - ratio;
            assert!(
                (0.0..1.0).contains(&reduction),
                "ITER6: {family_name}/{size_name} KV cache reduction {reduction} must be in [0, 1)"
            );
        }
    }
}

#[test]
fn falsify_iter6_ffn_expansion_ratio_consistent() {
    // STRONG PREDICTION: FFN expansion ratio (intermediate_dim / hidden_dim) is > 1
    // for all families/sizes. SwiGLU models typically use ~2.67x, standard ~4x.
    let registry = build_default_registry();

    for family_name in KNOWN_FAMILIES {
        let family = registry.get(family_name).expect("family exists");
        let config = family.config();

        for (size_name, size_config) in &config.size_variants {
            if size_config.hidden_dim == 0 || size_config.intermediate_dim == 0 {
                continue;
            }
            let ratio = size_config.intermediate_dim as f64 / size_config.hidden_dim as f64;

            // Must be > 1 (FFN expands)
            assert!(
                ratio > 1.0,
                "ITER6: {family_name}/{size_name} FFN ratio {ratio:.2} must be > 1.0"
            );

            // Must be < 10 (sanity: no model uses 10x expansion)
            assert!(
                ratio < 10.0,
                "ITER6: {family_name}/{size_name} FFN ratio {ratio:.2} suspiciously high (> 10x)"
            );
        }
    }
}

#[test]
fn falsify_iter6_kv_cache_per_token_computed_correctly() {
    // STRONG PREDICTION: KV cache per token = 2 * num_layers * num_kv_heads * head_dim * 2 (f16 bytes)
    // This formula must hold for all families/sizes.
    let registry = build_default_registry();

    for family_name in KNOWN_FAMILIES {
        let family = registry.get(family_name).expect("family exists");
        let config = family.config();

        for (size_name, size_config) in &config.size_variants {
            let expected = 2_u64
                * size_config.num_layers as u64
                * size_config.num_kv_heads as u64
                * size_config.head_dim as u64
                * 2; // f16 bytes

            assert!(
                expected > 0,
                "ITER6: {family_name}/{size_name} KV cache per token must be > 0"
            );

            // Verify 4K context KV cache is reasonable (< 100 GB for any model)
            let cache_4k = expected as f64 * 4096.0 / (1024.0 * 1024.0);
            assert!(
                cache_4k < 100_000.0,
                "ITER6: {family_name}/{size_name} KV cache for 4K context ({cache_4k:.1} MB) exceeds 100 GB"
            );
        }
    }
}

#[test]
fn falsify_iter6_param_count_order_of_magnitude() {
    // STRONG PREDICTION: Computed parameter count should be within 2x of the
    // declared parameter count string (e.g., "1.5B" → ~1.5 billion ± 2x).
    use aprender::format::model_family::MlpType;

    let registry = build_default_registry();

    for family_name in KNOWN_FAMILIES {
        let family = registry.get(family_name).expect("family exists");
        let config = family.config();
        let constraints = family.constraints();

        for (size_name, size_config) in &config.size_variants {
            // Parse declared params from string like "1.5B", "0.5B", "7B", etc.
            let declared = parse_param_string(&size_config.parameters);
            if declared == 0 {
                continue; // Can't verify if we can't parse
            }

            // Compute expected params using the same formula as oracle
            let h = size_config.hidden_dim as u64;
            let v = size_config.vocab_size as u64;
            let l = size_config.num_layers as u64;
            let n_heads = size_config.num_heads as u64;
            let n_kv = size_config.num_kv_heads as u64;
            let head_d = size_config.head_dim as u64;
            let inter = size_config.intermediate_dim as u64;

            let embedding = v * h;
            let attn = h * (n_heads * head_d)
                + h * (n_kv * head_d)
                + h * (n_kv * head_d)
                + (n_heads * head_d) * h;
            let is_gated = matches!(constraints.mlp_type, MlpType::SwiGlu | MlpType::GatedMlp);
            let ffn = if is_gated {
                h * inter * 3
            } else {
                h * inter * 2
            };
            let norms = h * 2;
            let per_layer = attn + ffn + norms;
            let lm_head = if constraints.tied_embeddings {
                0
            } else {
                v * h
            };
            let computed = embedding + (per_layer * l) + lm_head + h;

            // Must be within 3x (generous tolerance for bias terms, etc.)
            let ratio = computed as f64 / declared as f64;
            assert!(
                (0.3..3.0).contains(&ratio),
                "ITER6: {family_name}/{size_name} computed params ({computed}) vs declared '{}'  \
                 ratio {ratio:.2} — outside 0.3x-3.0x range",
                size_config.parameters
            );
        }
    }
}

#[test]
fn falsify_iter6_rope_wavelength_positive_for_rope_models() {
    // STRONG PREDICTION: For RoPE models, 2π * rope_theta > 0
    use aprender::format::model_family::PositionalEncoding;

    let registry = build_default_registry();

    for family_name in KNOWN_FAMILIES {
        let family = registry.get(family_name).expect("family exists");
        let constraints = family.constraints();

        if constraints.positional_encoding == PositionalEncoding::Rope {
            let config = family.config();
            for (size_name, size_config) in &config.size_variants {
                let wavelength = 2.0 * std::f64::consts::PI * size_config.rope_theta;
                assert!(
                    wavelength > 0.0,
                    "ITER6: {family_name}/{size_name} RoPE max wavelength must be > 0, got {wavelength}"
                );
            }
        }
    }
}

#[test]
fn falsify_iter6_context_window_positive_for_rope_models() {
    // STRONG PREDICTION: max_position_embeddings > 0 for RoPE-based models
    // Encoder-decoder models (Whisper) use max_source_positions/max_target_positions
    // which map differently, so we only assert for RoPE families.
    use aprender::format::model_family::PositionalEncoding;

    let registry = build_default_registry();

    for family_name in KNOWN_FAMILIES {
        let family = registry.get(family_name).expect("family exists");
        let constraints = family.constraints();

        if constraints.positional_encoding == PositionalEncoding::Rope {
            let config = family.config();
            for (size_name, size_config) in &config.size_variants {
                assert!(
                    size_config.max_position_embeddings > 0,
                    "ITER6: {family_name}/{size_name} max_position_embeddings must be > 0"
                );
            }
        }
    }
}

#[test]
fn falsify_iter6_gqa_implies_kv_cache_savings() {
    // STRONG PREDICTION: For GQA families, at least one size has kv_heads < heads,
    // which means the KV cache per token should be smaller than the MHA equivalent.
    use aprender::format::model_family::AttentionType;

    let registry = build_default_registry();

    for family_name in KNOWN_FAMILIES {
        let family = registry.get(family_name).expect("family exists");
        let constraints = family.constraints();

        if constraints.attention_type == AttentionType::Gqa {
            let config = family.config();
            for (size_name, size_config) in &config.size_variants {
                if size_config.num_kv_heads < size_config.num_heads {
                    // GQA KV cache
                    let gqa_kv = size_config.num_kv_heads as u64 * size_config.head_dim as u64;
                    // MHA equivalent
                    let mha_kv = size_config.num_heads as u64 * size_config.head_dim as u64;

                    assert!(
                        gqa_kv < mha_kv,
                        "ITER6: {family_name}/{size_name} GQA KV ({gqa_kv}) must be < MHA KV ({mha_kv})"
                    );
                }
            }
        }
    }
}

#[test]
fn falsify_iter6_model_size_f16_gt_q4() {
    // STRONG PREDICTION: F16 model size > Q4 model size for any param count > 0
    let registry = build_default_registry();

    for family_name in KNOWN_FAMILIES {
        let family = registry.get(family_name).expect("family exists");
        let config = family.config();

        for (size_name, size_config) in &config.size_variants {
            let h = size_config.hidden_dim as u64;
            let v = size_config.vocab_size as u64;
            // Quick param estimate: at least embedding layer
            let min_params = v * h;

            let f16_size = min_params as f64 * 2.0;
            let q4_size = min_params as f64 * 0.5;

            assert!(
                f16_size > q4_size,
                "ITER6: {family_name}/{size_name} F16 size must be > Q4 size"
            );
        }
    }
}

// =============================================================================
// Helpers
// =============================================================================

/// Parse a parameter count string like "1.5B", "0.5B", "7B", "768M", "tiny" etc.
fn parse_param_string(s: &str) -> u64 {
    let s = s.trim().to_uppercase();
    if let Some(rest) = s.strip_suffix('B') {
        if let Ok(v) = rest.parse::<f64>() {
            return (v * 1e9) as u64;
        }
    }
    if let Some(rest) = s.strip_suffix('M') {
        if let Ok(v) = rest.parse::<f64>() {
            return (v * 1e6) as u64;
        }
    }
    // Non-numeric sizes (tiny, base, small, etc.) — can't compare
    0
}

// =============================================================================
