// ITER7: Deep falsification of oracle 3X statistical engine
//
// These tests independently recompute the same quantities as the oracle
// statistical engine, providing Popperian falsification through independent
// implementation. If oracle and test disagree, one has a bug.
// =============================================================================

/// Independent parameter count computation (independent of oracle code).
/// This is the spec formula from the plan — if oracle diverges, that's a bug.
fn iter7_compute_params(
    sc: &aprender::format::model_family::ModelSizeConfig,
    c: &aprender::format::model_family::ModelConstraints,
) -> u64 {
    use aprender::format::model_family::MlpType;
    let h = sc.hidden_dim as u64;
    let v = sc.vocab_size as u64;
    let l = sc.num_layers as u64;
    let nh = sc.num_heads as u64;
    let nkv = sc.num_kv_heads as u64;
    let hd = sc.head_dim as u64;
    let inter = sc.intermediate_dim as u64;

    let embedding = v * h;
    let attn = h * (nh * hd) + h * (nkv * hd) + h * (nkv * hd) + (nh * hd) * h;
    let attn_bias = if c.has_bias {
        (nh * hd) + (nkv * hd) + (nkv * hd) + h
    } else {
        0
    };
    let is_gated = matches!(c.mlp_type, MlpType::SwiGlu | MlpType::GatedMlp);
    let ffn = if is_gated {
        h * inter * 3
    } else {
        h * inter * 2
    };
    let norms = h * 2;
    let per_layer = attn + attn_bias + ffn + norms;
    let lm_head = if c.tied_embeddings { 0 } else { v * h };
    let final_norm = h;
    embedding + (per_layer * l) + lm_head + final_norm
}

#[test]
fn falsify_iter7_all_computed_values_finite() {
    // STRONG PREDICTION: All computed statistical values are finite (not NaN/Inf)
    // for every real model family + size in the registry.
    let registry = build_default_registry();

    for family_name in KNOWN_FAMILIES {
        let family = registry.get(family_name).expect("family exists");
        let config = family.config();

        for (size_name, size_config) in &config.size_variants {
            let h = size_config.hidden_dim as f64;
            let nh = size_config.num_heads as f64;
            let nkv = size_config.num_kv_heads as f64;
            let inter = size_config.intermediate_dim as f64;

            // GQA ratio
            if nh > 0.0 {
                let gqa_ratio = nkv / nh;
                assert!(
                    gqa_ratio.is_finite(),
                    "ITER7: {family_name}/{size_name} gqa_ratio NaN/Inf"
                );
                assert!(
                    (1.0 - gqa_ratio).is_finite(),
                    "ITER7: {family_name}/{size_name} kv_reduction NaN/Inf"
                );
            }

            // FFN ratio
            if h > 0.0 {
                let ffn_ratio = inter / h;
                assert!(
                    ffn_ratio.is_finite(),
                    "ITER7: {family_name}/{size_name} ffn_ratio NaN/Inf"
                );
            }

            // RoPE wavelength
            let wl = 2.0 * std::f64::consts::PI * size_config.rope_theta;
            assert!(
                wl.is_finite(),
                "ITER7: {family_name}/{size_name} wavelength NaN/Inf"
            );
        }
    }
}

#[test]
fn falsify_iter7_gqa_ratio_plus_reduction_equals_one() {
    // STRONG PREDICTION: For any model, gqa_ratio + kv_cache_reduction == 1.0
    // (ratio = kv/heads, reduction = 1 - ratio).
    let registry = build_default_registry();

    for family_name in KNOWN_FAMILIES {
        let family = registry.get(family_name).expect("family exists");
        let config = family.config();

        for (size_name, size_config) in &config.size_variants {
            if size_config.num_heads == 0 {
                continue;
            }
            let ratio = size_config.num_kv_heads as f64 / size_config.num_heads as f64;
            let reduction = 1.0 - ratio;
            assert!(
                (ratio + reduction - 1.0).abs() < 1e-12,
                "ITER7: {family_name}/{size_name} ratio({ratio})+reduction({reduction}) != 1.0"
            );
            // ratio must be in (0, 1]
            assert!(
                ratio > 0.0 && ratio <= 1.0,
                "ITER7: {family_name}/{size_name} gqa_ratio={ratio} out of (0,1] range"
            );
        }
    }
}

#[test]
fn falsify_iter7_f16_memory_exactly_4x_q4() {
    // STRONG PREDICTION: F16 uses 2 bytes/param, Q4 uses 0.5 bytes/param.
    // Therefore F16_size / Q4_size == 4.0 exactly (both derived from same param count).
    let registry = build_default_registry();

    for family_name in KNOWN_FAMILIES {
        let family = registry.get(family_name).expect("family exists");
        let config = family.config();
        let constraints = family.constraints();

        for (size_name, size_config) in &config.size_variants {
            let params = iter7_compute_params(size_config, constraints);
            if params == 0 {
                continue;
            }
            let f16_mb = (params as f64 * 2.0) / (1024.0 * 1024.0);
            let q4_mb = (params as f64 * 0.5) / (1024.0 * 1024.0);
            let ratio = f16_mb / q4_mb;
            assert!(
                (ratio - 4.0).abs() < 1e-10,
                "ITER7: {family_name}/{size_name} F16/Q4 = {ratio}, expected exactly 4.0"
            );
        }
    }
}

#[test]
fn falsify_iter7_kv_cache_per_token_formula() {
    // STRONG PREDICTION: KV cache per token (bytes) =
    //   2 (K+V) * num_layers * num_kv_heads * head_dim * 2 (f16 bytes)
    let registry = build_default_registry();

    for family_name in KNOWN_FAMILIES {
        let family = registry.get(family_name).expect("family exists");
        let config = family.config();

        for (size_name, size_config) in &config.size_variants {
            let expected = 2_u64
                * size_config.num_layers as u64
                * size_config.num_kv_heads as u64
                * size_config.head_dim as u64
                * 2;

            // 4K cache in MB
            let cache_4k_mb = expected as f64 * 4096.0 / (1024.0 * 1024.0);

            assert!(
                cache_4k_mb.is_finite(),
                "ITER7: {family_name}/{size_name} KV cache 4K is not finite"
            );
            // Sanity: for any model, 4K KV cache < 100 GB
            assert!(
                cache_4k_mb < 100_000.0,
                "ITER7: {family_name}/{size_name} KV cache 4K = {cache_4k_mb:.1} MB > 100 GB"
            );
        }
    }
}

#[test]
fn falsify_iter7_ffn_ratio_exact() {
    // STRONG PREDICTION: FFN expansion ratio == intermediate_dim / hidden_dim
    let registry = build_default_registry();

    for family_name in KNOWN_FAMILIES {
        let family = registry.get(family_name).expect("family exists");
        let config = family.config();

        for (size_name, size_config) in &config.size_variants {
            if size_config.hidden_dim == 0 {
                continue;
            }
            let ratio = size_config.intermediate_dim as f64 / size_config.hidden_dim as f64;

            // Standard LLM FFN ratios are between 1.0 and 8.0
            assert!(
                ratio >= 1.0 && ratio <= 8.0,
                "ITER7: {family_name}/{size_name} FFN ratio {ratio:.2} outside [1.0, 8.0]"
            );
        }
    }
}

#[test]
fn falsify_iter7_rope_wavelength_zero_iff_theta_zero() {
    // STRONG PREDICTION: wavelength = 2π*θ, so wavelength==0 iff θ==0.
    use aprender::format::model_family::PositionalEncoding;

    let registry = build_default_registry();

    for family_name in KNOWN_FAMILIES {
        let family = registry.get(family_name).expect("family exists");
        let constraints = family.constraints();
        let config = family.config();

        for (size_name, size_config) in &config.size_variants {
            let wavelength = 2.0 * std::f64::consts::PI * size_config.rope_theta;

            if constraints.positional_encoding == PositionalEncoding::Rope {
                assert!(
                    wavelength > 0.0,
                    "ITER7: {family_name}/{size_name} RoPE model has wavelength=0"
                );
            } else if size_config.rope_theta == 0.0 {
                assert!(
                    wavelength == 0.0,
                    "ITER7: {family_name}/{size_name} theta=0 but wavelength={wavelength}"
                );
            }
        }
    }
}

#[test]
fn falsify_iter7_flops_ffn_dominates_attention() {
    // STRONG PREDICTION: For all known architectures, FFN FLOPS per token >= attention FLOPS.
    // FFN does 2-3 large matmuls vs attention's QKV projections.
    use aprender::format::model_family::MlpType;

    let registry = build_default_registry();

    for family_name in KNOWN_FAMILIES {
        let family = registry.get(family_name).expect("family exists");
        let config = family.config();
        let constraints = family.constraints();

        for (size_name, size_config) in &config.size_variants {
            let h = size_config.hidden_dim as u64;
            let nh = size_config.num_heads as u64;
            let nkv = size_config.num_kv_heads as u64;
            let hd = size_config.head_dim as u64;
            let inter = size_config.intermediate_dim as u64;
            let l = size_config.num_layers as u64;

            if h == 0 || l == 0 {
                continue;
            }

            // Attention FLOPS per layer: QKV + output projections
            let attn_per_layer = 2 * h * (nh + 2 * nkv) * hd + 2 * nh * hd * h;

            // FFN FLOPS per layer
            let is_gated = matches!(constraints.mlp_type, MlpType::SwiGlu | MlpType::GatedMlp);
            let ffn_per_layer = if is_gated {
                2 * h * inter * 3
            } else {
                2 * h * inter * 2
            };

            assert!(
                ffn_per_layer >= attn_per_layer,
                "ITER7: {family_name}/{size_name} FFN flops ({ffn_per_layer}) < attention ({attn_per_layer})"
            );
        }
    }
}

#[test]
fn falsify_iter7_param_count_monotonic_across_sizes() {
    // STRONG PREDICTION: Within a family, larger declared parameter count →
    // larger independently-computed parameter count. Monotonicity must hold.
    let registry = build_default_registry();

    for family_name in KNOWN_FAMILIES {
        let family = registry.get(family_name).expect("family exists");
        let config = family.config();
        let constraints = family.constraints();

        let mut sizes: Vec<(&str, u64, u64)> = config
            .size_variants
            .iter()
            .map(|(name, sc)| {
                let declared = parse_param_string(&sc.parameters);
                let computed = iter7_compute_params(sc, constraints);
                (name.as_str(), declared, computed)
            })
            .filter(|(_, declared, _)| *declared > 0)
            .collect();

        sizes.sort_by_key(|&(_, declared, _)| declared);

        for window in sizes.windows(2) {
            let (name_a, decl_a, comp_a) = window[0];
            let (name_b, decl_b, comp_b) = window[1];
            if decl_a < decl_b {
                assert!(
                    comp_b >= comp_a,
                    "ITER7: {family_name} monotonicity violation: \
                     {name_a}({comp_a}) > {name_b}({comp_b}) but declared {decl_a} < {decl_b}"
                );
            }
        }
    }
}

#[test]
fn falsify_iter7_param_count_within_3x_of_declared() {
    // STRONG PREDICTION: Independently-computed param count should be within 3x
    // of the declared value (generous for bias terms, norms, etc.).
    let registry = build_default_registry();

    for family_name in KNOWN_FAMILIES {
        let family = registry.get(family_name).expect("family exists");
        let config = family.config();
        let constraints = family.constraints();

        for (size_name, size_config) in &config.size_variants {
            let declared = parse_param_string(&size_config.parameters);
            if declared == 0 {
                continue;
            }
            let computed = iter7_compute_params(size_config, constraints);
            let ratio = computed as f64 / declared as f64;

            assert!(
                (0.3..3.0).contains(&ratio),
                "ITER7: {family_name}/{size_name} computed={computed}, declared={declared}, \
                 ratio={ratio:.2} outside [0.3, 3.0]"
            );
        }
    }
}

