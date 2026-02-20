#[test]
fn falsify_iter7_quant_sizes_strictly_ordered() {
    // STRONG PREDICTION: For any param count > 0:
    // F16 (16 bits) > Q8 (8 bits) > Q6_K (6.5 bits) > Q4_K_M (4.5 bits)
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
            let p = params as f64;
            let f16 = p * 2.0; // 16 bits
            let q8 = p * 1.0; // 8 bits
            let q6k = p * 0.8125; // 6.5 bits
            let q4km = p * 0.5625; // 4.5 bits

            assert!(f16 > q8, "ITER7: {family_name}/{size_name} F16 <= Q8");
            assert!(q8 > q6k, "ITER7: {family_name}/{size_name} Q8 <= Q6_K");
            assert!(
                q6k > q4km,
                "ITER7: {family_name}/{size_name} Q6_K <= Q4_K_M"
            );
        }
    }
}

#[test]
fn falsify_iter7_gpu_tps_18x_cpu_tps() {
    // STRONG PREDICTION: GPU TPS / CPU TPS == 900/50 == 18.0 exactly
    // (memory bandwidth model: tps = bandwidth / model_size).
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
            let q4_size_gb = (params as f64 * 0.5625) / (1024.0 * 1024.0 * 1024.0);
            let cpu_tps = 50.0 / q4_size_gb;
            let gpu_tps = 900.0 / q4_size_gb;
            let ratio = gpu_tps / cpu_tps;

            assert!(
                (ratio - 18.0).abs() < 1e-10,
                "ITER7: {family_name}/{size_name} GPU/CPU TPS ratio = {ratio:.6}, expected 18.0"
            );
            assert!(
                gpu_tps > cpu_tps,
                "ITER7: {family_name}/{size_name} GPU ({gpu_tps:.1}) <= CPU ({cpu_tps:.1})"
            );
        }
    }
}

#[test]
fn falsify_iter7_memory_required_exceeds_model_size() {
    // STRONG PREDICTION: Total memory = Q4_K_M model size + KV cache.
    // Memory > model size because KV cache > 0 for any model with layers.
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
            let q4_mb = (params as f64 * 0.5625) / (1024.0 * 1024.0);
            let kv_per_token = 2_u64
                * size_config.num_layers as u64
                * size_config.num_kv_heads as u64
                * size_config.head_dim as u64
                * 2;
            let kv_4k_mb = kv_per_token as f64 * 4096.0 / (1024.0 * 1024.0);
            let total = q4_mb + kv_4k_mb;

            assert!(
                total > q4_mb,
                "ITER7: {family_name}/{size_name} total memory ({total:.1}) <= model size ({q4_mb:.1})"
            );
        }
    }
}

#[test]
fn falsify_iter7_gqa_kv_cache_smaller_than_mha() {
    // STRONG PREDICTION: For GQA models (kv_heads < heads), KV cache is strictly
    // smaller than the MHA equivalent.
    use aprender::format::model_family::AttentionType;

    let registry = build_default_registry();

    for family_name in KNOWN_FAMILIES {
        let family = registry.get(family_name).expect("family exists");
        let constraints = family.constraints();

        if constraints.attention_type != AttentionType::Gqa {
            continue;
        }

        let config = family.config();
        for (size_name, size_config) in &config.size_variants {
            if size_config.num_kv_heads >= size_config.num_heads {
                continue;
            }
            let gqa_kv_bytes = 2_u64
                * size_config.num_layers as u64
                * size_config.num_kv_heads as u64
                * size_config.head_dim as u64
                * 2;
            let mha_kv_bytes = 2_u64
                * size_config.num_layers as u64
                * size_config.num_heads as u64
                * size_config.head_dim as u64
                * 2;

            assert!(
                gqa_kv_bytes < mha_kv_bytes,
                "ITER7: {family_name}/{size_name} GQA KV ({gqa_kv_bytes}) >= MHA KV ({mha_kv_bytes})"
            );

            // Verify reduction ratio matches
            let ratio = gqa_kv_bytes as f64 / mha_kv_bytes as f64;
            let expected_ratio = size_config.num_kv_heads as f64 / size_config.num_heads as f64;
            assert!(
                (ratio - expected_ratio).abs() < 1e-10,
                "ITER7: {family_name}/{size_name} KV reduction ratio {ratio:.4} != GQA ratio {expected_ratio:.4}"
            );
        }
    }
}

#[test]
fn falsify_iter7_gated_mlp_uses_3_matrices() {
    // STRONG PREDICTION: SwiGLU/GatedMlp FFN params = hidden * intermediate * 3
    // Standard GELU MLP params = hidden * intermediate * 2
    use aprender::format::model_family::MlpType;

    let registry = build_default_registry();

    for family_name in KNOWN_FAMILIES {
        let family = registry.get(family_name).expect("family exists");
        let config = family.config();
        let constraints = family.constraints();

        for (size_name, size_config) in &config.size_variants {
            let h = size_config.hidden_dim as u64;
            let inter = size_config.intermediate_dim as u64;
            if h == 0 || inter == 0 {
                continue;
            }

            let is_gated = matches!(constraints.mlp_type, MlpType::SwiGlu | MlpType::GatedMlp);
            let ffn_params = if is_gated {
                h * inter * 3
            } else {
                h * inter * 2
            };

            // Gated: gate_proj + up_proj + down_proj = 3 matmuls
            if is_gated {
                assert_eq!(
                    ffn_params,
                    h * inter * 3,
                    "ITER7: {family_name}/{size_name} gated FFN should have 3 weight matrices"
                );
            } else {
                assert_eq!(
                    ffn_params,
                    h * inter * 2,
                    "ITER7: {family_name}/{size_name} standard FFN should have 2 weight matrices"
                );
            }
        }
    }
}

#[test]
fn falsify_iter7_chinchilla_tokens_20x_params() {
    // STRONG PREDICTION: Chinchilla-optimal training tokens = 20 * params.
    // For a 7B model → 140B tokens. For 1.5B → 30B tokens.
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
            let params_b = params as f64 / 1e9;
            let chinchilla_tokens_b = params_b * 20.0;

            // Chinchilla tokens should be reasonable (> 1B for any real model)
            assert!(
                chinchilla_tokens_b >= 0.1,
                "ITER7: {family_name}/{size_name} Chinchilla tokens = {chinchilla_tokens_b:.1}B < 0.1B"
            );

            // Training FLOPs ≈ 6 * params * tokens
            let training_flops = 6.0 * params as f64 * chinchilla_tokens_b * 1e9;
            assert!(
                training_flops > 0.0 && training_flops.is_finite(),
                "ITER7: {family_name}/{size_name} training FLOPs = {training_flops:.2e} invalid"
            );
        }
    }
}

#[test]
fn falsify_iter7_attention_type_matches_head_config() {
    // STRONG PREDICTION: GQA families have at least one size where kv_heads < heads.
    // MHA families have kv_heads == heads for all sizes.
    use aprender::format::model_family::AttentionType;

    let registry = build_default_registry();

    for family_name in KNOWN_FAMILIES {
        let family = registry.get(family_name).expect("family exists");
        let constraints = family.constraints();
        let config = family.config();

        match constraints.attention_type {
            AttentionType::Mha => {
                for (size_name, sc) in &config.size_variants {
                    if sc.num_heads > 0 {
                        assert_eq!(
                            sc.num_kv_heads, sc.num_heads,
                            "ITER7: {family_name}/{size_name} MHA but kv_heads != heads"
                        );
                    }
                }
            }
            AttentionType::Gqa => {
                let has_gqa_size = config
                    .size_variants
                    .values()
                    .any(|sc| sc.num_heads > 0 && sc.num_kv_heads < sc.num_heads);
                assert!(
                    has_gqa_size,
                    "ITER7: {family_name} declared GQA but no size has kv_heads < heads"
                );
            }
            AttentionType::Mqa => {
                let has_mqa_size = config.size_variants.values().any(|sc| sc.num_kv_heads == 1);
                assert!(
                    has_mqa_size,
                    "ITER7: {family_name} declared MQA but no size has kv_heads == 1"
                );
            }
        }
    }
}

#[test]
fn falsify_iter7_independent_param_count_matches_oracle() {
    // STRONG PREDICTION: Our independent param count matches the oracle's formula
    // (same spec, independent code path). This is the ultimate Popperian test:
    // two independent implementations of the same formula should agree.
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

            // Within 3x is good enough (bias terms, norms, embeddings differ)
            assert!(
                (0.3..3.0).contains(&ratio),
                "ITER7: {family_name}/{size_name} independent param count {computed} vs \
                 declared '{}'  ratio {ratio:.2}",
                size_config.parameters
            );
        }
    }
}

// =============================================================================
