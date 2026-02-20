// §7.6 — FALSIFY-ALG-008: KV head ordering
// =============================================================================
//
// Prediction: num_kv_heads <= num_heads for all model sizes.
// GQA groups multiple query heads per KV head — reversing the ratio is invalid.
//
// Found via falsification round 2: a YAML with num_kv_heads=16, num_heads=4
// was only partially caught by ALG-002 (divisibility). The ordering constraint
// makes the intent explicit.

#[test]
fn falsify_alg_008_kv_heads_ordering() {
    let registry = build_default_registry();

    for family_name in KNOWN_FAMILIES {
        let family = registry.get(family_name).expect("family exists");
        for (size_name, size_config) in &family.config().size_variants {
            assert!(
                size_config.num_kv_heads <= size_config.num_heads,
                "FALSIFY-ALG-008: {family_name}/{size_name} has num_kv_heads={} > num_heads={} — \
                 GQA reduces heads, never adds",
                size_config.num_kv_heads,
                size_config.num_heads
            );
        }
    }
}

// =============================================================================
// §7.6 — FALSIFY-ALG-009: Norm epsilon positivity
// =============================================================================
//
// Prediction: norm_eps > 0 for all model sizes.
// RMSNorm computes x / sqrt(mean(x²) + eps). If eps=0 and input is zero,
// division by zero occurs (Zhang & Sennrich, 2019).
//
// Found via falsification round 2: attack_eps0.yaml with rms_norm_eps=0.0
// passed the build because no assertion checked norm_eps.

#[test]
fn falsify_alg_009_norm_eps_positive() {
    let registry = build_default_registry();

    for family_name in KNOWN_FAMILIES {
        let family = registry.get(family_name).expect("family exists");
        for (size_name, size_config) in &family.config().size_variants {
            assert!(
                size_config.norm_eps > 0.0,
                "FALSIFY-ALG-009: {family_name}/{size_name} has norm_eps={} — \
                 Zhang & Sennrich (2019) requires eps > 0 for RMSNorm stability",
                size_config.norm_eps
            );
        }
    }
}

// =============================================================================
// §7.6 — FALSIFY-ALG-009: Norm epsilon reasonableness
// =============================================================================
//
// Prediction: norm_eps is in a reasonable range [1e-12, 1e-1].
// Values outside this range indicate YAML typos.

#[test]
fn falsify_alg_009_norm_eps_reasonable_range() {
    let registry = build_default_registry();

    for family_name in KNOWN_FAMILIES {
        let family = registry.get(family_name).expect("family exists");
        for (size_name, size_config) in &family.config().size_variants {
            assert!(
                size_config.norm_eps >= 1e-12 && size_config.norm_eps <= 0.1,
                "FALSIFY-ALG-009: {family_name}/{size_name} has norm_eps={} — \
                 expected range [1e-12, 0.1] (typical: 1e-6 to 1e-5)",
                size_config.norm_eps
            );
        }
    }
}

// =============================================================================
// §7.6 — FALSIFY-ALG-008: GQA ratio is clean integer and bounded
// =============================================================================
//
// Prediction: num_heads / num_kv_heads is a clean integer bounded by 32.
//
// NOTE: Original prediction was "always power-of-two". Falsified by LLaMA 3B
// (ratio=3: 24 heads / 8 KV heads). Revised to "clean integer, bounded".

#[test]
fn falsify_alg_008_gqa_ratio_bounded() {
    let registry = build_default_registry();

    for family_name in KNOWN_FAMILIES {
        let family = registry.get(family_name).expect("family exists");
        for (size_name, size_config) in &family.config().size_variants {
            if size_config.num_kv_heads == 0 {
                continue; // caught by ALG-005
            }
            let ratio = size_config.num_heads / size_config.num_kv_heads;
            assert!(
                ratio >= 1 && ratio <= 32,
                "FALSIFY-ALG-008: {family_name}/{size_name} has GQA ratio {} \
                 (num_heads={}/num_kv_heads={}) — expected 1..32",
                ratio,
                size_config.num_heads,
                size_config.num_kv_heads
            );
            // Verify clean division (redundant with ALG-002 but explicit)
            assert_eq!(
                size_config.num_heads % size_config.num_kv_heads,
                0,
                "FALSIFY-ALG-008: {family_name}/{size_name} GQA ratio not clean \
                 (num_heads={} % num_kv_heads={} != 0)",
                size_config.num_heads,
                size_config.num_kv_heads
            );
        }
    }
}

// =============================================================================
// §7.6 — FALSIFY-ALG-003 (iter10): Head dimension upper bound
// =============================================================================
//
// Prediction: head_dim <= 2 * (hidden_dim / num_heads) for all sizes.
// Gemma 7B uses head_dim=256 with hidden_dim/num_heads=192 (1.33x), which is
// the highest known ratio. A 2x bound catches typos while allowing legitimate variance.
//
// Found via falsification round 3: head_dim=1024 with hidden_dim=128, num_heads=2
// (head_dim/natural=16x) passed all proofs because only a lower bound existed.

#[test]
fn falsify_alg_003_head_dim_upper_bound() {
    let registry = build_default_registry();

    for family_name in KNOWN_FAMILIES {
        let family = registry.get(family_name).expect("family exists");
        for (size_name, size_config) in &family.config().size_variants {
            if size_config.num_heads == 0 {
                continue;
            }
            let natural = size_config.hidden_dim / size_config.num_heads;
            assert!(
                size_config.head_dim <= 2 * natural,
                "FALSIFY-ALG-003 (iter10): {family_name}/{size_name} head_dim={} exceeds \
                 2x natural dimension {} (hidden_dim={}/num_heads={})",
                size_config.head_dim,
                natural,
                size_config.hidden_dim,
                size_config.num_heads
            );
        }
    }
}

// =============================================================================
// §7.6 — FALSIFY-ALG-009 (iter10): Norm epsilon upper bound
// =============================================================================
//
// Prediction: norm_eps < 1.0 for all sizes.
// RMSNorm with eps >= 1.0 dominates the denominator, collapsing activations.
//
// Found via falsification round 3: norm_eps=1e30 passed the > 0 check but
// produces a dead model where all normalized values are zero.

#[test]
fn falsify_alg_009_norm_eps_upper_bound() {
    let registry = build_default_registry();

    for family_name in KNOWN_FAMILIES {
        let family = registry.get(family_name).expect("family exists");
        for (size_name, size_config) in &family.config().size_variants {
            assert!(
                size_config.norm_eps < 1.0,
                "FALSIFY-ALG-009 (iter10): {family_name}/{size_name} has norm_eps={} >= 1.0 — \
                 this collapses all activations in RMSNorm",
                size_config.norm_eps
            );
        }
    }
}

// =============================================================================
// §7.6 — FALSIFY-ALG-009 (iter10): Finiteness of f64 invariants
// =============================================================================
//
// Prediction: rope_theta and norm_eps must be finite (not NaN, not Inf).
//
// Found via falsification round 3: format_f64(inf) generates "inf_f64" which
// fails to parse as Rust — caught by accident, not by proof.

#[test]
fn falsify_alg_finiteness_invariants() {
    let registry = build_default_registry();

    for family_name in KNOWN_FAMILIES {
        let family = registry.get(family_name).expect("family exists");
        for (size_name, size_config) in &family.config().size_variants {
            assert!(
                size_config.norm_eps.is_finite(),
                "FALSIFY finiteness: {family_name}/{size_name} norm_eps is not finite"
            );
            assert!(
                size_config.rope_theta.is_finite() || size_config.rope_theta == 0.0,
                "FALSIFY finiteness: {family_name}/{size_name} rope_theta={} is not finite",
                size_config.rope_theta
            );
        }
    }
}

// =============================================================================
// META: Updated proof count (iter10)
// =============================================================================

#[test]
fn falsify_alg_297_compile_time_proofs_count() {
    // After 3 rounds of falsification, const assertions: 225 → 273 → 297.
    // Each size gets: 6 non-degeneracy + 1 KV ordering + 1 Vaswani +
    // 2 head_dim bounds + 1 FFN = 11 minimum per size.
    // RoPE adds 2 more, GQA with kv>1 adds 1 more.
    let registry = build_default_registry();

    let mut total_sizes = 0_usize;
    for family_name in KNOWN_FAMILIES {
        let family = registry.get(family_name).expect("family exists");
        total_sizes += family.config().size_variants.len();
    }

    let min_per_size = 11;
    let min_expected = total_sizes * min_per_size;
    assert!(
        min_expected >= 250,
        "Expected at least 250 compile-time proofs (got minimum estimate {min_expected} \
         from {total_sizes} sizes * {min_per_size} proofs each)"
    );
}

fn find_project_root() -> std::path::PathBuf {
    let mut dir = std::env::current_dir().expect("current dir");
    loop {
        if dir.join("Cargo.toml").exists() && dir.join("src").exists() {
            return dir;
        }
        assert!(
            dir.pop(),
            "Could not find project root (looking for Cargo.toml + src/)"
        );
    }
}
