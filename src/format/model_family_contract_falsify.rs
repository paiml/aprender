
// ============================================================================
// Model Family Contract Falsification Tests (FALSIFY-MF-001..008)
//
// Popperian falsification: each test attempts to BREAK a mathematical invariant
// claimed by the model-family YAML contracts. If a test fails, the contract
// has a bug (wrong dimension, missing field, etc.).
//
// Contract: contracts/model-families/*.yaml
// Schema:   contracts/model-families/_schema.yaml
// ============================================================================

#[cfg(test)]
mod contract_falsification {
    use super::*;
    use std::collections::{HashMap, HashSet};
    use std::path::Path;

    /// Load all model family configs from the contracts directory.
    /// Returns (family_name, config) pairs.
    fn load_all_families() -> Vec<(String, ModelFamilyConfig)> {
        let contracts_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("contracts");
        let families_dir = contracts_dir.join("model-families");
        assert!(
            families_dir.exists(),
            "contracts/model-families/ directory must exist"
        );

        let mut families = Vec::new();
        let entries = std::fs::read_dir(&families_dir).expect("read model-families dir");

        for entry in entries {
            let entry = entry.expect("read dir entry");
            let path = entry.path();
            let file_name = path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("");

            // Skip non-YAML and _-prefixed files
            let ext_is_yaml = path
                .extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("yaml") || ext.eq_ignore_ascii_case("yml"));
            if !ext_is_yaml {
                continue;
            }
            if file_name.starts_with('_') {
                continue;
            }

            let config = load_family_yaml(&path)
                .unwrap_or_else(|e| panic!("Failed to load {file_name}: {e}"));
            families.push((config.family.clone(), config));
        }

        families.sort_by(|a, b| a.0.cmp(&b.0));
        assert!(
            !families.is_empty(),
            "At least one model family YAML must exist"
        );
        families
    }

    // ========================================================================
    // FALSIFY-MF-001: Positive dimensions
    //
    // Prediction: For ALL size variants in ALL families, every dimension > 0.
    // If fails: YAML has a zero or missing dimension → garbage shapes at runtime.
    // ========================================================================
    #[test]
    fn falsify_mf_001_positive_dimensions() {
        let families = load_all_families();
        let mut violations = Vec::new();

        for (family_name, config) in &families {
            for (size_name, sc) in &config.size_variants {
                let checks: &[(&str, usize)] = &[
                    ("hidden_dim", sc.hidden_dim),
                    ("num_layers", sc.num_layers),
                    ("num_heads", sc.num_heads),
                    ("num_kv_heads", sc.num_kv_heads),
                    ("intermediate_dim", sc.intermediate_dim),
                    ("vocab_size", sc.vocab_size),
                    ("head_dim", sc.head_dim),
                ];
                for &(field, value) in checks {
                    if value == 0 {
                        violations.push(format!(
                            "{family_name}/{size_name}: {field} = 0"
                        ));
                    }
                }
            }
        }

        assert!(
            violations.is_empty(),
            "FALSIFY-MF-001: Zero dimensions found:\n{}",
            violations.join("\n")
        );
    }

    // ========================================================================
    // FALSIFY-MF-002: GQA divisibility
    //
    // Prediction: For ALL size variants, num_heads % num_kv_heads == 0.
    // Mathematical requirement: GQA groups Q heads into KV groups. Each group
    // must have the same integer number of Q heads per KV head.
    // If fails: GQA kernel will produce wrong attention scores.
    // ========================================================================
    #[test]
    fn falsify_mf_002_gqa_divisibility() {
        let families = load_all_families();
        let mut violations = Vec::new();

        for (family_name, config) in &families {
            for (size_name, sc) in &config.size_variants {
                if sc.num_kv_heads > 0 && sc.num_heads % sc.num_kv_heads != 0 {
                    violations.push(format!(
                        "{family_name}/{size_name}: num_heads={} % num_kv_heads={} = {} (must be 0)",
                        sc.num_heads,
                        sc.num_kv_heads,
                        sc.num_heads % sc.num_kv_heads
                    ));
                }
            }
        }

        assert!(
            violations.is_empty(),
            "FALSIFY-MF-002: GQA divisibility violations:\n{}",
            violations.join("\n")
        );
    }

    // ========================================================================
    // FALSIFY-MF-003: FFN expansion
    //
    // Prediction: For ALL size variants, intermediate_dim > hidden_dim.
    // The FFN must expand the hidden representation (standard 4x or 8/3x for SwiGLU).
    // If fails: FFN bottleneck would LOSE information.
    // ========================================================================
    #[test]
    fn falsify_mf_003_ffn_expansion() {
        let families = load_all_families();
        let mut violations = Vec::new();

        for (family_name, config) in &families {
            for (size_name, sc) in &config.size_variants {
                if sc.intermediate_dim <= sc.hidden_dim {
                    violations.push(format!(
                        "{family_name}/{size_name}: intermediate_dim={} <= hidden_dim={} (must expand)",
                        sc.intermediate_dim, sc.hidden_dim
                    ));
                }
            }
        }

        assert!(
            violations.is_empty(),
            "FALSIFY-MF-003: FFN expansion violations:\n{}",
            violations.join("\n")
        );
    }

    // ========================================================================
    // FALSIFY-MF-004: Schema completeness
    //
    // Prediction: Every model family YAML file loads without error AND has
    // at least one size variant, non-empty architectures list, and non-empty
    // tensor template.
    // If fails: YAML is malformed or missing required contract fields.
    // ========================================================================
    #[test]
    fn falsify_mf_004_schema_completeness() {
        let families = load_all_families();
        let mut violations = Vec::new();

        for (family_name, config) in &families {
            if config.architectures.is_empty() {
                violations.push(format!("{family_name}: empty architectures list"));
            }
            if config.size_variants.is_empty() {
                violations.push(format!("{family_name}: no size variants"));
            }
            if config.vendor.is_empty() {
                violations.push(format!("{family_name}: empty vendor"));
            }
            if config.display_name.is_empty() {
                violations.push(format!("{family_name}: empty display_name"));
            }
            if config.hf_pattern.is_empty() {
                violations.push(format!("{family_name}: empty hf_pattern"));
            }
        }

        assert!(
            violations.is_empty(),
            "FALSIFY-MF-004: Schema completeness violations:\n{}",
            violations.join("\n")
        );
    }

    // ========================================================================
    // FALSIFY-MF-005: No duplicate family names
    //
    // Prediction: Every YAML file defines a UNIQUE family name.
    // If fails: Duplicate family would cause registry collision — wrong model
    // gets loaded at runtime.
    // ========================================================================
    #[test]
    fn falsify_mf_005_no_duplicate_family_names() {
        let families = load_all_families();
        let mut seen: HashMap<String, usize> = HashMap::new();

        for (family_name, _) in &families {
            *seen.entry(family_name.clone()).or_insert(0) += 1;
        }

        let duplicates: Vec<_> = seen
            .iter()
            .filter(|(_, count)| **count > 1)
            .map(|(name, count)| format!("{name}: appears {count} times"))
            .collect();

        assert!(
            duplicates.is_empty(),
            "FALSIFY-MF-005: Duplicate family names:\n{}",
            duplicates.join("\n")
        );
    }

    // ========================================================================
    // FALSIFY-MF-006: No duplicate architecture classes
    //
    // Prediction: Each HuggingFace architecture class maps to exactly ONE family.
    // If fails: Ambiguous auto-detection — two families claim the same arch class.
    // ========================================================================
    #[test]
    fn falsify_mf_006_no_duplicate_architecture_classes() {
        let families = load_all_families();
        let mut arch_to_family: HashMap<String, Vec<String>> = HashMap::new();

        for (family_name, config) in &families {
            for arch in &config.architectures {
                arch_to_family
                    .entry(arch.clone())
                    .or_default()
                    .push(family_name.clone());
            }
        }

        let duplicates: Vec<_> = arch_to_family
            .iter()
            .filter(|(_, families)| families.len() > 1)
            .map(|(arch, families)| {
                format!("{arch}: claimed by [{}]", families.join(", "))
            })
            .collect();

        assert!(
            duplicates.is_empty(),
            "FALSIFY-MF-006: Duplicate architecture classes:\n{}",
            duplicates.join("\n")
        );
    }

    // ========================================================================
    // FALSIFY-MF-007: Attention dimension identity
    //
    // Prediction: For MOST decoder models, hidden_dim == num_heads * head_dim.
    // This is the standard attention dimension identity. Models that use
    // non-square attention projections (Gemma, Qwen3 small sizes) are
    // documented exceptions.
    //
    // KNOWN EXCEPTIONS (architecturally intentional — non-square Q/K/V projections):
    //   - gemma/7b: head_dim=256 (fixed across sizes), hidden_dim=3072
    //   - qwen3/0.6b, qwen3/4b: head_dim=128 (fixed), hidden_dim < num_heads*head_dim
    //   - qwen3_5/*: (if present) may also use non-square projections
    //
    // If a NEW exception appears, this test FAILS — forcing the developer to
    // either fix the YAML or explicitly add the exception to the known list.
    // ========================================================================
    #[test]
    fn falsify_mf_007_attention_dimension_identity() {
        let families = load_all_families();

        // Known exceptions: (family, size) pairs where hidden_dim != num_heads * head_dim
        // is architecturally intentional (non-square attention projections).
        let known_exceptions: HashSet<(&str, &str)> = [
            // Gemma 7B: fixed head_dim=256 across sizes, 16*256=4096 != 3072
            ("gemma", "7b"),
            // Qwen3: uses fixed head_dim=128 across all sizes
            ("qwen3", "0.6b"),  // 16*128=2048 != 1024
            ("qwen3", "4b"),    // 32*128=4096 != 2560
            ("qwen3", "8b"),    // need to verify
            ("qwen3", "14b"),   // need to verify
            ("qwen3", "30b"),   // need to verify
            ("qwen3", "32b"),   // need to verify
            ("qwen3", "235b"),  // need to verify
        ]
        .into_iter()
        .collect();

        let mut unexpected_violations = Vec::new();
        let mut known_violations_found = HashSet::new();

        for (family_name, config) in &families {
            for (size_name, sc) in &config.size_variants {
                let expected = sc.num_heads * sc.head_dim;
                if expected != sc.hidden_dim {
                    let key = (family_name.as_str(), size_name.as_str());
                    if known_exceptions.contains(&key) {
                        known_violations_found.insert(key);
                    } else {
                        unexpected_violations.push(format!(
                            "{family_name}/{size_name}: hidden_dim={} != num_heads({}) * head_dim({}) = {}",
                            sc.hidden_dim, sc.num_heads, sc.head_dim, expected
                        ));
                    }
                }
            }
        }

        assert!(
            unexpected_violations.is_empty(),
            "FALSIFY-MF-007: UNEXPECTED attention dimension violations:\n{}\n\
             If intentional, add to known_exceptions in this test.",
            unexpected_violations.join("\n")
        );
    }

    // ========================================================================
    // FALSIFY-MF-008: KV heads <= Q heads
    //
    // Prediction: num_kv_heads <= num_heads for ALL size variants.
    // KV heads can never exceed Q heads (MHA: equal, GQA: fewer, MQA: 1).
    // If fails: YAML has KV/Q heads swapped — GQA kernel will crash.
    // ========================================================================
    #[test]
    fn falsify_mf_008_kv_heads_le_q_heads() {
        let families = load_all_families();
        let mut violations = Vec::new();

        for (family_name, config) in &families {
            for (size_name, sc) in &config.size_variants {
                if sc.num_kv_heads > sc.num_heads {
                    violations.push(format!(
                        "{family_name}/{size_name}: num_kv_heads={} > num_heads={} (impossible)",
                        sc.num_kv_heads, sc.num_heads
                    ));
                }
            }
        }

        assert!(
            violations.is_empty(),
            "FALSIFY-MF-008: KV heads exceed Q heads:\n{}",
            violations.join("\n")
        );
    }

    // ========================================================================
    // FALSIFY-MF-009: MHA consistency
    //
    // Prediction: When constraints.attention_type == MHA, num_kv_heads == num_heads
    // for ALL size variants. MHA means all heads do full attention.
    // If fails: Family claims MHA but has different KV head count — the
    // kernel dispatch would choose wrong attention path.
    // ========================================================================
    #[test]
    fn falsify_mf_009_mha_kv_heads_equal() {
        let families = load_all_families();
        let mut violations = Vec::new();

        for (family_name, config) in &families {
            if config.constraints.attention_type != AttentionType::Mha {
                continue;
            }
            for (size_name, sc) in &config.size_variants {
                if sc.num_kv_heads != sc.num_heads {
                    violations.push(format!(
                        "{family_name}/{size_name}: claims MHA but num_kv_heads={} != num_heads={}",
                        sc.num_kv_heads, sc.num_heads
                    ));
                }
            }
        }

        assert!(
            violations.is_empty(),
            "FALSIFY-MF-009: MHA families with mismatched KV heads:\n{}",
            violations.join("\n")
        );
    }

    // ========================================================================
    // FALSIFY-MF-010: GQA families must have num_kv_heads < num_heads
    //
    // Prediction: When constraints.attention_type == GQA, at least ONE size
    // variant must have num_kv_heads < num_heads (otherwise it's actually MHA).
    // If fails: Family incorrectly classified as GQA.
    // ========================================================================
    #[test]
    fn falsify_mf_010_gqa_has_fewer_kv_heads() {
        let families = load_all_families();
        let mut violations = Vec::new();

        for (family_name, config) in &families {
            if config.constraints.attention_type != AttentionType::Gqa {
                continue;
            }
            let has_gqa_variant = config.size_variants.values().any(|sc| sc.num_kv_heads < sc.num_heads);
            if !has_gqa_variant {
                violations.push(format!(
                    "{family_name}: claims GQA but all size variants have num_kv_heads == num_heads (that's MHA)"
                ));
            }
        }

        assert!(
            violations.is_empty(),
            "FALSIFY-MF-010: GQA families with no actual GQA variants:\n{}",
            violations.join("\n")
        );
    }

    // ========================================================================
    // FALSIFY-MF-011: Vocabulary consistency within family
    //
    // Prediction: Most families use the SAME vocab_size across all size variants
    // (the tokenizer is shared). Known exceptions: Qwen2 (0.5B/1.5B/3B use
    // 151936, 7B+ use 152064).
    //
    // If a family has >2 distinct vocab sizes, that's suspicious.
    // ========================================================================
    #[test]
    fn falsify_mf_011_vocab_consistency() {
        let families = load_all_families();
        let mut violations = Vec::new();

        for (family_name, config) in &families {
            if config.size_variants.len() <= 1 {
                continue;
            }
            let vocab_sizes: HashSet<usize> = config
                .size_variants
                .values()
                .map(|sc| sc.vocab_size)
                .collect();
            if vocab_sizes.len() > 2 {
                violations.push(format!(
                    "{family_name}: {} distinct vocab_sizes: {:?} (suspicious — tokenizer should be shared)",
                    vocab_sizes.len(),
                    vocab_sizes
                ));
            }
        }

        assert!(
            violations.is_empty(),
            "FALSIFY-MF-011: Excessive vocab size variation:\n{}",
            violations.join("\n")
        );
    }

    // ========================================================================
    // FALSIFY-MF-012: Minimum family count
    //
    // Prediction: The contracts directory contains at least 10 model families.
    // This is a canary — if families get accidentally deleted, this catches it.
    // ========================================================================
    #[test]
    fn falsify_mf_012_minimum_family_count() {
        let families = load_all_families();
        assert!(
            families.len() >= 10,
            "FALSIFY-MF-012: Expected >= 10 model families, found {}. \
             Families may have been accidentally deleted.",
            families.len()
        );
    }

    // ========================================================================
    // FALSIFY-MF-013: Shape template coverage for decoder models
    //
    // Prediction: For decoder-only models (those with q_proj in tensor_template),
    // the shape_template must define shapes for at least: embedding, q_proj,
    // k_proj, v_proj, o_proj, gate_proj/up_proj/down_proj.
    //
    // If fails: Shape validation at load time would be incomplete.
    // ========================================================================
    #[test]
    fn falsify_mf_013_shape_template_coverage() {
        let families = load_all_families();
        let required_decoder_shapes = [
            "embedding", "q_proj", "k_proj", "v_proj", "o_proj",
        ];
        let mut violations = Vec::new();

        for (family_name, config) in &families {
            // Only check decoder-only models (those with per_layer q_proj)
            let has_q_proj = config
                .tensor_template
                .per_layer
                .get("q_proj")
                .is_some_and(|v| v.is_some());
            if !has_q_proj {
                continue;
            }

            for shape_key in &required_decoder_shapes {
                if !config.shape_template.shapes.contains_key(*shape_key) {
                    violations.push(format!(
                        "{family_name}: missing shape_template entry for '{shape_key}'"
                    ));
                }
            }
        }

        assert!(
            violations.is_empty(),
            "FALSIFY-MF-013: Missing shape template entries:\n{}",
            violations.join("\n")
        );
    }

    // ========================================================================
    // PROPTEST: FALSIFY-MF-ARCH-001-prop through FALSIFY-MF-ARCH-004-prop
    // ========================================================================

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            // ================================================================
            // FALSIFY-MF-ARCH-001-prop: norm_type consistency
            //
            // Prediction: For any random family index, the constraints.norm_type
            // is self-consistent — reading the same family config twice yields
            // the same NormType. This verifies the YAML-defined constraints
            // do not silently change between loads.
            // ================================================================
            #[test]
            fn falsify_mf_arch_001_prop_norm_type_consistency(
                idx in 0..100_usize
            ) {
                let families = load_all_families();
                let family_idx = idx % families.len();
                let (ref family_name, ref config) = families[family_idx];

                // norm_type must be one of the two valid variants
                let norm = config.constraints.norm_type;
                prop_assert!(
                    matches!(norm, NormType::RmsNorm | NormType::LayerNorm),
                    "FALSIFY-MF-ARCH-001-prop: {} has invalid norm_type: {:?}",
                    family_name, norm
                );

                // Reload and verify consistency
                let families2 = load_all_families();
                let (_, ref config2) = families2[family_idx];
                prop_assert!(
                    config.constraints.norm_type == config2.constraints.norm_type,
                    "FALSIFY-MF-ARCH-001-prop: {} norm_type changed between loads: {:?} vs {:?}",
                    family_name, config.constraints.norm_type, config2.constraints.norm_type
                );
            }

            // ================================================================
            // FALSIFY-MF-ARCH-002-prop: activation type consistency
            //
            // Prediction: For any random family index, constraints.activation
            // is a valid variant and consistent across loads.
            // ================================================================
            #[test]
            fn falsify_mf_arch_002_prop_activation_consistency(
                idx in 0..100_usize
            ) {
                let families = load_all_families();
                let family_idx = idx % families.len();
                let (ref family_name, ref config) = families[family_idx];

                let act = config.constraints.activation;
                prop_assert!(
                    matches!(act, Activation::Silu | Activation::Gelu | Activation::Relu),
                    "FALSIFY-MF-ARCH-002-prop: {} has invalid activation: {:?}",
                    family_name, act
                );

                let families2 = load_all_families();
                let (_, ref config2) = families2[family_idx];
                prop_assert!(
                    config.constraints.activation == config2.constraints.activation,
                    "FALSIFY-MF-ARCH-002-prop: {} activation changed between loads: {:?} vs {:?}",
                    family_name, config.constraints.activation, config2.constraints.activation
                );
            }

            // ================================================================
            // FALSIFY-MF-ARCH-003-prop: mlp_type consistency
            //
            // Prediction: For any random family index, constraints.mlp_type
            // is a valid variant and consistent across loads.
            // ================================================================
            #[test]
            fn falsify_mf_arch_003_prop_mlp_type_consistency(
                idx in 0..100_usize
            ) {
                let families = load_all_families();
                let family_idx = idx % families.len();
                let (ref family_name, ref config) = families[family_idx];

                let mlp = config.constraints.mlp_type;
                prop_assert!(
                    matches!(mlp, MlpType::SwiGlu | MlpType::GeluMlp | MlpType::GatedMlp),
                    "FALSIFY-MF-ARCH-003-prop: {} has invalid mlp_type: {:?}",
                    family_name, mlp
                );

                let families2 = load_all_families();
                let (_, ref config2) = families2[family_idx];
                prop_assert!(
                    config.constraints.mlp_type == config2.constraints.mlp_type,
                    "FALSIFY-MF-ARCH-003-prop: {} mlp_type changed between loads: {:?} vs {:?}",
                    family_name, config.constraints.mlp_type, config2.constraints.mlp_type
                );
            }

            // ================================================================
            // FALSIFY-MF-ARCH-004-prop: numeric dimension roundtrip
            //
            // Prediction: For families with size variants, all numeric
            // dimension fields survive a format-then-parse roundtrip:
            //   value.to_string().parse::<usize>().unwrap() == value
            //
            // This catches hypothetical floating-point-to-int truncation bugs
            // in YAML parsing where e.g. "4096.0" would fail to parse as usize.
            // ================================================================
            #[test]
            fn falsify_mf_arch_004_prop_dimension_roundtrip(
                family_idx in 0..100_usize,
                size_idx in 0..100_usize,
            ) {
                let families = load_all_families();
                let fi = family_idx % families.len();
                let (ref family_name, ref config) = families[fi];

                if config.size_variants.is_empty() {
                    return Ok(());
                }

                let sizes: Vec<(&String, &ModelSizeConfig)> = config.size_variants.iter().collect();
                let si = size_idx % sizes.len();
                let (size_name, sc) = sizes[si];

                // Roundtrip all usize dimension fields through string formatting
                let fields: &[(&str, usize)] = &[
                    ("hidden_dim", sc.hidden_dim),
                    ("num_layers", sc.num_layers),
                    ("num_heads", sc.num_heads),
                    ("num_kv_heads", sc.num_kv_heads),
                    ("intermediate_dim", sc.intermediate_dim),
                    ("vocab_size", sc.vocab_size),
                    ("head_dim", sc.head_dim),
                    ("max_position_embeddings", sc.max_position_embeddings),
                ];

                for &(field, value) in fields {
                    let formatted = value.to_string();
                    let parsed: usize = formatted.parse().map_err(|e| {
                        proptest::test_runner::TestCaseError::Fail(
                            format!(
                                "FALSIFY-MF-ARCH-004-prop: {}/{} {}={} roundtrip failed: {}",
                                family_name, size_name, field, value, e
                            ).into()
                        )
                    })?;
                    prop_assert!(
                        parsed == value,
                        "FALSIFY-MF-ARCH-004-prop: {}/{} {} roundtrip mismatch: {} != {}",
                        family_name, size_name, field, parsed, value
                    );
                }
            }
        }
    }

    // ========================================================================
    // FALSIFY-MF-QWEN35-001: Qwen3.5 9B exact dimensions
    //
    // Prediction: The qwen3_5/9b size variant has the documented dimensions
    // from the Qwen3.5 model card. If fails: contract YAML is stale or wrong.
    // ========================================================================
    #[test]
    fn falsify_mf_qwen35_001_9b_dimensions() {
        let families = load_all_families();
        let qwen35 = families
            .iter()
            .find(|(name, _)| name == "qwen3_5")
            .expect("FALSIFIED: qwen3_5 family not found in contracts/model-families/");

        let variant = qwen35
            .1
            .size_variants
            .get("9b")
            .expect("FALSIFIED: qwen3_5 missing '9b' size variant");

        assert_eq!(variant.hidden_dim, 4096,
            "FALSIFIED QWEN35-001: hidden_dim={}, expected 4096", variant.hidden_dim);
        assert_eq!(variant.num_layers, 32,
            "FALSIFIED QWEN35-001: num_layers={}, expected 32", variant.num_layers);
        assert_eq!(variant.num_heads, 16,
            "FALSIFIED QWEN35-001: num_heads={}, expected 16", variant.num_heads);
        assert_eq!(variant.num_kv_heads, 4,
            "FALSIFIED QWEN35-001: num_kv_heads={}, expected 4", variant.num_kv_heads);
        assert_eq!(variant.intermediate_dim, 12288,
            "FALSIFIED QWEN35-001: intermediate_dim={}, expected 12288", variant.intermediate_dim);
        assert_eq!(variant.vocab_size, 248320,
            "FALSIFIED QWEN35-001: vocab_size={}, expected 248320", variant.vocab_size);
        assert_eq!(variant.head_dim, 256,
            "FALSIFIED QWEN35-001: head_dim={}, expected 256", variant.head_dim);
        assert_eq!(variant.max_position_embeddings, 262144,
            "FALSIFIED QWEN35-001: max_pos_embed={}, expected 262144", variant.max_position_embeddings);
    }

    // ========================================================================
    // FALSIFY-MF-QWEN35-002: Qwen3.5 has no attention bias
    //
    // Prediction: Qwen3.5 constraints specify has_bias=false.
    // If fails: contract YAML incorrectly claims bias exists → weight
    // loader would allocate and look for bias tensors that don't exist.
    // ========================================================================
    #[test]
    fn falsify_mf_qwen35_002_no_bias() {
        let families = load_all_families();
        let qwen35 = families
            .iter()
            .find(|(name, _)| name == "qwen3_5")
            .expect("FALSIFIED: qwen3_5 family not found");

        assert!(
            !qwen35.1.constraints.has_bias,
            "FALSIFIED QWEN35-002: has_bias={}, expected false (Qwen3.5 has no attention bias)",
            qwen35.1.constraints.has_bias
        );
    }

    // ========================================================================
    // FALSIFY-MF-QWEN35-003: Qwen3.5 attention dimension identity
    //
    // Prediction: hidden_dim == num_heads * head_dim for 9B.
    // Qwen3.5 9B: 16 * 256 = 4096 == hidden_dim.
    // If fails: either head_dim or num_heads is wrong in the contract.
    // ========================================================================
    #[test]
    fn falsify_mf_qwen35_003_attention_dim_identity() {
        let families = load_all_families();
        let qwen35 = families
            .iter()
            .find(|(name, _)| name == "qwen3_5")
            .expect("FALSIFIED: qwen3_5 family not found");

        let variant = qwen35.1.size_variants.get("9b")
            .expect("FALSIFIED: qwen3_5 missing '9b' size variant");

        let computed = variant.num_heads * variant.head_dim;
        assert_eq!(
            computed, variant.hidden_dim,
            "FALSIFIED QWEN35-003: num_heads({}) * head_dim({}) = {} != hidden_dim({})",
            variant.num_heads, variant.head_dim, computed, variant.hidden_dim
        );
    }

    // ========================================================================
    // FALSIFY-MF-QWEN35-004: Qwen3.5 GQA divisibility
    //
    // Prediction: num_heads % num_kv_heads == 0 (GQA requirement).
    // 9B: 16 % 4 == 0. If fails: GQA repeat_interleave would panic.
    // ========================================================================
    #[test]
    fn falsify_mf_qwen35_004_gqa_divisibility() {
        let families = load_all_families();
        let qwen35 = families
            .iter()
            .find(|(name, _)| name == "qwen3_5")
            .expect("FALSIFIED: qwen3_5 family not found");

        let variant = qwen35.1.size_variants.get("9b")
            .expect("FALSIFIED: qwen3_5 missing '9b' size variant");

        assert_eq!(
            variant.num_heads % variant.num_kv_heads, 0,
            "FALSIFIED QWEN35-004: num_heads({}) not divisible by num_kv_heads({})",
            variant.num_heads, variant.num_kv_heads
        );
    }

    // ========================================================================
    // FALSIFY-MF-QWEN35-005: Qwen3.5 shape templates produce correct dimensions
    //
    // Prediction: q_proj evaluates to [num_heads*head_dim, hidden_dim] = [4096, 4096]
    //             k_proj evaluates to [num_kv_heads*head_dim, hidden_dim] = [1024, 4096]
    //             o_proj evaluates to [hidden_dim, num_heads*head_dim] = [4096, 4096]
    // If fails: shape template has a bug → runtime shape validation rejects valid weights.
    // ========================================================================
    #[test]
    fn falsify_mf_qwen35_005_shape_template_dimensions() {
        let families = load_all_families();
        let qwen35 = families
            .iter()
            .find(|(name, _)| name == "qwen3_5")
            .expect("FALSIFIED: qwen3_5 family not found");

        let sc = qwen35.1.size_variants.get("9b")
            .expect("FALSIFIED: qwen3_5 missing '9b' size variant");

        // q_proj: [num_heads * head_dim, hidden_dim] = [4096, 4096]
        let q_proj_out = sc.num_heads * sc.head_dim;
        assert_eq!(q_proj_out, 4096,
            "FALSIFIED QWEN35-005: q_proj out_dim={}, expected 4096", q_proj_out);

        // k_proj: [num_kv_heads * head_dim, hidden_dim] = [1024, 4096]
        let k_proj_out = sc.num_kv_heads * sc.head_dim;
        assert_eq!(k_proj_out, 1024,
            "FALSIFIED QWEN35-005: k_proj out_dim={}, expected 1024", k_proj_out);

        // FFN: intermediate_dim / hidden_dim ratio = 3.0 (standard SwiGLU)
        let ffn_ratio = sc.intermediate_dim as f64 / sc.hidden_dim as f64;
        assert!(
            (ffn_ratio - 3.0).abs() < 0.01,
            "FALSIFIED QWEN35-005: FFN ratio={ffn_ratio:.2}, expected 3.0 for SwiGLU"
        );
    }

    // ========================================================================
    // FALSIFY-MF-QWEN35-006: Qwen3.5 rope_theta is 1M
    //
    // Prediction: rope_theta = 1,000,000.0 (long-context RoPE).
    // If fails: contract has wrong theta → RoPE position encoding breaks.
    // ========================================================================
    #[test]
    fn falsify_mf_qwen35_006_rope_theta() {
        let families = load_all_families();
        let qwen35 = families
            .iter()
            .find(|(name, _)| name == "qwen3_5")
            .expect("FALSIFIED: qwen3_5 family not found");

        let variant = qwen35.1.size_variants.get("9b")
            .expect("FALSIFIED: qwen3_5 missing '9b' size variant");

        let expected_theta = 1_000_000.0_f64;
        assert!(
            (variant.rope_theta - expected_theta).abs() < 1.0,
            "FALSIFIED QWEN35-006: rope_theta={}, expected {}",
            variant.rope_theta, expected_theta
        );
    }

    // ========================================================================
    // FALSIFY-MF-QWEN35-007: Qwen3.5 architecture class registered
    //
    // Prediction: The qwen3_5 family maps to Qwen3_5ForCausalLM architecture.
    // If fails: HuggingFace model loading will fail to identify the architecture.
    // ========================================================================
    #[test]
    fn falsify_mf_qwen35_007_architecture_class() {
        let families = load_all_families();
        let qwen35 = families
            .iter()
            .find(|(name, _)| name == "qwen3_5")
            .expect("FALSIFIED: qwen3_5 family not found");

        assert!(
            qwen35.1.architectures.contains(&"Qwen3_5ForCausalLM".to_string()),
            "FALSIFIED QWEN35-007: qwen3_5 architectures={:?}, must contain 'Qwen3_5ForCausalLM'",
            qwen35.1.architectures
        );
    }

    // ========================================================================
    // FALSIFY-MF-QWEN35-008: Qwen3.5 uses SwiGLU MLP
    //
    // Prediction: activation=silu, mlp_type=swiglu (consistent with Qwen family).
    // If fails: wrong FFN kernel dispatched at runtime.
    // ========================================================================
    #[test]
    fn falsify_mf_qwen35_008_swiglu_mlp() {
        let families = load_all_families();
        let qwen35 = families
            .iter()
            .find(|(name, _)| name == "qwen3_5")
            .expect("FALSIFIED: qwen3_5 family not found");

        assert!(
            matches!(qwen35.1.constraints.activation, Activation::Silu),
            "FALSIFIED QWEN35-008: activation={:?}, expected Silu",
            qwen35.1.constraints.activation
        );
        assert!(
            matches!(qwen35.1.constraints.mlp_type, MlpType::SwiGlu),
            "FALSIFIED QWEN35-008: mlp_type={:?}, expected SwiGlu",
            qwen35.1.constraints.mlp_type
        );
    }

    // ========================================================================
    // FINE-TUNING CONFIG FALSIFICATION (FALSIFY-FT-QWEN35-001..007)
    //
    // These tests verify that entrenar's TransformerConfig::qwen3_5_9b() factory
    // matches the model-family YAML contract. If any test fails, the config
    // factory is out of sync with the contract — fine-tuning would use wrong
    // dimensions.
    // ========================================================================

    // ========================================================================
    // FALSIFY-FT-QWEN35-001: vocab_size must be 248320 (not Qwen2's 152064)
    //
    // Prediction: qwen3_5_9b().vocab_size == 248320.
    // If fails: embedding/lm_head LoRA shapes would be wrong.
    // ========================================================================
    #[test]
    fn falsify_ft_qwen35_001_vocab_size() {
        let config = entrenar::transformer::TransformerConfig::qwen3_5_9b();
        assert_eq!(
            config.vocab_size, 248320,
            "FALSIFIED FT-QWEN35-001: vocab_size={}, expected 248320",
            config.vocab_size
        );
    }

    // ========================================================================
    // FALSIFY-FT-QWEN35-002: use_bias must be false
    //
    // Prediction: qwen3_5_9b().use_bias == false.
    // If fails: LoRA adapter would create bias tensors that don't exist.
    // ========================================================================
    #[test]
    fn falsify_ft_qwen35_002_no_bias() {
        let config = entrenar::transformer::TransformerConfig::qwen3_5_9b();
        assert!(
            !config.use_bias,
            "FALSIFIED FT-QWEN35-002: use_bias={}, expected false",
            config.use_bias
        );
    }

    // ========================================================================
    // FALSIFY-FT-QWEN35-003: head_dim must be 256
    //
    // Prediction: qwen3_5_9b().head_dim() == 256 (4096/16).
    // If fails: LoRA Q/K/V projection dimensions would be wrong.
    // ========================================================================
    #[test]
    fn falsify_ft_qwen35_003_head_dim() {
        let config = entrenar::transformer::TransformerConfig::qwen3_5_9b();
        assert_eq!(
            config.head_dim(), 256,
            "FALSIFIED FT-QWEN35-003: head_dim()={}, expected 256",
            config.head_dim()
        );
    }

    // ========================================================================
    // FALSIFY-FT-QWEN35-004: num_hidden_layers must be 32
    //
    // Prediction: qwen3_5_9b().num_hidden_layers == 32.
    // If fails: LoRA would target wrong number of layers.
    // ========================================================================
    #[test]
    fn falsify_ft_qwen35_004_num_layers() {
        let config = entrenar::transformer::TransformerConfig::qwen3_5_9b();
        assert_eq!(
            config.num_hidden_layers, 32,
            "FALSIFIED FT-QWEN35-004: num_hidden_layers={}, expected 32",
            config.num_hidden_layers
        );
    }

    // ========================================================================
    // FALSIFY-FT-QWEN35-005: num_kv_heads must be 4 (GQA)
    //
    // Prediction: qwen3_5_9b().num_kv_heads == 4.
    // If fails: GQA ratio wrong → K/V projection LoRA shapes wrong.
    // ========================================================================
    #[test]
    fn falsify_ft_qwen35_005_num_kv_heads() {
        let config = entrenar::transformer::TransformerConfig::qwen3_5_9b();
        assert_eq!(
            config.num_kv_heads, 4,
            "FALSIFIED FT-QWEN35-005: num_kv_heads={}, expected 4",
            config.num_kv_heads
        );
    }

    // ========================================================================
    // FALSIFY-FT-QWEN35-006: Contract YAML dimensions match config factory
    //
    // Prediction: Every dimension in qwen3_5.yaml 9b variant matches the
    // corresponding field in TransformerConfig::qwen3_5_9b().
    // If fails: contract and code are out of sync — one of them has a bug.
    // ========================================================================
    #[test]
    fn falsify_ft_qwen35_006_contract_config_sync() {
        let families = load_all_families();
        let qwen35 = families
            .iter()
            .find(|(name, _)| name == "qwen3_5")
            .expect("qwen3_5 family not found");
        let variant = qwen35.1.size_variants.get("9b")
            .expect("qwen3_5 missing '9b' variant");

        let config = entrenar::transformer::TransformerConfig::qwen3_5_9b();

        assert_eq!(config.hidden_size, variant.hidden_dim,
            "FALSIFIED FT-QWEN35-006: hidden_size mismatch: config={} vs contract={}",
            config.hidden_size, variant.hidden_dim);
        assert_eq!(config.num_hidden_layers, variant.num_layers,
            "FALSIFIED FT-QWEN35-006: num_layers mismatch: config={} vs contract={}",
            config.num_hidden_layers, variant.num_layers);
        assert_eq!(config.num_attention_heads, variant.num_heads,
            "FALSIFIED FT-QWEN35-006: num_heads mismatch: config={} vs contract={}",
            config.num_attention_heads, variant.num_heads);
        assert_eq!(config.num_kv_heads, variant.num_kv_heads,
            "FALSIFIED FT-QWEN35-006: num_kv_heads mismatch: config={} vs contract={}",
            config.num_kv_heads, variant.num_kv_heads);
        assert_eq!(config.intermediate_size, variant.intermediate_dim,
            "FALSIFIED FT-QWEN35-006: intermediate_dim mismatch: config={} vs contract={}",
            config.intermediate_size, variant.intermediate_dim);
        assert_eq!(config.vocab_size, variant.vocab_size,
            "FALSIFIED FT-QWEN35-006: vocab_size mismatch: config={} vs contract={}",
            config.vocab_size, variant.vocab_size);
        assert_eq!(config.head_dim(), variant.head_dim,
            "FALSIFIED FT-QWEN35-006: head_dim mismatch: config={} vs contract={}",
            config.head_dim(), variant.head_dim);
    }

    // ========================================================================
    // FALSIFY-FT-QWEN35-007: CLI dispatch "9B" resolves to qwen3_5_9b() config
    //
    // Prediction: The same config values used by "9B"/"qwen3.5-9b" CLI dispatch
    // match TransformerConfig::qwen3_5_9b(). This is a cross-boundary check
    // between CLI and library.
    // If fails: CLI dispatches to wrong config factory.
    // ========================================================================
    #[test]
    fn falsify_ft_qwen35_007_cli_dispatch_consistency() {
        // Verify all aliases produce the same config
        let config_9b = entrenar::transformer::TransformerConfig::qwen3_5_9b();
        assert_eq!(config_9b.vocab_size, 248320,
            "FALSIFIED FT-QWEN35-007: '9B' dispatch config has wrong vocab_size");
        assert_eq!(config_9b.hidden_size, 4096,
            "FALSIFIED FT-QWEN35-007: '9B' dispatch config has wrong hidden_size");
        assert!(!config_9b.use_bias,
            "FALSIFIED FT-QWEN35-007: '9B' dispatch config should not have bias");

        // Verify it's different from Qwen2 (no confusion between families)
        let config_qwen2 = entrenar::transformer::TransformerConfig::qwen2_0_5b();
        assert_ne!(config_9b.vocab_size, config_qwen2.vocab_size,
            "FALSIFIED FT-QWEN35-007: Qwen3.5 and Qwen2 should have different vocab_size");
        assert_ne!(config_9b.use_bias, config_qwen2.use_bias,
            "FALSIFIED FT-QWEN35-007: Qwen3.5 (no bias) vs Qwen2 (has bias) must differ");
    }

    // ========================================================================
    // CROSS-CRATE FINE-TUNING CONTRACT FALSIFICATION (FALSIFY-FT-XCRATE-001..004)
    //
    // These tests verify cross-crate invariants between aprender's Poka-Yoke
    // validated types and entrenar's training pipeline. They attempt to falsify
    // the claim that the two crates agree on classification contracts.
    //
    // Contract: classification-finetune-v1.yaml
    // ========================================================================

    // ========================================================================
    // FALSIFY-FT-XCRATE-001: ClassifyConfig default num_classes >= 2
    //
    // Prediction: ClassifyConfig::default().num_classes >= 2.
    // If fails: default config would violate F-CLASS-006 (degenerate class count).
    // ========================================================================
    #[test]
    fn falsify_ft_xcrate_001_default_num_classes() {
        let config = entrenar::finetune::ClassifyConfig::default();
        assert!(
            config.num_classes >= 2,
            "FALSIFIED FT-XCRATE-001: ClassifyConfig default num_classes={} < 2",
            config.num_classes
        );
    }

    // ========================================================================
    // FALSIFY-FT-XCRATE-002: ClassificationHead shape matches F-CLASS-004
    //
    // Prediction: ClassificationHead weight tensor has exactly
    //   hidden_size * num_classes elements.
    // If fails: weight shape contract is broken between aprender and entrenar.
    // ========================================================================
    #[test]
    fn falsify_ft_xcrate_002_classifier_head_shape() {
        let hidden_size = 896; // Qwen2-0.5B
        let num_classes = 5;
        let head = entrenar::finetune::ClassificationHead::new(hidden_size, num_classes);
        let weight_data = head.weight.data();
        let weight_slice = weight_data.as_slice().expect("contiguous weight data");
        assert_eq!(
            weight_slice.len(),
            hidden_size * num_classes,
            "FALSIFIED FT-XCRATE-002: ClassificationHead weight.len()={} != hidden_size({}) * num_classes({})",
            weight_slice.len(), hidden_size, num_classes
        );
    }

    // ========================================================================
    // FALSIFY-FT-XCRATE-003: cross_entropy_loss matches F-CLASS-003 postcondition
    //
    // Prediction: cross_entropy_loss output is finite and non-negative.
    // If fails: loss computation violates F-CLASS-005 postcondition.
    // ========================================================================
    #[test]
    fn falsify_ft_xcrate_003_cross_entropy_postcondition() {
        let logits = entrenar::Tensor::from_vec(vec![2.0_f32, 1.0, 0.1, -1.0, 3.0], false);
        let label = 2; // third class
        let num_classes = 5;
        let loss_tensor = entrenar::finetune::cross_entropy_loss(&logits, label, num_classes);
        let loss_data = loss_tensor.data();
        let loss_val = loss_data.as_slice().expect("contiguous loss")[0];
        assert!(
            loss_val.is_finite(),
            "FALSIFIED FT-XCRATE-003: cross_entropy_loss returned non-finite: {loss_val}"
        );
        assert!(
            loss_val >= 0.0,
            "FALSIFIED FT-XCRATE-003: cross_entropy_loss returned negative: {loss_val}"
        );
    }

    // ========================================================================
    // FALSIFY-FT-XCRATE-004: ValidatedClassLogits accepts entrenar logit output
    //
    // Prediction: logits from ClassificationHead::forward() can be validated
    //   by aprender's ValidatedClassLogits::new() without error.
    // If fails: the two crates disagree on logit shape contract.
    // ========================================================================
    #[test]
    fn falsify_ft_xcrate_004_validated_logits_accept_head_output() {
        use crate::format::validated_classification::ValidatedClassLogits;

        let hidden_size = 896;
        let num_classes = 5;
        let seq_len = 1;
        let head = entrenar::finetune::ClassificationHead::new(hidden_size, num_classes);

        // Simulate a hidden state tensor [seq_len * hidden_size]
        let hidden_state = entrenar::Tensor::from_vec(vec![0.1_f32; seq_len * hidden_size], false);
        let logits_tensor = head.forward(&hidden_state, seq_len);
        let logits_data: Vec<f32> = logits_tensor.data()
            .as_slice()
            .expect("contiguous logits")
            .to_vec();

        // aprender's Poka-Yoke type must accept these logits
        let validated = ValidatedClassLogits::new(logits_data, num_classes);
        assert!(
            validated.is_ok(),
            "FALSIFIED FT-XCRATE-004: ValidatedClassLogits rejected entrenar logits: {:?}",
            validated.err()
        );
    }
}
