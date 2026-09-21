
// ============================================================================
// Model Family Contract Falsification Tests (FALSIFY-MF-001..008)
//
// Popperian falsification: each test attempts to BREAK a mathematical invariant
// claimed by the model-family YAML contracts. If a test fails, the contract
// contains an error (wrong dimension, missing field, etc.).
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
        let contracts_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../contracts");
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

            // Skip ModelFamilyVariant contracts (start with `contract_id:`) that
            // co-locate under model-families/ for documentation purposes.
            let head = std::fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("Failed to read {file_name}: {e}"));
            if head.lines().any(|l| l.starts_with("contract_id:")) {
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
            // Qwen3.5 27B: head_dim=256 fixed, 24*256=6144 != 5120
            ("qwen3_5", "27b"),
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
    // FALSIFY-MF-QWEN35-010: the Gated DeltaNet shape keys survive the loader
    //
    // Prediction: loading contracts/model-families/qwen3_5.yaml yields
    //             constraints.deltanet = Some(inner 2048, state 128, conv 4,
    //             group 8, interval 4) — the values the descriptor declares —
    //             and EVERY other family yields None.
    // If fails: the keys are declared in YAML and dropped on the way into
    //           ModelConstraints, which is the #3346 defect. That drop made the
    //           arithmetic under-count a real Qwen3.5 file by 14.4%, because a
    //           DeltaNet layer's parameters live entirely in these dimensions.
    // ========================================================================
    #[test]
    fn falsify_mf_qwen35_010_deltanet_shape_reaches_constraints() {
        let families = load_all_families();
        let qwen35 = families
            .iter()
            .find(|(name, _)| name == "qwen3_5")
            .expect("FALSIFIED: qwen3_5 family not found");

        let shape = qwen35
            .1
            .constraints
            .deltanet
            .expect("FALSIFIED QWEN35-010: qwen3_5 declares inner_size/state_size, got None");

        assert_eq!(
            shape,
            DeltaNetShape {
                inner_size: 2048,
                state_size: 128,
                conv_kernel: 4,
                group_count: 8,
                full_attention_interval: 4,
            },
            "FALSIFIED QWEN35-010: loader did not reproduce the declared shape"
        );

        // No other family declares a DeltaNet mixer, so none may acquire one:
        // a false Some() here would change that family's parameter accounting.
        for (name, config) in &families {
            if name != "qwen3_5" {
                assert!(
                    config.constraints.deltanet.is_none(),
                    "FALSIFIED QWEN35-010: {name} acquired a DeltaNet shape it never declared"
                );
            }
        }
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
    // FALSIFY-MF-QWEN35-008b: Qwen3.5 27B exact dimensions
    //
    // Prediction: The qwen3_5/27b size variant has the documented dimensions
    // from the Qwen3.5-27B config.json. If fails: contract YAML is stale.
    // ========================================================================
    #[test]
    fn falsify_mf_qwen35_008b_27b_dimensions() {
        let families = load_all_families();
        let qwen35 = families
            .iter()
            .find(|(name, _)| name == "qwen3_5")
            .expect("FALSIFIED: qwen3_5 family not found");

        let variant = qwen35
            .1
            .size_variants
            .get("27b")
            .expect("FALSIFIED: qwen3_5 missing '27b' size variant");

        // hidden_dim = num_heads * head_dim = 24 * 256 = 6144
        assert_eq!(variant.hidden_dim, 6144,
            "FALSIFIED QWEN35-008b: hidden_dim={}, expected 6144 (24*256)", variant.hidden_dim);
        assert_eq!(variant.num_layers, 64,
            "FALSIFIED QWEN35-008b: num_layers={}, expected 64", variant.num_layers);
        assert_eq!(variant.num_heads, 24,
            "FALSIFIED QWEN35-008b: num_heads={}, expected 24", variant.num_heads);
        assert_eq!(variant.num_kv_heads, 4,
            "FALSIFIED QWEN35-008b: num_kv_heads={}, expected 4", variant.num_kv_heads);
        assert_eq!(variant.intermediate_dim, 17408,
            "FALSIFIED QWEN35-008b: intermediate_dim={}, expected 17408", variant.intermediate_dim);
        assert_eq!(variant.vocab_size, 248320,
            "FALSIFIED QWEN35-008b: vocab_size={}, expected 248320", variant.vocab_size);
        assert_eq!(variant.head_dim, 256,
            "FALSIFIED QWEN35-008b: head_dim={}, expected 256", variant.head_dim);
    }

    // ========================================================================
    // FALSIFY-MF-QWEN35-009: completeness_key returns "qwen3_5" (not "qwen3")
    //
    // Prediction: Architecture::Qwen3_5.completeness_key() == Some("qwen3_5").
    // Qwen3.5 has NO QK norm tensors (unlike Qwen3 which has q_norm/k_norm).
    // If mapped to "qwen3", import would require nonexistent q_norm/k_norm
    // tensors and fail validation.
    // ========================================================================
    #[test]
    fn falsify_mf_qwen35_009_completeness_key() {
        use crate::format::converter_types::Architecture;

        let key = Architecture::Qwen3_5.completeness_key();
        assert_eq!(
            key,
            Some("qwen3_5"),
            "FALSIFIED QWEN35-009: completeness_key()={:?}, expected Some(\"qwen3_5\"). \
             Using \"qwen3\" would require nonexistent q_norm/k_norm tensors.",
            key
        );

        // Also verify Qwen3 still maps to "qwen3" (regression guard)
        let qwen3_key = Architecture::Qwen3.completeness_key();
        assert_eq!(
            qwen3_key,
            Some("qwen3"),
            "FALSIFIED QWEN35-009: Qwen3 completeness_key()={:?}, expected Some(\"qwen3\")",
            qwen3_key
        );
    }

    // FALSIFY-FT-QWEN35-001..007 and FALSIFY-FT-XCRATE-001..004 moved to
    // tests/contracts/classification_finetune_xcrate_contract.rs (PMAT-1098).
    // They name `entrenar`, a PATH-ONLY dev-dep (PMAT-955: a version there is a
    // publish cycle) that `cargo publish` OMITS while publishing this
    // `#[cfg(test)]` code anyway -- so in published form the lib tests could not
    // compile (clean-room GATE B2: 19 x error[E0433]). Same mechanism as #3307.
}
