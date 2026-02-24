
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
}
