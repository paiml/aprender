
// ============================================================================
// Special Tokens Registry Contract Falsification (FALSIFY-ST-001..006)
//
// Verifies that the Rust SpecialTokens constructors in src/demo/mod.rs
// match the YAML source of truth in contracts/special-tokens-registry-v1.yaml.
//
// Contract: contracts/special-tokens-registry-v1.yaml
// ============================================================================

#[cfg(test)]
mod special_tokens_contract {
    use std::collections::HashMap;
    use std::path::Path;

    use crate::demo::SpecialTokens;

    /// Parsed special tokens entry from YAML.
    #[derive(Debug, Default)]
    struct YamlTokenEntry {
        vocab_size: u32,
        bos_id: u32,
        eos_id: u32,
        pad_id: u32,
        im_start_id: u32,
        im_end_id: u32,
    }

    /// Read the registry YAML file content.
    fn read_registry_content() -> String {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("contracts/special-tokens-registry-v1.yaml");
        assert!(
            path.exists(),
            "contracts/special-tokens-registry-v1.yaml must exist (P0 gap)"
        );
        std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("Failed to read special-tokens-registry: {e}"))
    }

    /// Parse a token ID field from a "key: value # comment" line.
    fn parse_token_field(entry: &mut YamlTokenEntry, key: &str, val: &str) {
        let val = val.split('#').next().unwrap_or("").trim();
        let parsed = val.parse().unwrap_or(0);
        match key {
            "vocab_size" => entry.vocab_size = parsed,
            "bos_id" => entry.bos_id = parsed,
            "eos_id" => entry.eos_id = parsed,
            "pad_id" => entry.pad_id = parsed,
            "im_start_id" => entry.im_start_id = parsed,
            "im_end_id" => entry.im_end_id = parsed,
            _ => {}
        }
    }

    /// Parse the families section from YAML lines.
    fn parse_families(lines: &[&str]) -> HashMap<String, YamlTokenEntry> {
        let mut families = HashMap::new();
        let mut current_family = String::new();
        let mut current_entry = YamlTokenEntry::default();

        for line in lines {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }
            let indent = line.len() - line.trim_start().len();

            // Family name at 2-space indent
            if indent == 2 && trimmed.ends_with(':') && !trimmed.contains(' ') {
                if !current_family.is_empty() && current_entry.vocab_size > 0 {
                    families.insert(current_family, current_entry);
                    current_entry = YamlTokenEntry::default();
                }
                current_family = trimmed.trim_end_matches(':').to_string();
                continue;
            }

            // Field at 4+ indent
            if indent >= 4 {
                if let Some((key, val)) = trimmed.split_once(':') {
                    parse_token_field(&mut current_entry, key.trim(), val.trim());
                }
            }
        }

        // Save last family
        if !current_family.is_empty() && current_entry.vocab_size > 0 {
            families.insert(current_family, current_entry);
        }

        families
    }

    /// Parse the architecture_mapping section from YAML lines.
    fn parse_arch_mapping(lines: &[&str]) -> HashMap<String, String> {
        let mut mapping = HashMap::new();

        for line in lines {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') || trimmed.starts_with('-') {
                continue;
            }
            let indent = line.len() - line.trim_start().len();
            if indent == 2 {
                if let Some((key, val)) = trimmed.split_once(':') {
                    let key = key.trim().to_string();
                    let val = val.split('#').next().unwrap_or("").trim().to_string();
                    if !key.is_empty() && !val.is_empty() {
                        mapping.insert(key, val);
                    }
                }
            }
        }

        mapping
    }

    /// Parse the special-tokens-registry-v1.yaml.
    /// Returns (families map, architecture_mapping).
    fn load_registry() -> (HashMap<String, YamlTokenEntry>, HashMap<String, String>) {
        let content = read_registry_content();
        let all_lines: Vec<&str> = content.lines().collect();

        // Find section boundaries
        let families_start = all_lines
            .iter()
            .position(|l| l.trim() == "families:")
            .map(|i| i + 1)
            .unwrap_or(0);
        let arch_start = all_lines
            .iter()
            .position(|l| l.trim() == "architecture_mapping:")
            .map(|i| i + 1)
            .unwrap_or(all_lines.len());

        // Find the end of families section (next unindented section)
        let families_end = all_lines[families_start..]
            .iter()
            .position(|l| {
                !l.is_empty()
                    && !l.starts_with(' ')
                    && !l.starts_with('\t')
                    && !l.trim().starts_with('#')
            })
            .map(|i| families_start + i)
            .unwrap_or(all_lines.len());

        // Find the end of arch mapping section
        let arch_end = if arch_start < all_lines.len() {
            all_lines[arch_start..]
                .iter()
                .position(|l| {
                    !l.is_empty()
                        && !l.starts_with(' ')
                        && !l.starts_with('\t')
                        && !l.trim().starts_with('#')
                })
                .map(|i| arch_start + i)
                .unwrap_or(all_lines.len())
        } else {
            all_lines.len()
        };

        let families = parse_families(&all_lines[families_start..families_end]);
        let arch_mapping = parse_arch_mapping(&all_lines[arch_start..arch_end]);

        (families, arch_mapping)
    }

    // ========================================================================
    // FALSIFY-ST-001: Rust constructors match YAML registry
    // ========================================================================
    #[test]
    fn falsify_st_001_rust_constructors_match_yaml() {
        let (families, _) = load_registry();

        let constructors: &[(&str, SpecialTokens)] = &[
            ("qwen2", SpecialTokens::qwen2()),
            ("qwen3_5", SpecialTokens::qwen3_5()),
            ("llama", SpecialTokens::llama()),
            ("mistral", SpecialTokens::mistral()),
            ("gemma", SpecialTokens::gemma()),
            ("deepseek", SpecialTokens::deepseek()),
            ("phi3", SpecialTokens::phi3()),
            ("phi2", SpecialTokens::phi2()),
            ("gpt2", SpecialTokens::gpt2()),
        ];

        let mut violations = Vec::new();

        for (family_name, rust_tokens) in constructors {
            let Some(yaml_entry) = families.get(*family_name) else {
                violations.push(format!(
                    "{family_name}: Rust constructor exists but no YAML entry"
                ));
                continue;
            };

            let checks: &[(&str, u32, u32)] = &[
                ("bos_id", rust_tokens.bos_id, yaml_entry.bos_id),
                ("eos_id", rust_tokens.eos_id, yaml_entry.eos_id),
                ("pad_id", rust_tokens.pad_id, yaml_entry.pad_id),
                ("im_start_id", rust_tokens.im_start_id, yaml_entry.im_start_id),
                ("im_end_id", rust_tokens.im_end_id, yaml_entry.im_end_id),
            ];

            for &(field, rust_val, yaml_val) in checks {
                if rust_val != yaml_val {
                    violations.push(format!(
                        "{family_name}.{field}: Rust={rust_val}, YAML={yaml_val}"
                    ));
                }
            }
        }

        assert!(
            violations.is_empty(),
            "FALSIFY-ST-001: Rust ↔ YAML divergence:\n{}",
            violations.join("\n")
        );
    }

    // ========================================================================
    // FALSIFY-ST-002: YAML self-consistency — all IDs within vocab bounds
    // ========================================================================
    #[test]
    fn falsify_st_002_yaml_ids_within_vocab_bounds() {
        let (families, _) = load_registry();
        let mut violations = Vec::new();

        for (family_name, entry) in &families {
            if entry.vocab_size == 0 {
                violations.push(format!("{family_name}: vocab_size is 0"));
                continue;
            }

            let checks: &[(&str, u32)] = &[
                ("bos_id", entry.bos_id),
                ("eos_id", entry.eos_id),
                ("pad_id", entry.pad_id),
                ("im_start_id", entry.im_start_id),
                ("im_end_id", entry.im_end_id),
            ];

            for &(field, id) in checks {
                if id > 0 && id >= entry.vocab_size {
                    violations.push(format!(
                        "{family_name}.{field}: {id} >= vocab_size {}",
                        entry.vocab_size
                    ));
                }
            }
        }

        assert!(
            violations.is_empty(),
            "FALSIFY-ST-002: Token IDs out of vocab bounds:\n{}",
            violations.join("\n")
        );
    }

    // ========================================================================
    // FALSIFY-ST-003: Architecture mapping references valid families
    // ========================================================================
    #[test]
    fn falsify_st_003_architecture_mapping_valid() {
        let (families, arch_mapping) = load_registry();
        let mut violations = Vec::new();

        for (arch, family) in &arch_mapping {
            if !families.contains_key(family) {
                violations.push(format!(
                    "architecture '{arch}' → '{family}' which doesn't exist in families"
                ));
            }
        }

        assert!(
            violations.is_empty(),
            "FALSIFY-ST-003: Invalid architecture mapping:\n{}",
            violations.join("\n")
        );
    }

    // ========================================================================
    // FALSIFY-ST-004: from_architecture() covers all YAML architecture mappings
    // ========================================================================
    #[test]
    fn falsify_st_004_from_architecture_covers_yaml() {
        let (_, arch_mapping) = load_registry();
        let mut violations = Vec::new();

        for (arch, _) in &arch_mapping {
            if SpecialTokens::from_architecture(arch).is_none() {
                violations.push(format!(
                    "SpecialTokens::from_architecture(\"{arch}\") returns None"
                ));
            }
        }

        assert!(
            violations.is_empty(),
            "FALSIFY-ST-004: from_architecture() missing mappings:\n{}",
            violations.join("\n")
        );
    }

    // ========================================================================
    // FALSIFY-ST-005: Every EOS is non-zero
    // ========================================================================
    #[test]
    fn falsify_st_005_eos_always_nonzero() {
        let (families, _) = load_registry();
        let mut violations = Vec::new();

        for (family_name, entry) in &families {
            if entry.eos_id == 0 {
                violations.push(format!(
                    "{family_name}: eos_id is 0 (model will never stop generating)"
                ));
            }
        }

        assert!(
            violations.is_empty(),
            "FALSIFY-ST-005: Families with zero EOS:\n{}",
            violations.join("\n")
        );
    }

    // ========================================================================
    // FALSIFY-ST-006: YAML family count matches Rust constructor count
    // ========================================================================
    #[test]
    fn falsify_st_006_family_count_parity() {
        let (families, _) = load_registry();

        // Count of Rust SpecialTokens constructors
        let rust_count = 9; // qwen2, qwen3_5, llama, mistral, gemma, deepseek, phi3, phi2, gpt2

        assert_eq!(
            families.len(),
            rust_count,
            "FALSIFY-ST-006: YAML has {} families but Rust has {rust_count} constructors. \
             Sync contracts/special-tokens-registry-v1.yaml with src/demo/mod.rs",
            families.len()
        );
    }
}
