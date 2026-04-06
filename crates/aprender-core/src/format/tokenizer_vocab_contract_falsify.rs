// ============================================================================
// Tokenizer-Vocabulary Contract Falsification (FALSIFY-TV-001..006)
//
// Cross-checks three contracts for consistency:
//   1. contracts/tokenizer-vocab-v1.yaml (this contract)
//   2. contracts/special-tokens-registry-v1.yaml (token IDs)
//   3. contracts/model-families/*.yaml (vocab per size variant)
//
// Contract: contracts/tokenizer-vocab-v1.yaml
// ============================================================================

#[cfg(test)]
mod tokenizer_vocab_contract {
    use std::collections::HashMap;
    use std::path::Path;

    use crate::format::model_family::ModelFamilyConfig;
    use crate::format::model_family_loader::load_family_yaml;

    // ========================================================================
    // Parsed types
    // ========================================================================

    #[derive(Debug)]
    struct TvFamily {
        tokenizer_type: String,
        vocab_size: u64,
        bos_id: u32,
        eos_id: u32,
        pad_id: u32,
        im_start_id: u32,
        im_end_id: u32,
    }

    #[derive(Debug)]
    struct StFamily {
        vocab_size: u64,
        bos_id: u32,
        eos_id: u32,
        pad_id: u32,
        im_start_id: u32,
        im_end_id: u32,
    }

    /// Accumulator for token fields during parsing.
    #[derive(Default)]
    struct TokenAcc {
        vocab_size: u64,
        bos_id: u32,
        eos_id: u32,
        pad_id: u32,
        im_start_id: u32,
        im_end_id: u32,
    }

    impl TokenAcc {
        fn reset(&mut self) {
            *self = Self::default();
        }

        /// Try to parse a token field from a trimmed line. Returns true if matched.
        fn try_parse_field(&mut self, trimmed: &str) -> bool {
            let pairs: &[(&str, fn(&mut Self, u32))] = &[
                ("vocab_size:", |s, v| s.vocab_size = u64::from(v)),
                ("bos_id:", |s, v| s.bos_id = v),
                ("eos_id:", |s, v| s.eos_id = v),
                ("pad_id:", |s, v| s.pad_id = v),
                ("im_start_id:", |s, v| s.im_start_id = v),
                ("im_end_id:", |s, v| s.im_end_id = v),
            ];

            for &(prefix, setter) in pairs {
                if let Some(rest) = trimmed.strip_prefix(prefix) {
                    let val: u32 = field_val(rest).parse().unwrap_or(0);
                    setter(self, val);
                    return true;
                }
            }
            false
        }

        /// Try to parse vocab_size as u64 (for large vocabs > u32).
        fn try_parse_vocab(&mut self, trimmed: &str) -> bool {
            if let Some(rest) = trimmed.strip_prefix("vocab_size:") {
                self.vocab_size = field_val(rest).parse().unwrap_or(0);
                return true;
            }
            false
        }
    }

    // ========================================================================
    // YAML parsers
    // ========================================================================

    fn read_file(name: &str) -> String {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(name);
        assert!(path.exists(), "{name} must exist");
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("Failed to read {name}: {e}"))
    }

    /// Parse field value from a "key: value # comment" string.
    fn field_val(line: &str) -> &str {
        line.split('#').next().unwrap_or("").trim()
    }

    /// Check if a line is a section-ending line (non-empty, non-indented, non-comment).
    fn is_section_end(line: &str) -> bool {
        let trimmed = line.trim();
        !trimmed.is_empty()
            && !line.starts_with(' ')
            && !line.starts_with('\t')
            && !trimmed.starts_with('#')
    }

    /// Check if trimmed line is a family-name key (2-space indent, ends with colon, no spaces).
    fn is_family_name(line: &str, trimmed: &str) -> bool {
        let indent = line.len() - line.trim_start().len();
        indent == 2 && trimmed.ends_with(':') && !trimmed.contains(' ')
    }

    /// Parse the tokenizer_types section — returns set of known type names.
    fn parse_tokenizer_types(content: &str) -> Vec<String> {
        let mut types = Vec::new();
        let mut in_section = false;

        for line in content.lines() {
            let trimmed = line.trim();
            if trimmed == "tokenizer_types:" {
                in_section = true;
                continue;
            }
            if in_section && is_section_end(line) {
                break;
            }
            if !in_section {
                continue;
            }
            if is_family_name(line, trimmed) {
                types.push(trimmed.trim_end_matches(':').to_string());
            }
        }
        types
    }

    /// Parse the families section from tokenizer-vocab-v1.yaml.
    fn parse_tv_families(content: &str) -> HashMap<String, TvFamily> {
        let mut families = HashMap::new();
        let mut in_section = false;
        let mut in_special_tokens = false;
        let mut current_name = String::new();
        let mut current_type = String::new();
        let mut acc = TokenAcc::default();

        for line in content.lines() {
            let trimmed = line.trim();

            if trimmed == "families:" {
                in_section = true;
                continue;
            }
            if in_section && is_section_end(line) {
                break;
            }
            if !in_section {
                continue;
            }

            let indent = line.len() - line.trim_start().len();

            if is_family_name(line, trimmed) {
                save_tv_family(&mut families, &current_name, &current_type, &acc);
                current_name = trimmed.trim_end_matches(':').to_string();
                current_type.clear();
                acc.reset();
                in_special_tokens = false;
                continue;
            }

            if indent == 4 && trimmed == "special_tokens:" {
                in_special_tokens = true;
                continue;
            }

            if indent == 4 && !trimmed.starts_with('#') {
                in_special_tokens = parse_tv_line(trimmed, &mut current_type, &mut acc);
            }

            if indent == 6 && in_special_tokens {
                acc.try_parse_field(trimmed);
            }
        }

        save_tv_family(&mut families, &current_name, &current_type, &acc);
        families
    }

    /// Parse a 4-indent tokenizer-vocab line. Returns true if entering special_tokens.
    fn parse_tv_line(trimmed: &str, current_type: &mut String, acc: &mut TokenAcc) -> bool {
        if let Some(rest) = trimmed.strip_prefix("tokenizer_type:") {
            *current_type = field_val(rest).to_string();
            return false;
        }
        if acc.try_parse_vocab(trimmed) {
            return false;
        }
        false
    }

    fn save_tv_family(
        families: &mut HashMap<String, TvFamily>,
        name: &str,
        tok_type: &str,
        acc: &TokenAcc,
    ) {
        if !name.is_empty() && acc.vocab_size > 0 {
            families.insert(
                name.to_string(),
                TvFamily {
                    tokenizer_type: tok_type.to_string(),
                    vocab_size: acc.vocab_size,
                    bos_id: acc.bos_id,
                    eos_id: acc.eos_id,
                    pad_id: acc.pad_id,
                    im_start_id: acc.im_start_id,
                    im_end_id: acc.im_end_id,
                },
            );
        }
    }

    /// Parse the families section from special-tokens-registry-v1.yaml.
    fn parse_st_families(content: &str) -> HashMap<String, StFamily> {
        let mut families = HashMap::new();
        let mut in_section = false;
        let mut current_name = String::new();
        let mut acc = TokenAcc::default();

        for line in content.lines() {
            let trimmed = line.trim();

            if trimmed == "families:" {
                in_section = true;
                continue;
            }
            if in_section && is_section_end(line) {
                break;
            }
            if !in_section {
                continue;
            }

            let indent = line.len() - line.trim_start().len();

            if is_family_name(line, trimmed) {
                save_st_family(&mut families, &current_name, &acc);
                current_name = trimmed.trim_end_matches(':').to_string();
                acc.reset();
                continue;
            }

            if indent >= 4 {
                acc.try_parse_field(trimmed);
            }
        }

        save_st_family(&mut families, &current_name, &acc);
        families
    }

    fn save_st_family(families: &mut HashMap<String, StFamily>, name: &str, acc: &TokenAcc) {
        if !name.is_empty() && acc.vocab_size > 0 {
            families.insert(
                name.to_string(),
                StFamily {
                    vocab_size: acc.vocab_size,
                    bos_id: acc.bos_id,
                    eos_id: acc.eos_id,
                    pad_id: acc.pad_id,
                    im_start_id: acc.im_start_id,
                    im_end_id: acc.im_end_id,
                },
            );
        }
    }

    /// Load all model-family YAML configs.
    fn load_model_family_configs() -> Vec<(String, ModelFamilyConfig)> {
        let families_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("contracts/model-families");
        let mut families = Vec::new();
        let entries = std::fs::read_dir(&families_dir).expect("read model-families dir");

        for entry in entries {
            let entry = entry.expect("dir entry");
            let path = entry.path();
            let file_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            let ext_ok = path
                .extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("yaml"));
            if !ext_ok || file_name.starts_with('_') {
                continue;
            }
            if let Ok(config) = load_family_yaml(&path) {
                let name = file_name
                    .trim_end_matches(".yaml")
                    .trim_end_matches(".yml")
                    .to_string();
                families.push((name, config));
            }
        }
        families
    }

    // ========================================================================
    // FALSIFY-TV-001: Special tokens match special-tokens-registry
    // ========================================================================
    #[test]
    fn falsify_tv_001_tokens_match_registry() {
        let tv_content = read_file("contracts/tokenizer-vocab-v1.yaml");
        let st_content = read_file("contracts/special-tokens-registry-v1.yaml");

        let tv_families = parse_tv_families(&tv_content);
        let st_families = parse_st_families(&st_content);

        let mut violations = Vec::new();

        for (name, tv) in &tv_families {
            let Some(st) = st_families.get(name) else {
                violations.push(format!(
                    "{name}: in tokenizer-vocab but not in special-tokens-registry"
                ));
                continue;
            };

            let checks: &[(&str, u32, u32)] = &[
                ("bos_id", tv.bos_id, st.bos_id),
                ("eos_id", tv.eos_id, st.eos_id),
                ("pad_id", tv.pad_id, st.pad_id),
                ("im_start_id", tv.im_start_id, st.im_start_id),
                ("im_end_id", tv.im_end_id, st.im_end_id),
            ];

            for &(field, tv_val, st_val) in checks {
                if tv_val != st_val {
                    violations.push(format!(
                        "{name}.{field}: tokenizer-vocab={tv_val}, special-tokens-registry={st_val}"
                    ));
                }
            }
        }

        assert!(
            violations.is_empty(),
            "FALSIFY-TV-001: Token ID divergence between contracts:\n{}",
            violations.join("\n")
        );
    }

    // ========================================================================
    // FALSIFY-TV-002: Vocab size matches special-tokens-registry
    // ========================================================================
    #[test]
    fn falsify_tv_002_vocab_size_matches_registry() {
        let tv_content = read_file("contracts/tokenizer-vocab-v1.yaml");
        let st_content = read_file("contracts/special-tokens-registry-v1.yaml");

        let tv_families = parse_tv_families(&tv_content);
        let st_families = parse_st_families(&st_content);

        let mut violations = Vec::new();

        for (name, tv) in &tv_families {
            if let Some(st) = st_families.get(name) {
                if tv.vocab_size != st.vocab_size {
                    violations.push(format!(
                        "{name}: tokenizer-vocab vocab_size={}, special-tokens-registry vocab_size={}",
                        tv.vocab_size, st.vocab_size
                    ));
                }
            }
        }

        assert!(
            violations.is_empty(),
            "FALSIFY-TV-002: Vocab size divergence between contracts:\n{}",
            violations.join("\n")
        );
    }

    // ========================================================================
    // FALSIFY-TV-003: Vocab size matches model-family size variants
    // ========================================================================
    #[test]
    fn falsify_tv_003_vocab_matches_model_families() {
        let tv_content = read_file("contracts/tokenizer-vocab-v1.yaml");
        let tv_families = parse_tv_families(&tv_content);
        let mf_configs = load_model_family_configs();

        let mut violations = Vec::new();

        for (tv_name, tv) in &tv_families {
            let matching_mf = mf_configs.iter().find(|(name, _)| name == tv_name);

            if let Some((_, mf_config)) = matching_mf {
                let has_match = mf_config
                    .size_variants
                    .values()
                    .any(|size| size.vocab_size as u64 == tv.vocab_size);

                if !has_match && !mf_config.size_variants.is_empty() {
                    let observed: Vec<String> = mf_config
                        .size_variants
                        .values()
                        .map(|s| format!("{}", s.vocab_size))
                        .collect();
                    violations.push(format!(
                        "{tv_name}: tokenizer-vocab vocab_size={}, model-family has [{}]",
                        tv.vocab_size,
                        observed.join(", ")
                    ));
                }
            }
        }

        assert!(
            violations.is_empty(),
            "FALSIFY-TV-003: Vocab size mismatch with model families:\n{}",
            violations.join("\n")
        );
    }

    // ========================================================================
    // FALSIFY-TV-004: All tokenizer_type references are valid
    // ========================================================================
    #[test]
    fn falsify_tv_004_tokenizer_types_valid() {
        let tv_content = read_file("contracts/tokenizer-vocab-v1.yaml");
        let known_types = parse_tokenizer_types(&tv_content);
        let tv_families = parse_tv_families(&tv_content);

        let mut violations = Vec::new();

        for (name, tv) in &tv_families {
            if !known_types.contains(&tv.tokenizer_type) {
                violations.push(format!(
                    "{name}: tokenizer_type '{}' not in tokenizer_types section (known: {:?})",
                    tv.tokenizer_type, known_types
                ));
            }
        }

        assert!(
            violations.is_empty(),
            "FALSIFY-TV-004: Invalid tokenizer type references:\n{}",
            violations.join("\n")
        );
    }

    // ========================================================================
    // FALSIFY-TV-005: Token IDs within vocab bounds
    // ========================================================================
    #[test]
    fn falsify_tv_005_token_ids_within_bounds() {
        let tv_content = read_file("contracts/tokenizer-vocab-v1.yaml");
        let tv_families = parse_tv_families(&tv_content);

        let mut violations = Vec::new();

        for (name, tv) in &tv_families {
            let checks: &[(&str, u32)] = &[
                ("bos_id", tv.bos_id),
                ("eos_id", tv.eos_id),
                ("pad_id", tv.pad_id),
                ("im_start_id", tv.im_start_id),
                ("im_end_id", tv.im_end_id),
            ];

            for &(field, id) in checks {
                if id > 0 && u64::from(id) >= tv.vocab_size {
                    violations.push(format!(
                        "{name}.{field}: {id} >= vocab_size {}",
                        tv.vocab_size
                    ));
                }
            }
        }

        assert!(
            violations.is_empty(),
            "FALSIFY-TV-005: Token IDs out of vocab bounds:\n{}",
            violations.join("\n")
        );
    }

    // ========================================================================
    // FALSIFY-TV-006: Family count parity with special-tokens-registry
    // ========================================================================
    #[test]
    fn falsify_tv_006_family_count_parity() {
        let tv_content = read_file("contracts/tokenizer-vocab-v1.yaml");
        let st_content = read_file("contracts/special-tokens-registry-v1.yaml");

        let tv_families = parse_tv_families(&tv_content);
        let st_families = parse_st_families(&st_content);

        assert_eq!(
            tv_families.len(),
            st_families.len(),
            "FALSIFY-TV-006: tokenizer-vocab has {} families, special-tokens-registry has {}. \
             Sync contracts/tokenizer-vocab-v1.yaml with contracts/special-tokens-registry-v1.yaml",
            tv_families.len(),
            st_families.len()
        );
    }
}
