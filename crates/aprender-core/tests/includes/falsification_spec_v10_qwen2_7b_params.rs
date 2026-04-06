// =============================================================================
// Supplementary: Qwen2 7B Parameter Verification
// =============================================================================

#[test]
fn f_model_qwen2_7b_hidden_dim_3584() {
    let registry = build_default_registry();
    let config = registry
        .get("qwen2")
        .expect("qwen2")
        .size_config("7b")
        .expect("7b");
    assert_eq!(config.hidden_dim, 3584);
}

#[test]
fn f_model_qwen2_7b_num_layers_28() {
    let registry = build_default_registry();
    let config = registry
        .get("qwen2")
        .expect("qwen2")
        .size_config("7b")
        .expect("7b");
    assert_eq!(config.num_layers, 28);
}

#[test]
fn f_model_qwen2_7b_num_heads_28() {
    let registry = build_default_registry();
    let config = registry
        .get("qwen2")
        .expect("qwen2")
        .size_config("7b")
        .expect("7b");
    assert_eq!(config.num_heads, 28);
}

#[test]
fn f_model_qwen2_7b_num_kv_heads_4() {
    let registry = build_default_registry();
    let config = registry
        .get("qwen2")
        .expect("qwen2")
        .size_config("7b")
        .expect("7b");
    assert_eq!(config.num_kv_heads, 4);
}

#[test]
fn f_model_qwen2_7b_intermediate_dim_18944() {
    let registry = build_default_registry();
    let config = registry
        .get("qwen2")
        .expect("qwen2")
        .size_config("7b")
        .expect("7b");
    assert_eq!(config.intermediate_dim, 18944);
}

#[test]
fn f_model_qwen2_7b_vocab_152064() {
    let registry = build_default_registry();
    let config = registry
        .get("qwen2")
        .expect("qwen2")
        .size_config("7b")
        .expect("7b");
    assert_eq!(config.vocab_size, 152_064);
}

#[test]
fn f_model_qwen2_constraints_correct() {
    let registry = build_default_registry();
    let qwen2 = registry.get("qwen2").expect("qwen2 must exist");
    let constraints = qwen2.constraints();

    assert_eq!(constraints.attention_type, AttentionType::Gqa);
    assert_eq!(constraints.activation, Activation::Silu);
    assert_eq!(constraints.norm_type, NormType::RmsNorm);
    assert_eq!(constraints.positional_encoding, PositionalEncoding::Rope);
    assert_eq!(constraints.mlp_type, MlpType::SwiGlu);
    assert!(constraints.has_bias, "Qwen2 must have bias tensors");
    assert!(
        !constraints.tied_embeddings,
        "Qwen2 must NOT have tied embeddings"
    );
}

#[test]
fn f_model_all_8_families_in_registry() {
    let registry = build_default_registry();
    let expected = [
        "bert", "deepseek", "gemma", "llama", "mistral", "phi", "qwen2", "whisper",
    ];

    for family in &expected {
        assert!(
            registry.get(family).is_some(),
            "Registry must contain '{family}'"
        );
    }

    assert!(
        registry.len() >= 8,
        "Registry must have >= 8 families, got {}",
        registry.len()
    );
}

// =============================================================================
// Enum variant counting helpers
// =============================================================================

fn count_enum_variants(source: &str, enum_header: &str) -> usize {
    extract_enum_variant_names(source, enum_header).len()
}

fn count_braces(line: &str) -> i32 {
    line.chars()
        .map(|ch| match ch {
            '{' => 1,
            '}' => -1,
            _ => 0,
        })
        .sum()
}

fn extract_variant_name(trimmed: &str) -> Option<String> {
    if trimmed.starts_with('#') || trimmed.starts_with("//") || trimmed.is_empty() {
        return None;
    }
    let first_char = trimmed.chars().next().unwrap_or(' ');
    if !first_char.is_ascii_uppercase() {
        return None;
    }
    let name: String = trimmed
        .chars()
        .take_while(|c| c.is_alphanumeric() || *c == '_')
        .collect();
    if name.is_empty() {
        None
    } else {
        Some(name)
    }
}

fn extract_enum_variant_names(source: &str, enum_header: &str) -> Vec<String> {
    let mut in_enum = false;
    let mut brace_depth: i32 = 0;
    let mut names = Vec::new();

    for line in source.lines() {
        let trimmed = line.trim();

        if !in_enum {
            if trimmed.contains(enum_header) {
                in_enum = true;
                brace_depth += count_braces(trimmed);
            }
            continue;
        }

        let depth_before = brace_depth;
        brace_depth += count_braces(trimmed);
        if brace_depth == 0 {
            break;
        }
        if depth_before == 1 {
            if let Some(name) = extract_variant_name(trimmed) {
                names.push(name);
            }
        }
    }
    names
}

fn count_unique_gate_ids(content: &str, prefix: &str) -> usize {
    use std::collections::HashSet;
    let mut ids = HashSet::new();

    for line in content.lines() {
        if let Some(pos) = line.find(prefix) {
            // Extract the gate ID (e.g., "F-GT-001")
            let remaining = &line[pos..];
            let id: String = remaining
                .chars()
                .take_while(|c| c.is_alphanumeric() || *c == '-')
                .collect();
            if id.len() > prefix.len() {
                ids.insert(id);
            }
        }
    }
    ids.len()
}

// =============================================================================
// Supplementary tests (merged from earlier test files)
// =============================================================================

#[test]
fn f_fmt_from_extension_covers_all_formats() {
    let cases = [
        ("model.gguf", FormatType::Gguf),
        ("model.safetensors", FormatType::SafeTensors),
        ("model.apr", FormatType::Apr),
    ];

    for (filename, expected) in &cases {
        let path = Path::new(filename);
        let format = FormatType::from_extension(path);
        assert!(
            format.is_ok(),
            "from_extension should detect format from '{filename}', got error: {:?}",
            format.err()
        );
        assert_eq!(
            format.expect("format"),
            *expected,
            "from_extension('{filename}') should return {expected:?}"
        );
    }

    // Unknown extension should fail
    let unknown = FormatType::from_extension(Path::new("model.pytorch"));
    assert!(
        unknown.is_err(),
        "from_extension should reject unknown extension .pytorch"
    );
}

#[test]
fn f_fmt_format_type_display_is_meaningful() {
    let formats = [FormatType::Gguf, FormatType::SafeTensors, FormatType::Apr];

    for format in &formats {
        let display = format.to_string();
        assert!(
            !display.is_empty(),
            "FormatType::{format:?} Display must be non-empty"
        );
    }
}

#[test]
fn f_arch_format_type_has_exactly_three_variants() {
    // Exhaustive match -- if a variant is added, this becomes a compile error
    let variants = [FormatType::Gguf, FormatType::SafeTensors, FormatType::Apr];

    for variant in &variants {
        match variant {
            FormatType::Gguf => {}
            FormatType::SafeTensors => {}
            FormatType::Apr => {}
        }
    }

    assert_eq!(
        variants.len(),
        3,
        "FormatType must have exactly 3 variants (Gguf, SafeTensors, Apr)"
    );
}

#[test]
fn f_contract_006_validated_weight_is_row_major_only() {
    // ValidatedWeight::new() produces ValidatedWeight<RowMajor>.
    // There is no ColumnMajor type, so this is the only possible layout.
    let data = vec![0.1_f32, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
    let result: Result<ValidatedWeight<RowMajor>, _> =
        ValidatedWeight::new(data, 2, 4, "test.weight");

    assert!(
        result.is_ok(),
        "ValidatedWeight::new() must produce ValidatedWeight<RowMajor>"
    );
}

#[test]
fn f_prove_known_families_contains_expected_families() {
    let expected_families = [
        "bert", "deepseek", "gemma", "llama", "mistral", "phi", "qwen2", "whisper",
    ];

    for family in &expected_families {
        assert!(
            KNOWN_FAMILIES.contains(family),
            "KNOWN_FAMILIES must contain '{family}' (from contracts/model-families/{family}.yaml)"
        );
    }

    assert!(
        KNOWN_FAMILIES.len() >= expected_families.len(),
        "At least {} families expected, found {}",
        expected_families.len(),
        KNOWN_FAMILIES.len()
    );
}

#[test]
fn f_arch_format_type_conversions_are_well_formed() {
    assert!(
        FormatType::Apr.can_convert_to(FormatType::Gguf),
        "APR must be able to convert to GGUF"
    );
    assert!(
        FormatType::Apr.can_convert_to(FormatType::SafeTensors),
        "APR must be able to convert to SafeTensors"
    );
    assert!(
        !FormatType::Apr.can_convert_to(FormatType::Apr),
        "APR->APR self-conversion should not exist"
    );
}

#[test]
fn f_fmt_truncated_file_returns_error() {
    let mut temp = NamedTempFile::with_suffix(".bin").expect("Create temp file");
    temp.write_all(b"AB").expect("Write short data");
    temp.flush().expect("Flush");

    let format = FormatType::from_magic(temp.path());
    assert!(
        format.is_err(),
        "Truncated file (2 bytes) must return error, got: {:?}",
        format.ok()
    );
}

#[test]
fn f_fmt_empty_file_returns_error() {
    let temp = NamedTempFile::with_suffix(".bin").expect("Create temp file");

    let format = FormatType::from_magic(temp.path());
    assert!(
        format.is_err(),
        "Empty file must return error, got: {:?}",
        format.ok()
    );
}

#[test]
fn f_dod_unsafe_code_is_deny_by_default() {
    let src_dir = project_root().join("src");
    let mut violations = Vec::new();

    for path in collect_rs_files(&src_dir) {
        // mmap.rs has a documented exception per bundle-mmap-spec.md
        if path.to_string_lossy().contains("mmap.rs") {
            continue;
        }

        let content = std::fs::read_to_string(&path).unwrap_or_default();
        for (line_no, line) in content.lines().enumerate() {
            let trimmed = line.trim();
            if trimmed.starts_with("//")
                || trimmed.starts_with("///")
                || trimmed.starts_with("#[")
                || trimmed.starts_with("#![")
            {
                continue;
            }
            if trimmed.contains("unsafe ") || trimmed.contains("unsafe{") {
                violations.push(format!("{}:{}: '{trimmed}'", path.display(), line_no + 1));
            }
        }
    }

    assert!(
        violations.is_empty(),
        "No unsafe code allowed outside mmap.rs (deny(unsafe_code)).\n\
         Violations:\n{}",
        violations.join("\n")
    );
}

#[test]
fn f_arch_007_enforce_import_contract_never_needs_data_transpose() {
    // GGUF->APR import never needs data transpose (shape reversal only)
    let test_tensors = [
        ("token_embd.weight", vec![896, 151936]),
        ("output.weight", vec![896, 151936]),
        ("blk.0.attn_q.weight", vec![896, 896]),
        ("blk.0.attn_k.weight", vec![896, 128]),
        ("blk.0.ffn_gate.weight", vec![896, 4864]),
        ("output_norm.weight", vec![896]),
        ("blk.0.attn_norm.weight", vec![896]),
    ];

    for (name, shape) in &test_tensors {
        let (_, needs_transpose) = enforce_import_contract(name, shape, 151936, 896);
        assert!(
            !needs_transpose,
            "GGUF tensor '{}' must NEVER need data transpose \
             (GH-208: GGUF data layout IS row-major when shape is reversed)",
            name
        );
    }
}
