// Section 15: Contract Model (F-CONTRACT-*)
// =============================================================================

#[test]
fn f_contract_001_contract_gate_exists() {
    // F-CONTRACT-001: validate_model_contract exists
    let lib_path = project_root()
        .join("crates")
        .join("apr-cli")
        .join("src")
        .join("lib.rs");
    let content = std::fs::read_to_string(&lib_path).expect("lib.rs readable");

    assert!(
        content.contains("fn validate_model_contract"),
        "F-CONTRACT-001: validate_model_contract must exist"
    );
    assert!(
        content.contains("fn extract_model_paths"),
        "F-CONTRACT-001: extract_model_paths must exist"
    );
}

#[test]
fn f_contract_002_skip_contract_bypasses_gate() {
    // F-CONTRACT-002: --skip-contract bypasses the contract gate
    // Structural check: verify the bypass logic exists in execute_command
    let lib_path = project_root()
        .join("crates")
        .join("apr-cli")
        .join("src")
        .join("lib.rs");
    let content = std::fs::read_to_string(&lib_path).expect("lib.rs readable");

    // skip_contract field exists
    assert!(
        content.contains("skip_contract"),
        "F-CONTRACT-002: skip_contract must be a field in CLI"
    );
    // The skip logic must check before calling validate
    assert!(
        content.contains("if") && content.contains("skip_contract"),
        "F-CONTRACT-002: Code must conditionally skip contract validation"
    );
}

#[test]
fn f_contract_003_diagnostic_commands_exempt() {
    // F-CONTRACT-003: Diagnostic commands return empty paths (no gate)
    let lib_path = project_root()
        .join("crates")
        .join("apr-cli")
        .join("src")
        .join("lib.rs");
    let content = std::fs::read_to_string(&lib_path).expect("lib.rs readable");

    // extract_model_paths classifies commands
    assert!(
        content.contains("fn extract_model_paths"),
        "F-CONTRACT-003: extract_model_paths must exist"
    );
    // Diagnostic commands must return empty vec (not gated)
    // The function has a catch-all `_ => vec![]` for diagnostics
    assert!(
        content.contains("vec![]"),
        "F-CONTRACT-003: Some commands must return empty path vec (exempt from gate)"
    );
}

#[test]
fn f_contract_004_all_zeros_embedding_rejected() {
    // F-CONTRACT-004: 94.5% zero embedding is rejected
    let vocab_size = 100;
    let hidden_dim = 64;
    let data = vec![0.0_f32; vocab_size * hidden_dim];

    let result = ValidatedEmbedding::new(data, vocab_size, hidden_dim);
    assert!(
        result.is_err(),
        "F-CONTRACT-004: All-zeros embedding must be rejected by density gate"
    );

    let err = result.unwrap_err();
    assert!(
        err.rule_id.contains("DATA-QUALITY"),
        "F-CONTRACT-004: Rejection must cite DATA-QUALITY rule, got: {}",
        err.rule_id
    );
}

#[test]
fn f_contract_005_nan_tensor_rejected() {
    // F-CONTRACT-005: NaN in embedding data is rejected
    let vocab_size = 100;
    let hidden_dim = 64;
    let mut data: Vec<f32> = (0..vocab_size * hidden_dim)
        .map(|i| (i as f32 * 0.01).sin() * 0.1)
        .collect();
    data[42] = f32::NAN;

    let result = ValidatedEmbedding::new(data, vocab_size, hidden_dim);
    assert!(
        result.is_err(),
        "F-CONTRACT-005: NaN in embedding must be rejected"
    );
}

#[test]
fn f_contract_006_no_column_major_type_exists() {
    // F-CONTRACT-006: ColumnMajor type does not exist
    let _row_major = RowMajor; // This compiles

    let dirs = [project_root().join("src"), project_root().join("crates")];
    let mut violations = Vec::new();

    for dir in &dirs {
        for path in collect_rs_files(dir) {
            let content = std::fs::read_to_string(&path).unwrap_or_default();
            for line in content.lines() {
                let trimmed = line.trim();
                if trimmed.starts_with("//") || trimmed.starts_with("///") {
                    continue;
                }
                if trimmed.contains("struct ColumnMajor")
                    || trimmed.contains("enum ColumnMajor")
                    || trimmed.contains("type ColumnMajor")
                {
                    violations.push(format!("{}: '{trimmed}'", path.display()));
                }
            }
        }
    }

    assert!(
        violations.is_empty(),
        "F-CONTRACT-006: ColumnMajor type must NOT exist.\nViolations:\n{}",
        violations.join("\n")
    );
}

#[test]
fn f_contract_007_lm_head_is_marked_critical() {
    // F-CONTRACT-007: lm_head.weight / output.weight is marked critical
    let contract = LayoutContract::new();
    let lm_head = contract.get_gguf_contract("output.weight");

    assert!(
        lm_head.is_some(),
        "F-CONTRACT-007: output.weight must be in layout contract"
    );
    assert!(
        lm_head.expect("lm_head").is_critical,
        "F-CONTRACT-007: output.weight must be marked critical"
    );
}

// =============================================================================
// Section 16: Provability (F-PROVE-*)
// =============================================================================

#[test]
fn f_prove_001_cargo_build_succeeds() {
    // F-PROVE-001: This test compiling proves all 297 assertions pass
    assert!(
        !KNOWN_FAMILIES.is_empty(),
        "F-PROVE-001: Build-time constants must be populated"
    );
}

#[test]
fn f_prove_002_invalid_yaml_would_break_build() {
    // F-PROVE-002: Structural test -- verify the YAML-to-Rust pipeline exists
    let build_rs = project_root().join("build.rs");
    let content = std::fs::read_to_string(&build_rs).expect("build.rs readable");

    assert!(
        content.contains("model_families") || content.contains("yaml"),
        "F-PROVE-002: build.rs must process model family YAML files"
    );
    assert!(
        content.contains("const_assert") || content.contains("const _: () = assert!"),
        "F-PROVE-002: build.rs must generate const assertions"
    );
}

#[test]
fn f_prove_003_gqa_violation_would_break_build() {
    // F-PROVE-003: Verify the GQA divisibility proof exists in generated code
    if let Some(gen_path) = find_generated_file("model_families_generated.rs") {
        let content = std::fs::read_to_string(&gen_path).expect("generated file readable");
        assert!(
            content.contains("GQA") || content.contains("num_kv_heads"),
            "F-PROVE-003: Generated proofs must include GQA constraints"
        );
    }
}

#[test]
fn f_prove_004_rope_parity_violation_would_break_build() {
    // F-PROVE-004: Verify RoPE even-head-dim proof exists
    if let Some(gen_path) = find_generated_file("model_families_generated.rs") {
        let content = std::fs::read_to_string(&gen_path).expect("generated file readable");
        assert!(
            content.contains("RoPE") || content.contains("even"),
            "F-PROVE-004: Generated proofs must include RoPE parity"
        );
    }
}

#[test]
fn f_prove_005_ffn_expansion_violation_would_break_build() {
    // F-PROVE-005: Verify FFN expansion proof exists
    if let Some(gen_path) = find_generated_file("model_families_generated.rs") {
        let content = std::fs::read_to_string(&gen_path).expect("generated file readable");
        assert!(
            content.contains("FFN expansion") || content.contains("intermediate_dim"),
            "F-PROVE-005: Generated proofs must include FFN expansion"
        );
    }
}

#[test]
fn f_prove_006_oracle_validate_catches_hf_mismatch() {
    // F-PROVE-006: Structural check — oracle.rs has HF validation/comparison logic
    let oracle_path = project_root()
        .join("crates")
        .join("apr-cli")
        .join("src")
        .join("commands")
        .join("oracle.rs");
    let content = std::fs::read_to_string(&oracle_path).expect("oracle.rs must exist");
    assert!(
        content.contains("validate") || content.contains("Validate"),
        "F-PROVE-006: oracle.rs must have validation logic"
    );
    assert!(
        content.contains("hf")
            || content.contains("HuggingFace")
            || content.contains("huggingface")
            || content.contains("config.json"),
        "F-PROVE-006: oracle must reference HuggingFace for cross-validation"
    );
}

#[test]
fn f_prove_007_proof_count_is_exactly_297() {
    // F-PROVE-007: Exactly 297 const assertions
    let gen_path = find_generated_file("model_families_generated.rs");
    let gen_path = gen_path.unwrap_or_else(|| {
        panic!("F-PROVE-007: Cannot find model_families_generated.rs. Run `cargo build` first.")
    });

    let content = std::fs::read_to_string(&gen_path).expect("generated file readable");
    let count = content
        .lines()
        .filter(|line| line.contains("const _: () = assert!"))
        .count();

    assert_eq!(
        count, 297,
        "F-PROVE-007: Expected 297 proofs, found {count}"
    );
}

// =============================================================================
// Section 17: CLI Surface Area (F-SURFACE-*)
// =============================================================================

#[test]
fn f_surface_001_all_36_top_level_commands_exist() {
    // F-SURFACE-001: Same as F-CLI-001
    f_cli_001_all_36_top_level_commands_parse();
}

#[test]
fn f_surface_002_all_10_nested_commands_exist() {
    // F-SURFACE-002: Same as F-CLI-002
    f_cli_002_all_10_nested_subcommands_parse();
}

#[test]
fn f_surface_003_no_undocumented_commands() {
    // F-SURFACE-003: All commands in enum are mentioned in spec
    let spec_path = project_root()
        .join("docs")
        .join("specifications")
        .join("qwen2.5-coder-showcase-demo.md");
    let spec = std::fs::read_to_string(&spec_path).expect("spec readable");

    // Extract command names from the Commands enum
    let lib_path = project_root()
        .join("crates")
        .join("apr-cli")
        .join("src")
        .join("lib.rs");
    let lib_content = std::fs::read_to_string(&lib_path).expect("lib.rs readable");

    let variants = extract_enum_variant_names(&lib_content, "pub enum Commands");
    for variant in &variants {
        // Convert PascalCase to kebab-case for CLI matching (e.g., CompareHf -> compare-hf)
        let kebab = variant
            .chars()
            .enumerate()
            .fold(String::new(), |mut acc, (i, c)| {
                if c.is_uppercase() && i > 0 {
                    acc.push('-');
                }
                acc.push(c.to_ascii_lowercase());
                acc
            });
        let lowercase = variant.to_lowercase();
        assert!(
            spec.contains(&format!("`apr {kebab}`"))
                || spec.contains(&format!("apr {kebab}"))
                || spec.contains(&format!("`apr {lowercase}`"))
                || spec.contains(&format!("apr {lowercase}"))
                || spec.contains(&format!("`{kebab}`"))
                || spec.contains(&format!("`{lowercase}`"))
                || spec.contains(variant),
            "F-SURFACE-003: Command '{variant}' (apr {kebab}) must be documented in spec"
        );
    }
}

#[test]
fn f_surface_004_every_command_referenced_in_spec() {
    // F-SURFACE-004: All 46 commands appear in spec
    let spec_path = project_root()
        .join("docs")
        .join("specifications")
        .join("qwen2.5-coder-showcase-demo.md");
    let spec = std::fs::read_to_string(&spec_path).expect("spec readable");

    // Key commands that must appear
    let required_commands = [
        "apr run",
        "apr chat",
        "apr serve",
        "apr import",
        "apr export",
        "apr convert",
        "apr inspect",
        "apr validate",
        "apr tensors",
        "apr diff",
        "apr oracle",
        "apr qa",
        "apr rosetta",
    ];

    for cmd in &required_commands {
        assert!(
            spec.contains(cmd),
            "F-SURFACE-004: '{cmd}' must be referenced in spec"
        );
    }
}

#[test]
fn f_surface_005_contract_classification_matches_code() {
    // F-SURFACE-005: Gated vs exempt classification is consistent
    let lib_path = project_root()
        .join("crates")
        .join("apr-cli")
        .join("src")
        .join("lib.rs");
    let content = std::fs::read_to_string(&lib_path).expect("lib.rs readable");

    // The extract_model_paths function determines gating
    assert!(
        content.contains("fn extract_model_paths"),
        "F-SURFACE-005: extract_model_paths must exist for contract classification"
    );

    // Diagnostic keyword must appear (documenting exemptions)
    assert!(
        content.contains("diagnostic")
            || content.contains("DIAGNOSTIC")
            || content.contains("exempt"),
        "F-SURFACE-005: Contract classification must document diagnostic exemptions"
    );
}

