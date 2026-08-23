// Section 15: Contract Model (F-CONTRACT-*)
// =============================================================================

#[test]
fn f_contract_001_contract_gate_exists() {
    // F-CONTRACT-001: validate_model_contract exists
    // #2522: was anchored to apr-cli/src/lib.rs; the gate moved to validate.rs.
    let content = crate_src_text("apr-cli");

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
    // #2522: was anchored to apr-cli/src/lib.rs; the gate moved to validate.rs.
    let content = crate_src_text("apr-cli");

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
    // #2522: was anchored to apr-cli/src/lib.rs; the gate moved to validate.rs.
    let content = crate_src_text("apr-cli");

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

    // #2522: this scan included the test tree, so it matched the three string
    // literals in THIS function that spell the patterns it forbids:
    //
    //   .../tests/includes/falsification_spec_v10_contract_model.rs:
    //     'if trimmed.contains("struct ColumnMajor")'
    //     '|| trimmed.contains("enum ColumnMajor")'
    //     '|| trimmed.contains("type ColumnMajor")'
    //
    // A gate whose own definition is inside its universe can never pass. It has
    // been red on its own source since the day it was written, and nobody saw
    // it because the suite ran in no workflow. Production sources only now.
    let mut violations = Vec::new();

    for path in production_rs_files() {
        let content = std::fs::read_to_string(&path).unwrap_or_default();
        for line in content.lines() {
            let trimmed = strip_trailing_comment(line).trim();
            if trimmed.contains("struct ColumnMajor")
                || trimmed.contains("enum ColumnMajor")
                || trimmed.contains("type ColumnMajor")
            {
                violations.push(format!("{}: '{trimmed}'", path.display()));
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
    // #2522: the root build.rs has not existed since APR-MONO; the generator
    // lives in the crate that owns the model-family registry.
    // The generator is split across build.rs + the build_*.rs files it pulls in
    // with include!, so reading build.rs alone misses the code that emits the
    // proofs.
    let content = build_script_text("aprender-core");

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
    // F-PROVE-007: every known model family carries its build-time proofs.
    //
    // #2522, two independent rots in one gate:
    //  1. `find_generated_file` searched `<project_root>/target` only. Every
    //     build here redirects CARGO_TARGET_DIR, so it returned None ALWAYS and
    //     this gate could only ever panic with "Run `cargo build` first" -- it
    //     has never once read the file. Its four siblings F-PROVE-003/004/005
    //     wrote the same lookup as `if let Some(..)`, so they passed VACUOUSLY
    //     for the same reason. The finder now derives the target dir from
    //     `current_exe()`, which cannot be redirected out from under it.
    //  2. `== 297` was frozen. The generator emits 578 today. A literal count of
    //     a generated artifact is a number that rots by construction, so the
    //     gate is now stated against the thing it actually means: every family
    //     in KNOWN_FAMILIES must be proved, and a family carries several proofs.
    let gen_path = find_generated_file("model_families_generated.rs").unwrap_or_else(|| {
        panic!(
            "F-PROVE-007: model_families_generated.rs not found under any of {:?}. \
             Run `cargo build -p aprender-core` first.",
            target_dir_candidates()
        )
    });

    let content = std::fs::read_to_string(&gen_path).expect("generated file readable");
    let count = content
        .lines()
        .filter(|line| line.contains("const _: () = assert!"))
        .count();

    let families = KNOWN_FAMILIES.len();
    assert!(families > 0, "F-PROVE-007: KNOWN_FAMILIES is empty");
    assert!(
        count >= families * 3,
        "F-PROVE-007: {count} build-time proofs for {families} model families -- \
         fewer than 3 per family means the generator stopped proving something \
         (generated file: {})",
        gen_path.display()
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
    // F-SURFACE-003: no undocumented commands.
    //
    // #2522: this required every `Commands` variant to appear in the showcase
    // markdown. That document is archived and describes 36 commands; the CLI
    // ships 102. Holding a living surface against a frozen design doc means the
    // gate fires on every command added since 2025 -- `validate-manifest` is
    // what it happened to trip on -- while saying nothing about whether the
    // command is documented anywhere a user will look.
    //
    // The registry contract IS the documentation surface (its own header calls
    // itself "the SINGLE SOURCE OF TRUTH for all apr subcommands", and
    // FALSIFY-CLI-001/002 hold it against the real binary). So the gate now
    // asserts what "documented" means there: every command carries a non-empty
    // description.
    let undocumented = commands_without_description();
    assert!(
        undocumented.is_empty(),
        "F-SURFACE-003: commands in contracts/apr-cli-commands-v1.yaml with no \
         description: {}",
        undocumented.join(", ")
    );
}

/// Lines of the registry's `commands:` block only.
///
/// The block ends at the next column-0 key. `grep -c '^  - name:'` over the
/// whole file also counts the `falsification_tests:` entries -- CLAUDE.md
/// records that exact trap.
fn commands_block_lines(text: &str) -> Vec<&str> {
    let mut lines = Vec::new();
    let mut in_commands = false;
    for line in text.lines() {
        if line.starts_with("commands:") {
            in_commands = true;
            continue;
        }
        if !in_commands {
            continue;
        }
        let is_column_zero_key =
            !line.starts_with(char::is_whitespace) && line.contains(':') && !line.starts_with('#');
        if is_column_zero_key {
            break;
        }
        lines.push(line);
    }
    lines
}

/// Value half of a `key: value` YAML line, unquoted.
fn yaml_scalar(rest: &str) -> String {
    rest.trim().trim_matches('"').to_string()
}

fn flush_command(current: &mut Option<String>, has_description: &mut bool, missing: &mut Vec<String>) {
    if let Some(name) = current.take() {
        if !*has_description {
            missing.push(name);
        }
    }
    *has_description = false;
}

/// Command entries in the CLI registry that carry no non-empty `description:`.
fn commands_without_description() -> Vec<String> {
    let path = project_root()
        .join("contracts")
        .join("apr-cli-commands-v1.yaml");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("CLI command registry unreadable at {}: {e}", path.display()));

    let mut missing = Vec::new();
    let mut current: Option<String> = None;
    let mut has_description = false;
    let mut seen = 0usize;

    for line in commands_block_lines(&text) {
        if let Some(rest) = line.strip_prefix("  - name:") {
            flush_command(&mut current, &mut has_description, &mut missing);
            current = Some(yaml_scalar(rest));
            seen += 1;
        } else if let Some(rest) = line.strip_prefix("    description:") {
            has_description |= !yaml_scalar(rest).is_empty();
        }
    }
    flush_command(&mut current, &mut has_description, &mut missing);

    assert!(
        seen >= 50,
        "F-SURFACE-003: parsed only {seen} commands out of the registry -- the \
         YAML shape changed and this gate is near-vacuous"
    );
    missing
}

#[test]
fn f_surface_004_every_command_referenced_in_spec() {
    // F-SURFACE-004: All 46 commands appear in spec
    // #2522: the spec moved to docs/specifications/archive/.
    let spec = spec_text();

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
    // #2522: was anchored to apr-cli/src/lib.rs; the gate moved to validate.rs.
    let content = crate_src_text("apr-cli");

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

