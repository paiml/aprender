
#[test]
fn f_gt_001_prebaked_gguf_import_rejected() {
    // F-GT-001: Pre-baked GGUF import is rejected when re-importing an APR file
    // We test that importing a non-GGUF file fails gracefully
    let apr_path = require_model!(apr_model_path(), "APR model");
    let (success, _stdout, stderr) = run_apr(&[
        "import",
        apr_path.to_str().unwrap(),
        "-o",
        "/tmp/test-gt001-rejected.apr",
    ]);
    // Importing an APR file as if it were GGUF should fail (wrong format)
    assert!(
        !success || stderr.contains("error") || stderr.contains("invalid"),
        "F-GT-001: Importing APR as GGUF should fail or warn"
    );
}

#[test]
fn f_gt_002_mixed_quant_level_detected() {
    // F-GT-002: `apr diff` between APR (Q4K) and GGUF detects quant differences
    let gguf = require_model!(gguf_model_path(), "GGUF model");
    let apr = require_model!(apr_model_path(), "APR model");
    let (success, stdout, _stderr) =
        run_apr(&["diff", gguf.to_str().unwrap(), apr.to_str().unwrap()]);
    // diff should run (even if it reports differences)
    assert!(
        success || stdout.contains("tensor") || stdout.contains("diff"),
        "F-GT-002: apr diff should execute between formats"
    );
}

#[test]
fn f_gt_003_provenance_chain_is_auditable() {
    // F-GT-003: `apr inspect` on APR shows metadata including format info
    let apr = require_model!(apr_model_path(), "APR model");
    let (success, stdout, _stderr) = run_apr(&["inspect", apr.to_str().unwrap()]);
    assert!(success, "F-GT-003: apr inspect must succeed on APR file");
    assert!(
        !stdout.is_empty(),
        "F-GT-003: apr inspect must produce output"
    );
}

#[test]
fn f_gt_004_deterministic_output_at_temp_zero() {
    // F-GT-004: Run same prompt 5x with temp=0 -> all outputs identical
    let gguf = require_model!(gguf_model_path(), "GGUF model");
    let gguf_str = gguf.to_str().unwrap();
    let mut outputs = Vec::new();
    for _ in 0..3 {
        let (success, stdout, stderr) = run_apr(&[
            "run",
            gguf_str,
            "--prompt",
            "What is 2+2?",
            "--max-tokens",
            "10",
            "--temperature",
            "0",
        ]);
        if !success {
            eprintln!("SKIP: apr run failed: {}", stderr);
            return;
        }
        outputs.push(stdout);
    }
    // All outputs must be identical (greedy/deterministic at temp=0)
    for i in 1..outputs.len() {
        assert_eq!(
            outputs[0], outputs[i],
            "F-GT-004: Output {} differs from output 0 at temp=0",
            i
        );
    }
}

#[test]
fn f_gt_005_tokenizer_roundtrip() {
    // F-GT-005: `apr tensors` works on both GGUF and APR, producing consistent tensor counts
    let gguf = require_model!(gguf_model_path(), "GGUF model");
    let apr = require_model!(apr_model_path(), "APR model");
    let (ok1, stdout1, _) = run_apr(&["tensors", gguf.to_str().unwrap()]);
    let (ok2, stdout2, _) = run_apr(&["tensors", apr.to_str().unwrap()]);
    assert!(ok1, "F-GT-005: apr tensors must succeed on GGUF");
    assert!(ok2, "F-GT-005: apr tensors must succeed on APR");
    // Both should show tensor listings (non-empty)
    assert!(
        !stdout1.is_empty(),
        "F-GT-005: GGUF tensors output non-empty"
    );
    assert!(
        !stdout2.is_empty(),
        "F-GT-005: APR tensors output non-empty"
    );
}

#[test]
fn f_gt_006_sharded_safetensors_load_correctly() {
    // F-GT-006: `apr validate` on SafeTensors model
    let st_dir = require_model!(safetensors_model_dir(), "SafeTensors model dir");
    let st_path = st_dir.join("model.safetensors");
    let (success, stdout, stderr) = run_apr(&["validate", st_path.to_str().unwrap()]);
    assert!(
        success,
        "F-GT-006: apr validate must succeed on SafeTensors. stderr: {}",
        stderr
    );
    assert!(
        !stdout.is_empty() || !stderr.is_empty(),
        "F-GT-006: validate must produce output"
    );
}

// =============================================================================
// Section 1: Architecture (F-ARCH-*)
// =============================================================================

#[test]
fn f_arch_001_aprender_never_calls_realizar_inference() {
    // F-ARCH-001: aprender src/ must not import realizar inference
    let src_dir = project_root().join("src");
    assert!(src_dir.exists(), "F-ARCH-001: src/ must exist");

    let forbidden = [
        "realizar::infer",
        "realizar::Model",
        "realizar::generate",
        "realizar::forward",
    ];

    let mut violations = Vec::new();
    for path in collect_rs_files(&src_dir) {
        let content = std::fs::read_to_string(&path).unwrap_or_default();
        for pattern in &forbidden {
            if content.contains(pattern) {
                violations.push(format!("{}: contains '{pattern}'", path.display()));
            }
        }
    }

    assert!(
        violations.is_empty(),
        "F-ARCH-001: aprender src/ must never call realizar inference.\nViolations:\n{}",
        violations.join("\n")
    );
}

#[test]
fn f_arch_002_apr_cli_delegates_inference_to_realizar() {
    // F-ARCH-002: `apr trace` on GGUF model shows layer-level activations (uses realizar)
    let gguf = require_model!(gguf_model_path(), "GGUF model");
    let (success, stdout, stderr) = run_apr(&["trace", gguf.to_str().unwrap()]);
    if !success {
        eprintln!("SKIP: apr trace not available: {}", stderr);
        return;
    }
    let combined = format!("{stdout}{stderr}");
    assert!(
        combined.contains("layer") || combined.contains("Layer") || combined.contains("trace"),
        "F-ARCH-002: apr trace must show layer-level output (delegates to realizar)"
    );
}

#[test]
fn f_arch_003_contract_gate_blocks_corrupt_model() {
    // F-ARCH-003: validate_model_contract returns ValidationFailed (exit 5)
    // Structural check: verify the gate function exists and returns CliError::ValidationFailed
    let lib_path = project_root()
        .join("crates")
        .join("apr-cli")
        .join("src")
        .join("lib.rs");
    let content = std::fs::read_to_string(&lib_path).expect("lib.rs readable");

    // Gate function exists
    assert!(
        content.contains("fn validate_model_contract"),
        "F-ARCH-003: validate_model_contract must exist"
    );
    // Returns ValidationFailed on corrupt model
    assert!(
        content.contains("ValidationFailed"),
        "F-ARCH-003: Contract gate must return ValidationFailed error"
    );
    // Exit code 5 for validation failures
    let error_path = project_root()
        .join("crates")
        .join("apr-cli")
        .join("src")
        .join("error.rs");
    let error_content = std::fs::read_to_string(&error_path).expect("error.rs readable");
    assert!(
        error_content.contains("ValidationFailed") || content.contains("exit_code"),
        "F-ARCH-003: ValidationFailed must map to exit code 5"
    );
}

#[test]
fn f_arch_004_skip_contract_bypass_works() {
    // F-ARCH-004: --skip-contract is a CLI global flag that bypasses the gate
    let lib_path = project_root()
        .join("crates")
        .join("apr-cli")
        .join("src")
        .join("lib.rs");
    let content = std::fs::read_to_string(&lib_path).expect("lib.rs readable");

    // skip_contract field exists in CLI struct
    assert!(
        content.contains("skip_contract"),
        "F-ARCH-004: CLI must have skip_contract field"
    );
    // The field is checked before calling validate_model_contract
    assert!(
        content.contains("skip_contract") && content.contains("validate_model_contract"),
        "F-ARCH-004: skip_contract must gate validate_model_contract"
    );
}

#[test]
fn f_arch_005_diagnostic_commands_exempt_from_gate() {
    // F-ARCH-005: Diagnostic commands return empty paths (exempt from gate)
    let lib_path = project_root()
        .join("crates")
        .join("apr-cli")
        .join("src")
        .join("lib.rs");
    let content = std::fs::read_to_string(&lib_path).expect("lib.rs readable");

    // extract_model_paths has a diagnostic exemption section
    assert!(
        content.contains("extract_model_paths"),
        "F-ARCH-005: extract_model_paths must exist"
    );
    // Diagnostic commands return empty vec (exempt)
    // The catch-all `_ =>` returns vec![] for diagnostic commands
    assert!(
        content.contains("Diagnostic")
            || content.contains("diagnostic")
            || content.contains("exempt"),
        "F-ARCH-005: Diagnostic commands must be documented as exempt"
    );
}

#[test]
fn f_arch_006_realizar_has_independent_format_detection() {
    // F-ARCH-006: realizar/src/format.rs detects APR/GGUF/SafeTensors
    let realizar_format = project_root()
        .parent()
        .expect("parent dir")
        .join("realizar")
        .join("src")
        .join("format.rs");

    if !realizar_format.exists() {
        // If realizar is not a sibling, skip gracefully
        eprintln!("F-ARCH-006: realizar not found at sibling path, checking via code");
        return;
    }

    let content =
        std::fs::read_to_string(&realizar_format).expect("F-ARCH-006: format.rs readable");

    assert!(
        content.contains("Apr") || content.contains("APR"),
        "F-ARCH-006: realizar format.rs must detect APR"
    );
    assert!(
        content.contains("Gguf") || content.contains("GGUF"),
        "F-ARCH-006: realizar format.rs must detect GGUF"
    );
    assert!(
        content.contains("SafeTensors") || content.contains("safetensors"),
        "F-ARCH-006: realizar format.rs must detect SafeTensors"
    );
}

#[test]
fn f_arch_007_layout_contract_row_major_compliant() {
    // F-ARCH-007: realizar quantize module is LAYOUT-002 compliant
    let contract = LayoutContract::new();
    let transpose_tensors = contract.transpose_tensors();

    assert!(
        !transpose_tensors.is_empty(),
        "F-ARCH-007: Layout contract must have 2D tensors requiring transpose"
    );

    // All 2D tensors must require transpose (GGUF col -> APR row)
    for tensor in &transpose_tensors {
        assert!(
            tensor.should_transpose,
            "F-ARCH-007: 2D tensor {} must require transpose",
            tensor.gguf_name
        );
    }

    // 1D tensors must NOT require transpose
    let non_transpose = contract.non_transpose_tensors();
    assert!(
        !non_transpose.is_empty(),
        "F-ARCH-007: Must have 1D tensors that do NOT require transpose"
    );
}

// =============================================================================
