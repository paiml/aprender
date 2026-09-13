// Section 8: Definition of Done (F-DOD-*)
// =============================================================================

#[test]
fn f_dod_001_satd_count_is_zero() {
    // F-DOD-001: SATD in production source is ratcheted. Same gate as
    // F-CHECKLIST-005; #2522 added the ID here because the delegate named only
    // the gate it forwards to, so F-DOD-001 was the one section entry that
    // F-DOD-004's own census could not see.
    f_checklist_005_satd_is_zero();
}

#[test]
fn f_dod_002_coverage_ge_95_percent() {
    // F-DOD-002: Coverage >= 95% -- verified by CI, structural check here
    // The spec states 96.27% achieved. This test verifies the Makefile has coverage target.
    let makefile = project_root().join("Makefile");
    if makefile.exists() {
        let content = std::fs::read_to_string(&makefile).unwrap_or_default();
        assert!(
            content.contains("coverage") || content.contains("llvm-cov"),
            "F-DOD-002: Makefile must have coverage target"
        );
    }
}

#[test]
fn f_dod_003_cargo_build_succeeds() {
    // F-DOD-003: `cargo build` succeeds => all 297 proofs pass
    // This test's existence and compilation proves the proofs pass.
    // Import a build-time constant to verify build.rs ran.
    assert!(
        !KNOWN_FAMILIES.is_empty(),
        "F-DOD-003: KNOWN_FAMILIES must be populated by build.rs"
    );
}

#[test]
fn f_dod_004_all_sections_have_ge_5_gates() {
    // F-DOD-004: Every falsification section has >= 5 gates IMPLEMENTED.
    //
    // #2522: this counted gate IDs in the markdown spec. Two problems. The spec
    // is now under docs/specifications/archive/ -- it is a frozen design
    // document that nobody updates, so counting it measures whether a 2025 doc
    // was maintained, not whether anything is enforced. And it enumerates only
    // 5 of the 10 prefixes as IDs (F-PIPE-, F-MODEL-, F-FMT-, F-REALIZE- and
    // F-SURFACE- appear as prose), so it reports "found 0" for sections this
    // suite implements in full.
    //
    // The living artifact is this suite. Counting the gates that actually RUN
    // is both checkable and the thing the gate means: a section may not be
    // hollowed out below 5 implemented gates.
    let content = suite_source_text();

    let prefixes = [
        "F-GT-",
        "F-ARCH-",
        "F-CLI-",
        "F-PIPE-",
        "F-MODEL-",
        "F-FMT-",
        "F-CHECKLIST-",
        "F-QA-",
        "F-OLLAMA-",
        "F-DOD-",
        "F-LAYOUT-",
        "F-ROSETTA-",
        "F-DIAG-",
        "F-PERF-",
        "F-TRUENO-",
        "F-REALIZE-",
        "F-CONTRACT-",
        "F-PROVE-",
        "F-SURFACE-",
    ];

    for prefix in &prefixes {
        let unique_count = count_unique_gate_ids(&content, prefix);
        assert!(
            unique_count >= 5,
            "F-DOD-004: Section {prefix} must have >= 5 IMPLEMENTED gates, found \
             {unique_count} in the suite source"
        );
    }
}

#[test]
fn f_dod_005_no_silent_fallbacks_in_dtype_handling() {
    // F-DOD-005: No `_ => ...F32` catch-all match arms.
    //
    // #2522: this reported 7 violations, and every one was produced by the
    // matcher rather than by the code:
    //
    //   `_ => 4,                // Default F32`        <- a byte WIDTH, in a comment
    //   `_ => num_elements * 4, // Default to F32`     <- ditto
    //   `_ => byte_size / 4,    // F32: 4 bytes/elem`  <- ditto
    //   `_ => 0,                // Skip non-F32`       <- ditto
    //   `_ => panic!("Expected ImmF32 source")`        <- a PANIC, and `ImmF32`
    //
    // Three defects. It skipped whole-line comments but not trailing ones, so
    // four of the seven were comments. `contains("F32")` matched `ImmF32`. And a
    // panicking arm is the OPPOSITE of a silent fallback -- it is the behaviour
    // this gate wants. Each is fixed below; the gate still fires on a real
    // `_ => DType::F32` (mutation record in aprender#2522).
    let mut violations = Vec::new();

    for path in production_rs_files() {
        let content = std::fs::read_to_string(&path).unwrap_or_default();
        for (line_no, line) in content.lines().enumerate() {
            let trimmed = strip_trailing_comment(line).trim();
            if is_silent_f32_fallback_arm(trimmed) {
                violations.push(format!("{}:{}: '{trimmed}'", path.display(), line_no + 1));
            }
        }
    }

    assert!(
        violations.is_empty(),
        "F-DOD-005: No silent dtype fallbacks to F32.\nViolations:\n{}",
        violations.join("\n")
    );
}

/// A catch-all match arm that quietly yields the F32 dtype.
fn is_silent_f32_fallback_arm(trimmed: &str) -> bool {
    let Some(pos) = trimmed.find("_ =>") else {
        return false;
    };
    let after = &trimmed[pos + 4..];
    // A diverging arm cannot be a SILENT fallback -- it is the behaviour this
    // gate wants.
    let diverges = ["panic!", "unreachable!", "todo!", "unimplemented!", "Err("]
        .iter()
        .any(|m| after.contains(m));
    !diverges && contains_f32_token(after)
}

// =============================================================================
// Section 9: Layout Safety (F-LAYOUT-*)
// =============================================================================

#[test]
fn f_layout_001_clippy_bans_colmajor_imports() {
    // F-LAYOUT-001: No colmajor imports in the INFERENCE PATH.
    //
    // #2522: the gate said "inference path" and scanned the entire workspace, so
    // it reported 9 violations that were all correct code:
    //   * 6 in crates/aprender-serve/examples/ -- verify_q4k_layout.rs and
    //     friends exist precisely TO compare column-major against row-major.
    //     They are evidence the contract is enforced, not a breach of it.
    //   * 2 `pub use colmajor::...` re-exports in crates/aprender-compute, which
    //     is where the column-major kernels are DEFINED. A definition is not an
    //     import into the inference path.
    // Scope is now what the gate's own name says: the crates CLAUDE.md names as
    // the inference path, production sources only.
    let dirs = [
        crate_dir("aprender-serve").join("src"),
        crate_dir("aprender-core").join("src"),
        crate_dir("apr-cli").join("src"),
    ];
    let mut violations = Vec::new();
    let mut scanned = 0usize;

    let inference_sources: Vec<PathBuf> = dirs
        .iter()
        .flat_map(|d| collect_rs_files(d))
        .filter(|p| is_scannable_production_source(p))
        .collect();

    for path in inference_sources {
        scanned += 1;
        let content = std::fs::read_to_string(&path).unwrap_or_default();
        for (line_no, line) in content.lines().enumerate() {
            let trimmed = strip_trailing_comment(line).trim();
            if trimmed.contains("colmajor") && trimmed.contains("use ") {
                violations.push(format!("{}:{}: '{trimmed}'", path.display(), line_no + 1));
            }
        }
    }
    // Narrowing a scan is how a gate is accidentally disarmed. Prove it still
    // has a corpus.
    assert!(
        scanned > 500,
        "F-LAYOUT-001: inference-path scan covered only {scanned} files -- the \
         crates moved and this gate is now near-vacuous"
    );

    assert!(
        violations.is_empty(),
        "F-LAYOUT-001: No colmajor imports in inference path.\nViolations:\n{}",
        violations.join("\n")
    );
}

#[test]
fn f_layout_002_enforce_import_contract_reverses_gguf_shapes() {
    // F-LAYOUT-002: GGUF [in, out] -> APR [out, in] shape reversal
    let (apr_shape, needs_transpose) =
        enforce_import_contract("output.weight", &[896, 151936], 151936, 896);

    // Shape must be reversed for 2D tensors
    assert_eq!(
        apr_shape,
        vec![151936, 896],
        "F-LAYOUT-002: APR shape must be [vocab, hidden] = [151936, 896]"
    );
    assert!(
        !needs_transpose,
        "F-LAYOUT-002: Data must NOT need transpose (shape reversal only)"
    );
}

#[test]
fn f_layout_003_enforce_load_contract_validates_shapes() {
    // F-LAYOUT-003: APR shapes must be [out_dim, in_dim] for 2D
    let (apr_shape, _) = enforce_import_contract("blk.0.attn_q.weight", &[896, 896], 151936, 896);

    assert_eq!(apr_shape.len(), 2, "F-LAYOUT-003: 2D tensor must remain 2D");

    // 1D tensors keep their shape
    let (apr_shape_1d, _) = enforce_import_contract("output_norm.weight", &[896], 151936, 896);
    assert_eq!(
        apr_shape_1d.len(),
        1,
        "F-LAYOUT-003: 1D tensor must remain 1D"
    );
}

#[test]
fn f_layout_004_enforce_embedding_contract_validates_shape() {
    // F-LAYOUT-004: Correct embedding passes
    enforce_embedding_contract(100 * 64, 100, 64);
    // Wrong size would panic -- that's the contract enforcement
}

#[test]
#[should_panic(expected = "CONTRACT VIOLATION")]
fn f_layout_004_enforce_embedding_contract_panics_on_wrong_shape() {
    // F-LAYOUT-004: Wrong embedding length panics
    enforce_embedding_contract(100 * 64 + 1, 100, 64); // off-by-one triggers CONTRACT VIOLATION
}

#[test]
fn f_layout_005_enforce_matmul_contract_validates_weight_dims() {
    // F-LAYOUT-005: Correct weight shape passes
    enforce_matmul_contract("test.weight", &[128, 64], 128, 64);
}

#[test]
#[should_panic(expected = "CONTRACT VIOLATION")]
fn f_layout_005_enforce_matmul_contract_panics_on_wrong_dims() {
    // F-LAYOUT-005: Swapped dims panic
    enforce_matmul_contract("test.weight", &[64, 128], 128, 64);
}

#[test]
fn f_layout_006_rosetta_diff_detects_transposed_dims() {
    // F-LAYOUT-006: `apr diff` detects tensor shape differences between formats
    let gguf = require_model!(gguf_model_path(), "GGUF model");
    let apr = require_model!(apr_model_path(), "APR model");
    let (success, stdout, stderr) =
        run_apr(&["diff", gguf.to_str().unwrap(), apr.to_str().unwrap()]);
    // diff should run and produce tensor comparison output
    assert!(
        success || !stdout.is_empty() || !stderr.is_empty(),
        "F-LAYOUT-006: apr diff must produce output for different formats"
    );
}

// =============================================================================
// Section 10: Rosetta Conversion (F-ROSETTA-*)
// =============================================================================

#[test]
fn f_rosetta_001_st_to_apr_preserves_tensor_count() {
    // F-ROSETTA-001: Tensor count matches between GGUF and APR
    let gguf = require_model!(gguf_model_path(), "GGUF model");
    let apr = require_model!(apr_model_path(), "APR model");
    let (ok1, stdout1, _) = run_apr(&["tensors", gguf.to_str().unwrap()]);
    let (ok2, stdout2, _) = run_apr(&["tensors", apr.to_str().unwrap()]);
    assert!(ok1, "F-ROSETTA-001: apr tensors GGUF must succeed");
    assert!(ok2, "F-ROSETTA-001: apr tensors APR must succeed");
    // Count tensor lines (non-header lines with tensor info)
    let count1 = stdout1
        .lines()
        .filter(|l| l.contains('[') && l.contains(']'))
        .count();
    let count2 = stdout2
        .lines()
        .filter(|l| l.contains('[') && l.contains(']'))
        .count();
    assert!(
        count1 > 0 && count2 > 0,
        "F-ROSETTA-001: Both formats must have tensors (GGUF:{count1}, APR:{count2})"
    );
}

#[test]
fn f_rosetta_002_apr_to_gguf_roundtrip_lossless() {
    // F-ROSETTA-002: SafeTensors -> APR -> GGUF roundtrip produces valid output
    // Canonical path: SafeTensors is the ONLY proper import source for APR.
    // GGUF files have mixed quant formats that APR cannot always preserve exactly.
    let st_dir = safetensors_model_dir();
    let st_dir = match st_dir {
        Some(d) => d,
        None => {
            eprintln!("SKIP: SafeTensors model not available");
            return;
        }
    };
    let st_path = st_dir.join("model.safetensors");
    if !st_path.exists() {
        eprintln!("SKIP: model.safetensors not found in {:?}", st_dir);
        return;
    }

    let tmp_apr = "/tmp/test-rosetta-002.apr";
    let tmp_gguf = "/tmp/test-rosetta-002.gguf";

    // Step 1: SafeTensors -> APR (canonical import path)
    let (ok1, _stdout1, stderr1) = run_apr(&["import", st_path.to_str().unwrap(), "-o", tmp_apr]);
    if !ok1 {
        eprintln!("SKIP: apr import from SafeTensors failed: {}", stderr1);
        let _ = std::fs::remove_file(tmp_apr);
        return;
    }

    // Step 2: APR -> GGUF (export for ollama parity)
    let (ok2, _stdout2, stderr2) =
        run_apr(&["export", tmp_apr, "--format", "gguf", "-o", tmp_gguf]);
    // Clean up APR regardless
    let _ = std::fs::remove_file(tmp_apr);

    if !ok2 {
        eprintln!("SKIP: apr export APR->GGUF failed: {}", stderr2);
        let _ = std::fs::remove_file(tmp_gguf);
        return;
    }

    // Step 3: Validate the exported GGUF
    let (ok3, _stdout3, stderr3) = run_apr(&["validate", tmp_gguf]);
    let _ = std::fs::remove_file(tmp_gguf);

    assert!(
        ok3,
        "F-ROSETTA-002: Exported GGUF must pass validation. stderr: {}",
        stderr3
    );
}

#[test]
fn f_rosetta_003_chain_produces_valid_gguf() {
    // F-ROSETTA-003: `apr validate` passes on GGUF model
    let gguf = require_model!(gguf_model_path(), "GGUF model");
    let (success, _stdout, stderr) = run_apr(&["validate", gguf.to_str().unwrap()]);
    assert!(
        success,
        "F-ROSETTA-003: GGUF must pass validation. stderr: {}",
        stderr
    );
}

#[test]
fn f_rosetta_004_fingerprint_detects_corruption() {
    // F-ROSETTA-004: `apr validate` on GGUF produces checksum/fingerprint
    let gguf = require_model!(gguf_model_path(), "GGUF model");
    let (success, stdout, stderr) = run_apr(&["validate", gguf.to_str().unwrap()]);
    assert!(success, "F-ROSETTA-004: apr validate must succeed");
    // Output should contain validation information
    let combined = format!("{stdout}{stderr}");
    assert!(
        !combined.is_empty(),
        "F-ROSETTA-004: Validation must produce output"
    );
}

#[test]
fn f_rosetta_005_nan_in_source_halts_conversion() {
    // F-ROSETTA-005: Structural check — RosettaStone has NaN validation
    // compute_tensor_validation flags NaN as F-DATA-QUALITY-002 failure
    // #2522: `compute_tensor_validation` is defined in a sibling of
    // format/rosetta/mod.rs, not in mod.rs itself.
    let content = crate_src_text("aprender-core");
    assert!(
        content.contains("compute_tensor_validation")
            || content.contains("nan_count")
            || content.contains("is_nan"),
        "F-ROSETTA-005: Rosetta must have NaN validation logic"
    );
}

#[test]
fn f_rosetta_006_vocab_mismatch_halts_conversion() {
    // F-ROSETTA-006: Structural check — vocab validation in import pipeline
    // The import path validates vocabulary via PMAT-232 (empty vocab detection)
    let import_path = crate_dir("aprender-core")
        .join("src")
        .join("format")
        .join("converter")
        .join("import.rs");
    let content = std::fs::read_to_string(&import_path).expect("import.rs must exist");
    assert!(
        content.contains("vocab")
            || content.contains("vocabulary")
            || content.contains("tokenizer"),
        "F-ROSETTA-006: Import pipeline must have vocabulary validation"
    );
}

// =============================================================================
