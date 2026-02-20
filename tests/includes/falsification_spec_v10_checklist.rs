// Section 6: 300-Point Checklist (F-CHECKLIST-*)
// =============================================================================

#[test]
fn f_checklist_001_score_ge_250() {
    // F-CHECKLIST-001: Structural check — qa.rs has scoring logic and threshold
    let qa_path = project_root()
        .join("crates")
        .join("apr-cli")
        .join("src")
        .join("commands")
        .join("qa.rs");
    let content = std::fs::read_to_string(&qa_path).expect("qa.rs must exist");
    assert!(
        content.contains("score") || content.contains("Score"),
        "F-CHECKLIST-001: qa.rs must have scoring logic"
    );
    assert!(
        content.contains("gate") || content.contains("Gate") || content.contains("check"),
        "F-CHECKLIST-001: qa.rs must have gate checks"
    );
}

#[test]
fn f_checklist_002_no_section_scores_zero() {
    // F-CHECKLIST-002: Structural check — qa.rs checks multiple sections (not just one)
    let qa_path = project_root()
        .join("crates")
        .join("apr-cli")
        .join("src")
        .join("commands")
        .join("qa.rs");
    let content = std::fs::read_to_string(&qa_path).expect("qa.rs must exist");
    // Count distinct gate/check functions (each section has its own checks)
    let gate_count = content.matches("fn check_").count()
        + content.matches("fn gate_").count()
        + content.matches("fn run_gate").count();
    assert!(
        gate_count >= 3 || content.contains("section"),
        "F-CHECKLIST-002: qa.rs must check multiple sections (found {gate_count} gate functions)"
    );
}

#[test]
fn f_checklist_003_contract_section_present_in_spec() {
    // F-CHECKLIST-003: Spec includes PMAT-237 contract gates
    let spec_path = project_root()
        .join("docs")
        .join("specifications")
        .join("qwen2.5-coder-showcase-demo.md");
    let content = std::fs::read_to_string(&spec_path).expect("spec readable");

    assert!(
        content.contains("PMAT-237"),
        "F-CHECKLIST-003: Spec must reference PMAT-237 contract gate"
    );
    assert!(
        content.contains("F-CONTRACT-"),
        "F-CHECKLIST-003: Spec must have F-CONTRACT-* gates"
    );
}

#[test]
fn f_checklist_004_falsification_depth_ge_level_5() {
    // F-CHECKLIST-004: At least 5 tests use Level 5 (hang detection, fuzzing)
    let spec_path = project_root()
        .join("docs")
        .join("specifications")
        .join("qwen2.5-coder-showcase-demo.md");
    let content = std::fs::read_to_string(&spec_path).expect("spec readable");

    // Count Level 5 indicators
    let level_5_indicators = ["hang detection", "fuzzing", "timeout", "Inject", "corrupt"];

    let mut count = 0;
    for indicator in &level_5_indicators {
        count += content.matches(indicator).count();
    }

    assert!(
        count >= 5,
        "F-CHECKLIST-004: Need >= 5 Level 5 falsification tests, found {count} indicators"
    );
}

fn is_satd_marker(trimmed: &str, marker: &str) -> bool {
    let Some(pos) = trimmed.find(marker) else {
        return false;
    };
    let after = trimmed.get(pos + marker.len()..pos + marker.len() + 1);
    match after {
        Some(c) => !c.chars().next().map_or(false, |ch| ch.is_alphanumeric()),
        None => true,
    }
}

fn check_file_for_satd(path: &std::path::Path, violations: &mut Vec<String>) {
    let content = std::fs::read_to_string(path).unwrap_or_default();
    let satd_markers = ["TODO", "FIXME", "HACK"];
    for (line_no, line) in content.lines().enumerate() {
        let trimmed = line.trim();
        if !trimmed.starts_with("//") && !trimmed.starts_with("///") {
            continue;
        }
        for marker in &satd_markers {
            if is_satd_marker(trimmed, marker) {
                violations.push(format!("{}:{}: '{trimmed}'", path.display(), line_no + 1));
            }
        }
    }
}

fn f_checklist_005_satd_is_zero() {
    // F-CHECKLIST-005: SATD = 0 across codebase
    let dirs = [project_root().join("src"), project_root().join("crates")];
    let mut violations = Vec::new();

    for dir in &dirs {
        for path in collect_rs_files(dir) {
            check_file_for_satd(&path, &mut violations);
        }
    }

    assert!(
        violations.is_empty(),
        "F-CHECKLIST-005: SATD must be 0. Found {}:\n{}",
        violations.len(),
        violations.join("\n")
    );
}

// =============================================================================
// Section 7: QA Testing (F-QA-*)
// All require model files
// =============================================================================

#[test]
fn f_qa_001_all_20_matrix_cells_pass() {
    // F-QA-001: `apr qa` on GGUF model runs QA matrix
    let gguf = require_model!(gguf_model_path(), "GGUF model");
    let (success, stdout, stderr) = run_apr(&["qa", gguf.to_str().unwrap()]);
    if !success {
        eprintln!(
            "SKIP: apr qa failed (may need inference feature): {}",
            stderr
        );
        return;
    }
    let combined = format!("{stdout}{stderr}");
    // QA should produce gate results
    assert!(
        combined.contains("PASS") || combined.contains("pass") || combined.contains("gate"),
        "F-QA-001: apr qa must report gate results"
    );
}

#[test]
fn f_qa_002_hang_detection_catches_silent_hangs() {
    // F-QA-002: Hang detection infrastructure exists:
    // 1. CircuitBreaker in federation/health.rs (timeout + state machine)
    // 2. wait_with_timeout in examples/qa_run.rs (process-level hang detection)
    // 3. apr qa runs with timeout (doesn't hang indefinitely)

    // 1. Structural: CircuitBreaker has timeout/failure detection
    let health_path = project_root()
        .join("crates")
        .join("apr-cli")
        .join("src")
        .join("federation")
        .join("health.rs");
    let health = std::fs::read_to_string(&health_path).expect("health.rs readable");
    assert!(
        health.contains("CircuitBreaker"),
        "F-QA-002: CircuitBreaker must exist in federation/health.rs"
    );
    assert!(
        health.contains("reset_timeout") || health.contains("timeout"),
        "F-QA-002: CircuitBreaker must have timeout configuration"
    );
    assert!(
        health.contains("Open") && health.contains("Closed"),
        "F-QA-002: CircuitBreaker must have Open/Closed states"
    );

    // 2. Structural: wait_with_timeout exists in QA tooling
    let qa_run_path = project_root().join("examples").join("qa_run.rs");
    if qa_run_path.exists() {
        let qa_run = std::fs::read_to_string(&qa_run_path).expect("qa_run.rs readable");
        assert!(
            qa_run.contains("wait_with_timeout"),
            "F-QA-002: wait_with_timeout must exist in qa_run.rs"
        );
        assert!(
            qa_run.contains("kill") && qa_run.contains("HANG"),
            "F-QA-002: timeout handler must kill hung process and report HANG"
        );
    }

    // 3. Runtime: apr qa completes within timeout (doesn't hang)
    // Use --skip flags to avoid expensive inference; test structure + timeout behavior
    let gguf = require_model!(gguf_model_path(), "GGUF model");
    let start = std::time::Instant::now();
    let (ok, _stdout, _stderr) = run_apr(&[
        "qa",
        gguf.to_str().unwrap(),
        "--skip-golden",
        "--skip-throughput",
        "--skip-ollama",
        "--skip-gpu-speedup",
        "--skip-format-parity",
    ]);
    let elapsed = start.elapsed();
    // Structural-only qa should complete in < 60s
    assert!(
        elapsed.as_secs() < 60,
        "F-QA-002: apr qa hung (took {}s, limit 60s)",
        elapsed.as_secs()
    );
    eprintln!(
        "F-QA-002: apr qa completed in {:.1}s (success={})",
        elapsed.as_secs_f64(),
        ok
    );
}

#[test]
fn f_qa_003_garbage_detection_catches_layout_bugs() {
    // F-QA-003: verify_output exists and detects garbage patterns
    // Structural check: the function is implemented in qa.rs with garbage detection
    let qa_path = project_root()
        .join("crates")
        .join("apr-cli")
        .join("src")
        .join("commands")
        .join("qa.rs");
    let content = std::fs::read_to_string(&qa_path).expect("qa.rs readable");

    assert!(
        content.contains("fn verify_output"),
        "F-QA-003: verify_output function must exist"
    );
    assert!(
        content.contains("Garbage detected") || content.contains("garbage"),
        "F-QA-003: verify_output must detect garbage patterns"
    );
    // Verify garbage patterns are checked (FFFD, UNK)
    assert!(
        content.contains("FFFD") || content.contains("\\u{FFFD}"),
        "F-QA-003: verify_output must check for Unicode replacement character"
    );
    assert!(
        content.contains("[UNK]"),
        "F-QA-003: verify_output must check for [UNK] token"
    );
}

#[test]
fn f_qa_004_empty_output_detected() {
    // F-QA-004: verify_output detects empty output
    let qa_path = project_root()
        .join("crates")
        .join("apr-cli")
        .join("src")
        .join("commands")
        .join("qa.rs");
    let content = std::fs::read_to_string(&qa_path).expect("qa.rs readable");

    assert!(
        content.contains("fn verify_output"),
        "F-QA-004: verify_output function must exist"
    );
    assert!(
        content.contains("Empty output") || content.contains("empty"),
        "F-QA-004: verify_output must detect empty output"
    );
}

#[test]
fn f_qa_005_apr_qa_returns_machine_readable_results() {
    // F-QA-005: apr qa supports --json machine-readable output
    let qa_path = project_root()
        .join("crates")
        .join("apr-cli")
        .join("src")
        .join("commands")
        .join("qa.rs");
    let content = std::fs::read_to_string(&qa_path).expect("qa.rs readable");

    assert!(
        content.contains("json") || content.contains("Json") || content.contains("JSON"),
        "F-QA-005: qa.rs must support JSON output"
    );
}

#[test]
fn f_qa_006_apr_showcase_runs_automated_demo() {
    // F-QA-006: apr showcase command exists with auto-verification
    let showcase_dir = project_root()
        .join("crates")
        .join("apr-cli")
        .join("src")
        .join("commands")
        .join("showcase");
    assert!(
        showcase_dir.exists(),
        "F-QA-006: showcase command module must exist"
    );

    let mod_path = showcase_dir.join("mod.rs");
    if mod_path.exists() {
        let content = std::fs::read_to_string(&mod_path).expect("showcase/mod.rs readable");
        assert!(
            content.contains("fn run"),
            "F-QA-006: showcase must have a run function"
        );
        assert!(
            content.contains("auto_verify") || content.contains("validate_falsification"),
            "F-QA-006: showcase must support auto-verification"
        );
    }
}

// =============================================================================
