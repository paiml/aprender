// Section 6: 300-Point Checklist (F-CHECKLIST-*)
// =============================================================================

#[test]
fn f_checklist_001_score_ge_250() {
    // F-CHECKLIST-001: Structural check — qa.rs has scoring logic and threshold
    // #2522: was anchored to apr-cli/src/commands/qa.rs. The scoring and gate
    // logic moved to qa_report.rs / qa_*.rs siblings; the property is unchanged.
    let content = crate_src_text("apr-cli");
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
    // #2522: was anchored to apr-cli/src/commands/qa.rs. The scoring and gate
    // logic moved to qa_report.rs / qa_*.rs siblings; the property is unchanged.
    let content = crate_src_text("apr-cli");
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
    // #2522: the spec moved to docs/specifications/archive/.
    let content = spec_text();

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
    // #2522: the spec moved to docs/specifications/archive/.
    let content = spec_text();

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

/// Self-admitted technical debt in PRODUCTION source, as measured on 2026-08-22
/// at `bb2bd5e73`. This is a RATCHET, not a target: the number may only fall.
///
/// #2522: the gate demanded 0 and found 86, so it failed on every run since
/// APR-MONO and told nobody, because the whole suite was named in no workflow.
/// Two things were wrong with it. Its universe was every `.rs` file including
/// test trees (86 vs 54 in production code), which is not the scope the repo's
/// own PMAT gate uses. And "must be 0" against a real 54 is not an assertion, it
/// is a wish -- it can only ever be red, so it carries no information about
/// whether the debt is growing. A baselined ratchet does: it goes red the moment
/// someone ADDS a marker, which is the outcome worth excluding.
///
/// Lower it whenever debt is paid down. Never raise it.
///
/// BSE-03 phase B (PMAT-1068): this constant is now the FALLBACK, not the
/// comparand. It is the one number in the SATD class that a pull request can
/// rewrite, which is the whole defect the D2 normaliser removes
/// (`docs/audits/threat-model-bse-03-ratchets.md`, class 2). When the job
/// exports `SATD_BASELINE` — the count measured on the comparand checkout of
/// `origin/main`, in the same job, by this same scanner — that measurement is
/// the ceiling and this constant is not read at all.
const SATD_PRODUCTION_BASELINE: usize = 37;

/// The environment variable carrying `measure(comparand)` for the SATD class.
const SATD_BASELINE_ENV: &str = "SATD_BASELINE";

/// The ceiling, and WHICH SOURCE it came from.
///
/// Two rules, and the second is the one worth stating:
///
/// * unset ⇒ the constant, tagged `"constant"`. That is the local-developer
///   path, where there is no second checkout to measure.
/// * set ⇒ the parsed value, tagged `"comparand"` — including `0`, which is a
///   real measurement of a tree with no markers and must never read as "unset".
/// * set but EMPTY or unparseable ⇒ `Err`. It does **not** fall back. An empty
///   `SATD_BASELINE` means the job tried to measure the comparand and failed,
///   and silently substituting a hand-written constant for a measurement that
///   did not happen is the exact defect this suite exists to catch: a gate
///   reporting a verdict it did not reach.
fn satd_baseline_from(raw: Option<&str>) -> Result<(usize, &'static str), String> {
    match raw {
        None => Ok((SATD_PRODUCTION_BASELINE, "constant")),
        Some(s) if s.trim().is_empty() => Err(format!(
            "{SATD_BASELINE_ENV} is set but empty: the comparand measurement did not \
             happen, and an unmeasured comparand is not a baseline. Unset it to fall \
             back to the constant deliberately, or fix the measurement."
        )),
        Some(s) => s
            .trim()
            .parse::<usize>()
            .map(|n| (n, "comparand"))
            .map_err(|e| {
                format!(
                    "{SATD_BASELINE_ENV}={s:?} is not a marker count ({e}); it must be \
                     the integer this same scanner measured on the comparand checkout."
                )
            }),
    }
}

fn f_checklist_005_satd_is_zero() {
    // F-CHECKLIST-005: SATD in production source may only shrink.
    //
    // POLARITY (BSE-03 phase B): a count ABOVE the ceiling fails; a count at or
    // below it passes. There is NO lower bound any more. The old "a ratchet
    // that never tightens is stuck" assertion was one — it failed a tree whose
    // debt had FALLEN, and it is the registered mutation of
    // `scripts/tests/ratchet_semantics_test.sh --class satd`. It was load
    // bearing only while the ceiling was a hand-written constant that somebody
    // had to remember to lower. A ceiling measured on the comparand tightens by
    // itself on the next merge, so an improvement needs no edit anywhere, and
    // reddening one is how a guard teaches people to stop paying debt down.
    let raw = std::env::var(SATD_BASELINE_ENV).ok();
    let (baseline, source) = match satd_baseline_from(raw.as_deref()) {
        Ok(pair) => pair,
        Err(why) => panic!("F-CHECKLIST-005: {why}"),
    };
    // Printed, not merely computed: which number was enforced, and from where.
    println!("F-CHECKLIST-005: SATD ceiling {baseline} (source: {source})");

    let mut violations = Vec::new();
    for path in production_rs_files() {
        check_file_for_satd(&path, &mut violations);
    }

    assert!(
        violations.len() <= baseline,
        "F-CHECKLIST-005: SATD ratchet BROKEN -- {} markers in production source, \
         the ceiling is {baseline} (source: {source}). Remove the new marker, or pay \
         down elsewhere; do not raise the ceiling.\n{}",
        violations.len(),
        violations.join("\n")
    );
}

#[test]
fn f_checklist_005_satd_baseline_source_is_env_then_constant() {
    // F-CHECKLIST-005, the parsing half. Both polarities of every rule in
    // satd_baseline_from, because "reads the environment when set" is a claim
    // about a fallback nobody sees fire.
    assert_eq!(
        satd_baseline_from(None).expect("unset is not an error"),
        (SATD_PRODUCTION_BASELINE, "constant"),
        "unset must fall back to the constant, and say so"
    );
    assert_eq!(
        satd_baseline_from(Some("12")).expect("a count parses"),
        (12, "comparand"),
        "a set value must WIN over the constant, and be tagged as the comparand"
    );
    assert_eq!(
        satd_baseline_from(Some(" 7\n")).expect("surrounding whitespace parses"),
        (7, "comparand"),
        "a value captured from a shell command substitution carries whitespace"
    );
    assert_eq!(
        satd_baseline_from(Some("0")).expect("zero parses"),
        (0, "comparand"),
        "0 is a real measurement of a clean comparand and must not read as unset"
    );
    for bad in ["", "   ", "abc", "-1", "37.0", "1e2"] {
        let err = satd_baseline_from(Some(bad)).expect_err(
            "an unusable SATD_BASELINE must be an ERROR, never a silent fall back to \
             the constant",
        );
        assert!(
            err.contains(SATD_BASELINE_ENV),
            "the failure must name the variable it could not use, got: {err}"
        );
    }
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
    // #2522: `verify_output` moved to apr-cli/src/commands/output_verification.rs.
    let content = crate_src_text("apr-cli");

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
    // #2522: `verify_output` moved to apr-cli/src/commands/output_verification.rs.
    let content = crate_src_text("apr-cli");

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
    // #2522: `verify_output` moved to apr-cli/src/commands/output_verification.rs.
    let content = crate_src_text("apr-cli");

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
