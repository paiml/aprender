//! REG-15 model admission (#2971, PMAT-1065; contract
//! `contracts/apr-gpu-cpu-parity-v1.yaml`, equation `model_admission_reg15`).
//!
//! Hermetic, CPU-only: no CUDA feature, no model file, no GPU. Exercises
//! `apr_cli::parity_admission` (re-exported from `crate::commands::parity_admission`)
//! directly — the same decision surface `apr chat --gpu` / `apr run --backend
//! cuda` reach at load time.

use apr_cli::error::parity_admission::{
    admit, override_line, parse_gate_error, Admission, ParityVerdict,
};
use apr_cli::error::CliError;

#[test]
fn forced_backend_over_a_failed_parity_gate_refuses_with_the_code_from_error_rs() {
    let verdict = ParityVerdict::fail(0.9508, 1, 0.98, "PARITY_GATE_COSINE_MIN [U]");
    let expected_code = i32::from(CliError::ParityFailed("x".to_string()).exit_code_value());

    match admit(true, &verdict) {
        Admission::Refuse { code, reason } => {
            assert_eq!(
                code, expected_code,
                "the refusal code must be READ from error.rs, never typed as a literal"
            );
            assert!(reason.contains("parity FAILED"), "reason={reason}");
            assert!(reason.contains("cosine=0.9508"), "reason={reason}");
            assert!(reason.contains("position=1"), "reason={reason}");
            assert!(reason.contains("threshold=0.98"), "reason={reason}");
        }
        other => panic!("forced + FAIL must Refuse, never silently downgrade: {other:?}"),
    }
}

#[test]
fn unforced_backend_over_a_failed_parity_gate_prints_the_exact_selected_cpu_line() {
    let verdict = ParityVerdict::fail(0.9508, 1, 0.98, "test");
    match admit(false, &verdict) {
        Admission::SelectCpu { line } => {
            assert_eq!(
                line,
                "selected: cpu (reason: parity FAILED cosine=0.9508 position=1 threshold=0.98)"
            );
        }
        other => panic!("unforced + FAIL must SelectCpu with the printed reason: {other:?}"),
    }
}

#[test]
fn a_passing_parity_gate_proceeds_with_the_exact_line_whether_forced_or_not() {
    let verdict = ParityVerdict::pass(0.9990, 64, 0.98, "test");
    for forced in [true, false] {
        match admit(forced, &verdict) {
            Admission::Proceed { line } => {
                assert_eq!(
                    line,
                    "selected: cuda (parity: PASS cosine=0.9990 positions=64)"
                );
            }
            other => panic!("PASS must Proceed regardless of forced={forced}: {other:?}"),
        }
    }
}

#[test]
fn override_line_reflects_skip_parity_gate_and_never_authorizes_a_downgrade() {
    // Env vars are process-global; this binary runs tests on multiple threads
    // by default, so guard the mutation with a lock shared across this file's
    // tests that touch SKIP_PARITY_GATE.
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _guard = LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);

    std::env::remove_var("SKIP_PARITY_GATE");
    assert_eq!(
        override_line(),
        None,
        "no override banner when the var is unset"
    );

    std::env::set_var("SKIP_PARITY_GATE", "1");
    let line = override_line().expect("override banner must be Some once the var is set");
    assert!(line.contains("SKIP_PARITY_GATE=1"), "line={line}");
    assert!(line.contains("INVALID-CORRECTNESS"), "line={line}");

    // The override is a print-only concern (mod.rs decides whether the gate
    // runs at all). admit() itself must still never downgrade a forced
    // request over a verdict that measured a failure.
    let verdict = ParityVerdict::fail(0.10, 1, 0.98, "test");
    assert!(
        matches!(admit(true, &verdict), Admission::Refuse { .. }),
        "SKIP_PARITY_GATE being set must never make admit() downgrade a forced request"
    );

    std::env::remove_var("SKIP_PARITY_GATE");
}

/// The exact message format `mod_parity_gate.rs` embeds in its `RealizarError`
/// on failure — apr-cli's only channel to cosine/threshold at load time.
const SAMPLE_GATE_ERROR: &str = "PARITY-GATE FAILED: GPU computes a DIFFERENT function than CPU.\n\
     \n\
     Cosine similarity: 0.950800 (required: \u{2265}0.98)\n\
     CPU argmax: 12 | GPU argmax: 884\n\
     Max absolute logit difference: 3.2100\n\
     \n\
     This model's dimensions (hidden=2048, heads=16, kv_heads=4) cause\n\
     GPU forward pass to diverge from CPU. The GPU CANNOT serve this model.\n\
     \n\
     Run `apr parity <model>` for full SPC diagnosis.\n\
     Set SKIP_PARITY_GATE=1 to bypass (for debugging only).";

#[test]
fn parse_gate_error_extracts_cosine_and_threshold_from_the_real_message_format() {
    let (cosine, threshold) =
        parse_gate_error(SAMPLE_GATE_ERROR).expect("must parse mod_parity_gate.rs's own format");
    assert!((cosine - 0.9508).abs() < 1e-4, "cosine={cosine}");
    assert!((threshold - 0.98).abs() < 1e-4, "threshold={threshold}");
}

#[test]
fn parse_gate_error_is_none_for_an_unrelated_cuda_failure() {
    assert_eq!(parse_gate_error("CUDA error: out of memory"), None);
}

#[test]
fn parity_verdict_json_shape_serialises_with_exactly_the_five_keys() {
    // PP-066 row L0-1a P3: GET /v1/effective-config.parity must carry
    // {status, cosine, positions, threshold, basis} and never be absent.
    let verdict = ParityVerdict::pass(0.99, 64, 0.98, "evidence/parity/thresholds.yaml");
    let json = serde_json::to_value(&verdict).expect("serialise");
    let obj = json.as_object().expect("object");
    for key in ["status", "cosine", "positions", "threshold", "basis"] {
        assert!(obj.contains_key(key), "missing key {key}: {json}");
    }
    assert_eq!(obj.len(), 5, "exactly these five keys, no more: {json}");

    let skipped = ParityVerdict::skipped();
    let skipped_json = serde_json::to_value(&skipped).expect("serialise");
    assert_eq!(skipped_json["status"], "skipped");
    assert_eq!(skipped_json["basis"], "no gate ran");
}
