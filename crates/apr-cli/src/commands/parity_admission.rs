//! REG-15 model admission (#2971, PMAT-1065; contract
//! `contracts/apr-gpu-cpu-parity-v1.yaml`, equation `model_admission_reg15`).
//!
//! `#2971`'s second half: when the CUDA load-time parity gate
//! (`crates/aprender-serve/src/gguf/cuda/mod_parity_gate.rs`) fails, `apr-cli`
//! caught the load error and SILENTLY ran on the CPU — even when the user
//! forced the GPU with `--gpu` / `--backend cuda`. This module is the one
//! place that decision is made, so it cannot drift per call site again:
//!
//! * a forced backend never downgrades — it is refused, loudly, with an exit
//!   code read from [`crate::error::CliError`];
//! * an unforced selection is printed with its reason (never silent);
//! * `SKIP_PARITY_GATE` is a printed override that marks every receipt of the
//!   run INVALID-CORRECTNESS, never a quiet bypass.

use crate::error::CliError;

/// The load-time parity gate's verdict for one model, in whatever backend
/// last measured it.
///
/// `status` is one of `"PASS"`, `"FAIL"`, or `"skipped"` (no gate ran — e.g.
/// the model loaded on CPU directly). Kept as a plain `String` rather than an
/// enum so it serialises directly into the `GET /v1/effective-config.parity`
/// wire shape (`{status, cosine, positions, threshold, basis}`) with no extra
/// mapping layer.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ParityVerdict {
    pub status: String,
    pub cosine: f32,
    pub positions: usize,
    pub threshold: f32,
    pub basis: String,
}

impl ParityVerdict {
    #[must_use]
    pub fn pass(cosine: f32, positions: usize, threshold: f32, basis: impl Into<String>) -> Self {
        Self {
            status: "PASS".to_string(),
            cosine,
            positions,
            threshold,
            basis: basis.into(),
        }
    }

    #[must_use]
    pub fn fail(cosine: f32, positions: usize, threshold: f32, basis: impl Into<String>) -> Self {
        Self {
            status: "FAIL".to_string(),
            cosine,
            positions,
            threshold,
            basis: basis.into(),
        }
    }

    /// No gate ran (e.g. the model loaded on CPU directly, or `SKIP_PARITY_GATE`
    /// bypassed it). `effective-config` must never simply omit the `parity`
    /// key — this is the honest "nothing was measured" value for it.
    #[must_use]
    pub fn skipped() -> Self {
        Self {
            status: "skipped".to_string(),
            cosine: 0.0,
            positions: 0,
            threshold: 0.0,
            basis: "no gate ran".to_string(),
        }
    }

    fn failed(&self) -> bool {
        self.status.eq_ignore_ascii_case("FAIL")
    }
}

/// What REG-15 admission decided for one load attempt.
#[derive(Debug, Clone, PartialEq)]
pub enum Admission {
    /// A forced backend refused to downgrade. `code` is read from
    /// [`CliError::ParityFailed`]'s `exit_code_value()`, never typed as a
    /// literal by a caller.
    Refuse { code: i32, reason: String },
    /// An unforced request selected CPU after a parity failure. `line` is the
    /// exact string to print (stderr) before falling back.
    SelectCpu { line: String },
    /// The GPU is admitted. `line` is the exact string to print (stderr).
    Proceed { line: String },
}

/// REG-15: decide whether to admit `verdict` onto the GPU.
///
/// `forced` is `true` when the user explicitly requested the GPU (`--gpu` /
/// `--backend cuda`), matching `crate::accel::asked_flag`'s callers. A forced
/// request whose parity gate FAILED is refused — never silently run on CPU
/// (the second half of #2971). An unforced request that fails is printed and
/// downgraded. A pass is admitted, printed either way.
#[must_use]
pub fn admit(forced: bool, verdict: &ParityVerdict) -> Admission {
    if verdict.failed() {
        let reason = format!(
            "parity FAILED cosine={:.4} position={} threshold={:.2}",
            verdict.cosine, verdict.positions, verdict.threshold
        );
        if forced {
            let code = i32::from(CliError::ParityFailed(reason.clone()).exit_code_value());
            Admission::Refuse { code, reason }
        } else {
            Admission::SelectCpu {
                line: format!("selected: cpu (reason: {reason})"),
            }
        }
    } else {
        Admission::Proceed {
            line: format!(
                "selected: cuda (parity: PASS cosine={:.4} positions={})",
                verdict.cosine, verdict.positions
            ),
        }
    }
}

/// The printed override banner when `SKIP_PARITY_GATE` is set — `None` when
/// it is not.
///
/// Every receipt taken while this is set is INVALID-CORRECTNESS: the gate
/// that would have caught a GPU computing a different function than the CPU
/// did not run. This must be printed once, on stderr, everywhere the env var
/// is read (`crates/aprender-serve/src/gguf/cuda/mod.rs`, and the two apr-cli
/// call sites that used to set it silently: `comparison.rs`, `diff_benchmark_report.rs`).
#[must_use]
pub fn override_line() -> Option<String> {
    let v = std::env::var("SKIP_PARITY_GATE").ok()?;
    if v.is_empty() {
        return None;
    }
    Some(format!(
        "override: SKIP_PARITY_GATE={v} — every receipt of this run is INVALID-CORRECTNESS"
    ))
}

/// Extract `(cosine, threshold)` out of the load-time parity gate's failure
/// message.
///
/// The gate (`mod_parity_gate.rs`) is the only channel apr-cli has for this
/// data at load time — it returns a `RealizarError` whose `Display` embeds
/// the line `Cosine similarity: <cosine> (required: ≥<threshold>)`. Kept as
/// ONE parser so the format is diagnosed in one place if it drifts.
#[must_use]
pub fn parse_gate_error(msg: &str) -> Option<(f32, f32)> {
    if !msg.contains("PARITY-GATE") {
        return None;
    }
    let after_label = msg.split("Cosine similarity:").nth(1)?;
    let cosine_str = after_label.split('(').next()?.trim();
    let cosine: f32 = cosine_str.parse().ok()?;

    let after_geq = after_label.split('\u{2265}').nth(1)?;
    let threshold_str: String = after_geq
        .chars()
        .take_while(|c| c.is_ascii_digit() || *c == '.')
        .collect();
    let threshold: f32 = threshold_str.parse().ok()?;

    Some((cosine, threshold))
}

/// Is this load-error message a parity-gate failure at all (vs. some other
/// CUDA init error, e.g. OOM)? Callers must not run REG-15 admission over an
/// unrelated failure.
#[must_use]
pub fn is_parity_gate_error(msg: &str) -> bool {
    msg.contains("PARITY-GATE FAILED")
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_GATE_ERROR: &str =
        "PARITY-GATE FAILED: GPU computes a DIFFERENT function than CPU.\n\
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
    fn forced_fail_refuses_with_the_code_from_error_rs() {
        let v = ParityVerdict::fail(0.9508, 1, 0.98, "PARITY_GATE_COSINE_MIN [U]");
        let expected_code = i32::from(CliError::ParityFailed("x".to_string()).exit_code_value());
        match admit(true, &v) {
            Admission::Refuse { code, reason } => {
                assert_eq!(
                    code, expected_code,
                    "must read the code from error.rs, not a literal"
                );
                assert!(reason.contains("parity FAILED"));
                assert!(reason.contains("cosine=0.9508"));
                assert!(reason.contains("threshold=0.98"));
            }
            other => panic!("forced + FAIL must Refuse, got {other:?}"),
        }
    }

    #[test]
    fn forced_never_downgrades_even_though_the_verdict_failed() {
        let v = ParityVerdict::fail(0.10, 1, 0.98, "test");
        assert!(
            matches!(admit(true, &v), Admission::Refuse { .. }),
            "a forced backend must never silently run on CPU (#2971)"
        );
    }

    #[test]
    fn unforced_fail_prints_the_exact_selected_cpu_line() {
        let v = ParityVerdict::fail(0.9508, 1, 0.98, "test");
        match admit(false, &v) {
            Admission::SelectCpu { line } => {
                assert_eq!(
                    line,
                    "selected: cpu (reason: parity FAILED cosine=0.9508 position=1 threshold=0.98)"
                );
            }
            other => panic!("unforced + FAIL must SelectCpu, got {other:?}"),
        }
    }

    #[test]
    fn pass_proceeds_with_the_exact_line_forced_or_not() {
        let v = ParityVerdict::pass(0.9990, 64, 0.98, "test");
        for forced in [true, false] {
            match admit(forced, &v) {
                Admission::Proceed { line } => {
                    assert_eq!(
                        line,
                        "selected: cuda (parity: PASS cosine=0.9990 positions=64)"
                    );
                }
                other => panic!("PASS must Proceed (forced={forced}), got {other:?}"),
            }
        }
    }

    #[test]
    fn override_line_is_none_when_unset() {
        // SERIAL_ENV_LOCK: env vars are process-global; guard against another
        // test in this binary racing on SKIP_PARITY_GATE.
        let _guard = env_lock();
        std::env::remove_var("SKIP_PARITY_GATE");
        assert_eq!(override_line(), None);
    }

    #[test]
    fn override_line_is_printed_and_admission_still_never_downgrades() {
        let _guard = env_lock();
        std::env::set_var("SKIP_PARITY_GATE", "1");
        let line = override_line().expect("override must be Some when the var is set");
        assert!(line.contains("SKIP_PARITY_GATE=1"));
        assert!(line.contains("INVALID-CORRECTNESS"));

        // The override banner is orthogonal to admission: a forced request
        // over a verdict that DID measure a failure must still refuse.
        let v = ParityVerdict::fail(0.10, 1, 0.98, "test");
        assert!(matches!(admit(true, &v), Admission::Refuse { .. }));

        std::env::remove_var("SKIP_PARITY_GATE");
    }

    #[test]
    fn parse_gate_error_extracts_cosine_and_threshold() {
        let (cosine, threshold) =
            parse_gate_error(SAMPLE_GATE_ERROR).expect("must parse the gate's own message format");
        assert!((cosine - 0.9508).abs() < 1e-4, "cosine={cosine}");
        assert!((threshold - 0.98).abs() < 1e-4, "threshold={threshold}");
    }

    #[test]
    fn parse_gate_error_is_none_for_an_unrelated_message() {
        assert_eq!(parse_gate_error("OOM: out of VRAM"), None);
    }

    #[test]
    fn is_parity_gate_error_recognises_the_marker() {
        assert!(is_parity_gate_error(SAMPLE_GATE_ERROR));
        assert!(!is_parity_gate_error("some other cuda init failure"));
    }

    #[test]
    fn parity_verdict_json_shape_has_the_five_keys() {
        let v = ParityVerdict::pass(0.99, 64, 0.98, "evidence/parity/thresholds.yaml");
        let json = serde_json::to_value(&v).expect("serialise");
        let obj = json.as_object().expect("object");
        for key in ["status", "cosine", "positions", "threshold", "basis"] {
            assert!(obj.contains_key(key), "missing key {key}: {json}");
        }
        assert_eq!(obj.len(), 5, "exactly these five keys, no more: {json}");
    }

    #[test]
    fn skipped_verdict_never_omits_the_parity_block() {
        let v = ParityVerdict::skipped();
        assert_eq!(v.status, "skipped");
        assert_eq!(v.basis, "no gate ran");
        assert!(!matches!(admit(true, &v), Admission::Refuse { .. }));
    }

    /// Serialises env-mutating tests in this module against a shared lock so
    /// they cannot race on the process-global `SKIP_PARITY_GATE` var.
    fn env_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        LOCK.lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}
