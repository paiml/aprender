//! `apr attn-parity-lint` — CRUX-L-02 flash-attn2 parity gate.
//!
//! Reads a captured `apr kernel parity --impl flash2 --ref naive --json`
//! body and/or `apr run --attn flash2 --json` body and/or a head_dim
//! error JSON, then dispatches the pure classifiers in
//! `attn_parity_classifier`. Exits non-zero on any failure.
//!
//! Spec: `contracts/crux-L-02-v1.yaml`. CRUX-SHIP-001 g2/g3 surface.

use std::path::{Path, PathBuf};

use serde_json::Value;

use super::attn_parity_classifier::{
    classify_head_dim_error, classify_parity_numerics, classify_provenance,
    AttnHeadDimErrorOutcome, AttnParityNumericsOutcome, AttnProvenanceOutcome,
    L02_DEFAULT_MAX_ABS_DIFF, L02_DEFAULT_MIN_COSINE_SIM,
};
use super::threshold_arg;
use crate::error::{CliError, Result};

pub(crate) fn run(
    parity_file: Option<&Path>,
    provenance_file: Option<&Path>,
    head_dim_error_file: Option<&Path>,
    tol_abs: f64,
    tol_cos: f64,
    json: bool,
) -> Result<()> {
    // Fail closed before any gate runs: `mad > tol_abs` / `cos < tol_cos` are
    // both false against a NaN threshold, so the gate could never fire.
    threshold_arg::guard("--tol-abs", tol_abs, threshold_arg::TOLERANCE)?;
    threshold_arg::guard("--tol-cos", tol_cos, threshold_arg::COSINE)?;
    if parity_file.is_none() && provenance_file.is_none() && head_dim_error_file.is_none() {
        return Err(CliError::ValidationFailed(
            "apr attn-parity-lint: at least one of --parity-file, --provenance-file, or --head-dim-error-file is required"
                .to_string(),
        ));
    }

    let parity = match parity_file {
        Some(p) => Some(classify_parity_numerics(&load_json(p)?, tol_abs, tol_cos)),
        None => None,
    };
    let provenance = match provenance_file {
        Some(p) => Some(classify_provenance(&load_json(p)?)),
        None => None,
    };
    let head_dim = match head_dim_error_file {
        Some(p) => Some(classify_head_dim_error(&load_json(p)?)),
        None => None,
    };

    print_report(
        parity_file,
        provenance_file,
        head_dim_error_file,
        parity.as_ref(),
        provenance.as_ref(),
        head_dim.as_ref(),
        json,
    );

    verdict("parity-numerics", parity.as_ref(), |o| {
        matches!(o, AttnParityNumericsOutcome::Ok { .. })
    })?;
    verdict("provenance", provenance.as_ref(), |o| {
        matches!(
            o,
            AttnProvenanceOutcome::OkFlash2 { .. } | AttnProvenanceOutcome::OkFallback { .. }
        )
    })?;
    verdict("head-dim-error", head_dim.as_ref(), |o| {
        matches!(o, AttnHeadDimErrorOutcome::Ok { .. })
    })
}

/// Turn one classifier outcome into this command's verdict.
///
/// A gate that was not asked for (`None`) says nothing; a gate whose outcome is
/// outside the accepting set fails the command with `ValidationFailed` (exit 5).
/// Written once so the three gates cannot drift apart — and so `run` stays under
/// the repo's cognitive-complexity ceiling.
fn verdict<T: std::fmt::Debug>(
    gate: &str,
    outcome: Option<&T>,
    accepts: impl Fn(&T) -> bool,
) -> Result<()> {
    match outcome {
        Some(o) if !accepts(o) => Err(CliError::ValidationFailed(format!(
            "attn-parity-lint {gate} gate rejected body: {o:?}"
        ))),
        _ => Ok(()),
    }
}

fn load_json(path: &Path) -> Result<Value> {
    super::lint_input::read_json_observation("apr attn-parity-lint", path)
}

#[allow(clippy::too_many_arguments)]
fn print_report(
    parity_file: Option<&Path>,
    provenance_file: Option<&Path>,
    head_dim_error_file: Option<&Path>,
    parity: Option<&AttnParityNumericsOutcome>,
    provenance: Option<&AttnProvenanceOutcome>,
    head_dim: Option<&AttnHeadDimErrorOutcome>,
    json: bool,
) {
    if json {
        let obj = serde_json::json!({
            "parity_file": parity_file.map(|p| p.display().to_string()),
            "provenance_file": provenance_file.map(|p| p.display().to_string()),
            "head_dim_error_file": head_dim_error_file.map(|p| p.display().to_string()),
            "parity_numerics": parity.map(|o| format!("{o:?}")),
            "provenance": provenance.map(|o| format!("{o:?}")),
            "head_dim_error": head_dim.map(|o| format!("{o:?}")),
        });
        println!("{}", serde_json::to_string_pretty(&obj).unwrap_or_default());
        return;
    }
    println!("attn-parity-lint report");
    if let Some(p) = parity_file {
        println!("  parity_file        : {}", p.display());
    }
    if let Some(p) = provenance_file {
        println!("  provenance_file    : {}", p.display());
    }
    if let Some(p) = head_dim_error_file {
        println!("  head_dim_error_file: {}", p.display());
    }
    if let Some(o) = parity {
        println!("  parity_numerics    : {o:?}");
    }
    if let Some(o) = provenance {
        println!("  provenance         : {o:?}");
    }
    if let Some(o) = head_dim {
        println!("  head_dim_error     : {o:?}");
    }
}

pub const ATTN_PARITY_DEFAULT_MAX_ABS_DIFF: f64 = L02_DEFAULT_MAX_ABS_DIFF;
pub const ATTN_PARITY_DEFAULT_MIN_COSINE_SIM: f64 = L02_DEFAULT_MIN_COSINE_SIM;

#[cfg(test)]
mod cov_tests {
    use super::*;
    #[test]
    fn no_inputs_is_validation_failed() {
        let err = run(
            None,
            None,
            None,
            ATTN_PARITY_DEFAULT_MAX_ABS_DIFF,
            ATTN_PARITY_DEFAULT_MIN_COSINE_SIM,
            false,
        )
        .unwrap_err();
        assert!(matches!(err, CliError::ValidationFailed(_)));
    }
    #[test]
    fn missing_parity_file_is_file_not_found() {
        let err = run(
            Some(Path::new("/no/such/parity.json")),
            None,
            None,
            ATTN_PARITY_DEFAULT_MAX_ABS_DIFF,
            ATTN_PARITY_DEFAULT_MIN_COSINE_SIM,
            false,
        )
        .unwrap_err();
        assert!(matches!(err, CliError::FileNotFound(_)));
    }
    /// A parity body that fails at the shipped defaults must keep failing: in
    /// 0.63.0 `--tol-abs nan --tol-cos nan` printed
    /// `parity_numerics : Ok { max_abs_diff: 999.0, cosine_sim: -0.5 }` and
    /// exited 0, because `mad > NaN` and `cos < NaN` are both false.
    #[test]
    fn nan_tolerance_cannot_disarm_the_parity_gate() {
        use std::io::Write;
        let mut f = tempfile::NamedTempFile::new().expect("tempfile");
        f.write_all(br#"{"max_abs_diff":999.0,"cosine_sim":-0.5}"#)
            .expect("write");
        f.flush().expect("flush");

        // Control: the defaults reject this body on the numerics gate.
        let err = run(
            Some(f.path()),
            None,
            None,
            ATTN_PARITY_DEFAULT_MAX_ABS_DIFF,
            ATTN_PARITY_DEFAULT_MIN_COSINE_SIM,
            false,
        )
        .unwrap_err();
        match err {
            CliError::ValidationFailed(msg) => {
                assert!(msg.contains("parity-numerics"), "got: {msg}");
            }
            other => panic!("expected the numerics gate to fire, got {other:?}"),
        }

        for (abs, cos, flag) in [
            (f64::NAN, ATTN_PARITY_DEFAULT_MIN_COSINE_SIM, "--tol-abs"),
            (ATTN_PARITY_DEFAULT_MAX_ABS_DIFF, f64::NAN, "--tol-cos"),
            (-1.0, ATTN_PARITY_DEFAULT_MIN_COSINE_SIM, "--tol-abs"),
        ] {
            let err = run(Some(f.path()), None, None, abs, cos, false).unwrap_err();
            match err {
                CliError::ValidationFailed(msg) => {
                    assert!(msg.contains(flag), "expected {flag} named; got: {msg}");
                }
                other => panic!("({abs}, {cos}) must fail closed, got {other:?}"),
            }
        }
    }

    #[test]
    fn missing_provenance_file_is_file_not_found() {
        let err = run(
            None,
            Some(Path::new("/no/such/prov.json")),
            None,
            ATTN_PARITY_DEFAULT_MAX_ABS_DIFF,
            ATTN_PARITY_DEFAULT_MIN_COSINE_SIM,
            true,
        )
        .unwrap_err();
        assert!(matches!(err, CliError::FileNotFound(_)));
    }
}
