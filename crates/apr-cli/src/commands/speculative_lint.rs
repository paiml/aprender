//! `apr speculative-lint` — CRUX-C-09 speculative-decoding observation linter.
//!
//! Reads a JSON observation file that captures a single speculative-decoding
//! run and dispatches all four classifiers (parity, uplift, compat,
//! acceptance_rate). Emits a text or `--json` report.
//!
//! Spec: `contracts/crux-C-09-v1.yaml`. CRUX-SHIP-001 g2/g3 surface.
//!
//! Observation schema (top-level keys; all optional — missing fields skip the
//! corresponding classifier):
//!
//!   {
//!     "base_tokens":                 [u32, ...],       // parity input 1
//!     "spec_tokens":                 [u32, ...],       // parity input 2
//!     "base_tps":                    f64,              // uplift input 1
//!     "spec_tps":                    f64,              // uplift input 2
//!     "draft_tokenizer_sha256":      "hex",            // compat input 1
//!     "target_tokenizer_sha256":     "hex",            // compat input 2
//!     "draft_vocab_size":            u32,              // compat input 3
//!     "target_vocab_size":           u32,              // compat input 4
//!     "speculative": { "acceptance_rate": f64 }        // ar input
//!   }

use std::path::{Path, PathBuf};

use serde_json::Value;

use crate::commands::speculative_decoding_classifier as clf;
use crate::error::{CliError, Result};

pub(crate) fn run(observation_file: &Path, alpha_min: f64, json: bool) -> Result<()> {
    if !observation_file.exists() {
        return Err(CliError::FileNotFound(PathBuf::from(observation_file)));
    }

    let body = std::fs::read_to_string(observation_file)?;
    let obs: Value = serde_json::from_str(&body).map_err(|e| {
        CliError::InvalidFormat(format!(
            "apr speculative-lint: failed to parse JSON from {}: {e}",
            observation_file.display()
        ))
    })?;

    if !alpha_min.is_finite() || alpha_min < 0.0 {
        return Err(CliError::ValidationFailed(format!(
            "--alpha-min must be finite and >= 0 (got {alpha_min})"
        )));
    }

    let parity = classify_parity(&obs);
    let uplift = classify_uplift(&obs, alpha_min);
    let compat = classify_compat(&obs);
    let acceptance = classify_acceptance(&obs);

    let fail_reasons: Vec<String> = [
        parity.as_ref().and_then(parity_fail_reason),
        uplift.as_ref().and_then(uplift_fail_reason),
        compat.as_ref().and_then(compat_fail_reason),
        acceptance.as_ref().and_then(acceptance_fail_reason),
    ]
    .into_iter()
    .flatten()
    .collect();

    print_report(
        observation_file,
        parity.as_ref(),
        uplift.as_ref(),
        compat.as_ref(),
        acceptance.as_ref(),
        alpha_min,
        json,
    );

    if fail_reasons.is_empty() {
        Ok(())
    } else {
        Err(CliError::ValidationFailed(fail_reasons.join("; ")))
    }
}

fn classify_parity(obs: &Value) -> Option<clf::SpecParityOutcome> {
    let base = obs.get("base_tokens")?.as_array()?;
    let spec = obs.get("spec_tokens")?.as_array()?;
    let base: Vec<u32> = base.iter().filter_map(|v| v.as_u64().map(|x| x as u32)).collect();
    let spec: Vec<u32> = spec.iter().filter_map(|v| v.as_u64().map(|x| x as u32)).collect();
    Some(clf::classify_speculative_parity(&base, &spec))
}

fn classify_uplift(obs: &Value, alpha_min: f64) -> Option<clf::ThroughputUpliftOutcome> {
    let base_tps = obs.get("base_tps")?.as_f64()?;
    let spec_tps = obs.get("spec_tps")?.as_f64()?;
    Some(clf::classify_throughput_uplift(base_tps, spec_tps, alpha_min))
}

fn classify_compat(obs: &Value) -> Option<clf::TokenizerCompatOutcome> {
    let draft_sha = obs.get("draft_tokenizer_sha256")?.as_str()?;
    let target_sha = obs.get("target_tokenizer_sha256")?.as_str()?;
    let draft_vocab = obs.get("draft_vocab_size")?.as_u64()? as u32;
    let target_vocab = obs.get("target_vocab_size")?.as_u64()? as u32;
    Some(clf::classify_tokenizer_compatibility(
        draft_sha,
        target_sha,
        draft_vocab,
        target_vocab,
    ))
}

fn classify_acceptance(obs: &Value) -> Option<clf::AcceptanceRateOutcome> {
    obs.get("speculative")?;
    Some(clf::classify_acceptance_rate(obs))
}

fn parity_fail_reason(o: &clf::SpecParityOutcome) -> Option<String> {
    match o {
        clf::SpecParityOutcome::Ok => None,
        clf::SpecParityOutcome::LengthMismatch { base_len, spec_len } => Some(format!(
            "FALSIFY-CRUX-C-09-001 parity: length mismatch base={base_len} spec={spec_len}"
        )),
        clf::SpecParityOutcome::TokenDivergence {
            at_index,
            base_token,
            spec_token,
        } => Some(format!(
            "FALSIFY-CRUX-C-09-001 parity: divergence at idx {at_index}: base={base_token} spec={spec_token}"
        )),
    }
}

fn uplift_fail_reason(o: &clf::ThroughputUpliftOutcome) -> Option<String> {
    match o {
        clf::ThroughputUpliftOutcome::Ok { .. } => None,
        clf::ThroughputUpliftOutcome::BelowThreshold {
            observed_alpha,
            required_alpha,
        } => Some(format!(
            "FALSIFY-CRUX-C-09-002 uplift: alpha={observed_alpha:.3} < required {required_alpha:.3}"
        )),
        clf::ThroughputUpliftOutcome::Regression { base_tps, spec_tps, .. } => Some(format!(
            "FALSIFY-CRUX-C-09-002 uplift: regression (base_tps={base_tps:.3}, spec_tps={spec_tps:.3})"
        )),
        clf::ThroughputUpliftOutcome::InvalidInput { reason } => Some(format!(
            "FALSIFY-CRUX-C-09-002 uplift: invalid input: {reason}"
        )),
    }
}

fn compat_fail_reason(o: &clf::TokenizerCompatOutcome) -> Option<String> {
    match o {
        clf::TokenizerCompatOutcome::Ok => None,
        clf::TokenizerCompatOutcome::TokenizerShaMismatch => Some(
            "FALSIFY-CRUX-C-09-003 compat: draft+target tokenizer sha256 differ".to_string(),
        ),
        clf::TokenizerCompatOutcome::VocabSizeMismatch { draft, target } => Some(format!(
            "FALSIFY-CRUX-C-09-003 compat: vocab_size mismatch draft={draft} target={target}"
        )),
        clf::TokenizerCompatOutcome::MalformedSha { reason } => Some(format!(
            "FALSIFY-CRUX-C-09-003 compat: malformed sha: {reason}"
        )),
        clf::TokenizerCompatOutcome::ZeroVocab { which } => Some(format!(
            "FALSIFY-CRUX-C-09-003 compat: {which} vocab_size is zero"
        )),
    }
}

fn acceptance_fail_reason(o: &clf::AcceptanceRateOutcome) -> Option<String> {
    match o {
        clf::AcceptanceRateOutcome::Ok { .. } => None,
        clf::AcceptanceRateOutcome::NotAnObject => Some(
            "FALSIFY-CRUX-C-09-004 acceptance: top-level is not a JSON object".to_string(),
        ),
        clf::AcceptanceRateOutcome::MissingSpeculative => Some(
            "FALSIFY-CRUX-C-09-004 acceptance: `speculative` key absent".to_string(),
        ),
        clf::AcceptanceRateOutcome::SpeculativeNotAnObject => Some(
            "FALSIFY-CRUX-C-09-004 acceptance: `speculative` is not an object".to_string(),
        ),
        clf::AcceptanceRateOutcome::MissingAcceptanceRate => Some(
            "FALSIFY-CRUX-C-09-004 acceptance: `speculative.acceptance_rate` absent".to_string(),
        ),
        clf::AcceptanceRateOutcome::AcceptanceRateNotNumeric => Some(
            "FALSIFY-CRUX-C-09-004 acceptance: `speculative.acceptance_rate` is not numeric"
                .to_string(),
        ),
        clf::AcceptanceRateOutcome::OutOfRange { value } => Some(format!(
            "FALSIFY-CRUX-C-09-004 acceptance: rate {value} out of [0,1]"
        )),
        clf::AcceptanceRateOutcome::NaN => Some(
            "FALSIFY-CRUX-C-09-004 acceptance: rate is NaN".to_string(),
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn print_report(
    path: &Path,
    parity: Option<&clf::SpecParityOutcome>,
    uplift: Option<&clf::ThroughputUpliftOutcome>,
    compat: Option<&clf::TokenizerCompatOutcome>,
    acceptance: Option<&clf::AcceptanceRateOutcome>,
    alpha_min: f64,
    json: bool,
) {
    if json {
        let v = serde_json::json!({
            "observation_path": path.display().to_string(),
            "alpha_min": alpha_min,
            "parity":     parity.map(|o| format!("{o:?}")),
            "uplift":     uplift.map(|o| format!("{o:?}")),
            "compat":     compat.map(|o| format!("{o:?}")),
            "acceptance": acceptance.map(|o| format!("{o:?}")),
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&v).unwrap_or_else(|_| v.to_string())
        );
    } else {
        println!("speculative-lint report for {}", path.display());
        println!("  alpha_min:  {alpha_min:.3}");
        print_line("  parity:     ", parity.map(|o| format!("{o:?}")));
        print_line("  uplift:     ", uplift.map(|o| format!("{o:?}")));
        print_line("  compat:     ", compat.map(|o| format!("{o:?}")));
        print_line("  acceptance: ", acceptance.map(|o| format!("{o:?}")));
    }
}

fn print_line(prefix: &str, v: Option<String>) {
    match v {
        Some(s) => println!("{prefix}{s}"),
        None => println!("{prefix}(missing fields — classifier skipped)"),
    }
}
