//! CRUX-B-07 imatrix calibration — algorithm-level classifiers.
//!
//! Partial discharge for the `apr quantize --imatrix` contract
//! (`contracts/crux-B-07-v1.yaml`). Three pure classifiers cover:
//!
//! 1. Perplexity-improvement threshold check (FALSIFY-001).
//! 2. `--imatrix <path>` CLI-flag parser (FALSIFY-002).
//! 3. Calibration provenance sha256 computation + match check (FALSIFY-003).
//!
//! Full discharge still requires a real calibration set, a real
//! held-out eval corpus, and a real `apr quantize` harness.
//!
//! Intentionally pure functions — no I/O — so the proofs run in
//! milliseconds under `cargo test -p apr-cli --lib`.

use std::collections::BTreeSet;

/// Minimum perplexity reduction demanded by the contract (0.5%).
pub const MIN_PPL_IMPROVEMENT: f64 = 0.005;

/// Outcome of comparing naive-quant vs calibrated-quant perplexity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ImprovementOutcome {
    /// Calibrated quant reduced PPL by at least the threshold.
    Improved { delta: f64 },
    /// Calibrated quant did not meet the contract threshold.
    Insufficient { delta: f64, threshold: f64 },
}

/// Classify whether imatrix calibration improved perplexity by at
/// least `threshold`. `delta = (ppl_naive - ppl_calib) / ppl_naive`.
///
/// Returns `Insufficient` (not panic) when `ppl_naive <= 0.0` — an
/// upstream pipeline bug, not a contract violation we should mask.
#[must_use]
pub fn classify_imatrix_improvement(
    ppl_naive: f64,
    ppl_calib: f64,
    threshold: f64,
) -> ImprovementOutcome {
    if !ppl_naive.is_finite() || ppl_naive <= 0.0 {
        return ImprovementOutcome::Insufficient {
            delta: f64::NAN,
            threshold,
        };
    }
    let delta = (ppl_naive - ppl_calib) / ppl_naive;
    if delta >= threshold {
        ImprovementOutcome::Improved { delta }
    } else {
        ImprovementOutcome::Insufficient { delta, threshold }
    }
}

/// Whether the calibration set and the eval set share any item.
/// Leakage here would invalidate the PPL improvement claim.
#[must_use]
pub fn calibration_eval_disjoint(
    calib_hashes: &BTreeSet<String>,
    eval_hashes: &BTreeSet<String>,
) -> bool {
    calib_hashes.intersection(eval_hashes).next().is_none()
}

/// Extract the `--imatrix <path>` value from an argv slice, mirroring
/// the competitor `llama-quantize --imatrix imat.dat` shape.
/// Accepts both `--imatrix PATH` and `--imatrix=PATH`.
#[must_use]
pub fn parse_imatrix_flag(argv: &[&str]) -> Option<String> {
    let mut i = 0;
    while i < argv.len() {
        let a = argv[i];
        if a == "--imatrix" {
            return argv.get(i + 1).map(|p| (*p).to_string());
        }
        if let Some(rest) = a.strip_prefix("--imatrix=") {
            return Some(rest.to_string());
        }
        i += 1;
    }
    None
}

/// Compute the canonical provenance hash for a calibration file.
/// Lowercase hex sha256 matches `sha256sum` output byte-for-byte.
#[must_use]
pub fn compute_provenance_sha256(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(bytes);
    let out = h.finalize();
    let mut s = String::with_capacity(64);
    for b in out {
        s.push_str(&format!("{:02x}", b));
    }
    s
}

/// Outcome of validating that an APR sidecar records the expected
/// imatrix provenance.
#[derive(Debug, Clone, PartialEq)]
pub enum ProvenanceOutcome {
    Match,
    Missing,
    Mismatch { recorded: String, expected: String },
}

/// Check the recorded provenance field against the expected sha256.
/// `recorded == None` means the sidecar has no `imatrix_source_sha256`
/// — a direct FALSIFY-003 failure, not a hash mismatch.
#[must_use]
pub fn validate_recorded_provenance(
    recorded: Option<&str>,
    expected: &str,
) -> ProvenanceOutcome {
    match recorded {
        None => ProvenanceOutcome::Missing,
        Some(r) if r.eq_ignore_ascii_case(expected) => ProvenanceOutcome::Match,
        Some(r) => ProvenanceOutcome::Mismatch {
            recorded: r.to_string(),
            expected: expected.to_string(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- FALSIFY-001 (perplexity improvement) ----

    #[test]
    fn improvement_meets_threshold_exact() {
        match classify_imatrix_improvement(100.0, 99.5, MIN_PPL_IMPROVEMENT) {
            ImprovementOutcome::Improved { delta } => assert!((delta - 0.005).abs() < 1e-9),
            o => panic!("expected Improved, got {:?}", o),
        }
    }

    #[test]
    fn improvement_above_threshold_classifies_improved() {
        assert!(matches!(
            classify_imatrix_improvement(100.0, 90.0, MIN_PPL_IMPROVEMENT),
            ImprovementOutcome::Improved { .. }
        ));
    }

    #[test]
    fn improvement_just_below_threshold_is_insufficient() {
        let o = classify_imatrix_improvement(100.0, 99.6, MIN_PPL_IMPROVEMENT);
        assert!(matches!(o, ImprovementOutcome::Insufficient { .. }));
    }

    #[test]
    fn regression_is_insufficient() {
        // Calibration made PPL worse — definitely fails the gate.
        let o = classify_imatrix_improvement(100.0, 110.0, MIN_PPL_IMPROVEMENT);
        assert!(matches!(o, ImprovementOutcome::Insufficient { .. }));
    }

    #[test]
    fn zero_or_negative_baseline_is_insufficient_not_panic() {
        assert!(matches!(
            classify_imatrix_improvement(0.0, 5.0, MIN_PPL_IMPROVEMENT),
            ImprovementOutcome::Insufficient { .. }
        ));
        assert!(matches!(
            classify_imatrix_improvement(-1.0, 5.0, MIN_PPL_IMPROVEMENT),
            ImprovementOutcome::Insufficient { .. }
        ));
    }

    #[test]
    fn improvement_is_deterministic() {
        let a = classify_imatrix_improvement(42.0, 40.0, MIN_PPL_IMPROVEMENT);
        let b = classify_imatrix_improvement(42.0, 40.0, MIN_PPL_IMPROVEMENT);
        assert_eq!(format!("{:?}", a), format!("{:?}", b));
    }

    // ---- leakage check (FALSIFY-001 / INV "disjoint") ----

    #[test]
    fn disjoint_sets_are_disjoint() {
        let calib: BTreeSet<String> = ["a", "b"].iter().map(|s| s.to_string()).collect();
        let eval: BTreeSet<String> = ["c", "d"].iter().map(|s| s.to_string()).collect();
        assert!(calibration_eval_disjoint(&calib, &eval));
    }

    #[test]
    fn overlapping_sets_are_not_disjoint() {
        let calib: BTreeSet<String> = ["a", "b", "c"].iter().map(|s| s.to_string()).collect();
        let eval: BTreeSet<String> = ["c", "d"].iter().map(|s| s.to_string()).collect();
        assert!(!calibration_eval_disjoint(&calib, &eval));
    }

    #[test]
    fn empty_sets_are_trivially_disjoint() {
        let empty: BTreeSet<String> = BTreeSet::new();
        assert!(calibration_eval_disjoint(&empty, &empty));
    }

    // ---- FALSIFY-002 (CLI flag) ----

    #[test]
    fn parse_imatrix_flag_finds_space_form() {
        let argv = &["quantize", "model.apr", "--imatrix", "calib.jsonl"];
        assert_eq!(parse_imatrix_flag(argv), Some("calib.jsonl".to_string()));
    }

    #[test]
    fn parse_imatrix_flag_finds_equals_form() {
        let argv = &["quantize", "model.apr", "--imatrix=calib.jsonl"];
        assert_eq!(parse_imatrix_flag(argv), Some("calib.jsonl".to_string()));
    }

    #[test]
    fn parse_imatrix_flag_missing_returns_none() {
        let argv = &["quantize", "model.apr", "--method", "q4k"];
        assert_eq!(parse_imatrix_flag(argv), None);
    }

    #[test]
    fn parse_imatrix_flag_without_value_returns_none() {
        // `--imatrix` at end of argv with no path following.
        let argv = &["quantize", "model.apr", "--imatrix"];
        assert_eq!(parse_imatrix_flag(argv), None);
    }

    #[test]
    fn parse_imatrix_flag_ignores_similar_names() {
        // `--imatrix-force` is a different flag, must not match.
        let argv = &["quantize", "--imatrix-force", "true"];
        assert_eq!(parse_imatrix_flag(argv), None);
    }

    // ---- FALSIFY-003 (provenance) ----

    #[test]
    fn sha256_of_empty_is_known_constant() {
        // sha256 of empty input (well-known constant)
        assert_eq!(
            compute_provenance_sha256(&[]),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn sha256_is_deterministic() {
        let a = compute_provenance_sha256(b"calibration-bytes");
        let b = compute_provenance_sha256(b"calibration-bytes");
        assert_eq!(a, b);
        assert_eq!(a.len(), 64);
    }

    #[test]
    fn sha256_differs_on_different_input() {
        let a = compute_provenance_sha256(b"a");
        let b = compute_provenance_sha256(b"b");
        assert_ne!(a, b);
    }

    #[test]
    fn provenance_match_ok() {
        let expected = compute_provenance_sha256(b"calib-v1");
        assert_eq!(
            validate_recorded_provenance(Some(&expected), &expected),
            ProvenanceOutcome::Match
        );
    }

    #[test]
    fn provenance_match_is_case_insensitive() {
        let expected = compute_provenance_sha256(b"calib-v1");
        let upper = expected.to_uppercase();
        assert_eq!(
            validate_recorded_provenance(Some(&upper), &expected),
            ProvenanceOutcome::Match
        );
    }

    #[test]
    fn provenance_missing_is_failure() {
        let expected = compute_provenance_sha256(b"calib-v1");
        assert_eq!(
            validate_recorded_provenance(None, &expected),
            ProvenanceOutcome::Missing
        );
    }

    #[test]
    fn provenance_mismatch_carries_both_values() {
        let expected = compute_provenance_sha256(b"calib-v1");
        let wrong = compute_provenance_sha256(b"calib-v2");
        match validate_recorded_provenance(Some(&wrong), &expected) {
            ProvenanceOutcome::Mismatch { recorded, expected: exp } => {
                assert_eq!(recorded, wrong);
                assert_eq!(exp, expected);
            }
            o => panic!("expected Mismatch, got {:?}", o),
        }
    }
}
