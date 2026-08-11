//! Shared domain validation for numeric tolerance/threshold CLI flags.
//!
//! Every CRUX lint gate is of the form `if observed > tolerance { fail }` or
//! `if observed < floor { fail }`. IEEE-754 says *every* comparison involving
//! NaN is false, so a NaN tolerance makes the failing branch unreachable: the
//! gate can never fire, the report prints a positive `Ok` for an observation it
//! never actually checked, and the command exits 0. A negative tolerance does
//! the same for the floor-style gates. Neither is a legitimate tolerance.
//!
//! The classifiers already refuse to judge a non-finite *observation*
//! (`AttnParityNumericsOutcome::NonFiniteMaxAbsDiff`); this module is the
//! symmetric guard on the *threshold* side. It is used twice:
//!
//! 1. as a clap `value_parser`, so a bad value is rejected at parse time
//!    (exit 2) before any gate runs, and
//! 2. as a `guard()` call at the top of each lint `run()`, so a caller that
//!    bypasses clap still fails closed instead of printing `Ok`.

use crate::error::{CliError, Result};

/// The closed interval a threshold flag must lie in, plus a human name used in
/// the error message.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ThresholdDomain {
    /// Inclusive lower bound.
    pub lo: f64,
    /// Inclusive upper bound.
    pub hi: f64,
    /// How the domain is described to the user, e.g. "a non-negative finite tolerance".
    pub what: &'static str,
}

/// A tolerance / epsilon: finite and non-negative, no upper bound.
pub(crate) const TOLERANCE: ThresholdDomain = ThresholdDomain {
    lo: 0.0,
    hi: f64::MAX,
    what: "a finite tolerance >= 0",
};

/// A fraction of one, e.g. a utilization threshold or a scaling-efficiency floor.
pub(crate) const FRACTION: ThresholdDomain = ThresholdDomain {
    lo: 0.0,
    hi: 1.0,
    what: "a finite fraction in [0.0, 1.0]",
};

/// A cosine-similarity floor, which legitimately spans the whole cosine range.
pub(crate) const COSINE: ThresholdDomain = ThresholdDomain {
    lo: -1.0,
    hi: 1.0,
    what: "a finite cosine similarity in [-1.0, 1.0]",
};

/// Reason a threshold value was rejected. Kept separate from the rendered
/// message so both the clap parser and `guard()` phrase it consistently.
fn reject_reason(value: f64, domain: ThresholdDomain) -> Option<String> {
    if value.is_nan() {
        return Some(format!(
            "NaN is not a threshold: every comparison against NaN is false, so the gate could never fail. Expected {}",
            domain.what
        ));
    }
    if value.is_infinite() {
        return Some(format!(
            "{value} disarms the gate rather than setting it. Expected {}",
            domain.what
        ));
    }
    if value < domain.lo || value > domain.hi {
        return Some(format!(
            "{value} is outside the valid domain. Expected {}",
            domain.what
        ));
    }
    None
}

/// Validate an already-parsed threshold. Returns the value unchanged when it is
/// usable, or a rendered rejection message.
pub(crate) fn check(value: f64, domain: ThresholdDomain) -> std::result::Result<f64, String> {
    match reject_reason(value, domain) {
        Some(msg) => Err(msg),
        None => Ok(value),
    }
}

/// clap `value_parser` for a tolerance/epsilon flag.
pub(crate) fn parse_tolerance(s: &str) -> std::result::Result<f64, String> {
    parse_in(s, TOLERANCE)
}

/// clap `value_parser` for a `[0.0, 1.0]` fraction flag.
pub(crate) fn parse_fraction(s: &str) -> std::result::Result<f64, String> {
    parse_in(s, FRACTION)
}

/// clap `value_parser` for a cosine-similarity floor flag.
pub(crate) fn parse_cosine(s: &str) -> std::result::Result<f64, String> {
    parse_in(s, COSINE)
}

fn parse_in(s: &str, domain: ThresholdDomain) -> std::result::Result<f64, String> {
    let value: f64 = s.parse().map_err(|_| "invalid float literal".to_string())?;
    check(value, domain)
}

/// Fail-closed guard for the `run()` entry points, so a non-clap caller cannot
/// disarm a gate either. Errors as `ValidationFailed` (exit 5), matching the
/// exit code the gate itself would have produced.
pub(crate) fn guard(flag: &str, value: f64, domain: ThresholdDomain) -> Result<()> {
    match reject_reason(value, domain) {
        Some(msg) => Err(CliError::ValidationFailed(format!(
            "invalid value for {flag}: {msg}"
        ))),
        None => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nan_is_rejected_in_every_domain() {
        for d in [TOLERANCE, FRACTION, COSINE] {
            let err = check(f64::NAN, d).unwrap_err();
            assert!(
                err.contains("NaN is not a threshold"),
                "NaN must be rejected for {d:?}; got: {err}"
            );
        }
    }

    #[test]
    fn infinities_are_rejected_in_every_domain() {
        for d in [TOLERANCE, FRACTION, COSINE] {
            assert!(check(f64::INFINITY, d).is_err(), "+inf must be rejected");
            assert!(
                check(f64::NEG_INFINITY, d).is_err(),
                "-inf must be rejected"
            );
        }
    }

    #[test]
    fn negative_tolerance_is_rejected() {
        let err = check(-1.0, TOLERANCE).unwrap_err();
        assert!(err.contains("outside the valid domain"), "got: {err}");
        assert!(check(-1.0, FRACTION).is_err());
        // A cosine floor legitimately reaches -1.0.
        assert_eq!(check(-1.0, COSINE), Ok(-1.0));
    }

    #[test]
    fn legitimate_values_pass_through_unchanged() {
        assert_eq!(check(0.0, TOLERANCE), Ok(0.0));
        assert_eq!(check(5e-3, TOLERANCE), Ok(5e-3));
        assert_eq!(check(1e9, TOLERANCE), Ok(1e9));
        assert_eq!(check(0.95, FRACTION), Ok(0.95));
        assert_eq!(check(1.0, FRACTION), Ok(1.0));
        assert_eq!(check(0.9999, COSINE), Ok(0.9999));
    }

    #[test]
    fn fraction_rejects_out_of_range_upper_bound() {
        let err = check(99.0, FRACTION).unwrap_err();
        assert!(err.contains("outside the valid domain"), "got: {err}");
    }

    #[test]
    fn parsers_reject_nan_and_keep_rejecting_garbage() {
        assert!(parse_tolerance("nan").is_err());
        assert!(parse_tolerance("NaN").is_err());
        assert!(parse_fraction("nan").is_err());
        assert!(parse_cosine("nan").is_err());
        assert_eq!(
            parse_tolerance("banana").unwrap_err(),
            "invalid float literal"
        );
        assert_eq!(parse_tolerance("1e-5"), Ok(1e-5));
        assert_eq!(parse_fraction("0.85"), Ok(0.85));
    }

    /// `Commands` is a very large enum; building the clap command tree needs
    /// more stack than the 2 MiB a test thread gets by default.
    fn on_big_stack(f: impl FnOnce() + Send + 'static) {
        std::thread::Builder::new()
            .stack_size(32 * 1024 * 1024)
            .spawn(f)
            .expect("spawn")
            .join()
            .expect("join");
    }

    /// The user-visible half of the fix: clap itself must refuse the value, so
    /// no gate ever runs and no `Ok` is ever printed. One case per flag in the
    /// family, each with the literal that shipped the disarm in 0.63.0.
    #[test]
    fn cli_rejects_nan_on_every_threshold_flag_in_the_lint_family() {
        on_big_stack(cli_rejects_nan_body);
    }

    fn cli_rejects_nan_body() {
        use clap::Parser;

        let disarming: &[&[&str]] = &[
            &[
                "apr",
                "kv-timeline-lint",
                "--timeline-file",
                "kv.json",
                "--preempt-threshold",
                "nan",
            ],
            &[
                "apr",
                "kv-timeline-lint",
                "--timeline-file",
                "kv.json",
                "--preempt-threshold=-1",
            ],
            &[
                "apr",
                "attn-parity-lint",
                "--parity-file",
                "p.json",
                "--tol-abs",
                "nan",
            ],
            &[
                "apr",
                "attn-parity-lint",
                "--parity-file",
                "p.json",
                "--tol-cos",
                "NaN",
            ],
            &[
                "apr",
                "attn-viz-lint",
                "--attn-file",
                "a.json",
                "--tolerance",
                "nan",
            ],
            &[
                "apr",
                "attn-viz-lint",
                "--attn-file",
                "a.json",
                "--epsilon",
                "nan",
            ],
            &[
                "apr",
                "explain-token-lint",
                "--jsonl-file",
                "e.jsonl",
                "--tolerance",
                "nan",
            ],
            &[
                "apr",
                "ddp-metrics-lint",
                "--metrics-1gpu-file",
                "a.json",
                "--metrics-ngpu-file",
                "b.json",
                "--world-size",
                "4",
                "--scaling-floor",
                "nan",
            ],
            &[
                "apr",
                "ddp-metrics-lint",
                "--metrics-1gpu-file",
                "a.json",
                "--metrics-ngpu-file",
                "b.json",
                "--world-size",
                "4",
                "--loss-tolerance",
                "nan",
            ],
            &[
                "apr",
                "ddp-metrics-lint",
                "--metrics-1gpu-file",
                "a.json",
                "--metrics-ngpu-file",
                "b.json",
                "--world-size",
                "4",
                "--scaling-floor=-1",
            ],
        ];

        for argv in disarming {
            let parsed = crate::Cli::try_parse_from(argv.iter().copied());
            assert!(
                parsed.is_err(),
                "clap accepted a gate-disarming threshold: {argv:?}"
            );
        }
    }

    /// The fix must not narrow the legitimate domain: every value the shipped
    /// falsification suites pass on the command line still parses.
    #[test]
    fn cli_still_accepts_the_documented_threshold_values() {
        on_big_stack(cli_accepts_documented_body);
    }

    fn cli_accepts_documented_body() {
        use clap::Parser;

        let legitimate: &[&[&str]] = &[
            &[
                "apr",
                "kv-timeline-lint",
                "--timeline-file",
                "kv.json",
                "--preempt-threshold",
                "0.80",
            ],
            &[
                "apr",
                "attn-parity-lint",
                "--parity-file",
                "p.json",
                "--tol-abs",
                "0.01",
                "--tol-cos",
                "0.9999",
            ],
            &[
                "apr",
                "attn-viz-lint",
                "--attn-file",
                "a.json",
                "--tolerance",
                "0.05",
                "--epsilon",
                "1e-9",
            ],
            &[
                "apr",
                "explain-token-lint",
                "--jsonl-file",
                "e.jsonl",
                "--tolerance",
                "0.05",
            ],
            &[
                "apr",
                "ddp-metrics-lint",
                "--metrics-1gpu-file",
                "a.json",
                "--metrics-ngpu-file",
                "b.json",
                "--world-size",
                "4",
                "--scaling-floor",
                "0.5",
                "--loss-tolerance",
                "0.01",
            ],
        ];

        for argv in legitimate {
            let parsed = crate::Cli::try_parse_from(argv.iter().copied());
            assert!(
                parsed.is_ok(),
                "clap rejected a legitimate threshold: {argv:?}"
            );
        }
    }

    #[test]
    fn guard_reports_the_flag_name_and_is_validation_failed() {
        let err = guard("--tol-abs", f64::NAN, TOLERANCE).unwrap_err();
        match err {
            CliError::ValidationFailed(msg) => {
                assert!(msg.contains("--tol-abs"), "got: {msg}");
                assert!(msg.contains("NaN"), "got: {msg}");
            }
            other => panic!("expected ValidationFailed, got {other:?}"),
        }
        assert!(guard("--tol-abs", 5e-3, TOLERANCE).is_ok());
    }
}
