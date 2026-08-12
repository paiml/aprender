//! The one SPDX-identifier table the CLI validates against.
//!
//! `apr validate-manifest` (FALSIFY-PM-004) has always enforced SPDX on
//! `license` / `provenance.parent_license` / `provenance.data_license` and
//! fails closed on anything else. The two commands that WRITE those same
//! fields — `apr stamp` and `apr publish` — did not: dogfooding 0.63.0
//! stamped `--license 'NOT-A-LICENSE-!!'` into a model's provenance block
//! and exited 0, and `apr publish --license 'NOT-A-LICENSE'` lower-cased it
//! straight into the Hugging Face model card's YAML front matter, where an
//! unrecognised `license:` value is rejected by the Hub (issue #2391).
//!
//! Both surfaces now share this table with the validator, so a value that
//! `apr stamp` accepts is a value `apr validate-manifest` accepts.

/// SPDX identifiers accepted without question. Not exhaustive; extend as new
/// licenses appear in provenance chains.
///
/// The non-SPDX tail (`llama2`, `gemma`, `custom`, …) is deliberate: those are
/// the identifiers the Hugging Face Hub itself uses for model licenses.
pub(crate) const SPDX_ALLOWLIST: &[&str] = &[
    "Apache-2.0",
    "MIT",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "MPL-2.0",
    "LGPL-2.1",
    "LGPL-2.1-only",
    "LGPL-3.0",
    "LGPL-3.0-only",
    "GPL-2.0",
    "GPL-2.0-only",
    "GPL-3.0",
    "GPL-3.0-only",
    "CC-BY-4.0",
    "CC-BY-SA-4.0",
    "CC-BY-NC-4.0",
    "CC0-1.0",
    "Unlicense",
    "ISC",
    "Apache-2.0 WITH LLVM-exception",
    "llama2",
    "llama3",
    "llama3.1",
    "gemma",
    "custom",
];

/// `true` iff `value` is an accepted identifier (case-insensitive, matching
/// `validate-manifest`'s long-standing comparison).
pub(crate) fn is_accepted(value: &str) -> bool {
    SPDX_ALLOWLIST
        .iter()
        .any(|a| a.eq_ignore_ascii_case(value.trim()))
}

/// Reject reason for a `--license`-style flag, or `None` when acceptable.
///
/// The message names the flag and lists the accepted values, because the
/// failure a user hits is "I typed the license the way my lawyer writes it".
pub(crate) fn reject_reason(flag: &str, value: &str) -> Option<String> {
    if is_accepted(value) {
        return None;
    }
    Some(format!(
        "{flag} {value:?} is not a recognised SPDX identifier. \
         `apr validate-manifest` (FALSIFY-PM-004) rejects it, and the Hugging Face \
         Hub ignores an unrecognised `license:` value, so stamping it would bake an \
         unverifiable license into the artifact. Accepted: {}",
        SPDX_ALLOWLIST.join(", ")
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_identifiers_are_accepted() {
        for ok in ["Apache-2.0", "MIT", "CC-BY-4.0", "llama3.1", "custom"] {
            assert!(is_accepted(ok), "{ok} must be accepted");
            assert!(reject_reason("--license", ok).is_none());
        }
    }

    #[test]
    fn case_and_surrounding_space_do_not_matter() {
        assert!(is_accepted("apache-2.0"));
        assert!(is_accepted("mit"));
        assert!(is_accepted("  MIT  "));
    }

    /// The exact value dogfooding stamped into a shipped artifact.
    #[test]
    fn punctuation_garbage_is_rejected() {
        assert!(!is_accepted("NOT-A-LICENSE-!!"));
        let why = reject_reason("--license", "NOT-A-LICENSE-!!").expect("must reject");
        assert!(why.contains("NOT-A-LICENSE-!!"), "{why}");
        assert!(
            why.contains("Apache-2.0"),
            "must list accepted values: {why}"
        );
    }

    #[test]
    fn near_misses_and_empty_are_rejected() {
        for bad in ["", "  ", "Apache2", "Apache-2", "MIT License", "GPL"] {
            assert!(!is_accepted(bad), "{bad:?} must be rejected");
        }
    }
}
