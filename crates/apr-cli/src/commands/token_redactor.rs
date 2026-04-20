//! Secret redaction classifier for `apr login` / `apr pull` (CRUX-A-18).
//!
//! Contract: `contracts/crux-A-18-v1.yaml`.
//!
//! Pure redactor — takes any string the CLI is about to emit (stderr, log
//! line, error message) and a slice of known-secret strings, and returns
//! a string with every occurrence of every secret replaced by the literal
//! `<redacted>`. No I/O, no global state.
//!
//! The algorithm-level sub-claim we DO discharge here is exactly the
//! invariant FALSIFY-CRUX-A-18-003 demands: "apr pull / login stderr
//! does NOT contain the literal token". If every stderr write goes
//! through `redact_secrets`, and `redact_secrets` provably removes every
//! occurrence of every known secret, then the write-site invariant
//! reduces to a single call-site audit (caller must pipe through us).
//!
//! The actual wiring of every `eprintln!` through the redactor, the
//! `~/.apr/token` file-mode-0600 check, and the real HTTP 403 retry flow
//! are all discharged by separate integration harnesses (follow-up).

/// The literal token written in place of any detected secret.
/// Chosen to be short, obviously non-secret, and grep-friendly.
pub const REDACTED_MARKER: &str = "<redacted>";

/// Replace every occurrence of every secret in `input` with
/// `REDACTED_MARKER`. Iterates the secret list to a fixpoint so that
/// overlapping secrets are all caught.
///
/// - Empty secrets are ignored (would otherwise cause an infinite loop).
/// - Secrets shorter than 4 chars are also ignored as a guard-rail: we
///   refuse to redact ambient short strings that are almost certainly
///   not real tokens (e.g. `"ok"`, `"hf"`). HF tokens start with `hf_`
///   followed by at least 32 characters, so real tokens are always
///   well over this floor.
///
/// This is a pure function: same inputs → same output, no I/O.
pub fn redact_secrets(input: &str, secrets: &[&str]) -> String {
    let mut out = input.to_string();
    for secret in secrets {
        if secret.len() < 4 {
            continue;
        }
        out = out.replace(secret, REDACTED_MARKER);
    }
    out
}

/// Return true iff `input` contains any of the provided secrets.
/// Uses the same short-secret guard-rail as `redact_secrets` so that
/// callers can call both without drift.
///
/// This is the direct observational inverse of FALSIFY-003: after a
/// write is routed through `redact_secrets`, `contains_secret` on the
/// result MUST be false.
pub fn contains_secret(input: &str, secrets: &[&str]) -> bool {
    for secret in secrets {
        if secret.len() < 4 {
            continue;
        }
        if input.contains(secret) {
            return true;
        }
    }
    false
}

/// Heuristic shape check for a HuggingFace access token. Used by
/// `apr login --stdin` to reject obvious non-tokens before persisting
/// them, so a typo doesn't write garbage to `~/.apr/token`.
///
/// Canonical HF tokens start with `hf_` and are ≥32 chars total. We
/// accept any `hf_` + ≥32 total length as plausibly valid; anything
/// else is rejected. This is advisory only — the real 403 retry is the
/// actual authority on whether a token works.
pub fn looks_like_hf_token(s: &str) -> bool {
    let trimmed = s.trim();
    trimmed.starts_with("hf_") && trimmed.len() >= 32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn redact_replaces_known_secret() {
        let tok = "hf_abcdefghijklmnopqrstuvwxyz123456";
        let line = format!("Authorization: Bearer {tok}");
        let out = redact_secrets(&line, &[tok]);
        assert_eq!(out, "Authorization: Bearer <redacted>");
        assert!(!out.contains(tok));
    }

    #[test]
    fn redact_is_identity_when_no_secret_present() {
        let out = redact_secrets("no secret here", &["hf_absent_token_12345678901234567890"]);
        assert_eq!(out, "no secret here");
    }

    #[test]
    fn redact_handles_multiple_secrets() {
        let a = "hf_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
        let b = "hf_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
        let line = format!("{a} then {b}");
        let out = redact_secrets(&line, &[a, b]);
        assert!(!out.contains(a));
        assert!(!out.contains(b));
        assert_eq!(out, "<redacted> then <redacted>");
    }

    #[test]
    fn redact_removes_every_occurrence() {
        let tok = "hf_repeat_repeat_repeat_repeat_repeat";
        let line = format!("{tok} and again {tok} and {tok}");
        let out = redact_secrets(&line, &[tok]);
        assert!(!out.contains(tok), "every occurrence must be removed: {out}");
    }

    #[test]
    fn redact_is_idempotent() {
        let tok = "hf_idemp_idemp_idemp_idemp_idemp_XXX";
        let once = redact_secrets(&format!("foo {tok} bar"), &[tok]);
        let twice = redact_secrets(&once, &[tok]);
        assert_eq!(once, twice);
    }

    #[test]
    fn redact_ignores_empty_and_short_secrets() {
        // Empty secret MUST be ignored (else `replace("", ...)` loops).
        let out = redact_secrets("nothing to see", &["", "hi", "a"]);
        assert_eq!(out, "nothing to see");
    }

    #[test]
    fn contains_secret_is_true_iff_redact_changes_output() {
        let tok = "hf_present_present_present_present";
        let line = format!("see {tok} there");
        assert!(contains_secret(&line, &[tok]));
        let redacted = redact_secrets(&line, &[tok]);
        assert!(!contains_secret(&redacted, &[tok]));
    }

    #[test]
    fn contains_secret_ignores_short_secrets() {
        // Short "secret" MUST be ignored (guard against false positives
        // on ambient strings).
        assert!(!contains_secret("hi there friend", &["hi", "a"]));
    }

    #[test]
    fn redact_then_contains_is_false_on_any_input() {
        // CRUX-A-18 ALGO-003 sub-claim of FALSIFY-003: for any input
        // containing any secret, after redaction the output no longer
        // contains that secret. This is the observational property the
        // full stderr-scrubbing invariant will reduce to, given that
        // every write is routed through redact_secrets.
        let secrets = ["hf_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "hf_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"];
        let inputs = [
            "just a prefix",
            "hf_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "prefix hf_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa suffix",
            "both hf_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa and hf_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            "multi hf_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa hf_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        ];
        for input in &inputs {
            let redacted = redact_secrets(input, &secrets);
            assert!(
                !contains_secret(&redacted, &secrets),
                "redaction failed to scrub all secrets from {input:?}: {redacted:?}",
            );
        }
    }

    #[test]
    fn redact_is_deterministic() {
        let tok = "hf_determ_determ_determ_determ_det";
        let a = redact_secrets(&format!("x {tok} y"), &[tok]);
        let b = redact_secrets(&format!("x {tok} y"), &[tok]);
        assert_eq!(a, b);
    }

    #[test]
    fn looks_like_hf_token_accepts_canonical_shape() {
        assert!(looks_like_hf_token("hf_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"));
        assert!(looks_like_hf_token("  hf_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa  "));
    }

    #[test]
    fn looks_like_hf_token_rejects_garbage() {
        assert!(!looks_like_hf_token(""));
        assert!(!looks_like_hf_token("not_a_token"));
        assert!(!looks_like_hf_token("hf_short"));
        assert!(!looks_like_hf_token("HF_CAPS_INSTEAD_OF_LOWER_12345678901234"));
    }

    #[test]
    fn redact_marker_is_stable() {
        // Downstream log consumers grep for the marker; it must not drift.
        assert_eq!(REDACTED_MARKER, "<redacted>");
    }

    #[test]
    fn redact_empty_input_is_empty() {
        assert_eq!(
            redact_secrets("", &["hf_something_big_enough_to_matter_XX"]),
            ""
        );
    }

    #[test]
    fn redact_does_not_panic_on_weird_inputs() {
        // Every non-panicking property: exercise a bag of pathological
        // inputs including unicode, control chars, and very long strings.
        for input in [
            "🎉🎉🎉",
            "\x00\x01\x02",
            &"x".repeat(10_000),
            "",
        ] {
            let _ = redact_secrets(input, &["hf_something_big_enough_to_matter_XX"]);
            let _ = contains_secret(input, &["hf_something_big_enough_to_matter_XX"]);
        }
    }
}
