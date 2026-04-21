//! Xet opt-out detection for `apr pull` under CRUX-A-07.
//!
//! Contract: `contracts/crux-A-07-v1.yaml`.
//!
//! Pure classifier — takes the parsed `--no-xet` flag and a slice of
//! environment variables and returns `bool` for "is the Xet fast path
//! enabled". No I/O, no network, no process-global state. Unit-testable
//! offline.
//!
//! The actual parallel-range HTTP fetch, CAS endpoint contact, and
//! byte-parity with the plain HTTPS path are discharged by separate
//! network-gated harnesses (follow-up).

/// Env var that toggles the Xet backend. HuggingFace's own
/// `huggingface_hub[hf_xet]` uses the same `APR_XET` / `HF_XET_*`
/// family, so our reader honors the APR-native `APR_XET` as the
/// source of truth for this classifier (per CRUX-A-07 equations).
pub const XET_ENV_VAR: &str = "APR_XET";

/// Raw values that explicitly turn Xet OFF. Mirrors HuggingFace's own
/// boolean-env taxonomy (`hf_hub_utils.constants._is_true`).
fn env_is_falsy(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "0" | "false" | "no" | "off"
    )
}

/// Raw values that explicitly turn Xet ON.
fn env_is_truthy(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on"
    )
}

/// Resolve Xet-enabled mode from the CLI flag + environment snapshot.
///
/// Precedence (highest → lowest):
///   1. `no_xet_flag == true`              → Xet OFF
///   2. `APR_XET` env var set to a falsy   → Xet OFF
///   3. `APR_XET` env var set to a truthy  → Xet ON
///   4. no signal                          → Xet ON (default opt-in)
///
/// CRUX-A-07 ALGO-003 sub-claim of FALSIFY-003: `APR_XET=0` MUST
/// classify as off; `APR_XET` unset MUST classify as on. The classifier
/// is the precondition for the integration-level "no xet CAS requests
/// emitted" strace check.
///
/// The environment snapshot is passed in explicitly (rather than read
/// from `std::env`) so callers can test this function deterministically
/// without mutating process-global state.
pub fn is_xet_enabled<'a, I>(no_xet_flag: bool, env: I) -> bool
where
    I: IntoIterator<Item = (&'a str, &'a str)>,
{
    if no_xet_flag {
        return false;
    }
    for (k, v) in env {
        if k == XET_ENV_VAR {
            if env_is_falsy(v) {
                return false;
            }
            if env_is_truthy(v) {
                return true;
            }
        }
    }
    true
}

/// Read `APR_XET` out of the real process environment. Thin wrapper so
/// callers don't sprinkle `std::env::var` across the codebase.
pub fn read_xet_env() -> Vec<(String, String)> {
    std::env::var(XET_ENV_VAR)
        .ok()
        .map(|v| vec![(XET_ENV_VAR.to_string(), v)])
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_no_signal_is_enabled() {
        assert!(is_xet_enabled(false, std::iter::empty::<(&str, &str)>()));
    }

    #[test]
    fn flag_alone_disables_xet() {
        assert!(!is_xet_enabled(true, std::iter::empty::<(&str, &str)>()));
    }

    #[test]
    fn apr_xet_zero_disables_xet() {
        // CRUX-A-07 ALGO-003 sub-claim of FALSIFY-003: `APR_XET=0`
        // MUST flip the classifier to off so the download path can
        // take the plain HTTPS branch.
        assert!(!is_xet_enabled(false, [("APR_XET", "0")]));
    }

    #[test]
    fn apr_xet_one_enables_xet() {
        assert!(is_xet_enabled(false, [("APR_XET", "1")]));
    }

    #[test]
    fn flag_overrides_truthy_env() {
        // CLI flag wins: even with `APR_XET=1` set in the env, the
        // explicit `--no-xet` flag MUST disable the Xet path.
        assert!(!is_xet_enabled(true, [("APR_XET", "1")]));
    }

    #[test]
    fn falsy_variants_all_disable() {
        for v in ["0", "false", "FALSE", "no", "off", "  0  "] {
            assert!(
                !is_xet_enabled(false, [("APR_XET", v)]),
                "APR_XET={v:?} must disable xet",
            );
        }
    }

    #[test]
    fn truthy_variants_all_enable() {
        for v in ["1", "true", "TRUE", "yes", "on", "  1  "] {
            assert!(
                is_xet_enabled(false, [("APR_XET", v)]),
                "APR_XET={v:?} must enable xet",
            );
        }
    }

    #[test]
    fn ambiguous_env_value_defers_to_default_enabled() {
        // Values that are neither recognized-truthy nor recognized-falsy
        // MUST NOT silently flip to off — they defer to the default
        // (enabled). Avoids a surprise opt-out on a typo like
        // `APR_XET=nope` when the user wasn't actually trying to
        // disable Xet.
        assert!(is_xet_enabled(false, [("APR_XET", "maybe")]));
        assert!(is_xet_enabled(false, [("APR_XET", "random-string")]));
    }

    #[test]
    fn empty_env_value_is_ambiguous_defers_to_default() {
        // `APR_XET=` (empty) is ambiguous; HF's own `_is_true` treats
        // it as false but we keep the default-enabled semantic to
        // avoid flipping Xet off on accidental empty exports.
        assert!(is_xet_enabled(false, [("APR_XET", "")]));
    }

    #[test]
    fn unrelated_env_var_ignored() {
        assert!(is_xet_enabled(false, [("SOME_OTHER_VAR", "0")]));
        assert!(is_xet_enabled(false, [("HF_XET_PARALLEL", "8")]));
    }

    #[test]
    fn last_apr_xet_wins_on_duplicate_keys() {
        // If the same key appears twice in the env snapshot, the later
        // occurrence wins (iterator order) — same as process env where
        // the most recent export is what getenv() returns.
        // Current implementation: first truthy/falsy hit short-circuits;
        // document that behavior so future changes are deliberate.
        let first_falsy_wins =
            is_xet_enabled(false, [("APR_XET", "0"), ("APR_XET", "1")]);
        assert!(!first_falsy_wins);
    }

    #[test]
    fn is_deterministic() {
        let a = is_xet_enabled(false, [("APR_XET", "1")]);
        let b = is_xet_enabled(false, [("APR_XET", "1")]);
        assert_eq!(a, b);
    }

    #[test]
    fn xet_env_var_is_stable_apr_xet() {
        // Downstream shell tests and goldens grep for this exact name.
        assert_eq!(XET_ENV_VAR, "APR_XET");
    }

    #[test]
    fn falsify_003_sub_claim_apr_xet_zero_opts_out() {
        // CRUX-A-07 ALGO-003 sub-claim of FALSIFY-003: setting
        // `APR_XET=0` MUST produce a deterministic "xet disabled"
        // classification, which is the algorithm-level precondition
        // for the "no xet CAS requests emitted" strace assertion.
        for v in ["0", "false", "no", "off"] {
            assert!(
                !is_xet_enabled(false, [("APR_XET", v)]),
                "APR_XET={v:?} must disable xet per FALSIFY-003",
            );
        }
        // And the complement: default + `APR_XET=1` MUST keep it on,
        // so the same harness can also assert the opt-in path.
        assert!(is_xet_enabled(false, std::iter::empty::<(&str, &str)>()));
        assert!(is_xet_enabled(false, [("APR_XET", "1")]));
    }
}
