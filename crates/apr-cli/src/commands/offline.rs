//! Offline mode detection for `apr pull --offline` (CRUX-A-20).
//!
//! Contract: `contracts/crux-A-20-v1.yaml`.
//!
//! Pure classifier — takes the parsed `--offline` flag and a slice of
//! environment variables, returns a `bool`. No I/O. Unit-testable offline.
//! The actual network-gating behavior (zero TCP connects, cache-miss
//! error messages) is discharged by a separate strace-gated harness.

/// Environment variables that trigger offline mode. APR-native and
/// HF-compatible. Both MUST be observationally equivalent to the
/// `--offline` CLI flag per FALSIFY-CRUX-A-20-004.
pub const OFFLINE_ENV_VARS: &[&str] = &["APR_OFFLINE", "HF_HUB_OFFLINE"];

/// Return true iff the raw env value is a truthy offline signal.
/// HF's own convention (`huggingface_hub.constants.HF_HUB_OFFLINE`) treats
/// "1", "true", "TRUE", "yes" as true and "0", "false", "" as false.
fn env_is_truthy(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on"
    )
}

/// Resolve offline mode from the CLI flag + environment snapshot.
///
/// Precedence: `cli_flag=true` OR any offline env var truthy → offline.
/// If none are set, or all env vars are falsy and the flag is false,
/// offline is false (default online behavior).
///
/// The environment snapshot is passed in explicitly (rather than read
/// from `std::env`) so callers can test this function deterministically
/// without mutating process-global state.
pub fn is_offline<'a, I>(cli_flag: bool, env: I) -> bool
where
    I: IntoIterator<Item = (&'a str, &'a str)>,
{
    if cli_flag {
        return true;
    }
    for (k, v) in env {
        if OFFLINE_ENV_VARS.contains(&k) && env_is_truthy(v) {
            return true;
        }
    }
    false
}

/// Read the two offline-relevant env vars out of the real process
/// environment. Thin wrapper so callers don't sprinkle `std::env::var`
/// across the codebase.
pub fn read_offline_env() -> Vec<(String, String)> {
    OFFLINE_ENV_VARS
        .iter()
        .filter_map(|k| std::env::var(k).ok().map(|v| ((*k).to_string(), v)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flag_alone_triggers_offline() {
        assert!(is_offline(true, std::iter::empty::<(&str, &str)>()));
    }

    #[test]
    fn no_flag_no_env_is_online() {
        assert!(!is_offline(false, std::iter::empty::<(&str, &str)>()));
    }

    #[test]
    fn apr_offline_one_triggers_offline() {
        assert!(is_offline(false, [("APR_OFFLINE", "1")]));
    }

    #[test]
    fn hf_hub_offline_one_triggers_offline() {
        assert!(is_offline(false, [("HF_HUB_OFFLINE", "1")]));
    }

    #[test]
    fn apr_offline_zero_is_online() {
        assert!(!is_offline(false, [("APR_OFFLINE", "0")]));
    }

    #[test]
    fn apr_offline_empty_is_online() {
        assert!(!is_offline(false, [("APR_OFFLINE", "")]));
    }

    #[test]
    fn truthy_variants_all_work() {
        for v in ["1", "true", "TRUE", "yes", "on", "  1  "] {
            assert!(
                is_offline(false, [("APR_OFFLINE", v)]),
                "APR_OFFLINE={v:?} must be truthy",
            );
        }
    }

    #[test]
    fn falsy_variants_all_fail() {
        for v in ["0", "false", "no", "off", "", "random-string"] {
            assert!(
                !is_offline(false, [("APR_OFFLINE", v)]),
                "APR_OFFLINE={v:?} must be falsy",
            );
        }
    }

    #[test]
    fn unrelated_env_var_ignored() {
        assert!(!is_offline(false, [("SOME_OTHER_VAR", "1")]));
    }

    #[test]
    fn flag_overrides_falsy_env() {
        assert!(is_offline(true, [("APR_OFFLINE", "0")]));
    }

    #[test]
    fn is_deterministic() {
        let a = is_offline(false, [("HF_HUB_OFFLINE", "1")]);
        let b = is_offline(false, [("HF_HUB_OFFLINE", "1")]);
        assert_eq!(a, b);
    }
}
