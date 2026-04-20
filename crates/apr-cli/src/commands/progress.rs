//! Progress suppression decision logic for `apr pull` (CRUX-A-19).
//!
//! Contract: `contracts/crux-A-19-v1.yaml`.
//!
//! Pure decision function — takes the observed TTY-ness of stderr, the
//! parsed `--quiet` flag, and a slice of environment variables, and
//! returns whether a progress bar should be emitted. No I/O.
//!
//! The actual progress-bar rendering, the live TTY check against
//! `stderr`, and the tqdm-parity regex match on emitted lines are all
//! discharged by a separate TTY-gated harness (follow-up).

/// Environment variable that force-disables progress output (HF-parity
/// to `HF_HUB_DISABLE_PROGRESS_BARS`). aprender uses `APR_PROGRESS=0`
/// per the CRUX-A-19 contract body.
pub const PROGRESS_OFF_ENV: &str = "APR_PROGRESS";

/// Return true iff the raw env value is a "falsy / disable" signal.
/// Accept the same taxonomy as CRUX-A-20: "0", "false", "no", "off"
/// disable progress; "1", "true", "yes", "on" (or absent) leave it on.
fn env_disables_progress(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "0" | "false" | "no" | "off"
    )
}

/// Decide whether a progress bar should be emitted.
///
/// Suppress progress iff ANY of:
///   - stderr is not a TTY (piped / redirected), OR
///   - `--quiet` flag is set, OR
///   - `APR_PROGRESS=0` (or equivalent falsy env value) is set.
///
/// Otherwise emit progress. This is the CRUX-A-19 `progress_suppression`
/// equation stated as a pure function.
///
/// The environment is passed in explicitly (rather than reading
/// `std::env`) so the decision is deterministic and unit-testable
/// without mutating process-global state.
pub fn progress_enabled<'a, I>(stderr_is_tty: bool, quiet_flag: bool, env: I) -> bool
where
    I: IntoIterator<Item = (&'a str, &'a str)>,
{
    if !stderr_is_tty {
        return false;
    }
    if quiet_flag {
        return false;
    }
    for (k, v) in env {
        if k == PROGRESS_OFF_ENV && env_disables_progress(v) {
            return false;
        }
    }
    true
}

/// Convenience wrapper reading the real process env. Thin layer so
/// callers don't sprinkle `std::env::var` across the codebase.
pub fn read_progress_env() -> Vec<(String, String)> {
    std::env::var(PROGRESS_OFF_ENV)
        .ok()
        .map(|v| vec![(PROGRESS_OFF_ENV.to_string(), v)])
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tty_no_quiet_no_env_emits_progress() {
        assert!(progress_enabled(true, false, std::iter::empty::<(&str, &str)>()));
    }

    #[test]
    fn non_tty_always_suppresses() {
        assert!(!progress_enabled(false, false, std::iter::empty::<(&str, &str)>()));
        assert!(!progress_enabled(false, true, std::iter::empty::<(&str, &str)>()));
        assert!(!progress_enabled(false, false, [("APR_PROGRESS", "1")]));
    }

    #[test]
    fn quiet_flag_suppresses_on_tty() {
        assert!(!progress_enabled(true, true, std::iter::empty::<(&str, &str)>()));
    }

    #[test]
    fn apr_progress_zero_suppresses_on_tty() {
        assert!(!progress_enabled(true, false, [("APR_PROGRESS", "0")]));
    }

    #[test]
    fn apr_progress_one_leaves_progress_on() {
        assert!(progress_enabled(true, false, [("APR_PROGRESS", "1")]));
    }

    #[test]
    fn apr_progress_missing_leaves_progress_on() {
        assert!(progress_enabled(true, false, [("SOME_OTHER", "0")]));
    }

    #[test]
    fn falsy_env_variants_all_suppress() {
        for v in ["0", "false", "FALSE", "no", "off", "  off  "] {
            assert!(
                !progress_enabled(true, false, [("APR_PROGRESS", v)]),
                "APR_PROGRESS={v:?} must suppress",
            );
        }
    }

    #[test]
    fn truthy_env_variants_leave_progress_on() {
        for v in ["1", "true", "yes", "on", ""] {
            assert!(
                progress_enabled(true, false, [("APR_PROGRESS", v)]),
                "APR_PROGRESS={v:?} must leave progress on",
            );
        }
    }

    #[test]
    fn three_suppression_signals_observationally_equivalent() {
        // Signal 1: non-TTY
        let a = progress_enabled(false, false, std::iter::empty::<(&str, &str)>());
        // Signal 2: --quiet
        let b = progress_enabled(true, true, std::iter::empty::<(&str, &str)>());
        // Signal 3: APR_PROGRESS=0
        let c = progress_enabled(true, false, [("APR_PROGRESS", "0")]);
        assert_eq!(a, b);
        assert_eq!(b, c);
        assert!(!a, "all three signals must produce false");
    }

    #[test]
    fn unrelated_env_var_ignored() {
        assert!(progress_enabled(true, false, [("HF_HUB_DISABLE_PROGRESS_BARS", "1")]));
    }

    #[test]
    fn quiet_overrides_truthy_env() {
        assert!(!progress_enabled(true, true, [("APR_PROGRESS", "1")]));
    }

    #[test]
    fn is_deterministic() {
        let a = progress_enabled(true, false, [("APR_PROGRESS", "0")]);
        let b = progress_enabled(true, false, [("APR_PROGRESS", "0")]);
        assert_eq!(a, b);
    }
}
