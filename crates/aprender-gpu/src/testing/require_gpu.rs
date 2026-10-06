//! `APR_REQUIRE_GPU`: where a GPU is promised, "no GPU" must fail a test, not skip it.
//!
//! A CUDA test written to be portable reads "no device" as "nothing to test",
//! prints a skip line and returns, and the harness reports `ok`. On a runner
//! whose whole job is to exercise a GPU that is a green which measured nothing:
//! a box that lost its driver, its device node or its permissions passes every
//! such test. Setting `APR_REQUIRE_GPU=1` turns each skip path that goes through
//! [`skip_or_panic`] into a panic, so the absence becomes the failure.
//!
//! The switch is read from the environment of the test process and nowhere else:
//! nothing in the library proper consults it, so production behaviour is the same
//! with or without it.
//!
//! Skip paths that do not go through this module are not covered by the panic. The
//! CI scripts that arm the switch also fail on any skip line in the test output
//! (see `scripts/ci_jetson_gpu.sh`), which is the second net for those.

use std::fmt::Debug;

/// Name of the environment variable that arms the check.
pub const REQUIRE_GPU_ENV: &str = "APR_REQUIRE_GPU";

/// Decide from the raw value of [`REQUIRE_GPU_ENV`] whether a GPU is required.
///
/// Unset, empty, `0` and `false` (any case) leave skipping allowed. Any other
/// value arms the check, so a value such as `yes` or `true` fails closed instead
/// of silently disarming the poka-yoke.
#[must_use]
pub fn gpu_required_from(value: Option<&str>) -> bool {
    match value.map(str::trim) {
        None | Some("" | "0") => false,
        Some(v) => !v.eq_ignore_ascii_case("false"),
    }
}

/// Is a GPU required for this test run (`APR_REQUIRE_GPU` armed)?
#[must_use]
pub fn gpu_required() -> bool {
    gpu_required_from(std::env::var(REQUIRE_GPU_ENV).ok().as_deref())
}

/// The refusal text, kept in one place so the tests pin it.
#[must_use]
pub fn refusal(what: &str, reason: &dyn Debug) -> String {
    format!("{REQUIRE_GPU_ENV} is set but the {what} cannot run: {reason:?}")
}

/// [`skip_or_panic`] with the decision injected, so the policy is testable
/// without touching the process environment.
///
/// # Panics
///
/// Panics (after printing the skip line) when `required` is true.
pub fn skip_or_panic_with(required: bool, what: &str, reason: &dyn Debug) {
    eprintln!("Skipping {what}: {reason:?}");
    assert!(!required, "{}", refusal(what, reason));
}

/// The one place a CUDA test's "no device" skip is decided.
///
/// Prints `Skipping <what>: <reason>` and returns, so the caller can `return`;
/// with `APR_REQUIRE_GPU` armed it panics instead, which fails the test.
///
/// # Panics
///
/// Panics when `APR_REQUIRE_GPU` is armed (see [`gpu_required_from`]).
pub fn skip_or_panic(what: &str, reason: &dyn Debug) {
    skip_or_panic_with(gpu_required(), what, reason);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unset_empty_zero_and_false_allow_skipping() {
        for v in [
            None,
            Some(""),
            Some("  "),
            Some("0"),
            Some("false"),
            Some("FALSE"),
            Some("False"),
        ] {
            assert!(!gpu_required_from(v), "{v:?} must leave skipping allowed");
        }
    }

    #[test]
    fn one_and_any_other_value_arm_the_check() {
        for v in ["1", " 1 ", "true", "TRUE", "yes", "on", "2", "anything"] {
            assert!(gpu_required_from(Some(v)), "{v:?} must arm the check");
        }
    }

    #[test]
    fn a_skip_path_returns_when_no_gpu_is_required() {
        skip_or_panic_with(false, "CUDA test", &"no device");
    }

    #[test]
    #[should_panic(expected = "APR_REQUIRE_GPU is set but the CUDA test cannot run")]
    fn a_skip_path_panics_when_a_gpu_is_required() {
        skip_or_panic_with(true, "CUDA test", &"no device");
    }

    #[test]
    fn the_refusal_names_the_switch_the_test_and_the_reason() {
        let text = refusal("CUDA test", &"no device");
        assert_eq!(
            text,
            "APR_REQUIRE_GPU is set but the CUDA test cannot run: \"no device\""
        );
    }

    /// Canary: with the switch armed a CUDA context must exist. Paths that skip
    /// without going through [`skip_or_panic`] cannot pass on a box with no GPU,
    /// because this test is in the same run.
    #[cfg(feature = "cuda")]
    #[test]
    fn an_armed_switch_means_a_cuda_context_exists() {
        if gpu_required() {
            if let Err(e) = crate::driver::CudaContext::new(0) {
                panic!("{}", refusal("CUDA context canary", &e));
            }
        }
    }
}
