//! Non-vacuity probe for the `provable-contracts-macros` facade.
//!
//! `compat/invoke.rs` proves the five 0.3.1 attribute macros still expand
//! through this facade. This file is the CONTROL for that claim: under
//! `--features __facade_probe_mutant` it names a macro the facade does not
//! export, and `scripts/check_facade_compat.sh --self-test` requires the build
//! to FAIL. A green-only corpus proves nothing.

#[cfg(feature = "__facade_probe_mutant")]
use provable_contracts_macros::this_macro_was_never_exported;

#[cfg(feature = "__facade_probe_mutant")]
#[this_macro_was_never_exported]
fn unreachable_mutant_arm() {}

#[test]
fn probe_target_is_wired() {
    // The green arm is deliberately trivial: `compat/invoke.rs` carries the
    // behavioural assertions. This target exists so the mutant arm has a home
    // that does not disturb the corpus.
    assert!(cfg!(not(feature = "__facade_probe_mutant")));
}
