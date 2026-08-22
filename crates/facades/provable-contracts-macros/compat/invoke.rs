//! COMPAT CORPUS for the macros facade.
//!
//! Re-exporting a proc macro and being able to INVOKE it through the re-export
//! are different claims: the attribute path has to resolve at expansion time,
//! and a `proc-macro = true` crate cannot forward someone else's macros at all.
//! This file invokes all five of `provable-contracts-macros 0.3.1`'s attribute
//! macros through the facade path and asserts the behaviour each one is
//! documented to inject. It is primarily a COMPILE-time assertion.
//!
//! The five, verbatim from 0.3.1's `#[proc_macro_attribute]` list:
//! `contract`, `requires`, `ensures`, `invariant`, `must_contract`.
//!
//! Non-vacuity control: `scripts/check_facade_compat.sh --mutate` breaks the
//! facade's re-export and requires this target to go RED. A corpus that has
//! only ever been green proves nothing.

use provable_contracts_macros::{contract, ensures, invariant, must_contract, requires};

/// `#[contract]` — the flagship. Binds a function to a YAML contract equation
/// and injects `debug_assert!`s from build-script env vars. With no env vars
/// set it degrades to `option_env!`, so it compiles standalone here.
#[contract("facade-compat-v1", equation = "identity")]
fn contract_target(x: i32) -> i32 {
    x
}

/// `#[requires]` — precondition, `debug_assert!`ed at entry.
#[requires(x > 0)]
fn requires_target(x: i32) -> i32 {
    x
}

/// `#[ensures]` — postcondition. 0.3.1 binds the return value to `ret`; that
/// binding name is part of the compatibility surface, so it is asserted here.
#[ensures(ret >= 0)]
fn ensures_target(x: i32) -> i32 {
    x.abs()
}

/// `#[invariant]` — checked before AND after. Applies to a fn, not a type.
#[invariant(n > 0)]
fn invariant_target(n: i32) -> i32 {
    n
}

/// `#[must_contract]` — marks an unbound `pub fn`, emitting `#[deprecated]`
/// when no `CONTRACT_*` env var names it. The deprecation IS the macro working.
#[must_contract]
pub fn must_contract_target(x: i32) -> i32 {
    x
}

#[test]
fn all_five_0_3_1_attribute_macros_resolve_and_expand_through_the_facade() {
    assert_eq!(contract_target(1), 1);
    assert_eq!(requires_target(1), 1);
    assert_eq!(ensures_target(-1), 1);
    assert_eq!(invariant_target(1), 1);
    #[allow(deprecated)] // the deprecation is `must_contract` doing its job
    {
        assert_eq!(must_contract_target(1), 1);
    }
}

/// `#[requires]` must EXCLUDE an outcome, not merely compile. A debug build
/// panics when the precondition is false; if the facade forwarded a no-op
/// macro this test would fail rather than pass silently.
#[test]
#[should_panic(expected = "Pre-condition violated")]
#[cfg(debug_assertions)]
fn requires_still_rejects_a_violating_input_through_the_facade() {
    let _ = requires_target(-1);
}
