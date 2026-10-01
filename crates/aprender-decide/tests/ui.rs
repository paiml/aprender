//! Compile-fail proofs for the aprender-decide public API (trybuild).
//!
//! FALSIFY-DECIDE-APR-006 / decide-apr-v1 rung 8: a [`aprender_decide::Decider`] can be
//! minted only by the load ladder (`Decider::load_bytes` / `Decider::load_path`). Each
//! case under `tests/ui/` tries a second minting path and must fail to compile with the
//! error recorded in its `.stderr`.
//!
//! This integration target runs in CI: the fragment
//! `ci/explicit-test-commands.d/510-aprender-decide-ui.cmd` carries
//! `cargo test -p aprender-decide --test ui`, run by the workspace-test "Integration tests"
//! step (added to ci.yml's explicit line by plan 08-12; header corrected by plan 08-31,
//! IN-04; re-homed to upstream's fragment directory by the 08-32 upstream merge).

#[test]
fn decider_has_no_second_minting_path() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/ui/*.rs");
}
