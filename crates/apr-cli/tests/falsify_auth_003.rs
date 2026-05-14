//! FALSIFY-AUTH-003 — the auth module compares digests via
//! `subtle::ConstantTimeEq::ct_eq`, never via `==` or `[u8]::eq`.
//!
//! Contract: `contracts/apr-serve-api-key-auth-v1.yaml`.
//!
//! This is a structural source-code gate. Runtime timing tests are too
//! noisy to be CI-tractable; instead we assert that the source under
//! `crates/apr-cli/src/commands/serve/auth.rs`:
//!   1. imports `subtle::ConstantTimeEq`,
//!   2. calls `.ct_eq(...)` at least once,
//!   3. does NOT contain a literal `==` between two `[u8; 32]` digests.
//!
//! A drive-by refactor that switches to `==` falls foul of #3. The test
//! is intentionally strict about the file being present at the exact path
//! the contract names — moving or renaming the module is itself a
//! contract change.

#![allow(clippy::unwrap_used)]

const AUTH_SOURCE: &str = include_str!("../src/commands/serve/auth.rs");

#[test]
fn auth_module_imports_subtle_constanttimeeq() {
    assert!(
        AUTH_SOURCE.contains("use subtle::ConstantTimeEq"),
        "auth.rs must `use subtle::ConstantTimeEq` — required by FALSIFY-AUTH-003.\n\
         If the import was renamed, update the contract before this test.",
    );
}

#[test]
fn auth_module_calls_ct_eq() {
    assert!(
        AUTH_SOURCE.contains(".ct_eq("),
        "auth.rs must call `.ct_eq(...)` somewhere — required by FALSIFY-AUTH-003.\n\
         If the comparison was extracted to a helper, that helper must \
         live in this module so this gate keeps catching regressions.",
    );
}

#[test]
fn auth_module_does_not_compare_digests_with_plain_eq() {
    // We can't ban every `==` in the file (false positives in tests, etc.),
    // but we CAN assert that no line of source compares an `expected` with a
    // `presented` digest via `==`. The patterns below are the exact shapes
    // a regression would take.
    let banned_patterns = [
        "expected == presented",
        "presented == expected",
        "expected.eq(&presented)",
        "presented.eq(&expected)",
        "*expected == *presented",
        "*presented == *expected",
    ];
    for pat in banned_patterns {
        assert!(
            !AUTH_SOURCE.contains(pat),
            "auth.rs must NOT contain `{pat}` — that would be a non-constant-time \
             comparison and break FALSIFY-AUTH-003. Use `expected.ct_eq(&presented)`.",
        );
    }
}

#[test]
fn auth_module_path_matches_contract_reference() {
    // If the file moves, this test stops compiling (include_str!) — that's
    // by design. The contract's `references:` list points at this exact
    // path; a rename without contract update would fail the workspace
    // contract integration test.
    assert!(!AUTH_SOURCE.is_empty());
    assert!(AUTH_SOURCE.contains("HELIX-IDEA-009"));
}
