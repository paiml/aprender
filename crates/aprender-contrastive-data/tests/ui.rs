//! DATA-06 — "cannot be constructed" is checked by the COMPILER, not by a test.
//!
//! Obligation: `OBLIG-CPP-LEAKAGE-NOT-CONSTRUCTIBLE`.
//!
//! Every other gate in this crate observes a value and rejects it. These five observe that
//! there is no value: each `tests/ui/*.rs` is a complete program using only the crate's
//! PUBLIC API which must fail to compile, with the diagnostic pinned by a committed
//! `.stderr` snapshot. A runtime rejection can be caught and ignored by a caller; a
//! non-compiling program cannot.
//!
//! # Why this became writable only after the typestate was corrected
//!
//! Under the pre-review design `profile` was a runtime FIELD, so
//! `Split::<Validation>::from_jsonl_bytes(bytes, &compat_decl)` compiled and returned
//! `Err(ConflictingSourceRole)`. There was no compile error to snapshot, and the obligation
//! could not have been discharged as written (review V1 / adjudicated claim 1). Plan 02-03
//! made the profile a TYPE PARAMETER; cases 1-3 exist because of that change.
//!
//! # What a snapshot must contain to be evidence
//!
//! Each `.stderr` must name the crate's real types, methods or visibility —
//! `PreparedDataset<Compatibility>`, `PreparedDataset<Canonical>`, `validation_witness`,
//! `Selection`, `from_jsonl_bytes`. A snapshot showing a syntax error or an unresolved
//! import would be a red case that proves nothing about leakage, and it must be fixed
//! rather than blessed.
//!
//! # Snapshots are rustc-version sensitive, and that is understood
//!
//! `.stderr` files pin rustc's exact wording. A toolchain bump can reword a diagnostic and
//! turn this suite red without any behaviour change; the re-baseline command is
//!
//! ```text
//! TRYBUILD=overwrite cargo test -p aprender-contrastive-data --test ui
//! ```
//!
//! and the resulting diff MUST be reviewed against the list of names above before it is
//! committed. The assertions here are about types and visibility, so a legitimate reword
//! keeps every one of those names present.

#[test]
fn ui() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/ui/*.rs");
}
