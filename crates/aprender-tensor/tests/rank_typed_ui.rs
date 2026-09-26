//! FALSIFY-TENSOR-RANK-001: rank and layout mismatches do not compile (#3150).
//!
//! Each `tests/ui/*.rs` must FAIL to compile with the recorded diagnostic. If a
//! change makes one compile (say a blanket `matmul` for every rank), this goes
//! red. The `.stderr` files pin the pinned toolchain's wording; refresh them with
//! `TRYBUILD=overwrite cargo test -p aprender-tensor --test rank_typed_ui`
//! only after checking that the new diagnostic still names the same error.

#[test]
fn rank_and_layout_mismatches_do_not_compile() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/ui/*.rs");
}
