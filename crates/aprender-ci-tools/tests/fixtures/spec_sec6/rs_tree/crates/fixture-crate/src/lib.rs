//! Fixture crate for the §6 `rs:` surface.

#[test]
fn alpha_ok() {}

// spacer a
// spacer b
pub fn not_a_test() {}

#[test]
// one
// two
// three
fn too_far() {}

#[test]
#[ignore]
pub fn ignored_case() {}

mod nested;
