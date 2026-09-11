//! An integration reader: rows for these are unchanged (crate --test it).
#[test]
fn integration_reads_the_tree() {
    let _ = std::fs::read_to_string("docs/specifications/ci-fleet-hygiene.md");
}
