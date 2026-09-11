//! Fixture crate for scripts/check_tree_reader_tests.sh --self-test. NOT a
//! workspace member and deliberately without a Cargo.toml: the derivation
//! reads .rs text, so cargo never sees this tree.
pub mod deep;
pub mod inc;

/// A reader in the crate ROOT module (src/lib.rs) -> the row's module is `<root>`.
pub fn root_reader() -> bool {
    std::fs::read_to_string("scripts/root_baseline.txt").is_ok()
}

#[cfg(test)]
mod tests {
    #[test]
    fn root_module_reads_the_tree() {
        assert!(!std::fs::read_to_string("scripts/root_baseline.txt").is_err() || true);
    }
}
