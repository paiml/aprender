//! Test-only support shared by unit tests (#4130).

/// A repo-root `contracts/<rel>` file, read at RUN time, or `None` (after printing why) when this build
/// is not in the aprender workspace.
///
/// Some unit tests pin the code to a contract under the repository's `contracts/`. They used
/// `include_str!("../../../../contracts/…")`, a path outside this crate, so `cargo test` from the
/// published .crate could not compile them (#4130, measured by `scripts/package_tarball_build.sh`, #4114).
///
/// The in-tree decision is not made here: it is the one shared rule,
/// `provable_contracts::workspace_file_or_skip!` (#4175). In tree a missing or unreadable file PANICS;
/// out of tree it prints `SKIP <test>: out of tree …` and returns `None`, and the caller returns.
pub(crate) fn workspace_contract_or_skip(test: &str, rel: &str) -> Option<String> {
    provable_contracts::workspace_file_or_skip!(test, &format!("contracts/{rel}"))
}

#[cfg(test)]
mod tests {
    /// The real workspace resolves: this build IS in tree, so the helper must find the contract.
    #[test]
    fn this_build_is_in_tree() {
        assert!(
            super::workspace_contract_or_skip("this_build_is_in_tree", "setfit-apr-v1.yaml")
                .is_some()
        );
    }

    /// In tree with the file MISSING (renamed, deleted): a failure, never a skip.
    #[test]
    #[should_panic(expected = "in tree")]
    fn in_tree_missing_file_panics() {
        let _ = super::workspace_contract_or_skip("in_tree_missing_file_panics", "renamed.yaml");
    }
}
