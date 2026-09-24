//! Test-only support shared by unit tests (#4130).

/// A repo-root `contracts/<rel>` file, read at RUN time, or `None` (after printing why) when this build
/// has no workspace at all.
///
/// Some unit tests pin the code to a contract under the repository's `contracts/`. They used
/// `include_str!("../../../../contracts/…")`, a path outside this crate, so `cargo test` from the
/// published .crate could not compile them (#4130, measured by `scripts/package_tarball_build.sh`, #4114).
///
/// The semantics are the ones aprender-contracts' `schema::workspace_contract_or_skip` (#4129) and
/// aprender-serve (#4048) use, so one pattern covers every crate:
/// - IN TREE (the workspace `contracts/` beside this crate): a missing or unreadable file PANICS. A
///   renamed contract is a failure, never a skip.
/// - OUT OF TREE (no `contracts/` at all, i.e. the crates.io tarball): prints `SKIP <test>: out of tree …`
///   at column 0 on stderr, which the tarball gate counts, and returns `None`. The caller returns.
pub(crate) fn workspace_contract_or_skip(test: &str, rel: &str) -> Option<String> {
    contract_at_or_skip(
        &std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../.."),
        test,
        rel,
    )
}

/// [`workspace_contract_or_skip`] against an explicit repository root, so both arms are testable.
fn contract_at_or_skip(root: &std::path::Path, test: &str, rel: &str) -> Option<String> {
    if !root.join("contracts").is_dir() {
        eprintln!(
            "SKIP {test}: out of tree (no {} beside this crate) - contracts/{rel} lives in the \
             workspace, which a published crate does not carry (#4130)",
            root.join("contracts").display()
        );
        return None;
    }
    let path = root.join("contracts").join(rel);
    Some(
        std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("in tree, {} must be readable: {e}", path.display())),
    )
}

#[cfg(test)]
mod tests {
    use super::contract_at_or_skip;

    /// Out of tree (no `contracts/` under the root): the test is skipped, never failed.
    #[test]
    fn no_workspace_skips() {
        let dir = tempfile::tempdir().expect("tempdir");
        assert_eq!(contract_at_or_skip(dir.path(), "t", "x.yaml"), None);
    }

    /// In tree: the file's bytes, read at run time.
    #[test]
    fn in_tree_reads_the_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::create_dir(dir.path().join("contracts")).expect("mkdir");
        std::fs::write(dir.path().join("contracts/x.yaml"), "k: v\n").expect("write");
        assert_eq!(
            contract_at_or_skip(dir.path(), "t", "x.yaml").as_deref(),
            Some("k: v\n")
        );
    }

    /// In tree with the file MISSING (renamed, deleted): a failure, never a skip — the one outcome that
    /// separates this from a silent guard.
    #[test]
    #[should_panic(expected = "in tree")]
    fn in_tree_missing_file_panics() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::create_dir(dir.path().join("contracts")).expect("mkdir");
        let _ = contract_at_or_skip(dir.path(), "t", "renamed.yaml");
    }

    /// The real workspace resolves: this build IS in tree, so the helper must find `contracts/`.
    #[test]
    fn this_build_is_in_tree() {
        assert!(
            super::workspace_contract_or_skip("this_build_is_in_tree", "setfit-apr-v1.yaml")
                .is_some()
        );
    }
}
