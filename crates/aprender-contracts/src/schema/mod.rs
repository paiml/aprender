pub mod artifact;
pub mod composition;
pub mod external_corpora;
pub mod kaizen;
mod kind;
mod parser;
mod types;
mod validator;

pub use artifact::{classify_artifact, validate_artifact, ArtifactKind};
pub use external_corpora::{
    is_external_corpora_schema, parse_external_corpora_str, validate_external_corpora,
    ExternalCorpora, ExternalCorpus,
};
pub use parser::{is_contract_yaml, parse_contract, parse_contract_str};
pub use types::*;
pub use validator::validate_contract;

/// A contract from the workspace's `contracts/` directory, read at RUN time, for tests (#4129).
///
/// Several tests here used `include_str!("../../../../contracts/…")`, which is a path OUTSIDE this
/// crate, so `cargo test` from the published aprender-contracts tarball could not compile at all.
/// In tree, a missing or renamed contract FAILS the test (panic). Only a build with no
/// `contracts/` directory beside the crate (the crates.io tarball) returns `None`, after
/// printing which test skipped and why.
#[cfg(test)]
pub(crate) fn workspace_contract_or_skip(test: &str, rel: &str) -> Option<String> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    if !root.join("contracts").is_dir() {
        eprintln!(
            "SKIP {test}: out of tree (no {} beside this crate) - contracts/{rel} lives in the \
             workspace, which a published crate does not carry (#4129)",
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
