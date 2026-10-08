pub mod artifact;
mod clause;
pub mod composition;
pub mod external_corpora;
pub mod kaizen;
mod kind;
mod parser;
mod types;
mod validator;

pub use artifact::{classify_artifact, validate_artifact, ArtifactKind};
pub use clause::{Clause, FormalStatus};
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
/// The in-tree decision is the one shared rule in [`crate::tree`]: in tree a missing contract
/// FAILS, and only a build outside the aprender workspace returns `None` with a named SKIP.
#[cfg(test)]
pub(crate) fn workspace_contract_or_skip(test: &str, rel: &str) -> Option<String> {
    crate::tree::workspace_file_or_skip_at(
        test,
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")),
        &format!("contracts/{rel}"),
    )
}
