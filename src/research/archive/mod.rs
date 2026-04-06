//! Archive Deposit for Zenodo/figshare (ENT-027)
//!
//! Provides deposit functionality for academic archives like
//! Zenodo, Figshare, Dryad, and Dataverse.

use regex::Regex;
use std::sync::LazyLock;

pub mod config;
pub mod deposit;
pub mod identifiers;
pub mod metadata;
pub mod provider;
pub mod result;

#[cfg(test)]
mod tests;

/// DOI validation pattern
pub(crate) static DOI_REGEX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^10\.\d{4,}/[^\s]+$").expect("Invalid DOI regex"));

// Re-export all public types
pub use config::{FigshareConfig, ZenodoConfig};
pub use deposit::ArchiveDeposit;
pub use identifiers::{IdentifierScheme, RelatedIdentifier, RelationType};
pub use metadata::{DepositMetadata, ResourceType};
pub use provider::ArchiveProvider;
pub use result::{validate_doi, DepositError, DepositResult};
