//! Research Artifact and Author structs (ENT-019)
//!
//! Provides core types for academic research artifacts with proper
//! attribution using CRediT taxonomy, ORCID, and ROR identifiers.

mod affiliation;
mod artifact_type;
mod author;
mod error;
mod license;
mod research_artifact;
mod role;

#[cfg(test)]
mod tests;

use regex::Regex;
use std::sync::LazyLock;

pub use affiliation::Affiliation;
pub use artifact_type::ArtifactType;
pub use author::Author;
pub use error::ValidationError;
pub use license::License;
pub use research_artifact::ResearchArtifact;
pub use role::ContributorRole;

/// ORCID validation pattern: 0000-0000-0000-000X
static ORCID_REGEX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^(\d{4}-){3}\d{3}[\dX]$").expect("Invalid ORCID regex"));

/// ROR ID validation pattern: https://ror.org/xxxxxxxxx
static ROR_REGEX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^https://ror\.org/[a-z0-9]{9}$").expect("Invalid ROR regex"));

/// Validate ORCID format: 0000-0000-0000-000X
pub fn validate_orcid(orcid: &str) -> bool {
    ORCID_REGEX.is_match(orcid)
}

/// Validate ROR ID format: <https://ror.org/xxxxxxxxx>
pub fn validate_ror_id(ror_id: &str) -> bool {
    ROR_REGEX.is_match(ror_id)
}
