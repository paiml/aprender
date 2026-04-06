//! Test helpers for crates_io module tests.

use crate::stack::crates_io::types::{CrateData, CrateResponse, VersionData};

/// Create a simple CrateResponse for testing (no versions, no stable).
pub fn make_response(name: &str, version: &str) -> CrateResponse {
    CrateResponse { krate: CrateData::new(name, version), versions: vec![] }
}

/// Create a full CrateResponse with stable version and version list.
pub fn make_full_response(
    name: &str,
    version: &str,
    description: Option<&str>,
    downloads: u64,
) -> CrateResponse {
    CrateResponse {
        krate: CrateData {
            max_stable_version: Some(version.to_string()),
            description: description.map(|s| s.to_string()),
            downloads,
            ..CrateData::new(name, version)
        },
        versions: vec![VersionData::new(version, downloads)],
    }
}
