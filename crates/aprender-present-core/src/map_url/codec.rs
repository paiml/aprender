//! `encode` and `decode` for the `k=v&...` map URL hash.
//!
//! RED stub: the real codec lands in the GREEN commit. These bodies compile
//! and return the wrong answer, so the FALSIFY tests fail on assertions.

use super::state::{MapUrlError, MapUrlState};

/// Encodes `state` as the hash body, without the leading `#`.
///
/// # Errors
/// Returns an error for a non-finite camera, an empty layer name, or a bad key.
pub fn encode(_state: &MapUrlState) -> Result<String, MapUrlError> {
    Ok(String::new())
}

/// Decodes a hash, with or without the leading `#`.
///
/// # Errors
/// Returns an error for a malformed pair, a duplicate key, bad percent
/// escapes, bad UTF-8, or a non-finite or malformed camera.
pub fn decode(_hash: &str) -> Result<MapUrlState, MapUrlError> {
    Ok(MapUrlState::default())
}
