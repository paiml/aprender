//! The crate's one sha256 helper (plan 08-31, S1).
//!
//! Every digest this crate computes — the D-11 artifact identity, the task's sha256, the
//! recipe_id, the pack-time input hashes and the verifier's re-derivations — goes through
//! [`sha256_hex`]. It lives in its own module so the LOAD path (`artifact`, `task`) hashes
//! without importing anything from [`crate::pack`]; `pack` and `verify` call it too.
//! [`crate::artifact::artifact_sha256_hex`] stays as the named D-11 identity and calls it.

use sha2::{Digest, Sha256};

/// Lowercase-hex sha256 of `bytes`.
#[must_use]
pub fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

#[cfg(test)]
mod tests {
    use super::sha256_hex;

    /// The FIPS 180-2 "abc" vector and the empty-input digest: the helper is sha256, not
    /// some other digest that happens to be 64 hex chars.
    #[test]
    fn sha256_hex_matches_known_vectors() {
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }
}
