//! Structural validation — the Structure-category half of the validator split
//! (issue #2231 Stage 1 spike, part b).
//!
//! These checks operate on **bytes only** (magic, version, header size, flags)
//! and have NO dependency on `f32` tensor data. They are the cleanly-separable
//! subset of `aprender-core/src/format/validation_impl.rs`'s `Category::Structure`
//! checks (`check_magic`, `check_gguf_version`, `check_header_size`,
//! `check_version`, `check_flags`, `AprHeader::parse`).
//!
//! The Physics-category checks (`check_no_nan`, `check_no_inf`, `validate_tensors`,
//! `TensorStats::compute`) need `&[f32]` and are deliberately NOT moved here —
//! they stay in `aprender-core` (decision: converter/physics stay in core).

use crate::types::{Header, HEADER_SIZE, MAGIC};

/// Outcome of a single structural check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StructureCheck {
    /// The check passed.
    Pass,
    /// The check failed (with a static reason).
    Fail,
}

impl StructureCheck {
    /// True iff this check passed.
    #[must_use]
    pub fn is_pass(self) -> bool {
        matches!(self, StructureCheck::Pass)
    }
}

/// Check that the leading bytes carry the v1 `APRN` magic.
#[must_use]
pub fn check_magic(data: &[u8]) -> StructureCheck {
    if data.len() >= 4 && data[0..4] == MAGIC {
        StructureCheck::Pass
    } else {
        StructureCheck::Fail
    }
}

/// Check that the file is at least one full header in size.
#[must_use]
pub fn check_header_size(data: &[u8]) -> StructureCheck {
    if data.len() >= HEADER_SIZE {
        StructureCheck::Pass
    } else {
        StructureCheck::Fail
    }
}

/// Check that a parsed header carries a supported major version.
#[must_use]
pub fn check_version(header: &Header) -> StructureCheck {
    if header.version.0 <= crate::types::FORMAT_VERSION.0 {
        StructureCheck::Pass
    } else {
        StructureCheck::Fail
    }
}

/// Check that the header's flags byte parses (reserved high bit clear).
#[must_use]
pub fn check_flags(header: &Header) -> StructureCheck {
    // `Flags::from_bits` masks the reserved bit; a round-trip that drops bits
    // signals a dirty reserved bit.
    if header.flags.bits() & 0b1000_0000 == 0 {
        StructureCheck::Pass
    } else {
        StructureCheck::Fail
    }
}

/// Run all structural (byte-only) checks against a candidate `.apr` buffer.
///
/// Returns `true` iff the structure is well-formed. Performs NO `f32` / tensor
/// physics validation — that is the responsibility of `aprender-core`.
#[must_use]
pub fn validate_structure(data: &[u8]) -> bool {
    if !check_magic(data).is_pass() || !check_header_size(data).is_pass() {
        return false;
    }
    match Header::from_bytes(&data[..HEADER_SIZE]) {
        Ok(header) => check_version(&header).is_pass() && check_flags(&header).is_pass(),
        Err(_) => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Header, ModelType};

    fn good_header_bytes() -> Vec<u8> {
        let mut v = Header::new(ModelType::LinearRegression).to_bytes().to_vec();
        v.resize(HEADER_SIZE, 0);
        v
    }

    #[test]
    fn test_check_magic_pass_and_fail() {
        assert!(check_magic(&good_header_bytes()).is_pass());
        assert!(!check_magic(b"GGUF").is_pass());
    }

    #[test]
    fn test_validate_structure_round_trip() {
        assert!(validate_structure(&good_header_bytes()));
        let mut bad = good_header_bytes();
        bad[0] = 0x00; // corrupt magic
        assert!(!validate_structure(&bad));
    }
}
