//! Data bundle manifest, part verification and lazy tables (MAP-4).
//!
//! This is the target-neutral slice of the bundle format: a canonical sha256
//! manifest over a set of named parts, verification of each part against it,
//! and a lazy-table loader that fetches a lazy part on first use, verifies it,
//! and caches it. Parquet encoding, the CLI builder and browser loading are not
//! part of this module.

pub mod lazy;
pub mod manifest;

pub use lazy::LazyTables;
pub use manifest::{Manifest, PartEntry, PartKind};

/// Every failure the bundle manifest and lazy loader can report.
///
/// Variants that concern a data part carry its name so the caller can show
/// which part was refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BundleError {
    /// A part was given an empty name.
    EmptyName,
    /// A part name contains a control character (tab, newline, ...).
    InvalidName {
        /// The rejected name.
        name: String,
    },
    /// The same part name appears more than once.
    DuplicatePart {
        /// The repeated name.
        part: String,
    },
    /// A decoded manifest lists this part out of name order.
    UnsortedPart {
        /// The part that sorts before its predecessor.
        part: String,
    },
    /// No part with this name is in the manifest.
    UnknownPart {
        /// The requested name.
        part: String,
    },
    /// A lazy fetch was requested for a part the manifest marks eager.
    EagerPart {
        /// The eager part.
        part: String,
    },
    /// The bytes of a part are not the manifest's length.
    LengthMismatch {
        /// The part that was checked.
        part: String,
        /// Length recorded in the manifest.
        expected: u64,
        /// Length of the bytes presented.
        actual: u64,
    },
    /// The bytes of a part do not hash to the manifest's sha256.
    HashMismatch {
        /// The part that was checked.
        part: String,
    },
    /// The manifest's self-digest does not match its bytes.
    ManifestDigest {
        /// Digest recorded in the manifest.
        expected: String,
        /// Digest computed over the manifest bytes.
        actual: String,
    },
    /// The first line is not the v1 manifest header.
    BadHeader {
        /// The first line as found.
        found: String,
    },
    /// A manifest line has the wrong shape.
    BadLine {
        /// 1-based line number.
        line: usize,
        /// What is wrong with the line.
        reason: &'static str,
    },
    /// A manifest line carries a hash that is not 64 lowercase hex digits.
    BadHex {
        /// 1-based line number.
        line: usize,
    },
    /// The manifest is not complete (no trailing newline, no digest line).
    Malformed {
        /// What is missing.
        reason: &'static str,
    },
}

impl std::fmt::Display for BundleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyName => write!(f, "bundle part name must not be empty"),
            Self::InvalidName { name } => {
                write!(f, "bundle part name {name:?} contains a control character")
            }
            Self::DuplicatePart { part } => {
                write!(f, "bundle part `{part}` is listed more than once")
            }
            Self::UnsortedPart { part } => {
                write!(f, "bundle part `{part}` is out of name order")
            }
            Self::UnknownPart { part } => write!(f, "bundle has no part `{part}`"),
            Self::EagerPart { part } => {
                write!(f, "bundle part `{part}` is eager and is not fetched lazily")
            }
            Self::LengthMismatch { part, expected, actual } => {
                write!(f, "bundle part `{part}`: manifest records {expected} bytes, got {actual}")
            }
            Self::HashMismatch { part } => {
                write!(f, "bundle part `{part}`: sha256 does not match the manifest")
            }
            Self::ManifestDigest { expected, actual } => {
                write!(f, "bundle manifest digest mismatch: recorded {expected}, computed {actual}")
            }
            Self::BadHeader { found } => {
                write!(f, "bundle manifest header is {found:?}, expected {:?}", manifest::HEADER)
            }
            Self::BadLine { line, reason } => {
                write!(f, "bundle manifest line {line}: {reason}")
            }
            Self::BadHex { line } => {
                write!(f, "bundle manifest line {line}: hash is not 64 lowercase hex digits")
            }
            Self::Malformed { reason } => write!(f, "bundle manifest is malformed: {reason}"),
        }
    }
}

impl std::error::Error for BundleError {}
