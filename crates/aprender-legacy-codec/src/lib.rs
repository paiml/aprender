//! Byte-compatible replacement for `bincode` 1.x `serialize` / `deserialize`
//! (`DefaultOptions`-free legacy layout: fixed-width integers, little-endian).
//!
//! The upstream crate is unmaintained (RUSTSEC-2025-0141, no patched version, and
//! bincode 2 is covered by the same advisory), while `.apr` payloads, cached indexes
//! and replay files already on disk were written by 1.x. This crate keeps those bytes
//! readable and writable without the dependency. Contract:
//! `contracts/aprender-legacy-codec-v1.yaml`.
//!
//! Layout (identical to bincode 1.3 `bincode::serialize`):
//! * `bool` = 1 byte (0/1); integers and floats = fixed width, little-endian;
//!   `char` = its UTF-8 bytes; `usize`/`isize` are not serde primitives (serde widens to u64).
//! * `str`, byte strings, sequences, maps = `u64` length, then the elements.
//! * `Option` = 1 tag byte (0 = `None`, 1 = `Some`) then the value.
//! * enum = `u32` variant index, then the payload.
//! * structs, tuples, newtypes, unit = fields in order, no names, no padding.
//!
//! Self-describing requests (`deserialize_any`, identifiers) are rejected, exactly as 1.x did.
#![forbid(unsafe_code)]

mod de;
mod ser;

use serde::de::DeserializeOwned;
use serde::Serialize;
use std::fmt;
use std::io::{Read, Write};

/// Codec error. Mirrors the information bincode's `ErrorKind` carried, as one type.
#[derive(Debug)]
pub enum Error {
    /// Underlying reader/writer failed (including unexpected end of input).
    Io(std::io::Error),
    /// A serde `custom` error from a `Serialize`/`Deserialize` impl.
    Custom(String),
    /// A `bool` byte other than 0 or 1.
    InvalidBool(u8),
    /// An `Option` tag byte other than 0 or 1.
    InvalidOptionTag(u8),
    /// A `char` that is not a valid scalar value.
    InvalidChar,
    /// A string whose bytes are not UTF-8.
    InvalidUtf8,
    /// A sequence or map whose length was unknown up front.
    SequenceMustHaveLength,
    /// `deserialize_any`/identifier: the format is not self-describing.
    NotSelfDescribing,
    /// A length prefix that does not fit this platform's `usize`.
    LengthOverflow(u64),
}

/// `Result` alias used across the crate.
pub type Result<T> = std::result::Result<T, Error>;

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "io error: {e}"),
            Self::Custom(m) => f.write_str(m),
            Self::InvalidBool(b) => write!(f, "invalid bool byte {b}, expected 0 or 1"),
            Self::InvalidOptionTag(b) => write!(f, "invalid Option tag {b}, expected 0 or 1"),
            Self::InvalidChar => f.write_str("invalid char encoding"),
            Self::InvalidUtf8 => f.write_str("string is not valid UTF-8"),
            Self::SequenceMustHaveLength => f.write_str("sequence length must be known up front"),
            Self::NotSelfDescribing => f.write_str("format is not self-describing"),
            Self::LengthOverflow(n) => write!(f, "length {n} does not fit usize"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl serde::ser::Error for Error {
    fn custom<T: fmt::Display>(msg: T) -> Self {
        Self::Custom(msg.to_string())
    }
}

impl serde::de::Error for Error {
    fn custom<T: fmt::Display>(msg: T) -> Self {
        Self::Custom(msg.to_string())
    }
}

/// Serialize `value` to a fresh `Vec<u8>` (drop-in for `bincode::serialize`).
pub fn serialize<T: Serialize + ?Sized>(value: &T) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    serialize_into(&mut out, value)?;
    Ok(out)
}

/// Serialize `value` into `writer` (drop-in for `bincode::serialize_into`).
pub fn serialize_into<W: Write, T: Serialize + ?Sized>(writer: W, value: &T) -> Result<()> {
    value.serialize(&mut ser::Serializer::new(writer))
}

/// Deserialize from a byte slice. Trailing bytes are ignored, as in `bincode::deserialize`.
pub fn deserialize<T: DeserializeOwned>(bytes: &[u8]) -> Result<T> {
    deserialize_from(bytes)
}

/// Deserialize from a reader, consuming only what the value needs (drop-in for
/// `bincode::deserialize_from`).
pub fn deserialize_from<R: Read, T: DeserializeOwned>(reader: R) -> Result<T> {
    T::deserialize(&mut de::Deserializer::new(reader))
}
