//! MAP-8 binary frame codec (PMAT-7553, epic #7507). Target-neutral: no socket, no axum.
//!
//! Every value is little-endian and no JSON is involved, so `f32` values
//! round-trip bit-exactly and [`super::column_hash`] stays comparable across the wire.
//!
//! Layouts:
//! - Hello (client to server): `u64 last_seen_seq`.
//! - Batch (server to client): tag `0`, `u64 seq`, `u32 n`, then `n` times `(u32 cell, f32 value)`.
//! - Snapshot (server to client): tag `1`, `u64 seq`, `u64 hash`, `u32 n`, then `n` times `f32`.
//!
//! Decoding never panics: truncated input, unknown tags and trailing bytes are [`CodecError`].

use super::Batch;

/// Wire tag of a Batch frame.
pub const TAG_BATCH: u8 = 0;
/// Wire tag of a Snapshot frame.
pub const TAG_SNAPSHOT: u8 = 1;

/// A decoded server-to-client frame.
#[derive(Debug, Clone, PartialEq)]
pub enum ServerFrame {
    /// One change batch.
    Batch(Batch),
    /// A full column at `seq`, with its [`super::column_hash`].
    Snapshot {
        /// Column sequence number the values correspond to.
        seq: u64,
        /// Hash the server computed over `values`.
        hash: u64,
        /// Full column values.
        values: Vec<f32>,
    },
}

/// Errors from decoding a frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CodecError {
    /// The input ended before the layout said it would.
    Truncated {
        /// Length of the input in bytes.
        have: usize,
    },
    /// The first byte is neither [`TAG_BATCH`] nor [`TAG_SNAPSHOT`].
    UnknownTag(u8),
    /// The layout ended but input bytes remain.
    TrailingBytes(usize),
}

impl std::fmt::Display for CodecError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Truncated { have } => write!(f, "truncated frame ({have} bytes)"),
            Self::UnknownTag(tag) => write!(f, "unknown frame tag {tag}"),
            Self::TrailingBytes(n) => write!(f, "{n} trailing bytes after frame"),
        }
    }
}

impl std::error::Error for CodecError {}

/// Encode a Hello frame.
#[must_use]
pub fn encode_hello(last_seen_seq: u64) -> Vec<u8> {
    todo!("RED: encode_hello({last_seen_seq})")
}

/// Decode a Hello frame into its `last_seen_seq`.
///
/// # Errors
/// [`CodecError::Truncated`] if shorter than 8 bytes, [`CodecError::TrailingBytes`] if longer.
pub fn decode_hello(bytes: &[u8]) -> Result<u64, CodecError> {
    todo!("RED: decode_hello({} bytes)", bytes.len())
}

/// Encode a Batch frame.
#[must_use]
pub fn encode_batch(batch: &Batch) -> Vec<u8> {
    todo!("RED: encode_batch(seq {})", batch.seq)
}

/// Encode a Snapshot frame.
#[must_use]
pub fn encode_snapshot(seq: u64, hash: u64, values: &[f32]) -> Vec<u8> {
    todo!(
        "RED: encode_snapshot({seq}, {hash}, {} values)",
        values.len()
    )
}

/// Decode a server-to-client frame (Batch or Snapshot).
///
/// # Errors
/// [`CodecError::Truncated`], [`CodecError::UnknownTag`] or [`CodecError::TrailingBytes`].
pub fn decode_server_frame(bytes: &[u8]) -> Result<ServerFrame, CodecError> {
    todo!("RED: decode_server_frame({} bytes)", bytes.len())
}
