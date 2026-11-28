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

/// Bounds-checked little-endian reader. Every read either succeeds inside the input or
/// returns [`CodecError::Truncated`]; nothing indexes past the end.
struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn truncated(&self) -> CodecError {
        CodecError::Truncated {
            have: self.buf.len(),
        }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], CodecError> {
        let end = self.pos.checked_add(n).ok_or_else(|| self.truncated())?;
        let slice = self
            .buf
            .get(self.pos..end)
            .ok_or_else(|| self.truncated())?;
        self.pos = end;
        Ok(slice)
    }

    fn array<const N: usize>(&mut self) -> Result<[u8; N], CodecError> {
        let slice = self.take(N)?;
        <[u8; N]>::try_from(slice).map_err(|_| self.truncated())
    }

    fn u8(&mut self) -> Result<u8, CodecError> {
        let [byte] = self.array::<1>()?;
        Ok(byte)
    }

    fn u32(&mut self) -> Result<u32, CodecError> {
        Ok(u32::from_le_bytes(self.array()?))
    }

    fn u64(&mut self) -> Result<u64, CodecError> {
        Ok(u64::from_le_bytes(self.array()?))
    }

    fn f32(&mut self) -> Result<f32, CodecError> {
        Ok(f32::from_le_bytes(self.array()?))
    }

    /// Succeed only if every input byte was consumed.
    fn finish(self) -> Result<(), CodecError> {
        match self.buf.len().saturating_sub(self.pos) {
            0 => Ok(()),
            rest => Err(CodecError::TrailingBytes(rest)),
        }
    }
}

/// Element count as the wire `u32`. A batch of more than `u32::MAX` changes cannot be framed.
fn wire_count(len: usize) -> u32 {
    u32::try_from(len).expect("frame holds at most u32::MAX entries")
}

/// Encode a Hello frame.
#[must_use]
pub fn encode_hello(last_seen_seq: u64) -> Vec<u8> {
    last_seen_seq.to_le_bytes().to_vec()
}

/// Decode a Hello frame into its `last_seen_seq`.
///
/// # Errors
/// [`CodecError::Truncated`] if shorter than 8 bytes, [`CodecError::TrailingBytes`] if longer.
pub fn decode_hello(bytes: &[u8]) -> Result<u64, CodecError> {
    let mut reader = Reader::new(bytes);
    let last_seen_seq = reader.u64()?;
    reader.finish()?;
    Ok(last_seen_seq)
}

/// Encode a Batch frame.
#[must_use]
pub fn encode_batch(batch: &Batch) -> Vec<u8> {
    let mut out = Vec::with_capacity(1 + 8 + 4 + 8 * batch.changes.len());
    out.push(TAG_BATCH);
    out.extend_from_slice(&batch.seq.to_le_bytes());
    out.extend_from_slice(&wire_count(batch.changes.len()).to_le_bytes());
    for &(cell, value) in &batch.changes {
        out.extend_from_slice(&cell.to_le_bytes());
        out.extend_from_slice(&value.to_le_bytes());
    }
    out
}

/// Encode a Snapshot frame.
#[must_use]
pub fn encode_snapshot(seq: u64, hash: u64, values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(1 + 8 + 8 + 4 + 4 * values.len());
    out.push(TAG_SNAPSHOT);
    out.extend_from_slice(&seq.to_le_bytes());
    out.extend_from_slice(&hash.to_le_bytes());
    out.extend_from_slice(&wire_count(values.len()).to_le_bytes());
    for value in values {
        out.extend_from_slice(&value.to_le_bytes());
    }
    out
}

/// Decode a server-to-client frame (Batch or Snapshot).
///
/// # Errors
/// [`CodecError::Truncated`], [`CodecError::UnknownTag`] or [`CodecError::TrailingBytes`].
pub fn decode_server_frame(bytes: &[u8]) -> Result<ServerFrame, CodecError> {
    let mut reader = Reader::new(bytes);
    let frame = match reader.u8()? {
        TAG_BATCH => {
            let seq = reader.u64()?;
            let n = reader.u32()?;
            let mut changes = Vec::new();
            for _ in 0..n {
                let cell = reader.u32()?;
                let value = reader.f32()?;
                changes.push((cell, value));
            }
            ServerFrame::Batch(Batch { seq, changes })
        },
        TAG_SNAPSHOT => {
            let seq = reader.u64()?;
            let hash = reader.u64()?;
            let n = reader.u32()?;
            let mut values = Vec::new();
            for _ in 0..n {
                values.push(reader.f32()?);
            }
            ServerFrame::Snapshot { seq, hash, values }
        },
        tag => return Err(CodecError::UnknownTag(tag)),
    };
    reader.finish()?;
    Ok(frame)
}
