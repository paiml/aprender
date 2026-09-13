//! Byte format helpers for `apr trace --save-tensor`.
//!
//! Contract: [`contracts/apr-cli-trace-save-tensor-v1.yaml`] v1.0.0 (PROPOSED).
//!
//! ## File format (per contract `byte_format` equation)
//!
//! ```text
//! offset 0-3:   magic "APRT" (b"APRT")
//! offset 4-7:   u32 LE — layer index (0..num_layers-1; or WHOLE_MODEL_LAYER for
//!               whole-model stages such as final_norm and lm_head)
//! offset 8-11:  u32 LE — dim_product (number of f32 elements following)
//! offset 12+:   f32 LE × dim_product values
//! ```
//!
//! Total file size = `12 + dim_product × 4` bytes.
//!
//! ## Why a separate module
//!
//! Per `apr-cli-trace-save-tensor-v1` `apr_diff_values_compat` invariant: the
//! 12-byte header lets `apr diff --values` skip past it and read f32 LE bodies
//! directly. Keeping the format primitives as pure byte-in-byte-out helpers
//! (no I/O, no model state) lets the same code be reused by:
//!
//! 1. `apr trace --save-tensor` writer (this module's `write_*` functions).
//! 2. `apr diff --values` reader (this module's `parse_header` + `read_*`).
//! 3. Future Python/uv-run analysis scripts that load these binaries.
//!
//! ## Discharge status
//!
//! Partial-discharge of `FALSIFY-APR-TRACE-SAVE-004` (header self-describing
//! format) at the parser/serializer level. Full discharge requires the
//! `apr trace --save-tensor` CLI implementation that calls `write_tensor_file`
//! at the right capture points; that work is the next slice in the contract's
//! staged plan.

use std::io::{self, Read, Write};

/// Magic bytes at offset 0-3 of every save-tensor file.
///
/// Allows `apr diff --values` and downstream tools to detect the format
/// without external metadata.
pub const MAGIC: [u8; 4] = *b"APRT";

/// Total size of the fixed-length header in bytes (offset 0..12).
pub const HEADER_SIZE: usize = 12;

/// Layer-index sentinel for whole-model stages (e.g., `final_norm`, `lm_head`).
///
/// Per contract: per-layer stages use `0..num_layers-1`; whole-model stages
/// use `0xFFFFFFFF` so `apr diff --values` can recognize them.
pub const WHOLE_MODEL_LAYER: u32 = 0xFFFFFFFF;

/// Parsed header from a save-tensor file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TensorHeader {
    /// Layer index in `0..num_layers-1`, or [`WHOLE_MODEL_LAYER`].
    pub layer: u32,
    /// Number of f32 elements that follow the header.
    pub dim_product: u32,
}

impl TensorHeader {
    /// Whether this header refers to a whole-model stage (vs a per-layer stage).
    #[must_use]
    pub fn is_whole_model(&self) -> bool {
        self.layer == WHOLE_MODEL_LAYER
    }

    /// Total file size in bytes given this header.
    #[must_use]
    pub fn total_file_size(&self) -> usize {
        HEADER_SIZE + (self.dim_product as usize) * 4
    }
}

/// Errors that can arise parsing a save-tensor header.
#[derive(Debug, thiserror::Error)]
pub enum HeaderError {
    /// Less than `HEADER_SIZE` bytes were available.
    #[error("save-tensor header truncated: got {got} bytes, need {HEADER_SIZE}")]
    Truncated {
        /// Number of bytes that WERE available.
        got: usize,
    },
    /// Magic mismatch.
    #[error("save-tensor magic mismatch: got {got:?}, expected {MAGIC:?}")]
    BadMagic {
        /// The first 4 bytes that we read but did not match [`MAGIC`].
        got: [u8; 4],
    },
}

/// Errors that can arise reading a full save-tensor file.
#[derive(Debug, thiserror::Error)]
pub enum ReadError {
    /// Underlying I/O failed.
    #[error("save-tensor I/O error: {0}")]
    Io(#[from] io::Error),
    /// Header parse failed.
    #[error("save-tensor header: {0}")]
    Header(#[from] HeaderError),
    /// Body length didn't match `dim_product * 4`.
    #[error("save-tensor body length mismatch: header says {expected} bytes, got {got}")]
    BodyLengthMismatch {
        /// Number of f32 elements claimed by the header.
        expected: usize,
        /// Number of f32 elements actually decoded from the body.
        got: usize,
    },
}

/// Serialize a header into a 12-byte little-endian byte array.
///
/// This is a pure function — no I/O, no allocation beyond the returned array.
#[must_use]
pub fn write_header(layer: u32, dim_product: u32) -> [u8; HEADER_SIZE] {
    let mut buf = [0u8; HEADER_SIZE];
    buf[0..4].copy_from_slice(&MAGIC);
    buf[4..8].copy_from_slice(&layer.to_le_bytes());
    buf[8..12].copy_from_slice(&dim_product.to_le_bytes());
    buf
}

/// Parse a 12-byte (or longer) byte slice as a save-tensor header.
///
/// Reads exactly the first 12 bytes; ignores any trailing body content.
///
/// # Errors
///
/// Returns `Truncated` if `bytes.len() < HEADER_SIZE`, or `BadMagic` if the
/// first four bytes do not equal [`MAGIC`].
pub fn parse_header(bytes: &[u8]) -> Result<TensorHeader, HeaderError> {
    if bytes.len() < HEADER_SIZE {
        return Err(HeaderError::Truncated { got: bytes.len() });
    }
    let mut magic = [0u8; 4];
    magic.copy_from_slice(&bytes[0..4]);
    if magic != MAGIC {
        return Err(HeaderError::BadMagic { got: magic });
    }
    let mut layer_bytes = [0u8; 4];
    layer_bytes.copy_from_slice(&bytes[4..8]);
    let layer = u32::from_le_bytes(layer_bytes);

    let mut dp_bytes = [0u8; 4];
    dp_bytes.copy_from_slice(&bytes[8..12]);
    let dim_product = u32::from_le_bytes(dp_bytes);

    Ok(TensorHeader { layer, dim_product })
}

/// Write a complete save-tensor file: 12-byte header followed by f32 LE values.
///
/// `dim_product` is taken from `values.len()`. Preserves NaN values verbatim.
///
/// # Errors
///
/// Forwards any `io::Error` from the writer.
///
/// # Panics
///
/// Panics if `values.len() > u32::MAX` (would overflow the `dim_product` field).
pub fn write_tensor_file<W: Write>(w: &mut W, layer: u32, values: &[f32]) -> io::Result<()> {
    let dim_product = u32::try_from(values.len())
        .expect("save-tensor: values.len() exceeds u32::MAX (4 GiB elements)");
    w.write_all(&write_header(layer, dim_product))?;
    for &v in values {
        w.write_all(&v.to_le_bytes())?;
    }
    Ok(())
}

/// Read a complete save-tensor file: 12-byte header followed by f32 LE values.
///
/// # Errors
///
/// Returns [`ReadError`] for I/O, header, or length-mismatch failures.
pub fn read_tensor_file<R: Read>(r: &mut R) -> Result<(TensorHeader, Vec<f32>), ReadError> {
    let mut header_bytes = [0u8; HEADER_SIZE];
    r.read_exact(&mut header_bytes)?;
    let header = parse_header(&header_bytes)?;

    let n = header.dim_product as usize;
    let mut body = vec![0u8; n * 4];
    r.read_exact(&mut body)?;

    let mut values = Vec::with_capacity(n);
    for chunk in body.as_chunks::<4>().0 {
        let mut v = [0u8; 4];
        v.copy_from_slice(chunk);
        values.push(f32::from_le_bytes(v));
    }

    if values.len() != n {
        return Err(ReadError::BodyLengthMismatch {
            expected: n,
            got: values.len(),
        });
    }
    Ok((header, values))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn magic_is_aprt() {
        assert_eq!(&MAGIC, b"APRT");
    }

    #[test]
    fn header_size_is_twelve() {
        assert_eq!(HEADER_SIZE, 12);
    }

    #[test]
    fn falsify_apr_trace_save_004_header_format_layer_zero() {
        let bytes = write_header(0, 7);
        assert_eq!(&bytes[0..4], b"APRT");
        assert_eq!(&bytes[4..8], &0u32.to_le_bytes());
        assert_eq!(&bytes[8..12], &7u32.to_le_bytes());
    }

    #[test]
    fn falsify_apr_trace_save_004_header_format_arbitrary_layer() {
        let bytes = write_header(3, 3584);
        assert_eq!(&bytes[0..4], b"APRT");
        let mut layer_bytes = [0u8; 4];
        layer_bytes.copy_from_slice(&bytes[4..8]);
        assert_eq!(u32::from_le_bytes(layer_bytes), 3);
        let mut dp = [0u8; 4];
        dp.copy_from_slice(&bytes[8..12]);
        assert_eq!(u32::from_le_bytes(dp), 3584);
    }

    #[test]
    fn header_roundtrip() {
        let original = TensorHeader {
            layer: 42,
            dim_product: 1024,
        };
        let bytes = write_header(original.layer, original.dim_product);
        let parsed = parse_header(&bytes).expect("parse must succeed");
        assert_eq!(parsed, original);
    }

    #[test]
    fn header_roundtrip_whole_model() {
        let original = TensorHeader {
            layer: WHOLE_MODEL_LAYER,
            dim_product: 151_936,
        };
        let bytes = write_header(original.layer, original.dim_product);
        let parsed = parse_header(&bytes).expect("parse must succeed");
        assert_eq!(parsed, original);
        assert!(parsed.is_whole_model());
    }

    #[test]
    fn parse_header_rejects_short_input() {
        let bytes = vec![0u8; 11];
        let result = parse_header(&bytes);
        assert!(matches!(result, Err(HeaderError::Truncated { got: 11 })));
    }

    #[test]
    fn parse_header_rejects_bad_magic() {
        let mut bytes = vec![0u8; 12];
        bytes[0..4].copy_from_slice(b"GGUF");
        let result = parse_header(&bytes);
        assert!(matches!(result, Err(HeaderError::BadMagic { got: g }) if &g == b"GGUF"));
    }

    #[test]
    fn parse_header_ignores_trailing_body() {
        let mut bytes = write_header(0, 2).to_vec();
        bytes.extend_from_slice(&1.0_f32.to_le_bytes());
        bytes.extend_from_slice(&2.0_f32.to_le_bytes());
        let parsed = parse_header(&bytes).expect("parse must succeed");
        assert_eq!(parsed.layer, 0);
        assert_eq!(parsed.dim_product, 2);
    }

    #[test]
    fn total_file_size_per_layer() {
        let h = TensorHeader {
            layer: 0,
            dim_product: 100,
        };
        assert_eq!(h.total_file_size(), 12 + 400);
    }

    #[test]
    fn total_file_size_empty() {
        let h = TensorHeader {
            layer: 0,
            dim_product: 0,
        };
        assert_eq!(h.total_file_size(), 12);
    }

    #[test]
    fn write_and_read_tensor_file_roundtrip() {
        let values: Vec<f32> = vec![1.0, 2.0, 3.0, -4.0, 0.0, 5.5];
        let mut buf = Vec::new();
        write_tensor_file(&mut buf, 5, &values).expect("write must succeed");

        assert_eq!(buf.len(), HEADER_SIZE + values.len() * 4);

        let mut cursor = std::io::Cursor::new(&buf);
        let (header, read_values) = read_tensor_file(&mut cursor).expect("read must succeed");
        assert_eq!(header.layer, 5);
        assert_eq!(header.dim_product as usize, values.len());
        assert_eq!(read_values, values);
    }

    #[test]
    fn write_tensor_file_empty() {
        let mut buf = Vec::new();
        write_tensor_file(&mut buf, 0, &[]).expect("write must succeed");
        assert_eq!(buf.len(), HEADER_SIZE);
        assert_eq!(&buf[0..4], b"APRT");
    }

    #[test]
    fn write_preserves_nan_verbatim() {
        // Per contract `byte_format` invariant: NaN values preserved verbatim.
        let values: Vec<f32> = vec![f32::NAN, 1.0, f32::INFINITY, f32::NEG_INFINITY];
        let mut buf = Vec::new();
        write_tensor_file(&mut buf, 0, &values).expect("write must succeed");

        let mut cursor = std::io::Cursor::new(&buf);
        let (_, read_values) = read_tensor_file(&mut cursor).expect("read must succeed");
        assert!(read_values[0].is_nan(), "NaN must be preserved");
        assert_eq!(read_values[1], 1.0);
        assert!(read_values[2].is_infinite() && read_values[2].is_sign_positive());
        assert!(read_values[3].is_infinite() && read_values[3].is_sign_negative());
    }

    #[test]
    fn read_tensor_file_truncated_body() {
        let header = write_header(0, 3);
        let mut buf = header.to_vec();
        // Only write 2 of the claimed 3 f32 values
        buf.extend_from_slice(&1.0_f32.to_le_bytes());
        buf.extend_from_slice(&2.0_f32.to_le_bytes());

        let mut cursor = std::io::Cursor::new(&buf);
        let result = read_tensor_file(&mut cursor);
        assert!(result.is_err(), "must error on truncated body");
    }

    #[test]
    fn write_layer_index_max_u32_minus_one() {
        // Last valid per-layer index is u32::MAX - 1; WHOLE_MODEL_LAYER is u32::MAX.
        let edge = u32::MAX - 1;
        let bytes = write_header(edge, 1);
        let parsed = parse_header(&bytes).expect("parse must succeed");
        assert_eq!(parsed.layer, edge);
        assert!(!parsed.is_whole_model());
    }
}
