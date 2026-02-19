//! GGUF reader and binary parsing (spec §7.2)

use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;

// BUG-GGUF-001 FIX: Define reasonable limits to prevent allocation attacks
// A malicious file with tensor_count=u64::MAX could cause OOM or panic.
// Even the largest models (Llama 405B) have <1000 tensors.
/// Maximum number of tensors allowed in a GGUF file (prevents OOM attack)
const MAX_TENSOR_COUNT: u64 = 100_000;
/// Maximum number of metadata entries allowed (prevents OOM attack)
const MAX_METADATA_COUNT: u64 = 100_000;
/// Maximum number of dimensions per tensor (no real tensor has > 8 dims)
const MAX_DIMS: u32 = 16;
/// BUG-GGUF-002 FIX: Maximum total elements per tensor (~16GB F32 tensor)
/// This prevents integer overflow in shape.iter().product() and subsequent
/// byte size calculations. 4B elements * 4 bytes = 16GB, reasonable for largest models.
const MAX_TENSOR_ELEMENTS: usize = 4_000_000_000;

use super::dequant::{
    dequantize_iq_approximate, dequantize_q2_k, dequantize_q3_k, dequantize_q4_k, dequantize_q5_1,
    dequantize_q5_k, dequantize_q6_k, f16_to_f32,
};
use super::types::{
    padding_for_alignment, GgufValue, TensorDataMap, GGUF_DEFAULT_ALIGNMENT, GGUF_MAGIC,
};
use crate::error::{AprenderError, Result};

// ============================================================================
// GGUF Reading/Import API
// ============================================================================

/// Read a u32 from bytes at offset
pub(crate) fn read_u32(data: &[u8], offset: usize) -> Result<u32> {
    if offset + 4 > data.len() {
        return Err(AprenderError::FormatError {
            message: format!("Unexpected EOF reading u32 at offset {offset}"),
        });
    }
    Ok(u32::from_le_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ]))
}

/// Read a u64 from bytes at offset
pub(crate) fn read_u64(data: &[u8], offset: usize) -> Result<u64> {
    if offset + 8 > data.len() {
        return Err(AprenderError::FormatError {
            message: format!("Unexpected EOF reading u64 at offset {offset}"),
        });
    }
    Ok(u64::from_le_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
        data[offset + 4],
        data[offset + 5],
        data[offset + 6],
        data[offset + 7],
    ]))
}

/// Read a length-prefixed string from bytes
pub(crate) fn read_string(data: &[u8], offset: usize) -> Result<(String, usize)> {
    let len = read_u64(data, offset)? as usize;
    let str_start = offset + 8;
    if str_start + len > data.len() {
        return Err(AprenderError::FormatError {
            message: format!("String length {len} exceeds data at offset {offset}"),
        });
    }
    let s = String::from_utf8_lossy(&data[str_start..str_start + len]).to_string();
    Ok((s, 8 + len))
}

/// Read a GGUF array value (type 9) and return (value, bytes_consumed).
fn read_metadata_array(data: &[u8], offset: usize) -> Result<(GgufValue, usize)> {
    let elem_type = read_u32(data, offset)?;
    let count = read_u64(data, offset + 4)? as usize;
    let mut consumed = 12; // type (4) + count (8)

    match elem_type {
        8 => {
            let mut strings = Vec::with_capacity(count);
            for _ in 0..count {
                let (s, len) = read_string(data, offset + consumed)?;
                strings.push(s);
                consumed += len;
            }
            Ok((GgufValue::ArrayString(strings), consumed))
        }
        4 => {
            let mut values = Vec::with_capacity(count);
            for _ in 0..count {
                values.push(read_u32(data, offset + consumed)?);
                consumed += 4;
            }
            Ok((GgufValue::ArrayUint32(values), consumed))
        }
        5 => {
            let mut values = Vec::with_capacity(count);
            for _ in 0..count {
                let v = i32::from_le_bytes([
                    data[offset + consumed],
                    data[offset + consumed + 1],
                    data[offset + consumed + 2],
                    data[offset + consumed + 3],
                ]);
                values.push(v);
                consumed += 4;
            }
            Ok((GgufValue::ArrayInt32(values), consumed))
        }
        6 => {
            let mut values = Vec::with_capacity(count);
            for _ in 0..count {
                let v = f32::from_le_bytes([
                    data[offset + consumed],
                    data[offset + consumed + 1],
                    data[offset + consumed + 2],
                    data[offset + consumed + 3],
                ]);
                values.push(v);
                consumed += 4;
            }
            Ok((GgufValue::ArrayFloat32(values), consumed))
        }
        _ => {
            let elem_size = match elem_type {
                0..=1 | 7 => 1,
                2..=3 => 2,
                10..=12 => 8,
                _ => 4,
            };
            consumed += count * elem_size;
            Ok((GgufValue::ArrayUint32(vec![]), consumed))
        }
    }
}

/// Read a metadata value and return (value, bytes_consumed)
/// Ensure `n` bytes are available at `offset`, returning a format error with `type_name` if not.
fn ensure_bytes(data: &[u8], offset: usize, n: usize, type_name: &str) -> Result<()> {
    if offset + n > data.len() {
        return Err(AprenderError::FormatError {
            message: format!("Unexpected EOF reading {type_name}"),
        });
    }
    Ok(())
}

/// Read a little-endian i16 from `data` at `offset`.
fn read_i16_le(data: &[u8], offset: usize) -> Result<i16> {
    ensure_bytes(data, offset, 2, "Int16")?;
    Ok(i16::from_le_bytes([data[offset], data[offset + 1]]))
}

/// Read a little-endian i32 from `data` at `offset`.
fn read_i32_le(data: &[u8], offset: usize) -> Result<i32> {
    ensure_bytes(data, offset, 4, "Int32")?;
    Ok(i32::from_le_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ]))
}

/// Read a little-endian f32 from `data` at `offset`.
fn read_f32_le(data: &[u8], offset: usize) -> Result<f32> {
    ensure_bytes(data, offset, 4, "Float32")?;
    Ok(f32::from_le_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ]))
}

/// Read a little-endian i64 from `data` at `offset`.
fn read_i64_le(data: &[u8], offset: usize) -> Result<i64> {
    ensure_bytes(data, offset, 8, "Int64")?;
    Ok(i64::from_le_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
        data[offset + 4],
        data[offset + 5],
        data[offset + 6],
        data[offset + 7],
    ]))
}

/// Read a little-endian f64 from `data` at `offset`.
fn read_f64_le(data: &[u8], offset: usize) -> Result<f64> {
    ensure_bytes(data, offset, 8, "Float64")?;
    Ok(f64::from_le_bytes([
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
        data[offset + 4],
        data[offset + 5],
        data[offset + 6],
        data[offset + 7],
    ]))
}

/// Read a single-byte metadata value (Uint8, Int8, or Bool).
fn read_metadata_byte(data: &[u8], offset: usize, value_type: u32) -> Result<(GgufValue, usize)> {
    let label = match value_type {
        0 => "Uint8",
        1 => "Int8",
        _ => "Bool",
    };
    ensure_bytes(data, offset, 1, label)?;
    let val = match value_type {
        0 => GgufValue::Uint8(data[offset]),
        1 => GgufValue::Int8(data[offset] as i8),
        _ => GgufValue::Bool(data[offset] != 0),
    };
    Ok((val, 1))
}

pub(crate) fn read_metadata_value(
    data: &[u8],
    offset: usize,
    value_type: u32,
) -> Result<(GgufValue, usize)> {
    match value_type {
        0 | 1 | 7 => read_metadata_byte(data, offset, value_type),
        2 => {
            ensure_bytes(data, offset, 2, "Uint16")?;
            Ok((
                GgufValue::Uint16(u16::from_le_bytes([data[offset], data[offset + 1]])),
                2,
            ))
        }
        3 => Ok((GgufValue::Int16(read_i16_le(data, offset)?), 2)),
        4 => Ok((GgufValue::Uint32(read_u32(data, offset)?), 4)),
        5 => Ok((GgufValue::Int32(read_i32_le(data, offset)?), 4)),
        6 => Ok((GgufValue::Float32(read_f32_le(data, offset)?), 4)),
        8 => {
            let (s, len) = read_string(data, offset)?;
            Ok((GgufValue::String(s), len))
        }
        9 => read_metadata_array(data, offset),
        10 => Ok((GgufValue::Uint64(read_u64(data, offset)?), 8)),
        11 => Ok((GgufValue::Int64(read_i64_le(data, offset)?), 8)),
        12 => Ok((GgufValue::Float64(read_f64_le(data, offset)?), 8)),
        _ => Ok((GgufValue::Uint32(0), 4)),
    }
}

/// Skip a metadata value (we don't need to parse all metadata types)
fn skip_metadata_value(data: &[u8], offset: usize, value_type: u32) -> Result<usize> {
    match value_type {
        0..=1 | 7 => Ok(1), // Uint8, Int8, Bool
        2..=3 => Ok(2),     // Uint16, Int16
        8 => {
            // String
            let len = read_u64(data, offset)? as usize;
            Ok(8 + len)
        }
        9 => {
            // Array - need to read element type and count, then skip elements
            let elem_type = read_u32(data, offset)?;
            let count = read_u64(data, offset + 4)? as usize;
            let elem_size = match elem_type {
                0..=1 | 7 => 1, // Uint8, Int8, Bool
                2..=3 => 2,     // Uint16, Int16
                8 => {
                    // Array of strings - need to iterate
                    let mut skip = 12; // type (4) + count (8)
                    for _ in 0..count {
                        let (_, slen) = read_string(data, offset + skip)?;
                        skip += slen;
                    }
                    return Ok(skip);
                }
                10..=12 => 8, // Uint64, Int64, Float64
                _ => 4,       // Uint32, Int32, Float32, Unknown
            };
            Ok(12 + count * elem_size)
        }
        10..=12 => Ok(8), // Uint64, Int64, Float64
        _ => Ok(4),       // Uint32, Int32, Float32, Unknown (4 bytes default)
    }
}

/// Parsed GGUF file for import
#[derive(Debug)]
pub struct GgufReader {
    /// Raw file data
    pub(crate) data: Vec<u8>,
    /// Format version
    pub version: u32,
    /// Number of tensors
    pub tensor_count: u64,
    /// Tensor infos (name, dims, dtype, `data_offset`)
    pub tensors: Vec<GgufTensorMeta>,
    /// Offset where tensor data section starts
    pub data_offset: usize,
    /// Metadata key-value pairs (extracted from GGUF)
    pub metadata: BTreeMap<String, GgufValue>,
}

/// Tensor metadata from GGUF file
#[derive(Debug, Clone)]
pub struct GgufTensorMeta {
    /// Tensor name
    pub name: String,
    /// Dimensions
    pub dims: Vec<u64>,
    /// Data type (`GgmlType` as u32)
    pub dtype: u32,
    /// Offset within tensor data section
    pub offset: u64,
}

include!("reader_part_02.rs");
include!("reader_part_03.rs");
