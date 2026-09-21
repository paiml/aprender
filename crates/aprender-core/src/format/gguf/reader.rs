//! GGUF reader and binary parsing (spec §7.2)

use provable_contracts_macros::ensures;
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
    dequantize_q2_k, dequantize_q3_k, dequantize_q4_k, dequantize_q5_1, dequantize_q5_k,
    dequantize_q6_k, f16_to_f32,
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

    // #3733: every header key is now parsed (keys outside the allowlist used to
    // be skipped by length arithmetic alone), so a count taken from the file must
    // not size an allocation or index past the end. The span is checked first;
    // a string needs at least its 8-byte length, which bounds that count too.
    let available = data.len().saturating_sub(offset + consumed);
    let min_elem = match elem_type {
        0..=1 | 7 => 1,
        2..=3 => 2,
        8 | 10..=12 => 8,
        _ => 4,
    };
    if count.checked_mul(min_elem).is_none_or(|need| need > available) {
        return Err(AprenderError::FormatError {
            message: format!(
                "GGUF metadata array of {count} elements (type {elem_type}) overruns the file"
            ),
        });
    }

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
                values.push(read_i32_le(data, offset + consumed)?);
                consumed += 4;
            }
            Ok((GgufValue::ArrayInt32(values), consumed))
        }
        6 => {
            let mut values = Vec::with_capacity(count);
            for _ in 0..count {
                values.push(read_f32_le(data, offset + consumed)?);
                consumed += 4;
            }
            Ok((GgufValue::ArrayFloat32(values), consumed))
        }
        _ => {
            consumed += count * min_elem;
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
    contract_pre_metadata_kv_safety!();
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
    /// Every header key OUTSIDE the parse allowlist, kept for inventory only
    /// (`apr inspect`). Empty when the reader was built with `keep_all`.
    ///
    /// #3733: the allowlist (`tokenizer.`, `general.`, and six arch prefixes)
    /// decided what `apr inspect --json` could show, so a Qwen3.5 file lost all
    /// 22 of its `qwen35.*` keys, `context_length` among them: `qwen35.` does not
    /// start with `qwen3.`. The keys live here rather than in `metadata` because
    /// the config accessors read `metadata`, and widening THEM would hand every
    /// unlisted architecture a config the import path has never been given.
    pub display_only_metadata: BTreeMap<String, GgufValue>,
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

include!("reader_loading.rs");
include!("reader_test_mod.rs");
