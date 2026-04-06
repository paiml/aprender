//! GGUF Primitive I/O Readers (PMAT-COMPLY)
//!
//! Extracted from loader.rs for file health compliance.
//! Low-level binary reading functions for GGUF format parsing.

use crate::error::{RealizarError, Result};
use std::io::{Cursor, Read};

/// Read a single u8 from the cursor.
pub(crate) fn read_u8(cursor: &mut Cursor<&[u8]>) -> Result<u8> {
    let mut buf = [0u8; 1];
    cursor
        .read_exact(&mut buf)
        .map_err(|e| RealizarError::UnsupportedOperation {
            operation: "read_u8".to_string(),
            reason: e.to_string(),
        })?;
    Ok(buf[0])
}

/// Read a single i8 from the cursor.
pub(crate) fn read_i8(cursor: &mut Cursor<&[u8]>) -> Result<i8> {
    let mut buf = [0u8; 1];
    cursor
        .read_exact(&mut buf)
        .map_err(|e| RealizarError::UnsupportedOperation {
            operation: "read_i8".to_string(),
            reason: e.to_string(),
        })?;
    Ok(i8::from_le_bytes(buf))
}

/// Read a u16 (little-endian) from the cursor.
pub(crate) fn read_u16(cursor: &mut Cursor<&[u8]>) -> Result<u16> {
    let mut buf = [0u8; 2];
    cursor
        .read_exact(&mut buf)
        .map_err(|e| RealizarError::UnsupportedOperation {
            operation: "read_u16".to_string(),
            reason: e.to_string(),
        })?;
    Ok(u16::from_le_bytes(buf))
}

/// Read an i16 (little-endian) from the cursor.
pub(crate) fn read_i16(cursor: &mut Cursor<&[u8]>) -> Result<i16> {
    let mut buf = [0u8; 2];
    cursor
        .read_exact(&mut buf)
        .map_err(|e| RealizarError::UnsupportedOperation {
            operation: "read_i16".to_string(),
            reason: e.to_string(),
        })?;
    Ok(i16::from_le_bytes(buf))
}

/// Read a u32 (little-endian) from the cursor.
pub(crate) fn read_u32(cursor: &mut Cursor<&[u8]>) -> Result<u32> {
    let mut buf = [0u8; 4];
    cursor
        .read_exact(&mut buf)
        .map_err(|e| RealizarError::UnsupportedOperation {
            operation: "read_u32".to_string(),
            reason: e.to_string(),
        })?;
    Ok(u32::from_le_bytes(buf))
}

/// Read an i32 (little-endian) from the cursor.
pub(crate) fn read_i32(cursor: &mut Cursor<&[u8]>) -> Result<i32> {
    let mut buf = [0u8; 4];
    cursor
        .read_exact(&mut buf)
        .map_err(|e| RealizarError::UnsupportedOperation {
            operation: "read_i32".to_string(),
            reason: e.to_string(),
        })?;
    Ok(i32::from_le_bytes(buf))
}

/// Read an f32 (little-endian) from the cursor.
pub(crate) fn read_f32(cursor: &mut Cursor<&[u8]>) -> Result<f32> {
    let mut buf = [0u8; 4];
    cursor
        .read_exact(&mut buf)
        .map_err(|e| RealizarError::UnsupportedOperation {
            operation: "read_f32".to_string(),
            reason: e.to_string(),
        })?;
    Ok(f32::from_le_bytes(buf))
}

/// Read a bool from the cursor.
pub(crate) fn read_bool(cursor: &mut Cursor<&[u8]>) -> Result<bool> {
    let mut buf = [0u8; 1];
    cursor
        .read_exact(&mut buf)
        .map_err(|e| RealizarError::UnsupportedOperation {
            operation: "read_bool".to_string(),
            reason: e.to_string(),
        })?;
    Ok(buf[0] != 0)
}

/// Read a u64 (little-endian) from the cursor.
pub(crate) fn read_u64(cursor: &mut Cursor<&[u8]>) -> Result<u64> {
    let mut buf = [0u8; 8];
    cursor
        .read_exact(&mut buf)
        .map_err(|e| RealizarError::UnsupportedOperation {
            operation: "read_u64".to_string(),
            reason: e.to_string(),
        })?;
    Ok(u64::from_le_bytes(buf))
}

/// Read an i64 (little-endian) from the cursor.
pub(crate) fn read_i64(cursor: &mut Cursor<&[u8]>) -> Result<i64> {
    let mut buf = [0u8; 8];
    cursor
        .read_exact(&mut buf)
        .map_err(|e| RealizarError::UnsupportedOperation {
            operation: "read_i64".to_string(),
            reason: e.to_string(),
        })?;
    Ok(i64::from_le_bytes(buf))
}

/// Read an f64 (little-endian) from the cursor.
pub(crate) fn read_f64(cursor: &mut Cursor<&[u8]>) -> Result<f64> {
    let mut buf = [0u8; 8];
    cursor
        .read_exact(&mut buf)
        .map_err(|e| RealizarError::UnsupportedOperation {
            operation: "read_f64".to_string(),
            reason: e.to_string(),
        })?;
    Ok(f64::from_le_bytes(buf))
}
