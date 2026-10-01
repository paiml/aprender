//! Minimal safetensors reader: header length (u64 LE) + JSON header + raw data. F32, F16 and BF16
//! are decoded to f32 (`half`); everything else is refused. Works on a byte slice so the embedded
//! copy and a file on disk go through the same door.
//!
//! Ported VERBATIM from `sources/007-chronos-mcp-thin-server/src/safetensors.rs` (41 lines) under
//! D-08. The only edits are rustfmt and the doc comments the workspace lints want; no branch, no
//! bound and no arithmetic moved. The `safetensors` crate (0.4/0.7/0.8, all already in
//! `Cargo.lock`) is deliberately NOT used: it does not decode F16/BF16 to f32, and this loader is
//! the one the spike-005 parity ladder measured 9.54e-7 through.

use std::collections::HashMap;

/// One decoded tensor: its shape and its values, always as `f32`.
pub struct Tensor {
    /// Row-major dimensions exactly as the file declares them.
    pub shape: Vec<usize>,
    /// `shape.iter().product()` values, decoded from F32/F16/BF16.
    pub data: Vec<f32>,
}

/// Every tensor in one safetensors file, keyed by its name.
pub type Weights = HashMap<String, Tensor>;

/// Decode a safetensors byte slice into tensors plus the dtype they were stored in.
///
/// The dtype string is `"F32"`, `"F16"`, `"BF16"` — or `"mixed"` when a single file holds more
/// than one. It is reported (not enforced) so `chronos::Model` can put the real provenance in
/// its diagnostics.
///
/// # Errors
///
/// A `String` naming the defect when the slice is shorter than the 8-byte length prefix, the
/// header is truncated or not JSON, a tensor's data range falls outside the slice, a tensor's
/// value count disagrees with its declared shape, or a dtype outside F32/F16/BF16 appears.
pub fn load_bytes(bytes: &[u8]) -> Result<(Weights, String), String> {
    if bytes.len() < 8 {
        return Err("safetensors: file too short".into());
    }
    let n = u64::from_le_bytes(bytes[..8].try_into().expect("8 bytes")) as usize;
    // `8 + n` is attacker arithmetic: a header prefix near `u64::MAX` overflows `usize`,
    // which panics in debug and wraps in release. `checked_add` keeps the documented
    // `Err` contract on both profiles.
    let base = 8usize
        .checked_add(n)
        .ok_or("safetensors: header length overflows")?;
    let header: serde_json::Value =
        serde_json::from_slice(bytes.get(8..base).ok_or("safetensors: truncated header")?)
            .map_err(|e| format!("header json: {e}"))?;
    let mut out = HashMap::new();
    let mut dtype_seen = String::new();
    for (name, meta) in header.as_object().ok_or("header not an object")? {
        if name == "__metadata__" {
            continue;
        }
        let dtype = meta["dtype"].as_str().ok_or("dtype")?;
        // Every field below is read out of a file this function documents itself as
        // REFUSING when malformed, so none of them may panic: a negative or float `shape`
        // entry, a short `data_offsets`, or an offset that overflows when rebased all have
        // to come back as the declared `Err(String)`.
        let shape: Vec<usize> = meta["shape"]
            .as_array()
            .ok_or("shape")?
            .iter()
            .map(|v| {
                v.as_u64()
                    .map(|d| d as usize)
                    .ok_or_else(|| format!("{name}: shape entry is not a non-negative integer"))
            })
            .collect::<Result<_, _>>()?;
        let offs = meta["data_offsets"].as_array().ok_or("offsets")?;
        let offset = |i: usize| -> Result<usize, String> {
            let v = offs
                .get(i)
                .and_then(serde_json::Value::as_u64)
                .ok_or_else(|| format!("{name}: data_offsets[{i}] missing or not an integer"))?;
            usize::try_from(v)
                .ok()
                .and_then(|v| base.checked_add(v))
                .ok_or_else(|| format!("{name}: data_offsets[{i}] out of range"))
        };
        let (a, b) = (offset(0)?, offset(1)?);
        let raw = bytes
            .get(a..b)
            .ok_or_else(|| format!("{name}: data out of range"))?;
        let data: Vec<f32> = match dtype {
            "F32" => raw
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            "F16" => raw
                .chunks_exact(2)
                .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect(),
            "BF16" => raw
                .chunks_exact(2)
                .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect(),
            other => {
                return Err(format!(
                    "{name}: dtype {other} not supported (F32, F16, BF16)"
                ))
            }
        };
        // `product()` wraps in release, so `[2^32, 2^32]` would collapse to 0 and make the
        // length check below pass on a truncated tensor.
        let numel: usize = shape
            .iter()
            .try_fold(1usize, |acc, d| acc.checked_mul(*d))
            .ok_or_else(|| format!("{name}: shape {shape:?} overflows"))?;
        if data.len() != numel {
            return Err(format!("{name}: {} values for shape {shape:?}", data.len()));
        }
        if dtype_seen.is_empty() {
            dtype_seen = dtype.to_string();
        } else if dtype_seen != dtype {
            dtype_seen = "mixed".into();
        }
        out.insert(name.clone(), Tensor { shape, data });
    }
    Ok((out, dtype_seen))
}
