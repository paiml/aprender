//! Minimal safetensors reader: header length (u64 LE) + JSON header + raw data. F32, F16 and BF16
//! are decoded to f32 (`half`); everything else is refused. Works on a byte slice so the embedded
//! copy and a file on disk go through the same door.
use std::collections::HashMap;

pub struct Tensor { pub shape: Vec<usize>, pub data: Vec<f32> }
pub type Weights = HashMap<String, Tensor>;

/// Returns the tensors and the dtype they were stored in.
pub fn load_bytes(bytes: &[u8]) -> Result<(Weights, String), String> {
    if bytes.len() < 8 { return Err("safetensors: file too short".into()); }
    let n = u64::from_le_bytes(bytes[..8].try_into().expect("8 bytes")) as usize;
    let header: serde_json::Value = serde_json::from_slice(bytes.get(8..8 + n).ok_or("safetensors: truncated header")?).map_err(|e| format!("header json: {e}"))?;
    let base = 8 + n;
    let mut out = HashMap::new();
    let mut dtype_seen = String::new();
    for (name, meta) in header.as_object().ok_or("header not an object")? {
        if name == "__metadata__" { continue; }
        let dtype = meta["dtype"].as_str().ok_or("dtype")?;
        let shape: Vec<usize> = meta["shape"].as_array().ok_or("shape")?.iter().map(|v| v.as_u64().expect("dim") as usize).collect();
        let offs = meta["data_offsets"].as_array().ok_or("offsets")?;
        let (a, b) = (offs[0].as_u64().expect("a") as usize, offs[1].as_u64().expect("b") as usize);
        let raw = bytes.get(base + a..base + b).ok_or_else(|| format!("{name}: data out of range"))?;
        let data: Vec<f32> = match dtype {
            "F32" => raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
            "F16" => raw.chunks_exact(2).map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32()).collect(),
            "BF16" => raw.chunks_exact(2).map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32()).collect(),
            other => return Err(format!("{name}: dtype {other} not supported (F32, F16, BF16)")),
        };
        let numel: usize = shape.iter().product();
        if data.len() != numel { return Err(format!("{name}: {} values for shape {shape:?}", data.len())); }
        if dtype_seen.is_empty() { dtype_seen = dtype.to_string(); } else if dtype_seen != dtype { dtype_seen = "mixed".into(); }
        out.insert(name.clone(), Tensor { shape, data });
    }
    Ok((out, dtype_seen))
}

pub fn load(path: &str) -> Result<(Weights, String), String> {
    let bytes = std::fs::read(path).map_err(|e| format!("read {path}: {e}"))?;
    load_bytes(&bytes)
}
