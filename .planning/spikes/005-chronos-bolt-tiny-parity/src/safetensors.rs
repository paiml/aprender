//! Minimal safetensors reader: header length (u64 LE) + JSON header + raw data. F32 only.
use std::collections::HashMap;

pub struct Tensor { pub shape: Vec<usize>, pub data: Vec<f32> }
pub type Weights = HashMap<String, Tensor>;

pub fn load(path: &str) -> Result<Weights, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("read {path}: {e}"))?;
    if bytes.len() < 8 { return Err("file too short".into()); }
    let n = u64::from_le_bytes(bytes[..8].try_into().expect("8 bytes")) as usize;
    let header: serde_json::Value = serde_json::from_slice(&bytes[8..8 + n]).map_err(|e| format!("header json: {e}"))?;
    let base = 8 + n;
    let mut out = HashMap::new();
    for (name, meta) in header.as_object().ok_or("header not an object")? {
        if name == "__metadata__" { continue; }
        let dtype = meta["dtype"].as_str().ok_or("dtype")?;
        if dtype != "F32" { return Err(format!("{name}: dtype {dtype} not supported (F32 only)")); }
        let shape: Vec<usize> = meta["shape"].as_array().ok_or("shape")?.iter().map(|v| v.as_u64().expect("dim") as usize).collect();
        let offs = meta["data_offsets"].as_array().ok_or("offsets")?;
        let (a, b) = (offs[0].as_u64().expect("a") as usize, offs[1].as_u64().expect("b") as usize);
        let raw = &bytes[base + a..base + b];
        let data: Vec<f32> = raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        let numel: usize = shape.iter().product();
        if data.len() != numel { return Err(format!("{name}: {} floats for shape {shape:?}", data.len())); }
        out.insert(name.clone(), Tensor { shape, data });
    }
    Ok(out)
}
