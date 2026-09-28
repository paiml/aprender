//! Falsifiers for qwen35-format-roundtrip-v1 QFR-001/002, in-tree on a pygmy.
//! S-R10 measured the lossless leg by hand on Qwen3.5-0.8B (489/489
//! identical); these bind it to CI: a qwen3_5 `.apr` → safetensors → `.apr`
//! round trip is bit-identical on EVERY tensor, and the gate that says so
//! turns RED on one transposed tensor.

use super::super::*;
use super::qfr_003_config_roundtrip::{import_hf_dir, source_config};
use std::collections::BTreeMap;
use std::fs;

/// name → (dtype, shape, raw bytes) for every tensor in an `.apr`.
type Tensors = BTreeMap<String, (String, Vec<usize>, Vec<u8>)>;

fn tensors(apr: &Path) -> Tensors {
    let bytes = fs::read(apr).expect("read apr");
    let reader = crate::format::v2::AprV2Reader::from_bytes(&bytes).expect("parse apr");
    reader
        .tensor_names()
        .into_iter()
        .map(|name| {
            let entry = reader.get_tensor(name).expect("index entry");
            let data = reader.get_tensor_data(name).expect("tensor data");
            (
                name.to_string(),
                (format!("{:?}", entry.dtype), entry.shape.clone(), data.to_vec()),
            )
        })
        .collect()
}

/// The QFR-001 gate: every tensor on either side that differs in dtype,
/// shape or any byte. It compares ALL tensors, never a sample (`apr diff
/// --values` defaults to `--limit 10`, the failure mode QFR-002 names).
fn value_gate(a: &Tensors, b: &Tensors) -> Vec<String> {
    let mut bad: Vec<String> = a
        .iter()
        .filter(|(name, t)| b.get(*name) != Some(t))
        .map(|(name, _)| name.clone())
        .collect();
    bad.extend(b.keys().filter(|name| !a.contains_key(*name)).cloned());
    bad
}

/// import(HF dir) → export safetensors (+ config.json) → bare re-import.
fn round_trip() -> (Tensors, Tensors) {
    let src = tempfile::tempdir().expect("tempdir");
    let apr = import_hf_dir(src.path(), Some(&source_config()));
    let out = tempfile::tempdir().expect("tempdir");
    let st = out.path().join("model.safetensors");
    let options = ExportOptions {
        format: ExportFormat::SafeTensors,
        include_config: true,
        include_tokenizer: false,
        ..Default::default()
    };
    apr_export(&apr, &st, options).expect("export");
    let back = out.path().join("back.apr");
    apr_import(st.to_str().expect("utf8"), &back, ImportOptions::default())
        .expect("the bare export re-imports");
    (tensors(&apr), tensors(&back))
}

/// FALSIFY-QFR-001: the lossless leg is bit-identical, tensor set unchanged.
#[test]
fn falsify_qfr_001_safetensors_roundtrip_is_bit_identical() {
    let (before, after) = round_trip();
    assert!(before.len() > 10, "pygmy too small to catch a --limit 10 sample: {}", before.len());
    assert_eq!(
        before.keys().collect::<Vec<_>>(),
        after.keys().collect::<Vec<_>>(),
        "tensor set changed"
    );
    assert_eq!(value_gate(&before, &after), Vec::<String>::new());
}

/// Transpose a row-major f32 matrix held as little-endian bytes.
fn transpose_f32(shape: &[usize], data: &[u8]) -> (Vec<usize>, Vec<u8>) {
    let (rows, cols) = (shape[0], shape[1]);
    let mut out = vec![0u8; data.len()];
    for r in 0..rows {
        for c in 0..cols {
            let (src, dst) = (4 * (r * cols + c), 4 * (c * rows + r));
            out[dst..dst + 4].copy_from_slice(&data[src..src + 4]);
        }
    }
    (vec![cols, rows], out)
}

/// FALSIFY-QFR-002: plant one transposed tensor in the round-tripped set, in
/// EVERY 2-D F32 tensor in turn; the gate names the planted one and nothing
/// else. At least one plant sits past the 10th tensor, so a gate that
/// samples the first 10 (`apr diff --values` default) goes RED here.
#[test]
fn falsify_qfr_002_one_transposed_tensor_turns_the_gate_red() {
    let (before, after) = round_trip();
    let names: Vec<String> = after.keys().cloned().collect();
    let mut planted = Vec::new();
    for (index, name) in names.iter().enumerate() {
        let (dtype, shape, data) = &after[name];
        if dtype != "F32" || shape.len() != 2 {
            continue;
        }
        let (t_shape, t_data) = transpose_f32(shape, data);
        if t_shape == *shape && t_data == *data {
            continue; // symmetric: a transpose plants nothing
        }
        let mut mutated = after.clone();
        let entry = mutated.get_mut(name).expect("planted tensor");
        (entry.1, entry.2) = (t_shape, t_data);
        assert_eq!(value_gate(&before, &mutated), vec![name.clone()], "plant in {name}");
        planted.push(index);
    }
    assert!(
        planted.iter().any(|&i| i >= 10),
        "no plant past the 10th tensor (plants at {planted:?} of {names:?})"
    );
}
