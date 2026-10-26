//! `load_qwen35_layers` fails closed on a GGUF that does not describe its depth or
//! carry its tensors, instead of returning a truncated model (#5678).
//! Header-only GGUFs are hand-built so no model file is needed.

use realizar::gguf::qwen35_load::load_qwen35_layers;
use realizar::gguf::GGUFModel;

fn gguf_string(s: &str) -> Vec<u8> {
    let mut v = (s.len() as u64).to_le_bytes().to_vec();
    v.extend_from_slice(s.as_bytes());
    v
}

/// GGUF v3, zero tensors, `arch` plus optional `{arch}.block_count`.
fn header_only(arch: &str, block_count: Option<u32>) -> Vec<u8> {
    let mut kvs = Vec::new();
    kvs.extend(gguf_string("general.architecture"));
    kvs.extend(8u32.to_le_bytes());
    kvs.extend(gguf_string(arch));
    let mut n = 1u64;
    if let Some(bc) = block_count {
        kvs.extend(gguf_string(&format!("{arch}.block_count")));
        kvs.extend(4u32.to_le_bytes());
        kvs.extend(bc.to_le_bytes());
        n += 1;
    }
    let mut out = b"GGUF".to_vec();
    out.extend(3u32.to_le_bytes());
    out.extend(0u64.to_le_bytes());
    out.extend(n.to_le_bytes());
    out.extend(kvs);
    out
}

fn load(data: &[u8]) -> Result<usize, String> {
    let model = GGUFModel::from_bytes(data).map_err(|e| format!("parse: {e}"))?;
    load_qwen35_layers(&model, data)
        .map(|l| l.len())
        .map_err(|e| e.to_string())
}

#[test]
fn missing_block_count_is_refused_not_guessed() {
    let err = load(&header_only("qwen35", None)).expect_err("no depth, no model");
    assert!(
        err.contains("block_count is missing"),
        "error must name the missing key: {err}"
    );
    assert!(
        err.contains("qwen35"),
        "error must name the architecture: {err}"
    );
}

#[test]
fn zero_layers_loads_as_empty_not_an_error() {
    assert_eq!(load(&header_only("qwen35", Some(0))).expect("empty"), 0);
}

#[test]
fn declared_layer_without_its_tensors_is_an_error() {
    // block_count = 1 -> layer 0 is DeltaNet (interval 4); its tensors are absent.
    let err = load(&header_only("qwen35", Some(1))).expect_err("tensors absent");
    assert!(
        !err.contains("block_count"),
        "failure is the tensor, not the depth: {err}"
    );
}

#[test]
fn full_attention_layer_without_tensors_is_an_error() {
    // block_count = 4 -> the 4th layer is full attention, but even layer 0 is missing.
    assert!(load(&header_only("qwen35", Some(4))).is_err());
}
