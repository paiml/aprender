//! PMAT-906 — Pillar-4 fail-closed: reject a structurally-INCONSISTENT APR model
//! at the LOAD path (`AprV2Model::from_bytes` / `load`), rather than accepting it
//! and emitting garbage at inference.
//!
//! THE BEAT (a real win, not just parity): an APR whose declared `config.vocab_size`
//! does not match the embedding/lm_head ROW count, or whose weight tensor shape does
//! not match the dims implied by `config.hidden_size`, is structurally broken — every
//! token lookup / matmul reads with the wrong stride (or indexes out of bounds) and
//! produces garbage. Before this fix, `from_model_data` accepted such a model `Ok`
//! and only blew up (or silently garbled) deep inside `forward()`.
//!
//! llama.cpp / Ollama ALSO load mismatched-metadata models (their `check_tensors`
//! defaults to off). So `apr` REJECTING it at LOAD is a genuine fail-closed BEAT
//! (PMAT-744 / PMAT-895 lineage).
//!
//!   - OBLIG-APR-VOCAB-EMBED-CONSISTENT
//!   - OBLIG-APR-WEIGHT-SHAPE-MATCHES-CONFIG

use crate::apr::{AprV2Model, HEADER_SIZE, MAGIC};

/// Tensor definition: (name, dtype byte, shape, byte_size).
type TensorDef = (&'static str, u8, Vec<u64>, usize);

/// Serialize one APR v2 tensor index entry (variable-size format):
/// name_len(2) + name + dtype(1) + ndim(1) + dims(8×ndim) + offset(8) + size(8).
fn tensor_entry(name: &str, dtype: u8, shape: &[u64], offset: u64, size: u64) -> Vec<u8> {
    let mut data = Vec::new();
    data.extend_from_slice(&(name.len() as u16).to_le_bytes());
    data.extend_from_slice(name.as_bytes());
    data.push(dtype);
    data.push(shape.len() as u8);
    for &dim in shape {
        data.extend_from_slice(&dim.to_le_bytes());
    }
    data.extend_from_slice(&offset.to_le_bytes());
    data.extend_from_slice(&size.to_le_bytes());
    data
}

/// Build a minimal F32 APR v2 model with caller-controlled metadata config dims
/// and tensor shapes. This lets a falsifier inject EXACTLY one mismatch
/// (vocab_size vs embedding rows, or hidden_size vs weight cols) while keeping
/// everything else self-consistent.
fn build_apr(
    meta_vocab_size: usize,
    meta_hidden_size: usize,
    tensor_defs: &[TensorDef],
) -> Vec<u8> {
    let metadata = format!(
        r#"{{
        "architecture": "llama",
        "hidden_size": {meta_hidden_size},
        "num_layers": 1,
        "num_heads": 2,
        "num_kv_heads": 2,
        "vocab_size": {meta_vocab_size},
        "intermediate_size": 16,
        "max_position_embeddings": 512,
        "rms_norm_eps": 1e-6
    }}"#
    );
    let metadata_bytes = metadata.as_bytes();
    let metadata_padded_size = metadata_bytes.len().div_ceil(64) * 64;

    let mut entries = Vec::new();
    let mut current_offset = 0u64;
    for (name, dtype, shape, byte_size) in tensor_defs {
        entries.push(tensor_entry(
            name,
            *dtype,
            shape,
            current_offset,
            *byte_size as u64,
        ));
        current_offset += *byte_size as u64;
    }
    let tensor_index: Vec<u8> = entries.iter().flat_map(|e| e.iter().copied()).collect();
    let tensor_count = tensor_defs.len() as u32;
    let total_data_size = current_offset as usize;

    let tensor_index_offset = HEADER_SIZE as u64 + metadata_padded_size as u64;
    let data_offset = tensor_index_offset + tensor_index.len() as u64;
    let total_size = data_offset as usize + total_data_size;

    let mut data = vec![0u8; total_size];
    data[0..4].copy_from_slice(&MAGIC);
    data[4] = 2; // version major
    data[5] = 0; // version minor
    data[8..12].copy_from_slice(&tensor_count.to_le_bytes());
    data[12..20].copy_from_slice(&(HEADER_SIZE as u64).to_le_bytes());
    data[20..24].copy_from_slice(&(metadata_bytes.len() as u32).to_le_bytes());
    data[24..32].copy_from_slice(&tensor_index_offset.to_le_bytes());
    data[32..40].copy_from_slice(&data_offset.to_le_bytes());

    data[HEADER_SIZE..HEADER_SIZE + metadata_bytes.len()].copy_from_slice(metadata_bytes);

    let idx_start = tensor_index_offset as usize;
    data[idx_start..idx_start + tensor_index.len()].copy_from_slice(&tensor_index);

    // Fill tensor data with small finite values (orthogonal to the config gate).
    let data_start = data_offset as usize;
    let num_floats = total_data_size / 4;
    for i in 0..num_floats {
        let val = ((i % 10) as f32 - 5.0) * 0.1;
        data[data_start + i * 4..data_start + i * 4 + 4].copy_from_slice(&val.to_le_bytes());
    }
    data
}

/// Full self-consistent tensor set for a pygmy llama (vocab=V, hidden=H, inter=16).
/// All weights match the config; mutate ONE entry to forge an inconsistency.
fn consistent_tensor_defs(vocab: usize, hidden: usize) -> Vec<TensorDef> {
    let inter = 16usize;
    vec![
        (
            "model.embed_tokens.weight",
            0,
            vec![vocab as u64, hidden as u64],
            vocab * hidden * 4,
        ),
        (
            "model.layers.0.input_layernorm.weight",
            0,
            vec![hidden as u64],
            hidden * 4,
        ),
        (
            "model.layers.0.self_attn.q_proj.weight",
            0,
            vec![hidden as u64, hidden as u64],
            hidden * hidden * 4,
        ),
        (
            "model.layers.0.self_attn.k_proj.weight",
            0,
            vec![hidden as u64, hidden as u64],
            hidden * hidden * 4,
        ),
        (
            "model.layers.0.self_attn.v_proj.weight",
            0,
            vec![hidden as u64, hidden as u64],
            hidden * hidden * 4,
        ),
        (
            "model.layers.0.self_attn.o_proj.weight",
            0,
            vec![hidden as u64, hidden as u64],
            hidden * hidden * 4,
        ),
        (
            "model.layers.0.post_attention_layernorm.weight",
            0,
            vec![hidden as u64],
            hidden * 4,
        ),
        (
            "model.layers.0.mlp.gate_proj.weight",
            0,
            vec![inter as u64, hidden as u64],
            hidden * inter * 4,
        ),
        (
            "model.layers.0.mlp.up_proj.weight",
            0,
            vec![inter as u64, hidden as u64],
            hidden * inter * 4,
        ),
        (
            "model.layers.0.mlp.down_proj.weight",
            0,
            vec![hidden as u64, inter as u64],
            hidden * inter * 4,
        ),
        ("model.norm.weight", 0, vec![hidden as u64], hidden * 4),
        (
            "lm_head.weight",
            0,
            vec![vocab as u64, hidden as u64],
            vocab * hidden * 4,
        ),
    ]
}

/// FP-bound: a fully self-consistent pygmy APR (vocab=10, hidden=8) still loads `Ok`.
/// The config gate must not reject a legitimate model.
#[test]
fn consistent_apr_still_loads_ok() {
    let defs = consistent_tensor_defs(10, 8);
    let bytes = build_apr(10, 8, &defs);
    let result = AprV2Model::from_bytes(bytes);
    assert!(
        result.is_ok(),
        "FP-bound FAIL: config-consistency gate rejected a self-consistent APR: {:?}",
        result.err()
    );
}

/// RED → GREEN falsifier for OBLIG-APR-VOCAB-EMBED-CONSISTENT.
///
/// The embedding/lm_head matrices have 10 rows (vocab=10), but the declared
/// `config.vocab_size` is 99. On `main` (before the fix) `from_bytes` returned
/// `Ok` — token IDs in [10,99) would index PAST the 10-row embedding table at
/// inference (garbage / OOB). After the fix it MUST return `Err`.
#[test]
fn vocab_embed_mismatch_rejected_at_load() {
    // Weights physically sized for vocab=10, hidden=8 ...
    let defs = consistent_tensor_defs(10, 8);
    // ... but the metadata LIES: it declares vocab_size=99.
    let bytes = build_apr(99, 8, &defs);

    let result = AprV2Model::from_bytes(bytes);
    assert!(
        result.is_err(),
        "OBLIG-APR-VOCAB-EMBED-CONSISTENT FAIL: from_bytes accepted an APR whose declared \
         vocab_size=99 != embedding rows=10. Token IDs would index out of bounds and inference \
         would produce garbage. apr must fail closed at load."
    );
    let msg = format!("{:?}", result.err().unwrap());
    assert!(
        msg.contains("OBLIG-APR-VOCAB-EMBED-CONSISTENT"),
        "rejection must name the obligation, got: {msg}"
    );
}

/// RED → GREEN falsifier for OBLIG-APR-WEIGHT-SHAPE-MATCHES-CONFIG.
///
/// The embedding/lm_head matrices have 8 columns (hidden=8), but the declared
/// `config.hidden_size` is 64. On `main` (before the fix) `from_bytes` returned
/// `Ok` — every matmul would read the hidden vector with the wrong stride
/// (garbage). After the fix it MUST return `Err`.
#[test]
fn weight_shape_hidden_mismatch_rejected_at_load() {
    // Weights physically sized for vocab=10, hidden=8 ...
    let defs = consistent_tensor_defs(10, 8);
    // ... but the metadata LIES: it declares hidden_size=64.
    let bytes = build_apr(10, 64, &defs);

    let result = AprV2Model::from_bytes(bytes);
    assert!(
        result.is_err(),
        "OBLIG-APR-WEIGHT-SHAPE-MATCHES-CONFIG FAIL: from_bytes accepted an APR whose declared \
         hidden_size=64 != embedding cols=8. Every matmul would read the hidden vector with the \
         wrong stride and inference would produce garbage. apr must fail closed at load."
    );
    let msg = format!("{:?}", result.err().unwrap());
    assert!(
        msg.contains("OBLIG-APR-WEIGHT-SHAPE-MATCHES-CONFIG"),
        "rejection must name the obligation, got: {msg}"
    );
}
