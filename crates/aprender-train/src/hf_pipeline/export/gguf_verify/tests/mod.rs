mod basic;
mod falsify;
mod proptest_roundtrip;
mod stress;

use super::parsing::{read_gguf_string, skip_gguf_value};
use super::types::GgufTensorInfo;
use super::*;
use crate::hf_pipeline::export::gguf_writer::{quantize_to_gguf_bytes, GgufQuantization};
use aprender::format::gguf::{
    export_tensors_to_gguf, padding_for_alignment, GgmlType, GgufTensor, GgufValue,
    GGUF_DEFAULT_ALIGNMENT,
};

/// Helper: serialize GGUF to a Vec<u8> via aprender
pub(super) fn write_gguf(tensors: &[GgufTensor], metadata: &[(String, GgufValue)]) -> Vec<u8> {
    let mut buf = Vec::new();
    export_tensors_to_gguf(&mut buf, tensors, metadata).expect("operation should succeed");
    buf
}

/// Extract f32 tensor data from raw GGUF bytes at the given tensor's offset.
/// `data_section_start` is the byte offset where the tensor data section begins.
/// Uses manual LE decoding to avoid alignment requirements of bytemuck::cast_slice.
pub(super) fn extract_f32_tensor_data(
    gguf_bytes: &[u8],
    data_section_start: usize,
    tensor_info: &GgufTensorInfo,
    num_elements: usize,
) -> Vec<f32> {
    let start = data_section_start + tensor_info.offset as usize;
    (0..num_elements)
        .map(|i| {
            let off = start + i * 4;
            f32::from_le_bytes(
                gguf_bytes[off..off + 4].try_into().expect("conversion should succeed"),
            )
        })
        .collect()
}

/// Find the start of the tensor data section by scanning past header + metadata + tensor info.
///
/// `aprender::format::gguf::export_tensors_to_gguf` writes alignment padding
/// (default 32 bytes) AFTER the tensor-info section so the tensor data starts
/// at a `GGUF_DEFAULT_ALIGNMENT`-aligned offset. The previous version of this
/// helper claimed "NO alignment padding" — that's incorrect; tensor data is
/// preceded by `padding_for_alignment(pos, GGUF_DEFAULT_ALIGNMENT)` zero bytes.
/// All `test_falsify_*_roundtrip` and friends were reading f32 bytes at the
/// padding offset (zeros) instead of the actual tensor offset, surfaced as
/// `expected [5.0, 6.0, 7.0, 8.0]` vs `got [0.0, 5.93e-39, ...]`.
pub(super) fn find_data_section_start(gguf_bytes: &[u8], summary: &GgufSummary) -> usize {
    let mut pos = 24; // skip header
                      // Skip metadata
    for _ in 0..summary.metadata_count {
        let (_, new_pos) = read_gguf_string(gguf_bytes, pos).expect("operation should succeed");
        pos = new_pos;
        let value_type = u32::from_le_bytes(
            gguf_bytes[pos..pos + 4].try_into().expect("conversion should succeed"),
        );
        pos += 4;
        pos = skip_gguf_value(gguf_bytes, pos, value_type).expect("operation should succeed");
    }
    // Skip tensor info
    for _ in 0..summary.tensor_count {
        let (_, new_pos) = read_gguf_string(gguf_bytes, pos).expect("operation should succeed");
        pos = new_pos;
        let n_dims = u32::from_le_bytes(
            gguf_bytes[pos..pos + 4].try_into().expect("conversion should succeed"),
        ) as usize;
        pos += 4 + n_dims * 8 + 4 + 8; // dims + dtype + offset
    }
    // Skip alignment padding before tensor data (matches writer at types.rs:445).
    pos += padding_for_alignment(pos, GGUF_DEFAULT_ALIGNMENT);
    pos
}
