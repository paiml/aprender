mod basic;
mod falsify;
mod proptest_roundtrip;
mod stress;

use super::parsing::{read_gguf_string, skip_gguf_value};
use super::types::GgufTensorInfo;
use super::*;
use crate::hf_pipeline::export::gguf_writer::{quantize_to_gguf_bytes, GgufQuantization};
use aprender::format::gguf::{export_tensors_to_gguf, GgmlType, GgufTensor, GgufValue};

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
/// Note: aprender's `export_tensors_to_gguf` places tensor data immediately after
/// tensor info with NO alignment padding. Tensor offsets in the info entries are
/// relative to this position.
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
    pos
}
