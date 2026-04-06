//! GGUF file verification
//!
//! Parses and validates GGUF v3 files, returning a summary of their contents.

mod parsing;
mod types;

#[cfg(test)]
mod tests;

use crate::hf_pipeline::error::FetchError;
use parsing::{parse_all_tensor_info, parse_header, skip_all_metadata};

pub use types::{GgufSummary, GgufTensorInfo};

/// Verify a GGUF file and return a summary
///
/// Parses the header and tensor info sections, validating:
/// - Magic bytes ("GGUF")
/// - Version (must be 3)
/// - Tensor count consistency
/// - No truncation (file large enough for declared tensors)
pub fn verify_gguf(data: &[u8]) -> Result<GgufSummary, FetchError> {
    let header = parse_header(data)?;
    let pos = skip_all_metadata(data, 24, header.metadata_count)?;
    let tensors = parse_all_tensor_info(data, pos, header.tensor_count)?;

    Ok(GgufSummary {
        version: header.version,
        tensor_count: header.tensor_count,
        metadata_count: header.metadata_count,
        file_size: data.len(),
        tensors,
    })
}
