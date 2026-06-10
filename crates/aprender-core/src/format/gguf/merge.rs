//! #1893: Merge sharded GGUF parts (`-NNNNN-of-MMMMM.gguf`) into one GGUF file.
//!
//! Sharded GGUFs carry no `index.json`; each part is a complete GGUF holding a
//! SUBSET of tensors plus `split.*` metadata. Merging the parts into a single
//! file lets the existing single-file loader (`GGUFModel::from_path`) run them
//! unchanged — no inference-hot-path refactor (the codebase's #1 garbage-output
//! risk class).
//!
//! **Type-agnostic.** Tensor data is copied as raw byte ranges sized from the
//! source part's own offset table (the gap to the next tensor's offset), then
//! re-padded to GGUF alignment in the merged file. So EVERY ggml quant type
//! works (Q5_K / Q3_K / IQ\* included) regardless of the `GgmlType` enum — we
//! never interpret tensor contents. `split.*` and `general.alignment` keys are
//! stripped so the merged file is self-consistent at the default 32-byte
//! alignment.

use super::reader::{GgufReader, GgufTensorMeta};
use super::types::{
    padding_for_alignment, write_metadata_kv, GgufHeader, GgufValue, GGUF_DEFAULT_ALIGNMENT,
    GGUF_VERSION,
};
use crate::error::{AprenderError, Result};
use std::io;
use std::path::{Path, PathBuf};

fn invalid(msg: String) -> AprenderError {
    AprenderError::Io(io::Error::new(io::ErrorKind::InvalidData, msg))
}

/// Write a length-prefixed UTF-8 string (GGUF spec §7).
fn write_string(buf: &mut Vec<u8>, s: &str) {
    buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
    buf.extend_from_slice(s.as_bytes());
}

/// One tensor pulled from a shard: name, dims, raw ggml type, and raw bytes.
struct MergedTensor {
    name: String,
    dims: Vec<u64>,
    dtype: u32,
    data: Vec<u8>,
}

/// Merge ordered sharded-GGUF `parts` (the complete set, in part order 1..=N)
/// into a single GGUF written to `output`.
///
/// Metadata is taken from the first part with `split.*` / `general.alignment`
/// stripped; tensors are the union across all parts in file order.
///
/// # Errors
/// Returns an error if fewer than 2 parts are given, a part fails to parse, a
/// part has corrupt tensor offsets, or the output cannot be written.
pub fn merge_gguf_shards(parts: &[PathBuf], output: &Path) -> Result<()> {
    if parts.len() < 2 {
        return Err(invalid(format!(
            "merge_gguf_shards needs >= 2 parts, got {}",
            parts.len()
        )));
    }

    let mut tensors: Vec<MergedTensor> = Vec::new();
    let mut metadata: Vec<(String, GgufValue)> = Vec::new();

    for (i, path) in parts.iter().enumerate() {
        let reader = GgufReader::from_file(path)?;

        // Metadata comes from the first part only (the others duplicate it),
        // minus the split bookkeeping and any alignment override.
        if i == 0 {
            for (k, v) in &reader.metadata {
                if !k.starts_with("split.") && k != "general.alignment" {
                    metadata.push((k.clone(), v.clone()));
                }
            }
        }

        // Size each tensor by the gap to the next offset (type-agnostic).
        let mut metas: Vec<&GgufTensorMeta> = reader.tensors.iter().collect();
        metas.sort_by_key(|t| t.offset);
        let section_len = reader.data.len().saturating_sub(reader.data_offset);

        for (j, meta) in metas.iter().enumerate() {
            let start = meta.offset as usize;
            let end = if j + 1 < metas.len() {
                metas[j + 1].offset as usize
            } else {
                section_len
            };
            let abs_start = reader.data_offset.saturating_add(start);
            let abs_end = reader.data_offset.saturating_add(end);
            if end < start || abs_end > reader.data.len() {
                return Err(invalid(format!(
                    "corrupt tensor offsets in shard {}",
                    path.display()
                )));
            }
            tensors.push(MergedTensor {
                name: meta.name.clone(),
                dims: meta.dims.clone(),
                dtype: meta.dtype,
                data: reader.data[abs_start..abs_end].to_vec(),
            });
        }
    }

    // --- Serialize the merged GGUF ---
    // Header + metadata + tensor infos go into a buffer first so the data
    // section can be aligned against the real header size (DEFECT-002 pattern).
    let mut head: Vec<u8> = Vec::new();
    GgufHeader {
        version: GGUF_VERSION,
        tensor_count: tensors.len() as u64,
        metadata_kv_count: metadata.len() as u64,
    }
    .write_to(&mut head)?;
    for (k, v) in &metadata {
        write_metadata_kv(&mut head, k, v)?;
    }

    // Tensor infos with cumulative, alignment-padded offsets.
    let mut running: u64 = 0;
    for t in &tensors {
        write_string(&mut head, &t.name);
        head.extend_from_slice(&(t.dims.len() as u32).to_le_bytes());
        for d in &t.dims {
            head.extend_from_slice(&d.to_le_bytes());
        }
        head.extend_from_slice(&t.dtype.to_le_bytes());
        head.extend_from_slice(&running.to_le_bytes());
        running = running.saturating_add(t.data.len() as u64);
        running = running
            .saturating_add(padding_for_alignment(running as usize, GGUF_DEFAULT_ALIGNMENT) as u64);
    }

    let mut out = head;
    // Pad the header section to alignment, then write each tensor's bytes
    // followed by its own alignment padding (mirrors the offset math above).
    let header_pad = padding_for_alignment(out.len(), GGUF_DEFAULT_ALIGNMENT);
    out.extend(std::iter::repeat(0u8).take(header_pad));
    for t in &tensors {
        out.extend_from_slice(&t.data);
        let data_pad = padding_for_alignment(t.data.len(), GGUF_DEFAULT_ALIGNMENT);
        out.extend(std::iter::repeat(0u8).take(data_pad));
    }

    std::fs::write(output, &out)
        .map_err(|e| AprenderError::Io(io::Error::new(e.kind(), e.to_string())))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::gguf::types::{export_tensors_to_gguf, GgmlType, GgufTensor};

    fn write_part(path: &Path, tensors: &[GgufTensor], meta: &[(String, GgufValue)]) {
        let mut buf = Vec::new();
        export_tensors_to_gguf(&mut buf, tensors, meta).expect("export part");
        std::fs::write(path, &buf).expect("write part");
    }

    /// FT-MERGE-001..003: a 2-part split round-trips — tensors unioned with bytes
    /// preserved, split.* stripped, general.* kept.
    #[test]
    fn merge_two_part_roundtrip() {
        let dir = std::env::temp_dir().join(format!("apr-merge-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("mkdir");
        let p0 = dir.join("model-00001-of-00002.gguf");
        let p1 = dir.join("model-00002-of-00002.gguf");
        let merged = dir.join("model.gguf");

        // F32 tensor (16B) in part 0; Q4_0 tensor (36B) in part 1.
        let a_data = vec![1u8; 16];
        let b_data = vec![2u8; 36];
        let a = GgufTensor {
            name: "blk.0.weight".into(),
            shape: vec![4],
            dtype: GgmlType::F32,
            data: a_data.clone(),
        };
        let b = GgufTensor {
            name: "blk.1.weight".into(),
            shape: vec![64],
            dtype: GgmlType::Q4_0,
            data: b_data.clone(),
        };

        write_part(
            &p0,
            &[a],
            &[
                (
                    "general.architecture".into(),
                    GgufValue::String("llama".into()),
                ),
                ("split.no".into(), GgufValue::Uint16(0)),
                ("split.count".into(), GgufValue::Uint16(2)),
            ],
        );
        write_part(
            &p1,
            &[b],
            &[
                ("split.no".into(), GgufValue::Uint16(1)),
                ("split.count".into(), GgufValue::Uint16(2)),
            ],
        );

        merge_gguf_shards(&[p0, p1], &merged).expect("merge");

        let r = GgufReader::from_file(&merged).expect("re-read merged");

        // FT-MERGE-001: union of tensors across parts.
        let names: Vec<&str> = r.tensors.iter().map(|t| t.name.as_str()).collect();
        assert!(
            names.contains(&"blk.0.weight") && names.contains(&"blk.1.weight"),
            "merged file must contain tensors from BOTH parts, got {names:?}"
        );

        // FT-MERGE-002: split.* stripped, general.* preserved.
        assert!(
            !r.metadata.keys().any(|k| k.starts_with("split.")),
            "split.* metadata must be stripped from the merged file"
        );
        assert!(
            r.metadata.contains_key("general.architecture"),
            "general.* metadata must be preserved"
        );

        // FT-MERGE-003: tensor bytes preserved (read the real prefix of each).
        for (name, want) in [("blk.0.weight", &a_data), ("blk.1.weight", &b_data)] {
            let m = r
                .tensors
                .iter()
                .find(|t| t.name == name)
                .unwrap_or_else(|| panic!("tensor {name} missing"));
            let start = r.data_offset + m.offset as usize;
            let got = &r.data[start..start + want.len()];
            assert_eq!(
                got,
                want.as_slice(),
                "tensor {name} bytes must survive merge"
            );
        }

        std::fs::remove_dir_all(&dir).ok();
    }
}
