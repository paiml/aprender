//! #1893: Merge sharded GGUF parts (`-NNNNN-of-MMMMM.gguf`) into one GGUF file.
//!
//! Sharded GGUFs carry no `index.json`; each part is a complete GGUF holding a
//! SUBSET of tensors plus `split.*` metadata. Merging the parts into a single
//! file lets the existing single-file loader (`GGUFModel::from_path`) run them
//! unchanged — no inference-hot-path refactor (the codebase's #1 garbage-output
//! risk class).
//!
//! **Type-agnostic data.** Tensor data is copied as raw byte ranges sized from
//! the source part's own offset table (the gap to the next tensor's offset),
//! then re-padded to GGUF alignment in the merged file. So EVERY ggml quant type
//! works (Q5_K / Q3_K / IQ\* included) regardless of the `GgmlType` enum.
//!
//! **Lossless metadata.** Part-0 metadata is read with [`GgufReader::from_file_full`]
//! (no architecture whitelist) so arbitrary `<arch>.*` config keys (gemma.*,
//! phi3.*, deepseek2.*, …) survive — without this the merged file would be
//! unloadable for any architecture outside the reader's parse whitelist.
//! `split.*` and `general.alignment` keys are stripped so the merged file is
//! self-consistent at the default 32-byte alignment.
//!
//! **Bounded memory.** The output is streamed to disk and each part is held in
//! RAM only while its tensors are copied (peak ≈ the largest single part, not
//! the whole model) — required for the 7B+ sharded models this targets.

use super::reader::{GgufReader, GgufTensorMeta};
use super::types::{
    padding_for_alignment, write_metadata_kv, GgufHeader, GgufValue, GGUF_DEFAULT_ALIGNMENT,
    GGUF_VERSION,
};
use crate::error::{AprenderError, Result};
use std::collections::HashSet;
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};

fn invalid(msg: String) -> AprenderError {
    AprenderError::Io(io::Error::new(io::ErrorKind::InvalidData, msg))
}

fn io_err(e: io::Error) -> AprenderError {
    AprenderError::Io(io::Error::new(e.kind(), e.to_string()))
}

/// Write a length-prefixed UTF-8 string (GGUF spec §7).
fn write_string(buf: &mut Vec<u8>, s: &str) {
    buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
    buf.extend_from_slice(s.as_bytes());
}

/// Where a merged tensor's data lives: the source part and its absolute byte
/// range within that part's file (so pass 2 can re-read and stream it).
struct TensorPlan {
    name: String,
    dims: Vec<u64>,
    dtype: u32,
    part: usize,
    abs_start: usize,
    abs_end: usize,
}

/// Merge ordered sharded-GGUF `parts` (the complete set, in part order 1..=N)
/// into a single GGUF written to `output`.
///
/// Metadata is taken verbatim from the first part (ALL keys) with `split.*` /
/// `general.alignment` stripped; tensors are the union across all parts in file
/// order. Duplicate tensor names across parts are rejected (a split must be
/// disjoint).
///
/// # Errors
/// Returns an error if fewer than 2 parts are given, a part fails to parse, a
/// part has corrupt tensor offsets, a tensor name appears in more than one
/// part, or the output cannot be written.
pub fn merge_gguf_shards(parts: &[PathBuf], output: &Path) -> Result<()> {
    if parts.len() < 2 {
        return Err(invalid(format!(
            "merge_gguf_shards needs >= 2 parts, got {}",
            parts.len()
        )));
    }

    let mut plans: Vec<TensorPlan> = Vec::new();
    let mut metadata: Vec<(String, GgufValue)> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();

    // Pass 1: gather metadata (part 0, ALL keys) and tensor plans (all parts).
    // Each part's whole-file buffer is freed at the end of its iteration.
    for (pi, path) in parts.iter().enumerate() {
        let reader = if pi == 0 {
            // keep every <arch>.* config key — the merged model is otherwise
            // unloadable for non-whitelisted architectures.
            GgufReader::from_file_full(path)?
        } else {
            GgufReader::from_file(path)?
        };

        if pi == 0 {
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

        for (j, m) in metas.iter().enumerate() {
            let start = m.offset as usize;
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
            if !seen.insert(m.name.clone()) {
                return Err(invalid(format!(
                    "duplicate tensor '{}' across shards (corrupt or non-disjoint split)",
                    m.name
                )));
            }
            plans.push(TensorPlan {
                name: m.name.clone(),
                dims: m.dims.clone(),
                dtype: m.dtype,
                part: pi,
                abs_start,
                abs_end,
            });
        }
    }

    // Build the header section (header + metadata + tensor infos) in RAM — it is
    // small (no tensor data) and lets the data section align to its exact size.
    let mut head: Vec<u8> = Vec::new();
    GgufHeader {
        version: GGUF_VERSION,
        tensor_count: plans.len() as u64,
        metadata_kv_count: metadata.len() as u64,
    }
    .write_to(&mut head)?;
    for (k, v) in &metadata {
        write_metadata_kv(&mut head, k, v)?;
    }
    let mut running: u64 = 0;
    for t in &plans {
        write_string(&mut head, &t.name);
        head.extend_from_slice(&(t.dims.len() as u32).to_le_bytes());
        for d in &t.dims {
            head.extend_from_slice(&d.to_le_bytes());
        }
        head.extend_from_slice(&t.dtype.to_le_bytes());
        head.extend_from_slice(&running.to_le_bytes());
        let len = (t.abs_end - t.abs_start) as u64;
        running = running.saturating_add(len);
        running = running
            .saturating_add(padding_for_alignment(running as usize, GGUF_DEFAULT_ALIGNMENT) as u64);
    }

    // Pass 2: stream to the output file — header, then each tensor's bytes,
    // re-reading one part at a time so peak RAM ≈ the largest single part.
    let file = File::create(output).map_err(io_err)?;
    let mut w = BufWriter::new(file);
    w.write_all(&head).map_err(io_err)?;
    let header_pad = padding_for_alignment(head.len(), GGUF_DEFAULT_ALIGNMENT);
    if header_pad > 0 {
        w.write_all(&vec![0u8; header_pad]).map_err(io_err)?;
    }

    let mut loaded: Option<(usize, GgufReader)> = None;
    for t in &plans {
        let reload = loaded.as_ref().map_or(true, |(pi, _)| *pi != t.part);
        if reload {
            loaded = Some((t.part, GgufReader::from_file(&parts[t.part])?));
        }
        let reader = &loaded.as_ref().expect("part just loaded").1;
        if t.abs_end > reader.data.len() {
            return Err(invalid(format!(
                "shard {} shorter than expected on re-read",
                parts[t.part].display()
            )));
        }
        let block = &reader.data[t.abs_start..t.abs_end];
        w.write_all(block).map_err(io_err)?;
        let pad = padding_for_alignment(block.len(), GGUF_DEFAULT_ALIGNMENT);
        if pad > 0 {
            w.write_all(&vec![0u8; pad]).map_err(io_err)?;
        }
    }
    w.flush().map_err(io_err)?;
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

    fn tmpdir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("apr-merge-{}-{}", tag, std::process::id()));
        std::fs::create_dir_all(&d).expect("mkdir");
        d
    }

    /// FT-MERGE-001..003: a 2-part split round-trips — tensors unioned with bytes
    /// preserved, split.* stripped, general.* kept.
    #[test]
    fn merge_two_part_roundtrip() {
        let dir = tmpdir("rt");
        let p0 = dir.join("model-00001-of-00002.gguf");
        let p1 = dir.join("model-00002-of-00002.gguf");
        let merged = dir.join("model.gguf");

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

        let r = GgufReader::from_file_full(&merged).expect("re-read merged");
        let names: Vec<&str> = r.tensors.iter().map(|t| t.name.as_str()).collect();
        assert!(
            names.contains(&"blk.0.weight") && names.contains(&"blk.1.weight"),
            "merged file must contain tensors from BOTH parts, got {names:?}"
        );
        assert!(
            !r.metadata.keys().any(|k| k.starts_with("split.")),
            "split.* metadata must be stripped"
        );
        assert!(
            r.metadata.contains_key("general.architecture"),
            "general.* metadata must be preserved"
        );
        for (name, want) in [("blk.0.weight", &a_data), ("blk.1.weight", &b_data)] {
            let m = r
                .tensors
                .iter()
                .find(|t| t.name == name)
                .unwrap_or_else(|| panic!("tensor {name} missing"));
            let start = r.data_offset + m.offset as usize;
            assert_eq!(
                &r.data[start..start + want.len()],
                want.as_slice(),
                "tensor {name} bytes must survive merge"
            );
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// FT-MERGE-004 (release-blocker regression): config metadata for a
    /// NON-whitelisted architecture (gemma) must survive the merge — otherwise
    /// the merged model is unloadable ("Missing embedding_length"). The earlier
    /// llama-only test masked this because llama is in the reader whitelist.
    #[test]
    fn merge_preserves_nonwhitelisted_arch_metadata() {
        let dir = tmpdir("gemma");
        let p0 = dir.join("model-00001-of-00002.gguf");
        let p1 = dir.join("model-00002-of-00002.gguf");
        let merged = dir.join("model.gguf");

        let t0 = GgufTensor {
            name: "blk.0.weight".into(),
            shape: vec![4],
            dtype: GgmlType::F32,
            data: vec![7u8; 16],
        };
        let t1 = GgufTensor {
            name: "blk.1.weight".into(),
            shape: vec![4],
            dtype: GgmlType::F32,
            data: vec![9u8; 16],
        };
        write_part(
            &p0,
            &[t0],
            &[
                (
                    "general.architecture".into(),
                    GgufValue::String("gemma".into()),
                ),
                ("gemma.embedding_length".into(), GgufValue::Uint32(2048)),
                ("gemma.block_count".into(), GgufValue::Uint32(18)),
                ("gemma.attention.head_count".into(), GgufValue::Uint32(8)),
                ("split.no".into(), GgufValue::Uint16(0)),
                ("split.count".into(), GgufValue::Uint16(2)),
            ],
        );
        write_part(&p1, &[t1], &[("split.no".into(), GgufValue::Uint16(1))]);

        merge_gguf_shards(&[p0, p1], &merged).expect("merge");

        // Must read back with from_file_full (the merged file carries gemma.*).
        let r = GgufReader::from_file_full(&merged).expect("re-read merged");
        for key in [
            "gemma.embedding_length",
            "gemma.block_count",
            "gemma.attention.head_count",
        ] {
            assert!(
                r.metadata.contains_key(key),
                "merged gemma model must retain {key}; got {:?}",
                r.metadata.keys().collect::<Vec<_>>()
            );
        }
        assert!(!r.metadata.keys().any(|k| k.starts_with("split.")));
        std::fs::remove_dir_all(&dir).ok();
    }

    /// FT-MERGE-005: a tensor name appearing in two parts is rejected (a split
    /// must be disjoint — duplicates would silently inflate tensor_count and
    /// ship wrong weights).
    #[test]
    fn merge_rejects_duplicate_tensor_names() {
        let dir = tmpdir("dup");
        let p0 = dir.join("model-00001-of-00002.gguf");
        let p1 = dir.join("model-00002-of-00002.gguf");
        let merged = dir.join("model.gguf");

        let dup = |fill: u8| GgufTensor {
            name: "blk.0.weight".into(),
            shape: vec![4],
            dtype: GgmlType::F32,
            data: vec![fill; 16],
        };
        write_part(
            &p0,
            &[dup(1)],
            &[(
                "general.architecture".into(),
                GgufValue::String("llama".into()),
            )],
        );
        write_part(&p1, &[dup(2)], &[]);

        let res = merge_gguf_shards(&[p0, p1], &merged);
        assert!(
            res.is_err(),
            "duplicate tensor name across shards must be rejected"
        );
        std::fs::remove_dir_all(&dir).ok();
    }
}
