// merge-output-fidelity-v1: what `apr_merge` writes, and what it refuses to merge.
//
// Before MOF-002 every merge went through `save_safetensors`, so `-o merged.apr`
// produced metadata-free F32 SafeTensors (first bytes `23 eb 00 00`) that
// `apr inspect` then read as an inferred llama. MOF-004 (dtype preservation) is
// still open: the merged tensors are F32 in either format.

/// True when the output path asks for the APR container (`.apr`, any case).
fn merge_output_is_apr(path: &Path) -> bool {
    path.extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("apr"))
}

/// Read the APR v2 metadata of `path` without loading its tensors: 64-byte
/// header, then `metadata_size` bytes at `metadata_offset`.
///
/// `Ok(None)` when the file is not APR v2 (SafeTensors, GGUF, APR v1).
fn read_apr_metadata(path: &Path) -> Result<Option<AprV2Metadata>> {
    use crate::format::v2::{AprV2Header, HEADER_SIZE_V2, MAGIC_V2, MAX_METADATA_SIZE};
    use std::io::{Read, Seek, SeekFrom};

    let fail = |what: &str, e: &dyn std::fmt::Display| AprenderError::FormatError {
        message: format!("{}: {what}: {e}", path.display()),
    };
    let mut file = fs::File::open(path).map_err(|e| fail("cannot open", &e))?;
    let mut head = [0u8; HEADER_SIZE_V2];
    if file.read_exact(&mut head).is_err() || head[0..4] != MAGIC_V2 {
        return Ok(None);
    }
    let header = AprV2Header::from_bytes(&head).map_err(|e| fail("bad APR v2 header", &e))?;
    if !header.verify_checksum() {
        return Err(fail("APR v2 header checksum mismatch", &"refusing to merge"));
    }
    let size = header.metadata_size as usize;
    if size > MAX_METADATA_SIZE {
        return Err(fail("APR v2 metadata too large", &size));
    }
    let mut buf = vec![0u8; size];
    file.seek(SeekFrom::Start(header.metadata_offset))
        .and_then(|_| file.read_exact(&mut buf))
        .map_err(|e| fail("cannot read APR v2 metadata", &e))?;
    AprV2Metadata::from_json(&buf)
        .map(Some)
        .map_err(|e| fail("bad APR v2 metadata", &e))
}

/// FALSIFY-MOF-005: two inputs whose APR metadata names different architectures
/// are different models, however well their tensor shapes line up. Each field
/// is compared only when both sides declare it.
///
/// Returns the metadata of the first APR input, to be carried into the output.
fn check_merge_architectures(paths: &[&Path]) -> Result<Option<AprV2Metadata>> {
    let mut first: Option<(&Path, AprV2Metadata)> = None;
    for &path in paths {
        let Some(meta) = read_apr_metadata(path)? else {
            continue;
        };
        let Some((first_path, first_meta)) = &first else {
            first = Some((path, meta));
            continue;
        };
        let fields = [
            ("architecture", &first_meta.architecture, &meta.architecture),
            (
                "hf_architecture",
                &first_meta.hf_architecture,
                &meta.hf_architecture,
            ),
        ];
        for (field, a, b) in fields {
            if let (Some(a), Some(b)) = (a, b) {
                if a != b {
                    return Err(AprenderError::FormatError {
                        message: format!(
                            "refusing to merge different architectures: {} has {field} {a}, \
                             {} has {field} {b} (merge-output-fidelity-v1 MOF-005)",
                            first_path.display(),
                            path.display()
                        ),
                    });
                }
            }
        }
    }
    Ok(first.map(|(_, meta)| meta))
}

/// Write the merged tensors in the format the output extension names.
///
/// `.apr` gets an APR v2 file carrying the first APR input's metadata
/// (FALSIFY-MOF-002/003), plus the merge provenance; anything else keeps the
/// SafeTensors writer.
fn save_merged_model(
    output_path: &Path,
    merged: &BTreeMap<String, (Vec<f32>, Vec<usize>)>,
    source_metadata: Option<AprV2Metadata>,
    strategy: MergeStrategy,
    inputs: &[&Path],
) -> Result<()> {
    if !merge_output_is_apr(output_path) {
        return save_safetensors(output_path, merged).map_err(|e| AprenderError::FormatError {
            message: format!("Failed to save merged model: {e}"),
        });
    }
    let mut metadata = source_metadata.unwrap_or_default();
    // Tensors are written as F32 below; a copied quantization block would lie.
    metadata.quantization = None;
    metadata.canonicalize_hf_aliases();
    metadata.custom.insert(
        "merge_strategy".to_string(),
        serde_json::Value::String(format!("{strategy:?}")),
    );
    metadata.custom.insert(
        "merge_inputs".to_string(),
        serde_json::Value::Array(
            inputs
                .iter()
                .map(|p| serde_json::Value::String(p.display().to_string()))
                .collect(),
        ),
    );
    let mut writer = AprV2Writer::new(metadata);
    for (name, (data, shape)) in merged {
        writer.add_f32_tensor(name.clone(), shape.clone(), data);
    }
    let bytes = writer.write().map_err(|e| AprenderError::FormatError {
        message: format!("Failed to encode merged model as APR v2: {e}"),
    })?;
    fs::write(output_path, bytes).map_err(|e| AprenderError::FormatError {
        message: format!("Failed to write {}: {e}", output_path.display()),
    })
}

#[cfg(test)]
#[path = "merge_output_tests.rs"]
mod merge_output_tests;
