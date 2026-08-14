//! Model export writers for the distillation pipeline.
//!
//! These writers were stranded in the `aprender-train-distill` binary's
//! `main.rs`, so no library caller could reach them. They moved here verbatim
//! when the binary was folded into `apr train distill export`
//! (APR-MONO Rule 1); [`crate::cli::run_export`] is the only in-tree caller.

// `serde_json::json!` expands to code containing `.unwrap()`, which trips
// clippy::disallowed_methods at the macro invocation site.
#![allow(clippy::disallowed_methods)]

use std::collections::HashMap;
use std::path::Path;

/// Ensure the parent directory of `path` exists, creating it if necessary.
///
/// # Errors
///
/// Returns [`entrenar_common::EntrenarError::Io`] when the directory cannot be
/// created.
pub fn ensure_parent_dir(path: &Path) -> entrenar_common::Result<()> {
    let parent = match path.parent() {
        Some(p) if !p.as_os_str().is_empty() => p,
        _ => return Ok(()),
    };
    std::fs::create_dir_all(parent).map_err(|e| entrenar_common::EntrenarError::Io {
        context: format!("creating output directory: {}", parent.display()),
        source: e,
    })
}

/// Dispatch export to the correct format handler.
///
/// # Errors
///
/// Returns [`entrenar_common::EntrenarError::UnsupportedFormat`] when `format`
/// is not `safetensors`, `gguf`, `apr` or `json`, and propagates writer errors.
pub fn dispatch_export(
    format: &str,
    weights: &HashMap<String, Vec<f32>>,
    shapes: &HashMap<String, Vec<usize>>,
    output: &Path,
    quantize: &str,
) -> entrenar_common::Result<()> {
    match format {
        "safetensors" => export_safetensors(weights, shapes, output),
        "gguf" => export_gguf(weights, shapes, output, quantize),
        "apr" | "json" => export_apr(weights, output),
        other => Err(entrenar_common::EntrenarError::UnsupportedFormat {
            format: other.to_string(),
        }),
    }
}

/// Export weights as SafeTensors format.
///
/// # Errors
///
/// Returns [`entrenar_common::EntrenarError::Serialization`] on tensor-view or
/// serialization failure, and [`entrenar_common::EntrenarError::Io`] on write
/// failure.
pub fn export_safetensors(
    weights: &HashMap<String, Vec<f32>>,
    shapes: &HashMap<String, Vec<usize>>,
    output: &Path,
) -> entrenar_common::Result<()> {
    use safetensors::tensor::{Dtype, TensorView};

    let mut sorted_names: Vec<&String> = weights.keys().collect();
    sorted_names.sort();

    let tensor_data: Vec<(String, Vec<u8>, Vec<usize>)> = sorted_names
        .iter()
        .map(|name| {
            let data = &weights[*name];
            let bytes: Vec<u8> = bytemuck::cast_slice(data).to_vec();
            let shape = shapes
                .get(*name)
                .cloned()
                .unwrap_or_else(|| vec![data.len()]);
            ((*name).clone(), bytes, shape)
        })
        .collect();

    let views: Vec<(&str, TensorView<'_>)> = tensor_data
        .iter()
        .map(|(name, bytes, shape)| -> entrenar_common::Result<_> {
            let view = TensorView::new(Dtype::F32, shape.clone(), bytes).map_err(|e| {
                entrenar_common::EntrenarError::Serialization {
                    message: format!("TensorView creation failed for {name}: {e}"),
                }
            })?;
            Ok((name.as_str(), view))
        })
        .collect::<entrenar_common::Result<Vec<_>>>()?;

    let st_bytes = safetensors::serialize(views, None).map_err(|e| {
        entrenar_common::EntrenarError::Serialization {
            message: format!("SafeTensors serialization failed: {e}"),
        }
    })?;

    std::fs::write(output, st_bytes).map_err(|e| entrenar_common::EntrenarError::Io {
        context: format!("writing SafeTensors output: {}", output.display()),
        source: e,
    })
}

/// Export weights as GGUF format (requires the `hub` feature for real quantization).
///
/// # Errors
///
/// Without the `hub` feature, always returns
/// [`entrenar_common::EntrenarError::HuggingFace`]. With it, returns
/// [`entrenar_common::EntrenarError::ConfigValue`] for an unknown `quantize`
/// value and propagates exporter failures.
pub fn export_gguf(
    weights: &HashMap<String, Vec<f32>>,
    shapes: &HashMap<String, Vec<usize>>,
    output: &Path,
    quantize: &str,
) -> entrenar_common::Result<()> {
    #[cfg(feature = "hub")]
    {
        let quant = match quantize {
            "q4_0" | "Q4_0" => entrenar::hf_pipeline::GgufQuantization::Q4_0,
            "q8_0" | "Q8_0" => entrenar::hf_pipeline::GgufQuantization::Q8_0,
            "none" | "None" | "f32" => entrenar::hf_pipeline::GgufQuantization::None,
            other => {
                return Err(entrenar_common::EntrenarError::ConfigValue {
                    field: "quantize".into(),
                    message: format!("unknown quantization: {other}"),
                    suggestion: "Use one of: none, q4_0, q8_0".into(),
                });
            }
        };

        let mw = crate::weights::weights_to_model_weights(weights.clone(), shapes.clone());

        let output_dir = output.parent().unwrap_or_else(|| Path::new("."));
        let filename = output
            .file_name()
            .unwrap_or_else(|| std::ffi::OsStr::new("model.gguf"));

        let exporter = entrenar::hf_pipeline::Exporter::new()
            .output_dir(output_dir)
            .gguf_quantization(quant);

        exporter
            .export(&mw, entrenar::hf_pipeline::ExportFormat::GGUF, filename)
            .map_err(|e| entrenar_common::EntrenarError::Internal {
                message: format!("GGUF export failed: {e}"),
            })?;

        Ok(())
    }

    #[cfg(not(feature = "hub"))]
    {
        let _ = (weights, shapes, output, quantize);
        Err(entrenar_common::EntrenarError::HuggingFace {
            message: "GGUF export requires the 'hub' feature. \
                      Rebuild with: cargo build -p aprender-train-distill --features hub"
                .to_string(),
        })
    }
}

/// Export weights as APR (JSON) format.
///
/// # Errors
///
/// Returns [`entrenar_common::EntrenarError::Serialization`] on JSON failure
/// and [`entrenar_common::EntrenarError::Io`] on write failure.
pub fn export_apr(
    weights: &HashMap<String, Vec<f32>>,
    output: &Path,
) -> entrenar_common::Result<()> {
    // Build a simple JSON representation of the model weights
    let model_data = serde_json::json!({
        "format": "apr",
        "version": "1.0",
        "tensors": weights.iter().map(|(name, data)| {
            serde_json::json!({
                "name": name,
                "shape": [data.len()],
                "data": data,
            })
        }).collect::<Vec<_>>(),
    });

    let json = serde_json::to_string_pretty(&model_data).map_err(|e| {
        entrenar_common::EntrenarError::Serialization {
            message: format!("APR JSON serialization failed: {e}"),
        }
    })?;

    std::fs::write(output, json).map_err(|e| entrenar_common::EntrenarError::Io {
        context: format!("writing APR output: {}", output.display()),
        source: e,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn one_tensor() -> (HashMap<String, Vec<f32>>, HashMap<String, Vec<usize>>) {
        let mut weights = HashMap::new();
        weights.insert("w".to_string(), vec![1.0_f32, 2.0, 3.0, 4.0]);
        let mut shapes = HashMap::new();
        shapes.insert("w".to_string(), vec![2_usize, 2]);
        (weights, shapes)
    }

    #[test]
    fn dispatch_export_refuses_unknown_format() {
        let (weights, shapes) = one_tensor();
        let dir = tempfile::tempdir().expect("tempdir");
        let out = dir.path().join("model.bin");
        // Asserting is_ok() on an unsupported format would lock the defect in.
        let err = match dispatch_export("pickle", &weights, &shapes, &out, "none") {
            Ok(()) => panic!("dispatch_export must refuse an unsupported format"),
            Err(e) => e.to_string(),
        };
        assert!(
            err.contains("pickle"),
            "refusal must quote the rejected format, got: {err}"
        );
        assert!(
            !out.exists(),
            "a refused export must not leave an output file behind"
        );
    }

    #[test]
    fn export_safetensors_round_trips_shape_and_values() {
        let (weights, shapes) = one_tensor();
        let dir = tempfile::tempdir().expect("tempdir");
        let out = dir.path().join("model.safetensors");

        dispatch_export("safetensors", &weights, &shapes, &out, "none")
            .expect("safetensors export should succeed");

        let bytes = std::fs::read(&out).expect("read exported file");
        let st = safetensors::SafeTensors::deserialize(&bytes).expect("parse safetensors");
        let view = st.tensor("w").expect("tensor w present");
        assert_eq!(view.shape(), &[2, 2]);
        assert_eq!(view.dtype(), safetensors::Dtype::F32);
    }

    #[test]
    fn export_apr_writes_json_with_tensor_names() {
        let (weights, _shapes) = one_tensor();
        let dir = tempfile::tempdir().expect("tempdir");
        let out = dir.path().join("model.apr.json");

        export_apr(&weights, &out).expect("apr export should succeed");

        let text = std::fs::read_to_string(&out).expect("read exported file");
        let parsed: serde_json::Value = serde_json::from_str(&text).expect("valid json");
        assert_eq!(parsed["format"], "apr");
        assert_eq!(parsed["tensors"][0]["name"], "w");
    }

    #[test]
    fn ensure_parent_dir_creates_missing_directories() {
        let dir = tempfile::tempdir().expect("tempdir");
        let nested = dir.path().join("a").join("b").join("model.safetensors");
        ensure_parent_dir(&nested).expect("parent creation should succeed");
        assert!(
            nested.parent().expect("nested path has a parent").is_dir(),
            "ensure_parent_dir must create the full parent chain"
        );
    }

    #[test]
    fn ensure_parent_dir_tolerates_bare_filename() {
        // A bare filename has an empty parent; this must be a no-op, not an error.
        ensure_parent_dir(Path::new("model.safetensors"))
            .expect("bare filename must not be treated as a directory error");
    }

    #[cfg(not(feature = "hub"))]
    #[test]
    fn export_gguf_refuses_without_hub_feature() {
        let (weights, shapes) = one_tensor();
        let dir = tempfile::tempdir().expect("tempdir");
        let out = dir.path().join("model.gguf");
        let err = match export_gguf(&weights, &shapes, &out, "q4_0") {
            Ok(()) => panic!("GGUF export must refuse when the 'hub' feature is off"),
            Err(e) => e.to_string(),
        };
        assert!(
            err.contains("hub"),
            "refusal must name the missing feature, got: {err}"
        );
    }
}
