//! Quantize-Export pipeline
//!
//! Quantizes model weights and exports them as GGUF files in a single operation.

use crate::hf_pipeline::error::{FetchError, Result};
use crate::hf_pipeline::export::gguf_writer::GgufQuantization;
use std::path::Path;

use super::exporter::Exporter;
use super::format::ExportFormat;
use super::result::ExportResult;
use super::weights::ModelWeights;

/// Result of the quantize-export pipeline
#[derive(Debug, Clone)]
pub struct QuantExportResult {
    /// Export result with file path and size
    pub export: ExportResult,
    /// Quantization mode used
    pub quantization: GgufQuantization,
    /// Generated README content (if any)
    pub readme: Option<String>,
}

/// Quantize model weights and export to GGUF format
///
/// Performs the full quantize→export pipeline:
/// 1. Quantize all tensors according to the config
/// 2. Export as GGUF with tensor data
/// 3. Generate a README with quantization metadata
pub fn quantize_and_export(
    weights: &ModelWeights,
    quantization: GgufQuantization,
    output_dir: impl AsRef<Path>,
    filename: impl AsRef<Path>,
) -> Result<QuantExportResult> {
    let output_dir = output_dir.as_ref();

    // Determine output filename
    let filename = filename.as_ref();

    // Build exporter with quantization config
    let exporter = Exporter::new().output_dir(output_dir).gguf_quantization(quantization);

    let export = exporter.export(weights, ExportFormat::GGUF, filename)?;

    // Generate README with quantization metadata
    let readme = generate_quant_readme(weights, quantization, &export);

    // Write README alongside the model
    let readme_path = output_dir.join("README.md");
    std::fs::write(&readme_path, &readme).map_err(|e| FetchError::GgufWriteError {
        message: format!("Failed to write README: {e}"),
    })?;

    Ok(QuantExportResult { export, quantization, readme: Some(readme) })
}

/// Generate a README with quantization metadata
fn generate_quant_readme(
    weights: &ModelWeights,
    quantization: GgufQuantization,
    export: &ExportResult,
) -> String {
    let quant_name = match quantization {
        GgufQuantization::None => "F32 (unquantized)",
        GgufQuantization::Q4_0 => "Q4_0 (4-bit)",
        GgufQuantization::Q8_0 => "Q8_0 (8-bit)",
    };

    let model_name = weights.metadata.model_name.as_deref().unwrap_or("Unknown Model");

    let arch = weights.metadata.architecture.as_deref().unwrap_or("unknown");

    format!(
        "---\ntags:\n- entrenar\n- gguf\n- quantized\n---\n\n\
         # {model_name} ({quant_name})\n\n\
         Quantized with [Entrenar](https://github.com/paiml/entrenar).\n\n\
         ## Model Details\n\n\
         | Property | Value |\n\
         |----------|-------|\n\
         | Architecture | {arch} |\n\
         | Parameters | {} |\n\
         | Quantization | {quant_name} |\n\
         | File Size | {} |\n\
         | Tensors | {} |\n\
         | Format | GGUF v3 |\n",
        weights.metadata.num_params,
        export.size_human(),
        export.num_tensors,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hf_pipeline::export::weights::{ModelMetadata, ModelWeights};
    use tempfile::TempDir;

    fn make_test_weights() -> ModelWeights {
        let mut weights = ModelWeights::new();
        weights.add_tensor("layer.0.weight", vec![1.0; 256], vec![16, 16]);
        weights.add_tensor("layer.0.bias", vec![0.1; 16], vec![16]);
        weights.metadata = ModelMetadata {
            model_name: Some("test-model".to_string()),
            architecture: Some("llama".to_string()),
            num_params: 272,
            ..Default::default()
        };
        weights
    }

    #[test]
    fn test_quantize_export_f32() {
        let weights = make_test_weights();
        let tmp = TempDir::new().expect("temp file creation should succeed");

        let result =
            quantize_and_export(&weights, GgufQuantization::None, tmp.path(), "model.gguf")
                .expect("operation should succeed");

        assert_eq!(result.quantization, GgufQuantization::None);
        assert!(result.export.size_bytes > 0);
        assert!(result.readme.is_some());
        assert!(tmp.path().join("model.gguf").exists());
        assert!(tmp.path().join("README.md").exists());
    }

    #[test]
    fn test_quantize_export_q4_0() {
        let weights = make_test_weights();
        let tmp = TempDir::new().expect("temp file creation should succeed");

        let result =
            quantize_and_export(&weights, GgufQuantization::Q4_0, tmp.path(), "model-q4.gguf")
                .expect("operation should succeed");

        assert_eq!(result.quantization, GgufQuantization::Q4_0);
        assert!(result.export.size_bytes > 0);
    }

    #[test]
    fn test_quantize_export_q8_0() {
        let weights = make_test_weights();
        let tmp = TempDir::new().expect("temp file creation should succeed");

        let result =
            quantize_and_export(&weights, GgufQuantization::Q8_0, tmp.path(), "model-q8.gguf")
                .expect("operation should succeed");

        assert_eq!(result.quantization, GgufQuantization::Q8_0);
    }

    #[test]
    fn test_quantize_export_readme_content() {
        let weights = make_test_weights();
        let tmp = TempDir::new().expect("temp file creation should succeed");

        let result =
            quantize_and_export(&weights, GgufQuantization::Q4_0, tmp.path(), "model.gguf")
                .expect("operation should succeed");

        let readme = result.readme.expect("operation should succeed");
        assert!(readme.contains("test-model"));
        assert!(readme.contains("Q4_0"));
        assert!(readme.contains("llama"));
        assert!(readme.contains("entrenar"));
    }

    #[test]
    fn test_quantize_export_q4_smaller_than_f32() {
        let weights = make_test_weights();
        let tmp_f32 = TempDir::new().expect("temp file creation should succeed");
        let tmp_q4 = TempDir::new().expect("temp file creation should succeed");

        let f32_result =
            quantize_and_export(&weights, GgufQuantization::None, tmp_f32.path(), "model.gguf")
                .expect("operation should succeed");
        let q4_result =
            quantize_and_export(&weights, GgufQuantization::Q4_0, tmp_q4.path(), "model.gguf")
                .expect("operation should succeed");

        assert!(
            q4_result.export.size_bytes < f32_result.export.size_bytes,
            "Q4_0 ({}) should be smaller than F32 ({})",
            q4_result.export.size_bytes,
            f32_result.export.size_bytes
        );
    }

    // =====================================================================
    // Falsification: pipeline roundtrip via verify_gguf
    // =====================================================================

    #[test]
    fn test_falsify_pipeline_f32_gguf_is_valid() {
        use crate::hf_pipeline::export::gguf_verify::verify_gguf;

        let weights = make_test_weights();
        let tmp = TempDir::new().expect("temp file creation should succeed");
        quantize_and_export(&weights, GgufQuantization::None, tmp.path(), "f32.gguf")
            .expect("operation should succeed");

        let file_data =
            std::fs::read(tmp.path().join("f32.gguf")).expect("file read should succeed");
        let summary = verify_gguf(&file_data).expect("operation should succeed");

        assert_eq!(summary.version, 3);
        assert_eq!(summary.tensor_count, 2);
        // Metadata: architecture + name + parameter_count = 3
        assert_eq!(summary.metadata_count, 3);
        // Tensors sorted alphabetically
        assert_eq!(summary.tensors[0].name, "layer.0.bias");
        assert_eq!(summary.tensors[1].name, "layer.0.weight");
        // Both F32
        assert_eq!(summary.tensors[0].dtype, 0);
        assert_eq!(summary.tensors[1].dtype, 0);
        // Shapes
        assert_eq!(summary.tensors[0].shape, vec![16]);
        assert_eq!(summary.tensors[1].shape, vec![16, 16]);
    }

    #[test]
    fn test_falsify_pipeline_q4_0_gguf_is_valid() {
        use crate::hf_pipeline::export::gguf_verify::verify_gguf;

        let weights = make_test_weights();
        let tmp = TempDir::new().expect("temp file creation should succeed");
        quantize_and_export(&weights, GgufQuantization::Q4_0, tmp.path(), "q4.gguf")
            .expect("operation should succeed");

        let file_data =
            std::fs::read(tmp.path().join("q4.gguf")).expect("file read should succeed");
        let summary = verify_gguf(&file_data).expect("operation should succeed");

        assert_eq!(summary.tensor_count, 2);
        // Both Q4_0
        assert_eq!(summary.tensors[0].dtype, 2);
        assert_eq!(summary.tensors[1].dtype, 2);
    }

    #[test]
    fn test_falsify_pipeline_q8_0_gguf_is_valid() {
        use crate::hf_pipeline::export::gguf_verify::verify_gguf;

        let weights = make_test_weights();
        let tmp = TempDir::new().expect("temp file creation should succeed");
        quantize_and_export(&weights, GgufQuantization::Q8_0, tmp.path(), "q8.gguf")
            .expect("operation should succeed");

        let file_data =
            std::fs::read(tmp.path().join("q8.gguf")).expect("file read should succeed");
        let summary = verify_gguf(&file_data).expect("operation should succeed");

        assert_eq!(summary.tensor_count, 2);
        // Both Q8_0
        assert_eq!(summary.tensors[0].dtype, 8);
        assert_eq!(summary.tensors[1].dtype, 8);
    }

    #[test]
    fn test_falsify_pipeline_q8_smaller_than_f32() {
        let weights = make_test_weights();
        let tmp_f32 = TempDir::new().expect("temp file creation should succeed");
        let tmp_q8 = TempDir::new().expect("temp file creation should succeed");

        let f32_result =
            quantize_and_export(&weights, GgufQuantization::None, tmp_f32.path(), "model.gguf")
                .expect("operation should succeed");
        let q8_result =
            quantize_and_export(&weights, GgufQuantization::Q8_0, tmp_q8.path(), "model.gguf")
                .expect("operation should succeed");

        assert!(
            q8_result.export.size_bytes < f32_result.export.size_bytes,
            "Q8_0 ({}) should be smaller than F32 ({})",
            q8_result.export.size_bytes,
            f32_result.export.size_bytes
        );
    }

    #[test]
    fn test_falsify_pipeline_q4_smaller_than_q8() {
        let weights = make_test_weights();
        let tmp_q4 = TempDir::new().expect("temp file creation should succeed");
        let tmp_q8 = TempDir::new().expect("temp file creation should succeed");

        let q4_result =
            quantize_and_export(&weights, GgufQuantization::Q4_0, tmp_q4.path(), "model.gguf")
                .expect("operation should succeed");
        let q8_result =
            quantize_and_export(&weights, GgufQuantization::Q8_0, tmp_q8.path(), "model.gguf")
                .expect("operation should succeed");

        assert!(
            q4_result.export.size_bytes < q8_result.export.size_bytes,
            "Q4_0 ({}) should be smaller than Q8_0 ({})",
            q4_result.export.size_bytes,
            q8_result.export.size_bytes
        );
    }

    #[test]
    fn test_falsify_pipeline_readme_contains_quantization_mode() {
        let weights = make_test_weights();

        for (quant, expected_str) in [
            (GgufQuantization::None, "F32 (unquantized)"),
            (GgufQuantization::Q4_0, "Q4_0 (4-bit)"),
            (GgufQuantization::Q8_0, "Q8_0 (8-bit)"),
        ] {
            let tmp = TempDir::new().expect("temp file creation should succeed");
            let result = quantize_and_export(&weights, quant, tmp.path(), "model.gguf")
                .expect("operation should succeed");
            let readme = result.readme.expect("operation should succeed");
            assert!(
                readme.contains(expected_str),
                "README for {quant:?} should contain '{expected_str}', got:\n{readme}"
            );
        }
    }

    #[test]
    fn test_falsify_pipeline_f32_data_integrity_through_pipeline() {
        // Verify actual tensor bytes survive the full pipeline
        use crate::hf_pipeline::export::gguf_verify::verify_gguf;

        let mut weights = ModelWeights::new();
        let original: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.1).collect();
        weights.add_tensor("test_data", original.clone(), vec![8, 8]);
        weights.metadata.num_params = 64;

        let tmp = TempDir::new().expect("temp file creation should succeed");
        quantize_and_export(&weights, GgufQuantization::None, tmp.path(), "data.gguf")
            .expect("operation should succeed");

        let file_data =
            std::fs::read(tmp.path().join("data.gguf")).expect("file read should succeed");
        let summary = verify_gguf(&file_data).expect("operation should succeed");
        assert_eq!(summary.tensors[0].name, "test_data");
        assert_eq!(summary.tensors[0].shape, vec![8, 8]);
        assert_eq!(summary.tensors[0].dtype, 0); // F32

        // Extract and verify actual data
        // Find data section: skip header (24) + metadata + tensor info
        let mut pos = 24;
        // Skip metadata
        for _ in 0..summary.metadata_count {
            // Skip key string
            let key_len = u64::from_le_bytes(
                file_data[pos..pos + 8].try_into().expect("conversion should succeed"),
            ) as usize;
            pos += 8 + key_len;
            let value_type = u32::from_le_bytes(
                file_data[pos..pos + 4].try_into().expect("conversion should succeed"),
            );
            pos += 4;
            match value_type {
                4..=6 => pos += 4, // U32/I32/F32
                8 => {
                    let len = u64::from_le_bytes(
                        file_data[pos..pos + 8].try_into().expect("conversion should succeed"),
                    ) as usize;
                    pos += 8 + len;
                }
                10..=12 => pos += 8, // U64/I64/F64
                _ => {}
            }
        }
        // Skip tensor info
        let name_len = u64::from_le_bytes(
            file_data[pos..pos + 8].try_into().expect("conversion should succeed"),
        ) as usize;
        pos += 8 + name_len;
        let n_dims = u32::from_le_bytes(
            file_data[pos..pos + 4].try_into().expect("conversion should succeed"),
        ) as usize;
        pos += 4 + n_dims * 8 + 4 + 8; // dims + dtype + offset

        // Now pos is at the start of tensor data
        let data_start = pos;
        let recovered: Vec<f32> = (0..64)
            .map(|i| {
                let off = data_start + i * 4;
                f32::from_le_bytes(
                    file_data[off..off + 4].try_into().expect("conversion should succeed"),
                )
            })
            .collect();
        assert_eq!(original, recovered, "f32 data must survive pipeline exactly");
    }

    // =====================================================================
    // TIER 3: File size monotonicity across varying tensor counts
    // =====================================================================

    #[test]
    fn test_falsify_pipeline_size_monotonic_with_tensor_count() {
        // For same quant mode, more tensors → larger file
        let mut prev_size = 0u64;
        for n in [1, 2, 4, 8] {
            let mut weights = ModelWeights::new();
            for i in 0..n {
                weights.add_tensor(format!("layer.{i}.weight"), vec![1.0; 64], vec![8, 8]);
            }
            weights.metadata.num_params = n as u64 * 64;

            let tmp = TempDir::new().expect("temp file creation should succeed");
            let result =
                quantize_and_export(&weights, GgufQuantization::None, tmp.path(), "model.gguf")
                    .expect("operation should succeed");

            assert!(
                result.export.size_bytes > prev_size,
                "F32 {n} tensors ({}) must be > prev ({prev_size})",
                result.export.size_bytes
            );
            prev_size = result.export.size_bytes;
        }
    }

    #[test]
    fn test_falsify_pipeline_q4_size_monotonic_with_tensor_count() {
        let mut prev_size = 0u64;
        for n in [1, 2, 4, 8] {
            let mut weights = ModelWeights::new();
            for i in 0..n {
                weights.add_tensor(format!("layer.{i}.weight"), vec![1.0; 64], vec![8, 8]);
            }
            weights.metadata.num_params = n as u64 * 64;

            let tmp = TempDir::new().expect("temp file creation should succeed");
            let result =
                quantize_and_export(&weights, GgufQuantization::Q4_0, tmp.path(), "model.gguf")
                    .expect("operation should succeed");

            assert!(
                result.export.size_bytes > prev_size,
                "Q4_0 {n} tensors ({}) must be > prev ({prev_size})",
                result.export.size_bytes
            );
            prev_size = result.export.size_bytes;
        }
    }

    #[test]
    fn test_falsify_pipeline_size_ordering_at_multiple_scales() {
        // For various tensor element counts, always: Q4 < Q8 < F32
        for n_elements in [32, 128, 512, 1024] {
            let mut weights = ModelWeights::new();
            weights.add_tensor("w", vec![0.5; n_elements], vec![n_elements]);
            weights.metadata.num_params = n_elements as u64;

            let sizes: Vec<(GgufQuantization, u64)> =
                [GgufQuantization::None, GgufQuantization::Q8_0, GgufQuantization::Q4_0]
                    .iter()
                    .map(|&quant| {
                        let tmp = TempDir::new().expect("temp file creation should succeed");
                        let result = quantize_and_export(&weights, quant, tmp.path(), "m.gguf")
                            .expect("operation should succeed");
                        (quant, result.export.size_bytes)
                    })
                    .collect();

            let (_, f32_size) = sizes[0];
            let (_, q8_size) = sizes[1];
            let (_, q4_size) = sizes[2];

            assert!(
                q4_size < q8_size,
                "at {n_elements} elements: Q4={q4_size} must be < Q8={q8_size}"
            );
            assert!(
                q8_size < f32_size,
                "at {n_elements} elements: Q8={q8_size} must be < F32={f32_size}"
            );
        }
    }

    #[test]
    fn test_falsify_pipeline_magic_bytes_all_quant_modes() {
        let weights = make_test_weights();
        for quant in [GgufQuantization::None, GgufQuantization::Q4_0, GgufQuantization::Q8_0] {
            let tmp = TempDir::new().expect("temp file creation should succeed");
            quantize_and_export(&weights, quant, tmp.path(), "model.gguf")
                .expect("operation should succeed");
            let file_data =
                std::fs::read(tmp.path().join("model.gguf")).expect("file read should succeed");
            assert_eq!(&file_data[0..4], b"GGUF", "magic bytes wrong for pipeline {quant:?}");
        }
    }

    #[test]
    fn test_falsify_pipeline_readme_file_size_field() {
        let weights = make_test_weights();
        let tmp = TempDir::new().expect("temp file creation should succeed");
        let result =
            quantize_and_export(&weights, GgufQuantization::None, tmp.path(), "model.gguf")
                .expect("operation should succeed");
        let readme = result.readme.expect("operation should succeed");
        // README should contain the human-readable size string
        let size_str = result.export.size_human();
        assert!(
            readme.contains(&size_str),
            "README should contain size '{size_str}', got:\n{readme}"
        );
    }

    #[test]
    fn test_falsify_pipeline_readme_tensor_count() {
        let weights = make_test_weights();
        let tmp = TempDir::new().expect("temp file creation should succeed");
        let result =
            quantize_and_export(&weights, GgufQuantization::None, tmp.path(), "model.gguf")
                .expect("operation should succeed");
        let readme = result.readme.expect("operation should succeed");
        assert!(
            readme.contains(&format!("{}", result.export.num_tensors)),
            "README should contain tensor count {}",
            result.export.num_tensors
        );
    }

    #[test]
    fn test_falsify_pipeline_readme_has_yaml_frontmatter() {
        let weights = make_test_weights();
        let tmp = TempDir::new().expect("temp file creation should succeed");
        let result =
            quantize_and_export(&weights, GgufQuantization::None, tmp.path(), "model.gguf")
                .expect("operation should succeed");
        let readme = result.readme.expect("operation should succeed");
        assert!(readme.starts_with("---\n"), "README must start with YAML frontmatter");
        assert!(readme.contains("tags:"), "README must have tags in frontmatter");
        assert!(readme.contains("- gguf"), "README frontmatter must tag 'gguf'");
        assert!(readme.contains("- entrenar"), "README frontmatter must tag 'entrenar'");
    }
}
