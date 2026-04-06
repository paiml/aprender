//! Model exporter implementation.

use crate::hf_pipeline::error::{FetchError, Result};
use serde::Serialize;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use super::format::ExportFormat;
use super::gguf_writer::{quantize_to_gguf_bytes, GgufQuantization};
use super::result::ExportResult;
use super::weights::{ModelMetadata, ModelWeights};

/// Model exporter
pub struct Exporter {
    /// Output directory
    pub(super) output_dir: PathBuf,
    /// Default format
    pub(super) default_format: ExportFormat,
    /// Include metadata
    pub(super) include_metadata: bool,
    /// GGUF quantization mode
    pub(super) gguf_quantization: GgufQuantization,
}

impl Default for Exporter {
    fn default() -> Self {
        Self::new()
    }
}

impl Exporter {
    /// Create new exporter
    #[must_use]
    pub fn new() -> Self {
        Self {
            output_dir: PathBuf::from("."),
            default_format: ExportFormat::SafeTensors,
            include_metadata: true,
            gguf_quantization: GgufQuantization::None,
        }
    }

    /// Set output directory
    #[must_use]
    pub fn output_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.output_dir = dir.into();
        self
    }

    /// Set default format
    #[must_use]
    pub fn default_format(mut self, format: ExportFormat) -> Self {
        self.default_format = format;
        self
    }

    /// Set whether to include metadata
    #[must_use]
    pub fn include_metadata(mut self, include: bool) -> Self {
        self.include_metadata = include;
        self
    }

    /// Set GGUF quantization mode
    #[must_use]
    pub fn gguf_quantization(mut self, quant: GgufQuantization) -> Self {
        self.gguf_quantization = quant;
        self
    }

    /// Export weights to file
    pub fn export(
        &self,
        weights: &ModelWeights,
        format: ExportFormat,
        filename: impl AsRef<Path>,
    ) -> Result<ExportResult> {
        let path = self.output_dir.join(filename);

        // Ensure parent directory exists
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| FetchError::ConfigParseError {
                message: format!("Failed to create output directory: {e}"),
            })?;
        }

        match format {
            ExportFormat::SafeTensors => self.export_safetensors(weights, &path),
            ExportFormat::APR => self.export_apr(weights, &path),
            ExportFormat::GGUF => self.export_gguf(weights, &path),
            ExportFormat::PyTorch => Err(FetchError::PickleSecurityRisk),
        }
    }

    /// Export to SafeTensors format
    fn export_safetensors(&self, weights: &ModelWeights, path: &Path) -> Result<ExportResult> {
        // Mock implementation - actual safetensors serialization would use the safetensors crate
        let mut output = Vec::new();

        // Header
        let header = serde_json::json!({
            "__metadata__": {
                "format": "safetensors",
                "version": "0.1.0",
                "num_tensors": weights.tensors.len(),
                "num_params": weights.param_count(),
            }
        });
        let header_bytes = serde_json::to_vec(&header).map_err(|e| {
            FetchError::ConfigParseError { message: format!("Failed to serialize header: {e}") }
        })?;

        // Write header length (8 bytes, little-endian)
        output.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        output.extend_from_slice(&header_bytes);

        // Write tensor data (mock - just count bytes)
        let data_size: usize = weights.tensors.values().map(|t| t.len() * 4).sum();
        output.extend(vec![0u8; data_size.min(1024)]); // Truncate for mock

        std::fs::write(path, &output).map_err(|e| FetchError::ConfigParseError {
            message: format!("Failed to write file: {e}"),
        })?;

        Ok(ExportResult {
            path: path.to_path_buf(),
            format: ExportFormat::SafeTensors,
            size_bytes: output.len() as u64,
            num_tensors: weights.tensors.len(),
        })
    }

    /// Export to APR format (JSON-based)
    fn export_apr(&self, weights: &ModelWeights, path: &Path) -> Result<ExportResult> {
        #[derive(Serialize)]
        struct AprFormat {
            version: String,
            metadata: ModelMetadata,
            tensors: HashMap<String, AprTensor>,
        }

        #[derive(Serialize)]
        struct AprTensor {
            shape: Vec<usize>,
            dtype: String,
            data: Vec<f32>,
        }

        let apr = AprFormat {
            version: "1.0".to_string(),
            metadata: weights.metadata.clone(),
            tensors: weights
                .tensors
                .iter()
                .map(|(name, data)| {
                    let shape = weights.shapes.get(name).cloned().unwrap_or_default();
                    (
                        name.clone(),
                        AprTensor { shape, dtype: "f32".to_string(), data: data.clone() },
                    )
                })
                .collect(),
        };

        let json = serde_json::to_string_pretty(&apr).map_err(|e| {
            FetchError::ConfigParseError { message: format!("Failed to serialize APR: {e}") }
        })?;

        std::fs::write(path, &json).map_err(|e| FetchError::ConfigParseError {
            message: format!("Failed to write file: {e}"),
        })?;

        Ok(ExportResult {
            path: path.to_path_buf(),
            format: ExportFormat::APR,
            size_bytes: json.len() as u64,
            num_tensors: weights.tensors.len(),
        })
    }

    /// Export to GGUF format with real tensor data (delegates to aprender)
    fn export_gguf(&self, weights: &ModelWeights, path: &Path) -> Result<ExportResult> {
        use aprender::format::gguf::{export_tensors_to_gguf, GgufTensor, GgufValue};

        // Build metadata
        let mut metadata: Vec<(String, GgufValue)> = Vec::new();
        if self.include_metadata {
            if let Some(arch) = &weights.metadata.architecture {
                metadata.push(("general.architecture".into(), GgufValue::String(arch.clone())));
            }
            if let Some(name) = &weights.metadata.model_name {
                metadata.push(("general.name".into(), GgufValue::String(name.clone())));
            }
            metadata.push((
                "general.parameter_count".into(),
                GgufValue::Uint64(weights.metadata.num_params),
            ));
            if let Some(hidden) = weights.metadata.hidden_size {
                metadata.push(("general.hidden_size".into(), GgufValue::Uint32(hidden as u32)));
            }
            if let Some(layers) = weights.metadata.num_layers {
                metadata.push(("general.num_layers".into(), GgufValue::Uint32(layers as u32)));
            }
        }

        // Build tensors — sort names for deterministic output
        let mut tensor_names: Vec<&String> = weights.tensors.keys().collect();
        tensor_names.sort();

        let mut tensors: Vec<GgufTensor> = Vec::new();
        for name in &tensor_names {
            let data = &weights.tensors[*name];
            let shape = weights.shapes.get(*name).cloned().unwrap_or_else(|| vec![data.len()]);
            let (bytes, dtype) = quantize_to_gguf_bytes(data, self.gguf_quantization);
            tensors.push(GgufTensor {
                name: (*name).clone(),
                shape: shape.iter().map(|&d| d as u64).collect(),
                dtype,
                data: bytes,
            });
        }

        // Write via aprender
        let mut file = std::fs::File::create(path).map_err(|e| FetchError::GgufWriteError {
            message: format!("Failed to create GGUF file: {e}"),
        })?;
        export_tensors_to_gguf(&mut file, &tensors, &metadata).map_err(|e| {
            FetchError::GgufWriteError { message: format!("Failed to write GGUF data: {e}") }
        })?;

        let size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);

        Ok(ExportResult {
            path: path.to_path_buf(),
            format: ExportFormat::GGUF,
            size_bytes: size,
            num_tensors: tensor_names.len(),
        })
    }

    /// Export with automatic format detection from filename
    pub fn export_auto(
        &self,
        weights: &ModelWeights,
        filename: impl AsRef<Path>,
    ) -> Result<ExportResult> {
        let path = filename.as_ref();
        let format = ExportFormat::from_path(path).unwrap_or(self.default_format);
        self.export(weights, format, path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hf_pipeline::export::weights::ModelMetadata;

    fn make_test_weights() -> ModelWeights {
        let mut weights = ModelWeights::new();
        weights.add_tensor("layer.0.weight", vec![1.0; 64], vec![8, 8]);
        weights.metadata = ModelMetadata {
            model_name: Some("test-model".to_string()),
            architecture: Some("llama".to_string()),
            num_params: 64,
            ..Default::default()
        };
        weights
    }

    // =================================================================
    // TIER 4: Builder pattern & defaults
    // =================================================================

    #[test]
    fn test_falsify_exporter_default_values() {
        let exp = Exporter::new();
        assert_eq!(exp.output_dir, PathBuf::from("."));
        assert_eq!(exp.default_format, ExportFormat::SafeTensors);
        assert!(exp.include_metadata);
        assert_eq!(exp.gguf_quantization, GgufQuantization::None);
    }

    #[test]
    fn test_falsify_exporter_default_eq_new() {
        let a = Exporter::new();
        let b = Exporter::default();
        assert_eq!(a.output_dir, b.output_dir);
        assert_eq!(a.default_format, b.default_format);
        assert_eq!(a.include_metadata, b.include_metadata);
        assert_eq!(a.gguf_quantization, b.gguf_quantization);
    }

    #[test]
    fn test_falsify_builder_order_independence() {
        let weights = make_test_weights();
        let dir = tempfile::tempdir().expect("temp file creation should succeed");

        let result1 = Exporter::new()
            .output_dir(dir.path())
            .gguf_quantization(GgufQuantization::Q4_0)
            .include_metadata(false)
            .export(&weights, ExportFormat::GGUF, "a.gguf")
            .expect("operation should succeed");

        let result2 = Exporter::new()
            .include_metadata(false)
            .gguf_quantization(GgufQuantization::Q4_0)
            .output_dir(dir.path())
            .export(&weights, ExportFormat::GGUF, "b.gguf")
            .expect("operation should succeed");

        assert_eq!(result1.size_bytes, result2.size_bytes);
        assert_eq!(result1.num_tensors, result2.num_tensors);
    }

    #[test]
    fn test_falsify_builder_setter_override() {
        let weights = make_test_weights();
        let dir = tempfile::tempdir().expect("temp file creation should succeed");

        // Set Q8_0 then override to Q4_0
        let _result = Exporter::new()
            .output_dir(dir.path())
            .gguf_quantization(GgufQuantization::Q8_0)
            .gguf_quantization(GgufQuantization::Q4_0)
            .include_metadata(false)
            .export(&weights, ExportFormat::GGUF, "override.gguf")
            .expect("operation should succeed");

        let file_data =
            std::fs::read(dir.path().join("override.gguf")).expect("file read should succeed");
        let summary = crate::hf_pipeline::export::gguf_verify::verify_gguf(&file_data)
            .expect("operation should succeed");
        // Should be Q4_0 (dtype=2), not Q8_0 (dtype=8)
        assert_eq!(summary.tensors[0].dtype, 2, "override should use Q4_0");
    }

    // =================================================================
    // TIER 4: Format rejection & regression
    // =================================================================

    #[test]
    fn test_falsify_pytorch_format_rejected() {
        let weights = make_test_weights();
        let dir = tempfile::tempdir().expect("temp file creation should succeed");
        let exporter = Exporter::new().output_dir(dir.path());
        let result = exporter.export(&weights, ExportFormat::PyTorch, "model.pt");
        assert!(result.is_err(), "PyTorch export must be rejected");
        let err = result.unwrap_err();
        assert!(
            matches!(err, FetchError::PickleSecurityRisk),
            "error must be PickleSecurityRisk, got {err:?}"
        );
    }

    #[test]
    fn test_falsify_safetensors_export_works() {
        let weights = make_test_weights();
        let dir = tempfile::tempdir().expect("temp file creation should succeed");
        let exporter = Exporter::new().output_dir(dir.path());
        let result = exporter
            .export(&weights, ExportFormat::SafeTensors, "model.safetensors")
            .expect("deserialization should succeed");
        assert_eq!(result.format, ExportFormat::SafeTensors);
        assert!(result.size_bytes > 0);
        assert!(dir.path().join("model.safetensors").exists());
    }

    #[test]
    fn test_falsify_apr_export_works() {
        let weights = make_test_weights();
        let dir = tempfile::tempdir().expect("temp file creation should succeed");
        let exporter = Exporter::new().output_dir(dir.path());
        let result = exporter
            .export(&weights, ExportFormat::APR, "model.apr.json")
            .expect("operation should succeed");
        assert_eq!(result.format, ExportFormat::APR);
        assert!(result.size_bytes > 0);
        assert!(dir.path().join("model.apr.json").exists());
    }

    #[test]
    fn test_falsify_safetensors_ignores_quantization_setting() {
        let weights = make_test_weights();
        let dir = tempfile::tempdir().expect("temp file creation should succeed");
        // Set Q4_0 quant — should be silently ignored for SafeTensors
        let exporter =
            Exporter::new().output_dir(dir.path()).gguf_quantization(GgufQuantization::Q4_0);
        let result = exporter
            .export(&weights, ExportFormat::SafeTensors, "model.safetensors")
            .expect("deserialization should succeed");
        assert_eq!(result.format, ExportFormat::SafeTensors);
        assert!(result.size_bytes > 0);
    }

    // =================================================================
    // TIER 4: export_auto() format detection
    // =================================================================

    #[test]
    fn test_falsify_export_auto_detects_gguf() {
        let weights = make_test_weights();
        let dir = tempfile::tempdir().expect("temp file creation should succeed");
        let exporter = Exporter::new().output_dir(dir.path()).default_format(ExportFormat::APR);
        let result =
            exporter.export_auto(&weights, "model.gguf").expect("operation should succeed");
        assert_eq!(result.format, ExportFormat::GGUF);
    }

    #[test]
    fn test_falsify_export_auto_detects_safetensors() {
        let weights = make_test_weights();
        let dir = tempfile::tempdir().expect("temp file creation should succeed");
        let exporter = Exporter::new().output_dir(dir.path()).default_format(ExportFormat::GGUF);
        let result =
            exporter.export_auto(&weights, "model.safetensors").expect("operation should succeed");
        assert_eq!(result.format, ExportFormat::SafeTensors);
    }

    #[test]
    fn test_falsify_export_auto_detects_apr() {
        let weights = make_test_weights();
        let dir = tempfile::tempdir().expect("temp file creation should succeed");
        let exporter = Exporter::new().output_dir(dir.path()).default_format(ExportFormat::GGUF);
        let result =
            exporter.export_auto(&weights, "model.apr.json").expect("operation should succeed");
        assert_eq!(result.format, ExportFormat::APR);
    }

    #[test]
    fn test_falsify_export_auto_unknown_extension_uses_default() {
        let weights = make_test_weights();
        let dir = tempfile::tempdir().expect("temp file creation should succeed");
        let exporter = Exporter::new().output_dir(dir.path()).default_format(ExportFormat::GGUF);
        let result =
            exporter.export_auto(&weights, "model.unknown").expect("operation should succeed");
        assert_eq!(result.format, ExportFormat::GGUF);
    }

    // =================================================================
    // TIER 4: num_tensors invariant
    // =================================================================

    #[test]
    fn test_falsify_num_tensors_matches_input() {
        for n in [0, 1, 3, 10] {
            let mut weights = ModelWeights::new();
            for i in 0..n {
                weights.add_tensor(format!("t.{i}"), vec![1.0], vec![1]);
            }

            let dir = tempfile::tempdir().expect("temp file creation should succeed");
            let exporter = Exporter::new().output_dir(dir.path()).include_metadata(false);
            let result = exporter
                .export(&weights, ExportFormat::GGUF, "count.gguf")
                .expect("operation should succeed");
            assert_eq!(result.num_tensors, n, "num_tensors mismatch for {n} input tensors");

            let file_data =
                std::fs::read(dir.path().join("count.gguf")).expect("file read should succeed");
            let summary = crate::hf_pipeline::export::gguf_verify::verify_gguf(&file_data)
                .expect("operation should succeed");
            assert_eq!(summary.tensor_count, n as u64, "GGUF header tensor_count mismatch for {n}");
        }
    }
}
