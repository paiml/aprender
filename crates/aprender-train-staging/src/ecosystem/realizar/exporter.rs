//! GGUF exporter implementation.

use super::error::GgufExportError;
use super::metadata::{GeneralMetadata, GgufMetadata};
use super::provenance::ExperimentProvenance;
use super::quantization::QuantizationType;
use std::path::Path;

/// GGUF exporter for model conversion.
#[derive(Debug, Clone)]
pub struct GgufExporter {
    /// Quantization type to apply
    quantization: QuantizationType,
    /// Metadata to embed
    metadata: GgufMetadata,
    /// Whether to validate model structure
    validate: bool,
    /// Number of threads for quantization
    threads: usize,
}

impl Default for GgufExporter {
    fn default() -> Self {
        Self::new(QuantizationType::Q4KM)
    }
}

impl GgufExporter {
    /// Create a new exporter with specified quantization.
    pub fn new(quantization: QuantizationType) -> Self {
        Self {
            quantization,
            metadata: GgufMetadata::default(),
            validate: true,
            threads: num_cpus(),
        }
    }

    /// Set metadata.
    pub fn with_metadata(mut self, metadata: GgufMetadata) -> Self {
        self.metadata = metadata;
        self
    }

    /// Set general metadata.
    pub fn with_general(mut self, general: GeneralMetadata) -> Self {
        self.metadata.general = general;
        self
    }

    /// Set experiment provenance.
    pub fn with_provenance(mut self, provenance: ExperimentProvenance) -> Self {
        self.metadata.provenance = Some(provenance);
        self
    }

    /// Disable validation.
    pub fn without_validation(mut self) -> Self {
        self.validate = false;
        self
    }

    /// Set thread count.
    pub fn with_threads(mut self, threads: usize) -> Self {
        self.threads = threads.max(1);
        self
    }

    /// Get the quantization type.
    pub fn quantization(&self) -> QuantizationType {
        self.quantization
    }

    /// Get the metadata.
    pub fn metadata(&self) -> &GgufMetadata {
        &self.metadata
    }

    /// Export model to GGUF format.
    ///
    /// This is a placeholder that prepares export configuration.
    /// Actual export requires Realizar crate integration.
    pub fn export(
        &self,
        _input_path: impl AsRef<Path>,
        output_path: impl AsRef<Path>,
    ) -> Result<GgufExportResult, GgufExportError> {
        let output = output_path.as_ref();

        // Validate output path
        if let Some(parent) = output.parent() {
            if !parent.exists() {
                return Err(GgufExportError::IoError(format!(
                    "Output directory does not exist: {}",
                    parent.display()
                )));
            }
        }

        // In a real implementation, this would:
        // 1. Load model from input_path using Realizar
        // 2. Apply quantization
        // 3. Embed metadata
        // 4. Write to output_path

        // Return export result with metadata
        Ok(GgufExportResult {
            output_path: output.to_path_buf(),
            quantization: self.quantization,
            metadata_keys: self
                .metadata
                .provenance
                .as_ref()
                .map_or(0, |p| p.to_metadata_pairs().len())
                + self.metadata.custom.len(),
            estimated_size_bytes: 0, // Would be calculated from actual model
        })
    }

    /// Collect all metadata as key-value pairs.
    pub fn collect_metadata(&self) -> Vec<(String, String)> {
        let mut pairs = Vec::new();

        // General metadata
        pairs
            .push(("general.architecture".to_string(), self.metadata.general.architecture.clone()));
        pairs.push(("general.name".to_string(), self.metadata.general.name.clone()));

        if let Some(ref author) = self.metadata.general.author {
            pairs.push(("general.author".to_string(), author.clone()));
        }
        if let Some(ref desc) = self.metadata.general.description {
            pairs.push(("general.description".to_string(), desc.clone()));
        }
        if let Some(ref license) = self.metadata.general.license {
            pairs.push(("general.license".to_string(), license.clone()));
        }
        if let Some(ref url) = self.metadata.general.url {
            pairs.push(("general.url".to_string(), url.clone()));
        }

        pairs.push(("general.file_type".to_string(), self.quantization.as_str().to_string()));

        // Provenance metadata
        if let Some(ref prov) = self.metadata.provenance {
            pairs.extend(prov.to_metadata_pairs());
        }

        // Custom metadata
        for (key, value) in &self.metadata.custom {
            pairs.push((format!("custom.{key}"), value.clone()));
        }

        pairs
    }
}

/// Result of a GGUF export operation.
#[derive(Debug, Clone)]
pub struct GgufExportResult {
    /// Path to exported file
    pub output_path: std::path::PathBuf,
    /// Quantization type used
    pub quantization: QuantizationType,
    /// Number of metadata keys embedded
    pub metadata_keys: usize,
    /// Estimated file size in bytes
    pub estimated_size_bytes: u64,
}

/// Get number of CPUs (simplified).
fn num_cpus() -> usize {
    std::thread::available_parallelism().map(std::num::NonZero::get).unwrap_or(4)
}
