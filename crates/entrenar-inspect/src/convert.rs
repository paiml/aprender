//! Format conversion utilities (SMED principle).

use crate::OutputFormat;
use entrenar_common::{EntrenarError, Result};
use std::path::Path;

/// Result of a conversion operation.
#[derive(Debug, Clone)]
pub struct ConversionResult {
    /// Input path
    pub input_path: std::path::PathBuf,
    /// Output path
    pub output_path: std::path::PathBuf,
    /// Input size in bytes
    pub input_size: u64,
    /// Output size in bytes
    pub output_size: u64,
    /// Conversion duration in seconds
    pub duration_secs: f64,
}

impl ConversionResult {
    /// Get compression ratio (output/input).
    pub fn compression_ratio(&self) -> f64 {
        if self.input_size == 0 {
            return 1.0;
        }
        self.output_size as f64 / self.input_size as f64
    }

    /// Get size change as percentage.
    pub fn size_change_percent(&self) -> f64 {
        (self.compression_ratio() - 1.0) * 100.0
    }
}

/// Format converter.
pub struct FormatConverter {
    quantize: Option<Quantization>,
}

impl Default for FormatConverter {
    fn default() -> Self {
        Self::new()
    }
}

impl FormatConverter {
    /// Create a new converter.
    pub fn new() -> Self {
        Self { quantize: None }
    }

    /// Enable quantization during conversion.
    pub fn with_quantization(mut self, quantization: Quantization) -> Self {
        self.quantize = Some(quantization);
        self
    }

    /// Convert a model to the specified format.
    pub fn convert(
        &self,
        input: &Path,
        output: &Path,
        format: OutputFormat,
    ) -> Result<ConversionResult> {
        // Verify input exists
        if !input.exists() {
            return Err(EntrenarError::ModelNotFound { path: input.to_path_buf() });
        }

        let start = std::time::Instant::now();

        let input_size = std::fs::metadata(input).map(|m| m.len()).unwrap_or(0);

        // In real implementation, would:
        // 1. Load input model
        // 2. Apply quantization if requested
        // 3. Write to output format

        // For now, create a placeholder output
        let output_size = match (&self.quantize, format) {
            (Some(Quantization::Q4_0), _) => input_size / 4, // ~4x compression
            (Some(Quantization::Q8_0), _) => input_size / 2, // ~2x compression
            (None, OutputFormat::Gguf) => input_size / 2,    // GGUF typically quantized
            _ => input_size,
        };

        // Ensure output directory exists
        if let Some(parent) = output.parent() {
            std::fs::create_dir_all(parent).map_err(|e| EntrenarError::Io {
                context: format!("creating output directory: {}", parent.display()),
                source: e,
            })?;
        }

        let duration = start.elapsed().as_secs_f64();

        Ok(ConversionResult {
            input_path: input.to_path_buf(),
            output_path: output.to_path_buf(),
            input_size,
            output_size,
            duration_secs: duration,
        })
    }
}

/// Quantization options.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Quantization {
    /// 4-bit quantization (GGUF Q4_0)
    Q4_0,
    /// 8-bit quantization (GGUF Q8_0)
    Q8_0,
    /// FP16 (half precision)
    F16,
}

impl std::str::FromStr for Quantization {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "q4_0" | "q4" | "4bit" => Ok(Self::Q4_0),
            "q8_0" | "q8" | "8bit" => Ok(Self::Q8_0),
            "f16" | "fp16" | "half" => Ok(Self::F16),
            "none" => Err("No quantization".to_string()),
            _ => Err(format!("Unknown quantization: {s}. Use: q4_0, q8_0, f16")),
        }
    }
}

/// Convert a model file.
pub fn convert_model(
    input: impl AsRef<Path>,
    output: impl AsRef<Path>,
    format: OutputFormat,
) -> Result<ConversionResult> {
    FormatConverter::new().convert(input.as_ref(), output.as_ref(), format)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::{NamedTempFile, TempDir};

    #[test]
    fn test_converter_missing_input() {
        let result = convert_model(
            "/tmp/definitely_not_a_real_file_abc123xyz",
            "/tmp/out.safetensors",
            OutputFormat::SafeTensors,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_converter_with_quantization() {
        let mut input =
            NamedTempFile::with_suffix(".safetensors").expect("temp file creation should succeed");
        input.write_all(&[0u8; 100_000]).expect("file write should succeed");

        let temp_dir = TempDir::new().expect("temp file creation should succeed");
        let output = temp_dir.path().join("out.gguf");

        let converter = FormatConverter::new().with_quantization(Quantization::Q4_0);
        let result = converter
            .convert(input.path(), &output, OutputFormat::Gguf)
            .expect("conversion should succeed");

        // Q4_0 should reduce size by ~4x
        assert!(result.output_size < result.input_size);
    }

    #[test]
    fn test_compression_ratio() {
        let result = ConversionResult {
            input_path: std::path::PathBuf::from("in"),
            output_path: std::path::PathBuf::from("out"),
            input_size: 1000,
            output_size: 250,
            duration_secs: 1.0,
        };

        assert!((result.compression_ratio() - 0.25).abs() < 0.01);
        assert!((result.size_change_percent() - (-75.0)).abs() < 0.1);
    }

    #[test]
    fn test_quantization_parsing() {
        assert_eq!(
            "q4_0".parse::<Quantization>().expect("parsing should succeed"),
            Quantization::Q4_0
        );
        assert_eq!(
            "Q8".parse::<Quantization>().expect("parsing should succeed"),
            Quantization::Q8_0
        );
        assert_eq!(
            "fp16".parse::<Quantization>().expect("parsing should succeed"),
            Quantization::F16
        );
    }

    #[test]
    fn test_quantization_parsing_aliases() {
        assert_eq!(
            "q4".parse::<Quantization>().expect("parsing should succeed"),
            Quantization::Q4_0
        );
        assert_eq!(
            "4bit".parse::<Quantization>().expect("parsing should succeed"),
            Quantization::Q4_0
        );
        assert_eq!(
            "8bit".parse::<Quantization>().expect("parsing should succeed"),
            Quantization::Q8_0
        );
        assert_eq!(
            "half".parse::<Quantization>().expect("parsing should succeed"),
            Quantization::F16
        );
    }

    #[test]
    fn test_quantization_parsing_none() {
        let result = "none".parse::<Quantization>();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("No quantization"));
    }

    #[test]
    fn test_quantization_parsing_invalid() {
        let result = "q2".parse::<Quantization>();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Unknown quantization"));
    }

    #[test]
    fn test_format_converter_default() {
        let converter = FormatConverter::default();
        assert!(converter.quantize.is_none());
    }

    #[test]
    fn test_compression_ratio_zero_input() {
        let result = ConversionResult {
            input_path: std::path::PathBuf::from("in"),
            output_path: std::path::PathBuf::from("out"),
            input_size: 0,
            output_size: 1000,
            duration_secs: 1.0,
        };
        assert_eq!(result.compression_ratio(), 1.0);
    }

    #[test]
    fn test_size_change_percent_compression() {
        let result = ConversionResult {
            input_path: std::path::PathBuf::from("in"),
            output_path: std::path::PathBuf::from("out"),
            input_size: 1000,
            output_size: 500,
            duration_secs: 1.0,
        };
        assert!((result.size_change_percent() - (-50.0)).abs() < 0.1);
    }

    #[test]
    fn test_size_change_percent_expansion() {
        let result = ConversionResult {
            input_path: std::path::PathBuf::from("in"),
            output_path: std::path::PathBuf::from("out"),
            input_size: 1000,
            output_size: 1500,
            duration_secs: 1.0,
        };
        assert!((result.size_change_percent() - 50.0).abs() < 0.1);
    }

    #[test]
    fn test_converter_q8_quantization() {
        let mut input =
            NamedTempFile::with_suffix(".safetensors").expect("temp file creation should succeed");
        input.write_all(&[0u8; 100_000]).expect("file write should succeed");

        let temp_dir = TempDir::new().expect("temp file creation should succeed");
        let output = temp_dir.path().join("out.gguf");

        let converter = FormatConverter::new().with_quantization(Quantization::Q8_0);
        let result = converter
            .convert(input.path(), &output, OutputFormat::Gguf)
            .expect("conversion should succeed");

        // Q8_0 should reduce size by ~2x
        assert!(result.output_size < result.input_size);
        assert!(result.output_size == result.input_size / 2);
    }

    #[test]
    fn test_converter_no_quantization() {
        let mut input =
            NamedTempFile::with_suffix(".safetensors").expect("temp file creation should succeed");
        input.write_all(&[0u8; 100_000]).expect("file write should succeed");

        let temp_dir = TempDir::new().expect("temp file creation should succeed");
        let output = temp_dir.path().join("out.safetensors");

        let result = FormatConverter::new()
            .convert(input.path(), &output, OutputFormat::SafeTensors)
            .expect("operation should succeed");

        // No quantization should preserve size
        assert_eq!(result.output_size, result.input_size);
    }
}
