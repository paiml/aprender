//! SafeTensors model inspection and format conversion.
//!
//! This crate provides tools for:
//! - Inspecting model architectures and layer structures
//! - Analyzing memory requirements
//! - Converting between formats (SafeTensors, GGUF, APR)
//!
//! # Toyota Way Principles
//!
//! - **Andon**: Integrity checker surfaces data problems immediately
//! - **Genchi Genbutsu**: Direct inspection of actual model weights
//! - **SMED**: Quick format conversion for deployment changeover

pub mod architecture;
pub mod convert;
pub mod inspect;
pub mod validate;

pub use architecture::{Architecture, ArchitectureDetector};
pub use convert::{ConversionResult, FormatConverter};
pub use inspect::{ModelInfo, TensorInfo};
pub use validate::{IntegrityChecker, ValidationResult};

use entrenar_common::Result;
use std::path::Path;

/// Inspect a model file and return detailed information.
pub fn inspect(path: impl AsRef<Path>) -> Result<ModelInfo> {
    inspect::inspect_model(path)
}

/// Validate a model file's integrity.
pub fn validate(path: impl AsRef<Path>) -> Result<ValidationResult> {
    validate::validate_model(path)
}

/// Convert a model between formats.
pub fn convert(
    input: impl AsRef<Path>,
    output: impl AsRef<Path>,
    format: OutputFormat,
) -> Result<ConversionResult> {
    convert::convert_model(input, output, format)
}

/// Output format for model conversion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    /// SafeTensors format (secure, recommended)
    SafeTensors,
    /// GGUF format (llama.cpp compatible)
    Gguf,
    /// APR format (JSON metadata)
    Apr,
}

impl std::str::FromStr for OutputFormat {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "safetensors" | "st" => Ok(Self::SafeTensors),
            "gguf" => Ok(Self::Gguf),
            "apr" => Ok(Self::Apr),
            _ => Err(format!("Unknown format: {s}. Use: safetensors, gguf, apr")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_output_format_parsing() {
        assert_eq!(
            "safetensors".parse::<OutputFormat>().expect("parsing should succeed"),
            OutputFormat::SafeTensors
        );
        assert_eq!(
            "GGUF".parse::<OutputFormat>().expect("parsing should succeed"),
            OutputFormat::Gguf
        );
        assert_eq!(
            "apr".parse::<OutputFormat>().expect("parsing should succeed"),
            OutputFormat::Apr
        );
    }

    #[test]
    fn test_output_format_st_alias() {
        assert_eq!(
            "st".parse::<OutputFormat>().expect("parsing should succeed"),
            OutputFormat::SafeTensors
        );
        assert_eq!(
            "ST".parse::<OutputFormat>().expect("parsing should succeed"),
            OutputFormat::SafeTensors
        );
    }

    #[test]
    fn test_output_format_invalid() {
        let result = "pickle".parse::<OutputFormat>();
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("Unknown format"));
        assert!(err.contains("pickle"));
    }

    #[test]
    fn test_output_format_equality() {
        assert_eq!(OutputFormat::SafeTensors, OutputFormat::SafeTensors);
        assert_ne!(OutputFormat::SafeTensors, OutputFormat::Gguf);
        assert_ne!(OutputFormat::Gguf, OutputFormat::Apr);
    }
}
