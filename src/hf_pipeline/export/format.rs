//! Export format selection and detection.

use serde::{Deserialize, Serialize};
use std::path::Path;

/// Export format selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExportFormat {
    /// SafeTensors format (recommended)
    SafeTensors,
    /// Aprender format (JSON-based)
    APR,
    /// GGUF quantized format
    GGUF,
    /// PyTorch state dict (for compatibility)
    PyTorch,
}

impl ExportFormat {
    /// Get file extension for format
    #[must_use]
    pub fn extension(&self) -> &'static str {
        match self {
            Self::SafeTensors => "safetensors",
            Self::APR => "apr.json",
            Self::GGUF => "gguf",
            Self::PyTorch => "pt",
        }
    }

    /// Check if format is safe (no pickle/arbitrary code)
    #[must_use]
    pub fn is_safe(&self) -> bool {
        matches!(self, Self::SafeTensors | Self::APR | Self::GGUF)
    }

    /// Detect format from file path
    #[must_use]
    pub fn from_path(path: &Path) -> Option<Self> {
        let name = path.file_name()?.to_str()?;
        if name.ends_with(".safetensors") {
            Some(Self::SafeTensors)
        } else if name.ends_with(".apr.json") || name.ends_with(".apr") {
            Some(Self::APR)
        } else if name.ends_with(".gguf") {
            Some(Self::GGUF)
        } else if name.ends_with(".pt") || name.ends_with(".bin") {
            Some(Self::PyTorch)
        } else {
            None
        }
    }
}

impl std::fmt::Display for ExportFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SafeTensors => write!(f, "SafeTensors"),
            Self::APR => write!(f, "APR"),
            Self::GGUF => write!(f, "GGUF"),
            Self::PyTorch => write!(f, "PyTorch"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =================================================================
    // TIER 4: from_path() exhaustive coverage
    // =================================================================

    #[test]
    fn test_falsify_from_path_gguf() {
        assert_eq!(ExportFormat::from_path(Path::new("model.gguf")), Some(ExportFormat::GGUF));
    }

    #[test]
    fn test_falsify_from_path_safetensors() {
        assert_eq!(
            ExportFormat::from_path(Path::new("model.safetensors")),
            Some(ExportFormat::SafeTensors)
        );
    }

    #[test]
    fn test_falsify_from_path_apr_json() {
        assert_eq!(ExportFormat::from_path(Path::new("model.apr.json")), Some(ExportFormat::APR));
    }

    #[test]
    fn test_falsify_from_path_apr() {
        assert_eq!(ExportFormat::from_path(Path::new("model.apr")), Some(ExportFormat::APR));
    }

    #[test]
    fn test_falsify_from_path_pytorch_pt() {
        assert_eq!(ExportFormat::from_path(Path::new("model.pt")), Some(ExportFormat::PyTorch));
    }

    #[test]
    fn test_falsify_from_path_pytorch_bin() {
        assert_eq!(ExportFormat::from_path(Path::new("model.bin")), Some(ExportFormat::PyTorch));
    }

    #[test]
    fn test_falsify_from_path_unknown() {
        assert_eq!(ExportFormat::from_path(Path::new("model.xyz")), None);
    }

    #[test]
    fn test_falsify_from_path_no_extension() {
        assert_eq!(ExportFormat::from_path(Path::new("model")), None);
    }

    #[test]
    fn test_falsify_from_path_nested_path() {
        assert_eq!(
            ExportFormat::from_path(Path::new("/deep/nested/dir/model.gguf")),
            Some(ExportFormat::GGUF)
        );
    }

    // =================================================================
    // Extension & safety coverage
    // =================================================================

    #[test]
    fn test_falsify_extension_roundtrip() {
        // extension() output should be recognized by from_path()
        for fmt in [
            ExportFormat::SafeTensors,
            ExportFormat::APR,
            ExportFormat::GGUF,
            ExportFormat::PyTorch,
        ] {
            let filename = format!("model.{}", fmt.extension());
            let detected = ExportFormat::from_path(Path::new(&filename));
            assert_eq!(
                detected,
                Some(fmt),
                "extension '{}' should roundtrip for {fmt:?}",
                fmt.extension()
            );
        }
    }

    #[test]
    fn test_falsify_is_safe() {
        assert!(ExportFormat::SafeTensors.is_safe());
        assert!(ExportFormat::APR.is_safe());
        assert!(ExportFormat::GGUF.is_safe());
        assert!(!ExportFormat::PyTorch.is_safe());
    }

    #[test]
    fn test_falsify_display() {
        assert_eq!(ExportFormat::SafeTensors.to_string(), "SafeTensors");
        assert_eq!(ExportFormat::APR.to_string(), "APR");
        assert_eq!(ExportFormat::GGUF.to_string(), "GGUF");
        assert_eq!(ExportFormat::PyTorch.to_string(), "PyTorch");
    }
}
