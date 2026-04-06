//! Serialization format definitions

use serde::{Deserialize, Serialize};

/// Supported model serialization formats
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelFormat {
    /// JSON format (human-readable, larger file size)
    Json,

    /// YAML format (human-readable, good for configs)
    Yaml,

    /// SafeTensors format (HuggingFace compatible, efficient binary)
    SafeTensors,

    /// APR format (sovereign stack universal format — atomic, binary, training+inference)
    Apr,

    /// Placeholder for future GGUF support
    #[cfg(feature = "gguf")]
    Gguf,
}

impl ModelFormat {
    /// Get file extension for this format
    pub fn extension(&self) -> &str {
        match self {
            ModelFormat::Json => "json",
            ModelFormat::Yaml => "yaml",
            ModelFormat::SafeTensors => "safetensors",
            ModelFormat::Apr => "apr",
            #[cfg(feature = "gguf")]
            ModelFormat::Gguf => "gguf",
        }
    }

    /// Detect format from file extension
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext.to_lowercase().as_str() {
            "json" => Some(ModelFormat::Json),
            "yaml" | "yml" => Some(ModelFormat::Yaml),
            "safetensors" => Some(ModelFormat::SafeTensors),
            "apr" => Some(ModelFormat::Apr),
            #[cfg(feature = "gguf")]
            "gguf" => Some(ModelFormat::Gguf),
            _ => None,
        }
    }
}

/// Configuration for saving models
#[derive(Debug, Clone)]
pub struct SaveConfig {
    /// Serialization format
    pub format: ModelFormat,

    /// Whether to pretty-print (for text formats)
    pub pretty: bool,

    /// Whether to compress the output
    pub compress: bool,
}

impl SaveConfig {
    /// Create new save config with format
    pub fn new(format: ModelFormat) -> Self {
        Self { format, pretty: true, compress: false }
    }

    /// Enable/disable pretty printing
    pub fn with_pretty(mut self, pretty: bool) -> Self {
        self.pretty = pretty;
        self
    }

    /// Enable/disable compression
    pub fn with_compress(mut self, compress: bool) -> Self {
        self.compress = compress;
        self
    }
}

impl Default for SaveConfig {
    fn default() -> Self {
        Self::new(ModelFormat::Json).with_pretty(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_extension() {
        assert_eq!(ModelFormat::Json.extension(), "json");
        assert_eq!(ModelFormat::Yaml.extension(), "yaml");
        assert_eq!(ModelFormat::SafeTensors.extension(), "safetensors");
        assert_eq!(ModelFormat::Apr.extension(), "apr");
    }

    #[test]
    fn test_format_from_extension() {
        assert_eq!(ModelFormat::from_extension("json"), Some(ModelFormat::Json));
        assert_eq!(ModelFormat::from_extension("JSON"), Some(ModelFormat::Json));
        assert_eq!(ModelFormat::from_extension("yaml"), Some(ModelFormat::Yaml));
        assert_eq!(ModelFormat::from_extension("yml"), Some(ModelFormat::Yaml));
        assert_eq!(ModelFormat::from_extension("safetensors"), Some(ModelFormat::SafeTensors));
        assert_eq!(ModelFormat::from_extension("SAFETENSORS"), Some(ModelFormat::SafeTensors));
        assert_eq!(ModelFormat::from_extension("apr"), Some(ModelFormat::Apr));
        assert_eq!(ModelFormat::from_extension("unknown"), None);
    }

    #[test]
    fn test_safetensors_format_serde() {
        let format = ModelFormat::SafeTensors;
        let serialized = serde_json::to_string(&format).expect("JSON serialization should succeed");
        let deserialized: ModelFormat =
            serde_json::from_str(&serialized).expect("JSON deserialization should succeed");
        assert_eq!(format, deserialized);
    }

    #[test]
    fn test_save_config_safetensors() {
        let config = SaveConfig::new(ModelFormat::SafeTensors);
        assert_eq!(config.format, ModelFormat::SafeTensors);
        // pretty/compress don't apply to binary formats
        assert!(config.pretty);
    }

    #[test]
    fn test_save_config_builder() {
        let config = SaveConfig::new(ModelFormat::Json).with_pretty(false).with_compress(true);

        assert_eq!(config.format, ModelFormat::Json);
        assert!(!config.pretty);
        assert!(config.compress);
    }

    #[test]
    fn test_save_config_default() {
        let config = SaveConfig::default();
        assert_eq!(config.format, ModelFormat::Json);
        assert!(config.pretty);
        assert!(!config.compress);
    }

    #[test]
    fn test_model_format_serde() {
        // Test serialization/deserialization
        let format = ModelFormat::Json;
        let serialized = serde_json::to_string(&format).expect("JSON serialization should succeed");
        let deserialized: ModelFormat =
            serde_json::from_str(&serialized).expect("JSON deserialization should succeed");
        assert_eq!(format, deserialized);

        let format_yaml = ModelFormat::Yaml;
        let serialized =
            serde_json::to_string(&format_yaml).expect("JSON serialization should succeed");
        let deserialized: ModelFormat =
            serde_json::from_str(&serialized).expect("JSON deserialization should succeed");
        assert_eq!(format_yaml, deserialized);
    }

    #[test]
    fn test_save_config_clone() {
        let config = SaveConfig::new(ModelFormat::Yaml).with_compress(true);
        let cloned = config.clone();
        assert_eq!(config.format, cloned.format);
        assert_eq!(config.compress, cloned.compress);
    }
}
