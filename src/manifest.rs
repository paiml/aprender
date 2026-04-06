//! Model Manifest (Modelfile) Support
//!
//! Provides a Modelfile-like configuration for custom model definitions,
//! similar to ollama's Modelfile format.
//!
//! ## Modelfile Format
//!
//! ```text
//! FROM llama3:8b
//! SYSTEM You are a helpful coding assistant.
//! PARAMETER temperature 0.7
//! PARAMETER top_p 0.9
//! PARAMETER stop "<|endoftext|>"
//! TEMPLATE "{{ .System }}\nUser: {{ .Prompt }}\nAssistant:"
//! ```
//!
//! ## Example
//!
//! ```rust,ignore
//! use pacha::manifest::ModelManifest;
//!
//! let manifest = ModelManifest::parse(r#"
//!     FROM llama3:8b
//!     SYSTEM You are helpful.
//!     PARAMETER temperature 0.7
//! "#)?;
//!
//! println!("Base: {}", manifest.base_model);
//! println!("System: {:?}", manifest.system_prompt);
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

use crate::error::{PachaError, Result};

// ============================================================================
// MANIFEST-001: Model Manifest
// ============================================================================

/// Model manifest defining a custom model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelManifest {
    /// Base model reference (FROM directive)
    pub base_model: String,
    /// System prompt (SYSTEM directive)
    pub system_prompt: Option<String>,
    /// Generation parameters (PARAMETER directives)
    pub parameters: ManifestParameters,
    /// Custom prompt template (TEMPLATE directive)
    pub template: Option<String>,
    /// Model adapter/LoRA path (ADAPTER directive)
    pub adapter: Option<String>,
    /// License information (LICENSE directive)
    pub license: Option<String>,
    /// Model description
    pub description: Option<String>,
    /// Custom metadata
    pub metadata: HashMap<String, String>,
}

impl Default for ModelManifest {
    fn default() -> Self {
        Self {
            base_model: String::new(),
            system_prompt: None,
            parameters: ManifestParameters::default(),
            template: None,
            adapter: None,
            license: None,
            description: None,
            metadata: HashMap::new(),
        }
    }
}

/// Generation parameters from manifest
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifestParameters {
    /// Sampling temperature
    pub temperature: Option<f32>,
    /// Top-p (nucleus) sampling
    pub top_p: Option<f32>,
    /// Top-k sampling
    pub top_k: Option<usize>,
    /// Maximum tokens to generate
    pub max_tokens: Option<usize>,
    /// Stop sequences
    pub stop: Vec<String>,
    /// Repetition penalty
    pub repeat_penalty: Option<f32>,
    /// Number of tokens to consider for repetition penalty
    pub repeat_last_n: Option<usize>,
    /// Context window size
    pub context_length: Option<usize>,
    /// Seed for reproducibility
    pub seed: Option<u64>,
}

impl Default for ManifestParameters {
    fn default() -> Self {
        Self {
            temperature: None,
            top_p: None,
            top_k: None,
            max_tokens: None,
            stop: Vec::new(),
            repeat_penalty: None,
            repeat_last_n: None,
            context_length: None,
            seed: None,
        }
    }
}

impl ModelManifest {
    /// Create a new manifest with a base model
    #[must_use]
    pub fn new(base_model: impl Into<String>) -> Self {
        Self { base_model: base_model.into(), ..Default::default() }
    }

    /// Set system prompt
    #[must_use]
    pub fn with_system(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Set temperature
    #[must_use]
    pub fn with_temperature(mut self, temp: f32) -> Self {
        self.parameters.temperature = Some(temp);
        self
    }

    /// Set top_p
    #[must_use]
    pub fn with_top_p(mut self, top_p: f32) -> Self {
        self.parameters.top_p = Some(top_p);
        self
    }

    /// Set top_k
    #[must_use]
    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.parameters.top_k = Some(top_k);
        self
    }

    /// Set max tokens
    #[must_use]
    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.parameters.max_tokens = Some(max_tokens);
        self
    }

    /// Add stop sequence
    #[must_use]
    pub fn with_stop(mut self, stop: impl Into<String>) -> Self {
        self.parameters.stop.push(stop.into());
        self
    }

    /// Set template
    #[must_use]
    pub fn with_template(mut self, template: impl Into<String>) -> Self {
        self.template = Some(template.into());
        self
    }

    /// Set adapter path
    #[must_use]
    pub fn with_adapter(mut self, adapter: impl Into<String>) -> Self {
        self.adapter = Some(adapter.into());
        self
    }

    /// Set description
    #[must_use]
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Add metadata
    #[must_use]
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Parse a Modelfile-format string
    pub fn parse(content: &str) -> Result<Self> {
        let mut manifest = Self::default();

        for line in content.lines() {
            let line = line.trim();

            // Skip empty lines and comments
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            // Parse directive
            let (directive, value) = if let Some(idx) = line.find(char::is_whitespace) {
                let (d, v) = line.split_at(idx);
                (d.to_uppercase(), v.trim())
            } else {
                (line.to_uppercase(), "")
            };

            match directive.as_str() {
                "FROM" => {
                    if value.is_empty() {
                        return Err(PachaError::Validation(
                            "FROM requires a model reference".to_string(),
                        ));
                    }
                    manifest.base_model = value.to_string();
                }
                "SYSTEM" => {
                    manifest.system_prompt = Some(value.to_string());
                }
                "PARAMETER" => {
                    parse_parameter(&mut manifest.parameters, value)?;
                }
                "TEMPLATE" => {
                    // Template can be multi-line with quotes
                    let template = value.trim_matches('"').trim_matches('\'');
                    manifest.template = Some(template.to_string());
                }
                "ADAPTER" => {
                    manifest.adapter = Some(value.to_string());
                }
                "LICENSE" => {
                    manifest.license = Some(value.to_string());
                }
                "MESSAGE" => {
                    // MESSAGE role content - add to metadata for now
                    manifest.metadata.insert("message".to_string(), value.to_string());
                }
                _ => {
                    // Unknown directive - store as metadata
                    manifest.metadata.insert(directive.to_lowercase(), value.to_string());
                }
            }
        }

        if manifest.base_model.is_empty() {
            return Err(PachaError::Validation("Modelfile must have FROM directive".to_string()));
        }

        Ok(manifest)
    }

    /// Load manifest from file
    pub fn load(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path).map_err(|e| {
            PachaError::Io(std::io::Error::new(
                e.kind(),
                format!("Failed to read {}: {}", path.display(), e),
            ))
        })?;
        Self::parse(&content)
    }

    /// Save manifest to file in Modelfile format
    pub fn save(&self, path: &Path) -> Result<()> {
        let content = self.to_modelfile();
        std::fs::write(path, content).map_err(|e| {
            PachaError::Io(std::io::Error::new(
                e.kind(),
                format!("Failed to write {}: {}", path.display(), e),
            ))
        })
    }

    /// Convert to Modelfile format string
    #[must_use]
    pub fn to_modelfile(&self) -> String {
        let mut lines = Vec::new();

        // FROM directive (required)
        lines.push(format!("FROM {}", self.base_model));

        // SYSTEM directive
        if let Some(ref system) = self.system_prompt {
            lines.push(format!("SYSTEM {}", system));
        }

        // PARAMETER directives
        if let Some(temp) = self.parameters.temperature {
            lines.push(format!("PARAMETER temperature {}", temp));
        }
        if let Some(top_p) = self.parameters.top_p {
            lines.push(format!("PARAMETER top_p {}", top_p));
        }
        if let Some(top_k) = self.parameters.top_k {
            lines.push(format!("PARAMETER top_k {}", top_k));
        }
        if let Some(max_tokens) = self.parameters.max_tokens {
            lines.push(format!("PARAMETER max_tokens {}", max_tokens));
        }
        for stop in &self.parameters.stop {
            lines.push(format!("PARAMETER stop \"{}\"", stop));
        }
        if let Some(repeat_penalty) = self.parameters.repeat_penalty {
            lines.push(format!("PARAMETER repeat_penalty {}", repeat_penalty));
        }
        if let Some(context_length) = self.parameters.context_length {
            lines.push(format!("PARAMETER context_length {}", context_length));
        }
        if let Some(seed) = self.parameters.seed {
            lines.push(format!("PARAMETER seed {}", seed));
        }

        // TEMPLATE directive
        if let Some(ref template) = self.template {
            lines.push(format!("TEMPLATE \"{}\"", template));
        }

        // ADAPTER directive
        if let Some(ref adapter) = self.adapter {
            lines.push(format!("ADAPTER {}", adapter));
        }

        // LICENSE directive
        if let Some(ref license) = self.license {
            lines.push(format!("LICENSE {}", license));
        }

        lines.join("\n")
    }

    /// Convert to JSON
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self)
            .map_err(|e| PachaError::Validation(format!("Failed to serialize manifest: {}", e)))
    }

    /// Parse from JSON
    pub fn from_json(json: &str) -> Result<Self> {
        serde_json::from_str(json)
            .map_err(|e| PachaError::Validation(format!("Failed to parse manifest JSON: {}", e)))
    }
}

/// Parse a PARAMETER directive value
fn parse_parameter(params: &mut ManifestParameters, value: &str) -> Result<()> {
    let parts: Vec<&str> = value.splitn(2, char::is_whitespace).collect();
    if parts.len() != 2 {
        return Err(PachaError::Validation(format!("Invalid PARAMETER syntax: {}", value)));
    }

    let (name, val) = (parts[0].to_lowercase(), parts[1].trim());

    match name.as_str() {
        "temperature" => {
            params.temperature =
                Some(val.parse().map_err(|_| {
                    PachaError::Validation(format!("Invalid temperature: {}", val))
                })?);
        }
        "top_p" => {
            params.top_p = Some(
                val.parse()
                    .map_err(|_| PachaError::Validation(format!("Invalid top_p: {}", val)))?,
            );
        }
        "top_k" => {
            params.top_k = Some(
                val.parse()
                    .map_err(|_| PachaError::Validation(format!("Invalid top_k: {}", val)))?,
            );
        }
        "max_tokens" | "num_predict" => {
            params.max_tokens = Some(
                val.parse()
                    .map_err(|_| PachaError::Validation(format!("Invalid max_tokens: {}", val)))?,
            );
        }
        "stop" => {
            let stop = val.trim_matches('"').trim_matches('\'');
            params.stop.push(stop.to_string());
        }
        "repeat_penalty" => {
            params.repeat_penalty =
                Some(val.parse().map_err(|_| {
                    PachaError::Validation(format!("Invalid repeat_penalty: {}", val))
                })?);
        }
        "repeat_last_n" => {
            params.repeat_last_n =
                Some(val.parse().map_err(|_| {
                    PachaError::Validation(format!("Invalid repeat_last_n: {}", val))
                })?);
        }
        "context_length" | "num_ctx" => {
            params.context_length =
                Some(val.parse().map_err(|_| {
                    PachaError::Validation(format!("Invalid context_length: {}", val))
                })?);
        }
        "seed" => {
            params.seed = Some(
                val.parse()
                    .map_err(|_| PachaError::Validation(format!("Invalid seed: {}", val)))?,
            );
        }
        _ => {
            // Ignore unknown parameters
        }
    }

    Ok(())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // Parse Tests
    // ========================================================================

    #[test]
    fn test_parse_minimal() {
        let manifest = ModelManifest::parse("FROM llama3").unwrap();
        assert_eq!(manifest.base_model, "llama3");
    }

    #[test]
    fn test_parse_with_system() {
        let manifest = ModelManifest::parse(
            r#"
            FROM llama3:8b
            SYSTEM You are a helpful assistant.
            "#,
        )
        .unwrap();

        assert_eq!(manifest.base_model, "llama3:8b");
        assert_eq!(manifest.system_prompt, Some("You are a helpful assistant.".to_string()));
    }

    #[test]
    fn test_parse_with_parameters() {
        let manifest = ModelManifest::parse(
            r#"
            FROM mistral
            PARAMETER temperature 0.7
            PARAMETER top_p 0.9
            PARAMETER top_k 40
            PARAMETER max_tokens 256
            "#,
        )
        .unwrap();

        assert_eq!(manifest.parameters.temperature, Some(0.7));
        assert_eq!(manifest.parameters.top_p, Some(0.9));
        assert_eq!(manifest.parameters.top_k, Some(40));
        assert_eq!(manifest.parameters.max_tokens, Some(256));
    }

    #[test]
    fn test_parse_with_stop_sequences() {
        let manifest = ModelManifest::parse(
            r#"
            FROM llama3
            PARAMETER stop "<|endoftext|>"
            PARAMETER stop "User:"
            "#,
        )
        .unwrap();

        assert_eq!(manifest.parameters.stop.len(), 2);
        assert!(manifest.parameters.stop.contains(&"<|endoftext|>".to_string()));
        assert!(manifest.parameters.stop.contains(&"User:".to_string()));
    }

    #[test]
    fn test_parse_with_template() {
        let manifest = ModelManifest::parse(
            r#"
            FROM llama3
            TEMPLATE "{{ .System }}\nUser: {{ .Prompt }}\nAssistant:"
            "#,
        )
        .unwrap();

        assert!(manifest.template.is_some());
        assert!(manifest.template.as_ref().unwrap().contains("System"));
    }

    #[test]
    fn test_parse_with_adapter() {
        let manifest = ModelManifest::parse(
            r#"
            FROM llama3:8b
            ADAPTER /path/to/lora.safetensors
            "#,
        )
        .unwrap();

        assert_eq!(manifest.adapter, Some("/path/to/lora.safetensors".to_string()));
    }

    #[test]
    fn test_parse_with_comments() {
        let manifest = ModelManifest::parse(
            r#"
            # This is a comment
            FROM llama3
            # Another comment
            SYSTEM Be helpful
            "#,
        )
        .unwrap();

        assert_eq!(manifest.base_model, "llama3");
        assert!(manifest.system_prompt.is_some());
    }

    #[test]
    fn test_parse_missing_from() {
        let result = ModelManifest::parse("SYSTEM You are helpful.");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_empty_from() {
        let result = ModelManifest::parse("FROM");
        assert!(result.is_err());
    }

    // ========================================================================
    // Builder Tests
    // ========================================================================

    #[test]
    fn test_builder() {
        let manifest = ModelManifest::new("llama3:8b")
            .with_system("You are a coding assistant.")
            .with_temperature(0.8)
            .with_top_p(0.95)
            .with_max_tokens(1024)
            .with_stop("<|end|>")
            .with_description("My custom model");

        assert_eq!(manifest.base_model, "llama3:8b");
        assert!(manifest.system_prompt.is_some());
        assert_eq!(manifest.parameters.temperature, Some(0.8));
        assert_eq!(manifest.parameters.top_p, Some(0.95));
        assert_eq!(manifest.parameters.max_tokens, Some(1024));
        assert_eq!(manifest.parameters.stop.len(), 1);
        assert!(manifest.description.is_some());
    }

    #[test]
    fn test_builder_with_metadata() {
        let manifest = ModelManifest::new("llama3")
            .with_metadata("author", "test")
            .with_metadata("version", "1.0");

        assert_eq!(manifest.metadata.get("author"), Some(&"test".to_string()));
        assert_eq!(manifest.metadata.get("version"), Some(&"1.0".to_string()));
    }

    // ========================================================================
    // Serialization Tests
    // ========================================================================

    #[test]
    fn test_to_modelfile() {
        let manifest =
            ModelManifest::new("llama3:8b").with_system("Be helpful").with_temperature(0.7);

        let modelfile = manifest.to_modelfile();
        assert!(modelfile.contains("FROM llama3:8b"));
        assert!(modelfile.contains("SYSTEM Be helpful"));
        assert!(modelfile.contains("PARAMETER temperature 0.7"));
    }

    #[test]
    fn test_roundtrip() {
        let original = ModelManifest::new("mixtral:8x7b")
            .with_system("You are an expert.")
            .with_temperature(0.9)
            .with_top_k(50)
            .with_max_tokens(2048);

        let modelfile = original.to_modelfile();
        let parsed = ModelManifest::parse(&modelfile).unwrap();

        assert_eq!(parsed.base_model, original.base_model);
        assert_eq!(parsed.system_prompt, original.system_prompt);
        assert_eq!(parsed.parameters.temperature, original.parameters.temperature);
        assert_eq!(parsed.parameters.top_k, original.parameters.top_k);
        assert_eq!(parsed.parameters.max_tokens, original.parameters.max_tokens);
    }

    #[test]
    fn test_json_roundtrip() {
        let original = ModelManifest::new("llama3").with_system("Test").with_temperature(0.5);

        let json = original.to_json().unwrap();
        let parsed = ModelManifest::from_json(&json).unwrap();

        assert_eq!(parsed.base_model, original.base_model);
        assert_eq!(parsed.system_prompt, original.system_prompt);
    }

    // ========================================================================
    // Parameter Parsing Tests
    // ========================================================================

    #[test]
    fn test_parse_context_length_alias() {
        let manifest = ModelManifest::parse(
            r#"
            FROM llama3
            PARAMETER num_ctx 4096
            "#,
        )
        .unwrap();

        assert_eq!(manifest.parameters.context_length, Some(4096));
    }

    #[test]
    fn test_parse_max_tokens_alias() {
        let manifest = ModelManifest::parse(
            r#"
            FROM llama3
            PARAMETER num_predict 512
            "#,
        )
        .unwrap();

        assert_eq!(manifest.parameters.max_tokens, Some(512));
    }

    #[test]
    fn test_parse_repeat_penalty() {
        let manifest = ModelManifest::parse(
            r#"
            FROM llama3
            PARAMETER repeat_penalty 1.1
            PARAMETER repeat_last_n 64
            "#,
        )
        .unwrap();

        assert_eq!(manifest.parameters.repeat_penalty, Some(1.1));
        assert_eq!(manifest.parameters.repeat_last_n, Some(64));
    }

    #[test]
    fn test_parse_seed() {
        let manifest = ModelManifest::parse(
            r#"
            FROM llama3
            PARAMETER seed 42
            "#,
        )
        .unwrap();

        assert_eq!(manifest.parameters.seed, Some(42));
    }

    #[test]
    fn test_invalid_parameter_value() {
        let result = ModelManifest::parse(
            r#"
            FROM llama3
            PARAMETER temperature not_a_number
            "#,
        );
        assert!(result.is_err());
    }

    // ========================================================================
    // Default Tests
    // ========================================================================

    #[test]
    fn test_default_parameters() {
        let params = ManifestParameters::default();
        assert!(params.temperature.is_none());
        assert!(params.top_p.is_none());
        assert!(params.stop.is_empty());
    }

    #[test]
    fn test_default_manifest() {
        let manifest = ModelManifest::default();
        assert!(manifest.base_model.is_empty());
        assert!(manifest.system_prompt.is_none());
    }
}
