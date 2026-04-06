//! Model Aliases and Shortcuts
//!
//! Provides short aliases for common models, similar to ollama's model naming.
//!
//! ## Features
//!
//! - Short names: `llama3` -> `meta-llama/Meta-Llama-3-8B`
//! - Version shortcuts: `llama3:70b` -> `meta-llama/Meta-Llama-3-70B`
//! - Quantization tags: `llama3:8b-q4` -> Q4_K_M quantized
//! - Custom alias configuration
//!
//! ## Example
//!
//! ```rust,ignore
//! use pacha::aliases::AliasRegistry;
//!
//! let aliases = AliasRegistry::default();
//!
//! let resolved = aliases.resolve("llama3:8b-q4")?;
//! // Returns: hf://meta-llama/Meta-Llama-3-8B-Instruct-GGUF:Q4_K_M
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// ALIAS-001: Alias Entry
// ============================================================================

/// An alias entry mapping short name to full reference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AliasEntry {
    /// Short name (e.g., "llama3")
    pub alias: String,
    /// Full model reference
    pub target: String,
    /// Default quantization
    pub default_quant: Option<String>,
    /// Available variants (size tags)
    pub variants: HashMap<String, String>,
    /// Description
    pub description: Option<String>,
}

impl AliasEntry {
    /// Create a new alias entry
    #[must_use]
    pub fn new(alias: impl Into<String>, target: impl Into<String>) -> Self {
        Self {
            alias: alias.into(),
            target: target.into(),
            default_quant: None,
            variants: HashMap::new(),
            description: None,
        }
    }

    /// Set default quantization
    #[must_use]
    pub fn with_default_quant(mut self, quant: impl Into<String>) -> Self {
        self.default_quant = Some(quant.into());
        self
    }

    /// Add a size variant
    #[must_use]
    pub fn with_variant(mut self, tag: impl Into<String>, target: impl Into<String>) -> Self {
        self.variants.insert(tag.into(), target.into());
        self
    }

    /// Set description
    #[must_use]
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Resolve a variant tag to full target
    #[must_use]
    pub fn resolve_variant(&self, variant: Option<&str>) -> &str {
        match variant {
            Some(v) => self.variants.get(v).map(String::as_str).unwrap_or(&self.target),
            None => &self.target,
        }
    }
}

// ============================================================================
// ALIAS-002: Parsed Model Reference
// ============================================================================

/// Parsed model reference from alias format
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedRef {
    /// Model name (alias or full name)
    pub name: String,
    /// Size variant (e.g., "8b", "70b")
    pub variant: Option<String>,
    /// Quantization tag (e.g., "q4", "q8")
    pub quantization: Option<String>,
}

impl ParsedRef {
    /// Parse a model reference string
    ///
    /// Formats:
    /// - `name` -> name only
    /// - `name:variant` -> name with variant
    /// - `name:variant-quant` -> name with variant and quantization
    /// - `name-quant` -> name with quantization (no variant)
    /// - `scheme://path` -> full URI (not parsed as alias)
    #[must_use]
    pub fn parse(s: &str) -> Self {
        // Check if this is a full URI with scheme (contains "://")
        if s.contains("://") {
            return Self { name: s.to_string(), variant: None, quantization: None };
        }

        // Split on colon first (but only if not part of a scheme)
        let (name_part, tag_part) =
            if let Some(idx) = s.find(':') { (&s[..idx], Some(&s[idx + 1..])) } else { (s, None) };

        // Check for quantization suffix in name part
        let (name, name_quant) = extract_quant_suffix(name_part);

        // Parse tag part for variant and quantization
        let (variant, tag_quant) =
            if let Some(tag) = tag_part { parse_tag(tag) } else { (None, None) };

        // Quantization from tag takes precedence
        let quantization = tag_quant.or(name_quant);

        Self { name: name.to_string(), variant, quantization }
    }

    /// Format as string
    #[must_use]
    pub fn to_string_repr(&self) -> String {
        let mut s = self.name.clone();
        if let Some(ref v) = self.variant {
            s.push(':');
            s.push_str(v);
        }
        if let Some(ref q) = self.quantization {
            if self.variant.is_some() {
                s.push('-');
            } else {
                s.push(':');
            }
            s.push_str(q);
        }
        s
    }
}

/// Extract quantization suffix from name (e.g., "llama-q4" -> ("llama", Some("q4")))
fn extract_quant_suffix(s: &str) -> (&str, Option<String>) {
    // Common quant suffixes
    let quant_patterns = [
        "-q4", "-q5", "-q6", "-q8", "-Q4_K_M", "-Q4_K_S", "-Q5_K_M", "-Q5_K_S", "-Q6_K", "-Q8_0",
        "-Q8_K", "-f16", "-f32", "-bf16",
    ];

    for pattern in quant_patterns {
        if let Some(idx) = s.to_lowercase().rfind(&pattern.to_lowercase()) {
            return (&s[..idx], Some(s[idx + 1..].to_string()));
        }
    }

    (s, None)
}

/// Parse tag into variant and quantization
fn parse_tag(tag: &str) -> (Option<String>, Option<String>) {
    // Check for variant-quant pattern (e.g., "8b-q4")
    if let Some(idx) = tag.rfind('-') {
        let (variant, quant) = (&tag[..idx], &tag[idx + 1..]);
        if is_quant_tag(quant) {
            return (Some(variant.to_string()), Some(quant.to_string()));
        }
    }

    // Check if entire tag is quantization
    if is_quant_tag(tag) {
        return (None, Some(tag.to_string()));
    }

    // Otherwise it's a variant
    (Some(tag.to_string()), None)
}

/// Check if a string is a quantization tag
fn is_quant_tag(s: &str) -> bool {
    let lower = s.to_lowercase();
    lower.starts_with("q") && lower.chars().nth(1).is_some_and(|c| c.is_ascii_digit())
        || lower.starts_with("iq")
        || lower == "f16"
        || lower == "f32"
        || lower == "fp16"
        || lower == "fp32"
        || lower == "bf16"
}

// ============================================================================
// ALIAS-003: Alias Registry
// ============================================================================

/// Registry of model aliases
#[derive(Debug, Clone, Default)]
pub struct AliasRegistry {
    /// Alias entries by short name
    aliases: HashMap<String, AliasEntry>,
}

impl AliasRegistry {
    /// Create a new empty registry
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create registry with default aliases
    #[must_use]
    pub fn with_defaults() -> Self {
        let mut registry = Self::new();

        // Llama 3 family
        registry.add(
            AliasEntry::new("llama3", "hf://meta-llama/Meta-Llama-3-8B-Instruct")
                .with_default_quant("Q4_K_M")
                .with_variant("8b", "hf://meta-llama/Meta-Llama-3-8B-Instruct")
                .with_variant("70b", "hf://meta-llama/Meta-Llama-3-70B-Instruct")
                .with_description("Meta's Llama 3 family"),
        );

        // Llama 3.1 family
        registry.add(
            AliasEntry::new("llama3.1", "hf://meta-llama/Meta-Llama-3.1-8B-Instruct")
                .with_default_quant("Q4_K_M")
                .with_variant("8b", "hf://meta-llama/Meta-Llama-3.1-8B-Instruct")
                .with_variant("70b", "hf://meta-llama/Meta-Llama-3.1-70B-Instruct")
                .with_variant("405b", "hf://meta-llama/Meta-Llama-3.1-405B-Instruct")
                .with_description("Meta's Llama 3.1 family"),
        );

        // Mistral family
        registry.add(
            AliasEntry::new("mistral", "hf://mistralai/Mistral-7B-Instruct-v0.3")
                .with_default_quant("Q4_K_M")
                .with_variant("7b", "hf://mistralai/Mistral-7B-Instruct-v0.3")
                .with_description("Mistral AI's 7B model"),
        );

        // Mixtral
        registry.add(
            AliasEntry::new("mixtral", "hf://mistralai/Mixtral-8x7B-Instruct-v0.1")
                .with_default_quant("Q4_K_M")
                .with_variant("8x7b", "hf://mistralai/Mixtral-8x7B-Instruct-v0.1")
                .with_variant("8x22b", "hf://mistralai/Mixtral-8x22B-Instruct-v0.1")
                .with_description("Mistral AI's Mixtral MoE"),
        );

        // Phi family
        registry.add(
            AliasEntry::new("phi3", "hf://microsoft/Phi-3-mini-4k-instruct")
                .with_default_quant("Q4_K_M")
                .with_variant("mini", "hf://microsoft/Phi-3-mini-4k-instruct")
                .with_variant("small", "hf://microsoft/Phi-3-small-8k-instruct")
                .with_variant("medium", "hf://microsoft/Phi-3-medium-4k-instruct")
                .with_description("Microsoft's Phi-3 family"),
        );

        // Gemma family
        registry.add(
            AliasEntry::new("gemma", "hf://google/gemma-7b-it")
                .with_default_quant("Q4_K_M")
                .with_variant("2b", "hf://google/gemma-2b-it")
                .with_variant("7b", "hf://google/gemma-7b-it")
                .with_description("Google's Gemma family"),
        );

        registry.add(
            AliasEntry::new("gemma2", "hf://google/gemma-2-9b-it")
                .with_default_quant("Q4_K_M")
                .with_variant("2b", "hf://google/gemma-2-2b-it")
                .with_variant("9b", "hf://google/gemma-2-9b-it")
                .with_variant("27b", "hf://google/gemma-2-27b-it")
                .with_description("Google's Gemma 2 family"),
        );

        // Qwen family
        registry.add(
            AliasEntry::new("qwen2", "hf://Qwen/Qwen2-7B-Instruct")
                .with_default_quant("Q4_K_M")
                .with_variant("0.5b", "hf://Qwen/Qwen2-0.5B-Instruct")
                .with_variant("1.5b", "hf://Qwen/Qwen2-1.5B-Instruct")
                .with_variant("7b", "hf://Qwen/Qwen2-7B-Instruct")
                .with_variant("72b", "hf://Qwen/Qwen2-72B-Instruct")
                .with_description("Alibaba's Qwen2 family"),
        );

        // CodeLlama
        registry.add(
            AliasEntry::new("codellama", "hf://codellama/CodeLlama-7b-Instruct-hf")
                .with_default_quant("Q4_K_M")
                .with_variant("7b", "hf://codellama/CodeLlama-7b-Instruct-hf")
                .with_variant("13b", "hf://codellama/CodeLlama-13b-Instruct-hf")
                .with_variant("34b", "hf://codellama/CodeLlama-34b-Instruct-hf")
                .with_description("Meta's CodeLlama"),
        );

        // DeepSeek Coder
        registry.add(
            AliasEntry::new("deepseek-coder", "hf://deepseek-ai/deepseek-coder-6.7b-instruct")
                .with_default_quant("Q4_K_M")
                .with_variant("1.3b", "hf://deepseek-ai/deepseek-coder-1.3b-instruct")
                .with_variant("6.7b", "hf://deepseek-ai/deepseek-coder-6.7b-instruct")
                .with_variant("33b", "hf://deepseek-ai/deepseek-coder-33b-instruct")
                .with_description("DeepSeek AI Coder"),
        );

        // StarCoder
        registry.add(
            AliasEntry::new("starcoder2", "hf://bigcode/starcoder2-7b")
                .with_default_quant("Q4_K_M")
                .with_variant("3b", "hf://bigcode/starcoder2-3b")
                .with_variant("7b", "hf://bigcode/starcoder2-7b")
                .with_variant("15b", "hf://bigcode/starcoder2-15b")
                .with_description("BigCode's StarCoder 2"),
        );

        // Embedding models
        registry.add(
            AliasEntry::new("nomic-embed", "hf://nomic-ai/nomic-embed-text-v1.5")
                .with_description("Nomic AI embedding model"),
        );

        registry.add(
            AliasEntry::new("bge", "hf://BAAI/bge-large-en-v1.5")
                .with_variant("small", "hf://BAAI/bge-small-en-v1.5")
                .with_variant("base", "hf://BAAI/bge-base-en-v1.5")
                .with_variant("large", "hf://BAAI/bge-large-en-v1.5")
                .with_description("BGE embedding models"),
        );

        registry
    }

    /// Add an alias entry
    pub fn add(&mut self, entry: AliasEntry) {
        self.aliases.insert(entry.alias.clone(), entry);
    }

    /// Get an alias entry
    #[must_use]
    pub fn get(&self, alias: &str) -> Option<&AliasEntry> {
        self.aliases.get(alias)
    }

    /// Check if an alias exists
    #[must_use]
    pub fn contains(&self, alias: &str) -> bool {
        self.aliases.contains_key(alias)
    }

    /// Get all aliases
    #[must_use]
    pub fn list(&self) -> Vec<&AliasEntry> {
        let mut entries: Vec<_> = self.aliases.values().collect();
        entries.sort_by(|a, b| a.alias.cmp(&b.alias));
        entries
    }

    /// Resolve a model reference
    ///
    /// Returns the full URI and optional quantization
    #[must_use]
    pub fn resolve(&self, reference: &str) -> ResolvedAlias {
        let parsed = ParsedRef::parse(reference);

        // Check if it's a known alias
        if let Some(entry) = self.aliases.get(&parsed.name) {
            let target = entry.resolve_variant(parsed.variant.as_deref());
            let quant = parsed.quantization.or_else(|| entry.default_quant.clone());

            ResolvedAlias { uri: target.to_string(), quantization: quant, is_alias: true }
        } else {
            // Not an alias, return as-is
            // Determine scheme based on format:
            // - Contains "://" -> use as-is
            // - Contains "/" (like "org/repo") -> assume HuggingFace (hf://)
            // - Otherwise -> assume pacha://
            let uri = if parsed.name.contains("://") {
                parsed.name.clone()
            } else if parsed.name.contains('/') {
                // HuggingFace-style org/repo format
                format!("hf://{}", parsed.name)
            } else {
                format!("pacha://{}", parsed.name)
            };

            ResolvedAlias { uri, quantization: parsed.quantization, is_alias: false }
        }
    }

    /// Get number of aliases
    #[must_use]
    pub fn len(&self) -> usize {
        self.aliases.len()
    }

    /// Check if registry is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.aliases.is_empty()
    }
}

/// Resolved alias result
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedAlias {
    /// Full model URI
    pub uri: String,
    /// Quantization (if specified)
    pub quantization: Option<String>,
    /// Whether this was resolved from an alias
    pub is_alias: bool,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // ALIAS-001: Entry Tests
    // ========================================================================

    #[test]
    fn test_alias_entry_new() {
        let entry = AliasEntry::new("llama3", "hf://meta-llama/Llama-3-8B");
        assert_eq!(entry.alias, "llama3");
        assert_eq!(entry.target, "hf://meta-llama/Llama-3-8B");
    }

    #[test]
    fn test_alias_entry_builder() {
        let entry = AliasEntry::new("llama3", "hf://meta-llama/Llama-3-8B")
            .with_default_quant("Q4_K_M")
            .with_variant("70b", "hf://meta-llama/Llama-3-70B")
            .with_description("Llama 3 family");

        assert_eq!(entry.default_quant, Some("Q4_K_M".to_string()));
        assert!(entry.variants.contains_key("70b"));
        assert!(entry.description.is_some());
    }

    #[test]
    fn test_alias_entry_resolve_variant() {
        let entry = AliasEntry::new("llama3", "hf://default")
            .with_variant("8b", "hf://8b-model")
            .with_variant("70b", "hf://70b-model");

        assert_eq!(entry.resolve_variant(None), "hf://default");
        assert_eq!(entry.resolve_variant(Some("8b")), "hf://8b-model");
        assert_eq!(entry.resolve_variant(Some("70b")), "hf://70b-model");
        assert_eq!(entry.resolve_variant(Some("unknown")), "hf://default");
    }

    // ========================================================================
    // ALIAS-002: Parsing Tests
    // ========================================================================

    #[test]
    fn test_parsed_ref_name_only() {
        let parsed = ParsedRef::parse("llama3");
        assert_eq!(parsed.name, "llama3");
        assert!(parsed.variant.is_none());
        assert!(parsed.quantization.is_none());
    }

    #[test]
    fn test_parsed_ref_with_variant() {
        let parsed = ParsedRef::parse("llama3:8b");
        assert_eq!(parsed.name, "llama3");
        assert_eq!(parsed.variant, Some("8b".to_string()));
        assert!(parsed.quantization.is_none());
    }

    #[test]
    fn test_parsed_ref_with_variant_and_quant() {
        let parsed = ParsedRef::parse("llama3:8b-q4");
        assert_eq!(parsed.name, "llama3");
        assert_eq!(parsed.variant, Some("8b".to_string()));
        assert_eq!(parsed.quantization, Some("q4".to_string()));
    }

    #[test]
    fn test_parsed_ref_quant_only() {
        let parsed = ParsedRef::parse("llama3:q4");
        assert_eq!(parsed.name, "llama3");
        assert!(parsed.variant.is_none());
        assert_eq!(parsed.quantization, Some("q4".to_string()));
    }

    #[test]
    fn test_parsed_ref_uppercase_quant() {
        let parsed = ParsedRef::parse("llama3:8b-Q4_K_M");
        assert_eq!(parsed.name, "llama3");
        assert_eq!(parsed.variant, Some("8b".to_string()));
        assert_eq!(parsed.quantization, Some("Q4_K_M".to_string()));
    }

    #[test]
    fn test_parsed_ref_to_string() {
        let parsed = ParsedRef::parse("llama3:8b-q4");
        assert_eq!(parsed.to_string_repr(), "llama3:8b-q4");

        let parsed = ParsedRef::parse("llama3:q4");
        assert_eq!(parsed.to_string_repr(), "llama3:q4");

        let parsed = ParsedRef::parse("llama3");
        assert_eq!(parsed.to_string_repr(), "llama3");
    }

    #[test]
    fn test_is_quant_tag() {
        assert!(is_quant_tag("q4"));
        assert!(is_quant_tag("Q4_K_M"));
        assert!(is_quant_tag("q8"));
        assert!(is_quant_tag("f16"));
        assert!(is_quant_tag("fp16"));
        assert!(is_quant_tag("bf16"));
        assert!(is_quant_tag("iq4"));
        assert!(!is_quant_tag("8b"));
        assert!(!is_quant_tag("70b"));
        assert!(!is_quant_tag("instruct"));
    }

    // ========================================================================
    // ALIAS-003: Registry Tests
    // ========================================================================

    #[test]
    fn test_alias_registry_new() {
        let registry = AliasRegistry::new();
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);
    }

    #[test]
    fn test_alias_registry_with_defaults() {
        let registry = AliasRegistry::with_defaults();
        assert!(!registry.is_empty());
        assert!(registry.contains("llama3"));
        assert!(registry.contains("mistral"));
        assert!(registry.contains("phi3"));
    }

    #[test]
    fn test_alias_registry_add() {
        let mut registry = AliasRegistry::new();
        registry.add(AliasEntry::new("test", "hf://test/model"));

        assert!(registry.contains("test"));
        assert_eq!(registry.len(), 1);
    }

    #[test]
    fn test_alias_registry_get() {
        let registry = AliasRegistry::with_defaults();
        let entry = registry.get("llama3");

        assert!(entry.is_some());
        assert!(entry.unwrap().target.contains("meta-llama"));
    }

    #[test]
    fn test_alias_registry_list() {
        let mut registry = AliasRegistry::new();
        registry.add(AliasEntry::new("zzz", "hf://zzz"));
        registry.add(AliasEntry::new("aaa", "hf://aaa"));

        let list = registry.list();
        assert_eq!(list.len(), 2);
        assert_eq!(list[0].alias, "aaa"); // Sorted
        assert_eq!(list[1].alias, "zzz");
    }

    #[test]
    fn test_alias_registry_resolve_known() {
        let registry = AliasRegistry::with_defaults();

        let resolved = registry.resolve("llama3");
        assert!(resolved.is_alias);
        assert!(resolved.uri.contains("meta-llama"));
    }

    #[test]
    fn test_alias_registry_resolve_with_variant() {
        let registry = AliasRegistry::with_defaults();

        let resolved = registry.resolve("llama3:70b");
        assert!(resolved.is_alias);
        assert!(resolved.uri.contains("70B"));
    }

    #[test]
    fn test_alias_registry_resolve_with_quant() {
        let registry = AliasRegistry::with_defaults();

        let resolved = registry.resolve("llama3:8b-q8");
        assert!(resolved.is_alias);
        assert_eq!(resolved.quantization, Some("q8".to_string()));
    }

    #[test]
    fn test_alias_registry_resolve_default_quant() {
        let registry = AliasRegistry::with_defaults();

        let resolved = registry.resolve("llama3");
        assert!(resolved.is_alias);
        // Should have default quant
        assert_eq!(resolved.quantization, Some("Q4_K_M".to_string()));
    }

    #[test]
    fn test_alias_registry_resolve_unknown() {
        let registry = AliasRegistry::with_defaults();

        let resolved = registry.resolve("unknown-model");
        assert!(!resolved.is_alias);
        assert_eq!(resolved.uri, "pacha://unknown-model");
    }

    #[test]
    fn test_alias_registry_resolve_huggingface_style() {
        let registry = AliasRegistry::with_defaults();

        // HuggingFace-style org/repo format should default to hf:// scheme
        let resolved = registry.resolve("Qwen/Qwen2.5-Coder-0.5B-Instruct");
        assert!(!resolved.is_alias);
        assert_eq!(resolved.uri, "hf://Qwen/Qwen2.5-Coder-0.5B-Instruct");

        // Another example
        let resolved = registry.resolve("TheBloke/Llama-2-7B-GGUF");
        assert!(!resolved.is_alias);
        assert_eq!(resolved.uri, "hf://TheBloke/Llama-2-7B-GGUF");
    }

    #[test]
    fn test_alias_registry_resolve_full_uri() {
        let registry = AliasRegistry::with_defaults();

        let resolved = registry.resolve("hf://some/model");
        assert!(!resolved.is_alias);
        assert_eq!(resolved.uri, "hf://some/model");
    }

    // ========================================================================
    // Serialization Tests
    // ========================================================================

    #[test]
    fn test_alias_entry_serialization() {
        let entry = AliasEntry::new("llama3", "hf://test")
            .with_default_quant("Q4_K_M")
            .with_variant("70b", "hf://test-70b");

        let json = serde_json::to_string(&entry).unwrap();
        assert!(json.contains("llama3"));
        assert!(json.contains("Q4_K_M"));

        let parsed: AliasEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.alias, "llama3");
        assert_eq!(parsed.default_quant, Some("Q4_K_M".to_string()));
    }

    // ========================================================================
    // Edge Cases
    // ========================================================================

    #[test]
    fn test_parsed_ref_complex_name() {
        let parsed = ParsedRef::parse("deepseek-coder:6.7b-q4");
        assert_eq!(parsed.name, "deepseek-coder");
        assert_eq!(parsed.variant, Some("6.7b".to_string()));
        assert_eq!(parsed.quantization, Some("q4".to_string()));
    }

    #[test]
    fn test_parsed_ref_numbers_in_name() {
        let parsed = ParsedRef::parse("llama3.1:8b");
        assert_eq!(parsed.name, "llama3.1");
        assert_eq!(parsed.variant, Some("8b".to_string()));
    }

    #[test]
    fn test_resolve_quant_override() {
        let registry = AliasRegistry::with_defaults();

        // Explicit quant should override default
        let resolved = registry.resolve("llama3:q8");
        assert_eq!(resolved.quantization, Some("q8".to_string()));
    }

    #[test]
    fn test_gemma_variants() {
        let registry = AliasRegistry::with_defaults();

        let resolved_2b = registry.resolve("gemma:2b");
        assert!(resolved_2b.uri.contains("2b"));

        let resolved_7b = registry.resolve("gemma:7b");
        assert!(resolved_7b.uri.contains("7b"));
    }

    #[test]
    fn test_embedding_models() {
        let registry = AliasRegistry::with_defaults();

        assert!(registry.contains("nomic-embed"));
        assert!(registry.contains("bge"));

        let resolved = registry.resolve("bge:large");
        assert!(resolved.uri.contains("large"));
    }
}
