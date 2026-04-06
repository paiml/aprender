/// Plugin Architecture for Custom Transpilers
///
/// Provides extensible plugin system for custom transpiler implementations.
/// Plugins can register as pipeline stages and integrate seamlessly with Batuta.
///
/// # Architecture
///
/// - **TranspilerPlugin**: Core trait for custom transpiler plugins
/// - **PluginRegistry**: Central registry for discovering and managing plugins
/// - **PluginMetadata**: Plugin information (name, version, supported languages)
/// - **PluginLoader**: Dynamic plugin loading mechanism
///
/// # Example
///
/// ```rust,no_run
/// use batuta::plugin::{TranspilerPlugin, PluginMetadata, PluginRegistry};
/// use batuta::pipeline::{PipelineContext, PipelineStage};
/// use batuta::types::Language;
/// use anyhow::Result;
///
/// // Define a custom transpiler plugin
/// struct MyCustomTranspiler;
///
/// impl TranspilerPlugin for MyCustomTranspiler {
///     fn metadata(&self) -> PluginMetadata {
///         PluginMetadata {
///             name: "my-custom-transpiler".to_string(),
///             version: "0.1.0".to_string(),
///             description: "Custom transpiler for my language".to_string(),
///             author: "Your Name".to_string(),
///             supported_languages: vec![Language::Python],
///         }
///     }
///
///     fn initialize(&mut self) -> Result<()> {
///         println!("Initializing custom transpiler");
///         Ok(())
///     }
///
///     fn transpile(&self, source: &str, language: Language) -> Result<String> {
///         // Custom transpilation logic
///         Ok(format!("// Transpiled from {:?}\n{}", language, source))
///     }
/// }
///
/// // Register and use the plugin
/// let mut registry = PluginRegistry::new();
/// registry.register(Box::new(MyCustomTranspiler))?;
///
/// // Get plugins for a language
/// let plugins = registry.get_for_language(&Language::Python);
/// if let Some(plugin) = plugins.first() {
///     let output = plugin.transpile("print('hello')", Language::Python)?;
///     println!("{}", output);
/// }
/// # Ok::<(), anyhow::Error>(())
/// ```
use anyhow::{anyhow, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

use crate::pipeline::{PipelineContext, PipelineStage, ValidationResult};
use crate::types::Language;

/// Metadata describing a transpiler plugin
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginMetadata {
    /// Unique plugin identifier (kebab-case recommended)
    pub name: String,

    /// Semantic version (e.g., "1.0.0")
    pub version: String,

    /// Human-readable description
    pub description: String,

    /// Plugin author or organization
    pub author: String,

    /// Languages this plugin can transpile
    pub supported_languages: Vec<Language>,
}

impl PluginMetadata {
    /// Check if this plugin supports a given language
    pub fn supports_language(&self, lang: &Language) -> bool {
        self.supported_languages.contains(lang)
    }
}

/// Core trait for transpiler plugins
///
/// Implement this trait to create custom transpilers that integrate with Batuta's pipeline.
pub trait TranspilerPlugin: Send + Sync {
    /// Get plugin metadata
    fn metadata(&self) -> PluginMetadata;

    /// Initialize the plugin
    ///
    /// Called once when the plugin is loaded. Use this for setup tasks like:
    /// - Loading configuration
    /// - Initializing caches
    /// - Validating dependencies
    fn initialize(&mut self) -> Result<()> {
        Ok(())
    }

    /// Transpile source code to Rust
    ///
    /// # Arguments
    ///
    /// * `source` - Source code to transpile
    /// * `language` - Source language
    ///
    /// # Returns
    ///
    /// Rust code as a String
    fn transpile(&self, source: &str, language: Language) -> Result<String>;

    /// Transpile a file
    ///
    /// Default implementation reads the file and calls `transpile()`.
    /// Override for custom file handling (e.g., imports, multi-file projects).
    fn transpile_file(&self, path: &Path, language: Language) -> Result<String> {
        let source = std::fs::read_to_string(path)?;
        self.transpile(&source, language)
    }

    /// Validate transpiled output
    ///
    /// Optional validation hook. Override to add custom validation logic.
    fn validate(&self, _original: &str, _transpiled: &str) -> Result<()> {
        Ok(())
    }

    /// Cleanup resources
    ///
    /// Called when the plugin is unloaded or the pipeline completes.
    fn cleanup(&mut self) -> Result<()> {
        Ok(())
    }
}

/// Wrapper to integrate a plugin as a pipeline stage
pub struct PluginStage {
    plugin: Box<dyn TranspilerPlugin>,
    name: String,
}

impl PluginStage {
    pub fn new(plugin: Box<dyn TranspilerPlugin>) -> Self {
        let name = plugin.metadata().name.clone();
        Self { plugin, name }
    }
}

#[async_trait]
impl PipelineStage for PluginStage {
    fn name(&self) -> &str {
        &self.name
    }

    async fn execute(&self, mut ctx: PipelineContext) -> Result<PipelineContext> {
        let metadata = self.plugin.metadata();

        // Check if we have a language to transpile
        let language =
            ctx.primary_language.clone().ok_or_else(|| anyhow!("No primary language detected"))?;

        // Check if plugin supports this language
        if !metadata.supports_language(&language) {
            return Err(anyhow!("Plugin '{}' does not support {:?}", metadata.name, language));
        }

        // Transpile files
        for (source_path, output_path) in &ctx.file_mappings.clone() {
            let transpiled = self.plugin.transpile_file(source_path, language.clone())?;

            // Write output
            if let Some(parent) = output_path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(output_path, transpiled)?;
        }

        // Record plugin execution in metadata
        ctx.metadata.insert(
            format!("plugin_{}", metadata.name),
            serde_json::json!({
                "version": metadata.version,
                "files_processed": ctx.file_mappings.len(),
            }),
        );

        Ok(ctx)
    }

    fn validate(&self, ctx: &PipelineContext) -> Result<ValidationResult> {
        let metadata = self.plugin.metadata();

        // Validate all transpiled files
        for (source_path, output_path) in &ctx.file_mappings {
            let original = std::fs::read_to_string(source_path)?;
            let transpiled = std::fs::read_to_string(output_path)?;

            self.plugin.validate(&original, &transpiled)?;
        }

        Ok(ValidationResult {
            stage: metadata.name.clone(),
            passed: true,
            message: format!(
                "Plugin '{}' validation passed for {} files",
                metadata.name,
                ctx.file_mappings.len()
            ),
            details: None,
        })
    }
}

/// Plugin registry for managing transpiler plugins
pub struct PluginRegistry {
    plugins: Vec<Box<dyn TranspilerPlugin>>,
    language_map: HashMap<Language, Vec<String>>, // Language -> plugin names
}

impl PluginRegistry {
    /// Create a new empty plugin registry
    pub fn new() -> Self {
        Self { plugins: Vec::new(), language_map: HashMap::new() }
    }

    /// Register a transpiler plugin
    pub fn register(&mut self, mut plugin: Box<dyn TranspilerPlugin>) -> Result<()> {
        // Initialize the plugin
        plugin.initialize()?;

        let metadata = plugin.metadata();

        // Update language map
        for lang in &metadata.supported_languages {
            self.language_map.entry(lang.clone()).or_default().push(metadata.name.clone());
        }

        // Store plugin
        self.plugins.push(plugin);

        Ok(())
    }

    /// Get plugin by name
    pub fn get(&self, name: &str) -> Option<&dyn TranspilerPlugin> {
        self.plugins.iter().find(|p| p.metadata().name == name).map(|p| &**p)
    }

    /// Get mutable reference to plugin by name
    pub fn get_mut(&mut self, name: &str) -> Option<&mut dyn TranspilerPlugin> {
        for plugin in &mut self.plugins {
            if plugin.metadata().name == name {
                return Some(&mut **plugin);
            }
        }
        None
    }

    /// Get all plugins that support a language
    pub fn get_for_language(&self, language: &Language) -> Vec<&dyn TranspilerPlugin> {
        self.plugins
            .iter()
            .filter(|p| p.metadata().supports_language(language))
            .map(|p| &**p as &dyn TranspilerPlugin)
            .collect()
    }

    /// Get all registered plugin names
    pub fn list_plugins(&self) -> Vec<String> {
        self.plugins.iter().map(|p| p.metadata().name.clone()).collect()
    }

    /// Get languages supported by all plugins
    pub fn supported_languages(&self) -> Vec<Language> {
        self.language_map.keys().cloned().collect()
    }

    /// Unregister a plugin and cleanup
    pub fn unregister(&mut self, name: &str) -> Result<()> {
        if let Some(pos) = self.plugins.iter().position(|p| p.metadata().name == name) {
            let mut plugin = self.plugins.remove(pos);
            plugin.cleanup()?;

            // Update language map
            let metadata = plugin.metadata();
            for lang in &metadata.supported_languages {
                if let Some(names) = self.language_map.get_mut(lang) {
                    names.retain(|n| n != &metadata.name);
                    if names.is_empty() {
                        self.language_map.remove(lang);
                    }
                }
            }
        }

        Ok(())
    }

    /// Cleanup all plugins
    pub fn cleanup_all(&mut self) -> Result<()> {
        for plugin in &mut self.plugins {
            plugin.cleanup()?;
        }
        self.plugins.clear();
        self.language_map.clear();
        Ok(())
    }

    /// Get number of registered plugins
    pub fn len(&self) -> usize {
        self.plugins.len()
    }

    /// Check if registry is empty
    pub fn is_empty(&self) -> bool {
        self.plugins.is_empty()
    }
}

impl Default for PluginRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for PluginRegistry {
    fn drop(&mut self) {
        // Best-effort cleanup on drop
        let _ = self.cleanup_all();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestPlugin {
        name: String,
        languages: Vec<Language>,
    }

    impl TranspilerPlugin for TestPlugin {
        fn metadata(&self) -> PluginMetadata {
            PluginMetadata {
                name: self.name.clone(),
                version: "1.0.0".to_string(),
                description: "Test plugin".to_string(),
                author: "Test Author".to_string(),
                supported_languages: self.languages.clone(),
            }
        }

        fn transpile(&self, source: &str, _language: Language) -> Result<String> {
            Ok(format!("// Transpiled by {}\n{}", self.name, source))
        }
    }

    #[test]
    fn test_plugin_registration() {
        let mut registry = PluginRegistry::new();

        let plugin = Box::new(TestPlugin {
            name: "test-plugin".to_string(),
            languages: vec![Language::Python],
        });

        assert!(registry.register(plugin).is_ok());
        assert_eq!(registry.len(), 1);
        assert!(registry.get("test-plugin").is_some());
    }

    #[test]
    fn test_language_lookup() {
        let mut registry = PluginRegistry::new();

        let plugin = Box::new(TestPlugin {
            name: "python-plugin".to_string(),
            languages: vec![Language::Python],
        });

        registry.register(plugin).expect("registration failed");

        let plugins = registry.get_for_language(&Language::Python);
        assert_eq!(plugins.len(), 1);

        let plugins = registry.get_for_language(&Language::C);
        assert_eq!(plugins.len(), 0);
    }

    #[test]
    fn test_plugin_unregister() {
        let mut registry = PluginRegistry::new();

        let plugin = Box::new(TestPlugin {
            name: "test-plugin".to_string(),
            languages: vec![Language::Python],
        });

        registry.register(plugin).expect("registration failed");
        assert_eq!(registry.len(), 1);

        registry.unregister("test-plugin").expect("unregistration failed");
        assert_eq!(registry.len(), 0);
        assert!(registry.get("test-plugin").is_none());
    }

    // ============================================================================
    // PLUGIN METADATA TESTS
    // ============================================================================

    #[test]
    fn test_plugin_metadata_construction() {
        let metadata = PluginMetadata {
            name: "my-plugin".to_string(),
            version: "1.2.3".to_string(),
            description: "A test plugin".to_string(),
            author: "Test Author".to_string(),
            supported_languages: vec![Language::Python, Language::Rust],
        };

        assert_eq!(metadata.name, "my-plugin");
        assert_eq!(metadata.version, "1.2.3");
        assert_eq!(metadata.description, "A test plugin");
        assert_eq!(metadata.author, "Test Author");
        assert_eq!(metadata.supported_languages.len(), 2);
    }

    #[test]
    fn test_plugin_metadata_supports_language() {
        let metadata = PluginMetadata {
            name: "test".to_string(),
            version: "1.0.0".to_string(),
            description: "Test".to_string(),
            author: "Author".to_string(),
            supported_languages: vec![Language::Python, Language::C],
        };

        assert!(metadata.supports_language(&Language::Python));
        assert!(metadata.supports_language(&Language::C));
        assert!(!metadata.supports_language(&Language::Rust));
        assert!(!metadata.supports_language(&Language::Shell));
    }

    #[test]
    fn test_plugin_metadata_serialization() {
        let metadata = PluginMetadata {
            name: "serialize-test".to_string(),
            version: "0.1.0".to_string(),
            description: "Serialization test".to_string(),
            author: "Tester".to_string(),
            supported_languages: vec![Language::Python],
        };

        let json = serde_json::to_string(&metadata).expect("json serialize failed");
        let deserialized: PluginMetadata =
            serde_json::from_str(&json).expect("json deserialize failed");

        assert_eq!(metadata.name, deserialized.name);
        assert_eq!(metadata.version, deserialized.version);
        assert_eq!(metadata.description, deserialized.description);
        assert_eq!(metadata.author, deserialized.author);
        assert_eq!(metadata.supported_languages, deserialized.supported_languages);
    }

    #[test]
    fn test_plugin_metadata_empty_languages() {
        let metadata = PluginMetadata {
            name: "no-lang".to_string(),
            version: "1.0.0".to_string(),
            description: "No languages".to_string(),
            author: "Test".to_string(),
            supported_languages: vec![],
        };

        assert!(!metadata.supports_language(&Language::Python));
        assert_eq!(metadata.supported_languages.len(), 0);
    }

    // ============================================================================
    // TRANSPILER PLUGIN TRAIT TESTS
    // ============================================================================

    struct MinimalPlugin;

    impl TranspilerPlugin for MinimalPlugin {
        fn metadata(&self) -> PluginMetadata {
            PluginMetadata {
                name: "minimal".to_string(),
                version: "1.0.0".to_string(),
                description: "Minimal plugin".to_string(),
                author: "Test".to_string(),
                supported_languages: vec![Language::Python],
            }
        }

        fn transpile(&self, source: &str, _language: Language) -> Result<String> {
            Ok(format!("fn main() {{\n    // {}\n}}", source))
        }
    }

    #[test]
    fn test_plugin_default_initialize() {
        let mut plugin = MinimalPlugin;
        assert!(plugin.initialize().is_ok());
    }

    #[test]
    fn test_plugin_transpile() {
        let plugin = MinimalPlugin;
        let result =
            plugin.transpile("print('hello')", Language::Python).expect("unexpected failure");
        assert!(result.contains("fn main()"));
        assert!(result.contains("print('hello')"));
    }

    #[test]
    fn test_plugin_transpile_file() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let plugin = MinimalPlugin;

        let mut temp_file = NamedTempFile::new().expect("tempfile creation failed");
        temp_file.write_all(b"print('test')").expect("fs write failed");
        temp_file.flush().expect("unexpected failure");

        let result =
            plugin.transpile_file(temp_file.path(), Language::Python).expect("unexpected failure");
        assert!(result.contains("print('test')"));
    }

    #[test]
    fn test_plugin_default_validate() {
        let plugin = MinimalPlugin;
        assert!(plugin.validate("original", "transpiled").is_ok());
    }

    #[test]
    fn test_plugin_default_cleanup() {
        let mut plugin = MinimalPlugin;
        assert!(plugin.cleanup().is_ok());
    }

    // ============================================================================
    // PLUGIN STAGE TESTS
    // ============================================================================

    #[test]
    fn test_plugin_stage_construction() {
        let plugin = Box::new(TestPlugin {
            name: "stage-test".to_string(),
            languages: vec![Language::Python],
        });

        let stage = PluginStage::new(plugin);
        assert_eq!(stage.name(), "stage-test");
    }

    #[tokio::test]
    async fn test_plugin_stage_execute_no_language() {
        let plugin =
            Box::new(TestPlugin { name: "test".to_string(), languages: vec![Language::Python] });

        let stage = PluginStage::new(plugin);
        let ctx = PipelineContext::new(
            std::path::PathBuf::from("/tmp/source"),
            std::path::PathBuf::from("/tmp/output"),
        );

        let result = stage.execute(ctx).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("No primary language"));
    }

    #[tokio::test]
    async fn test_plugin_stage_execute_unsupported_language() {
        let plugin = Box::new(TestPlugin {
            name: "python-only".to_string(),
            languages: vec![Language::Python],
        });

        let stage = PluginStage::new(plugin);
        let mut ctx = PipelineContext::new(
            std::path::PathBuf::from("/tmp/source"),
            std::path::PathBuf::from("/tmp/output"),
        );
        ctx.primary_language = Some(Language::Rust);

        let result = stage.execute(ctx).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("does not support"));
    }

    #[tokio::test]
    async fn test_plugin_stage_execute_success() {
        use std::fs;
        use tempfile::TempDir;

        let temp_dir = TempDir::new().expect("tempdir creation failed");
        let source_path = temp_dir.path().join("input.py");
        let output_path = temp_dir.path().join("output.rs");

        fs::write(&source_path, "print('hello')").expect("fs write failed");

        let plugin = Box::new(TestPlugin {
            name: "transpiler".to_string(),
            languages: vec![Language::Python],
        });

        let stage = PluginStage::new(plugin);
        let mut ctx =
            PipelineContext::new(temp_dir.path().to_path_buf(), temp_dir.path().to_path_buf());
        ctx.primary_language = Some(Language::Python);
        ctx.file_mappings.push((source_path.clone(), output_path.clone()));

        let result = stage.execute(ctx).await;
        assert!(result.is_ok());

        let ctx = result.expect("operation failed");
        assert!(ctx.metadata.contains_key("plugin_transpiler"));
        assert!(output_path.exists());

        let content = fs::read_to_string(&output_path).expect("fs read failed");
        assert!(content.contains("Transpiled by transpiler"));
    }

    #[test]
    fn test_plugin_stage_validate_success() {
        use std::fs;
        use tempfile::TempDir;

        let temp_dir = TempDir::new().expect("tempdir creation failed");
        let source_path = temp_dir.path().join("source.py");
        let output_path = temp_dir.path().join("output.rs");

        fs::write(&source_path, "original").expect("fs write failed");
        fs::write(&output_path, "transpiled").expect("fs write failed");

        let plugin = Box::new(TestPlugin {
            name: "validator".to_string(),
            languages: vec![Language::Python],
        });

        let stage = PluginStage::new(plugin);
        let mut ctx =
            PipelineContext::new(temp_dir.path().to_path_buf(), temp_dir.path().to_path_buf());
        ctx.file_mappings.push((source_path, output_path));

        let result = stage.validate(&ctx).expect("validation failed");
        assert!(result.passed);
        assert_eq!(result.stage, "validator");
        assert!(result.message.contains("validation passed"));
    }

    // ============================================================================
    // PLUGIN REGISTRY TESTS
    // ============================================================================

    #[test]
    fn test_registry_default() {
        let registry = PluginRegistry::default();
        assert_eq!(registry.len(), 0);
        assert!(registry.is_empty());
    }

    #[test]
    fn test_registry_is_empty() {
        let mut registry = PluginRegistry::new();
        assert!(registry.is_empty());

        let plugin =
            Box::new(TestPlugin { name: "test".to_string(), languages: vec![Language::Python] });
        registry.register(plugin).expect("registration failed");

        assert!(!registry.is_empty());
    }

    #[test]
    fn test_registry_len() {
        let mut registry = PluginRegistry::new();
        assert_eq!(registry.len(), 0);

        registry
            .register(Box::new(TestPlugin {
                name: "plugin1".to_string(),
                languages: vec![Language::Python],
            }))
            .expect("unexpected failure");
        assert_eq!(registry.len(), 1);

        registry
            .register(Box::new(TestPlugin {
                name: "plugin2".to_string(),
                languages: vec![Language::Rust],
            }))
            .expect("unexpected failure");
        assert_eq!(registry.len(), 2);
    }

    #[test]
    fn test_registry_get_mut() {
        let mut registry = PluginRegistry::new();

        registry
            .register(Box::new(TestPlugin {
                name: "mutable-test".to_string(),
                languages: vec![Language::Python],
            }))
            .expect("unexpected failure");

        let plugin = registry.get_mut("mutable-test");
        assert!(plugin.is_some());
        assert_eq!(plugin.expect("unexpected failure").metadata().name, "mutable-test");

        let none_plugin = registry.get_mut("nonexistent");
        assert!(none_plugin.is_none());
    }

    #[test]
    fn test_registry_list_plugins() {
        let mut registry = PluginRegistry::new();

        registry
            .register(Box::new(TestPlugin {
                name: "plugin-a".to_string(),
                languages: vec![Language::Python],
            }))
            .expect("unexpected failure");

        registry
            .register(Box::new(TestPlugin {
                name: "plugin-b".to_string(),
                languages: vec![Language::Rust],
            }))
            .expect("unexpected failure");

        let list = registry.list_plugins();
        assert_eq!(list.len(), 2);
        assert!(list.contains(&"plugin-a".to_string()));
        assert!(list.contains(&"plugin-b".to_string()));
    }

    #[test]
    fn test_registry_supported_languages() {
        let mut registry = PluginRegistry::new();

        registry
            .register(Box::new(TestPlugin {
                name: "python-plugin".to_string(),
                languages: vec![Language::Python],
            }))
            .expect("unexpected failure");

        registry
            .register(Box::new(TestPlugin {
                name: "multi-plugin".to_string(),
                languages: vec![Language::Rust, Language::C],
            }))
            .expect("unexpected failure");

        let langs = registry.supported_languages();
        assert!(langs.len() >= 3);
        assert!(langs.contains(&Language::Python));
        assert!(langs.contains(&Language::Rust));
        assert!(langs.contains(&Language::C));
    }

    #[test]
    fn test_registry_multiple_plugins_same_language() {
        let mut registry = PluginRegistry::new();

        registry
            .register(Box::new(TestPlugin {
                name: "python-plugin-1".to_string(),
                languages: vec![Language::Python],
            }))
            .expect("unexpected failure");

        registry
            .register(Box::new(TestPlugin {
                name: "python-plugin-2".to_string(),
                languages: vec![Language::Python],
            }))
            .expect("unexpected failure");

        let plugins = registry.get_for_language(&Language::Python);
        assert_eq!(plugins.len(), 2);
    }

    #[test]
    fn test_registry_cleanup_all() {
        let mut registry = PluginRegistry::new();

        registry
            .register(Box::new(TestPlugin {
                name: "cleanup1".to_string(),
                languages: vec![Language::Python],
            }))
            .expect("unexpected failure");

        registry
            .register(Box::new(TestPlugin {
                name: "cleanup2".to_string(),
                languages: vec![Language::Rust],
            }))
            .expect("unexpected failure");

        assert_eq!(registry.len(), 2);

        registry.cleanup_all().expect("unexpected failure");

        assert_eq!(registry.len(), 0);
        assert!(registry.is_empty());
        assert_eq!(registry.supported_languages().len(), 0);
    }

    #[test]
    fn test_registry_unregister_nonexistent() {
        let mut registry = PluginRegistry::new();

        // Unregistering nonexistent plugin should not error
        let result = registry.unregister("nonexistent");
        assert!(result.is_ok());
    }

    #[test]
    fn test_registry_unregister_updates_language_map() {
        let mut registry = PluginRegistry::new();

        registry
            .register(Box::new(TestPlugin {
                name: "only-python".to_string(),
                languages: vec![Language::Python],
            }))
            .expect("unexpected failure");

        assert!(registry.supported_languages().contains(&Language::Python));

        registry.unregister("only-python").expect("unregistration failed");

        // Language map should be updated
        assert!(!registry.supported_languages().contains(&Language::Python));
    }

    #[test]
    fn test_registry_get_nonexistent() {
        let registry = PluginRegistry::new();
        assert!(registry.get("nonexistent").is_none());
    }

    #[test]
    fn test_registry_get_for_language_empty() {
        let registry = PluginRegistry::new();
        let plugins = registry.get_for_language(&Language::Python);
        assert_eq!(plugins.len(), 0);
    }

    #[test]
    fn test_plugin_multiple_languages() {
        let mut registry = PluginRegistry::new();

        registry
            .register(Box::new(TestPlugin {
                name: "multi-lang".to_string(),
                languages: vec![Language::Python, Language::Rust, Language::C],
            }))
            .expect("unexpected failure");

        // Should be accessible from all three languages
        assert_eq!(registry.get_for_language(&Language::Python).len(), 1);
        assert_eq!(registry.get_for_language(&Language::Rust).len(), 1);
        assert_eq!(registry.get_for_language(&Language::C).len(), 1);
        assert_eq!(registry.get_for_language(&Language::Shell).len(), 0);
    }

    // Test initialization failure handling
    struct FailingInitPlugin;

    impl TranspilerPlugin for FailingInitPlugin {
        fn metadata(&self) -> PluginMetadata {
            PluginMetadata {
                name: "failing".to_string(),
                version: "1.0.0".to_string(),
                description: "Fails on init".to_string(),
                author: "Test".to_string(),
                supported_languages: vec![Language::Python],
            }
        }

        fn initialize(&mut self) -> Result<()> {
            Err(anyhow!("Initialization failed"))
        }

        fn transpile(&self, _source: &str, _language: Language) -> Result<String> {
            Ok("".to_string())
        }
    }

    #[test]
    fn test_plugin_initialization_failure() {
        let mut registry = PluginRegistry::new();

        let plugin = Box::new(FailingInitPlugin);
        let result = registry.register(plugin);

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Initialization failed"));
        assert_eq!(registry.len(), 0);
    }
}
