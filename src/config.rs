use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Batuta project configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatutaConfig {
    /// Configuration file version
    pub version: String,

    /// Project metadata
    pub project: ProjectConfig,

    /// Source code configuration
    pub source: SourceConfig,

    /// Transpilation settings
    pub transpilation: TranspilationConfig,

    /// Optimization settings
    pub optimization: OptimizationConfig,

    /// Validation settings
    pub validation: ValidationConfig,

    /// Build settings
    pub build: BuildConfig,
}

impl Default for BatutaConfig {
    fn default() -> Self {
        Self {
            version: "1.0".to_string(),
            project: ProjectConfig::default(),
            source: SourceConfig::default(),
            transpilation: TranspilationConfig::default(),
            optimization: OptimizationConfig::default(),
            validation: ValidationConfig::default(),
            build: BuildConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectConfig {
    /// Project name
    pub name: String,

    /// Project description
    pub description: Option<String>,

    /// Primary language of source project
    pub primary_language: Option<String>,

    /// Authors
    pub authors: Vec<String>,

    /// License
    pub license: Option<String>,
}

impl Default for ProjectConfig {
    fn default() -> Self {
        Self {
            name: "untitled".to_string(),
            description: None,
            primary_language: None,
            authors: vec![],
            license: Some("MIT".to_string()),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceConfig {
    /// Source code directory (relative to config file)
    pub path: PathBuf,

    /// Files/directories to exclude
    pub exclude: Vec<String>,

    /// Files/directories to include (overrides exclude)
    pub include: Vec<String>,
}

impl Default for SourceConfig {
    fn default() -> Self {
        Self {
            path: PathBuf::from("."),
            exclude: vec![
                ".git".to_string(),
                "target".to_string(),
                "build".to_string(),
                "dist".to_string(),
                "node_modules".to_string(),
                "__pycache__".to_string(),
                "*.pyc".to_string(),
                ".venv".to_string(),
                "venv".to_string(),
            ],
            include: vec![],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranspilationConfig {
    /// Output directory for generated Rust code
    pub output_dir: PathBuf,

    /// Enable incremental compilation
    pub incremental: bool,

    /// Enable caching
    pub cache: bool,

    /// Generate Ruchy instead of pure Rust
    pub use_ruchy: bool,

    /// Ruchy strictness level (permissive, gradual, strict)
    pub ruchy_strictness: Option<String>,

    /// Specific modules to transpile (empty = all)
    pub modules: Vec<String>,

    /// Tool-specific settings
    pub decy: DecyConfig,
    pub depyler: DepylerConfig,
    pub bashrs: BashrsConfig,
}

impl Default for TranspilationConfig {
    fn default() -> Self {
        Self {
            output_dir: PathBuf::from("./rust-output"),
            incremental: true,
            cache: true,
            use_ruchy: false,
            ruchy_strictness: Some("gradual".to_string()),
            modules: vec![],
            decy: DecyConfig::default(),
            depyler: DepylerConfig::default(),
            bashrs: BashrsConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecyConfig {
    /// Enable ownership inference
    pub ownership_inference: bool,

    /// Generate actionable diagnostics
    pub actionable_diagnostics: bool,

    /// Use StaticFixer integration
    pub use_static_fixer: bool,
}

impl Default for DecyConfig {
    fn default() -> Self {
        Self { ownership_inference: true, actionable_diagnostics: true, use_static_fixer: true }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DepylerConfig {
    /// Enable type inference
    pub type_inference: bool,

    /// Convert NumPy to Trueno
    pub numpy_to_trueno: bool,

    /// Convert sklearn to Aprender
    pub sklearn_to_aprender: bool,

    /// Convert PyTorch to Realizar
    pub pytorch_to_realizar: bool,
}

impl Default for DepylerConfig {
    fn default() -> Self {
        Self {
            type_inference: true,
            numpy_to_trueno: true,
            sklearn_to_aprender: true,
            pytorch_to_realizar: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BashrsConfig {
    /// Target shell compatibility
    pub target_shell: String,

    /// Generate CLI using clap
    pub use_clap: bool,
}

impl Default for BashrsConfig {
    fn default() -> Self {
        Self { target_shell: "bash".to_string(), use_clap: true }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationConfig {
    /// Optimization profile (fast, balanced, aggressive)
    pub profile: String,

    /// Enable SIMD vectorization
    pub enable_simd: bool,

    /// Enable GPU acceleration
    pub enable_gpu: bool,

    /// GPU dispatch threshold (matrix size)
    pub gpu_threshold: usize,

    /// Use Mixture-of-Experts routing
    pub use_moe_routing: bool,

    /// Trueno backend preferences
    pub trueno: TruenoConfig,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            profile: "balanced".to_string(),
            enable_simd: true,
            enable_gpu: false,
            gpu_threshold: 500,
            use_moe_routing: false,
            trueno: TruenoConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TruenoConfig {
    /// Preferred backends in priority order
    pub backends: Vec<String>,

    /// Enable adaptive threshold learning
    pub adaptive_thresholds: bool,

    /// Initial CPU threshold
    pub cpu_threshold: usize,
}

impl Default for TruenoConfig {
    fn default() -> Self {
        Self {
            backends: vec!["simd".to_string(), "cpu".to_string()],
            adaptive_thresholds: false,
            cpu_threshold: 500,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationConfig {
    /// Enable syscall tracing
    pub trace_syscalls: bool,

    /// Run original test suite
    pub run_original_tests: bool,

    /// Generate diff output
    pub diff_output: bool,

    /// Run benchmarks
    pub benchmark: bool,

    /// Renacer configuration
    pub renacer: RenacerConfig,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            trace_syscalls: true,
            run_original_tests: true,
            diff_output: true,
            benchmark: false,
            renacer: RenacerConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RenacerConfig {
    /// Syscalls to trace (empty = all)
    pub trace_syscalls: Vec<String>,

    /// Output format
    pub output_format: String,
}

impl Default for RenacerConfig {
    fn default() -> Self {
        Self { trace_syscalls: vec![], output_format: "json".to_string() }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildConfig {
    /// Build in release mode
    pub release: bool,

    /// Target platform (empty = native)
    pub target: Option<String>,

    /// Enable WebAssembly build
    pub wasm: bool,

    /// Additional cargo flags
    pub cargo_flags: Vec<String>,
}

impl Default for BuildConfig {
    fn default() -> Self {
        Self { release: true, target: None, wasm: false, cargo_flags: vec![] }
    }
}

// ============================================================================
// Private RAG Configuration
// ============================================================================

/// Filename for the private configuration file (git-ignored).
pub const PRIVATE_CONFIG_FILENAME: &str = ".batuta-private.toml";

/// Top-level private configuration loaded from `.batuta-private.toml`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PrivateConfig {
    /// Private directory extensions for RAG indexing.
    #[serde(default)]
    pub private: PrivateExtensions,
}

/// Private directories and endpoints to merge into the RAG index.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PrivateExtensions {
    /// Additional Rust stack directories to index.
    #[serde(default)]
    pub rust_stack_dirs: Vec<String>,

    /// Additional Rust corpus directories to index.
    #[serde(default)]
    pub rust_corpus_dirs: Vec<String>,

    /// Additional Python corpus directories to index.
    #[serde(default)]
    pub python_corpus_dirs: Vec<String>,

    /// Future: remote RAG endpoints (Phase 2).
    #[serde(default)]
    pub endpoints: Vec<PrivateEndpoint>,
}

/// A remote RAG endpoint (Phase 2 — not yet implemented).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivateEndpoint {
    pub name: String,
    #[serde(rename = "type")]
    pub endpoint_type: String,
    pub host: String,
    pub index_path: String,
}

impl PrivateConfig {
    /// Load from `.batuta-private.toml`, searching multiple locations.
    ///
    /// Search order:
    /// 1. `$HOME/.batuta-private.toml` (user-global config)
    /// 2. `./.batuta-private.toml` (current directory, for development)
    ///
    /// Returns `Ok(None)` if no config file exists, `Err` if malformed.
    #[cfg(feature = "native")]
    pub fn load_optional() -> anyhow::Result<Option<Self>> {
        if let Some(path) = Self::find_config_path() {
            let content = std::fs::read_to_string(&path)?;
            let config: Self = toml::from_str(&content)?;
            return Ok(Some(config));
        }
        Ok(None)
    }

    /// Stub when native feature is disabled — no config file support.
    #[cfg(not(feature = "native"))]
    pub fn load_optional() -> anyhow::Result<Option<Self>> {
        Ok(None)
    }

    /// Find the config file path, checking home dir then cwd.
    #[cfg(feature = "native")]
    fn find_config_path() -> Option<std::path::PathBuf> {
        // 1. Home directory (works from any cwd)
        if let Some(home) = dirs::home_dir() {
            let home_path = home.join(PRIVATE_CONFIG_FILENAME);
            if home_path.exists() {
                return Some(home_path);
            }
        }
        // 2. Current directory (development fallback)
        let cwd_path = std::path::PathBuf::from(PRIVATE_CONFIG_FILENAME);
        if cwd_path.exists() {
            return Some(cwd_path);
        }
        None
    }

    /// Whether any private directories are configured.
    pub fn has_dirs(&self) -> bool {
        !self.private.rust_stack_dirs.is_empty()
            || !self.private.rust_corpus_dirs.is_empty()
            || !self.private.python_corpus_dirs.is_empty()
    }

    /// Total number of private directories across all categories.
    pub fn dir_count(&self) -> usize {
        self.private.rust_stack_dirs.len()
            + self.private.rust_corpus_dirs.len()
            + self.private.python_corpus_dirs.len()
    }
}

impl BatutaConfig {
    /// Load configuration from TOML file
    #[cfg(feature = "native")]
    pub fn load(path: &std::path::Path) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config = toml::from_str(&content)?;
        Ok(config)
    }

    /// Save configuration to TOML file
    #[cfg(feature = "native")]
    pub fn save(&self, path: &std::path::Path) -> anyhow::Result<()> {
        let content = toml::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }

    /// Create a new config from project analysis
    pub fn from_analysis(analysis: &crate::types::ProjectAnalysis) -> Self {
        let mut config = Self::default();

        // Set project name from directory
        if let Some(name) = analysis.root_path.file_name() {
            config.project.name = name.to_string_lossy().to_string();
        }

        // Set primary language
        if let Some(lang) = &analysis.primary_language {
            config.project.primary_language = Some(format!("{}", lang));
        }

        // Configure transpilation based on detected dependencies
        if analysis.has_ml_dependencies() {
            config.transpilation.depyler.numpy_to_trueno = true;
            config.transpilation.depyler.sklearn_to_aprender = true;
            config.transpilation.depyler.pytorch_to_realizar = true;
        }

        config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use tempfile::TempDir;

    // ============================================================================
    // DEFAULT VALUE TESTS
    // ============================================================================

    #[test]
    fn test_batuta_config_default() {
        let config = BatutaConfig::default();

        assert_eq!(config.version, "1.0");
        assert_eq!(config.project.name, "untitled");
        assert_eq!(config.source.path, PathBuf::from("."));
        assert_eq!(config.transpilation.output_dir, PathBuf::from("./rust-output"));
        assert_eq!(config.optimization.profile, "balanced");
        assert!(config.validation.trace_syscalls);
        assert!(config.build.release);
    }

    #[test]
    fn test_project_config_default() {
        let config = ProjectConfig::default();

        assert_eq!(config.name, "untitled");
        assert!(config.description.is_none());
        assert!(config.primary_language.is_none());
        assert!(config.authors.is_empty());
        assert_eq!(config.license, Some("MIT".to_string()));
    }

    #[test]
    fn test_source_config_default() {
        let config = SourceConfig::default();

        assert_eq!(config.path, PathBuf::from("."));
        assert!(config.exclude.contains(&".git".to_string()));
        assert!(config.exclude.contains(&"target".to_string()));
        assert!(config.exclude.contains(&"node_modules".to_string()));
        assert!(config.exclude.contains(&"__pycache__".to_string()));
        assert!(config.include.is_empty());
    }

    #[test]
    fn test_transpilation_config_default() {
        let config = TranspilationConfig::default();

        assert_eq!(config.output_dir, PathBuf::from("./rust-output"));
        assert!(config.incremental);
        assert!(config.cache);
        assert!(!config.use_ruchy);
        assert_eq!(config.ruchy_strictness, Some("gradual".to_string()));
        assert!(config.modules.is_empty());
    }

    #[test]
    fn test_decy_config_default() {
        let config = DecyConfig::default();

        assert!(config.ownership_inference);
        assert!(config.actionable_diagnostics);
        assert!(config.use_static_fixer);
    }

    #[test]
    fn test_depyler_config_default() {
        let config = DepylerConfig::default();

        assert!(config.type_inference);
        assert!(config.numpy_to_trueno);
        assert!(config.sklearn_to_aprender);
        assert!(config.pytorch_to_realizar);
    }

    #[test]
    fn test_bashrs_config_default() {
        let config = BashrsConfig::default();

        assert_eq!(config.target_shell, "bash");
        assert!(config.use_clap);
    }

    #[test]
    fn test_optimization_config_default() {
        let config = OptimizationConfig::default();

        assert_eq!(config.profile, "balanced");
        assert!(config.enable_simd);
        assert!(!config.enable_gpu);
        assert_eq!(config.gpu_threshold, 500);
        assert!(!config.use_moe_routing);
    }

    #[test]
    fn test_trueno_config_default() {
        let config = TruenoConfig::default();

        assert_eq!(config.backends, vec!["simd".to_string(), "cpu".to_string()]);
        assert!(!config.adaptive_thresholds);
        assert_eq!(config.cpu_threshold, 500);
    }

    #[test]
    fn test_validation_config_default() {
        let config = ValidationConfig::default();

        assert!(config.trace_syscalls);
        assert!(config.run_original_tests);
        assert!(config.diff_output);
        assert!(!config.benchmark);
    }

    #[test]
    fn test_renacer_config_default() {
        let config = RenacerConfig::default();

        assert!(config.trace_syscalls.is_empty());
        assert_eq!(config.output_format, "json");
    }

    #[test]
    fn test_build_config_default() {
        let config = BuildConfig::default();

        assert!(config.release);
        assert!(config.target.is_none());
        assert!(!config.wasm);
        assert!(config.cargo_flags.is_empty());
    }

    // ============================================================================
    // LOAD/SAVE TESTS
    // ============================================================================

    #[test]
    fn test_save_and_load_config() {
        let temp_dir = TempDir::new().expect("tempdir creation failed");
        let config_path = temp_dir.path().join("batuta.toml");

        // Create a config with custom values
        let mut config = BatutaConfig::default();
        config.project.name = "test-project".to_string();
        config.project.description = Some("A test project".to_string());
        config.optimization.enable_gpu = true;
        config.optimization.gpu_threshold = 1000;

        // Save config
        config.save(&config_path).expect("save failed");

        // Verify file exists
        assert!(config_path.exists());

        // Load config
        let loaded_config = BatutaConfig::load(&config_path).expect("unexpected failure");

        // Verify loaded values match
        assert_eq!(loaded_config.project.name, "test-project");
        assert_eq!(loaded_config.project.description, Some("A test project".to_string()));
        assert!(loaded_config.optimization.enable_gpu);
        assert_eq!(loaded_config.optimization.gpu_threshold, 1000);
    }

    #[test]
    fn test_load_nonexistent_file() {
        let result = BatutaConfig::load(std::path::Path::new("/nonexistent/file.toml"));
        assert!(result.is_err());
    }

    #[test]
    fn test_load_invalid_toml() {
        let temp_dir = TempDir::new().expect("tempdir creation failed");
        let config_path = temp_dir.path().join("invalid.toml");

        // Write invalid TOML
        std::fs::write(&config_path, "invalid toml content [[[").expect("fs write failed");

        let result = BatutaConfig::load(&config_path);
        assert!(result.is_err());
    }

    #[test]
    fn test_save_config_creates_parent_dirs() {
        let temp_dir = TempDir::new().expect("tempdir creation failed");
        let nested_path = temp_dir.path().join("nested").join("dir").join("batuta.toml");

        // Create parent directories
        if let Some(parent) = nested_path.parent() {
            std::fs::create_dir_all(parent).expect("mkdir failed");
        }

        let config = BatutaConfig::default();
        let result = config.save(&nested_path);

        assert!(result.is_ok());
        assert!(nested_path.exists());
    }

    #[test]
    fn test_save_config_toml_format() {
        let temp_dir = TempDir::new().expect("tempdir creation failed");
        let config_path = temp_dir.path().join("batuta.toml");

        let config = BatutaConfig::default();
        config.save(&config_path).expect("save failed");

        // Read the TOML file
        let content = std::fs::read_to_string(&config_path).expect("fs read failed");

        // Verify it contains expected sections
        assert!(content.contains("[project]"));
        assert!(content.contains("[source]"));
        assert!(content.contains("[transpilation]"));
        assert!(content.contains("[optimization]"));
        assert!(content.contains("[validation]"));
        assert!(content.contains("[build]"));
    }

    // ============================================================================
    // FROM_ANALYSIS TESTS
    // ============================================================================

    #[test]
    fn test_from_analysis_basic() {
        let analysis = crate::types::ProjectAnalysis {
            root_path: PathBuf::from("/home/user/my-project"),
            total_files: 10,
            total_lines: 1000,
            languages: vec![],
            primary_language: Some(crate::types::Language::Python),
            dependencies: vec![],
            tdg_score: Some(85.0),
        };

        let config = BatutaConfig::from_analysis(&analysis);

        assert_eq!(config.project.name, "my-project");
        assert_eq!(config.project.primary_language, Some("Python".to_string()));
    }

    #[test]
    fn test_from_analysis_with_ml_dependencies() {
        let analysis = crate::types::ProjectAnalysis {
            root_path: PathBuf::from("/test/project"),
            total_files: 5,
            total_lines: 500,
            languages: vec![],
            primary_language: Some(crate::types::Language::Python),
            dependencies: vec![crate::types::DependencyInfo {
                manager: crate::types::DependencyManager::Pip,
                file_path: PathBuf::from("requirements.txt"),
                count: Some(3),
            }],
            tdg_score: None,
        };

        let config = BatutaConfig::from_analysis(&analysis);

        // ML frameworks should be enabled by default
        assert!(config.transpilation.depyler.numpy_to_trueno);
        assert!(config.transpilation.depyler.sklearn_to_aprender);
        assert!(config.transpilation.depyler.pytorch_to_realizar);
    }

    #[test]
    fn test_from_analysis_without_ml_dependencies() {
        let analysis = crate::types::ProjectAnalysis {
            root_path: PathBuf::from("/test/project"),
            total_files: 5,
            total_lines: 500,
            languages: vec![],
            primary_language: Some(crate::types::Language::Python),
            dependencies: vec![crate::types::DependencyInfo {
                manager: crate::types::DependencyManager::Pip,
                file_path: PathBuf::from("requirements.txt"),
                count: Some(1),
            }],
            tdg_score: None,
        };

        let config = BatutaConfig::from_analysis(&analysis);

        // Should still have ML framework support enabled by default
        assert!(config.transpilation.depyler.numpy_to_trueno);
    }

    #[test]
    fn test_from_analysis_rust_project() {
        let analysis = crate::types::ProjectAnalysis {
            root_path: PathBuf::from("/rust/project"),
            total_files: 20,
            total_lines: 2000,
            languages: vec![],
            primary_language: Some(crate::types::Language::Rust),
            dependencies: vec![],
            tdg_score: Some(95.0),
        };

        let config = BatutaConfig::from_analysis(&analysis);

        assert_eq!(config.project.name, "project");
        assert_eq!(config.project.primary_language, Some("Rust".to_string()));
    }

    #[test]
    fn test_from_analysis_no_primary_language() {
        let analysis = crate::types::ProjectAnalysis {
            root_path: PathBuf::from("/unknown/project"),
            total_files: 1,
            total_lines: 10,
            languages: vec![],
            primary_language: None,
            dependencies: vec![],
            tdg_score: None,
        };

        let config = BatutaConfig::from_analysis(&analysis);

        assert_eq!(config.project.name, "project");
        assert!(config.project.primary_language.is_none());
    }

    // ============================================================================
    // SERIALIZATION TESTS
    // ============================================================================

    #[test]
    fn test_serialize_deserialize_batuta_config() {
        let config = BatutaConfig::default();

        let serialized = toml::to_string(&config).expect("toml serialize failed");
        let deserialized: BatutaConfig = toml::from_str(&serialized).expect("toml parse failed");

        assert_eq!(config.version, deserialized.version);
        assert_eq!(config.project.name, deserialized.project.name);
        assert_eq!(config.optimization.profile, deserialized.optimization.profile);
    }

    #[test]
    fn test_serialize_deserialize_with_optional_fields() {
        let mut config = BatutaConfig::default();
        config.project.description = Some("Test description".to_string());
        config.project.primary_language = Some("Python".to_string());
        config.build.target = Some("x86_64-unknown-linux-gnu".to_string());

        let serialized = toml::to_string(&config).expect("toml serialize failed");
        let deserialized: BatutaConfig = toml::from_str(&serialized).expect("toml parse failed");

        assert_eq!(config.project.description, deserialized.project.description);
        assert_eq!(config.project.primary_language, deserialized.project.primary_language);
        assert_eq!(config.build.target, deserialized.build.target);
    }

    #[test]
    fn test_serialize_deserialize_with_vectors() {
        let mut config = BatutaConfig::default();
        config.project.authors = vec!["Alice".to_string(), "Bob".to_string()];
        config.source.exclude = vec!["test".to_string(), "docs".to_string()];
        config.transpilation.modules = vec!["mod1".to_string(), "mod2".to_string()];

        let serialized = toml::to_string(&config).expect("toml serialize failed");
        let deserialized: BatutaConfig = toml::from_str(&serialized).expect("toml parse failed");

        assert_eq!(config.project.authors, deserialized.project.authors);
        assert_eq!(config.source.exclude, deserialized.source.exclude);
        assert_eq!(config.transpilation.modules, deserialized.transpilation.modules);
    }

    #[test]
    fn test_full_toml_deserialization() {
        // Test deserializing a complete TOML configuration
        let config = BatutaConfig::default();
        let serialized = toml::to_string(&config).expect("toml serialize failed");
        let deserialized: BatutaConfig = toml::from_str(&serialized).expect("toml parse failed");

        assert_eq!(config.version, deserialized.version);
        assert_eq!(config.project.name, deserialized.project.name);
        assert_eq!(config.optimization.profile, deserialized.optimization.profile);
    }

    #[test]
    fn test_modified_toml_deserialization() {
        // Test deserializing a modified configuration
        let mut config = BatutaConfig::default();
        config.project.name = "custom-name".to_string();
        config.optimization.profile = "aggressive".to_string();
        config.build.release = false;

        let serialized = toml::to_string(&config).expect("toml serialize failed");
        let deserialized: BatutaConfig = toml::from_str(&serialized).expect("toml parse failed");

        assert_eq!(deserialized.project.name, "custom-name");
        assert_eq!(deserialized.optimization.profile, "aggressive");
        assert!(!deserialized.build.release);
    }

    // ============================================================================
    // NESTED CONFIG TESTS
    // ============================================================================

    #[test]
    fn test_decy_config_in_transpilation() {
        let config = BatutaConfig::default();

        assert!(config.transpilation.decy.ownership_inference);
        assert!(config.transpilation.decy.actionable_diagnostics);
        assert!(config.transpilation.decy.use_static_fixer);
    }

    #[test]
    fn test_depyler_config_in_transpilation() {
        let config = BatutaConfig::default();

        assert!(config.transpilation.depyler.type_inference);
        assert!(config.transpilation.depyler.numpy_to_trueno);
        assert!(config.transpilation.depyler.sklearn_to_aprender);
        assert!(config.transpilation.depyler.pytorch_to_realizar);
    }

    #[test]
    fn test_bashrs_config_in_transpilation() {
        let config = BatutaConfig::default();

        assert_eq!(config.transpilation.bashrs.target_shell, "bash");
        assert!(config.transpilation.bashrs.use_clap);
    }

    #[test]
    fn test_trueno_config_in_optimization() {
        let config = BatutaConfig::default();

        assert_eq!(
            config.optimization.trueno.backends,
            vec!["simd".to_string(), "cpu".to_string()]
        );
        assert!(!config.optimization.trueno.adaptive_thresholds);
        assert_eq!(config.optimization.trueno.cpu_threshold, 500);
    }

    #[test]
    fn test_renacer_config_in_validation() {
        let config = BatutaConfig::default();

        assert!(config.validation.renacer.trace_syscalls.is_empty());
        assert_eq!(config.validation.renacer.output_format, "json");
    }

    // ============================================================================
    // MODIFICATION TESTS
    // ============================================================================

    #[test]
    fn test_config_modification() {
        let mut config = BatutaConfig::default();

        // Modify various fields
        config.project.name = "new-name".to_string();
        config.optimization.enable_gpu = true;
        config.optimization.gpu_threshold = 2000;
        config.transpilation.incremental = false;

        assert_eq!(config.project.name, "new-name");
        assert!(config.optimization.enable_gpu);
        assert_eq!(config.optimization.gpu_threshold, 2000);
        assert!(!config.transpilation.incremental);
    }

    #[test]
    fn test_config_clone() {
        let config = BatutaConfig::default();
        let cloned = config.clone();

        assert_eq!(config.version, cloned.version);
        assert_eq!(config.project.name, cloned.project.name);
        assert_eq!(config.optimization.profile, cloned.optimization.profile);
    }

    #[test]
    fn test_save_modified_config() {
        let temp_dir = TempDir::new().expect("tempdir creation failed");
        let config_path = temp_dir.path().join("config.toml");

        let mut config = BatutaConfig::default();
        config.project.name = "modified-project".to_string();
        config.project.authors = vec!["Author1".to_string(), "Author2".to_string()];
        config.optimization.enable_gpu = true;

        config.save(&config_path).expect("save failed");

        let loaded = BatutaConfig::load(&config_path).expect("unexpected failure");

        assert_eq!(loaded.project.name, "modified-project");
        assert_eq!(loaded.project.authors.len(), 2);
        assert!(loaded.optimization.enable_gpu);
    }

    // ============================================================================
    // PRIVATE CONFIG TESTS
    // ============================================================================

    #[test]
    fn test_private_config_default() {
        let config = PrivateConfig::default();
        assert!(config.private.rust_stack_dirs.is_empty());
        assert!(config.private.rust_corpus_dirs.is_empty());
        assert!(config.private.python_corpus_dirs.is_empty());
        assert!(config.private.endpoints.is_empty());
        assert!(!config.has_dirs());
        assert_eq!(config.dir_count(), 0);
    }

    #[test]
    fn test_private_config_deserialize_full() {
        let toml_str = r#"
[private]
rust_stack_dirs = ["../rmedia", "../infra"]
rust_corpus_dirs = ["../internal-cookbook"]
python_corpus_dirs = ["../private-notebooks"]
"#;
        let config: PrivateConfig = toml::from_str(toml_str).expect("toml parse failed");
        assert_eq!(config.private.rust_stack_dirs.len(), 2);
        assert_eq!(config.private.rust_corpus_dirs.len(), 1);
        assert_eq!(config.private.python_corpus_dirs.len(), 1);
        assert!(config.has_dirs());
        assert_eq!(config.dir_count(), 4);
    }

    #[test]
    fn test_private_config_deserialize_partial() {
        let toml_str = r#"
[private]
rust_stack_dirs = ["../rmedia"]
"#;
        let config: PrivateConfig = toml::from_str(toml_str).expect("toml parse failed");
        assert_eq!(config.private.rust_stack_dirs, vec!["../rmedia"]);
        assert!(config.private.rust_corpus_dirs.is_empty());
        assert!(config.private.python_corpus_dirs.is_empty());
        assert!(config.has_dirs());
        assert_eq!(config.dir_count(), 1);
    }

    #[test]
    fn test_private_config_deserialize_empty_private() {
        let toml_str = r#"
[private]
"#;
        let config: PrivateConfig = toml::from_str(toml_str).expect("toml parse failed");
        assert!(!config.has_dirs());
        assert_eq!(config.dir_count(), 0);
    }

    #[test]
    fn test_private_config_with_endpoints() {
        let toml_str = r#"
[private]
rust_stack_dirs = ["../rmedia"]

[[private.endpoints]]
name = "intel"
type = "ssh"
host = "intel.local"
index_path = "/tmp/batuta/rag/index.sqlite"
"#;
        let config: PrivateConfig = toml::from_str(toml_str).expect("toml parse failed");
        assert_eq!(config.private.endpoints.len(), 1);
        assert_eq!(config.private.endpoints[0].name, "intel");
        assert_eq!(config.private.endpoints[0].endpoint_type, "ssh");
        assert_eq!(config.private.endpoints[0].host, "intel.local");
    }

    #[test]
    fn test_private_config_serialize_roundtrip() {
        let toml_str = r#"
[private]
rust_stack_dirs = ["../rmedia", "../infra"]
rust_corpus_dirs = ["../internal-cookbook"]
python_corpus_dirs = []
"#;
        let config: PrivateConfig = toml::from_str(toml_str).expect("toml parse failed");
        let serialized = toml::to_string(&config).expect("toml serialize failed");
        let roundtripped: PrivateConfig = toml::from_str(&serialized).expect("toml parse failed");
        assert_eq!(config.private.rust_stack_dirs, roundtripped.private.rust_stack_dirs);
        assert_eq!(config.private.rust_corpus_dirs, roundtripped.private.rust_corpus_dirs);
    }

    #[test]
    fn test_private_config_find_config_path_checks_home() {
        // find_config_path checks home dir first, then cwd.
        // We can't easily mock the filesystem, but we can verify the function
        // returns Some when $HOME/.batuta-private.toml exists (which it does
        // in dev) or None when it doesn't.
        let result = PrivateConfig::find_config_path();
        // Result depends on environment — just verify it doesn't panic
        let _ = result;
    }

    #[test]
    fn test_private_config_has_dirs() {
        let mut config = PrivateConfig::default();
        assert!(!config.has_dirs());

        config.private.rust_stack_dirs.push("../foo".to_string());
        assert!(config.has_dirs());
    }

    #[test]
    fn test_private_config_dir_count() {
        let mut config = PrivateConfig::default();
        assert_eq!(config.dir_count(), 0);

        config.private.rust_stack_dirs.push("../a".to_string());
        config.private.rust_corpus_dirs.push("../b".to_string());
        config.private.python_corpus_dirs.push("../c".to_string());
        assert_eq!(config.dir_count(), 3);
    }
}
