//! WASM Runner Configuration
//!
//! Configuration types for the WASM development server.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Optimization level for WASM builds
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum OptLevel {
    /// Debug build (no optimization)
    #[default]
    Debug,
    /// Standard release optimization
    Release,
    /// Optimize for size
    Size,
    /// Minimize size (aggressive)
    MinSize,
}

impl OptLevel {
    /// Get the optimization flag for cargo
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            OptLevel::Debug => "0",
            OptLevel::Release => "3",
            OptLevel::Size => "s",
            OptLevel::MinSize => "z",
        }
    }

    /// Check if this is a release build
    #[must_use]
    pub fn is_release(&self) -> bool {
        !matches!(self, OptLevel::Debug)
    }
}

/// Configuration for the WASM runner
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmRunnerConfig {
    /// HTTP server port
    pub http_port: u16,
    /// WebSocket server port for hot reload
    pub ws_port: u16,
    /// Enable hot reload
    pub hot_reload: bool,
    /// Preserve application state during reload
    pub preserve_state: bool,
    /// Generate source maps
    pub source_maps: bool,
    /// Optimization level
    pub opt_level: OptLevel,
    /// Watch patterns (glob)
    pub watch_patterns: Vec<String>,
    /// Output directory for WASM
    pub output_dir: PathBuf,
    /// Static files directory
    pub static_dir: Option<PathBuf>,
    /// WASM file name
    pub wasm_filename: String,
}

impl Default for WasmRunnerConfig {
    fn default() -> Self {
        Self {
            http_port: super::DEFAULT_HTTP_PORT,
            ws_port: super::DEFAULT_WS_PORT,
            hot_reload: true,
            preserve_state: true,
            source_maps: true,
            opt_level: OptLevel::Debug,
            watch_patterns: vec!["src/**/*.rs".to_string(), "Cargo.toml".to_string()],
            output_dir: PathBuf::from("target/wasm32-unknown-unknown/debug"),
            static_dir: None,
            wasm_filename: "app.wasm".to_string(),
        }
    }
}

impl WasmRunnerConfig {
    /// Create a new builder
    #[must_use]
    pub fn builder() -> WasmRunnerConfigBuilder {
        WasmRunnerConfigBuilder::default()
    }

    /// Get the full WASM file path
    #[must_use]
    pub fn wasm_path(&self) -> PathBuf {
        self.output_dir.join(&self.wasm_filename)
    }
}

/// Builder for `WasmRunnerConfig`
#[derive(Debug, Clone, Default)]
pub struct WasmRunnerConfigBuilder {
    config: WasmRunnerConfig,
}

impl WasmRunnerConfigBuilder {
    /// Set HTTP server port
    #[must_use]
    pub fn http_port(mut self, port: u16) -> Self {
        self.config.http_port = port;
        self
    }

    /// Set WebSocket server port
    #[must_use]
    pub fn ws_port(mut self, port: u16) -> Self {
        self.config.ws_port = port;
        self
    }

    /// Enable/disable hot reload
    #[must_use]
    pub fn hot_reload(mut self, enabled: bool) -> Self {
        self.config.hot_reload = enabled;
        self
    }

    /// Enable/disable state preservation
    #[must_use]
    pub fn preserve_state(mut self, enabled: bool) -> Self {
        self.config.preserve_state = enabled;
        self
    }

    /// Enable/disable source maps
    #[must_use]
    pub fn source_maps(mut self, enabled: bool) -> Self {
        self.config.source_maps = enabled;
        self
    }

    /// Set optimization level
    #[must_use]
    pub fn opt_level(mut self, level: OptLevel) -> Self {
        self.config.opt_level = level;
        self
    }

    /// Set watch patterns
    #[must_use]
    pub fn watch_patterns(mut self, patterns: Vec<String>) -> Self {
        self.config.watch_patterns = patterns;
        self
    }

    /// Set output directory
    #[must_use]
    pub fn output_dir(mut self, dir: PathBuf) -> Self {
        self.config.output_dir = dir;
        self
    }

    /// Set static files directory
    #[must_use]
    pub fn static_dir(mut self, dir: PathBuf) -> Self {
        self.config.static_dir = Some(dir);
        self
    }

    /// Set WASM filename
    #[must_use]
    pub fn wasm_filename(mut self, name: impl Into<String>) -> Self {
        self.config.wasm_filename = name.into();
        self
    }

    /// Build the configuration
    #[must_use]
    pub fn build(self) -> WasmRunnerConfig {
        self.config
    }
}

/// General runner configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunnerConfig {
    /// Project root directory
    pub project_root: PathBuf,
    /// Target triple
    pub target: String,
    /// Package name (optional)
    pub package: Option<String>,
    /// Features to enable
    pub features: Vec<String>,
    /// Environment variables
    pub env: std::collections::HashMap<String, String>,
}

impl Default for RunnerConfig {
    fn default() -> Self {
        Self {
            project_root: PathBuf::from("."),
            target: "wasm32-unknown-unknown".to_string(),
            package: None,
            features: Vec::new(),
            env: std::collections::HashMap::new(),
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn test_opt_level_as_str() {
        assert_eq!(OptLevel::Debug.as_str(), "0");
        assert_eq!(OptLevel::Release.as_str(), "3");
        assert_eq!(OptLevel::Size.as_str(), "s");
        assert_eq!(OptLevel::MinSize.as_str(), "z");
    }

    #[test]
    fn test_opt_level_is_release() {
        assert!(!OptLevel::Debug.is_release());
        assert!(OptLevel::Release.is_release());
        assert!(OptLevel::Size.is_release());
        assert!(OptLevel::MinSize.is_release());
    }

    #[test]
    fn test_wasm_config_default() {
        let config = WasmRunnerConfig::default();
        assert!(config.hot_reload);
        assert!(config.preserve_state);
        assert!(config.source_maps);
        assert_eq!(config.opt_level, OptLevel::Debug);
    }

    #[test]
    fn test_wasm_config_builder_chain() {
        let config = WasmRunnerConfig::builder()
            .http_port(9000)
            .ws_port(9001)
            .hot_reload(false)
            .opt_level(OptLevel::Release)
            .build();

        assert_eq!(config.http_port, 9000);
        assert_eq!(config.ws_port, 9001);
        assert!(!config.hot_reload);
        assert_eq!(config.opt_level, OptLevel::Release);
    }

    #[test]
    fn test_wasm_path() {
        let config = WasmRunnerConfig::builder()
            .output_dir(PathBuf::from("target/wasm"))
            .wasm_filename("myapp.wasm")
            .build();

        assert_eq!(config.wasm_path(), PathBuf::from("target/wasm/myapp.wasm"));
    }

    #[test]
    fn test_runner_config_default() {
        let config = RunnerConfig::default();
        assert_eq!(config.target, "wasm32-unknown-unknown");
        assert!(config.features.is_empty());
    }

    #[test]
    fn test_opt_level_default() {
        let opt = OptLevel::default();
        assert_eq!(opt, OptLevel::Debug);
        assert!(!opt.is_release());
    }

    #[test]
    fn test_wasm_config_builder_preserve_state() {
        let config = WasmRunnerConfig::builder().preserve_state(false).build();
        assert!(!config.preserve_state);

        let config2 = WasmRunnerConfig::builder().preserve_state(true).build();
        assert!(config2.preserve_state);
    }

    #[test]
    fn test_wasm_config_builder_source_maps() {
        let config = WasmRunnerConfig::builder().source_maps(false).build();
        assert!(!config.source_maps);

        let config2 = WasmRunnerConfig::builder().source_maps(true).build();
        assert!(config2.source_maps);
    }

    #[test]
    fn test_wasm_config_builder_watch_patterns() {
        let patterns = vec![
            "src/**/*.rs".to_string(),
            "tests/**/*.rs".to_string(),
            "Cargo.toml".to_string(),
        ];
        let config = WasmRunnerConfig::builder()
            .watch_patterns(patterns.clone())
            .build();

        assert_eq!(config.watch_patterns.len(), 3);
        assert_eq!(config.watch_patterns, patterns);
    }

    #[test]
    fn test_wasm_config_builder_static_dir() {
        let config = WasmRunnerConfig::builder()
            .static_dir(PathBuf::from("public"))
            .build();

        assert_eq!(config.static_dir, Some(PathBuf::from("public")));
    }

    #[test]
    fn test_wasm_config_default_static_dir_none() {
        let config = WasmRunnerConfig::default();
        assert!(config.static_dir.is_none());
    }

    #[test]
    fn test_runner_config_with_all_fields() {
        let mut config = RunnerConfig::default();
        config.project_root = PathBuf::from("/home/user/project");
        config.target = "wasm32-wasi".to_string();
        config.package = Some("my-wasm-lib".to_string());
        config.features = vec!["feature1".to_string(), "feature2".to_string()];
        config
            .env
            .insert("RUST_LOG".to_string(), "debug".to_string());
        config
            .env
            .insert("CARGO_TERM_COLOR".to_string(), "always".to_string());

        assert_eq!(config.project_root, PathBuf::from("/home/user/project"));
        assert_eq!(config.target, "wasm32-wasi");
        assert_eq!(config.package, Some("my-wasm-lib".to_string()));
        assert_eq!(config.features.len(), 2);
        assert_eq!(config.env.len(), 2);
        assert_eq!(config.env.get("RUST_LOG"), Some(&"debug".to_string()));
    }

    #[test]
    fn test_wasm_config_builder_all_options() {
        let config = WasmRunnerConfig::builder()
            .http_port(3000)
            .ws_port(3001)
            .hot_reload(true)
            .preserve_state(false)
            .source_maps(true)
            .opt_level(OptLevel::MinSize)
            .watch_patterns(vec!["*.rs".to_string()])
            .output_dir(PathBuf::from("dist"))
            .static_dir(PathBuf::from("assets"))
            .wasm_filename("game.wasm")
            .build();

        assert_eq!(config.http_port, 3000);
        assert_eq!(config.ws_port, 3001);
        assert!(config.hot_reload);
        assert!(!config.preserve_state);
        assert!(config.source_maps);
        assert_eq!(config.opt_level, OptLevel::MinSize);
        assert_eq!(config.watch_patterns, vec!["*.rs".to_string()]);
        assert_eq!(config.output_dir, PathBuf::from("dist"));
        assert_eq!(config.static_dir, Some(PathBuf::from("assets")));
        assert_eq!(config.wasm_filename, "game.wasm");
    }

    #[test]
    fn test_wasm_path_with_nested_output_dir() {
        let config = WasmRunnerConfig::builder()
            .output_dir(PathBuf::from("target/wasm32-unknown-unknown/release"))
            .wasm_filename("my_game.wasm")
            .build();

        assert_eq!(
            config.wasm_path(),
            PathBuf::from("target/wasm32-unknown-unknown/release/my_game.wasm")
        );
    }

    #[test]
    fn test_opt_level_serialization() {
        for opt in [
            OptLevel::Debug,
            OptLevel::Release,
            OptLevel::Size,
            OptLevel::MinSize,
        ] {
            let json = serde_json::to_string(&opt).unwrap();
            let deserialized: OptLevel = serde_json::from_str(&json).unwrap();
            assert_eq!(opt, deserialized);
        }
    }

    #[test]
    fn test_wasm_runner_config_serialization() {
        let config = WasmRunnerConfig::builder()
            .http_port(8000)
            .ws_port(8001)
            .hot_reload(true)
            .opt_level(OptLevel::Release)
            .wasm_filename("app.wasm")
            .build();

        let json = serde_json::to_string(&config).unwrap();
        let deserialized: WasmRunnerConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.http_port, 8000);
        assert_eq!(deserialized.ws_port, 8001);
        assert!(deserialized.hot_reload);
        assert_eq!(deserialized.opt_level, OptLevel::Release);
        assert_eq!(deserialized.wasm_filename, "app.wasm");
    }

    #[test]
    fn test_runner_config_serialization() {
        let mut config = RunnerConfig::default();
        config.package = Some("test-package".to_string());
        config.features = vec!["web".to_string()];

        let json = serde_json::to_string(&config).unwrap();
        let deserialized: RunnerConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.target, "wasm32-unknown-unknown");
        assert_eq!(deserialized.package, Some("test-package".to_string()));
        assert_eq!(deserialized.features, vec!["web".to_string()]);
    }

    #[test]
    fn test_wasm_config_default_watch_patterns() {
        let config = WasmRunnerConfig::default();
        assert!(config.watch_patterns.contains(&"src/**/*.rs".to_string()));
        assert!(config.watch_patterns.contains(&"Cargo.toml".to_string()));
    }

    #[test]
    fn test_wasm_config_default_output_dir() {
        let config = WasmRunnerConfig::default();
        assert_eq!(
            config.output_dir,
            PathBuf::from("target/wasm32-unknown-unknown/debug")
        );
    }

    #[test]
    fn test_wasm_config_default_wasm_filename() {
        let config = WasmRunnerConfig::default();
        assert_eq!(config.wasm_filename, "app.wasm");
    }

    #[test]
    fn test_wasm_config_builder_default_values() {
        // Test that builder starts with defaults
        let builder = WasmRunnerConfigBuilder::default();
        let config = builder.build();

        // Should match WasmRunnerConfig::default()
        let default_config = WasmRunnerConfig::default();
        assert_eq!(config.http_port, default_config.http_port);
        assert_eq!(config.ws_port, default_config.ws_port);
        assert_eq!(config.hot_reload, default_config.hot_reload);
        assert_eq!(config.preserve_state, default_config.preserve_state);
        assert_eq!(config.source_maps, default_config.source_maps);
        assert_eq!(config.opt_level, default_config.opt_level);
    }

    #[test]
    fn test_runner_config_empty_env() {
        let config = RunnerConfig::default();
        assert!(config.env.is_empty());
    }

    #[test]
    fn test_runner_config_project_root_default() {
        let config = RunnerConfig::default();
        assert_eq!(config.project_root, PathBuf::from("."));
    }

    #[test]
    fn test_wasm_config_builder_chaining() {
        // Test that builder methods can be chained in any order
        let config1 = WasmRunnerConfig::builder()
            .http_port(9000)
            .opt_level(OptLevel::Size)
            .hot_reload(false)
            .build();

        let config2 = WasmRunnerConfig::builder()
            .hot_reload(false)
            .http_port(9000)
            .opt_level(OptLevel::Size)
            .build();

        assert_eq!(config1.http_port, config2.http_port);
        assert_eq!(config1.opt_level, config2.opt_level);
        assert_eq!(config1.hot_reload, config2.hot_reload);
    }

    #[test]
    fn test_opt_level_all_variants_as_str() {
        // Comprehensive test for all variants
        let variants = [
            (OptLevel::Debug, "0"),
            (OptLevel::Release, "3"),
            (OptLevel::Size, "s"),
            (OptLevel::MinSize, "z"),
        ];

        for (opt, expected) in variants {
            assert_eq!(opt.as_str(), expected);
        }
    }

    #[test]
    fn test_opt_level_is_release_all_variants() {
        // Debug is not release
        assert!(!OptLevel::Debug.is_release());

        // All others are release
        assert!(OptLevel::Release.is_release());
        assert!(OptLevel::Size.is_release());
        assert!(OptLevel::MinSize.is_release());
    }
}
