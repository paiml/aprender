//! Dynamic Stack Discovery for Sovereign AI Stack components.
//!
//! Per the Batuta Stack Review, hardcoded component lists introduce
//! Muda of Processing (maintenance waste). This module uses dynamic
//! discovery to detect available stack components from Cargo.toml.

use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

use crate::error::{SimError, SimResult};

/// Semantic version representation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Version {
    /// Major version number.
    pub major: u32,
    /// Minor version number.
    pub minor: u32,
    /// Patch version number.
    pub patch: u32,
    /// Pre-release identifier (e.g., "alpha", "beta").
    pub pre_release: Option<String>,
}

impl Version {
    /// Parse a version string like "1.2.3" or "1.2.3-beta".
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        let s = s.trim();

        // Handle version requirements like "^1.0", ">=1.2", "~1.0"
        let s = s
            .trim_start_matches('^')
            .trim_start_matches('~')
            .trim_start_matches(">=")
            .trim_start_matches("<=")
            .trim_start_matches('>')
            .trim_start_matches('<')
            .trim_start_matches('=')
            .trim();

        // Split on hyphen for pre-release
        let (version_part, pre_release) = s
            .find('-')
            .map_or((s, None), |idx| (&s[..idx], Some(s[idx + 1..].to_string())));

        let parts: Vec<&str> = version_part.split('.').collect();
        if parts.is_empty() || parts.len() > 3 {
            return None;
        }

        let major = parts[0].parse().ok()?;
        let minor = parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(0);
        let patch = parts.get(2).and_then(|s| s.parse().ok()).unwrap_or(0);

        Some(Self {
            major,
            minor,
            patch,
            pre_release,
        })
    }

    /// Check if this version satisfies a minimum requirement.
    #[must_use]
    pub fn satisfies_minimum(&self, min: &Self) -> bool {
        if self.major != min.major {
            return self.major > min.major;
        }
        if self.minor != min.minor {
            return self.minor > min.minor;
        }
        self.patch >= min.patch
    }
}

impl std::fmt::Display for Version {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(ref pre) = self.pre_release {
            write!(f, "{}.{}.{}-{}", self.major, self.minor, self.patch, pre)
        } else {
            write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
        }
    }
}

/// Sovereign AI Stack components that simular can integrate with.
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq)]
pub enum StackComponent {
    /// trueno: SIMD-accelerated tensor operations.
    Trueno,
    /// trueno-db: Analytics database.
    TruenoDB,
    /// trueno-graph: Code analysis graphs.
    TruenoGraph,
    /// trueno-rag: RAG pipeline.
    TruenoRag,
    /// aprender: ML algorithms (regression, trees, GNN).
    Aprender,
    /// entrenar: Training (autograd, `LoRA`, quantization).
    Entrenar,
    /// realizar: Inference (GGUF serving, `SafeTensors`).
    Realizar,
    /// alimentar: Data loading (Parquet, Arrow).
    Alimentar,
    /// pacha: Registry (Ed25519 signatures, versioning).
    Pacha,
    /// renacer: Tracing (syscall trace, source correlation).
    Renacer,
}

impl StackComponent {
    /// Get the crate name for this component.
    #[must_use]
    pub const fn crate_name(&self) -> &'static str {
        match self {
            Self::Trueno => "trueno",
            Self::TruenoDB => "trueno-db",
            Self::TruenoGraph => "trueno-graph",
            Self::TruenoRag => "trueno-rag",
            Self::Aprender => "aprender",
            Self::Entrenar => "entrenar",
            Self::Realizar => "realizar",
            Self::Alimentar => "alimentar",
            Self::Pacha => "pacha",
            Self::Renacer => "renacer",
        }
    }

    /// Get all known stack components.
    #[must_use]
    pub const fn all() -> &'static [Self] {
        &[
            Self::Trueno,
            Self::TruenoDB,
            Self::TruenoGraph,
            Self::TruenoRag,
            Self::Aprender,
            Self::Entrenar,
            Self::Realizar,
            Self::Alimentar,
            Self::Pacha,
            Self::Renacer,
        ]
    }
}

impl std::fmt::Display for StackComponent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.crate_name())
    }
}

/// Dynamic discovery of Sovereign AI Stack components.
///
/// Eliminates hardcoded lists (Batuta Review §2.2) by parsing
/// Cargo.toml at runtime to detect available integrations.
#[derive(Debug, Clone, Default)]
pub struct StackDiscovery {
    /// Discovered stack crates and versions.
    components: HashMap<StackComponent, Version>,
}

impl StackDiscovery {
    /// Create an empty discovery instance.
    #[must_use]
    pub fn new() -> Self {
        Self {
            components: HashMap::new(),
        }
    }

    /// Discover available stack components from a Cargo.toml file.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or parsed.
    pub fn from_cargo_toml(path: &Path) -> SimResult<Self> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| SimError::config(format!("Failed to read Cargo.toml: {e}")))?;

        Self::from_toml_str(&content)
    }

    /// Discover available stack components from TOML content.
    ///
    /// # Errors
    ///
    /// Returns an error if the TOML cannot be parsed.
    pub fn from_toml_str(content: &str) -> SimResult<Self> {
        let manifest: CargoManifest = toml::from_str(content)
            .map_err(|e| SimError::config(format!("Failed to parse Cargo.toml: {e}")))?;

        let mut discovery = Self::new();

        // Parse dependencies section
        if let Some(deps) = manifest.dependencies {
            discovery.parse_dependencies(&deps);
        }

        // Parse dev-dependencies section
        if let Some(dev_deps) = manifest.dev_dependencies {
            discovery.parse_dependencies(&dev_deps);
        }

        Ok(discovery)
    }

    /// Parse a dependencies table for stack components.
    fn parse_dependencies(&mut self, deps: &HashMap<String, toml::Value>) {
        for (name, value) in deps {
            if let Some(component) = Self::parse_stack_component(name) {
                if let Some(version) = Self::extract_version(value) {
                    self.components.insert(component, version);
                }
            }
        }
    }

    /// Parse component name with fuzzy matching.
    ///
    /// Handles both hyphenated and underscored variants.
    #[must_use]
    pub fn parse_stack_component(name: &str) -> Option<StackComponent> {
        let normalized = name.to_lowercase().replace('_', "-");
        match normalized.as_str() {
            "trueno" => Some(StackComponent::Trueno),
            "trueno-db" => Some(StackComponent::TruenoDB),
            "trueno-graph" => Some(StackComponent::TruenoGraph),
            "trueno-rag" => Some(StackComponent::TruenoRag),
            "aprender" => Some(StackComponent::Aprender),
            "entrenar" => Some(StackComponent::Entrenar),
            "realizar" => Some(StackComponent::Realizar),
            "alimentar" => Some(StackComponent::Alimentar),
            "pacha" => Some(StackComponent::Pacha),
            "renacer" => Some(StackComponent::Renacer),
            _ => None,
        }
    }

    /// Extract version from a TOML dependency value.
    fn extract_version(value: &toml::Value) -> Option<Version> {
        match value {
            // Simple version string: dependency = "1.0"
            toml::Value::String(s) => Version::parse(s),
            // Table format: dependency = { version = "1.0", ... }
            toml::Value::Table(t) => t
                .get("version")
                .and_then(|v| v.as_str())
                .and_then(Version::parse),
            _ => None,
        }
    }

    /// Check if a component is available.
    #[must_use]
    pub fn has(&self, component: StackComponent) -> bool {
        self.components.contains_key(&component)
    }

    /// Get component version if available.
    #[must_use]
    pub fn version(&self, component: StackComponent) -> Option<&Version> {
        self.components.get(&component)
    }

    /// Get all discovered components.
    #[must_use]
    pub fn discovered(&self) -> &HashMap<StackComponent, Version> {
        &self.components
    }

    /// Get count of discovered components.
    #[must_use]
    pub fn count(&self) -> usize {
        self.components.len()
    }

    /// Check if no components were discovered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.components.is_empty()
    }

    /// Manually register a component (for testing or manual configuration).
    pub fn register(&mut self, component: StackComponent, version: Version) {
        self.components.insert(component, version);
    }

    /// Check version compatibility for a component.
    ///
    /// Returns `true` if the component is available and satisfies
    /// the minimum version requirement.
    #[must_use]
    pub fn check_version(&self, component: StackComponent, min_version: &Version) -> bool {
        self.version(component)
            .is_some_and(|v| v.satisfies_minimum(min_version))
    }

    /// Get a summary of available integrations.
    #[must_use]
    pub fn summary(&self) -> String {
        if self.is_empty() {
            return String::from("No Sovereign AI Stack components detected");
        }

        let mut lines = vec![format!("Detected {} stack components:", self.count())];

        for component in StackComponent::all() {
            if let Some(version) = self.version(*component) {
                lines.push(format!("  - {}: v{version}", component.crate_name()));
            }
        }

        lines.join("\n")
    }
}

/// Minimal Cargo.toml structure for parsing.
#[derive(Deserialize)]
struct CargoManifest {
    #[serde(default)]
    dependencies: Option<HashMap<String, toml::Value>>,
    #[serde(default, rename = "dev-dependencies")]
    dev_dependencies: Option<HashMap<String, toml::Value>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_parse_simple() {
        let v = Version::parse("1.2.3").unwrap();
        assert_eq!(v.major, 1);
        assert_eq!(v.minor, 2);
        assert_eq!(v.patch, 3);
        assert!(v.pre_release.is_none());
    }

    #[test]
    fn test_version_parse_with_prerelease() {
        let v = Version::parse("2.0.0-beta").unwrap();
        assert_eq!(v.major, 2);
        assert_eq!(v.minor, 0);
        assert_eq!(v.patch, 0);
        assert_eq!(v.pre_release.as_deref(), Some("beta"));
    }

    #[test]
    fn test_version_parse_partial() {
        let v = Version::parse("1.5").unwrap();
        assert_eq!(v.major, 1);
        assert_eq!(v.minor, 5);
        assert_eq!(v.patch, 0);

        let v = Version::parse("3").unwrap();
        assert_eq!(v.major, 3);
        assert_eq!(v.minor, 0);
        assert_eq!(v.patch, 0);
    }

    #[test]
    fn test_version_parse_with_prefix() {
        let v = Version::parse("^1.2.3").unwrap();
        assert_eq!(v.major, 1);
        assert_eq!(v.minor, 2);
        assert_eq!(v.patch, 3);

        let v = Version::parse(">=2.0").unwrap();
        assert_eq!(v.major, 2);
        assert_eq!(v.minor, 0);
    }

    #[test]
    fn test_version_satisfies_minimum() {
        let v1 = Version::parse("1.2.3").unwrap();
        let v2 = Version::parse("1.2.0").unwrap();
        let v3 = Version::parse("1.3.0").unwrap();
        let v4 = Version::parse("2.0.0").unwrap();

        assert!(v1.satisfies_minimum(&v2)); // 1.2.3 >= 1.2.0
        assert!(!v1.satisfies_minimum(&v3)); // 1.2.3 < 1.3.0
        assert!(!v1.satisfies_minimum(&v4)); // 1.2.3 < 2.0.0
    }

    #[test]
    fn test_version_display() {
        let v = Version::parse("1.2.3").unwrap();
        assert_eq!(v.to_string(), "1.2.3");

        let v = Version::parse("1.0.0-alpha").unwrap();
        assert_eq!(v.to_string(), "1.0.0-alpha");
    }

    #[test]
    fn test_stack_component_crate_name() {
        assert_eq!(StackComponent::Trueno.crate_name(), "trueno");
        assert_eq!(StackComponent::TruenoDB.crate_name(), "trueno-db");
        assert_eq!(StackComponent::Aprender.crate_name(), "aprender");
    }

    #[test]
    fn test_parse_stack_component() {
        assert_eq!(
            StackDiscovery::parse_stack_component("trueno"),
            Some(StackComponent::Trueno)
        );
        assert_eq!(
            StackDiscovery::parse_stack_component("trueno-db"),
            Some(StackComponent::TruenoDB)
        );
        assert_eq!(
            StackDiscovery::parse_stack_component("trueno_db"),
            Some(StackComponent::TruenoDB)
        );
        assert_eq!(
            StackDiscovery::parse_stack_component("APRENDER"),
            Some(StackComponent::Aprender)
        );
        assert_eq!(StackDiscovery::parse_stack_component("unknown"), None);
    }

    #[test]
    fn test_discovery_from_toml_simple() {
        let toml = r#"
[package]
name = "test"
version = "0.1.0"

[dependencies]
trueno = "1.0.0"
aprender = "0.5.0"
serde = "1.0"
"#;

        let discovery = StackDiscovery::from_toml_str(toml).unwrap();

        assert!(discovery.has(StackComponent::Trueno));
        assert!(discovery.has(StackComponent::Aprender));
        assert!(!discovery.has(StackComponent::Entrenar));

        let trueno_v = discovery.version(StackComponent::Trueno).unwrap();
        assert_eq!(trueno_v.major, 1);
        assert_eq!(trueno_v.minor, 0);
    }

    #[test]
    fn test_discovery_from_toml_table_format() {
        let toml = r#"
[dependencies]
trueno = { version = "2.1.0", features = ["simd"] }
entrenar = { version = "0.3.0", optional = true }
"#;

        let discovery = StackDiscovery::from_toml_str(toml).unwrap();

        assert!(discovery.has(StackComponent::Trueno));
        assert!(discovery.has(StackComponent::Entrenar));

        let trueno_v = discovery.version(StackComponent::Trueno).unwrap();
        assert_eq!(trueno_v.to_string(), "2.1.0");
    }

    #[test]
    fn test_discovery_dev_dependencies() {
        let toml = r#"
[dev-dependencies]
renacer = "0.1.0"
"#;

        let discovery = StackDiscovery::from_toml_str(toml).unwrap();
        assert!(discovery.has(StackComponent::Renacer));
    }

    #[test]
    fn test_discovery_empty() {
        let toml = r#"
[package]
name = "test"
version = "0.1.0"

[dependencies]
serde = "1.0"
"#;

        let discovery = StackDiscovery::from_toml_str(toml).unwrap();

        assert!(discovery.is_empty());
        assert_eq!(discovery.count(), 0);
    }

    #[test]
    fn test_discovery_check_version() {
        let toml = r#"
[dependencies]
trueno = "1.5.0"
"#;

        let discovery = StackDiscovery::from_toml_str(toml).unwrap();

        let min_ok = Version::parse("1.0.0").unwrap();
        let min_exact = Version::parse("1.5.0").unwrap();
        let min_too_high = Version::parse("2.0.0").unwrap();

        assert!(discovery.check_version(StackComponent::Trueno, &min_ok));
        assert!(discovery.check_version(StackComponent::Trueno, &min_exact));
        assert!(!discovery.check_version(StackComponent::Trueno, &min_too_high));
        assert!(!discovery.check_version(StackComponent::Aprender, &min_ok)); // Not present
    }

    #[test]
    fn test_discovery_register() {
        let mut discovery = StackDiscovery::new();
        assert!(discovery.is_empty());

        discovery.register(StackComponent::Trueno, Version::parse("1.0.0").unwrap());

        assert!(discovery.has(StackComponent::Trueno));
        assert_eq!(discovery.count(), 1);
    }

    #[test]
    fn test_discovery_summary() {
        let toml = r#"
[dependencies]
trueno = "1.0.0"
aprender = "0.5.0"
"#;

        let discovery = StackDiscovery::from_toml_str(toml).unwrap();
        let summary = discovery.summary();

        assert!(summary.contains("2 stack components"));
        assert!(summary.contains("trueno: v1.0.0"));
        assert!(summary.contains("aprender: v0.5.0"));
    }

    #[test]
    fn test_discovery_summary_empty() {
        let discovery = StackDiscovery::new();
        let summary = discovery.summary();

        assert!(summary.contains("No Sovereign AI Stack components detected"));
    }

    #[test]
    fn test_stack_component_all() {
        let all = StackComponent::all();
        assert_eq!(all.len(), 10);

        // Verify all components are unique
        let mut seen = std::collections::HashSet::new();
        for component in all {
            assert!(seen.insert(*component));
        }
    }

    #[test]
    fn test_stack_component_display() {
        assert_eq!(format!("{}", StackComponent::Trueno), "trueno");
        assert_eq!(format!("{}", StackComponent::TruenoGraph), "trueno-graph");
        assert_eq!(format!("{}", StackComponent::TruenoRag), "trueno-rag");
        assert_eq!(format!("{}", StackComponent::Entrenar), "entrenar");
        assert_eq!(format!("{}", StackComponent::Realizar), "realizar");
        assert_eq!(format!("{}", StackComponent::Alimentar), "alimentar");
        assert_eq!(format!("{}", StackComponent::Pacha), "pacha");
        assert_eq!(format!("{}", StackComponent::Renacer), "renacer");
    }

    #[test]
    fn test_version_clone() {
        let v = Version::parse("1.2.3-beta").unwrap();
        let cloned = v.clone();
        assert_eq!(cloned.major, v.major);
        assert_eq!(cloned.pre_release, v.pre_release);
    }

    #[test]
    fn test_version_eq() {
        let v1 = Version::parse("1.2.3").unwrap();
        let v2 = Version::parse("1.2.3").unwrap();
        let v3 = Version::parse("1.2.4").unwrap();
        assert_eq!(v1, v2);
        assert_ne!(v1, v3);
    }

    #[test]
    fn test_version_parse_invalid() {
        // Empty string
        assert!(Version::parse("").is_none());
        // Too many parts
        assert!(Version::parse("1.2.3.4.5").is_none());
        // Non-numeric
        assert!(Version::parse("abc.def.ghi").is_none());
    }

    #[test]
    fn test_version_satisfies_major_gt() {
        let v_new = Version::parse("2.0.0").unwrap();
        let v_old = Version::parse("1.5.0").unwrap();
        assert!(v_new.satisfies_minimum(&v_old));
    }

    #[test]
    fn test_version_satisfies_minor_gt() {
        let v_new = Version::parse("1.5.0").unwrap();
        let v_old = Version::parse("1.3.0").unwrap();
        assert!(v_new.satisfies_minimum(&v_old));
    }

    #[test]
    fn test_stack_component_clone_eq() {
        let c1 = StackComponent::Trueno;
        let c2 = c1.clone();
        assert_eq!(c1, c2);
    }

    #[test]
    fn test_stack_component_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(StackComponent::Trueno);
        set.insert(StackComponent::Aprender);
        set.insert(StackComponent::Trueno); // Duplicate
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_version_parse_with_tilde() {
        let v = Version::parse("~1.2.3").unwrap();
        assert_eq!(v.major, 1);
        assert_eq!(v.minor, 2);
        assert_eq!(v.patch, 3);
    }

    #[test]
    fn test_version_parse_with_lt_gt() {
        let v = Version::parse(">1.0.0").unwrap();
        assert_eq!(v.major, 1);

        let v = Version::parse("<2.0.0").unwrap();
        assert_eq!(v.major, 2);

        let v = Version::parse("<=3.0.0").unwrap();
        assert_eq!(v.major, 3);
    }
}
