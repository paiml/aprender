//! Individual component manifest

use serde::{Deserialize, Serialize};

/// Individual component manifest
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ComponentManifest {
    /// Component name
    pub name: String,
    /// Component version
    pub version: String,
    /// Crate name on crates.io
    pub crate_name: String,
    /// Enabled features
    pub features: Vec<String>,
}

impl ComponentManifest {
    /// Create a new component manifest
    pub fn new(
        name: impl Into<String>,
        version: impl Into<String>,
        crate_name: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            version: version.into(),
            crate_name: crate_name.into(),
            features: Vec::new(),
        }
    }

    /// Add features to the component
    pub fn with_features(mut self, features: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.features = features.into_iter().map(Into::into).collect();
        self
    }

    /// Create component for entrenar-core
    pub fn entrenar_core(version: &str) -> Self {
        Self::new("entrenar-core", version, "entrenar")
    }

    /// Create component for trueno
    pub fn trueno(version: &str) -> Self {
        Self::new("trueno", version, "trueno")
    }

    /// Create component for aprender
    pub fn aprender(version: &str) -> Self {
        Self::new("aprender", version, "aprender")
    }

    /// Create component for renacer
    pub fn renacer(version: &str) -> Self {
        Self::new("renacer", version, "renacer")
    }
}
