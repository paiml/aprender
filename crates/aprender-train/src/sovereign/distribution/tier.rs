//! Distribution tier levels

use serde::{Deserialize, Serialize};

/// Distribution tier levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum DistributionTier {
    /// ~50MB: entrenar-core, trueno, aprender
    #[default]
    Core,
    /// ~200MB: + renacer, trueno-db, ruchy
    Standard,
    /// ~500MB: + GPU support, all tooling
    Full,
}

impl DistributionTier {
    /// Get the approximate size in megabytes
    pub fn approximate_size_mb(&self) -> u64 {
        match self {
            Self::Core => 50,
            Self::Standard => 200,
            Self::Full => 500,
        }
    }

    /// Get the core component names for this tier
    pub fn component_names(&self) -> Vec<&'static str> {
        match self {
            Self::Core => vec!["entrenar-core", "trueno", "aprender"],
            Self::Standard => {
                vec!["entrenar-core", "trueno", "aprender", "renacer", "trueno-db", "ruchy"]
            }
            Self::Full => vec![
                "entrenar-core",
                "trueno",
                "aprender",
                "renacer",
                "trueno-db",
                "ruchy",
                "entrenar-gpu",
                "entrenar-bench",
                "entrenar-inspect",
                "entrenar-lora",
                "entrenar-shell",
            ],
        }
    }

    /// Check if this tier includes a specific component
    pub fn includes(&self, component: &str) -> bool {
        self.component_names().contains(&component)
    }
}

impl std::fmt::Display for DistributionTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Core => write!(f, "core"),
            Self::Standard => write!(f, "standard"),
            Self::Full => write!(f, "full"),
        }
    }
}
