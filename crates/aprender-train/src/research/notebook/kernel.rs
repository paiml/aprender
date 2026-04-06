//! Jupyter kernel specifications.

use serde::{Deserialize, Serialize};

/// Jupyter kernel specification
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KernelSpec {
    /// Display name
    pub display_name: String,
    /// Language
    pub language: String,
    /// Kernel name
    pub name: String,
}

impl Default for KernelSpec {
    fn default() -> Self {
        Self::python3()
    }
}

impl KernelSpec {
    /// Python 3 kernel
    pub fn python3() -> Self {
        Self {
            display_name: "Python 3".to_string(),
            language: "python".to_string(),
            name: "python3".to_string(),
        }
    }

    /// evcxr Rust kernel
    pub fn evcxr() -> Self {
        Self {
            display_name: "Rust".to_string(),
            language: "rust".to_string(),
            name: "rust".to_string(),
        }
    }

    /// Julia kernel
    pub fn julia() -> Self {
        Self {
            display_name: "Julia 1.9".to_string(),
            language: "julia".to_string(),
            name: "julia-1.9".to_string(),
        }
    }
}
