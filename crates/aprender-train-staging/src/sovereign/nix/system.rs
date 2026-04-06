//! Target system for Nix builds

use serde::{Deserialize, Serialize};

/// Target system for Nix builds
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NixSystem {
    /// x86_64 Linux
    X86_64Linux,
    /// aarch64 Linux
    Aarch64Linux,
    /// x86_64 macOS
    X86_64Darwin,
    /// aarch64 macOS (Apple Silicon)
    Aarch64Darwin,
}

impl NixSystem {
    /// Get the Nix system string
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::X86_64Linux => "x86_64-linux",
            Self::Aarch64Linux => "aarch64-linux",
            Self::X86_64Darwin => "x86_64-darwin",
            Self::Aarch64Darwin => "aarch64-darwin",
        }
    }

    /// Get all supported systems
    pub fn all() -> Vec<Self> {
        vec![Self::X86_64Linux, Self::Aarch64Linux, Self::X86_64Darwin, Self::Aarch64Darwin]
    }

    /// Get Linux systems only
    pub fn linux_only() -> Vec<Self> {
        vec![Self::X86_64Linux, Self::Aarch64Linux]
    }
}

impl std::fmt::Display for NixSystem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}
