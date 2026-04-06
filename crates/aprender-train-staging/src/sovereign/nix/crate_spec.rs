//! Crate specification for Nix packaging

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Crate specification for Nix packaging
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CrateSpec {
    /// Crate name
    pub name: String,
    /// Local path (if building from source)
    pub path: Option<PathBuf>,
    /// Crates.io version (if using published version)
    pub version: Option<String>,
    /// Git repository URL (if using git source)
    pub git: Option<String>,
    /// Git revision/tag
    pub rev: Option<String>,
}

impl CrateSpec {
    /// Create a crate spec for a local path
    pub fn local(name: impl Into<String>, path: impl Into<PathBuf>) -> Self {
        Self { name: name.into(), path: Some(path.into()), version: None, git: None, rev: None }
    }

    /// Create a crate spec for crates.io version
    pub fn crates_io(name: impl Into<String>, version: impl Into<String>) -> Self {
        Self { name: name.into(), path: None, version: Some(version.into()), git: None, rev: None }
    }

    /// Create a crate spec for git source
    pub fn git(name: impl Into<String>, url: impl Into<String>, rev: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            path: None,
            version: None,
            git: Some(url.into()),
            rev: Some(rev.into()),
        }
    }

    /// Check if this is a local source
    pub fn is_local(&self) -> bool {
        self.path.is_some()
    }

    /// Check if this is a crates.io source
    pub fn is_crates_io(&self) -> bool {
        self.version.is_some() && self.git.is_none()
    }

    /// Check if this is a git source
    pub fn is_git(&self) -> bool {
        self.git.is_some()
    }

    /// Get source string for Nix
    pub fn nix_source(&self) -> String {
        if let Some(path) = &self.path {
            format!("./{}", path.display())
        } else if let Some(version) = &self.version {
            format!("crates.io:{version}")
        } else if let (Some(git), Some(rev)) = (&self.git, &self.rev) {
            format!("git:{git}?rev={rev}")
        } else {
            "unknown".to_string()
        }
    }
}
