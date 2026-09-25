//! The training write perimeter (EXT-001 §3.2, R-7; FALSIFY-EXT-011).
//!
//! A training process must run **without** read access to publish
//! credentials and **without** write access to the sealed eval manifest or
//! the promotion contract. Otherwise the candidate could influence its own
//! promotion evidence (I-9).
//!
//! This guard is enforced by the released `apr` binary, which the candidate
//! (weights, recipes) cannot rewrite. It fails closed. A training verb
//! refuses to start when:
//! - a publish credential file under [`PUBLISH_CREDENTIALS_DIR`] is readable
//!   by this process: on the driver host the publisher must be a separate
//!   account that owns the tokens;
//! - a publish credential is in the environment ([`PUBLISH_TOKEN_ENV`]):
//!   the publisher reads its token from the driver-host path only;
//! - one of its output paths resolves inside [`sealed_dir`], where the
//!   sealed manifest and the promotion contract live.
//!
//! The locations are fixed, with no environment override, so a training run
//! cannot move the perimeter.

use std::path::{Path, PathBuf};

/// Where the publisher's HF and GHCR tokens live on the driver host (EXT-14).
pub const PUBLISH_CREDENTIALS_DIR: &str = "/etc/apr/publish";

/// Environment names a publish credential could leak through.
pub const PUBLISH_TOKEN_ENV: [&str; 3] =
    ["APR_PUBLISH_HF_TOKEN", "APR_PUBLISH_GHCR_TOKEN", "APR_PUBLISH_TOKEN"];

/// The sealed store: sealed eval manifests and the promotion contract.
#[must_use]
pub fn sealed_dir() -> Option<PathBuf> {
    dirs::home_dir().map(|h| h.join(".pacha").join("sealed"))
}

/// One way a training process reaches past the perimeter.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum PerimeterViolation {
    #[error("publish credential {0} is readable by the training process (R-7): the publisher must own it under a separate account")]
    CredentialReadable(PathBuf),
    #[error("publish credential in the training environment: ${0} (R-7)")]
    CredentialInEnv(String),
    #[error("training output {0} is inside the sealed store {1} (I-9)")]
    WritesSealed(PathBuf, PathBuf),
}

/// The perimeter a training process is checked against.
#[derive(Debug, Clone)]
pub struct Perimeter {
    credentials_dir: PathBuf,
    sealed_dir: Option<PathBuf>,
}

impl Perimeter {
    /// The perimeter at its fixed locations.
    #[must_use]
    pub fn fixed() -> Self {
        Self { credentials_dir: PathBuf::from(PUBLISH_CREDENTIALS_DIR), sealed_dir: sealed_dir() }
    }

    /// A perimeter at explicit locations (tests plant these).
    #[must_use]
    pub fn at(credentials_dir: impl Into<PathBuf>, sealed_dir: impl Into<PathBuf>) -> Self {
        Self { credentials_dir: credentials_dir.into(), sealed_dir: Some(sealed_dir.into()) }
    }

    /// Every violation for a training process with environment `env` that
    /// will write `outputs`. Empty means the run may start.
    pub fn violations<'a>(
        &self,
        env: impl IntoIterator<Item = (String, String)>,
        outputs: impl IntoIterator<Item = &'a Path>,
    ) -> Vec<PerimeterViolation> {
        let mut bad = Vec::new();
        if let Ok(entries) = std::fs::read_dir(&self.credentials_dir) {
            for path in entries.filter_map(std::result::Result::ok).map(|e| e.path()) {
                // Readable = this process could open it. Opening reads nothing.
                if path.is_file() && std::fs::File::open(&path).is_ok() {
                    bad.push(PerimeterViolation::CredentialReadable(path));
                }
            }
        }
        for (name, value) in env {
            if PUBLISH_TOKEN_ENV.contains(&name.as_str()) && !value.is_empty() {
                bad.push(PerimeterViolation::CredentialInEnv(name));
            }
        }
        if let Some(sealed) = &self.sealed_dir {
            let sealed_abs = resolve(sealed);
            for out in outputs {
                let out_abs = resolve(out);
                if out_abs.starts_with(&sealed_abs) {
                    bad.push(PerimeterViolation::WritesSealed(out_abs, sealed_abs.clone()));
                }
            }
        }
        bad
    }

    /// Check this process: its environment and the outputs it will write.
    pub fn enforce<'a>(
        &self,
        outputs: impl IntoIterator<Item = &'a Path>,
    ) -> std::result::Result<(), Vec<PerimeterViolation>> {
        let bad = self.violations(std::env::vars(), outputs);
        if bad.is_empty() {
            Ok(())
        } else {
            Err(bad)
        }
    }
}

/// `path` made absolute, with its longest existing prefix canonicalized so a
/// symlink or `..` into the sealed store is caught before the file exists.
fn resolve(path: &Path) -> PathBuf {
    let abs = std::path::absolute(path).unwrap_or_else(|_| path.to_path_buf());
    let mut existing = abs.as_path();
    let mut rest = Vec::new();
    loop {
        if let Ok(canon) = existing.canonicalize() {
            return rest.iter().rev().fold(canon, |acc, part| acc.join(part));
        }
        match (existing.parent(), existing.file_name()) {
            (Some(parent), Some(name)) => {
                rest.push(name.to_os_string());
                existing = parent;
            }
            _ => return abs,
        }
    }
}

#[cfg(test)]
#[path = "perimeter_tests.rs"]
mod tests;
