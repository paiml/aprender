//! Semantic versioning for ML models.
//!
//! Follows Semantic Versioning 2.0.0 with ML-specific semantics:
//! - MAJOR: Architecture change (incompatible inputs/outputs)
//! - MINOR: Retraining with new data (backward compatible)
//! - PATCH: Bug fixes, quantization, optimization

use crate::error::{PachaError, Result};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::fmt;
use std::str::FromStr;

/// Semantic version for a model.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ModelVersion {
    /// Major version (architecture changes).
    pub major: u32,
    /// Minor version (retraining).
    pub minor: u32,
    /// Patch version (optimizations).
    pub patch: u32,
    /// Optional pre-release identifier (e.g., "beta.1").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prerelease: Option<String>,
    /// Build metadata (e.g., training run ID).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub build: Option<String>,
}

impl ModelVersion {
    /// Create a new version.
    #[must_use]
    pub fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self { major, minor, patch, prerelease: None, build: None }
    }

    /// Create version 0.0.0.
    #[must_use]
    pub fn zero() -> Self {
        Self::new(0, 0, 0)
    }

    /// Create version 1.0.0.
    #[must_use]
    pub fn initial() -> Self {
        Self::new(1, 0, 0)
    }

    /// Set pre-release identifier.
    #[must_use]
    pub fn with_prerelease(mut self, prerelease: impl Into<String>) -> Self {
        self.prerelease = Some(prerelease.into());
        self
    }

    /// Set build metadata.
    #[must_use]
    pub fn with_build(mut self, build: impl Into<String>) -> Self {
        self.build = Some(build.into());
        self
    }

    /// Increment major version (resets minor and patch).
    #[must_use]
    pub fn bump_major(&self) -> Self {
        Self::new(self.major + 1, 0, 0)
    }

    /// Increment minor version (resets patch).
    #[must_use]
    pub fn bump_minor(&self) -> Self {
        Self::new(self.major, self.minor + 1, 0)
    }

    /// Increment patch version.
    #[must_use]
    pub fn bump_patch(&self) -> Self {
        Self::new(self.major, self.minor, self.patch + 1)
    }

    /// Check if this is a pre-release version.
    #[must_use]
    pub fn is_prerelease(&self) -> bool {
        self.prerelease.is_some()
    }

    /// Check if this is a stable version (>= 1.0.0, no prerelease).
    #[must_use]
    pub fn is_stable(&self) -> bool {
        self.major >= 1 && self.prerelease.is_none()
    }

    /// Parse a version string.
    ///
    /// # Errors
    ///
    /// Returns an error if the string is not a valid semantic version.
    pub fn parse(s: &str) -> Result<Self> {
        s.parse()
    }
}

impl Default for ModelVersion {
    fn default() -> Self {
        Self::initial()
    }
}

impl fmt::Display for ModelVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)?;
        if let Some(ref pre) = self.prerelease {
            write!(f, "-{pre}")?;
        }
        if let Some(ref build) = self.build {
            write!(f, "+{build}")?;
        }
        Ok(())
    }
}

impl FromStr for ModelVersion {
    type Err = PachaError;

    fn from_str(s: &str) -> Result<Self> {
        // Split off build metadata first (after +)
        let (version_pre, build) = match s.split_once('+') {
            Some((v, b)) => (v, Some(b.to_string())),
            None => (s, None),
        };

        // Split off prerelease (after -)
        let (version, prerelease) = match version_pre.split_once('-') {
            Some((v, p)) => (v, Some(p.to_string())),
            None => (version_pre, None),
        };

        // Parse major.minor.patch
        let parts: Vec<&str> = version.split('.').collect();
        if parts.len() != 3 {
            return Err(PachaError::InvalidVersion(format!(
                "expected MAJOR.MINOR.PATCH, got '{s}'"
            )));
        }

        let major = parts[0]
            .parse::<u32>()
            .map_err(|_| PachaError::InvalidVersion(format!("invalid major version in '{s}'")))?;
        let minor = parts[1]
            .parse::<u32>()
            .map_err(|_| PachaError::InvalidVersion(format!("invalid minor version in '{s}'")))?;
        let patch = parts[2]
            .parse::<u32>()
            .map_err(|_| PachaError::InvalidVersion(format!("invalid patch version in '{s}'")))?;

        Ok(Self { major, minor, patch, prerelease, build })
    }
}

impl PartialOrd for ModelVersion {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ModelVersion {
    fn cmp(&self, other: &Self) -> Ordering {
        // Compare major, minor, patch
        match self.major.cmp(&other.major) {
            Ordering::Equal => {}
            ord => return ord,
        }
        match self.minor.cmp(&other.minor) {
            Ordering::Equal => {}
            ord => return ord,
        }
        match self.patch.cmp(&other.patch) {
            Ordering::Equal => {}
            ord => return ord,
        }

        // Pre-release versions have lower precedence
        match (&self.prerelease, &other.prerelease) {
            (None, None) => Ordering::Equal,
            (Some(_), None) => Ordering::Less,
            (None, Some(_)) => Ordering::Greater,
            (Some(a), Some(b)) => a.cmp(b),
        }
        // Build metadata is ignored for precedence
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn test_version_new() {
        let v = ModelVersion::new(1, 2, 3);
        assert_eq!(v.major, 1);
        assert_eq!(v.minor, 2);
        assert_eq!(v.patch, 3);
        assert!(v.prerelease.is_none());
        assert!(v.build.is_none());
    }

    #[test]
    fn test_version_display() {
        assert_eq!(ModelVersion::new(1, 2, 3).to_string(), "1.2.3");
        assert_eq!(
            ModelVersion::new(1, 0, 0).with_prerelease("beta.1").to_string(),
            "1.0.0-beta.1"
        );
        assert_eq!(ModelVersion::new(1, 0, 0).with_build("run-123").to_string(), "1.0.0+run-123");
        assert_eq!(
            ModelVersion::new(1, 0, 0).with_prerelease("rc.1").with_build("abc").to_string(),
            "1.0.0-rc.1+abc"
        );
    }

    #[test]
    fn test_version_parse() {
        assert_eq!("1.2.3".parse::<ModelVersion>().unwrap(), ModelVersion::new(1, 2, 3));
        assert_eq!("0.0.0".parse::<ModelVersion>().unwrap(), ModelVersion::zero());

        let with_pre: ModelVersion = "1.0.0-beta.1".parse().unwrap();
        assert_eq!(with_pre.prerelease, Some("beta.1".to_string()));

        let with_build: ModelVersion = "1.0.0+run-123".parse().unwrap();
        assert_eq!(with_build.build, Some("run-123".to_string()));

        let full: ModelVersion = "2.1.0-rc.1+build.456".parse().unwrap();
        assert_eq!(full.major, 2);
        assert_eq!(full.minor, 1);
        assert_eq!(full.patch, 0);
        assert_eq!(full.prerelease, Some("rc.1".to_string()));
        assert_eq!(full.build, Some("build.456".to_string()));
    }

    #[test]
    fn test_version_parse_errors() {
        assert!("1.2".parse::<ModelVersion>().is_err());
        assert!("1.2.3.4".parse::<ModelVersion>().is_err());
        assert!("a.b.c".parse::<ModelVersion>().is_err());
        assert!("1.2.three".parse::<ModelVersion>().is_err());
    }

    #[test]
    fn test_version_bump() {
        let v = ModelVersion::new(1, 2, 3);

        assert_eq!(v.bump_major(), ModelVersion::new(2, 0, 0));
        assert_eq!(v.bump_minor(), ModelVersion::new(1, 3, 0));
        assert_eq!(v.bump_patch(), ModelVersion::new(1, 2, 4));
    }

    #[test]
    fn test_version_ordering() {
        let v100 = ModelVersion::new(1, 0, 0);
        let v110 = ModelVersion::new(1, 1, 0);
        let v111 = ModelVersion::new(1, 1, 1);
        let v200 = ModelVersion::new(2, 0, 0);

        assert!(v100 < v110);
        assert!(v110 < v111);
        assert!(v111 < v200);
    }

    #[test]
    fn test_prerelease_ordering() {
        let stable = ModelVersion::new(1, 0, 0);
        let beta = ModelVersion::new(1, 0, 0).with_prerelease("beta");
        let alpha = ModelVersion::new(1, 0, 0).with_prerelease("alpha");

        // Pre-release < stable
        assert!(beta < stable);
        assert!(alpha < stable);
        // Alphabetic ordering for pre-releases
        assert!(alpha < beta);
    }

    #[test]
    fn test_is_stable() {
        assert!(ModelVersion::new(1, 0, 0).is_stable());
        assert!(ModelVersion::new(2, 5, 3).is_stable());
        assert!(!ModelVersion::new(0, 9, 0).is_stable());
        assert!(!ModelVersion::new(1, 0, 0).with_prerelease("beta").is_stable());
    }

    #[test]
    fn test_serialization() {
        let v = ModelVersion::new(1, 2, 3).with_prerelease("rc.1");
        let json = serde_json::to_string(&v).unwrap();
        let deserialized: ModelVersion = serde_json::from_str(&json).unwrap();
        assert_eq!(v, deserialized);
    }

    // Property-based tests
    proptest! {
        #[test]
        fn prop_version_roundtrip(major: u32, minor: u32, patch: u32) {
            let v = ModelVersion::new(major, minor, patch);
            let s = v.to_string();
            let parsed: ModelVersion = s.parse().unwrap();
            prop_assert_eq!(v, parsed);
        }

        #[test]
        fn prop_bump_major_resets(major in 0u32..1000, minor in 0u32..1000, patch in 0u32..1000) {
            let v = ModelVersion::new(major, minor, patch);
            let bumped = v.bump_major();
            prop_assert_eq!(bumped.major, major + 1);
            prop_assert_eq!(bumped.minor, 0);
            prop_assert_eq!(bumped.patch, 0);
        }

        #[test]
        fn prop_bump_minor_resets_patch(major in 0u32..1000, minor in 0u32..1000, patch in 0u32..1000) {
            let v = ModelVersion::new(major, minor, patch);
            let bumped = v.bump_minor();
            prop_assert_eq!(bumped.major, major);
            prop_assert_eq!(bumped.minor, minor + 1);
            prop_assert_eq!(bumped.patch, 0);
        }

        #[test]
        fn prop_ordering_transitive(
            a_major in 0u32..10, a_minor in 0u32..10, a_patch in 0u32..10,
            b_major in 0u32..10, b_minor in 0u32..10, b_patch in 0u32..10,
            c_major in 0u32..10, c_minor in 0u32..10, c_patch in 0u32..10,
        ) {
            let a = ModelVersion::new(a_major, a_minor, a_patch);
            let b = ModelVersion::new(b_major, b_minor, b_patch);
            let c = ModelVersion::new(c_major, c_minor, c_patch);

            if a < b && b < c {
                prop_assert!(a < c);
            }
        }
    }
}

// ─── Kani Formal Verification ────────────────────────────────────────────

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    #[kani::proof]
    fn verify_bump_major_resets() {
        let major: u32 = kani::any();
        let minor: u32 = kani::any();
        let patch: u32 = kani::any();
        kani::assume(major < u32::MAX);
        let v = ModelVersion::new(major, minor, patch);
        let bumped = v.bump_major();
        assert!(bumped.major == major + 1);
        assert!(bumped.minor == 0);
        assert!(bumped.patch == 0);
    }

    #[kani::proof]
    fn verify_bump_ordering() {
        let major: u32 = kani::any();
        let minor: u32 = kani::any();
        let patch: u32 = kani::any();
        kani::assume(major < 100 && minor < 100 && patch < u32::MAX);
        let v = ModelVersion::new(major, minor, patch);
        let bumped = v.bump_patch();
        assert!(bumped > v);
    }

    #[kani::proof]
    fn verify_version_equality_reflexive() {
        let major: u32 = kani::any();
        let minor: u32 = kani::any();
        let patch: u32 = kani::any();
        kani::assume(major <= 100 && minor <= 100 && patch <= 100);
        let v = ModelVersion::new(major, minor, patch);
        assert!(v == v.clone());
    }
}
