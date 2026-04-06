//! Dataset versioning.
//!
//! Uses content-based versioning similar to Git but optimized for large binary files.

use crate::error::{PachaError, Result};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::fmt;
use std::str::FromStr;

/// Version for a dataset.
///
/// Uses semantic versioning:
/// - MAJOR: Schema breaking change
/// - MINOR: New data added (backward compatible)
/// - PATCH: Data corrections, documentation updates
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DatasetVersion {
    /// Major version (schema changes).
    pub major: u32,
    /// Minor version (new data).
    pub minor: u32,
    /// Patch version (corrections).
    pub patch: u32,
}

impl DatasetVersion {
    /// Create a new version.
    #[must_use]
    pub fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self { major, minor, patch }
    }

    /// Create version 1.0.0.
    #[must_use]
    pub fn initial() -> Self {
        Self::new(1, 0, 0)
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

    /// Parse a version string.
    ///
    /// # Errors
    ///
    /// Returns an error if the string is not a valid version.
    pub fn parse(s: &str) -> Result<Self> {
        s.parse()
    }
}

impl Default for DatasetVersion {
    fn default() -> Self {
        Self::initial()
    }
}

impl fmt::Display for DatasetVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

impl FromStr for DatasetVersion {
    type Err = PachaError;

    fn from_str(s: &str) -> Result<Self> {
        let parts: Vec<&str> = s.split('.').collect();
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

        Ok(Self { major, minor, patch })
    }
}

impl PartialOrd for DatasetVersion {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for DatasetVersion {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.major.cmp(&other.major) {
            Ordering::Equal => {}
            ord => return ord,
        }
        match self.minor.cmp(&other.minor) {
            Ordering::Equal => {}
            ord => return ord,
        }
        self.patch.cmp(&other.patch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn test_version_new() {
        let v = DatasetVersion::new(1, 2, 3);
        assert_eq!(v.major, 1);
        assert_eq!(v.minor, 2);
        assert_eq!(v.patch, 3);
    }

    #[test]
    fn test_version_display() {
        assert_eq!(DatasetVersion::new(1, 2, 3).to_string(), "1.2.3");
    }

    #[test]
    fn test_version_parse() {
        assert_eq!("1.2.3".parse::<DatasetVersion>().unwrap(), DatasetVersion::new(1, 2, 3));
    }

    #[test]
    fn test_version_parse_errors() {
        assert!("1.2".parse::<DatasetVersion>().is_err());
        assert!("a.b.c".parse::<DatasetVersion>().is_err());
    }

    #[test]
    fn test_version_bump() {
        let v = DatasetVersion::new(1, 2, 3);
        assert_eq!(v.bump_major(), DatasetVersion::new(2, 0, 0));
        assert_eq!(v.bump_minor(), DatasetVersion::new(1, 3, 0));
        assert_eq!(v.bump_patch(), DatasetVersion::new(1, 2, 4));
    }

    #[test]
    fn test_version_ordering() {
        let v100 = DatasetVersion::new(1, 0, 0);
        let v110 = DatasetVersion::new(1, 1, 0);
        let v200 = DatasetVersion::new(2, 0, 0);

        assert!(v100 < v110);
        assert!(v110 < v200);
    }

    proptest! {
        #[test]
        fn prop_version_roundtrip(major: u32, minor: u32, patch: u32) {
            let v = DatasetVersion::new(major, minor, patch);
            let s = v.to_string();
            let parsed: DatasetVersion = s.parse().unwrap();
            prop_assert_eq!(v, parsed);
        }
    }
}
