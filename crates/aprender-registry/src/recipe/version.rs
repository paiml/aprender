//! Recipe versioning.

use crate::error::{PachaError, Result};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::fmt;
use std::str::FromStr;

/// Version for a recipe.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RecipeVersion {
    /// Major version.
    pub major: u32,
    /// Minor version.
    pub minor: u32,
    /// Patch version.
    pub patch: u32,
}

impl RecipeVersion {
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

    /// Increment major version.
    #[must_use]
    pub fn bump_major(&self) -> Self {
        Self::new(self.major + 1, 0, 0)
    }

    /// Increment minor version.
    #[must_use]
    pub fn bump_minor(&self) -> Self {
        Self::new(self.major, self.minor + 1, 0)
    }

    /// Increment patch version.
    #[must_use]
    pub fn bump_patch(&self) -> Self {
        Self::new(self.major, self.minor, self.patch + 1)
    }
}

impl Default for RecipeVersion {
    fn default() -> Self {
        Self::initial()
    }
}

impl fmt::Display for RecipeVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

impl FromStr for RecipeVersion {
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

impl PartialOrd for RecipeVersion {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for RecipeVersion {
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
        let v = RecipeVersion::new(1, 2, 3);
        assert_eq!(v.major, 1);
        assert_eq!(v.minor, 2);
        assert_eq!(v.patch, 3);
    }

    #[test]
    fn test_version_display() {
        assert_eq!(RecipeVersion::new(1, 2, 3).to_string(), "1.2.3");
    }

    #[test]
    fn test_version_parse() {
        assert_eq!("1.2.3".parse::<RecipeVersion>().unwrap(), RecipeVersion::new(1, 2, 3));
    }

    #[test]
    fn test_version_parse_errors() {
        assert!("1.2".parse::<RecipeVersion>().is_err());
        assert!("a.b.c".parse::<RecipeVersion>().is_err());
    }

    #[test]
    fn test_version_bump() {
        let v = RecipeVersion::new(1, 2, 3);
        assert_eq!(v.bump_major(), RecipeVersion::new(2, 0, 0));
        assert_eq!(v.bump_minor(), RecipeVersion::new(1, 3, 0));
        assert_eq!(v.bump_patch(), RecipeVersion::new(1, 2, 4));
    }

    #[test]
    fn test_version_ordering() {
        let v100 = RecipeVersion::new(1, 0, 0);
        let v110 = RecipeVersion::new(1, 1, 0);
        let v200 = RecipeVersion::new(2, 0, 0);

        assert!(v100 < v110);
        assert!(v110 < v200);
    }

    proptest! {
        #[test]
        fn prop_version_roundtrip(major: u32, minor: u32, patch: u32) {
            let v = RecipeVersion::new(major, minor, patch);
            let s = v.to_string();
            let parsed: RecipeVersion = s.parse().unwrap();
            prop_assert_eq!(v, parsed);
        }
    }
}
