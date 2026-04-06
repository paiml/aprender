//! Software license types.

use serde::{Deserialize, Serialize};

/// Software license
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum License {
    /// MIT License
    Mit,
    /// Apache License 2.0
    Apache2,
    /// BSD 3-Clause
    Bsd3,
    /// GNU GPL v3
    Gpl3,
    /// Creative Commons Attribution 4.0
    CcBy4,
    /// Creative Commons Zero (public domain)
    Cc0,
    /// Custom license with SPDX identifier
    Custom(String),
}

impl std::fmt::Display for License {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Mit => write!(f, "MIT"),
            Self::Apache2 => write!(f, "Apache-2.0"),
            Self::Bsd3 => write!(f, "BSD-3-Clause"),
            Self::Gpl3 => write!(f, "GPL-3.0"),
            Self::CcBy4 => write!(f, "CC-BY-4.0"),
            Self::Cc0 => write!(f, "CC0-1.0"),
            Self::Custom(spdx) => write!(f, "{spdx}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_license_display() {
        assert_eq!(License::Mit.to_string(), "MIT");
        assert_eq!(License::Apache2.to_string(), "Apache-2.0");
        assert_eq!(License::Bsd3.to_string(), "BSD-3-Clause");
        assert_eq!(License::Gpl3.to_string(), "GPL-3.0");
        assert_eq!(License::CcBy4.to_string(), "CC-BY-4.0");
        assert_eq!(License::Cc0.to_string(), "CC0-1.0");
        assert_eq!(License::Custom("LGPL-2.1".to_string()).to_string(), "LGPL-2.1");
    }

    #[test]
    fn test_license_clone() {
        let license = License::Mit;
        let cloned = license.clone();
        assert_eq!(license, cloned);

        let custom = License::Custom("WTFPL".to_string());
        let custom_cloned = custom.clone();
        assert_eq!(custom, custom_cloned);
    }

    #[test]
    fn test_license_eq() {
        assert_eq!(License::Mit, License::Mit);
        assert_ne!(License::Mit, License::Apache2);
        assert_eq!(License::Custom("ISC".to_string()), License::Custom("ISC".to_string()));
        assert_ne!(License::Custom("ISC".to_string()), License::Custom("MPL-2.0".to_string()));
    }

    #[test]
    fn test_license_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(License::Mit);
        set.insert(License::Mit);
        assert_eq!(set.len(), 1);
        set.insert(License::Apache2);
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_license_serde() {
        let license = License::Mit;
        let json = serde_json::to_string(&license).expect("JSON serialization should succeed");
        let deserialized: License =
            serde_json::from_str(&json).expect("JSON deserialization should succeed");
        assert_eq!(license, deserialized);

        let custom = License::Custom("Unlicense".to_string());
        let custom_json =
            serde_json::to_string(&custom).expect("JSON serialization should succeed");
        let custom_deserialized: License =
            serde_json::from_str(&custom_json).expect("JSON deserialization should succeed");
        assert_eq!(custom, custom_deserialized);
    }

    #[test]
    fn test_license_debug() {
        assert_eq!(format!("{:?}", License::Mit), "Mit");
        assert_eq!(format!("{:?}", License::Custom("ISC".to_string())), "Custom(\"ISC\")");
    }
}
