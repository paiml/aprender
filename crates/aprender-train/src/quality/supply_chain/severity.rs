//! Severity levels for security advisories.

use serde::{Deserialize, Serialize};

/// Severity level for security advisories
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub enum Severity {
    /// Informational only
    None,
    /// Low severity
    Low,
    /// Medium severity
    Medium,
    /// High severity
    High,
    /// Critical severity
    Critical,
}

impl Severity {
    /// Parse severity from string (infallible, returns None for unknown)
    pub fn parse(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "critical" => Self::Critical,
            "high" => Self::High,
            "medium" => Self::Medium,
            "low" => Self::Low,
            _ => Self::None,
        }
    }
}

impl std::fmt::Display for Severity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::None => write!(f, "none"),
            Self::Low => write!(f, "low"),
            Self::Medium => write!(f, "medium"),
            Self::High => write!(f, "high"),
            Self::Critical => write!(f, "critical"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_severity_parse_critical() {
        assert_eq!(Severity::parse("critical"), Severity::Critical);
        assert_eq!(Severity::parse("CRITICAL"), Severity::Critical);
        assert_eq!(Severity::parse("Critical"), Severity::Critical);
    }

    #[test]
    fn test_severity_parse_high() {
        assert_eq!(Severity::parse("high"), Severity::High);
        assert_eq!(Severity::parse("HIGH"), Severity::High);
    }

    #[test]
    fn test_severity_parse_medium() {
        assert_eq!(Severity::parse("medium"), Severity::Medium);
        assert_eq!(Severity::parse("MEDIUM"), Severity::Medium);
    }

    #[test]
    fn test_severity_parse_low() {
        assert_eq!(Severity::parse("low"), Severity::Low);
        assert_eq!(Severity::parse("LOW"), Severity::Low);
    }

    #[test]
    fn test_severity_parse_unknown() {
        assert_eq!(Severity::parse("unknown"), Severity::None);
        assert_eq!(Severity::parse(""), Severity::None);
        assert_eq!(Severity::parse("info"), Severity::None);
    }

    #[test]
    fn test_severity_display() {
        assert_eq!(Severity::None.to_string(), "none");
        assert_eq!(Severity::Low.to_string(), "low");
        assert_eq!(Severity::Medium.to_string(), "medium");
        assert_eq!(Severity::High.to_string(), "high");
        assert_eq!(Severity::Critical.to_string(), "critical");
    }

    #[test]
    fn test_severity_ord() {
        assert!(Severity::None < Severity::Low);
        assert!(Severity::Low < Severity::Medium);
        assert!(Severity::Medium < Severity::High);
        assert!(Severity::High < Severity::Critical);
    }

    #[test]
    fn test_severity_eq() {
        assert_eq!(Severity::Critical, Severity::Critical);
        assert_ne!(Severity::Critical, Severity::High);
    }

    #[test]
    fn test_severity_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(Severity::Critical);
        set.insert(Severity::Critical);
        assert_eq!(set.len(), 1);
        set.insert(Severity::High);
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_severity_serde() {
        let sev = Severity::High;
        let json = serde_json::to_string(&sev).expect("JSON serialization should succeed");
        let deserialized: Severity =
            serde_json::from_str(&json).expect("JSON deserialization should succeed");
        assert_eq!(sev, deserialized);
    }

    #[test]
    fn test_severity_clone() {
        let sev = Severity::Critical;
        let cloned = sev;
        assert_eq!(sev, cloned);
    }

    #[test]
    fn test_severity_debug() {
        assert_eq!(format!("{:?}", Severity::Critical), "Critical");
        assert_eq!(format!("{:?}", Severity::None), "None");
    }
}
