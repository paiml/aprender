//! Build-time trace assertion DSL parser (Sprint 44)
//!
//! This module parses renacer.toml files containing trace assertions.
//! The DSL enables declarative specification of performance constraints
//! that are validated at build time.
//!
//! # Example renacer.toml
//!
//! ```toml
//! # Critical path latency assertion
//! [[assertion]]
//! name = "api_max_latency"
//! type = "critical_path"
//! max_duration_ms = 100
//! trace_name_pattern = "api_.*"
//! fail_on_violation = true
//!
//! # Anti-pattern detection
//! [[assertion]]
//! name = "no_god_process"
//! type = "anti_pattern"
//! pattern = "GodProcess"
//! threshold = 0.8
//! fail_on_violation = true
//! ```

use crate::assertion_types::Assertion;
use anyhow::{Context, Result};
use serde::Deserialize;
use std::fs;
use std::path::Path;

/// Root configuration for renacer.toml
#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct AssertionConfig {
    /// List of assertions to evaluate
    #[serde(default)]
    pub assertion: Vec<Assertion>,
}

impl AssertionConfig {
    /// Load assertions from a TOML file
    ///
    /// # Arguments
    ///
    /// * `path` - Path to renacer.toml file
    ///
    /// # Returns
    ///
    /// Parsed assertion configuration
    ///
    /// # Example
    ///
    /// ```no_run
    /// use renacer::assertion_dsl::AssertionConfig;
    ///
    /// # fn main() -> anyhow::Result<()> {
    /// let config = AssertionConfig::from_file("renacer.toml")?;
    /// println!("Loaded {} assertions", config.assertion.len());
    /// # Ok(())
    /// # }
    /// ```
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let content = fs::read_to_string(path)
            .with_context(|| format!("Failed to read {}", path.display()))?;

        Self::from_toml_str(&content)
    }

    /// Load assertions from a TOML string
    ///
    /// # Arguments
    ///
    /// * `content` - TOML content as string
    ///
    /// # Returns
    ///
    /// Parsed assertion configuration
    pub fn from_toml_str(content: &str) -> Result<Self> {
        toml::from_str(content).context("Failed to parse TOML")
    }

    /// Get enabled assertions only
    ///
    /// Filters out assertions where `enabled = false`.
    pub fn enabled_assertions(&self) -> Vec<&Assertion> {
        self.assertion.iter().filter(|a| a.enabled).collect()
    }

    /// Get assertions that fail on violation
    ///
    /// Filters assertions where `fail_on_violation = true`.
    pub fn fail_on_violation_assertions(&self) -> Vec<&Assertion> {
        self.assertion.iter().filter(|a| a.fail_on_violation).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assertion_types::{AntiPatternType, AssertionType};

    #[test]
    fn test_parse_critical_path_assertion() {
        let toml = r#"
            [[assertion]]
            name = "api_latency"
            type = "critical_path"
            max_duration_ms = 100
            fail_on_violation = true
        "#;

        let config = AssertionConfig::from_toml_str(toml).expect("test");

        assert_eq!(config.assertion.len(), 1);

        let assertion = &config.assertion[0];
        assert_eq!(assertion.name, "api_latency");
        assert!(assertion.fail_on_violation);

        match &assertion.assertion_type {
            AssertionType::CriticalPath(cp) => {
                assert_eq!(cp.max_duration_ms, 100);
            }
            _ => panic!("Expected CriticalPath assertion"),
        }
    }

    #[test]
    fn test_parse_anti_pattern_assertion() {
        let toml = r#"
            [[assertion]]
            name = "no_god_process"
            type = "anti_pattern"
            pattern = "GodProcess"
            threshold = 0.8
        "#;

        let config = AssertionConfig::from_toml_str(toml).expect("test");

        assert_eq!(config.assertion.len(), 1);

        let assertion = &config.assertion[0];
        assert_eq!(assertion.name, "no_god_process");

        match &assertion.assertion_type {
            AssertionType::AntiPattern(ap) => {
                assert_eq!(ap.pattern, AntiPatternType::GodProcess);
                assert_eq!(ap.threshold, 0.8);
            }
            _ => panic!("Expected AntiPattern assertion"),
        }
    }

    #[test]
    fn test_parse_multiple_assertions() {
        let toml = r#"
            [[assertion]]
            name = "critical_path_max_latency"
            type = "critical_path"
            max_duration_ms = 100

            [[assertion]]
            name = "no_god_process"
            type = "anti_pattern"
            pattern = "GodProcess"
            threshold = 0.8

            [[assertion]]
            name = "max_syscalls"
            type = "span_count"
            max_spans = 1000
        "#;

        let config = AssertionConfig::from_toml_str(toml).expect("test");

        assert_eq!(config.assertion.len(), 3);
        assert_eq!(config.assertion[0].name, "critical_path_max_latency");
        assert_eq!(config.assertion[1].name, "no_god_process");
        assert_eq!(config.assertion[2].name, "max_syscalls");
    }

    #[test]
    fn test_parse_disabled_assertion() {
        let toml = r#"
            [[assertion]]
            name = "disabled_check"
            type = "critical_path"
            max_duration_ms = 100
            enabled = false
        "#;

        let config = AssertionConfig::from_toml_str(toml).expect("test");

        assert_eq!(config.assertion.len(), 1);
        assert!(!config.assertion[0].enabled);

        // Should be filtered out by enabled_assertions()
        assert_eq!(config.enabled_assertions().len(), 0);
    }

    #[test]
    fn test_enabled_assertions_filter() {
        let toml = r#"
            [[assertion]]
            name = "enabled1"
            type = "critical_path"
            max_duration_ms = 100
            enabled = true

            [[assertion]]
            name = "disabled1"
            type = "critical_path"
            max_duration_ms = 100
            enabled = false

            [[assertion]]
            name = "enabled2"
            type = "critical_path"
            max_duration_ms = 100
        "#;

        let config = AssertionConfig::from_toml_str(toml).expect("test");

        assert_eq!(config.assertion.len(), 3);
        assert_eq!(config.enabled_assertions().len(), 2);
    }

    #[test]
    fn test_fail_on_violation_filter() {
        let toml = r#"
            [[assertion]]
            name = "fail1"
            type = "critical_path"
            max_duration_ms = 100
            fail_on_violation = true

            [[assertion]]
            name = "warn1"
            type = "critical_path"
            max_duration_ms = 100
            fail_on_violation = false

            [[assertion]]
            name = "fail2"
            type = "critical_path"
            max_duration_ms = 100
        "#;

        let config = AssertionConfig::from_toml_str(toml).expect("test");

        assert_eq!(config.assertion.len(), 3);
        assert_eq!(config.fail_on_violation_assertions().len(), 2);
    }

    #[test]
    fn test_parse_empty_config() {
        let toml = r"";

        let config = AssertionConfig::from_toml_str(toml).expect("parse should succeed");

        assert_eq!(config.assertion.len(), 0);
    }

    #[test]
    fn test_parse_with_trace_name_pattern() {
        let toml = r#"
            [[assertion]]
            name = "api_latency"
            type = "critical_path"
            max_duration_ms = 100
            trace_name_pattern = "api_.*"
        "#;

        let config = AssertionConfig::from_toml_str(toml).expect("test");

        match &config.assertion[0].assertion_type {
            AssertionType::CriticalPath(cp) => {
                assert_eq!(cp.trace_name_pattern, Some("api_.*".to_string()));
            }
            _ => panic!("Expected CriticalPath assertion"),
        }
    }

    #[test]
    fn test_parse_invalid_toml() {
        let toml = r#"
            [[assertion
            name = "broken"
        "#;

        let result = AssertionConfig::from_toml_str(toml);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_missing_required_field() {
        let toml = r#"
            [[assertion]]
            name = "broken"
            type = "critical_path"
            # Missing max_duration_ms
        "#;

        let result = AssertionConfig::from_toml_str(toml);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_real_world_config() {
        let toml = r#"
            # Critical path assertions
            [[assertion]]
            name = "api_max_latency"
            type = "critical_path"
            max_duration_ms = 100
            trace_name_pattern = "api_.*"
            fail_on_violation = true

            [[assertion]]
            name = "db_query_max_latency"
            type = "critical_path"
            max_duration_ms = 50
            trace_name_pattern = "db_.*"
            fail_on_violation = true

            # Anti-pattern detection
            [[assertion]]
            name = "no_god_process"
            type = "anti_pattern"
            pattern = "GodProcess"
            threshold = 0.8
            fail_on_violation = true

            [[assertion]]
            name = "no_tight_loop"
            type = "anti_pattern"
            pattern = "TightLoop"
            threshold = 0.9
            fail_on_violation = false

            # Resource constraints
            [[assertion]]
            name = "max_syscalls"
            type = "span_count"
            max_spans = 10000
            fail_on_violation = true

            [[assertion]]
            name = "max_memory"
            type = "memory_usage"
            max_bytes = 100000000
            tracking_mode = "allocations"
            fail_on_violation = true
        "#;

        let config = AssertionConfig::from_toml_str(toml).expect("test");

        assert_eq!(config.assertion.len(), 6);
        assert_eq!(config.enabled_assertions().len(), 6);
        assert_eq!(config.fail_on_violation_assertions().len(), 5);
    }
}
