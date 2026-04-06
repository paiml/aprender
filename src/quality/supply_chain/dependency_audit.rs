//! Dependency audit results and cargo-deny parsing.

use serde::{Deserialize, Serialize};

use super::{Advisory, AuditStatus, Result, Severity, SupplyChainError};

// Field name constants for cargo-deny JSON parsing (CB-525)
const FIELD_TYPE: &str = "type";
const FIELD_FIELDS: &str = "fields";
const FIELD_SEVERITY: &str = "severity";
const FIELD_CODE: &str = "code";
const FIELD_LABELS: &str = "labels";
const FIELD_SPAN: &str = "span";
const FIELD_CRATE: &str = "crate";
const FIELD_NAME: &str = "name";
const FIELD_VERSION: &str = "version";
const FIELD_MESSAGE: &str = "message";

/// Dependency audit result
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DependencyAudit {
    /// Crate name
    pub crate_name: String,

    /// Version string
    pub version: String,

    /// Security advisories affecting this crate
    pub advisories: Vec<Advisory>,

    /// License identifier (e.g., "MIT", "Apache-2.0")
    pub license: String,

    /// Overall audit status
    pub audit_status: AuditStatus,
}

impl DependencyAudit {
    /// Create a new clean dependency audit
    pub fn clean(
        crate_name: impl Into<String>,
        version: impl Into<String>,
        license: impl Into<String>,
    ) -> Self {
        Self {
            crate_name: crate_name.into(),
            version: version.into(),
            advisories: Vec::new(),
            license: license.into(),
            audit_status: AuditStatus::Clean,
        }
    }

    /// Create a vulnerable dependency audit
    pub fn vulnerable(
        crate_name: impl Into<String>,
        version: impl Into<String>,
        license: impl Into<String>,
        advisories: Vec<Advisory>,
    ) -> Self {
        Self {
            crate_name: crate_name.into(),
            version: version.into(),
            advisories,
            license: license.into(),
            audit_status: AuditStatus::Vulnerable,
        }
    }

    /// Parse dependency audits from cargo-deny JSON output
    ///
    /// # Arguments
    ///
    /// * `json` - JSON output from `cargo deny check --format json`
    ///
    /// # Example JSON format
    ///
    /// ```json
    /// {
    ///   "type": "diagnostic",
    ///   "fields": {
    ///     "graphs": [...],
    ///     "severity": "error",
    ///     "code": "A001",
    ///     "message": "Detected security vulnerability",
    ///     "labels": [{"span": {"crate": {"name": "foo", "version": "1.0.0"}}}]
    ///   }
    /// }
    /// ```
    pub fn from_cargo_deny_output(json: &str) -> Result<Vec<Self>> {
        let mut audits = Vec::new();

        // cargo deny outputs newline-delimited JSON
        for line in json.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            let value: serde_json::Value = serde_json::from_str(line)
                .map_err(|e| SupplyChainError::ParseError(e.to_string()))?;

            // Skip non-diagnostic messages
            if value.get(FIELD_TYPE).and_then(|t| t.as_str()) != Some("diagnostic") {
                continue;
            }

            let fields = match value.get(FIELD_FIELDS) {
                Some(f) => f,
                None => continue,
            };

            // Extract severity
            let severity_str =
                fields.get(FIELD_SEVERITY).and_then(|s| s.as_str()).unwrap_or("none");

            let is_vulnerability = severity_str == "error"
                && fields
                    .get(FIELD_CODE)
                    .and_then(|c| c.as_str())
                    .is_some_and(|c| c.starts_with('A'));

            if !is_vulnerability {
                continue;
            }

            // Extract crate info from labels
            if let Some(labels) = fields.get(FIELD_LABELS).and_then(|l| l.as_array()) {
                for label in labels {
                    if let Some(span) = label.get(FIELD_SPAN) {
                        if let Some(krate) = span.get(FIELD_CRATE) {
                            let crate_name = krate
                                .get(FIELD_NAME)
                                .and_then(|n| n.as_str())
                                .unwrap_or("unknown")
                                .to_string();

                            let version = krate
                                .get(FIELD_VERSION)
                                .and_then(|v| v.as_str())
                                .unwrap_or("unknown")
                                .to_string();

                            let message = fields
                                .get(FIELD_MESSAGE)
                                .and_then(|m| m.as_str())
                                .unwrap_or("Unknown vulnerability")
                                .to_string();

                            let code = fields
                                .get(FIELD_CODE)
                                .and_then(|c| c.as_str())
                                .unwrap_or("UNKNOWN")
                                .to_string();

                            let advisory = Advisory::new(code, Severity::High, message);

                            audits.push(Self::vulnerable(
                                crate_name,
                                version,
                                "unknown",
                                vec![advisory],
                            ));
                        }
                    }
                }
            }
        }

        Ok(audits)
    }

    /// Returns true if this dependency has vulnerabilities
    pub fn is_vulnerable(&self) -> bool {
        self.audit_status == AuditStatus::Vulnerable
    }

    /// Returns the highest severity advisory
    pub fn max_severity(&self) -> Severity {
        self.advisories.iter().map(|a| a.severity).max().unwrap_or(Severity::None)
    }
}
