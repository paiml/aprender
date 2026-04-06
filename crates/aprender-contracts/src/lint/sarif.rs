//! SARIF v2.1.0 output for `pv lint`.
//!
//! Converts `LintReport` findings into OASIS SARIF format for
//! GitHub Code Scanning, VS Code SARIF Viewer, and CI toolchains.
//!
//! Spec: `docs/specifications/sub/lint.md` Section 2
//! Reference: OASIS SARIF v2.1.0 (2020), arXiv:2403.05986

use serde::Serialize;

use super::finding::LintFinding;
use super::rules::{RULES, RuleSeverity};

const SARIF_SCHEMA: &str =
    "https://docs.oasis-open.org/sarif/sarif/v2.1.0/errata01/os/schemas/sarif-schema-2.1.0.json";
const SARIF_VERSION: &str = "2.1.0";
const TOOL_NAME: &str = "pv-lint";
const TOOL_URI: &str = "https://github.com/paiml/provable-contracts";

/// Top-level SARIF log.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SarifLog {
    #[serde(rename = "$schema")]
    pub schema: String,
    pub version: String,
    pub runs: Vec<SarifRun>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SarifRun {
    pub tool: SarifTool,
    pub results: Vec<SarifResult>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SarifTool {
    pub driver: SarifDriver,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SarifDriver {
    pub name: String,
    pub version: String,
    pub information_uri: String,
    pub rules: Vec<SarifRuleDescriptor>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SarifRuleDescriptor {
    pub id: String,
    pub short_description: SarifMessage,
    pub default_configuration: SarifConfiguration,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SarifConfiguration {
    pub level: String,
}

#[derive(Debug, Serialize)]
pub struct SarifMessage {
    pub text: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SarifResult {
    pub rule_id: String,
    pub level: String,
    pub message: SarifMessage,
    pub locations: Vec<SarifLocation>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub suppressions: Vec<SarifSuppression>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SarifLocation {
    pub physical_location: SarifPhysicalLocation,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SarifPhysicalLocation {
    pub artifact_location: SarifArtifactLocation,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub region: Option<SarifRegion>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SarifArtifactLocation {
    pub uri: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SarifRegion {
    pub start_line: u32,
    pub start_column: u32,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SarifSuppression {
    pub kind: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub justification: Option<String>,
}

/// Build rule descriptors from the static rule catalog.
fn build_rule_descriptors() -> Vec<SarifRuleDescriptor> {
    RULES
        .iter()
        .map(|r| SarifRuleDescriptor {
            id: r.id.to_string(),
            short_description: SarifMessage {
                text: r.description.to_string(),
            },
            default_configuration: SarifConfiguration {
                level: r.default_severity.sarif_level().to_string(),
            },
        })
        .collect()
}

/// Convert a list of lint findings to a SARIF log.
pub fn findings_to_sarif(findings: &[LintFinding], tool_version: &str) -> SarifLog {
    let results: Vec<SarifResult> = findings
        .iter()
        .filter(|f| f.severity != RuleSeverity::Off)
        .map(|f| {
            let suppressions = if f.suppressed {
                vec![SarifSuppression {
                    kind: "inSource".to_string(),
                    justification: f.suppression_reason.clone(),
                }]
            } else {
                vec![]
            };
            SarifResult {
                rule_id: f.rule_id.clone(),
                level: f.severity.sarif_level().to_string(),
                message: SarifMessage {
                    text: f.message.clone(),
                },
                locations: vec![SarifLocation {
                    physical_location: SarifPhysicalLocation {
                        artifact_location: SarifArtifactLocation {
                            uri: f.file.clone(),
                        },
                        region: Some(SarifRegion {
                            start_line: f.line.unwrap_or(1),
                            start_column: 1,
                        }),
                    },
                }],
                suppressions,
            }
        })
        .collect();

    SarifLog {
        schema: SARIF_SCHEMA.to_string(),
        version: SARIF_VERSION.to_string(),
        runs: vec![SarifRun {
            tool: SarifTool {
                driver: SarifDriver {
                    name: TOOL_NAME.to_string(),
                    version: tool_version.to_string(),
                    information_uri: TOOL_URI.to_string(),
                    rules: build_rule_descriptors(),
                },
            },
            results,
        }],
    }
}

/// Serialize SARIF log to JSON string.
pub fn sarif_to_json(log: &SarifLog, pretty: bool) -> String {
    if pretty {
        serde_json::to_string_pretty(log).unwrap_or_default()
    } else {
        serde_json::to_string(log).unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lint::finding::LintFinding;
    use crate::lint::rules::RuleSeverity;

    fn sample_finding() -> LintFinding {
        LintFinding {
            rule_id: "PV-VAL-001".into(),
            severity: RuleSeverity::Error,
            message: "Missing proof_obligations".into(),
            file: "contracts/example-v1.yaml".into(),
            line: Some(1),
            contract_stem: Some("example-v1".into()),
            suppressed: false,
            suppression_reason: None,
            is_new: false,
            snippet: None,
            suggestion: None,
            evidence: None,
        }
    }

    #[test]
    fn sarif_log_has_schema() {
        let log = findings_to_sarif(&[sample_finding()], "0.1.0");
        assert!(log.schema.contains("sarif-schema-2.1.0"));
        assert_eq!(log.version, "2.1.0");
    }

    #[test]
    fn sarif_log_has_tool_info() {
        let log = findings_to_sarif(&[], "0.2.0");
        assert_eq!(log.runs.len(), 1);
        assert_eq!(log.runs[0].tool.driver.name, "pv-lint");
        assert_eq!(log.runs[0].tool.driver.version, "0.2.0");
    }

    #[test]
    fn sarif_log_has_rules() {
        let log = findings_to_sarif(&[], "0.1.0");
        assert!(!log.runs[0].tool.driver.rules.is_empty());
        let rule_ids: Vec<&str> = log.runs[0]
            .tool
            .driver
            .rules
            .iter()
            .map(|r| r.id.as_str())
            .collect();
        assert!(rule_ids.contains(&"PV-VAL-001"));
        assert!(rule_ids.contains(&"PV-PRV-001"));
    }

    #[test]
    fn sarif_result_maps_finding() {
        let log = findings_to_sarif(&[sample_finding()], "0.1.0");
        assert_eq!(log.runs[0].results.len(), 1);
        let result = &log.runs[0].results[0];
        assert_eq!(result.rule_id, "PV-VAL-001");
        assert_eq!(result.level, "error");
        assert!(result.message.text.contains("Missing proof_obligations"));
        assert_eq!(
            result.locations[0].physical_location.artifact_location.uri,
            "contracts/example-v1.yaml"
        );
    }

    #[test]
    fn sarif_suppressed_finding() {
        let mut f = sample_finding();
        f.suppressed = true;
        f.suppression_reason = Some("Known gap".into());
        let log = findings_to_sarif(&[f], "0.1.0");
        let result = &log.runs[0].results[0];
        assert_eq!(result.suppressions.len(), 1);
        assert_eq!(result.suppressions[0].kind, "inSource");
        assert_eq!(
            result.suppressions[0].justification.as_deref(),
            Some("Known gap")
        );
    }

    #[test]
    fn sarif_off_severity_filtered() {
        let mut f = sample_finding();
        f.severity = RuleSeverity::Off;
        let log = findings_to_sarif(&[f], "0.1.0");
        assert!(log.runs[0].results.is_empty());
    }

    #[test]
    fn sarif_to_json_pretty() {
        let log = findings_to_sarif(&[sample_finding()], "0.1.0");
        let json = sarif_to_json(&log, true);
        assert!(json.contains('\n'));
        assert!(json.contains("$schema"));
    }

    #[test]
    fn sarif_to_json_compact() {
        let log = findings_to_sarif(&[sample_finding()], "0.1.0");
        let json = sarif_to_json(&log, false);
        assert!(!json.contains('\n'));
        assert!(json.contains("$schema"));
    }

    #[test]
    fn sarif_valid_json() {
        let log = findings_to_sarif(&[sample_finding()], "0.1.0");
        let json = sarif_to_json(&log, true);
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["version"], "2.1.0");
        assert!(parsed["runs"].is_array());
    }

    #[test]
    fn sarif_multiple_findings() {
        let mut f2 = sample_finding();
        f2.rule_id = "PV-AUD-001".into();
        f2.severity = RuleSeverity::Warning;
        f2.message = "Obligation without test".into();
        let log = findings_to_sarif(&[sample_finding(), f2], "0.1.0");
        assert_eq!(log.runs[0].results.len(), 2);
    }
}
