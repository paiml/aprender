//! Evidence Export for Oracle Integration (PMAT-261)
//!
//! This module provides the structured evidence export format consumed by
//! aprender's `apr oracle` CLI for certification status lookup.
//!
//! # Theoretical Foundation
//!
//! - **Reproducibility (Hamming, 1962)**: Same evidence → same MQS score
//! - **Contract Programming (Meyer, 1992)**: Schema defines oracle expectations
//! - **Defensive Programming (Hunt & Thomas, 1999)**: Handle missing/malformed data
//!
//! # JSON Schema
//!
//! The export format follows the apr-qa-evidence.schema.json specification:
//! - Model metadata for identification
//! - Playbook metadata for tier/version tracking
//! - Summary statistics for quick lookup
//! - MQS score and breakdown for certification
//! - Gateway results for compliance checking
//! - Full evidence array for reproducibility

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Model metadata in the evidence export.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMeta {
    /// HuggingFace repository ID (e.g., "Qwen/Qwen2.5-Coder-0.5B-Instruct")
    pub hf_repo: String,
    /// Model family (e.g., "qwen2")
    pub family: String,
    /// Size variant (e.g., "0.5b")
    pub size: String,
    /// Ground truth format (e.g., "safetensors")
    pub format: String,
}

/// Playbook metadata in the evidence export.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlaybookMeta {
    /// Playbook name (e.g., "qwen2.5-coder-0.5b-mvp")
    pub name: String,
    /// Playbook version
    pub version: String,
    /// Certification tier (smoke, mvp, full)
    pub tier: String,
}

/// Summary statistics for the evidence export.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportSummary {
    /// Total number of scenarios
    pub total_scenarios: usize,
    /// Number of passed scenarios
    pub passed: usize,
    /// Number of failed scenarios
    pub failed: usize,
    /// Number of skipped scenarios
    pub skipped: usize,
    /// Pass rate (0.0 - 1.0)
    pub pass_rate: f64,
    /// Total duration in milliseconds
    pub duration_ms: u64,
    /// Run timestamp
    pub timestamp: DateTime<Utc>,
}

/// MQS (Model Qualification Score) in the evidence export.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MqsExport {
    /// Raw MQS score (0-1000)
    pub score: u32,
    /// Letter grade (A, B, C, D, F)
    pub grade: String,
    /// Whether all gateways passed
    pub gateway_passed: bool,
    /// Category score breakdown
    pub category_scores: HashMap<String, u32>,
}

/// Gate result in the evidence export.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateResult {
    /// Whether the gate passed
    pub passed: bool,
    /// Human-readable reason
    pub reason: String,
}

/// Complete evidence export structure for oracle consumption.
///
/// This structure is serialized to JSON and consumed by:
/// - `apr oracle` for certification status lookup
/// - CI/CD pipelines for quality gates
/// - Audit trails for reproducibility
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceExport {
    /// JSON Schema URL
    #[serde(rename = "$schema")]
    pub schema: String,
    /// Format version
    pub version: String,
    /// Model metadata
    pub model: ModelMeta,
    /// Playbook metadata
    pub playbook: PlaybookMeta,
    /// Summary statistics
    pub summary: ExportSummary,
    /// MQS score and breakdown
    pub mqs: MqsExport,
    /// Gateway/gate results
    pub gates: HashMap<String, GateResult>,
    /// Full evidence array
    pub evidence: Vec<serde_json::Value>,
}

impl Default for EvidenceExport {
    /// Create a default evidence export with empty fields and "F" grade
    fn default() -> Self {
        Self {
            schema: "https://paiml.com/schemas/apr-qa-evidence.schema.json".to_string(),
            version: "1.0.0".to_string(),
            model: ModelMeta {
                hf_repo: String::new(),
                family: String::new(),
                size: String::new(),
                format: "safetensors".to_string(),
            },
            playbook: PlaybookMeta {
                name: String::new(),
                version: "1.0.0".to_string(),
                tier: "mvp".to_string(),
            },
            summary: ExportSummary {
                total_scenarios: 0,
                passed: 0,
                failed: 0,
                skipped: 0,
                pass_rate: 0.0,
                duration_ms: 0,
                timestamp: Utc::now(),
            },
            mqs: MqsExport {
                score: 0,
                grade: "F".to_string(),
                gateway_passed: false,
                category_scores: HashMap::new(),
            },
            gates: HashMap::new(),
            evidence: Vec::new(),
        }
    }
}

/// Core operations for evidence export creation, serialization, and analysis
impl EvidenceExport {
    /// Create a new evidence export builder.
    #[must_use]
    pub fn builder() -> EvidenceExportBuilder {
        EvidenceExportBuilder::default()
    }

    /// Create an EvidenceExport from MqsScore and evidence.
    ///
    /// This method bridges the internal MQS calculation with the external
    /// evidence export format consumed by the oracle.
    ///
    /// # Arguments
    ///
    /// * `mqs` - The calculated MQS score
    /// * `evidence` - Raw evidence as serialized JSON values
    /// * `model` - Model metadata
    /// * `playbook` - Playbook metadata
    #[must_use]
    pub fn from_mqs_score(
        mqs: &crate::mqs::MqsScore,
        evidence: Vec<serde_json::Value>,
        model: ModelMeta,
        playbook: PlaybookMeta,
    ) -> Self {
        let total_scenarios = mqs.total_tests;
        let failed = mqs.tests_failed;

        // Count skipped from evidence (MqsScore.tests_passed includes Skipped
        // via is_pass(), so we must separate them to avoid always-zero skipped)
        let skipped = evidence
            .iter()
            .filter(|e| e.get("outcome").and_then(|o| o.as_str()) == Some("Skipped"))
            .count();
        let passed = mqs.tests_passed.saturating_sub(skipped);

        let pass_rate = if total_scenarios > 0 {
            passed as f64 / total_scenarios as f64
        } else {
            0.0
        };

        // Calculate total duration from evidence metrics
        let duration_ms = evidence
            .iter()
            .filter_map(|e| e.get("metrics").and_then(|m| m.get("duration_ms")))
            .filter_map(serde_json::Value::as_u64)
            .sum();

        // Build category scores from MQS categories
        let mut category_scores = HashMap::new();
        let breakdown = mqs.categories.breakdown();
        for (cat, (score, _max)) in breakdown {
            category_scores.insert(cat.to_lowercase(), score);
        }

        // Build gate results from MQS gateways
        let mut gates = HashMap::new();
        for gateway in &mqs.gateways {
            gates.insert(
                match gateway.id.as_str() {
                    "G0" => "G0-INTEGRITY".to_string(),
                    "G1" => "G1-MODEL-LOADS".to_string(),
                    "G2" => "G2-BASIC-INFERENCE".to_string(),
                    "G3" => "G3-NO-CRASHES".to_string(),
                    "G4" => "G4-OUTPUT-QUALITY".to_string(),
                    other => other.to_string(),
                },
                GateResult {
                    passed: gateway.passed,
                    reason: gateway
                        .failure_reason
                        .clone()
                        .unwrap_or_else(|| gateway.description.clone()),
                },
            );
        }

        Self {
            schema: "https://paiml.com/schemas/apr-qa-evidence.schema.json".to_string(),
            version: "1.0.0".to_string(),
            model,
            playbook,
            summary: ExportSummary {
                total_scenarios,
                passed,
                failed,
                skipped,
                pass_rate,
                duration_ms,
                timestamp: Utc::now(),
            },
            mqs: MqsExport {
                score: mqs.raw_score,
                grade: mqs.grade.clone(),
                gateway_passed: mqs.gateways_passed,
                category_scores,
            },
            gates,
            evidence,
        }
    }

    /// Serialize to JSON string.
    ///
    /// # Errors
    ///
    /// Returns an error if serialization fails.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Deserialize from JSON string.
    ///
    /// # Errors
    ///
    /// Returns an error if deserialization fails.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Calculate pass rate from summary.
    #[must_use]
    pub fn calculate_pass_rate(&self) -> f64 {
        if self.summary.total_scenarios == 0 {
            0.0
        } else {
            self.summary.passed as f64 / self.summary.total_scenarios as f64
        }
    }

    /// Check if all mandatory gateways passed.
    ///
    /// All 5 gateways (G0-G4) are mandatory. Any single failure zeros MQS
    /// per the gateway zeroing invariant.
    #[must_use]
    pub fn all_gateways_passed(&self) -> bool {
        let mandatory = [
            "G0-INTEGRITY",
            "G1-MODEL-LOADS",
            "G2-BASIC-INFERENCE",
            "G3-NO-CRASHES",
            "G4-OUTPUT-QUALITY",
        ];
        mandatory
            .iter()
            .all(|gate| self.gates.get(*gate).is_some_and(|result| result.passed))
    }

    /// Derive certification status from MQS and gateways.
    ///
    /// UNTESTED requires both zero score AND no evidence collected.
    /// A model with evidence but score 0 was tested and failed (BLOCKED).
    #[must_use]
    pub fn derive_status(&self) -> &'static str {
        if self.mqs.score >= 800 && self.mqs.gateway_passed {
            "CERTIFIED"
        } else if self.mqs.score == 0 && self.evidence.is_empty() {
            "UNTESTED"
        } else {
            "BLOCKED"
        }
    }
}

/// Builder for `EvidenceExport`.
#[derive(Debug, Clone, Default)]
pub struct EvidenceExportBuilder {
    /// The evidence export being constructed
    export: EvidenceExport,
}

/// Builder methods for constructing evidence exports step by step
impl EvidenceExportBuilder {
    /// Set model metadata.
    #[must_use]
    pub fn model(
        mut self,
        hf_repo: impl Into<String>,
        family: impl Into<String>,
        size: impl Into<String>,
    ) -> Self {
        self.export.model.hf_repo = hf_repo.into();
        self.export.model.family = family.into();
        self.export.model.size = size.into();
        self
    }

    /// Set model format.
    #[must_use]
    pub fn format(mut self, format: impl Into<String>) -> Self {
        self.export.model.format = format.into();
        self
    }

    /// Set playbook metadata.
    #[must_use]
    pub fn playbook(
        mut self,
        name: impl Into<String>,
        version: impl Into<String>,
        tier: impl Into<String>,
    ) -> Self {
        self.export.playbook.name = name.into();
        self.export.playbook.version = version.into();
        self.export.playbook.tier = tier.into();
        self
    }

    /// Set summary statistics.
    #[must_use]
    pub fn summary(
        mut self,
        total: usize,
        passed: usize,
        failed: usize,
        skipped: usize,
        duration_ms: u64,
    ) -> Self {
        self.export.summary.total_scenarios = total;
        self.export.summary.passed = passed;
        self.export.summary.failed = failed;
        self.export.summary.skipped = skipped;
        self.export.summary.pass_rate = if total > 0 {
            passed as f64 / total as f64
        } else {
            0.0
        };
        self.export.summary.duration_ms = duration_ms;
        self.export.summary.timestamp = Utc::now();
        self
    }

    /// Set MQS score.
    #[must_use]
    pub fn mqs(mut self, score: u32, grade: impl Into<String>, gateway_passed: bool) -> Self {
        self.export.mqs.score = score;
        self.export.mqs.grade = grade.into();
        self.export.mqs.gateway_passed = gateway_passed;
        self
    }

    /// Add a category score.
    #[must_use]
    pub fn category_score(mut self, category: impl Into<String>, score: u32) -> Self {
        self.export
            .mqs
            .category_scores
            .insert(category.into(), score);
        self
    }

    /// Add a gate result.
    #[must_use]
    pub fn gate(
        mut self,
        gate_id: impl Into<String>,
        passed: bool,
        reason: impl Into<String>,
    ) -> Self {
        self.export.gates.insert(
            gate_id.into(),
            GateResult {
                passed,
                reason: reason.into(),
            },
        );
        self
    }

    /// Add evidence items.
    #[must_use]
    pub fn evidence(mut self, evidence: Vec<serde_json::Value>) -> Self {
        self.export.evidence = evidence;
        self
    }

    /// Build the export.
    #[must_use]
    pub fn build(self) -> EvidenceExport {
        self.export
    }
}

#[cfg(test)]
#[path = "evidence_export_tests.rs"]
mod evidence_export_tests;
