//! Proof status reader — deserializes `pv proof-status --format json` output.
//!
//! Provides mirror types for the JSON bridge from provable-contracts and
//! a `ProofBonus` derivation function for MQS scoring integration.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

// ── Mirror types (deserialization only) ───────────────────────────

/// Top-level proof status report from `pv proof-status --format json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExternalProofStatus {
    /// Schema version (e.g., "1.0.0").
    pub schema_version: String,
    /// Generation timestamp.
    pub timestamp: String,
    /// Per-contract proof status entries.
    pub contracts: Vec<ExternalContractStatus>,
    /// Kernel equivalence class summaries.
    pub kernel_classes: Vec<ExternalKernelClass>,
    /// Aggregate totals.
    pub totals: ExternalTotals,
}

/// Proof status for a single contract.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExternalContractStatus {
    /// Contract stem (e.g., "softmax-kernel-v1").
    pub stem: String,
    /// Proof level: "L1" through "L5".
    pub proof_level: String,
    /// Total proof obligations.
    pub obligations: u32,
    /// Falsification test count.
    pub falsification_tests: u32,
    /// Kani harness count.
    pub kani_harnesses: u32,
    /// Lean 4 proved obligation count.
    pub lean_proved: u32,
    /// Bindings implemented count.
    pub bindings_implemented: u32,
    /// Total bindings count.
    pub bindings_total: u32,
}

/// Summary for a kernel equivalence class.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExternalKernelClass {
    /// Class label (A, B, C, D, E).
    pub label: String,
    /// Description (e.g., "GQA+RMSNorm+SiLU+SwiGLU+RoPE").
    pub description: String,
    /// Contract stems in this class.
    pub contract_stems: Vec<String>,
    /// Minimum proof level across all contracts in this class.
    pub min_proof_level: String,
    /// Whether all contracts have complete bindings.
    pub all_bound: bool,
}

/// Aggregate totals from the report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExternalTotals {
    /// Total contracts analyzed.
    pub contracts: u32,
    /// Total proof obligations.
    pub obligations: u32,
    /// Total falsification tests.
    pub falsification_tests: u32,
    /// Total Kani harnesses.
    pub kani_harnesses: u32,
    /// Total Lean-proved obligations.
    pub lean_proved: u32,
    /// Total bindings implemented.
    pub bindings_implemented: u32,
    /// Total bindings.
    pub bindings_total: u32,
}

// ── Proof bonus for MQS ──────────────────────────────────────────

/// Bonus points awarded for kernel proof level.
///
/// Added to MQS raw score before normalization.
/// Gateway failures still zero everything — bonus cannot override.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProofBonus {
    /// Kernel equivalence class (A, B, C, D, E).
    pub kernel_class: Option<String>,
    /// Proof level string (L1–L5).
    pub proof_level: Option<String>,
    /// Bonus points (0, 10, 25, 40, or 50).
    pub bonus_points: u32,
}

/// Maximum bonus points (for L5).
pub const MAX_PROOF_BONUS: u32 = 50;

// ── Public API ───────────────────────────────────────────────────

/// Read a proof status JSON file.
///
/// Discovery order:
/// 1. `PROVABLE_CONTRACTS_PROOF_STATUS` environment variable
/// 2. Explicit `path` argument
/// 3. Fallback to `../provable-contracts/proof-status.json`
///
/// # Errors
///
/// Returns an error if the file cannot be read or parsed.
pub fn read_proof_status(path: Option<&Path>) -> Result<ExternalProofStatus> {
    let resolved = if let Ok(env_path) = std::env::var("PROVABLE_CONTRACTS_PROOF_STATUS") {
        std::path::PathBuf::from(env_path)
    } else if let Some(p) = path {
        p.to_path_buf()
    } else {
        std::path::PathBuf::from("../provable-contracts/proof-status.json")
    };

    let content = std::fs::read_to_string(&resolved).map_err(|e| {
        Error::Io(format!(
            "Failed to read proof status from {}: {e}",
            resolved.display()
        ))
    })?;

    let status: ExternalProofStatus = serde_json::from_str(&content).map_err(|e| {
        Error::Validation(format!(
            "Failed to parse proof status JSON from {}: {e}",
            resolved.display()
        ))
    })?;

    Ok(status)
}

/// Derive a proof bonus for a kernel equivalence class.
///
/// Looks up the class by label in the proof status report and converts
/// its minimum proof level to bonus points.
///
/// | Level | Points |
/// |-------|--------|
/// | L1    | 0      |
/// | L2    | 10     |
/// | L3    | 25     |
/// | L4    | 40     |
/// | L5    | 50     |
#[must_use]
pub fn proof_bonus_for_class(status: &ExternalProofStatus, class: &str) -> ProofBonus {
    let kc = status.kernel_classes.iter().find(|kc| kc.label == class);

    match kc {
        Some(kc) => {
            let points = level_to_bonus(&kc.min_proof_level);
            ProofBonus {
                kernel_class: Some(kc.label.clone()),
                proof_level: Some(kc.min_proof_level.clone()),
                bonus_points: points,
            }
        }
        None => ProofBonus::default(),
    }
}

/// Convert a proof level string to bonus points.
#[must_use]
pub fn level_to_bonus(level: &str) -> u32 {
    match level {
        "L1" => 0,
        "L2" => 10,
        "L3" => 25,
        "L4" => 40,
        "L5" => 50,
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a sample proof status report for test assertions
    fn sample_report() -> ExternalProofStatus {
        ExternalProofStatus {
            schema_version: "1.0.0".to_string(),
            timestamp: "1234567890Z".to_string(),
            contracts: vec![
                ExternalContractStatus {
                    stem: "softmax-kernel-v1".to_string(),
                    proof_level: "L3".to_string(),
                    obligations: 6,
                    falsification_tests: 6,
                    kani_harnesses: 3,
                    lean_proved: 0,
                    bindings_implemented: 1,
                    bindings_total: 1,
                },
                ExternalContractStatus {
                    stem: "rmsnorm-kernel-v1".to_string(),
                    proof_level: "L3".to_string(),
                    obligations: 5,
                    falsification_tests: 5,
                    kani_harnesses: 2,
                    lean_proved: 0,
                    bindings_implemented: 1,
                    bindings_total: 1,
                },
            ],
            kernel_classes: vec![
                ExternalKernelClass {
                    label: "A".to_string(),
                    description: "GQA+RMSNorm+SiLU+SwiGLU+RoPE".to_string(),
                    contract_stems: vec![
                        "softmax-kernel-v1".to_string(),
                        "rmsnorm-kernel-v1".to_string(),
                    ],
                    min_proof_level: "L3".to_string(),
                    all_bound: true,
                },
                ExternalKernelClass {
                    label: "B".to_string(),
                    description: "MHA+LayerNorm+GELU+AbsPos".to_string(),
                    contract_stems: vec!["softmax-kernel-v1".to_string()],
                    min_proof_level: "L2".to_string(),
                    all_bound: false,
                },
            ],
            totals: ExternalTotals {
                contracts: 2,
                obligations: 11,
                falsification_tests: 11,
                kani_harnesses: 5,
                lean_proved: 0,
                bindings_implemented: 2,
                bindings_total: 2,
            },
        }
    }

    #[test]
    fn test_level_to_bonus() {
        assert_eq!(level_to_bonus("L1"), 0);
        assert_eq!(level_to_bonus("L2"), 10);
        assert_eq!(level_to_bonus("L3"), 25);
        assert_eq!(level_to_bonus("L4"), 40);
        assert_eq!(level_to_bonus("L5"), 50);
        assert_eq!(level_to_bonus("unknown"), 0);
    }

    #[test]
    fn test_proof_bonus_for_class_found() {
        let report = sample_report();
        let bonus = proof_bonus_for_class(&report, "A");
        assert_eq!(bonus.kernel_class, Some("A".to_string()));
        assert_eq!(bonus.proof_level, Some("L3".to_string()));
        assert_eq!(bonus.bonus_points, 25);
    }

    #[test]
    fn test_proof_bonus_for_class_not_found() {
        let report = sample_report();
        let bonus = proof_bonus_for_class(&report, "Z");
        assert!(bonus.kernel_class.is_none());
        assert!(bonus.proof_level.is_none());
        assert_eq!(bonus.bonus_points, 0);
    }

    #[test]
    fn test_proof_bonus_for_class_b() {
        let report = sample_report();
        let bonus = proof_bonus_for_class(&report, "B");
        assert_eq!(bonus.bonus_points, 10); // L2 = 10 points
    }

    #[test]
    fn test_proof_bonus_default() {
        let bonus = ProofBonus::default();
        assert!(bonus.kernel_class.is_none());
        assert!(bonus.proof_level.is_none());
        assert_eq!(bonus.bonus_points, 0);
    }

    #[test]
    fn test_json_roundtrip() {
        let report = sample_report();
        let json = serde_json::to_string(&report).unwrap();
        let parsed: ExternalProofStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.contracts.len(), 2);
        assert_eq!(parsed.kernel_classes.len(), 2);
        assert_eq!(parsed.totals.contracts, 2);
    }

    #[test]
    fn test_read_proof_status_missing_file() {
        let result = read_proof_status(Some(Path::new("/nonexistent/proof-status.json")));
        assert!(result.is_err());
    }

    #[test]
    fn test_read_proof_status_fallback_path() {
        // Without env var, explicit path takes precedence
        let result = read_proof_status(Some(Path::new("/tmp/nonexistent-proof-status.json")));
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("nonexistent-proof-status"));
    }

    #[test]
    fn test_read_proof_status_valid_json() {
        let report = sample_report();
        let json = serde_json::to_string_pretty(&report).unwrap();

        let temp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(temp.path(), &json).unwrap();

        let parsed = read_proof_status(Some(temp.path())).unwrap();
        assert_eq!(parsed.schema_version, "1.0.0");
        assert_eq!(parsed.contracts.len(), 2);
    }

    #[test]
    fn test_external_contract_status_fields() {
        let status = ExternalContractStatus {
            stem: "test-v1".to_string(),
            proof_level: "L3".to_string(),
            obligations: 5,
            falsification_tests: 5,
            kani_harnesses: 3,
            lean_proved: 0,
            bindings_implemented: 1,
            bindings_total: 1,
        };
        assert_eq!(status.stem, "test-v1");
        assert_eq!(status.proof_level, "L3");
    }

    #[test]
    fn test_max_proof_bonus_constant() {
        assert_eq!(MAX_PROOF_BONUS, 50);
    }

    #[test]
    fn test_proof_bonus_serialize() {
        let bonus = ProofBonus {
            kernel_class: Some("A".to_string()),
            proof_level: Some("L3".to_string()),
            bonus_points: 25,
        };
        let json = serde_json::to_string(&bonus).unwrap();
        assert!(json.contains("\"bonus_points\":25"));
        let parsed: ProofBonus = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.bonus_points, 25);
    }

    #[test]
    fn test_read_proof_status_invalid_json() {
        let temp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(temp.path(), "not valid json {{").unwrap();

        let result = read_proof_status(Some(temp.path()));
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("parse"));
    }
}
