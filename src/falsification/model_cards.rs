//! Section 8: Model Cards & Auditability (MA-01 to MA-10)
//!
//! Governance artifacts for ML models and datasets.
//!
//! # TPS Principles
//!
//! - **Governance documentation**: Model Cards, Datasheets
//! - **Genchi Genbutsu**: Verify claims match behavior
//! - **Kaizen**: Learning from incidents

use super::types::{CheckItem, Evidence, EvidenceType, Severity};
use std::path::Path;
use std::time::Instant;

/// Evaluate all Model Cards & Auditability checks.
pub fn evaluate_all(project_path: &Path) -> Vec<CheckItem> {
    vec![
        check_model_card_completeness(project_path),
        check_datasheet_completeness(project_path),
        check_model_card_accuracy(project_path),
        check_decision_logging(project_path),
        check_provenance_chain(project_path),
        check_version_tracking(project_path),
        check_rollback_capability(project_path),
        check_ab_test_logging(project_path),
        check_bias_audit(project_path),
        check_incident_response(project_path),
    ]
}

/// MA-01: Model Card Completeness
pub fn check_model_card_completeness(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item =
        CheckItem::new("MA-01", "Model Card Completeness", "Every model has complete Model Card")
            .with_severity(Severity::Major)
            .with_tps("Governance documentation");

    let model_card_paths = [
        project_path.join("MODEL_CARD.md"),
        project_path.join("docs/model_card.md"),
        project_path.join("docs/MODEL_CARD.md"),
    ];

    let has_model_card = model_card_paths.iter().any(|p| p.exists());
    let has_model_cards_dir = project_path.join("docs/model_cards/").exists();

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!("Model card: exists={}, dir={}", has_model_card, has_model_cards_dir),
        data: None,
        files: Vec::new(),
    });

    let has_models = check_for_pattern(project_path, &["model", "Model", "train", "predict"]);
    if !has_models || has_model_card || has_model_cards_dir {
        item = item.pass();
    } else {
        item = item.partial("Models without Model Card documentation");
    }

    item.finish_timed(start)
}

/// MA-02: Datasheet Completeness
pub fn check_datasheet_completeness(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new("MA-02", "Datasheet Completeness", "Every dataset has Datasheet")
        .with_severity(Severity::Major)
        .with_tps("Data governance");

    let has_datasheet = project_path.join("docs/datasheets/").exists()
        || project_path.join("DATASHEET.md").exists();

    let has_data_docs = check_for_pattern(project_path, &["datasheet", "Datasheet", "data_card"]);

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!("Datasheet: exists={}, docs={}", has_datasheet, has_data_docs),
        data: None,
        files: Vec::new(),
    });

    let uses_data = check_for_pattern(project_path, &["Dataset", "DataLoader", "load_data"]);
    if !uses_data || has_datasheet || has_data_docs {
        item = item.pass();
    } else {
        item = item.partial("Datasets without Datasheet documentation");
    }

    item.finish_timed(start)
}

/// MA-03: Model Card Accuracy
pub fn check_model_card_accuracy(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item =
        CheckItem::new("MA-03", "Model Card Accuracy", "Model Card reflects current behavior")
            .with_severity(Severity::Major)
            .with_tps("Genchi Genbutsu — verify claims");

    let has_drift_detection =
        check_for_pattern(project_path, &["drift", "model_drift", "behavior_test"]);
    let has_validation =
        check_for_pattern(project_path, &["validate_card", "card_accuracy", "claim_verification"]);

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!(
            "Card accuracy: drift={}, validation={}",
            has_drift_detection, has_validation
        ),
        data: None,
        files: Vec::new(),
    });

    if has_drift_detection || has_validation {
        item = item.pass();
    } else {
        item = item.partial("No model card accuracy verification");
    }

    item.finish_timed(start)
}

/// MA-04: Decision Logging Completeness
pub fn check_decision_logging(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "MA-04",
        "Decision Logging Completeness",
        "Model decisions logged with context",
    )
    .with_severity(Severity::Major)
    .with_tps("Auditability requirement");

    let has_logging =
        check_for_pattern(project_path, &["decision_log", "prediction_log", "audit_log"]);
    let has_context =
        check_for_pattern(project_path, &["input_hash", "timestamp", "model_version"]);

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!("Decision logging: logging={}, context={}", has_logging, has_context),
        data: None,
        files: Vec::new(),
    });

    let does_inference = check_for_pattern(project_path, &["inference", "predict", "forward"]);
    if !does_inference || (has_logging && has_context) {
        item = item.pass();
    } else if has_logging {
        item = item.partial("Decision logging (verify context)");
    } else {
        item = item.partial("Inference without decision logging");
    }

    item.finish_timed(start)
}

/// MA-05: Provenance Chain Completeness
pub fn check_provenance_chain(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "MA-05",
        "Provenance Chain Completeness",
        "Full provenance from data to prediction",
    )
    .with_severity(Severity::Major)
    .with_tps("Audit trail integrity");

    let has_provenance = check_for_pattern(project_path, &["provenance", "lineage", "trace"]);
    let has_data_lineage = check_for_pattern(project_path, &["data_lineage", "training_lineage"]);

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!(
            "Provenance: tracking={}, lineage={}",
            has_provenance, has_data_lineage
        ),
        data: None,
        files: Vec::new(),
    });

    if has_provenance && has_data_lineage {
        item = item.pass();
    } else if has_provenance {
        item = item.partial("Provenance tracking (verify completeness)");
    } else {
        item = item.partial("No provenance chain tracking");
    }

    item.finish_timed(start)
}

/// MA-06: Version Tracking
pub fn check_version_tracking(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item =
        CheckItem::new("MA-06", "Version Tracking", "All model versions uniquely identified")
            .with_severity(Severity::Major)
            .with_tps("Configuration management");

    let has_versioning =
        check_for_pattern(project_path, &["version", "Version", "model_id", "hash"]);
    let has_registry = check_for_pattern(project_path, &["registry", "Registry", "model_store"]);

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!(
            "Version tracking: versioning={}, registry={}",
            has_versioning, has_registry
        ),
        data: None,
        files: Vec::new(),
    });

    let has_models = check_for_pattern(project_path, &["save_model", "load_model", "Model"]);
    if !has_models || has_versioning {
        item = item.pass();
    } else {
        item = item.partial("Models without version tracking");
    }

    item.finish_timed(start)
}

/// MA-07: Rollback Capability
pub fn check_rollback_capability(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item =
        CheckItem::new("MA-07", "Rollback Capability", "Any model version can be restored")
            .with_severity(Severity::Major)
            .with_tps("Recovery capability");

    let has_rollback = check_for_pattern(project_path, &["rollback", "restore", "revert"]);
    let has_retention = check_for_pattern(project_path, &["retention", "archive", "backup"]);

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!("Rollback: capability={}, retention={}", has_rollback, has_retention),
        data: None,
        files: Vec::new(),
    });

    let has_deployment = check_for_pattern(project_path, &["deploy", "serve", "production"]);
    if !has_deployment || has_rollback || has_retention {
        item = item.pass();
    } else {
        item = item.partial("Deployment without rollback capability");
    }

    item.finish_timed(start)
}

/// MA-08: A/B Test Logging
pub fn check_ab_test_logging(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item =
        CheckItem::new("MA-08", "A/B Test Logging", "A/B tests fully logged for analysis")
            .with_severity(Severity::Minor)
            .with_tps("Scientific experimentation");

    let has_ab_testing = check_for_pattern(project_path, &["ab_test", "experiment", "variant"]);
    let has_ab_logging = check_for_pattern(project_path, &["experiment_log", "assignment_log"]);

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!("A/B testing: impl={}, logging={}", has_ab_testing, has_ab_logging),
        data: None,
        files: Vec::new(),
    });

    if !has_ab_testing || has_ab_logging {
        item = item.pass();
    } else {
        item = item.partial("A/B testing without logging");
    }

    item.finish_timed(start)
}

/// MA-09: Bias Audit Trail
pub fn check_bias_audit(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item =
        CheckItem::new("MA-09", "Bias Audit Trail", "Bias assessments documented per model")
            .with_severity(Severity::Major)
            .with_tps("Ethical governance");

    let has_bias_testing =
        check_for_pattern(project_path, &["bias", "fairness", "demographic_parity"]);
    let has_audit = check_for_pattern(project_path, &["bias_audit", "fairness_report"]);

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!("Bias audit: testing={}, audit={}", has_bias_testing, has_audit),
        data: None,
        files: Vec::new(),
    });

    let is_ml = check_for_pattern(project_path, &["classifier", "predict", "model"]);
    if !is_ml || has_bias_testing || has_audit {
        item = item.pass();
    } else {
        item = item.partial("ML without bias audit documentation");
    }

    item.finish_timed(start)
}

/// MA-10: Incident Response Logging
pub fn check_incident_response(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item =
        CheckItem::new("MA-10", "Incident Response Logging", "Model incidents fully documented")
            .with_severity(Severity::Major)
            .with_tps("Kaizen — learning from failures");

    let has_incident_log =
        check_for_pattern(project_path, &["incident", "postmortem", "root_cause"]);
    let has_incident_docs =
        project_path.join("docs/incidents/").exists() || project_path.join("INCIDENTS.md").exists();

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!(
            "Incident logging: code={}, docs={}",
            has_incident_log, has_incident_docs
        ),
        data: None,
        files: Vec::new(),
    });

    if has_incident_log || has_incident_docs {
        item = item.pass();
    } else {
        item = item.partial("No incident response documentation");
    }

    item.finish_timed(start)
}

fn check_for_pattern(project_path: &Path, patterns: &[&str]) -> bool {
    super::helpers::source_contains_pattern(project_path, patterns)
        || super::helpers::files_contain_pattern_ci(project_path, &["**/*.md"], patterns)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_evaluate_all_returns_10_items() {
        let path = PathBuf::from(".");
        let items = evaluate_all(&path);
        assert_eq!(items.len(), 10);
    }

    #[test]
    fn test_all_items_have_tps_principle() {
        let path = PathBuf::from(".");
        for item in evaluate_all(&path) {
            assert!(!item.tps_principle.is_empty(), "Item {} missing TPS", item.id);
        }
    }

    #[test]
    fn test_all_items_have_evidence() {
        let path = PathBuf::from(".");
        for item in evaluate_all(&path) {
            assert!(!item.evidence.is_empty(), "Item {} missing evidence", item.id);
        }
    }

    // ========================================================================
    // Additional Coverage Tests for Model Cards
    // ========================================================================

    #[test]
    fn test_ma01_model_card_completeness_id() {
        let result = check_model_card_completeness(Path::new("."));
        assert_eq!(result.id, "MA-01");
        assert_eq!(result.severity, Severity::Major);
    }

    #[test]
    fn test_ma02_datasheet_completeness_id() {
        let result = check_datasheet_completeness(Path::new("."));
        assert_eq!(result.id, "MA-02");
        assert_eq!(result.severity, Severity::Major);
    }

    #[test]
    fn test_ma03_model_card_accuracy_id() {
        let result = check_model_card_accuracy(Path::new("."));
        assert_eq!(result.id, "MA-03");
        assert_eq!(result.severity, Severity::Major);
    }

    #[test]
    fn test_ma04_decision_logging_id() {
        let result = check_decision_logging(Path::new("."));
        assert_eq!(result.id, "MA-04");
        assert_eq!(result.severity, Severity::Major);
    }

    #[test]
    fn test_ma05_provenance_chain_id() {
        let result = check_provenance_chain(Path::new("."));
        assert_eq!(result.id, "MA-05");
        assert_eq!(result.severity, Severity::Major);
    }

    #[test]
    fn test_ma06_version_tracking_id() {
        let result = check_version_tracking(Path::new("."));
        assert_eq!(result.id, "MA-06");
        assert_eq!(result.severity, Severity::Major);
    }

    #[test]
    fn test_ma07_rollback_capability_id() {
        let result = check_rollback_capability(Path::new("."));
        assert_eq!(result.id, "MA-07");
        assert_eq!(result.severity, Severity::Major);
    }

    #[test]
    fn test_ma08_ab_test_logging_id() {
        let result = check_ab_test_logging(Path::new("."));
        assert_eq!(result.id, "MA-08");
        assert_eq!(result.severity, Severity::Minor);
    }

    #[test]
    fn test_ma09_bias_audit_id() {
        let result = check_bias_audit(Path::new("."));
        assert_eq!(result.id, "MA-09");
        assert_eq!(result.severity, Severity::Major);
    }

    #[test]
    fn test_ma10_incident_response_id() {
        let result = check_incident_response(Path::new("."));
        assert_eq!(result.id, "MA-10");
        assert_eq!(result.severity, Severity::Major);
    }

    #[test]
    fn test_model_card_with_temp_dir() {
        let temp_dir = std::env::temp_dir().join("test_model_cards");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(&temp_dir).expect("mkdir failed");

        // Create MODEL_CARD.md
        std::fs::write(temp_dir.join("MODEL_CARD.md"), "# Model Card").expect("fs write failed");

        let result = check_model_card_completeness(&temp_dir);
        assert_eq!(result.id, "MA-01");

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_datasheet_with_dir() {
        let temp_dir = std::env::temp_dir().join("test_datasheets");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(temp_dir.join("docs/datasheets")).expect("mkdir failed");

        let result = check_datasheet_completeness(&temp_dir);
        assert_eq!(result.id, "MA-02");

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_nonexistent_path() {
        let path = Path::new("/nonexistent/path/for/model_cards");
        let items = evaluate_all(path);
        assert_eq!(items.len(), 10);
    }
}
