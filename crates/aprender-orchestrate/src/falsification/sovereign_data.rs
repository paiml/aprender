//! Section 1: Sovereign Data Governance (SDG-01 to SDG-15)
//!
//! Implements the Five Pillars verification: Data Residency, Privacy, Legal Controls.
//!
//! # TPS Principles
//!
//! - **Jidoka**: Automatic boundary violation detection
//! - **Poka-Yoke**: Classification-based routing, privacy by design
//! - **Muda**: Data inventory hygiene, prevent hoarding liability
//! - **Genchi Genbutsu**: Verify physical data locations

use super::types::{CheckItem, Evidence, EvidenceType, Severity};
use std::path::Path;
use std::time::Instant;

/// Evaluate all Sovereign Data Governance checks.
pub fn evaluate_all(project_path: &Path) -> Vec<CheckItem> {
    vec![
        check_data_residency_boundary(project_path),
        check_data_inventory_completeness(project_path),
        check_privacy_preserving_computation(project_path),
        check_federated_learning_isolation(project_path),
        check_supply_chain_provenance(project_path),
        check_vpc_isolation(project_path),
        check_data_classification_enforcement(project_path),
        check_consent_purpose_limitation(project_path),
        check_rtbf_compliance(project_path),
        check_cross_border_logging(project_path),
        check_model_weight_sovereignty(project_path),
        check_inference_classification(project_path),
        check_audit_log_immutability(project_path),
        check_third_party_isolation(project_path),
        check_secure_computation(project_path),
    ]
}

/// SDG-01: Data Residency Boundary Enforcement
///
/// **Claim:** Sovereign-critical data never crosses defined geographic boundaries.
///
/// **Rejection Criteria (Critical):**
/// - Any network call to non-compliant regions during Sovereign-tier operations
pub fn check_data_residency_boundary(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "SDG-01",
        "Data Residency Boundary Enforcement",
        "Sovereign-critical data never crosses geographic boundaries",
    )
    .with_severity(Severity::Critical)
    .with_tps("Jidoka — automatic boundary violation detection");

    // Check for residency configuration
    let config_files = [
        project_path.join("config/residency.yaml"),
        project_path.join("config/residency.toml"),
        project_path.join(".sovereignty/residency.yaml"),
    ];

    let has_residency_config = config_files.iter().any(|p| p.exists());

    // Check for VPC/network isolation configuration
    let infra_files = [
        project_path.join("infrastructure/vpc.tf"),
        project_path.join("terraform/vpc.tf"),
        project_path.join("infra/network.yaml"),
    ];

    let has_network_isolation = infra_files.iter().any(|p| p.exists());

    // Check for residency-aware code patterns
    let has_residency_checks = check_for_pattern(
        project_path,
        &["residency", "region_check", "boundary_enforce", "geo_fence"],
    );

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!(
            "Residency: config={}, network_isolation={}, code_checks={}",
            has_residency_config, has_network_isolation, has_residency_checks
        ),
        data: None,
        files: Vec::new(),
    });

    let is_cli_only = !has_network_code(project_path);
    item = apply_check_outcome(
        item,
        &[
            (has_residency_config && has_network_isolation, None),
            (
                has_residency_config || has_residency_checks,
                Some("Partial residency enforcement (missing network isolation)"),
            ),
            (is_cli_only, None),
            (true, Some("No explicit residency configuration")),
        ],
    );

    item.finish_timed(start)
}

/// SDG-02: Data Inventory Completeness
///
/// **Claim:** All ingested data has documented purpose, classification, and lifecycle policy.
///
/// **Rejection Criteria (Major):**
/// - Any data asset without classification tag or lifecycle policy
pub fn check_data_inventory_completeness(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "SDG-02",
        "Data Inventory Completeness",
        "All data has documented purpose, classification, and lifecycle",
    )
    .with_severity(Severity::Major)
    .with_tps("Muda (Inventory waste) — prevent data hoarding");

    // Check for data catalog/inventory
    let inventory_files = [
        project_path.join("data/inventory.yaml"),
        project_path.join("docs/data-catalog.md"),
        project_path.join(".datasheets/"),
        project_path.join("data/schema/"),
    ];

    let has_inventory = inventory_files.iter().any(|p| p.exists());

    // Check for data classification schema
    let has_classification = check_for_pattern(
        project_path,
        &["DataClassification", "data_class", "classification_level"],
    );

    // Check for lifecycle policies
    let has_lifecycle = check_for_pattern(
        project_path,
        &["lifecycle_policy", "retention_days", "data_ttl", "expiration"],
    );

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!(
            "Data inventory: catalog={}, classification={}, lifecycle={}",
            has_inventory, has_classification, has_lifecycle
        ),
        data: None,
        files: Vec::new(),
    });

    let handles_data = check_for_pattern(project_path, &["DataFrame", "Dataset", "DataLoader"]);
    item = apply_check_outcome(
        item,
        &[
            (has_inventory && has_classification && has_lifecycle, None),
            (
                has_inventory || has_classification,
                Some("Partial data inventory (missing lifecycle or classification)"),
            ),
            (!handles_data, None),
            (true, Some("No data inventory documentation")),
        ],
    );

    item.finish_timed(start)
}

/// SDG-03: Privacy-Preserving Computation
///
/// **Claim:** Differential privacy correctly implemented where specified.
///
/// **Rejection Criteria (Major):**
/// - ε-δ guarantees not met, composition budget exceeded
pub fn check_privacy_preserving_computation(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "SDG-03",
        "Privacy-Preserving Computation",
        "Differential privacy correctly implemented",
    )
    .with_severity(Severity::Major)
    .with_tps("Poka-Yoke — privacy by design");

    // Check for differential privacy implementation
    let has_dp = check_for_pattern(
        project_path,
        &[
            "differential_privacy",
            "epsilon",
            "privacy_budget",
            "laplace_noise",
            "gaussian_mechanism",
        ],
    );

    // Check for privacy budget tracking
    let has_budget_tracking =
        check_for_pattern(project_path, &["privacy_accountant", "budget_consumed", "composition"]);

    // Check for privacy tests
    let has_privacy_tests = check_for_test_pattern(project_path, &["privacy", "dp_test"]);

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!(
            "Privacy: dp_impl={}, budget_tracking={}, tests={}",
            has_dp, has_budget_tracking, has_privacy_tests
        ),
        data: None,
        files: Vec::new(),
    });

    let needs_privacy = check_for_pattern(project_path, &["pii", "personal_data", "gdpr"]);
    item = apply_check_outcome(
        item,
        &[
            (has_dp && has_budget_tracking && has_privacy_tests, None),
            (has_dp, Some("DP implemented but missing budget tracking or tests")),
            (!needs_privacy, None),
            (true, Some("PII handling without differential privacy")),
        ],
    );

    item.finish_timed(start)
}

/// SDG-04: Federated Learning Client Isolation
///
/// **Claim:** Federated learning clients send only model updates, never raw data.
///
/// **Rejection Criteria (Critical):**
/// - Any payload containing raw training samples in network trace
pub fn check_federated_learning_isolation(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "SDG-04",
        "Federated Learning Client Isolation",
        "FL clients send only model updates, never raw data",
    )
    .with_severity(Severity::Critical)
    .with_tps("Jidoka — client-side enforcement");

    // Check for federated learning implementation
    let has_fl = check_for_pattern(
        project_path,
        &["federated", "FederatedClient", "model_update", "gradient_only"],
    );

    // Check for secure aggregation
    let has_secure_agg = check_for_pattern(
        project_path,
        &["secure_aggregation", "SecureAggregator", "encrypted_gradient"],
    );

    // Check for data isolation tests
    let has_isolation_tests =
        check_for_test_pattern(project_path, &["client_isolation", "no_raw_data"]);

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!(
            "FL isolation: fl_impl={}, secure_agg={}, tests={}",
            has_fl, has_secure_agg, has_isolation_tests
        ),
        data: None,
        files: Vec::new(),
    });

    item = apply_check_outcome(
        item,
        &[
            (!has_fl, None),
            (has_fl && has_secure_agg && has_isolation_tests, None),
            (
                has_fl && has_secure_agg,
                Some("FL with secure aggregation (missing isolation tests)"),
            ),
            (true, Some("FL without secure aggregation")),
        ],
    );

    item.finish_timed(start)
}

/// SDG-05: Supply Chain Provenance (AI BOM)
///
/// **Claim:** All model weights and training data have verified provenance.
///
/// **Rejection Criteria (Critical):**
/// - Pre-trained weight without signature, training data without chain-of-custody
pub fn check_supply_chain_provenance(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "SDG-05",
        "Supply Chain Provenance (AI BOM)",
        "All model weights and data have verified provenance",
    )
    .with_severity(Severity::Critical)
    .with_tps("Jidoka — supply chain circuit breaker");

    // Check for AI BOM / SBOM
    let bom_files = [
        project_path.join("ai-bom.json"),
        project_path.join("sbom.json"),
        project_path.join("bom.xml"),
        project_path.join(".sbom/"),
    ];

    let has_bom = bom_files.iter().any(|p| p.exists());

    // Check for cargo-audit / cargo-deny
    let cargo_toml = project_path.join("Cargo.toml");
    let has_audit = cargo_toml
        .exists()
        .then(|| std::fs::read_to_string(&cargo_toml).ok())
        .flatten()
        .map(|c| c.contains("cargo-deny") || c.contains("cargo-audit"))
        .unwrap_or(false);

    // Check deny.toml for license/source policies
    let deny_toml = project_path.join("deny.toml");
    let has_deny_config = deny_toml.exists();

    // Check for signature verification
    let has_signature_check = check_for_pattern(
        project_path,
        &["verify_signature", "Ed25519", "sign_model", "provenance"],
    );

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!(
            "Supply chain: bom={}, audit={}, deny={}, signatures={}",
            has_bom, has_audit, has_deny_config, has_signature_check
        ),
        data: None,
        files: Vec::new(),
    });

    item = apply_check_outcome(
        item,
        &[
            (has_deny_config && (has_bom || has_signature_check), None),
            (has_deny_config || has_audit, Some("Dependency audit configured (missing AI BOM)")),
            (true, Some("No supply chain verification configured")),
        ],
    );

    item.finish_timed(start)
}

/// SDG-06: VPC Isolation Verification
///
/// **Claim:** Compute resources spin up only in compliant sovereign regions.
///
/// **Rejection Criteria (Major):**
/// - Any resource in non-approved region
pub fn check_vpc_isolation(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "SDG-06",
        "VPC Isolation Verification",
        "Compute only in compliant sovereign regions",
    )
    .with_severity(Severity::Major)
    .with_tps("Genchi Genbutsu — verify physical location");

    // Check for IaC with region constraints
    let iac_files = [
        project_path.join("terraform/"),
        project_path.join("infrastructure/"),
        project_path.join("pulumi/"),
        project_path.join("cloudformation/"),
    ];

    let has_iac = iac_files.iter().any(|p| p.exists());

    // Check for region policy
    let has_region_policy = check_for_pattern(
        project_path,
        &["allowed_regions", "region_whitelist", "sovereign_region"],
    );

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!("VPC: iac={}, region_policy={}", has_iac, has_region_policy),
        data: None,
        files: Vec::new(),
    });

    item = apply_check_outcome(
        item,
        &[
            (!has_iac, None),
            (has_iac && has_region_policy, None),
            (true, Some("IaC without explicit region constraints")),
        ],
    );

    item.finish_timed(start)
}

/// SDG-07: Data Classification Enforcement
///
/// **Claim:** Code paths enforce data classification levels.
///
/// **Rejection Criteria (Major):**
/// - Sovereign-classified data processed by Public-tier code path
pub fn check_data_classification_enforcement(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "SDG-07",
        "Data Classification Enforcement",
        "Code paths enforce data classification levels",
    )
    .with_severity(Severity::Major)
    .with_tps("Poka-Yoke — classification-based routing");

    // Check for type-level classification
    let has_type_classification = check_for_pattern(
        project_path,
        &["Classification", "DataTier", "SecurityLevel", "Sovereign<", "Confidential<"],
    );

    // Check for runtime enforcement
    let has_runtime_check = check_for_pattern(
        project_path,
        &["check_classification", "enforce_tier", "validate_access_level"],
    );

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!(
            "Classification: type_level={}, runtime={}",
            has_type_classification, has_runtime_check
        ),
        data: None,
        files: Vec::new(),
    });

    let handles_classified =
        check_for_pattern(project_path, &["sovereign", "confidential", "restricted"]);
    item = apply_check_outcome(
        item,
        &[
            (has_type_classification && has_runtime_check, None),
            (
                has_type_classification || has_runtime_check,
                Some("Partial classification enforcement"),
            ),
            (!handles_classified, None),
            (true, Some("Classified data without enforcement")),
        ],
    );

    item.finish_timed(start)
}

/// SDG-08: Consent and Purpose Limitation
///
/// **Claim:** Data usage matches documented consent scope.
///
/// **Rejection Criteria (Major):**
/// - Any data processing without matching consent record
pub fn check_consent_purpose_limitation(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "SDG-08",
        "Consent and Purpose Limitation",
        "Data usage matches documented consent scope",
    )
    .with_severity(Severity::Major)
    .with_tps("Legal controls pillar");

    // Check for consent management
    let has_consent =
        check_for_pattern(project_path, &["consent", "purpose_limitation", "data_usage_agreement"]);

    // Check for purpose binding
    let has_purpose_binding =
        check_for_pattern(project_path, &["purpose_id", "usage_scope", "consent_scope"]);

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!(
            "Consent: management={}, purpose_binding={}",
            has_consent, has_purpose_binding
        ),
        data: None,
        files: Vec::new(),
    });

    let handles_user_data = check_for_pattern(project_path, &["user_data", "personal", "pii"]);
    item = apply_check_outcome(
        item,
        &[
            (!handles_user_data, None),
            (has_consent && has_purpose_binding, None),
            (has_consent, Some("Consent tracking without purpose binding")),
            (true, Some("User data handling without consent management")),
        ],
    );

    item.finish_timed(start)
}

/// SDG-09: Right to Erasure (RTBF) Compliance
///
/// **Claim:** Data deletion requests fully propagate through all storage.
///
/// **Rejection Criteria (Major):**
/// - Any trace of erased identity in storage or model
pub fn check_rtbf_compliance(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "SDG-09",
        "Right to Erasure (RTBF) Compliance",
        "Deletion requests propagate through all storage",
    )
    .with_severity(Severity::Major)
    .with_tps("Muda — data inventory hygiene");

    // Check for deletion cascade
    let has_deletion = check_for_pattern(
        project_path,
        &["delete_user", "erasure", "rtbf", "forget_user", "cascade_delete"],
    );

    // Check for model unlearning
    let has_unlearning =
        check_for_pattern(project_path, &["unlearn", "model_forget", "data_removal"]);

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!(
            "RTBF: deletion_cascade={}, model_unlearning={}",
            has_deletion, has_unlearning
        ),
        data: None,
        files: Vec::new(),
    });

    let stores_user_data = check_for_pattern(project_path, &["user_store", "persist_user", "save"]);
    item = apply_check_outcome(
        item,
        &[
            (!stores_user_data, None),
            (has_deletion && has_unlearning, None),
            (has_deletion, Some("Deletion without model unlearning")),
            (true, Some("Data persistence without erasure mechanism")),
        ],
    );

    item.finish_timed(start)
}

/// SDG-10: Cross-Border Transfer Logging
///
/// **Claim:** All cross-border data transfers are logged and justified.
///
/// **Rejection Criteria (Major):**
/// - Any cross-border transfer without audit log entry
pub fn check_cross_border_logging(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "SDG-10",
        "Cross-Border Transfer Logging",
        "All cross-border transfers logged and justified",
    )
    .with_severity(Severity::Major)
    .with_tps("Auditability requirement");

    // Check for transfer logging
    let has_transfer_log = check_for_pattern(
        project_path,
        &["transfer_log", "cross_border", "data_export", "international_transfer"],
    );

    // Check for legal basis documentation
    let has_legal_basis = check_for_pattern(
        project_path,
        &["legal_basis", "transfer_agreement", "adequacy_decision"],
    );

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!(
            "Cross-border: logging={}, legal_basis={}",
            has_transfer_log, has_legal_basis
        ),
        data: None,
        files: Vec::new(),
    });

    let does_transfers = has_network_code(project_path);
    item = apply_check_outcome(
        item,
        &[
            (!does_transfers, None),
            (has_transfer_log && has_legal_basis, None),
            (has_transfer_log, Some("Transfer logging without legal basis")),
            (true, Some("Network operations without transfer logging")),
        ],
    );

    item.finish_timed(start)
}

/// SDG-11: Model Weight Sovereignty
///
/// **Claim:** Model weights trained on sovereign data remain under sovereign control.
///
/// **Rejection Criteria (Critical):**
/// - Model weights accessible from non-sovereign context
pub fn check_model_weight_sovereignty(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "SDG-11",
        "Model Weight Sovereignty",
        "Weights from sovereign data remain under sovereign control",
    )
    .with_severity(Severity::Critical)
    .with_tps("Jidoka — weight exfiltration prevention");

    // Check for weight access control
    let has_access_control = check_for_pattern(
        project_path,
        &["weight_access", "model_acl", "sovereign_model", "protected_weights"],
    );

    // Check for encryption
    let has_encryption =
        check_for_pattern(project_path, &["encrypt_weights", "sealed_model", "encrypted_model"]);

    // Check for key management
    let has_key_mgmt = check_for_pattern(project_path, &["key_management", "kms", "key_rotation"]);

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!(
            "Weight sovereignty: access_control={}, encryption={}, key_mgmt={}",
            has_access_control, has_encryption, has_key_mgmt
        ),
        data: None,
        files: Vec::new(),
    });

    let handles_weights =
        check_for_pattern(project_path, &["model_weights", "load_model", "save_model"]);
    item = apply_check_outcome(
        item,
        &[
            (!handles_weights, None),
            (has_access_control && has_encryption, None),
            (has_access_control || has_encryption, Some("Partial weight protection")),
            (true, Some("Model weights without sovereignty controls")),
        ],
    );

    item.finish_timed(start)
}

/// SDG-12: Inference Result Classification
///
/// **Claim:** Inference outputs inherit classification from input data.
///
/// **Rejection Criteria (Major):**
/// - Sovereign input produces unclassified output
pub fn check_inference_classification(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "SDG-12",
        "Inference Result Classification",
        "Outputs inherit classification from inputs",
    )
    .with_severity(Severity::Major)
    .with_tps("Poka-Yoke — automatic classification propagation");

    // Check for classification inheritance
    let has_inheritance = check_for_pattern(
        project_path,
        &["inherit_classification", "propagate_tier", "output_classification"],
    );

    // Check for output tagging
    let has_output_tagging =
        check_for_pattern(project_path, &["tag_output", "classify_result", "result_tier"]);

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!(
            "Inference classification: inheritance={}, tagging={}",
            has_inheritance, has_output_tagging
        ),
        data: None,
        files: Vec::new(),
    });

    let does_inference = check_for_pattern(project_path, &["inference", "predict", "infer"]);
    item = apply_check_outcome(
        item,
        &[
            (!does_inference, None),
            (has_inheritance && has_output_tagging, None),
            (has_inheritance || has_output_tagging, Some("Partial classification propagation")),
            (true, Some("Inference without classification propagation")),
        ],
    );

    item.finish_timed(start)
}

/// SDG-13: Audit Log Immutability
///
/// **Claim:** Audit logs cannot be modified or deleted.
///
/// **Rejection Criteria (Critical):**
/// - Any successful modification of historical log entry
pub fn check_audit_log_immutability(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "SDG-13",
        "Audit Log Immutability",
        "Audit logs cannot be modified or deleted",
    )
    .with_severity(Severity::Critical)
    .with_tps("Governance layer integrity");

    // Check for append-only logging
    let has_append_only =
        check_for_pattern(project_path, &["append_only", "immutable_log", "write_once"]);

    // Check for cryptographic chaining
    let has_chaining = check_for_pattern(
        project_path,
        &["merkle", "hash_chain", "log_signature", "tamper_evident"],
    );

    // Check for audit trail implementation
    let has_audit_trail =
        check_for_pattern(project_path, &["audit_trail", "audit_log", "AuditEntry"]);

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!(
            "Audit immutability: append_only={}, chaining={}, trail={}",
            has_append_only, has_chaining, has_audit_trail
        ),
        data: None,
        files: Vec::new(),
    });

    item = apply_check_outcome(
        item,
        &[
            (has_audit_trail && has_chaining, None),
            (has_audit_trail, Some("Audit trail without cryptographic verification")),
            (true, Some("No immutable audit logging")),
        ],
    );

    item.finish_timed(start)
}

/// SDG-14: Third-Party API Isolation
///
/// **Claim:** No third-party API calls during sovereign-tier operations.
///
/// **Rejection Criteria (Critical):**
/// - Any outbound call to non-sovereign endpoint
pub fn check_third_party_isolation(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "SDG-14",
        "Third-Party API Isolation",
        "No third-party API calls during sovereign operations",
    )
    .with_severity(Severity::Critical)
    .with_tps("Jidoka — network isolation");

    // Check for network allowlist
    let has_allowlist = check_for_pattern(
        project_path,
        &["allowlist", "whitelist", "approved_endpoints", "sovereign_endpoints"],
    );

    // Check for offline mode
    let has_offline_mode =
        check_for_pattern(project_path, &["offline_mode", "airgap", "no_network", "local_only"]);

    // Check for network guards
    let has_network_guard =
        check_for_pattern(project_path, &["network_guard", "egress_filter", "outbound_check"]);

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!(
            "Third-party isolation: allowlist={}, offline={}, guard={}",
            has_allowlist, has_offline_mode, has_network_guard
        ),
        data: None,
        files: Vec::new(),
    });

    let does_network = has_network_code(project_path);
    item = apply_check_outcome(
        item,
        &[
            (!does_network || has_offline_mode, None),
            (has_allowlist && has_network_guard, None),
            (has_allowlist, Some("Allowlist without runtime guard")),
            (true, Some("Network operations without isolation controls")),
        ],
    );

    item.finish_timed(start)
}

/// SDG-15: Homomorphic/Secure Computation Verification
///
/// **Claim:** Secure computation primitives correctly implemented.
///
/// **Rejection Criteria (Major):**
/// - Any information leakage beyond specified bounds
pub fn check_secure_computation(project_path: &Path) -> CheckItem {
    let start = Instant::now();
    let mut item = CheckItem::new(
        "SDG-15",
        "Secure Computation Verification",
        "Cryptographic primitives correctly implemented",
    )
    .with_severity(Severity::Major)
    .with_tps("Formal verification requirement");

    // Check for homomorphic encryption
    let has_he = check_for_pattern(project_path, &["homomorphic", "fhe", "seal", "paillier"]);

    // Check for MPC
    let has_mpc = check_for_pattern(project_path, &["mpc", "secure_multiparty", "secret_sharing"]);

    // Check for TEE
    let has_tee = check_for_pattern(project_path, &["sgx", "enclave", "trusted_execution", "tee"]);

    // Check for crypto tests
    let has_crypto_tests =
        check_for_test_pattern(project_path, &["crypto", "encryption", "secure_compute"]);

    item = item.with_evidence(Evidence {
        evidence_type: EvidenceType::StaticAnalysis,
        description: format!(
            "Secure compute: he={}, mpc={}, tee={}, tests={}",
            has_he, has_mpc, has_tee, has_crypto_tests
        ),
        data: None,
        files: Vec::new(),
    });

    let needs_secure_compute = has_he || has_mpc || has_tee;
    item = apply_check_outcome(
        item,
        &[
            (!needs_secure_compute, None),
            (needs_secure_compute && has_crypto_tests, None),
            (true, Some("Secure computation without comprehensive tests")),
        ],
    );

    item.finish_timed(start)
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Apply the first matching outcome to a CheckItem.
///
/// Each entry is `(condition, outcome)` where `None` means pass and
/// `Some(msg)` means partial with that message.  The first `true` condition
/// wins; if nothing matches the item is returned unchanged.
fn apply_check_outcome(item: CheckItem, checks: &[(bool, Option<&str>)]) -> CheckItem {
    for &(condition, msg) in checks {
        if condition {
            return match msg {
                Some(m) => item.partial(m),
                None => item.pass(),
            };
        }
    }
    item
}

/// Check if any Rust source or config file contains any of the given patterns.
fn check_for_pattern(project_path: &Path, patterns: &[&str]) -> bool {
    super::helpers::source_or_config_contains_pattern(project_path, patterns)
}

/// Check if any test file contains any of the given patterns.
fn check_for_test_pattern(project_path: &Path, patterns: &[&str]) -> bool {
    super::helpers::test_contains_pattern(project_path, patterns)
}

/// Check if project has network code (HTTP, gRPC, etc.).
fn has_network_code(project_path: &Path) -> bool {
    check_for_pattern(
        project_path,
        &["reqwest", "hyper", "tonic", "TcpStream", "HttpClient", "fetch", "grpc"],
    )
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_evaluate_all_returns_15_items() {
        let path = PathBuf::from(".");
        let items = evaluate_all(&path);
        assert_eq!(items.len(), 15);
    }

    #[test]
    fn test_all_items_have_tps_principle() {
        let path = PathBuf::from(".");
        let items = evaluate_all(&path);
        for item in items {
            assert!(!item.tps_principle.is_empty(), "Item {} missing TPS principle", item.id);
        }
    }

    #[test]
    fn test_all_items_have_evidence() {
        let path = PathBuf::from(".");
        let items = evaluate_all(&path);
        for item in items {
            assert!(!item.evidence.is_empty(), "Item {} missing evidence", item.id);
        }
    }

    #[test]
    fn test_sdg_01_data_residency() {
        let path = PathBuf::from(".");
        let item = check_data_residency_boundary(&path);
        assert_eq!(item.id, "SDG-01");
        assert_eq!(item.severity, Severity::Critical);
    }

    #[test]
    fn test_sdg_05_supply_chain() {
        let path = PathBuf::from(".");
        let item = check_supply_chain_provenance(&path);
        assert_eq!(item.id, "SDG-05");
        assert_eq!(item.severity, Severity::Critical);
    }

    #[test]
    fn test_sdg_13_audit_log() {
        let path = PathBuf::from(".");
        let item = check_audit_log_immutability(&path);
        assert_eq!(item.id, "SDG-13");
        assert_eq!(item.severity, Severity::Critical);
    }

    // =========================================================================
    // Additional Coverage Tests
    // =========================================================================

    #[test]
    fn test_sdg_02_data_inventory() {
        let path = PathBuf::from(".");
        let item = check_data_inventory_completeness(&path);
        assert_eq!(item.id, "SDG-02");
    }

    #[test]
    fn test_sdg_03_privacy_preserving() {
        let path = PathBuf::from(".");
        let item = check_privacy_preserving_computation(&path);
        assert_eq!(item.id, "SDG-03");
    }

    #[test]
    fn test_sdg_04_federated_learning() {
        let path = PathBuf::from(".");
        let item = check_federated_learning_isolation(&path);
        assert_eq!(item.id, "SDG-04");
    }

    #[test]
    fn test_sdg_06_vpc_isolation() {
        let path = PathBuf::from(".");
        let item = check_vpc_isolation(&path);
        assert_eq!(item.id, "SDG-06");
    }

    #[test]
    fn test_sdg_07_data_classification() {
        let path = PathBuf::from(".");
        let item = check_data_classification_enforcement(&path);
        assert_eq!(item.id, "SDG-07");
    }

    #[test]
    fn test_sdg_08_consent_purpose() {
        let path = PathBuf::from(".");
        let item = check_consent_purpose_limitation(&path);
        assert_eq!(item.id, "SDG-08");
    }

    #[test]
    fn test_sdg_09_rtbf_compliance() {
        let path = PathBuf::from(".");
        let item = check_rtbf_compliance(&path);
        assert_eq!(item.id, "SDG-09");
    }

    #[test]
    fn test_sdg_10_cross_border() {
        let path = PathBuf::from(".");
        let item = check_cross_border_logging(&path);
        assert_eq!(item.id, "SDG-10");
    }

    #[test]
    fn test_sdg_11_model_weight() {
        let path = PathBuf::from(".");
        let item = check_model_weight_sovereignty(&path);
        assert_eq!(item.id, "SDG-11");
    }

    #[test]
    fn test_sdg_12_inference_classification() {
        let path = PathBuf::from(".");
        let item = check_inference_classification(&path);
        assert_eq!(item.id, "SDG-12");
    }

    #[test]
    fn test_sdg_14_third_party() {
        let path = PathBuf::from(".");
        let item = check_third_party_isolation(&path);
        assert_eq!(item.id, "SDG-14");
    }

    #[test]
    fn test_sdg_15_secure_computation() {
        let path = PathBuf::from(".");
        let item = check_secure_computation(&path);
        assert_eq!(item.id, "SDG-15");
    }

    #[test]
    fn test_nonexistent_path() {
        let path = PathBuf::from("/nonexistent/path");
        let items = evaluate_all(&path);
        assert_eq!(items.len(), 15);
    }

    #[test]
    fn test_temp_dir_with_privacy_lib() {
        let temp_dir = std::env::temp_dir().join("test_privacy_lib");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(&temp_dir).expect("mkdir failed");

        std::fs::write(
            temp_dir.join("Cargo.toml"),
            r#"
[package]
name = "test"
version = "0.1.0"

[dependencies]
differential-privacy = "0.1"
"#,
        )
        .expect("unexpected failure");

        let item = check_privacy_preserving_computation(&temp_dir);
        assert_eq!(item.id, "SDG-03");

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_temp_dir_with_data_types() {
        let temp_dir = std::env::temp_dir().join("test_data_types");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(temp_dir.join("src")).expect("mkdir failed");

        std::fs::write(
            temp_dir.join("src/lib.rs"),
            r#"
pub struct DataInventory {
    pub pii: bool,
    pub classification: String,
}
"#,
        )
        .expect("unexpected failure");

        let item = check_data_inventory_completeness(&temp_dir);
        assert_eq!(item.id, "SDG-02");

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_temp_dir_with_audit_logging() {
        let temp_dir = std::env::temp_dir().join("test_audit_log");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(temp_dir.join("src")).expect("mkdir failed");

        std::fs::write(
            temp_dir.join("src/lib.rs"),
            r#"
pub struct AuditLog {
    pub append_only: bool,
    pub immutable: bool,
}
"#,
        )
        .expect("unexpected failure");

        let item = check_audit_log_immutability(&temp_dir);
        assert_eq!(item.id, "SDG-13");

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_all_have_severity() {
        let path = PathBuf::from(".");
        let items = evaluate_all(&path);
        for item in items {
            // All should have a severity set
            assert!(
                item.severity == Severity::Critical
                    || item.severity == Severity::Major
                    || item.severity == Severity::Minor
            );
        }
    }
}
