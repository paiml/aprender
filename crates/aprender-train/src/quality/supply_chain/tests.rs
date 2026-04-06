//! Tests for supply chain auditing.

use super::*;

#[test]
fn test_severity_from_str() {
    assert_eq!(Severity::parse("critical"), Severity::Critical);
    assert_eq!(Severity::parse("CRITICAL"), Severity::Critical);
    assert_eq!(Severity::parse("high"), Severity::High);
    assert_eq!(Severity::parse("medium"), Severity::Medium);
    assert_eq!(Severity::parse("low"), Severity::Low);
    assert_eq!(Severity::parse("unknown"), Severity::None);
}

#[test]
fn test_severity_ordering() {
    assert!(Severity::Critical > Severity::High);
    assert!(Severity::High > Severity::Medium);
    assert!(Severity::Medium > Severity::Low);
    assert!(Severity::Low > Severity::None);
}

#[test]
fn test_severity_display() {
    assert_eq!(format!("{}", Severity::Critical), "critical");
    assert_eq!(format!("{}", Severity::None), "none");
}

#[test]
fn test_audit_status_is_failure() {
    assert!(AuditStatus::Vulnerable.is_failure());
    assert!(!AuditStatus::Warning.is_failure());
    assert!(!AuditStatus::Clean.is_failure());
}

#[test]
fn test_audit_status_has_issues() {
    assert!(AuditStatus::Vulnerable.has_issues());
    assert!(AuditStatus::Warning.has_issues());
    assert!(!AuditStatus::Clean.has_issues());
}

#[test]
fn test_advisory_new() {
    let advisory = Advisory::new("RUSTSEC-2021-0001", Severity::High, "Test vulnerability");

    assert_eq!(advisory.id, "RUSTSEC-2021-0001");
    assert_eq!(advisory.severity, Severity::High);
    assert_eq!(advisory.title, "Test vulnerability");
}

#[test]
fn test_dependency_audit_clean() {
    let audit = DependencyAudit::clean("serde", "1.0.0", "MIT OR Apache-2.0");

    assert_eq!(audit.crate_name, "serde");
    assert_eq!(audit.version, "1.0.0");
    assert_eq!(audit.license, "MIT OR Apache-2.0");
    assert_eq!(audit.audit_status, AuditStatus::Clean);
    assert!(audit.advisories.is_empty());
    assert!(!audit.is_vulnerable());
}

#[test]
fn test_dependency_audit_vulnerable() {
    let advisory = Advisory::new("RUSTSEC-2021-0001", Severity::Critical, "RCE vulnerability");
    let audit = DependencyAudit::vulnerable("unsafe-crate", "0.1.0", "MIT", vec![advisory]);

    assert_eq!(audit.crate_name, "unsafe-crate");
    assert_eq!(audit.audit_status, AuditStatus::Vulnerable);
    assert!(audit.is_vulnerable());
    assert_eq!(audit.advisories.len(), 1);
    assert_eq!(audit.max_severity(), Severity::Critical);
}

#[test]
fn test_dependency_audit_max_severity() {
    let audit = DependencyAudit::vulnerable(
        "multi-vuln",
        "1.0.0",
        "MIT",
        vec![
            Advisory::new("A001", Severity::Low, "Low issue"),
            Advisory::new("A002", Severity::Critical, "Critical issue"),
            Advisory::new("A003", Severity::Medium, "Medium issue"),
        ],
    );

    assert_eq!(audit.max_severity(), Severity::Critical);
}

#[test]
fn test_dependency_audit_max_severity_empty() {
    let audit = DependencyAudit::clean("safe-crate", "1.0.0", "MIT");
    assert_eq!(audit.max_severity(), Severity::None);
}

#[test]
fn test_from_cargo_deny_output_empty() {
    let json = "";
    let audits = DependencyAudit::from_cargo_deny_output(json).expect("operation should succeed");
    assert!(audits.is_empty());
}

#[test]
fn test_from_cargo_deny_output_no_vulnerabilities() {
    let json = r#"{"type": "summary", "fields": {"total": 100}}"#;
    let audits = DependencyAudit::from_cargo_deny_output(json).expect("operation should succeed");
    assert!(audits.is_empty());
}

#[test]
fn test_from_cargo_deny_output_with_vulnerability() {
    let json = r#"{"type":"diagnostic","fields":{"graphs":[],"severity":"error","code":"A001","message":"Detected security vulnerability","labels":[{"span":{"crate":{"name":"vulnerable-crate","version":"0.1.0"}}}]}}"#;

    let audits = DependencyAudit::from_cargo_deny_output(json).expect("operation should succeed");

    assert_eq!(audits.len(), 1);
    assert_eq!(audits[0].crate_name, "vulnerable-crate");
    assert_eq!(audits[0].version, "0.1.0");
    assert!(audits[0].is_vulnerable());
}

#[test]
fn test_from_cargo_deny_output_invalid_json() {
    let json = "not valid json";
    let result = DependencyAudit::from_cargo_deny_output(json);
    assert!(result.is_err());
}

#[test]
fn test_audit_summary_from_audits() {
    let audits = vec![
        DependencyAudit::clean("serde", "1.0.0", "MIT"),
        DependencyAudit::clean("tokio", "1.0.0", "MIT"),
        DependencyAudit::vulnerable(
            "vuln-crate",
            "0.1.0",
            "MIT",
            vec![Advisory::new("A001", Severity::High, "Vuln")],
        ),
    ];

    let summary = AuditSummary::from_audits(audits);

    assert_eq!(summary.total_dependencies, 3);
    assert_eq!(summary.clean_count, 2);
    assert_eq!(summary.warning_count, 0);
    assert_eq!(summary.vulnerable_count, 1);
    assert!(summary.has_vulnerabilities());
    assert!(summary.has_issues());
}

#[test]
fn test_audit_summary_all_clean() {
    let audits = vec![
        DependencyAudit::clean("serde", "1.0.0", "MIT"),
        DependencyAudit::clean("tokio", "1.0.0", "MIT"),
    ];

    let summary = AuditSummary::from_audits(audits);

    assert!(!summary.has_vulnerabilities());
    assert!(!summary.has_issues());
    assert!(summary.vulnerable_deps().is_empty());
}

#[test]
fn test_audit_summary_vulnerable_deps() {
    let audits = vec![
        DependencyAudit::clean("serde", "1.0.0", "MIT"),
        DependencyAudit::vulnerable(
            "vuln1",
            "0.1.0",
            "MIT",
            vec![Advisory::new("A001", Severity::High, "Vuln")],
        ),
        DependencyAudit::vulnerable(
            "vuln2",
            "0.2.0",
            "MIT",
            vec![Advisory::new("A002", Severity::Critical, "Vuln")],
        ),
    ];

    let summary = AuditSummary::from_audits(audits);
    let vulnerable = summary.vulnerable_deps();

    assert_eq!(vulnerable.len(), 2);
    assert_eq!(vulnerable[0].crate_name, "vuln1");
    assert_eq!(vulnerable[1].crate_name, "vuln2");
}

#[test]
fn test_dependency_audit_serialization() {
    let audit = DependencyAudit::vulnerable(
        "test-crate",
        "1.0.0",
        "MIT",
        vec![Advisory::new("A001", Severity::High, "Test")],
    );

    let json = serde_json::to_string(&audit).expect("JSON serialization should succeed");
    let parsed: DependencyAudit =
        serde_json::from_str(&json).expect("JSON deserialization should succeed");

    assert_eq!(parsed.crate_name, audit.crate_name);
    assert_eq!(parsed.audit_status, audit.audit_status);
}
