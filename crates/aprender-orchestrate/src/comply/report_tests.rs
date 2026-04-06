use super::*;
use crate::comply::rule::RuleResult;

#[test]
fn test_report_creation() {
    let report = ComplyReport::new();
    assert!(report.results.is_empty());
    assert!(!report.finalized);
}

#[test]
fn test_add_result() {
    let mut report = ComplyReport::new();
    report.add_result("test-project", "test-rule", RuleResult::pass());
    assert!(report.results.contains_key("test-project"));
}

#[test]
fn test_finalize() {
    let mut report = ComplyReport::new();
    report.add_result("project1", "rule1", RuleResult::pass());
    report.add_result("project2", "rule1", RuleResult::fail(vec![]));
    report.finalize();

    assert!(report.finalized);
    assert_eq!(report.summary.total_projects, 2);
    assert_eq!(report.summary.passing_projects, 1);
}

#[test]
fn test_format_text() {
    let mut report = ComplyReport::new();
    report.add_result("trueno", "makefile-targets", RuleResult::pass());
    report.finalize();

    let text = report.format_text();
    assert!(text.contains("STACK COMPLIANCE REPORT"));
    assert!(text.contains("trueno"));
}

#[test]
fn test_format_json() {
    let mut report = ComplyReport::new();
    report.add_result("trueno", "makefile-targets", RuleResult::pass());
    report.finalize();

    let json = report.format_json();
    assert!(json.contains("trueno"));
    assert!(json.contains("summary"));
}

#[test]
fn test_is_compliant() {
    let mut report = ComplyReport::new();
    report.add_result("project1", "rule1", RuleResult::pass());
    report.finalize();
    assert!(report.is_compliant());

    let mut report2 = ComplyReport::new();
    report2.add_result("project1", "rule1", RuleResult::fail(vec![]));
    report2.finalize();
    assert!(!report2.is_compliant());
}

#[test]
fn test_add_exemption() {
    let mut report = ComplyReport::new();
    report.add_exemption("project", "rule");
    assert_eq!(report.exemptions.len(), 1);
    assert_eq!(report.exemptions[0].project, "project");
    assert_eq!(report.exemptions[0].rule, "rule");
}

#[test]
fn test_add_error() {
    let mut report = ComplyReport::new();
    report.add_error("project", "rule", "Error message".to_string());
    let result = report.results.get("project").unwrap().get("rule").unwrap();
    assert!(matches!(result, ProjectRuleResult::Error(_)));
}

#[test]
fn test_add_global_error() {
    let mut report = ComplyReport::new();
    report.add_global_error("Global error".to_string());
    assert_eq!(report.errors.len(), 1);
    assert_eq!(report.errors[0], "Global error");
}

#[test]
fn test_add_fix_result() {
    let mut report = ComplyReport::new();
    report.add_fix_result("project", "rule", FixResult::success(5));
    let result = report.results.get("project").unwrap().get("rule").unwrap();
    assert!(matches!(result, ProjectRuleResult::Fixed(_)));
}

#[test]
fn test_add_dry_run_fix() {
    let mut report = ComplyReport::new();
    let violations = vec![RuleViolation::new("V-001", "Test violation")];
    report.add_dry_run_fix("project", "rule", &violations);
    let result = report.results.get("project").unwrap().get("rule").unwrap();
    assert!(matches!(result, ProjectRuleResult::DryRunFix(_)));
}

#[test]
fn test_violations() {
    let mut report = ComplyReport::new();
    let violations = vec![RuleViolation::new("V-001", "Test violation")];
    report.add_result("project", "rule", RuleResult::fail(violations));
    report.finalize();
    let vs = report.violations();
    assert_eq!(vs.len(), 1);
    assert_eq!(vs[0].code, "V-001");
}

#[test]
fn test_format_markdown() {
    let mut report = ComplyReport::new();
    report.add_result("project", "rule", RuleResult::pass());
    report.finalize();
    let md = report.format_markdown();
    assert!(md.contains("# Stack Compliance Report"));
    assert!(md.contains("project"));
}

#[test]
fn test_format_html() {
    let mut report = ComplyReport::new();
    report.add_result("project", "rule", RuleResult::pass());
    report.finalize();
    let html = report.format_html();
    assert!(html.contains("<!DOCTYPE html>"));
    assert!(html.contains("Stack Compliance Report"));
}

#[test]
fn test_format_dispatch() {
    let mut report = ComplyReport::new();
    report.add_result("project", "rule", RuleResult::pass());
    report.finalize();

    let text = report.format(ComplyReportFormat::Text);
    assert!(text.contains("STACK COMPLIANCE REPORT"));

    let json = report.format(ComplyReportFormat::Json);
    assert!(json.contains("summary"));

    let md = report.format(ComplyReportFormat::Markdown);
    assert!(md.contains("# Stack"));

    let html = report.format(ComplyReportFormat::Html);
    assert!(html.contains("<html>"));
}

#[test]
fn test_violation_severity_from() {
    assert!(matches!(ViolationSeverity::from(ViolationLevel::Info), ViolationSeverity::Info));
    assert!(matches!(ViolationSeverity::from(ViolationLevel::Warning), ViolationSeverity::Warning));
    assert!(matches!(ViolationSeverity::from(ViolationLevel::Error), ViolationSeverity::Error));
    assert!(matches!(
        ViolationSeverity::from(ViolationLevel::Critical),
        ViolationSeverity::Critical
    ));
}

#[test]
fn test_compliance_summary_default() {
    let summary = ComplianceSummary::default();
    assert_eq!(summary.total_projects, 0);
    assert_eq!(summary.total_violations, 0);
    assert_eq!(summary.pass_rate, 0.0);
}

#[test]
fn test_report_default() {
    let report = ComplyReport::default();
    assert!(report.results.is_empty());
}

#[test]
fn test_finalize_idempotent() {
    let mut report = ComplyReport::new();
    report.add_result("project", "rule", RuleResult::pass());
    report.finalize();
    let summary1 = report.summary.clone();
    report.finalize(); // Should not change
    assert_eq!(report.summary.total_projects, summary1.total_projects);
}

#[test]
fn test_exemption_fields() {
    let exemption = Exemption {
        project: "test-project".to_string(),
        rule: "test-rule".to_string(),
        reason: Some("Legacy code".to_string()),
    };
    assert_eq!(exemption.project, "test-project");
    assert_eq!(exemption.rule, "test-rule");
    assert_eq!(exemption.reason, Some("Legacy code".to_string()));
}

#[test]
fn test_exemption_clone() {
    let exemption = Exemption { project: "p".to_string(), rule: "r".to_string(), reason: None };
    let cloned = exemption.clone();
    assert_eq!(cloned.project, exemption.project);
}

#[test]
fn test_violation_fields() {
    let v = Violation {
        project: "proj".to_string(),
        rule: "rule".to_string(),
        code: "V-001".to_string(),
        message: "msg".to_string(),
        severity: ViolationSeverity::Error,
        location: Some("file.rs:10".to_string()),
        fixable: true,
    };
    assert_eq!(v.project, "proj");
    assert_eq!(v.code, "V-001");
    assert!(v.location.is_some());
    assert!(v.fixable);
}

#[test]
fn test_violation_clone() {
    let v = Violation {
        project: "p".to_string(),
        rule: "r".to_string(),
        code: "C".to_string(),
        message: "m".to_string(),
        severity: ViolationSeverity::Warning,
        location: None,
        fixable: false,
    };
    let cloned = v.clone();
    assert_eq!(cloned.code, v.code);
    assert!(!cloned.fixable);
}

#[test]
fn test_violation_severity_debug() {
    let info = ViolationSeverity::Info;
    let warning = ViolationSeverity::Warning;
    let error = ViolationSeverity::Error;
    let critical = ViolationSeverity::Critical;

    assert!(format!("{:?}", info).contains("Info"));
    assert!(format!("{:?}", warning).contains("Warning"));
    assert!(format!("{:?}", error).contains("Error"));
    assert!(format!("{:?}", critical).contains("Critical"));
}

#[test]
fn test_comply_report_format_default() {
    let fmt = ComplyReportFormat::default();
    assert!(matches!(fmt, ComplyReportFormat::Text));
}

#[test]
fn test_comply_report_format_clone() {
    let fmt = ComplyReportFormat::Json;
    let cloned = fmt;
    assert!(matches!(cloned, ComplyReportFormat::Json));
}

#[test]
fn test_finalize_with_exemptions() {
    let mut report = ComplyReport::new();
    report.add_exemption("project", "rule");
    report.finalize();
    assert_eq!(report.summary.passed_checks, 1);
}

#[test]
fn test_finalize_with_errors() {
    let mut report = ComplyReport::new();
    report.add_error("project", "rule", "Error".to_string());
    report.finalize();
    assert_eq!(report.summary.failed_checks, 1);
    assert_eq!(report.summary.failing_projects, 1);
}

#[test]
fn test_finalize_with_fix_success() {
    let mut report = ComplyReport::new();
    report.add_fix_result("project", "rule", FixResult::success(3));
    report.finalize();
    assert_eq!(report.summary.passed_checks, 1);
}

#[test]
fn test_finalize_with_fix_failure() {
    let mut report = ComplyReport::new();
    report.add_fix_result("project", "rule", FixResult::failure("Fix failed"));
    report.finalize();
    assert_eq!(report.summary.failed_checks, 1);
}

#[test]
fn test_finalize_with_dry_run_fix() {
    let mut report = ComplyReport::new();
    let violations = vec![
        RuleViolation::new("V-001", "Test").fixable(),
        RuleViolation::new("V-002", "Test2"), // not fixable
    ];
    report.add_dry_run_fix("project", "rule", &violations);
    report.finalize();
    assert_eq!(report.summary.total_violations, 2);
    assert_eq!(report.summary.fixable_violations, 1);
}

#[test]
fn test_finalize_with_fixable_violations() {
    let mut report = ComplyReport::new();
    let violations = vec![RuleViolation::new("V-001", "Test").fixable()];
    report.add_result("project", "rule", RuleResult::fail(violations));
    report.finalize();
    assert_eq!(report.summary.fixable_violations, 1);
}

#[test]
fn test_finalize_violations_by_severity() {
    let mut report = ComplyReport::new();
    let mut v = RuleViolation::new("V-001", "Error violation");
    v.severity = ViolationLevel::Error;
    report.add_result("project", "rule", RuleResult::fail(vec![v]));
    report.finalize();
    assert!(report.summary.violations_by_severity.contains_key("ERROR"));
}

#[test]
fn test_format_text_with_violations() {
    let mut report = ComplyReport::new();
    let v = RuleViolation::new("V-001", "Test violation").with_location("file.rs:10");
    report.add_result("project", "rule", RuleResult::fail(vec![v]));
    report.finalize();
    let text = report.format_text();
    assert!(text.contains("V-001"));
    assert!(text.contains("Test violation"));
    assert!(text.contains("at file.rs:10"));
}

#[test]
fn test_format_text_with_exempt() {
    let mut report = ComplyReport::new();
    report.add_exemption("project", "rule");
    report.finalize();
    let text = report.format_text();
    assert!(text.contains("[EXEMPT]"));
}

#[test]
fn test_format_text_with_error() {
    let mut report = ComplyReport::new();
    report.add_error("project", "rule", "Something went wrong".to_string());
    report.finalize();
    let text = report.format_text();
    assert!(text.contains("[ERROR]"));
    assert!(text.contains("Something went wrong"));
}

#[test]
fn test_format_text_with_fixed() {
    let mut report = ComplyReport::new();
    report.add_fix_result("project", "rule", FixResult::success(5));
    report.finalize();
    let text = report.format_text();
    assert!(text.contains("[FIXED]"));
    assert!(text.contains("5 fixes applied"));
}

#[test]
fn test_format_text_with_dry_run() {
    let mut report = ComplyReport::new();
    report.add_dry_run_fix("project", "rule", &[RuleViolation::new("V-001", "Test")]);
    report.finalize();
    let text = report.format_text();
    assert!(text.contains("[DRY-RUN]"));
}

#[test]
fn test_format_text_with_global_errors() {
    let mut report = ComplyReport::new();
    report.add_global_error("Global error 1".to_string());
    report.add_global_error("Global error 2".to_string());
    report.finalize();
    let text = report.format_text();
    assert!(text.contains("Global Errors:"));
    assert!(text.contains("Global error 1"));
}

#[test]
fn test_format_text_fixable_stats() {
    let mut report = ComplyReport::new();
    let v = RuleViolation::new("V-001", "Test").fixable();
    report.add_result("project", "rule", RuleResult::fail(vec![v]));
    report.finalize();
    let text = report.format_text();
    assert!(text.contains("Fixable:"));
}

#[test]
fn test_format_markdown_with_exempt() {
    let mut report = ComplyReport::new();
    report.add_exemption("project", "rule");
    report.finalize();
    let md = report.format_markdown();
    assert!(md.contains("Exempt"));
}

#[test]
fn test_format_markdown_with_error() {
    let mut report = ComplyReport::new();
    report.add_error("project", "rule", "Error msg".to_string());
    report.finalize();
    let md = report.format_markdown();
    assert!(md.contains("Error"));
}

#[test]
fn test_format_markdown_with_violations() {
    let mut report = ComplyReport::new();
    let v = RuleViolation::new("V-001", "Test violation");
    report.add_result("project", "rule", RuleResult::fail(vec![v]));
    report.finalize();
    let md = report.format_markdown();
    assert!(md.contains("V-001"));
}

#[test]
fn test_format_html_with_violations() {
    let mut report = ComplyReport::new();
    let v = RuleViolation::new("V-001", "Test");
    report.add_result("project", "rule", RuleResult::fail(vec![v]));
    report.finalize();
    let html = report.format_html();
    assert!(html.contains("project"));
    assert!(html.contains("FAIL"));
}

#[test]
fn test_is_compliant_with_errors() {
    let mut report = ComplyReport::new();
    report.add_result("project", "rule", RuleResult::pass());
    report.add_global_error("Error".to_string());
    report.finalize();
    assert!(!report.is_compliant());
}

#[test]
fn test_compliance_summary_fields() {
    let summary = ComplianceSummary {
        total_projects: 10,
        passing_projects: 8,
        failing_projects: 2,
        total_checks: 40,
        passed_checks: 35,
        failed_checks: 5,
        total_violations: 15,
        violations_by_severity: HashMap::new(),
        fixable_violations: 10,
        pass_rate: 87.5,
    };
    assert_eq!(summary.total_projects, 10);
    assert_eq!(summary.pass_rate, 87.5);
}

#[test]
fn test_compliance_summary_clone() {
    let summary = ComplianceSummary::default();
    let cloned = summary.clone();
    assert_eq!(cloned.total_projects, summary.total_projects);
}

#[test]
fn test_project_rule_result_variants() {
    let checked = ProjectRuleResult::Checked(RuleResult::pass());
    let exempt = ProjectRuleResult::Exempt("reason".to_string());
    let error = ProjectRuleResult::Error("error".to_string());
    let fixed = ProjectRuleResult::Fixed(FixResult::success(1));
    let dry_run = ProjectRuleResult::DryRunFix(vec![]);

    assert!(matches!(checked, ProjectRuleResult::Checked(_)));
    assert!(matches!(exempt, ProjectRuleResult::Exempt(_)));
    assert!(matches!(error, ProjectRuleResult::Error(_)));
    assert!(matches!(fixed, ProjectRuleResult::Fixed(_)));
    assert!(matches!(dry_run, ProjectRuleResult::DryRunFix(_)));
}

#[test]
fn test_violations_empty() {
    let report = ComplyReport::new();
    assert!(report.violations().is_empty());
}

#[test]
fn test_finalize_zero_checks_pass_rate() {
    let mut report = ComplyReport::new();
    report.finalize();
    assert_eq!(report.summary.pass_rate, 100.0);
}
