use super::*;

fn test_entry(
    stem: &str,
    equations: Vec<&str>,
    obligs: usize,
    kani: usize,
) -> types::ContractEntry {
    types::ContractEntry {
        stem: stem.into(),
        path: format!("{stem}.yaml"),
        description: "desc".into(),
        equations: equations.into_iter().map(Into::into).collect(),
        obligation_types: vec![],
        properties: vec![],
        references: vec![],
        depends_on: vec![],
        is_registry: false,
        kind: Default::default(),
        obligation_count: obligs,
        falsification_count: 0,
        kani_count: kani,
        corpus_text: String::new(),
    }
}

#[test]
fn parse_iso_days_ago_valid_date() {
    let now = 1_767_225_600; // approx 2026-01-01 00:00 UTC
    let days = parse_iso_days_ago("2025-12-31", now);
    assert!(days <= 3, "Expected ~1 day, got {days}");
}

#[test]
fn parse_iso_days_ago_invalid_format() {
    // Empty and single-segment strings have < 3 parts → returns 0
    assert_eq!(parse_iso_days_ago("", 1_000_000), 0);
    assert_eq!(parse_iso_days_ago("2025", 1_000_000), 0);
    // "not-a-date" has 3 dash-separated parts but parse fails → doesn't panic
    let _ = parse_iso_days_ago("not-a-date", 1_000_000);
}

#[test]
fn parse_iso_days_ago_recent() {
    // Use a known epoch and date: 2025-01-02 vs now=2025-01-03
    let now = 1_735_862_400; // approx 2025-01-03 UTC
    let days = parse_iso_days_ago("2025-01-02", now);
    assert!(days <= 5, "Expected ~1 day, got {days}");
}

#[test]
fn month_days_boundaries() {
    assert_eq!(month_days(0), 0);
    assert_eq!(month_days(1), 0);
    assert_eq!(month_days(2), 31);
    assert_eq!(month_days(12), 334);
    assert_eq!(month_days(13), 0);
    assert_eq!(month_days(100), 0);
}

#[test]
fn parse_proof_level_all_variants() {
    assert_eq!(parse_proof_level("L1"), crate::proof_status::ProofLevel::L1);
    assert_eq!(parse_proof_level("L2"), crate::proof_status::ProofLevel::L2);
    assert_eq!(parse_proof_level("L3"), crate::proof_status::ProofLevel::L3);
    assert_eq!(parse_proof_level("L4"), crate::proof_status::ProofLevel::L4);
    assert_eq!(parse_proof_level("L5"), crate::proof_status::ProofLevel::L5);
}

#[test]
fn parse_proof_level_case_insensitive() {
    assert_eq!(parse_proof_level("l3"), crate::proof_status::ProofLevel::L3);
    assert_eq!(parse_proof_level("l5"), crate::proof_status::ProofLevel::L5);
}

#[test]
fn parse_proof_level_unknown_defaults_l1() {
    assert_eq!(parse_proof_level("L9"), crate::proof_status::ProofLevel::L1);
    assert_eq!(
        parse_proof_level("garbage"),
        crate::proof_status::ProofLevel::L1
    );
}

#[test]
fn filter_by_project_none_returns_all() {
    let sites = vec![
        CallSiteInfo {
            project: "alpha".into(),
            file: "a.rs".into(),
            line: 1,
            equation: Some("eq1".into()),
        },
        CallSiteInfo {
            project: "beta".into(),
            file: "b.rs".into(),
            line: 2,
            equation: Some("eq2".into()),
        },
    ];
    let result = filter_by_project(sites.clone(), None);
    assert_eq!(result.len(), 2);
}

#[test]
fn filter_by_project_some_filters() {
    let sites = vec![
        CallSiteInfo {
            project: "alpha".into(),
            file: "a.rs".into(),
            line: 1,
            equation: Some("eq1".into()),
        },
        CallSiteInfo {
            project: "beta".into(),
            file: "b.rs".into(),
            line: 2,
            equation: Some("eq2".into()),
        },
    ];
    let result = filter_by_project(sites, Some("alpha"));
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].project, "alpha");
}

#[test]
fn filter_by_project_no_match() {
    let sites = vec![CallSiteInfo {
        project: "alpha".into(),
        file: "a.rs".into(),
        line: 1,
        equation: Some("eq1".into()),
    }];
    let result = filter_by_project(sites, Some("gamma"));
    assert!(result.is_empty());
}

#[test]
fn filter_violations_none_returns_all() {
    let v = vec![
        ViolationInfo {
            project: "a".into(),
            kind: "gap".into(),
            detail: "d".into(),
        },
        ViolationInfo {
            project: "b".into(),
            kind: "gap".into(),
            detail: "d".into(),
        },
    ];
    let result = filter_violations(v, None);
    assert_eq!(result.len(), 2);
}

#[test]
fn filter_violations_some_filters() {
    let v = vec![
        ViolationInfo {
            project: "a".into(),
            kind: "gap".into(),
            detail: "d".into(),
        },
        ViolationInfo {
            project: "b".into(),
            kind: "gap".into(),
            detail: "d".into(),
        },
    ];
    let result = filter_violations(v, Some("a"));
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].project, "a");
}

#[test]
fn filter_coverage_none_returns_all() {
    let c = vec![
        ProjectCoverage {
            project: "x".into(),
            call_sites: 5,
            binding_refs: 2,
            binding_implemented: 1,
            binding_total: 2,
        },
        ProjectCoverage {
            project: "y".into(),
            call_sites: 3,
            binding_refs: 1,
            binding_implemented: 1,
            binding_total: 1,
        },
    ];
    let result = filter_coverage(c, None);
    assert_eq!(result.len(), 2);
}

#[test]
fn filter_coverage_some_filters() {
    let c = vec![
        ProjectCoverage {
            project: "x".into(),
            call_sites: 5,
            binding_refs: 2,
            binding_implemented: 1,
            binding_total: 2,
        },
        ProjectCoverage {
            project: "y".into(),
            call_sites: 3,
            binding_refs: 1,
            binding_implemented: 1,
            binding_total: 1,
        },
    ];
    let result = filter_coverage(c, Some("y"));
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].project, "y");
}

#[test]
fn build_call_sites_no_index() {
    let result = build_call_sites("softmax-kernel-v1", None);
    assert!(result.is_empty());
}

#[test]
fn build_violations_show_false() {
    let entry = test_entry("test", vec!["eq1"], 1, 0);
    let result = build_violations(&entry, None, false);
    assert!(result.is_empty());
}

#[test]
fn build_violations_show_true_no_index() {
    let entry = test_entry("test", vec!["eq1"], 1, 0);
    let result = build_violations(&entry, None, true);
    assert!(result.is_empty());
}

#[test]
fn build_coverage_map_show_false() {
    let result = build_coverage_map("test", None, false);
    assert!(result.is_empty());
}

#[test]
fn build_coverage_map_show_true_no_index() {
    let result = build_coverage_map("test", None, true);
    assert!(result.is_empty());
}

#[test]
fn build_binding_info_no_registry() {
    let entry = test_entry("test", vec!["alpha", "beta"], 0, 0);
    let result = build_binding_info(&entry, None);
    assert_eq!(result.len(), 2);
    assert_eq!(result[0].equation, "alpha");
    assert_eq!(result[0].status, "no binding registry");
    assert!(result[0].module_path.is_none());
}

#[test]
fn build_binding_info_with_registry() {
    let entry = test_entry("test", vec!["f", "g"], 0, 0);
    let binding = crate::binding::parse_binding_str(
        r#"
version: "1.0.0"
target_crate: test
bindings:
  - contract: test.yaml
    equation: f
    module_path: "test::f"
    function: f
    status: implemented
"#,
    )
    .unwrap();
    let result = build_binding_info(&entry, Some(&binding));
    assert_eq!(result.len(), 2);
    let f_binding = result.iter().find(|b| b.equation == "f").unwrap();
    assert_eq!(f_binding.status, "implemented");
    assert!(f_binding.module_path.is_some());
    let g_binding = result.iter().find(|b| b.equation == "g").unwrap();
    assert_eq!(g_binding.status, "unbound");
    assert!(g_binding.module_path.is_none());
}
