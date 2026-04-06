use super::*;
use crate::stack::graph::DependencyEdge;
use std::path::PathBuf;

fn create_test_graph() -> DependencyGraph {
    let mut graph = DependencyGraph::new();

    // trueno - healthy crate
    graph.add_crate(CrateInfo::new(
        "trueno",
        semver::Version::new(1, 2, 0),
        PathBuf::from("trueno/Cargo.toml"),
    ));

    // aprender - depends on trueno
    let mut aprender = CrateInfo::new(
        "aprender",
        semver::Version::new(0, 8, 1),
        PathBuf::from("aprender/Cargo.toml"),
    );
    aprender.paiml_dependencies.push(DependencyInfo::new("trueno", "^1.0"));
    graph.add_crate(aprender);

    // entrenar - has PATH DEPENDENCY (the bug!)
    let mut entrenar = CrateInfo::new(
        "entrenar",
        semver::Version::new(0, 2, 2),
        PathBuf::from("entrenar/Cargo.toml"),
    );
    entrenar.paiml_dependencies.push(DependencyInfo::new("aprender", "^0.8"));
    entrenar
        .paiml_dependencies
        .push(DependencyInfo::path("alimentar", PathBuf::from("../alimentar")));
    graph.add_crate(entrenar);

    // alimentar
    graph.add_crate(CrateInfo::new(
        "alimentar",
        semver::Version::new(0, 3, 0),
        PathBuf::from("alimentar/Cargo.toml"),
    ));

    // Add edges
    graph.add_dependency(
        "aprender",
        "trueno",
        DependencyEdge {
            version_req: "^1.0".to_string(),
            is_path: false,
            kind: DependencyKind::Normal,
        },
    );

    graph.add_dependency(
        "entrenar",
        "aprender",
        DependencyEdge {
            version_req: "^0.8".to_string(),
            is_path: false,
            kind: DependencyKind::Normal,
        },
    );

    graph.add_dependency(
        "entrenar",
        "alimentar",
        DependencyEdge { version_req: String::new(), is_path: true, kind: DependencyKind::Normal },
    );

    graph.add_dependency(
        "alimentar",
        "trueno",
        DependencyEdge {
            version_req: "^1.0".to_string(),
            is_path: false,
            kind: DependencyKind::Normal,
        },
    );

    graph
}

fn create_mock_client() -> MockCratesIoClient {
    let mut mock = MockCratesIoClient::new();
    mock.add_crate("trueno", "1.2.0")
        .add_crate("aprender", "0.8.1")
        .add_crate("alimentar", "0.3.0")
        .add_crate("entrenar", "0.2.2");
    mock
}

#[test]
fn test_checker_creation() {
    let graph = create_test_graph();
    let checker = StackChecker::with_graph(graph);
    assert_eq!(checker.crate_count(), 4);
}

#[test]
fn test_checker_finds_path_dependencies() {
    let graph = create_test_graph();
    let mut checker = StackChecker::with_graph(graph);
    let mock = create_mock_client();

    let report = checker.check_with_mock(&mock).unwrap();

    // Find entrenar in the report
    let entrenar = report.crates.iter().find(|c| c.name == "entrenar").unwrap();

    // Should have path dependency issue
    assert!(entrenar.issues.iter().any(|i| i.issue_type == IssueType::PathDependency));
    assert_eq!(entrenar.status, CrateStatus::Error);
}

#[test]
fn test_checker_healthy_crates() {
    let graph = create_test_graph();
    let mut checker = StackChecker::with_graph(graph);
    let mock = create_mock_client();

    let report = checker.check_with_mock(&mock).unwrap();

    // trueno should be healthy
    let trueno = report.crates.iter().find(|c| c.name == "trueno").unwrap();
    assert_eq!(trueno.status, CrateStatus::Healthy);
    assert!(trueno.issues.is_empty());
}

#[test]
fn test_checker_crates_io_versions() {
    let graph = create_test_graph();
    let mut checker = StackChecker::with_graph(graph).verify_published(true);
    let mock = create_mock_client();

    let report = checker.check_with_mock(&mock).unwrap();

    // All crates should have crates.io versions
    for crate_info in &report.crates {
        assert!(crate_info.crates_io_version.is_some());
    }
}

#[test]
fn test_checker_strict_mode() {
    let mut graph = DependencyGraph::new();

    // Create crate with warning (version behind)
    let crate_info = CrateInfo::new(
        "test-crate",
        semver::Version::new(0, 9, 0), // Behind crates.io
        PathBuf::new(),
    );
    graph.add_crate(crate_info);

    let mut mock = MockCratesIoClient::new();
    mock.add_crate("test-crate", "1.0.0"); // Ahead

    // Non-strict mode
    let mut checker = StackChecker::with_graph(graph.clone()).verify_published(true);
    let report = checker.check_with_mock(&mock).unwrap();
    let crate_info = report.crates.iter().find(|c| c.name == "test-crate").unwrap();
    assert_eq!(crate_info.status, CrateStatus::Warning);

    // Strict mode
    let mut checker = StackChecker::with_graph(graph).verify_published(true).strict(true);
    let report = checker.check_with_mock(&mock).unwrap();
    let crate_info = report.crates.iter().find(|c| c.name == "test-crate").unwrap();
    assert_eq!(crate_info.status, CrateStatus::Error); // Warning becomes error in strict mode
}

#[test]
fn test_checker_release_order() {
    let graph = create_test_graph();
    let checker = StackChecker::with_graph(graph);

    let order = checker.release_order_for("entrenar").unwrap();

    // trueno should be first
    assert_eq!(order[0], "trueno");

    // entrenar should be last
    assert_eq!(order.last().unwrap(), "entrenar");

    // All dependencies should be included
    assert!(order.contains(&"aprender".to_string()));
    assert!(order.contains(&"alimentar".to_string()));
}

#[test]
fn test_health_report_is_healthy() {
    let mut graph = DependencyGraph::new();

    let mut healthy_crate =
        CrateInfo::new("healthy", semver::Version::new(1, 0, 0), PathBuf::new());
    healthy_crate.status = CrateStatus::Healthy;
    graph.add_crate(healthy_crate);

    let report = StackHealthReport::new(graph.all_crates().cloned().collect(), vec![]);

    assert!(report.is_healthy());
}

#[test]
fn test_health_report_not_healthy_with_errors() {
    let graph = create_test_graph();
    let mut checker = StackChecker::with_graph(graph);
    let mock = create_mock_client();

    let report = checker.check_with_mock(&mock).unwrap();

    // Should not be healthy due to path dependency in entrenar
    assert!(!report.is_healthy());
}

#[test]
fn test_format_report_text() {
    let graph = create_test_graph();
    let mut checker = StackChecker::with_graph(graph);
    let mock = create_mock_client();

    let report = checker.check_with_mock(&mock).unwrap();
    let text = format_report_text(&report);

    // Should contain key information
    assert!(text.contains("PAIML Stack Health Check"));
    assert!(text.contains("trueno"));
    assert!(text.contains("entrenar"));
    assert!(text.contains("Summary:"));
}

#[test]
fn test_format_report_json() {
    let graph = create_test_graph();
    let mut checker = StackChecker::with_graph(graph);
    let mock = create_mock_client();

    let report = checker.check_with_mock(&mock).unwrap();
    let json = format_report_json(&report).unwrap();

    // Should be valid JSON
    let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert!(parsed.get("crates").is_some());
    assert!(parsed.get("summary").is_some());
}

#[test]
fn test_version_conflict_detection() {
    let mut graph = DependencyGraph::new();

    // Create crates with conflicting arrow versions
    let mut crate_a = CrateInfo::new("a", semver::Version::new(1, 0, 0), PathBuf::new());
    crate_a.external_dependencies.push(DependencyInfo::new("arrow", "54.0"));
    graph.add_crate(crate_a);

    let mut crate_b = CrateInfo::new("b", semver::Version::new(1, 0, 0), PathBuf::new());
    crate_b.external_dependencies.push(DependencyInfo::new("arrow", "53.0"));
    graph.add_crate(crate_b);

    let mut checker = StackChecker::with_graph(graph);
    let mock = MockCratesIoClient::new();

    let report = checker.check_with_mock(&mock).unwrap();

    // Should detect conflict
    assert_eq!(report.conflicts.len(), 1);
    assert_eq!(report.conflicts[0].dependency, "arrow");
}

#[test]
fn test_determine_status() {
    // No issues = healthy
    assert_eq!(StackChecker::determine_status(&[], false), CrateStatus::Healthy);

    // Info only = healthy
    let info_issue = CrateIssue::new(IssueSeverity::Info, IssueType::NotPublished, "test");
    assert_eq!(StackChecker::determine_status(&[info_issue], false), CrateStatus::Healthy);

    // Warning = warning (non-strict)
    let warning_issue = CrateIssue::new(IssueSeverity::Warning, IssueType::VersionBehind, "test");
    assert_eq!(
        StackChecker::determine_status(std::slice::from_ref(&warning_issue), false),
        CrateStatus::Warning
    );

    // Warning = error (strict)
    assert_eq!(StackChecker::determine_status(&[warning_issue], true), CrateStatus::Error);

    // Error = error
    let error_issue = CrateIssue::new(IssueSeverity::Error, IssueType::PathDependency, "test");
    assert_eq!(StackChecker::determine_status(&[error_issue], false), CrateStatus::Error);
}

#[test]
fn test_format_report_markdown() {
    let graph = create_test_graph();
    let mut checker = StackChecker::with_graph(graph);
    let mock = create_mock_client();

    let report = checker.check_with_mock(&mock).unwrap();
    let md = format_report_markdown(&report);

    // Should contain markdown table
    assert!(md.contains("# PAIML Stack Health Report"));
    assert!(md.contains("| Status | Crate | Version | Crates.io |"));
    assert!(md.contains("## Summary"));
    assert!(md.contains("**Total crates**"));
}

#[test]
fn test_format_report_markdown_healthy() {
    let mut graph = DependencyGraph::new();
    let mut healthy_crate =
        CrateInfo::new("healthy", semver::Version::new(1, 0, 0), PathBuf::new());
    healthy_crate.status = CrateStatus::Healthy;
    graph.add_crate(healthy_crate);

    let report = StackHealthReport::new(graph.all_crates().cloned().collect(), vec![]);
    let md = format_report_markdown(&report);

    assert!(md.contains("✅ **All crates are healthy**"));
}

#[test]
fn test_format_report_markdown_unhealthy() {
    let graph = create_test_graph();
    let mut checker = StackChecker::with_graph(graph);
    let mock = create_mock_client();

    let report = checker.check_with_mock(&mock).unwrap();
    let md = format_report_markdown(&report);

    assert!(md.contains("⚠️ **Some crates need attention**"));
}

#[test]
fn test_format_report_markdown_with_issues() {
    let mut graph = DependencyGraph::new();
    let mut crate_with_issues =
        CrateInfo::new("broken", semver::Version::new(1, 0, 0), PathBuf::new());
    crate_with_issues.status = CrateStatus::Error;
    crate_with_issues.issues.push(CrateIssue::new(
        IssueSeverity::Error,
        IssueType::PathDependency,
        "Path dependency detected",
    ));
    graph.add_crate(crate_with_issues);

    let report = StackHealthReport::new(graph.all_crates().cloned().collect(), vec![]);
    let md = format_report_markdown(&report);

    assert!(md.contains("❌"));
    assert!(md.contains("Path dependency detected"));
}

#[test]
fn test_format_report_text_with_suggestion() {
    let mut graph = DependencyGraph::new();
    let mut crate_with_suggestion =
        CrateInfo::new("suggest", semver::Version::new(1, 0, 0), PathBuf::new());
    crate_with_suggestion.status = CrateStatus::Warning;
    let issue = CrateIssue::new(IssueSeverity::Warning, IssueType::VersionBehind, "Version behind")
        .with_suggestion("Update to 2.0.0".to_string());
    crate_with_suggestion.issues.push(issue);
    graph.add_crate(crate_with_suggestion);

    let report = StackHealthReport::new(graph.all_crates().cloned().collect(), vec![]);
    let text = format_report_text(&report);

    assert!(text.contains("→ Update to 2.0.0"));
}

#[test]
fn test_format_report_text_with_conflicts() {
    let mut graph = DependencyGraph::new();
    graph.add_crate(CrateInfo::new("a", semver::Version::new(1, 0, 0), PathBuf::new()));

    let conflicts = vec![VersionConflict {
        dependency: "arrow".to_string(),
        usages: vec![
            ConflictUsage { crate_name: "a".to_string(), version_req: "54.0".to_string() },
            ConflictUsage { crate_name: "b".to_string(), version_req: "53.0".to_string() },
        ],
        recommendation: Some("Use 54.0 everywhere".to_string()),
    }];

    let report = StackHealthReport::new(graph.all_crates().cloned().collect(), conflicts);
    let text = format_report_text(&report);

    assert!(text.contains("Version Conflicts:"));
    assert!(text.contains("arrow conflict:"));
    assert!(text.contains("Recommendation: Use 54.0 everywhere"));
}

#[test]
fn test_format_report_text_path_dependency_count() {
    let mut graph = DependencyGraph::new();
    let mut crate_with_path =
        CrateInfo::new("pathcrate", semver::Version::new(1, 0, 0), PathBuf::new());
    crate_with_path.status = CrateStatus::Error;
    // has_path_dependencies() checks paiml_dependencies/external_dependencies
    crate_with_path
        .paiml_dependencies
        .push(DependencyInfo::path("somelib", PathBuf::from("../somelib")));
    graph.add_crate(crate_with_path);

    let report = StackHealthReport::new(graph.all_crates().cloned().collect(), vec![]);
    let text = format_report_text(&report);

    assert!(text.contains("Path dependencies: 1"));
}

#[test]
fn test_format_report_text_unknown_status() {
    let mut graph = DependencyGraph::new();
    let mut crate_unknown =
        CrateInfo::new("unknown", semver::Version::new(1, 0, 0), PathBuf::new());
    crate_unknown.status = CrateStatus::Unknown;
    graph.add_crate(crate_unknown);

    let report = StackHealthReport::new(graph.all_crates().cloned().collect(), vec![]);
    let text = format_report_text(&report);

    assert!(text.contains("❓"));
}

#[test]
fn test_format_report_markdown_unknown_status() {
    let mut graph = DependencyGraph::new();
    let mut crate_unknown =
        CrateInfo::new("unknown", semver::Version::new(1, 0, 0), PathBuf::new());
    crate_unknown.status = CrateStatus::Unknown;
    graph.add_crate(crate_unknown);

    let report = StackHealthReport::new(graph.all_crates().cloned().collect(), vec![]);
    let md = format_report_markdown(&report);

    assert!(md.contains("❓"));
}

#[test]
fn test_checker_topological_order() {
    let graph = create_test_graph();
    let checker = StackChecker::with_graph(graph);

    let order = checker.topological_order().unwrap();

    // trueno should be first (has no dependencies)
    assert_eq!(order[0], "trueno");

    // All crates should be present
    assert!(order.contains(&"aprender".to_string()));
    assert!(order.contains(&"entrenar".to_string()));
    assert!(order.contains(&"alimentar".to_string()));
}

#[test]
fn test_checker_not_published_detection() {
    let mut graph = DependencyGraph::new();
    graph.add_crate(CrateInfo::new("unpublished", semver::Version::new(1, 0, 0), PathBuf::new()));

    // Empty mock = nothing published
    let mock = MockCratesIoClient::new();
    let mut checker = StackChecker::with_graph(graph).verify_published(true);

    let report = checker.check_with_mock(&mock).unwrap();
    let crate_info = report.crates.iter().find(|c| c.name == "unpublished").unwrap();

    assert!(crate_info.issues.iter().any(|i| i.issue_type == IssueType::NotPublished));
}

#[test]
fn test_checker_version_behind_detection() {
    let mut graph = DependencyGraph::new();
    graph.add_crate(CrateInfo::new("behind", semver::Version::new(0, 9, 0), PathBuf::new()));

    let mut mock = MockCratesIoClient::new();
    mock.add_crate("behind", "1.0.0");

    let mut checker = StackChecker::with_graph(graph).verify_published(true);
    let report = checker.check_with_mock(&mock).unwrap();

    let crate_info = report.crates.iter().find(|c| c.name == "behind").unwrap();
    assert!(crate_info.issues.iter().any(|i| i.issue_type == IssueType::VersionBehind));
}

#[test]
fn test_format_report_text_not_published() {
    let mut graph = DependencyGraph::new();
    let mut crate_info = CrateInfo::new("notpub", semver::Version::new(1, 0, 0), PathBuf::new());
    crate_info.crates_io_version = None;
    graph.add_crate(crate_info);

    let report = StackHealthReport::new(graph.all_crates().cloned().collect(), vec![]);
    let text = format_report_text(&report);

    assert!(text.contains("(not published)"));
}

#[test]
fn test_format_report_markdown_not_published() {
    let mut graph = DependencyGraph::new();
    let mut crate_info = CrateInfo::new("notpub", semver::Version::new(1, 0, 0), PathBuf::new());
    crate_info.crates_io_version = None;
    graph.add_crate(crate_info);

    let report = StackHealthReport::new(graph.all_crates().cloned().collect(), vec![]);
    let md = format_report_markdown(&report);

    assert!(md.contains("not published"));
}

#[test]
fn test_format_report_markdown_warning_issue() {
    let mut graph = DependencyGraph::new();
    let mut crate_info = CrateInfo::new("warncrate", semver::Version::new(1, 0, 0), PathBuf::new());
    crate_info.status = CrateStatus::Warning;
    crate_info.issues.push(CrateIssue::new(
        IssueSeverity::Warning,
        IssueType::VersionBehind,
        "Version behind",
    ));
    graph.add_crate(crate_info);

    let report = StackHealthReport::new(graph.all_crates().cloned().collect(), vec![]);
    let md = format_report_markdown(&report);

    assert!(md.contains("⚠️"));
}

#[test]
fn test_format_report_markdown_info_issue() {
    let mut graph = DependencyGraph::new();
    let mut crate_info = CrateInfo::new("infocrate", semver::Version::new(1, 0, 0), PathBuf::new());
    crate_info.issues.push(CrateIssue::new(
        IssueSeverity::Info,
        IssueType::NotPublished,
        "Not published",
    ));
    graph.add_crate(crate_info);

    let report = StackHealthReport::new(graph.all_crates().cloned().collect(), vec![]);
    let md = format_report_markdown(&report);

    assert!(md.contains("ℹ️"));
}

#[test]
fn test_format_report_text_info_issue() {
    let mut graph = DependencyGraph::new();
    let mut crate_info = CrateInfo::new("infocrate", semver::Version::new(1, 0, 0), PathBuf::new());
    crate_info.issues.push(CrateIssue::new(
        IssueSeverity::Info,
        IssueType::NotPublished,
        "Not published",
    ));
    graph.add_crate(crate_info);

    let report = StackHealthReport::new(graph.all_crates().cloned().collect(), vec![]);
    let text = format_report_text(&report);

    assert!(text.contains("ℹ"));
}

// =========================================================================
// Coverage gap tests: StackChecker::find_path_dependencies delegate
// =========================================================================

/// Test find_path_dependencies on StackChecker (covers lines 220-222 in checker.rs).
#[test]
fn test_checker_find_path_dependencies_delegate() {
    let graph = create_test_graph();
    let checker = StackChecker::with_graph(graph);

    let path_deps = checker.find_path_dependencies();

    // The test graph has entrenar -> alimentar as path dependency
    assert_eq!(path_deps.len(), 1);
    assert_eq!(path_deps[0].crate_name, "entrenar");
    assert_eq!(path_deps[0].dependency, "alimentar");
}

/// Test find_path_dependencies on StackChecker with no path deps.
#[test]
fn test_checker_find_path_dependencies_empty() {
    let mut graph = DependencyGraph::new();
    graph.add_crate(CrateInfo::new("clean", semver::Version::new(1, 0, 0), PathBuf::new()));

    let checker = StackChecker::with_graph(graph);
    let path_deps = checker.find_path_dependencies();
    assert!(path_deps.is_empty());
}

/// Test get_crate on StackChecker.
#[test]
fn test_checker_get_crate_delegate() {
    let graph = create_test_graph();
    let checker = StackChecker::with_graph(graph);

    let trueno = checker.get_crate("trueno");
    assert!(trueno.is_some());
    assert_eq!(trueno.unwrap().name, "trueno");

    let missing = checker.get_crate("nonexistent");
    assert!(missing.is_none());
}

// =========================================================================
// Coverage gap tests: StackChecker::from_workspace (native-only)
// =========================================================================

/// Test from_workspace on StackChecker using the batuta project.
#[cfg(feature = "native")]
#[test]
fn test_checker_from_workspace_batuta() {
    let workspace_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let result = StackChecker::from_workspace(workspace_path);
    assert!(result.is_ok(), "from_workspace failed: {:?}", result.err());

    let checker = result.unwrap();
    assert!(checker.crate_count() >= 1);
    assert!(checker.get_crate("batuta").is_some());
}

/// Test from_workspace on StackChecker with invalid Cargo.toml.
///
/// A nonexistent path falls through to `from_installed_binaries()` which
/// may return Ok.  To test the error path we create a directory with a
/// malformed Cargo.toml so `cargo metadata` fails.
#[cfg(feature = "native")]
#[test]
fn test_checker_from_workspace_invalid() {
    let dir = std::env::temp_dir().join("batuta_checker_invalid_ws");
    let _ = std::fs::create_dir_all(&dir);
    std::fs::write(dir.join("Cargo.toml"), "THIS IS NOT VALID TOML {{{")
        .expect("write invalid Cargo.toml");

    let result = StackChecker::from_workspace(&dir);
    assert!(result.is_err(), "malformed Cargo.toml should produce an error");

    let _ = std::fs::remove_dir_all(&dir);
}

// =========================================================================
// Coverage gap tests: version conflict involvement in checker
// =========================================================================

/// Test that a crate involved in a version conflict gets the VersionConflict issue.
#[test]
fn test_checker_version_conflict_issue_on_crate() {
    let mut graph = DependencyGraph::new();

    let mut crate_a = CrateInfo::new("a", semver::Version::new(1, 0, 0), PathBuf::new());
    crate_a.external_dependencies.push(DependencyInfo::new("serde", "1.0"));
    graph.add_crate(crate_a);

    let mut crate_b = CrateInfo::new("b", semver::Version::new(1, 0, 0), PathBuf::new());
    crate_b.external_dependencies.push(DependencyInfo::new("serde", "2.0"));
    graph.add_crate(crate_b);

    let mut checker = StackChecker::with_graph(graph);
    let mock = MockCratesIoClient::new();

    let report = checker.check_with_mock(&mock).unwrap();

    // Both crates should have version conflict issues
    for name in &["a", "b"] {
        let crate_info = report.crates.iter().find(|c| c.name == *name).unwrap();
        assert!(
            crate_info.issues.iter().any(|i| i.issue_type == IssueType::VersionConflict),
            "Crate '{}' should have VersionConflict issue",
            name
        );
    }
}

/// Test path dependency with suggestion when crates.io version is available.
/// This covers lines 108-111 in checker.rs (the suggestion branch).
#[test]
fn test_checker_path_dep_with_crates_io_suggestion() {
    let mut graph = DependencyGraph::new();

    // Create a crate with a path dependency
    let mut main_crate = CrateInfo::new("entrenar", semver::Version::new(0, 2, 0), PathBuf::new());
    main_crate
        .paiml_dependencies
        .push(DependencyInfo::path("alimentar", PathBuf::from("../alimentar")));
    graph.add_crate(main_crate);

    graph.add_crate(CrateInfo::new("alimentar", semver::Version::new(0, 3, 0), PathBuf::new()));

    // Add path dependency edge
    graph.add_dependency(
        "entrenar",
        "alimentar",
        DependencyEdge { version_req: String::new(), is_path: true, kind: DependencyKind::Normal },
    );

    // Mock with crates.io version for entrenar (the one with the path dep)
    let mut mock = MockCratesIoClient::new();
    mock.add_crate("entrenar", "0.2.0");
    mock.add_crate("alimentar", "0.3.0");

    let mut checker = StackChecker::with_graph(graph).verify_published(true);
    let report = checker.check_with_mock(&mock).unwrap();

    // entrenar should have a path dependency issue
    let entrenar = report.crates.iter().find(|c| c.name == "entrenar").unwrap();
    let path_issue = entrenar.issues.iter().find(|i| i.issue_type == IssueType::PathDependency);
    assert!(path_issue.is_some(), "Expected PathDependency issue");

    // The issue should have a suggestion since crates.io version is available
    let issue = path_issue.unwrap();
    assert!(
        issue.suggestion.is_some(),
        "Expected suggestion for path dependency when crates.io version is available"
    );
}

/// Test format_conflicts_text with a conflict that has no recommendation (None).
#[test]
fn test_format_conflicts_text_no_recommendation() {
    let mut graph = DependencyGraph::new();
    graph.add_crate(CrateInfo::new("a", semver::Version::new(1, 0, 0), PathBuf::new()));

    let conflicts = vec![VersionConflict {
        dependency: "tokio".to_string(),
        usages: vec![
            ConflictUsage { crate_name: "a".to_string(), version_req: "1.0".to_string() },
            ConflictUsage { crate_name: "b".to_string(), version_req: "2.0".to_string() },
        ],
        recommendation: None,
    }];

    let report = StackHealthReport::new(graph.all_crates().cloned().collect(), conflicts);
    let text = format_report_text(&report);

    assert!(text.contains("Version Conflicts:"));
    assert!(text.contains("tokio conflict:"));
    // Should NOT contain recommendation line
    assert!(!text.contains("Recommendation:"));
}

// =========================================================================
// Coverage gap tests: StackChecker::check async method (native-only)
// =========================================================================

/// Test the async `check` method with verify_published=false.
/// This covers lines 66-68, 78-79 of checker.rs (the async function entry,
/// HashMap creation, and the final run_checks call).
#[cfg(feature = "native")]
#[tokio::test]
async fn test_checker_check_async_no_verify() {
    let graph = create_test_graph();
    let mut checker = StackChecker::with_graph(graph);
    // verify_published defaults to false, so the loop body is skipped
    let mut client = CratesIoClient::new();

    let report = checker.check(&mut client).await.unwrap();

    // Should still produce a valid report with all crates
    assert_eq!(report.crates.len(), 4);

    // entrenar should still have path dependency issue (detected by run_checks)
    let entrenar = report.crates.iter().find(|c| c.name == "entrenar").unwrap();
    assert!(entrenar.issues.iter().any(|i| i.issue_type == IssueType::PathDependency));
}

/// Test the async `check` method with verify_published=true.
/// This covers lines 70-76 of checker.rs (the crates.io fetch loop).
#[cfg(feature = "native")]
#[tokio::test]
async fn test_checker_check_async_with_verify_published() {
    let graph = create_test_graph();
    let mut checker = StackChecker::with_graph(graph).verify_published(true);
    let mut client = CratesIoClient::new();

    let report = checker.check(&mut client).await.unwrap();

    // Should still produce a valid report
    assert!(!report.crates.is_empty());

    // With real crates.io lookup, trueno/aprender should have remote versions
    // (these are real published crates), but the test graph uses
    // artificial names that may or may not exist on crates.io.
    // The key assertion is that the method completes without error.
    assert_eq!(report.crates.len(), 4);
}
