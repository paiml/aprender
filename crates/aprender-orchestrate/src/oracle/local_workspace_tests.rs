use super::*;

#[test]
fn test_version_compare() {
    use std::cmp::Ordering;
    assert_eq!(compare_versions("1.0.0", "1.0.0"), Ordering::Equal);
    assert_eq!(compare_versions("1.0.1", "1.0.0"), Ordering::Greater);
    assert_eq!(compare_versions("1.0.0", "1.0.1"), Ordering::Less);
    assert_eq!(compare_versions("2.0.0", "1.9.9"), Ordering::Greater);
    assert_eq!(compare_versions("0.1.0", "0.0.9"), Ordering::Greater);
}

#[test]
fn test_paiml_crate_detection() {
    assert!(is_paiml_crate("trueno"));
    assert!(is_paiml_crate("aprender"));
    assert!(is_paiml_crate("bashrs"));
    assert!(!is_paiml_crate("serde"));
}

#[test]
fn test_drift_type_variants() {
    assert_ne!(DriftType::LocalAhead, DriftType::LocalBehind);
    assert_ne!(DriftType::InSync, DriftType::NotPublished);
}

#[test]
fn test_workspace_summary_default() {
    let summary = WorkspaceSummary {
        total_projects: 0,
        projects_with_changes: 0,
        projects_with_unpushed: 0,
        workspace_count: 0,
    };
    assert_eq!(summary.total_projects, 0);
}

#[test]
fn test_dev_state_use_local_version() {
    assert!(DevState::Clean.use_local_version());
    assert!(!DevState::Dirty.use_local_version());
    assert!(!DevState::Unpushed.use_local_version());
}

#[test]
fn test_dev_state_safe_to_release() {
    assert!(DevState::Clean.safe_to_release());
    assert!(!DevState::Dirty.safe_to_release());
    assert!(DevState::Unpushed.safe_to_release());
}

#[test]
fn test_local_project_effective_version() {
    let project = LocalProject {
        name: "test".to_string(),
        path: PathBuf::from("/tmp/test"),
        local_version: "1.0.0".to_string(),
        published_version: Some("0.9.0".to_string()),
        git_status: GitStatus {
            branch: "main".to_string(),
            has_changes: false,
            modified_count: 0,
            unpushed_commits: 0,
            up_to_date: true,
        },
        dev_state: DevState::Clean,
        paiml_dependencies: vec![],
        is_workspace: false,
        workspace_members: vec![],
    };
    assert_eq!(project.effective_version(), "1.0.0");

    let dirty_project = LocalProject { dev_state: DevState::Dirty, ..project.clone() };
    assert_eq!(dirty_project.effective_version(), "0.9.0");
}

#[test]
fn test_local_project_is_blocking() {
    let project = LocalProject {
        name: "test".to_string(),
        path: PathBuf::from("/tmp/test"),
        local_version: "0.8.0".to_string(),
        published_version: Some("0.9.0".to_string()),
        git_status: GitStatus {
            branch: "main".to_string(),
            has_changes: false,
            modified_count: 0,
            unpushed_commits: 0,
            up_to_date: true,
        },
        dev_state: DevState::Clean,
        paiml_dependencies: vec![],
        is_workspace: false,
        workspace_members: vec![],
    };
    // Clean project with local behind published is blocking
    assert!(project.is_blocking());

    let dirty_project = LocalProject { dev_state: DevState::Dirty, ..project.clone() };
    // Dirty projects don't block
    assert!(!dirty_project.is_blocking());

    let ahead_project = LocalProject { local_version: "1.0.0".to_string(), ..project.clone() };
    // Local ahead is not blocking
    assert!(!ahead_project.is_blocking());
}

#[test]
fn test_local_project_is_blocking_no_published() {
    let project = LocalProject {
        name: "test".to_string(),
        path: PathBuf::from("/tmp/test"),
        local_version: "1.0.0".to_string(),
        published_version: None,
        git_status: GitStatus {
            branch: "main".to_string(),
            has_changes: false,
            modified_count: 0,
            unpushed_commits: 0,
            up_to_date: true,
        },
        dev_state: DevState::Clean,
        paiml_dependencies: vec![],
        is_workspace: false,
        workspace_members: vec![],
    };
    // Not published => not blocking
    assert!(!project.is_blocking());
}

#[test]
fn test_git_status_fields() {
    let status = GitStatus {
        branch: "feature".to_string(),
        has_changes: true,
        modified_count: 5,
        unpushed_commits: 2,
        up_to_date: false,
    };
    assert_eq!(status.branch, "feature");
    assert!(status.has_changes);
    assert_eq!(status.modified_count, 5);
    assert_eq!(status.unpushed_commits, 2);
    assert!(!status.up_to_date);
}

#[test]
fn test_dependency_info_fields() {
    let dep = DependencyInfo {
        name: "trueno".to_string(),
        required_version: "0.14.0".to_string(),
        is_path_dep: true,
        version_satisfied: Some(true),
    };
    assert_eq!(dep.name, "trueno");
    assert!(dep.is_path_dep);
    assert_eq!(dep.version_satisfied, Some(true));
}

#[test]
fn test_version_drift_fields() {
    let drift = VersionDrift {
        name: "aprender".to_string(),
        local_version: "0.25.0".to_string(),
        published_version: "0.24.0".to_string(),
        drift_type: DriftType::LocalAhead,
    };
    assert_eq!(drift.name, "aprender");
    assert_eq!(drift.drift_type, DriftType::LocalAhead);
}

#[test]
fn test_publish_order_fields() {
    let order = PublishOrder {
        order: vec![PublishStep {
            name: "trueno".to_string(),
            version: "0.14.0".to_string(),
            blocked_by: vec![],
            needs_publish: true,
        }],
        cycles: vec![],
    };
    assert_eq!(order.order.len(), 1);
    assert!(order.cycles.is_empty());
}

#[test]
fn test_publish_step_fields() {
    let step = PublishStep {
        name: "aprender".to_string(),
        version: "0.24.0".to_string(),
        blocked_by: vec!["trueno".to_string()],
        needs_publish: false,
    };
    assert_eq!(step.blocked_by.len(), 1);
    assert!(!step.needs_publish);
}

#[test]
fn test_workspace_summary_fields() {
    let summary = WorkspaceSummary {
        total_projects: 10,
        projects_with_changes: 3,
        projects_with_unpushed: 2,
        workspace_count: 1,
    };
    assert_eq!(summary.total_projects, 10);
    assert_eq!(summary.projects_with_changes, 3);
    assert_eq!(summary.projects_with_unpushed, 2);
    assert_eq!(summary.workspace_count, 1);
}

#[test]
fn test_drift_type_equality() {
    assert_eq!(DriftType::LocalAhead, DriftType::LocalAhead);
    assert_eq!(DriftType::LocalBehind, DriftType::LocalBehind);
    assert_eq!(DriftType::InSync, DriftType::InSync);
    assert_eq!(DriftType::NotPublished, DriftType::NotPublished);
}

#[test]
fn test_dev_state_equality() {
    assert_eq!(DevState::Clean, DevState::Clean);
    assert_eq!(DevState::Dirty, DevState::Dirty);
    assert_eq!(DevState::Unpushed, DevState::Unpushed);
    assert_ne!(DevState::Clean, DevState::Dirty);
}

#[test]
fn test_compare_versions_edge_cases() {
    use std::cmp::Ordering;
    // Short versions
    assert_eq!(compare_versions("1", "1.0.0"), Ordering::Equal);
    assert_eq!(compare_versions("1.0", "1.0.0"), Ordering::Equal);
    // Invalid versions default to 0
    assert_eq!(compare_versions("invalid", "0.0.0"), Ordering::Equal);
}

#[test]
fn test_local_project_effective_version_unpushed() {
    let project = LocalProject {
        name: "test".to_string(),
        path: PathBuf::from("/tmp/test"),
        local_version: "1.0.0".to_string(),
        published_version: Some("0.9.0".to_string()),
        git_status: GitStatus {
            branch: "main".to_string(),
            has_changes: false,
            modified_count: 0,
            unpushed_commits: 1,
            up_to_date: false,
        },
        dev_state: DevState::Unpushed,
        paiml_dependencies: vec![],
        is_workspace: false,
        workspace_members: vec![],
    };
    // Unpushed uses published version
    assert_eq!(project.effective_version(), "0.9.0");
}

#[test]
fn test_local_project_effective_version_no_published() {
    let project = LocalProject {
        name: "test".to_string(),
        path: PathBuf::from("/tmp/test"),
        local_version: "1.0.0".to_string(),
        published_version: None,
        git_status: GitStatus {
            branch: "main".to_string(),
            has_changes: true,
            modified_count: 1,
            unpushed_commits: 0,
            up_to_date: false,
        },
        dev_state: DevState::Dirty,
        paiml_dependencies: vec![],
        is_workspace: false,
        workspace_members: vec![],
    };
    // Dirty with no published falls back to local
    assert_eq!(project.effective_version(), "1.0.0");
}

#[test]
fn test_local_workspace_oracle_with_base_dir() {
    let temp_dir = std::env::temp_dir();
    let oracle = LocalWorkspaceOracle::with_base_dir(temp_dir.clone()).unwrap();
    assert!(oracle.projects().is_empty());
}

#[test]
fn test_local_workspace_oracle_summary_empty() {
    let oracle = LocalWorkspaceOracle::with_base_dir(std::env::temp_dir()).unwrap();
    let summary = oracle.summary();
    assert_eq!(summary.total_projects, 0);
    assert_eq!(summary.projects_with_changes, 0);
}

#[test]
fn test_local_workspace_oracle_detect_drift_empty() {
    let oracle = LocalWorkspaceOracle::with_base_dir(std::env::temp_dir()).unwrap();
    let drifts = oracle.detect_drift();
    assert!(drifts.is_empty());
}

#[test]
fn test_local_workspace_oracle_suggest_publish_order_empty() {
    let oracle = LocalWorkspaceOracle::with_base_dir(std::env::temp_dir()).unwrap();
    let order = oracle.suggest_publish_order();
    assert!(order.order.is_empty());
    assert!(order.cycles.is_empty());
}

// =========================================================================
// Coverage Gap Tests — extract_version
// =========================================================================

#[test]
fn test_extract_version_simple() {
    let parsed: toml::Value = toml::from_str(
        r#"
        [package]
        name = "test"
        version = "1.2.3"
    "#,
    )
    .unwrap();
    let package = parsed.get("package").unwrap();
    let version = LocalWorkspaceOracle::extract_version(package, &parsed);
    assert_eq!(version, "1.2.3");
}

#[test]
fn test_extract_version_workspace_inheritance() {
    let parsed: toml::Value = toml::from_str(
        r#"
        [package]
        name = "test"
        version.workspace = true

        [workspace.package]
        version = "4.5.6"
    "#,
    )
    .unwrap();
    let package = parsed.get("package").unwrap();
    let version = LocalWorkspaceOracle::extract_version(package, &parsed);
    assert_eq!(version, "4.5.6");
}

#[test]
fn test_extract_version_missing_defaults() {
    let parsed: toml::Value = toml::from_str(
        r#"
        [package]
        name = "test"
    "#,
    )
    .unwrap();
    let package = parsed.get("package").unwrap();
    let version = LocalWorkspaceOracle::extract_version(package, &parsed);
    assert_eq!(version, "0.0.0");
}

// =========================================================================
// Coverage Gap Tests — collect_paiml_deps
// =========================================================================

#[test]
fn test_collect_paiml_deps_string_version() {
    let oracle = LocalWorkspaceOracle::with_base_dir(std::env::temp_dir()).unwrap();
    let deps: toml::Value = toml::from_str(
        r#"
        trueno = "0.14.0"
        serde = "1.0"
    "#,
    )
    .unwrap();
    let mut result = Vec::new();
    oracle.collect_paiml_deps(&deps, &mut result);
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].name, "trueno");
    assert_eq!(result[0].required_version, "0.14.0");
    assert!(!result[0].is_path_dep);
}

#[test]
fn test_collect_paiml_deps_table_version() {
    let oracle = LocalWorkspaceOracle::with_base_dir(std::env::temp_dir()).unwrap();
    let deps: toml::Value = toml::from_str(
        r#"
        aprender = { version = "0.24.0", features = ["default"] }
    "#,
    )
    .unwrap();
    let mut result = Vec::new();
    oracle.collect_paiml_deps(&deps, &mut result);
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].name, "aprender");
    assert!(!result[0].is_path_dep);
}

#[test]
fn test_collect_paiml_deps_path_dep() {
    let oracle = LocalWorkspaceOracle::with_base_dir(std::env::temp_dir()).unwrap();
    let deps: toml::Value = toml::from_str(
        r#"
        realizar = { version = "0.5.0", path = "../realizar" }
    "#,
    )
    .unwrap();
    let mut result = Vec::new();
    oracle.collect_paiml_deps(&deps, &mut result);
    assert_eq!(result.len(), 1);
    assert!(result[0].is_path_dep);
}

#[test]
fn test_collect_paiml_deps_no_paiml() {
    let oracle = LocalWorkspaceOracle::with_base_dir(std::env::temp_dir()).unwrap();
    let deps: toml::Value = toml::from_str(
        r#"
        serde = "1.0"
        tokio = { version = "1.0", features = ["full"] }
    "#,
    )
    .unwrap();
    let mut result = Vec::new();
    oracle.collect_paiml_deps(&deps, &mut result);
    assert!(result.is_empty());
}

// =========================================================================
// Coverage Gap Tests — analyze_project
// =========================================================================

#[test]
fn test_analyze_project_simple_crate() {
    let temp = std::env::temp_dir().join("test_analyze_project_simple");
    let _ = std::fs::create_dir_all(&temp);
    std::fs::write(
        temp.join("Cargo.toml"),
        r#"
[package]
name = "trueno"
version = "0.14.0"

[dependencies]
serde = "1.0"
"#,
    )
    .unwrap();

    let oracle = LocalWorkspaceOracle::with_base_dir(std::env::temp_dir()).unwrap();
    let project = oracle.analyze_project(&temp).unwrap();

    assert_eq!(project.name, "trueno");
    assert_eq!(project.local_version, "0.14.0");
    assert!(!project.is_workspace);

    let _ = std::fs::remove_dir_all(&temp);
}

#[test]
fn test_analyze_project_workspace() {
    let temp = std::env::temp_dir().join("test_analyze_project_ws");
    let _ = std::fs::create_dir_all(&temp);
    std::fs::write(
        temp.join("Cargo.toml"),
        r#"
[workspace]
members = ["crate-a", "crate-b"]

[workspace.package]
version = "2.0.0"
"#,
    )
    .unwrap();

    let oracle = LocalWorkspaceOracle::with_base_dir(std::env::temp_dir()).unwrap();
    let project = oracle.analyze_project(&temp).unwrap();

    assert!(project.is_workspace);
    assert_eq!(project.workspace_members, vec!["crate-a", "crate-b"]);
    assert_eq!(project.local_version, "2.0.0");

    let _ = std::fs::remove_dir_all(&temp);
}

#[test]
fn test_analyze_project_with_paiml_deps() {
    let temp = std::env::temp_dir().join("test_analyze_project_deps");
    let _ = std::fs::create_dir_all(&temp);
    std::fs::write(
        temp.join("Cargo.toml"),
        r#"
[package]
name = "realizar"
version = "0.5.0"

[dependencies]
trueno = "0.14.0"
aprender = { version = "0.24.0", path = "../aprender" }
serde = "1.0"
"#,
    )
    .unwrap();

    let oracle = LocalWorkspaceOracle::with_base_dir(std::env::temp_dir()).unwrap();
    let project = oracle.analyze_project(&temp).unwrap();

    assert_eq!(project.paiml_dependencies.len(), 2);
    let trueno = project.paiml_dependencies.iter().find(|d| d.name == "trueno").unwrap();
    assert!(!trueno.is_path_dep);
    let aprender = project.paiml_dependencies.iter().find(|d| d.name == "aprender").unwrap();
    assert!(aprender.is_path_dep);

    let _ = std::fs::remove_dir_all(&temp);
}

#[test]
fn test_analyze_project_no_package_section() {
    let temp = std::env::temp_dir().join("test_analyze_project_nopackage");
    let _ = std::fs::create_dir_all(&temp);
    std::fs::write(
        temp.join("Cargo.toml"),
        r#"
[profile.release]
opt-level = 3
"#,
    )
    .unwrap();

    let oracle = LocalWorkspaceOracle::with_base_dir(std::env::temp_dir()).unwrap();
    let result = oracle.analyze_project(&temp);
    assert!(result.is_err());

    let _ = std::fs::remove_dir_all(&temp);
}

// =========================================================================
// Coverage Gap Tests — get_git_status
// =========================================================================

#[test]
fn test_get_git_status_current_repo() {
    let oracle = LocalWorkspaceOracle::with_base_dir(std::env::temp_dir()).unwrap();
    let status = oracle.get_git_status(Path::new("."));

    // In a git repo (local dev), branch should be a real name.
    // Outside a git repo (clean-room container), git is absent or cwd
    // has no .git — branch will be empty or "unknown".  Both are valid.
    let in_git_repo = std::process::Command::new("git")
        .args(["rev-parse", "--git-dir"])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    if in_git_repo {
        assert!(!status.branch.is_empty());
        assert_ne!(status.branch, "unknown");
    }
    // Outside a git repo we just verify it doesn't panic (already exercised above)
}

#[test]
fn test_get_git_status_non_git_dir() {
    let oracle = LocalWorkspaceOracle::with_base_dir(std::env::temp_dir()).unwrap();
    let status = oracle.get_git_status(Path::new("/tmp"));

    // Should return defaults without panic (branch may be empty for non-git dirs)
    let _ = status.branch;
    let _ = status.has_changes;
}

// =========================================================================
// Coverage Gap Tests — discover_projects
// =========================================================================

#[test]
fn test_discover_projects_nonexistent_dir() {
    let mut oracle =
        LocalWorkspaceOracle::with_base_dir(PathBuf::from("/nonexistent/unlikely/path")).unwrap();
    let projects = oracle.discover_projects().unwrap();
    assert!(projects.is_empty());
}

#[test]
fn test_discover_projects_with_paiml_crate() {
    let temp = std::env::temp_dir().join("test_discover_paiml");
    let _ = std::fs::create_dir_all(temp.join("trueno"));
    std::fs::write(
        temp.join("trueno/Cargo.toml"),
        r#"
[package]
name = "trueno"
version = "0.14.0"
"#,
    )
    .unwrap();

    let mut oracle = LocalWorkspaceOracle::with_base_dir(temp.clone()).unwrap();
    let projects = oracle.discover_projects().unwrap();
    assert!(projects.contains_key("trueno"));

    let _ = std::fs::remove_dir_all(&temp);
}

#[test]
fn test_discover_projects_skips_non_paiml() {
    let temp = std::env::temp_dir().join("test_discover_non_paiml");
    let _ = std::fs::create_dir_all(temp.join("random-crate"));
    std::fs::write(
        temp.join("random-crate/Cargo.toml"),
        r#"
[package]
name = "random-crate"
version = "0.1.0"

[dependencies]
serde = "1.0"
"#,
    )
    .unwrap();

    let mut oracle = LocalWorkspaceOracle::with_base_dir(temp.clone()).unwrap();
    let projects = oracle.discover_projects().unwrap();
    assert!(projects.is_empty(), "Non-PAIML crates should be skipped");

    let _ = std::fs::remove_dir_all(&temp);
}

// =========================================================================
// Coverage Gap Tests — suggest_publish_order with projects
// =========================================================================

fn make_project(name: &str, version: &str, deps: Vec<DependencyInfo>) -> LocalProject {
    LocalProject {
        name: name.to_string(),
        path: PathBuf::from(format!("/tmp/{}", name)),
        local_version: version.to_string(),
        published_version: None,
        git_status: GitStatus {
            branch: "main".to_string(),
            has_changes: false,
            modified_count: 0,
            unpushed_commits: 0,
            up_to_date: true,
        },
        dev_state: DevState::Clean,
        paiml_dependencies: deps,
        is_workspace: false,
        workspace_members: vec![],
    }
}

#[test]
fn test_suggest_publish_order_with_deps() {
    let mut oracle = LocalWorkspaceOracle::with_base_dir(std::env::temp_dir()).unwrap();

    // trueno has no deps, aprender depends on trueno
    oracle.projects.insert("trueno".to_string(), make_project("trueno", "0.14.0", vec![]));
    oracle.projects.insert(
        "aprender".to_string(),
        make_project(
            "aprender",
            "0.24.0",
            vec![DependencyInfo {
                name: "trueno".to_string(),
                required_version: "0.14.0".to_string(),
                is_path_dep: false,
                version_satisfied: None,
            }],
        ),
    );

    let order = oracle.suggest_publish_order();
    assert_eq!(order.order.len(), 2);
    assert!(order.cycles.is_empty());

    // trueno should come before aprender
    let trueno_idx = order.order.iter().position(|s| s.name == "trueno").unwrap();
    let aprender_idx = order.order.iter().position(|s| s.name == "aprender").unwrap();
    assert!(trueno_idx < aprender_idx);
}

#[test]
fn test_suggest_publish_order_no_deps() {
    let mut oracle = LocalWorkspaceOracle::with_base_dir(std::env::temp_dir()).unwrap();
    oracle.projects.insert("trueno".to_string(), make_project("trueno", "0.14.0", vec![]));

    let order = oracle.suggest_publish_order();
    assert_eq!(order.order.len(), 1);
    assert_eq!(order.order[0].name, "trueno");
}

#[test]
fn test_suggest_publish_order_needs_publish_dirty() {
    let mut oracle = LocalWorkspaceOracle::with_base_dir(std::env::temp_dir()).unwrap();
    let mut project = make_project("trueno", "0.14.0", vec![]);
    project.git_status.has_changes = true;
    project.dev_state = DevState::Dirty;
    oracle.projects.insert("trueno".to_string(), project);

    let order = oracle.suggest_publish_order();
    assert!(order.order[0].needs_publish);
}

// =========================================================================
// Coverage Gap Tests — detect_drift with projects
// =========================================================================

#[test]
fn test_detect_drift_not_published() {
    let mut oracle = LocalWorkspaceOracle::with_base_dir(std::env::temp_dir()).unwrap();
    oracle.projects.insert("trueno".to_string(), make_project("trueno", "0.14.0", vec![]));

    let drifts = oracle.detect_drift();
    let trueno_drift = drifts.iter().find(|d| d.name == "trueno");
    assert!(trueno_drift.is_some());
    assert_eq!(trueno_drift.unwrap().drift_type, DriftType::NotPublished);
}

#[test]
fn test_detect_drift_local_ahead() {
    let mut oracle = LocalWorkspaceOracle::with_base_dir(std::env::temp_dir()).unwrap();
    let mut project = make_project("trueno", "0.15.0", vec![]);
    project.published_version = Some("0.14.0".to_string());
    oracle.projects.insert("trueno".to_string(), project);

    let drifts = oracle.detect_drift();
    let trueno_drift = drifts.iter().find(|d| d.name == "trueno");
    assert!(trueno_drift.is_some());
    assert_eq!(trueno_drift.unwrap().drift_type, DriftType::LocalAhead);
}

#[test]
fn test_detect_drift_in_sync() {
    let mut oracle = LocalWorkspaceOracle::with_base_dir(std::env::temp_dir()).unwrap();
    let mut project = make_project("trueno", "0.14.0", vec![]);
    project.published_version = Some("0.14.0".to_string());
    oracle.projects.insert("trueno".to_string(), project);

    let drifts = oracle.detect_drift();
    assert!(drifts.is_empty(), "Same versions should not drift");
}

// =========================================================================
// Coverage Gap Tests — summary with projects
// =========================================================================

#[test]
fn test_summary_with_projects() {
    let mut oracle = LocalWorkspaceOracle::with_base_dir(std::env::temp_dir()).unwrap();
    oracle.projects.insert("trueno".to_string(), make_project("trueno", "0.14.0", vec![]));
    let mut dirty = make_project("aprender", "0.24.0", vec![]);
    dirty.git_status.has_changes = true;
    dirty.git_status.modified_count = 3;
    oracle.projects.insert("aprender".to_string(), dirty);

    let summary = oracle.summary();
    assert_eq!(summary.total_projects, 2);
    assert_eq!(summary.projects_with_changes, 1);
}

// =========================================================================
// Coverage Gap Tests — extract_paiml_deps (workspace deps)
// =========================================================================

#[test]
fn test_extract_paiml_deps_workspace_section() {
    let oracle = LocalWorkspaceOracle::with_base_dir(std::env::temp_dir()).unwrap();
    let parsed: toml::Value = toml::from_str(
        r#"
[package]
name = "test"
version = "0.1.0"

[workspace.dependencies]
trueno = "0.14.0"
serde = "1.0"
"#,
    )
    .unwrap();
    let deps = oracle.extract_paiml_deps(&parsed);
    assert_eq!(deps.len(), 1);
    assert_eq!(deps[0].name, "trueno");
}

#[test]
fn test_extract_paiml_deps_dev_dependencies() {
    let oracle = LocalWorkspaceOracle::with_base_dir(std::env::temp_dir()).unwrap();
    let parsed: toml::Value = toml::from_str(
        r#"
[package]
name = "test"
version = "0.1.0"

[dev-dependencies]
trueno = "0.14.0"
"#,
    )
    .unwrap();
    let deps = oracle.extract_paiml_deps(&parsed);
    assert_eq!(deps.len(), 1);
}

// =========================================================================
// Coverage Gap Tests — fetch_published_versions
// =========================================================================

#[tokio::test]
async fn test_fetch_published_versions_empty_projects() {
    let mut oracle =
        LocalWorkspaceOracle::with_base_dir(PathBuf::from("/nonexistent/unlikely/path")).unwrap();
    // No projects -> fetch should complete without error
    let result = oracle.fetch_published_versions().await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_fetch_published_versions_with_known_crate() {
    let mut oracle = LocalWorkspaceOracle::with_base_dir(std::env::temp_dir()).unwrap();

    // Insert a known crate
    oracle.projects.insert("trueno".to_string(), make_project("trueno", "0.14.0", vec![]));

    // This will attempt to fetch from crates.io. May succeed or fail depending
    // on network, but should not panic.
    let result = oracle.fetch_published_versions().await;
    assert!(result.is_ok());

    // If crates.io was reachable, published_version should be set
    if let Some(project) = oracle.projects.get("trueno") {
        // published_version may or may not be set depending on network
        let _ = project.published_version.as_ref();
    }
}

#[tokio::test]
async fn test_fetch_published_versions_unknown_crate() {
    let mut oracle = LocalWorkspaceOracle::with_base_dir(std::env::temp_dir()).unwrap();

    // Insert a crate that definitely doesn't exist on crates.io
    oracle.projects.insert(
        "nonexistent-paiml-crate-xyz-abc-123".to_string(),
        make_project("nonexistent-paiml-crate-xyz-abc-123", "0.1.0", vec![]),
    );

    let result = oracle.fetch_published_versions().await;
    assert!(result.is_ok());

    // Should remain None since crate doesn't exist
    let project = oracle.projects.get("nonexistent-paiml-crate-xyz-abc-123").unwrap();
    assert!(project.published_version.is_none());
}

// =========================================================================
// Coverage Gap Tests — compare_versions edge cases
// =========================================================================

#[test]
fn test_compare_versions_major_only() {
    use std::cmp::Ordering;
    assert_eq!(compare_versions("2", "1"), Ordering::Greater);
    assert_eq!(compare_versions("0", "1"), Ordering::Less);
}

#[test]
fn test_compare_versions_pre_release_ignored() {
    use std::cmp::Ordering;
    // Pre-release suffixes are parsed as 0 due to unwrap_or(0)
    assert_eq!(compare_versions("1.0.0-alpha", "1.0.0"), Ordering::Equal);
}

#[test]
fn test_compare_versions_extra_segments_ignored() {
    use std::cmp::Ordering;
    // Only first 3 segments are used
    assert_eq!(compare_versions("1.0.0.1", "1.0.0.2"), Ordering::Equal);
}

// =========================================================================
// Coverage Gap Tests — is_blocking edge cases
// =========================================================================

#[test]
fn test_is_blocking_equal_versions() {
    let project = LocalProject {
        name: "test".to_string(),
        path: PathBuf::from("/tmp/test"),
        local_version: "1.0.0".to_string(),
        published_version: Some("1.0.0".to_string()),
        git_status: GitStatus {
            branch: "main".to_string(),
            has_changes: false,
            modified_count: 0,
            unpushed_commits: 0,
            up_to_date: true,
        },
        dev_state: DevState::Clean,
        paiml_dependencies: vec![],
        is_workspace: false,
        workspace_members: vec![],
    };
    // Equal versions don't block
    assert!(!project.is_blocking());
}

#[test]
fn test_is_blocking_unpushed_state() {
    let project = LocalProject {
        name: "test".to_string(),
        path: PathBuf::from("/tmp/test"),
        local_version: "0.8.0".to_string(),
        published_version: Some("0.9.0".to_string()),
        git_status: GitStatus {
            branch: "main".to_string(),
            has_changes: false,
            modified_count: 0,
            unpushed_commits: 1,
            up_to_date: false,
        },
        dev_state: DevState::Unpushed,
        paiml_dependencies: vec![],
        is_workspace: false,
        workspace_members: vec![],
    };
    // Unpushed doesn't use local version, so not blocking
    assert!(!project.is_blocking());
}

// =========================================================================
// Coverage Gap Tests — detect_drift LocalBehind
// =========================================================================

#[test]
fn test_detect_drift_local_behind() {
    let mut oracle = LocalWorkspaceOracle::with_base_dir(std::env::temp_dir()).unwrap();
    let mut project = make_project("trueno", "0.13.0", vec![]);
    project.published_version = Some("0.14.0".to_string());
    oracle.projects.insert("trueno".to_string(), project);

    let drifts = oracle.detect_drift();
    let trueno_drift = drifts.iter().find(|d| d.name == "trueno");
    assert!(trueno_drift.is_some());
    assert_eq!(trueno_drift.unwrap().drift_type, DriftType::LocalBehind);
}

// =========================================================================
// Coverage Gap Tests — summary with workspace project
// =========================================================================

#[test]
fn test_summary_with_workspace() {
    let mut oracle = LocalWorkspaceOracle::with_base_dir(std::env::temp_dir()).unwrap();
    let mut ws = make_project("batuta", "0.6.0", vec![]);
    ws.is_workspace = true;
    ws.workspace_members = vec!["crate-a".to_string(), "crate-b".to_string()];
    oracle.projects.insert("batuta".to_string(), ws);

    let summary = oracle.summary();
    assert_eq!(summary.total_projects, 1);
    assert_eq!(summary.workspace_count, 1);
}

#[test]
fn test_summary_with_unpushed() {
    let mut oracle = LocalWorkspaceOracle::with_base_dir(std::env::temp_dir()).unwrap();
    let mut project = make_project("trueno", "0.14.0", vec![]);
    project.git_status.unpushed_commits = 3;
    oracle.projects.insert("trueno".to_string(), project);

    let summary = oracle.summary();
    assert_eq!(summary.projects_with_unpushed, 1);
}

// =========================================================================
// Coverage Gap Tests — extract_version workspace inheritance missing ws section
// =========================================================================

#[test]
fn test_extract_version_workspace_inheritance_missing_ws() {
    let parsed: toml::Value = toml::from_str(
        r#"
[package]
name = "test"
version.workspace = true
"#,
    )
    .unwrap();
    let package = parsed.get("package").unwrap();
    // workspace.package.version doesn't exist -> defaults to "0.0.0"
    let version = LocalWorkspaceOracle::extract_version(package, &parsed);
    assert_eq!(version, "0.0.0");
}

// =========================================================================
// Coverage Gap Tests — suggest_publish_order with unpushed commits
// =========================================================================

#[test]
fn test_suggest_publish_order_needs_publish_unpushed() {
    let mut oracle = LocalWorkspaceOracle::with_base_dir(std::env::temp_dir()).unwrap();
    let mut project = make_project("trueno", "0.14.0", vec![]);
    project.git_status.unpushed_commits = 2;
    project.dev_state = DevState::Unpushed;
    oracle.projects.insert("trueno".to_string(), project);

    let order = oracle.suggest_publish_order();
    assert!(order.order[0].needs_publish);
}

// =========================================================================
// Coverage Gap Tests — collect_paiml_deps with non-table value
// =========================================================================

#[test]
fn test_collect_paiml_deps_non_table_ignored() {
    let oracle = LocalWorkspaceOracle::with_base_dir(std::env::temp_dir()).unwrap();
    // Pass a non-table value -> should return empty
    let deps_val = toml::Value::String("not a table".to_string());
    let mut result = Vec::new();
    oracle.collect_paiml_deps(&deps_val, &mut result);
    assert!(result.is_empty());
}

// =========================================================================
// Coverage Gap Tests -- LocalWorkspaceOracle::new()
// =========================================================================

#[test]
fn test_local_workspace_oracle_new() {
    // Exercise the new() constructor that reads home_dir
    let result = LocalWorkspaceOracle::new();
    // Should succeed on most systems (home dir exists)
    assert!(result.is_ok());
    let oracle = result.unwrap();
    assert!(oracle.projects().is_empty());
}

// =========================================================================
// Coverage Gap Tests -- analyze_project with git dirty/unpushed states
// =========================================================================

#[test]
fn test_analyze_project_dirty_state() {
    // Create a temp dir with a git repo that has uncommitted changes
    let temp = std::env::temp_dir().join("test_analyze_dirty_state");
    let _ = std::fs::remove_dir_all(&temp);
    let _ = std::fs::create_dir_all(&temp);

    // Init git repo
    let _ = std::process::Command::new("git").args(["init"]).current_dir(&temp).output();
    let _ = std::process::Command::new("git")
        .args(["config", "user.email", "test@test.com"])
        .current_dir(&temp)
        .output();
    let _ = std::process::Command::new("git")
        .args(["config", "user.name", "Test"])
        .current_dir(&temp)
        .output();

    // Write Cargo.toml
    std::fs::write(
        temp.join("Cargo.toml"),
        r#"
[package]
name = "trueno"
version = "0.14.0"
"#,
    )
    .expect("write Cargo.toml");

    // Add and commit Cargo.toml
    let _ =
        std::process::Command::new("git").args(["add", "Cargo.toml"]).current_dir(&temp).output();
    let _ = std::process::Command::new("git")
        .args(["commit", "-m", "init"])
        .current_dir(&temp)
        .output();

    // Create an uncommitted file to make state Dirty
    std::fs::write(temp.join("dirty.rs"), "fn dirty() {}").expect("write");

    let oracle = LocalWorkspaceOracle::with_base_dir(std::env::temp_dir()).unwrap();
    let project = oracle.analyze_project(&temp).unwrap();

    // Should detect dirty state from uncommitted file
    assert!(project.git_status.has_changes);
    assert_eq!(project.dev_state, DevState::Dirty);

    let _ = std::fs::remove_dir_all(&temp);
}
