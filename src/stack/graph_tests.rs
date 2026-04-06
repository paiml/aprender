use super::*;

// =========================================================================
// Test Helpers: Reduce structural repetition (PMAT entropy)
// =========================================================================

/// Add a test crate with the given name and version to the graph.
/// Uses an empty PathBuf since test crates don't need real paths.
fn add_test_crate(graph: &mut DependencyGraph, name: &str, major: u64, minor: u64, patch: u64) {
    graph.add_crate(CrateInfo::new(
        name,
        semver::Version::new(major, minor, patch),
        std::path::PathBuf::new(),
    ));
}

/// Add a normal (non-path) dependency edge between two crates.
fn add_normal_dep(graph: &mut DependencyGraph, from: &str, to: &str, version: &str) {
    graph.add_dependency(
        from,
        to,
        DependencyEdge {
            version_req: version.to_string(),
            is_path: false,
            kind: DependencyKind::Normal,
        },
    );
}

/// Add a dev dependency edge between two crates.
fn add_dev_dep(graph: &mut DependencyGraph, from: &str, to: &str, version: &str) {
    graph.add_dependency(
        from,
        to,
        DependencyEdge {
            version_req: version.to_string(),
            is_path: false,
            kind: DependencyKind::Dev,
        },
    );
}

/// Add a path dependency edge between two crates.
fn add_path_dep(graph: &mut DependencyGraph, from: &str, to: &str) {
    graph.add_dependency(
        from,
        to,
        DependencyEdge { version_req: String::new(), is_path: true, kind: DependencyKind::Normal },
    );
}

/// Assert that a list of crate names contains the given name.
fn assert_contains_crate(list: &[String], name: &str) {
    assert!(
        list.contains(&name.to_string()),
        "Expected list to contain '{}', but it was not found in: {:?}",
        name,
        list
    );
}

fn create_test_graph() -> DependencyGraph {
    let mut graph = DependencyGraph::new();

    // Create crates (uses PathBuf::from for create_test_graph to preserve original paths)
    graph.add_crate(CrateInfo::new(
        "trueno",
        semver::Version::new(1, 2, 0),
        std::path::PathBuf::from("trueno/Cargo.toml"),
    ));
    graph.add_crate(CrateInfo::new(
        "aprender",
        semver::Version::new(0, 8, 0),
        std::path::PathBuf::from("aprender/Cargo.toml"),
    ));
    graph.add_crate(CrateInfo::new(
        "entrenar",
        semver::Version::new(0, 2, 0),
        std::path::PathBuf::from("entrenar/Cargo.toml"),
    ));
    graph.add_crate(CrateInfo::new(
        "alimentar",
        semver::Version::new(0, 3, 0),
        std::path::PathBuf::from("alimentar/Cargo.toml"),
    ));

    // Add dependencies
    add_normal_dep(&mut graph, "aprender", "trueno", "^1.0");
    add_normal_dep(&mut graph, "entrenar", "aprender", "^0.8");
    add_path_dep(&mut graph, "entrenar", "alimentar");
    add_normal_dep(&mut graph, "alimentar", "trueno", "^1.0");

    graph
}

#[test]
fn test_graph_creation() {
    let graph = DependencyGraph::new();
    assert_eq!(graph.crate_count(), 0);
    assert!(!graph.has_cycles());
}

#[test]
fn test_add_crate() {
    let mut graph = DependencyGraph::new();
    add_test_crate(&mut graph, "trueno", 1, 0, 0);

    assert_eq!(graph.crate_count(), 1);
    assert!(graph.get_crate("trueno").is_some());
    assert!(graph.get_crate("serde").is_none());
}

#[test]
fn test_dependency_edges() {
    let graph = create_test_graph();

    // Check dependencies
    let aprender_deps = graph.all_dependencies("aprender");
    assert_contains_crate(&aprender_deps, "trueno");

    let entrenar_deps = graph.all_dependencies("entrenar");
    assert_contains_crate(&entrenar_deps, "aprender");
    assert_contains_crate(&entrenar_deps, "alimentar");
    assert_contains_crate(&entrenar_deps, "trueno"); // transitive
}

#[test]
fn test_no_cycles() {
    let graph = create_test_graph();
    assert!(!graph.has_cycles());
}

#[test]
fn test_cycle_detection() {
    let mut graph = DependencyGraph::new();

    add_test_crate(&mut graph, "a", 1, 0, 0);
    add_test_crate(&mut graph, "b", 1, 0, 0);
    add_test_crate(&mut graph, "c", 1, 0, 0);

    // Create cycle: a -> b -> c -> a
    add_normal_dep(&mut graph, "a", "b", "1.0");
    add_normal_dep(&mut graph, "b", "c", "1.0");
    add_normal_dep(&mut graph, "c", "a", "1.0");

    assert!(graph.has_cycles());
    assert!(graph.topological_order().is_err());
}

#[test]
fn test_topological_order() {
    let graph = create_test_graph();
    let order = graph.topological_order().unwrap();

    // trueno should come before aprender
    let trueno_pos = order.iter().position(|n| n == "trueno").unwrap();
    let aprender_pos = order.iter().position(|n| n == "aprender").unwrap();
    assert!(trueno_pos < aprender_pos);

    // aprender should come before entrenar
    let entrenar_pos = order.iter().position(|n| n == "entrenar").unwrap();
    assert!(aprender_pos < entrenar_pos);

    // alimentar should come before entrenar
    let alimentar_pos = order.iter().position(|n| n == "alimentar").unwrap();
    assert!(alimentar_pos < entrenar_pos);
}

#[test]
fn test_release_order_for_crate() {
    let graph = create_test_graph();
    let order = graph.release_order_for("entrenar").unwrap();

    // Should include entrenar and its dependencies
    assert_contains_crate(&order, "trueno");
    assert_contains_crate(&order, "aprender");
    assert_contains_crate(&order, "alimentar");
    assert_contains_crate(&order, "entrenar");

    // entrenar should be last
    assert_eq!(order.last().unwrap(), "entrenar");

    // trueno should be first (base dependency)
    assert_eq!(order.first().unwrap(), "trueno");
}

#[test]
fn test_find_path_dependencies() {
    let graph = create_test_graph();
    let path_deps = graph.find_path_dependencies();

    // Should find the entrenar -> alimentar path dependency
    assert_eq!(path_deps.len(), 1);
    assert_eq!(path_deps[0].crate_name, "entrenar");
    assert_eq!(path_deps[0].dependency, "alimentar");
}

#[test]
fn test_dependents() {
    let graph = create_test_graph();

    // trueno has dependents: aprender, alimentar
    let trueno_dependents = graph.dependents("trueno");
    assert_contains_crate(&trueno_dependents, "aprender");
    assert_contains_crate(&trueno_dependents, "alimentar");

    // aprender has dependent: entrenar
    let aprender_dependents = graph.dependents("aprender");
    assert_contains_crate(&aprender_dependents, "entrenar");
}

#[test]
fn test_version_conflict_detection() {
    let mut graph = DependencyGraph::new();

    // Create crates with conflicting arrow versions
    let mut renacer =
        CrateInfo::new("renacer", semver::Version::new(0, 6, 0), std::path::PathBuf::new());
    renacer.external_dependencies.push(DependencyInfo::new("arrow", "54.0"));

    let mut trueno_graph =
        CrateInfo::new("trueno-graph", semver::Version::new(0, 2, 0), std::path::PathBuf::new());
    trueno_graph.external_dependencies.push(DependencyInfo::new("arrow", "53.0"));

    graph.add_crate(renacer);
    graph.add_crate(trueno_graph);

    let conflicts = graph.detect_conflicts();
    assert_eq!(conflicts.len(), 1);
    assert_eq!(conflicts[0].dependency, "arrow");
    assert_eq!(conflicts[0].usages.len(), 2);
}

#[test]
fn test_no_conflict_same_version() {
    let mut graph = DependencyGraph::new();

    // Create crates with same arrow version
    let mut crate_a = CrateInfo::new("a", semver::Version::new(1, 0, 0), std::path::PathBuf::new());
    crate_a.external_dependencies.push(DependencyInfo::new("arrow", "54.0"));

    let mut crate_b = CrateInfo::new("b", semver::Version::new(1, 0, 0), std::path::PathBuf::new());
    crate_b.external_dependencies.push(DependencyInfo::new("arrow", "54.0"));

    graph.add_crate(crate_a);
    graph.add_crate(crate_b);

    let conflicts = graph.detect_conflicts();
    assert!(conflicts.is_empty());
}

// ============================================================================
// ISSUE-13: False circular dependency detection
// https://github.com/paiml/batuta/issues/13
// ============================================================================

/// RED PHASE: Dev-dependencies should NOT create cycles for release ordering
/// This reproduces issue #13: presentar workspace reports false cycle
#[test]
fn test_issue_13_dev_dependency_not_cycle() {
    // ARRANGE: Create graph where:
    // - presentar depends on trueno (normal)
    // - trueno has dev-dependency on presentar (for testing)
    // This is NOT a real cycle for release purposes
    let mut graph = DependencyGraph::new();

    add_test_crate(&mut graph, "trueno", 1, 0, 0);
    add_test_crate(&mut graph, "presentar", 0, 1, 0);

    // presentar -> trueno (normal dependency)
    add_normal_dep(&mut graph, "presentar", "trueno", "^1.0");

    // trueno -> presentar (DEV dependency - for testing only)
    add_dev_dep(&mut graph, "trueno", "presentar", "^0.1");

    // ACT & ASSERT: Should NOT have cycles when excluding dev deps
    assert!(!graph.has_cycles(), "Dev dependencies should not create cycles");

    // Topological order should work
    let order = graph.topological_order();
    assert!(order.is_ok(), "Should compute topological order: {:?}", order.err());

    // trueno should come before presentar
    let order = order.unwrap();
    let trueno_pos = order.iter().position(|n| n == "trueno").unwrap();
    let presentar_pos = order.iter().position(|n| n == "presentar").unwrap();
    assert!(trueno_pos < presentar_pos, "trueno should be released before presentar");
}

/// RED PHASE: Multiple dev dependencies should not create cycle
#[test]
fn test_multiple_dev_deps_no_cycle() {
    let mut graph = DependencyGraph::new();

    add_test_crate(&mut graph, "a", 1, 0, 0);
    add_test_crate(&mut graph, "b", 1, 0, 0);
    add_test_crate(&mut graph, "c", 1, 0, 0);

    // a -> b (normal)
    add_normal_dep(&mut graph, "a", "b", "^1.0");

    // b -> a (dev)
    add_dev_dep(&mut graph, "b", "a", "^1.0");

    // c -> a (dev)
    add_dev_dep(&mut graph, "c", "a", "^1.0");

    assert!(!graph.has_cycles());
    let order = graph.topological_order();
    assert!(order.is_ok());
}

/// Test graph with build dependencies
#[test]
fn test_build_dependencies() {
    let mut graph = DependencyGraph::new();

    add_test_crate(&mut graph, "main", 1, 0, 0);
    add_test_crate(&mut graph, "build-dep", 1, 0, 0);

    graph.add_dependency(
        "main",
        "build-dep",
        DependencyEdge {
            version_req: "^1.0".to_string(),
            is_path: false,
            kind: DependencyKind::Build,
        },
    );

    // Build deps should be in the graph
    let deps = graph.all_dependencies("main");
    assert!(deps.iter().any(|d| d == "build-dep"));
}

/// Test graph removal of crate
#[test]
fn test_graph_get_crate() {
    let mut graph = DependencyGraph::new();
    add_test_crate(&mut graph, "test", 1, 0, 0);

    let crate_info = graph.get_crate("test");
    assert!(crate_info.is_some());
    assert_eq!(crate_info.unwrap().name, "test");

    let missing = graph.get_crate("nonexistent");
    assert!(missing.is_none());
}

/// Test graph contains
#[test]
fn test_graph_contains() {
    let mut graph = DependencyGraph::new();
    add_test_crate(&mut graph, "exists", 1, 0, 0);

    assert!(graph.get_crate("exists").is_some());
    assert!(graph.get_crate("not-exists").is_none());
}

/// Test graph all_crates iteration
#[test]
fn test_graph_all_crates() {
    let mut graph = DependencyGraph::new();
    add_test_crate(&mut graph, "a", 1, 0, 0);
    add_test_crate(&mut graph, "b", 2, 0, 0);

    let crates: Vec<_> = graph.all_crates().collect();
    assert_eq!(crates.len(), 2);
}

/// Test crate count
#[test]
fn test_graph_crate_count() {
    let mut graph = DependencyGraph::new();
    assert_eq!(graph.crate_count(), 0);

    add_test_crate(&mut graph, "test", 1, 0, 0);
    assert_eq!(graph.crate_count(), 1);
}

/// Test graph with no edges
#[test]
fn test_graph_no_dependencies() {
    let mut graph = DependencyGraph::new();
    add_test_crate(&mut graph, "lone", 1, 0, 0);

    let deps = graph.all_dependencies("lone");
    assert!(deps.is_empty());

    let dependents = graph.dependents("lone");
    assert!(dependents.is_empty());
}

/// Test get_crate_mut
#[test]
fn test_graph_get_crate_mut() {
    let mut graph = DependencyGraph::new();
    add_test_crate(&mut graph, "mutable", 1, 0, 0);

    if let Some(crate_info) = graph.get_crate_mut("mutable") {
        crate_info.status = CrateStatus::Healthy;
    }

    assert_eq!(graph.get_crate("mutable").unwrap().status, CrateStatus::Healthy);
}

/// RED PHASE: Real cycles (normal deps) should still be detected
#[test]
fn test_issue_13_real_cycle_still_detected() {
    // ARRANGE: Create actual cycle with normal dependencies
    let mut graph = DependencyGraph::new();

    add_test_crate(&mut graph, "a", 1, 0, 0);
    add_test_crate(&mut graph, "b", 1, 0, 0);

    // a -> b (normal)
    add_normal_dep(&mut graph, "a", "b", "1.0");

    // b -> a (normal) - REAL CYCLE!
    add_normal_dep(&mut graph, "b", "a", "1.0");

    // ACT & ASSERT: Should detect this real cycle
    assert!(graph.has_cycles(), "Real cycles should still be detected");
    assert!(graph.topological_order().is_err());
}

/// RED PHASE: Build dependencies should also be considered for cycles
#[test]
fn test_issue_13_build_dep_creates_cycle() {
    // Build deps are needed at compile time, so they create real cycles
    let mut graph = DependencyGraph::new();

    add_test_crate(&mut graph, "a", 1, 0, 0);
    add_test_crate(&mut graph, "b", 1, 0, 0);

    // a -> b (normal)
    add_normal_dep(&mut graph, "a", "b", "1.0");

    // b -> a (build) - Build deps are needed at compile time
    graph.add_dependency(
        "b",
        "a",
        DependencyEdge {
            version_req: "1.0".to_string(),
            is_path: false,
            kind: DependencyKind::Build,
        },
    );

    // ACT & ASSERT: Build dependency cycle should be detected
    assert!(graph.has_cycles(), "Build dependency cycles should be detected");
}

// =========================================================================
// Additional Coverage Tests
// =========================================================================

#[test]
fn test_graph_cov_001_all_deps_missing_crate() {
    let graph = DependencyGraph::new();
    let deps = graph.all_dependencies("nonexistent");
    assert!(deps.is_empty());
}

#[test]
fn test_graph_cov_002_dependents_missing_crate() {
    let graph = DependencyGraph::new();
    let deps = graph.dependents("nonexistent");
    assert!(deps.is_empty());
}

#[test]
fn test_graph_cov_003_release_order_missing_crate() {
    let graph = DependencyGraph::new();
    // Should still work with empty graph
    let order = graph.topological_order();
    assert!(order.is_ok());
    assert!(order.unwrap().is_empty());
}

#[test]
fn test_graph_cov_004_dependency_edge_debug() {
    let edge = DependencyEdge {
        version_req: "^1.0".to_string(),
        is_path: true,
        kind: DependencyKind::Normal,
    };
    let debug = format!("{:?}", edge);
    assert!(debug.contains("DependencyEdge"));
    assert!(debug.contains("is_path: true"));
}

#[test]
fn test_graph_cov_005_dependency_edge_clone() {
    let edge = DependencyEdge {
        version_req: "^1.0".to_string(),
        is_path: false,
        kind: DependencyKind::Dev,
    };
    let cloned = edge.clone();
    assert_eq!(cloned.version_req, edge.version_req);
    assert_eq!(cloned.is_path, edge.is_path);
}

#[test]
fn test_graph_cov_006_path_dep_issue_debug() {
    let issue = PathDependencyIssue {
        crate_name: "test".to_string(),
        dependency: "dep".to_string(),
        current: "path = \"../dep\"".to_string(),
        recommended: Some("1.0.0".to_string()),
    };
    let debug = format!("{:?}", issue);
    assert!(debug.contains("PathDependencyIssue"));
    assert!(debug.contains("test"));
}

#[test]
fn test_graph_cov_007_path_dep_issue_clone() {
    let issue = PathDependencyIssue {
        crate_name: "test".to_string(),
        dependency: "dep".to_string(),
        current: "path = \"../dep\"".to_string(),
        recommended: None,
    };
    let cloned = issue.clone();
    assert_eq!(cloned.crate_name, issue.crate_name);
    assert!(cloned.recommended.is_none());
}

#[test]
fn test_graph_cov_008_graph_debug() {
    let graph = DependencyGraph::new();
    let debug = format!("{:?}", graph);
    assert!(debug.contains("DependencyGraph"));
}

#[test]
fn test_graph_cov_009_graph_clone() {
    let mut graph = DependencyGraph::new();
    add_test_crate(&mut graph, "test", 1, 0, 0);
    let cloned = graph.clone();
    assert_eq!(cloned.crate_count(), 1);
}

#[test]
fn test_graph_cov_010_default() {
    let graph = DependencyGraph::default();
    assert_eq!(graph.crate_count(), 0);
}

#[test]
fn test_graph_cov_011_add_dep_creates_nodes() {
    let mut graph = DependencyGraph::new();
    // Add dependency without first adding crates
    add_normal_dep(&mut graph, "new_from", "new_to", "1.0");

    // Both nodes should be created
    assert!(graph.node_indices_contains("new_from"));
    assert!(graph.node_indices_contains("new_to"));
}

#[test]
fn test_graph_cov_012_add_crate_duplicate() {
    let mut graph = DependencyGraph::new();
    add_test_crate(&mut graph, "dup", 1, 0, 0);
    // Add again - should update, not duplicate
    add_test_crate(&mut graph, "dup", 2, 0, 0);
    assert_eq!(graph.crate_count(), 1);
    assert_eq!(graph.get_crate("dup").unwrap().local_version, semver::Version::new(2, 0, 0));
}

#[test]
fn test_graph_cov_013_release_order_for_leaf() {
    let graph = create_test_graph();
    // trueno is a leaf node (no dependencies)
    let order = graph.release_order_for("trueno").unwrap();
    assert_eq!(order.len(), 1);
    assert_eq!(order[0], "trueno");
}

#[test]
fn test_graph_cov_014_no_path_deps() {
    let mut graph = DependencyGraph::new();
    add_test_crate(&mut graph, "a", 1, 0, 0);
    add_test_crate(&mut graph, "b", 1, 0, 0);
    add_normal_dep(&mut graph, "a", "b", "1.0");

    let path_deps = graph.find_path_dependencies();
    assert!(path_deps.is_empty());
}

#[test]
fn test_graph_cov_015_detect_conflicts_no_deps() {
    let mut graph = DependencyGraph::new();
    add_test_crate(&mut graph, "empty", 1, 0, 0);

    let conflicts = graph.detect_conflicts();
    assert!(conflicts.is_empty());
}

#[test]
fn test_graph_cov_016_single_usage_no_conflict() {
    let mut graph = DependencyGraph::new();
    let mut crate_a = CrateInfo::new("a", semver::Version::new(1, 0, 0), std::path::PathBuf::new());
    crate_a.external_dependencies.push(DependencyInfo::new("serde", "1.0"));
    graph.add_crate(crate_a);

    let conflicts = graph.detect_conflicts();
    assert!(conflicts.is_empty()); // Single usage = no conflict
}

#[test]
fn test_graph_cov_017_dependency_kind_variants() {
    // Test all DependencyKind variants
    let normal = DependencyKind::Normal;
    let dev = DependencyKind::Dev;
    let build = DependencyKind::Build;

    assert!(matches!(normal, DependencyKind::Normal));
    assert!(matches!(dev, DependencyKind::Dev));
    assert!(matches!(build, DependencyKind::Build));
}

// =========================================================================
// Fallback graph primitives coverage
// =========================================================================

#[test]
fn test_fallback_graph_cov_018_incoming_neighbors_empty() {
    let mut graph = DependencyGraph::new();
    add_test_crate(&mut graph, "solo", 1, 0, 0);
    // Node exists but has no incoming edges
    let dependents = graph.dependents("solo");
    assert!(dependents.is_empty());
}

#[test]
fn test_fallback_graph_cov_019_collect_deps_visited_dedup() {
    // Test that transitive dependency traversal properly deduplicates
    let mut graph = DependencyGraph::new();
    add_test_crate(&mut graph, "root", 1, 0, 0);
    add_test_crate(&mut graph, "mid", 1, 0, 0);
    add_test_crate(&mut graph, "leaf", 1, 0, 0);

    // root -> mid -> leaf
    // root -> leaf (direct)
    add_normal_dep(&mut graph, "root", "mid", "1.0");
    add_normal_dep(&mut graph, "mid", "leaf", "1.0");
    add_normal_dep(&mut graph, "root", "leaf", "1.0");

    let deps = graph.all_dependencies("root");
    // leaf should appear only once
    assert_eq!(deps.iter().filter(|d| *d == "leaf").count(), 1);
    assert_contains_crate(&deps, "mid");
    assert_contains_crate(&deps, "leaf");
}

#[test]
fn test_fallback_graph_cov_020_build_release_graph_filters_dev() {
    let mut graph = DependencyGraph::new();
    add_test_crate(&mut graph, "a", 1, 0, 0);
    add_test_crate(&mut graph, "b", 1, 0, 0);

    // a -> b (dev) should be excluded from release graph
    add_dev_dep(&mut graph, "a", "b", "1.0");

    // Release graph should have no edges, so no cycles
    assert!(!graph.has_cycles());
    let order = graph.topological_order().unwrap();
    // Dev edges are filtered from the release graph.
    // build_release_graph uses from_edge_list with an empty edge list,
    // so nodes only reachable via dev edges won't be in the release graph.
    // The key assertion: no cycle detected and topological order succeeds.
    assert!(order.len() <= 2);
}

#[test]
fn test_fallback_graph_cov_021_toposort_empty_graph() {
    let graph = DependencyGraph::new();
    let order = graph.topological_order().unwrap();
    assert!(order.is_empty());
}

#[test]
fn test_fallback_graph_cov_022_diamond_dependency() {
    // Diamond: root -> A -> C, root -> B -> C
    let mut graph = DependencyGraph::new();
    add_test_crate(&mut graph, "root", 1, 0, 0);
    add_test_crate(&mut graph, "a", 1, 0, 0);
    add_test_crate(&mut graph, "b", 1, 0, 0);
    add_test_crate(&mut graph, "c", 1, 0, 0);

    add_normal_dep(&mut graph, "root", "a", "1.0");
    add_normal_dep(&mut graph, "root", "b", "1.0");
    add_normal_dep(&mut graph, "a", "c", "1.0");
    add_normal_dep(&mut graph, "b", "c", "1.0");

    assert!(!graph.has_cycles());
    let order = graph.topological_order().unwrap();
    assert_eq!(order.len(), 4);

    // c should come before a, b, and root
    let c_pos = order.iter().position(|n| n == "c").unwrap();
    let root_pos = order.iter().position(|n| n == "root").unwrap();
    assert!(c_pos < root_pos);
}

#[test]
fn test_fallback_graph_cov_023_release_order_for_nonexistent() {
    let graph = DependencyGraph::new();
    let order = graph.release_order_for("nonexistent").unwrap();
    assert!(order.is_empty());
}

#[test]
fn test_fallback_graph_cov_024_multiple_path_deps() {
    let mut graph = DependencyGraph::new();
    add_test_crate(&mut graph, "a", 1, 0, 0);
    add_test_crate(&mut graph, "b", 1, 0, 0);
    add_test_crate(&mut graph, "c", 1, 0, 0);

    add_path_dep(&mut graph, "a", "b");
    add_path_dep(&mut graph, "a", "c");

    let path_deps = graph.find_path_dependencies();
    assert_eq!(path_deps.len(), 2);
}

#[test]
fn test_fallback_graph_cov_025_detect_conflicts_multiple_deps() {
    let mut graph = DependencyGraph::new();

    let mut crate_a = CrateInfo::new("a", semver::Version::new(1, 0, 0), std::path::PathBuf::new());
    crate_a.external_dependencies.push(DependencyInfo::new("serde", "1.0"));
    crate_a.external_dependencies.push(DependencyInfo::new("arrow", "53.0"));

    let mut crate_b = CrateInfo::new("b", semver::Version::new(1, 0, 0), std::path::PathBuf::new());
    crate_b.external_dependencies.push(DependencyInfo::new("serde", "1.0"));
    crate_b.external_dependencies.push(DependencyInfo::new("arrow", "54.0"));

    let mut crate_c = CrateInfo::new("c", semver::Version::new(1, 0, 0), std::path::PathBuf::new());
    crate_c.external_dependencies.push(DependencyInfo::new("serde", "1.0"));

    graph.add_crate(crate_a);
    graph.add_crate(crate_b);
    graph.add_crate(crate_c);

    let conflicts = graph.detect_conflicts();
    // serde is same version across all -> no conflict
    // arrow is 53.0 vs 54.0 -> conflict
    assert_eq!(conflicts.len(), 1);
    assert_eq!(conflicts[0].dependency, "arrow");
}

#[test]
fn test_fallback_graph_cov_026_cycle_error_topological_order() {
    let mut graph = DependencyGraph::new();
    add_test_crate(&mut graph, "x", 1, 0, 0);
    add_test_crate(&mut graph, "y", 1, 0, 0);

    add_normal_dep(&mut graph, "x", "y", "1.0");
    add_normal_dep(&mut graph, "y", "x", "1.0");

    let result = graph.topological_order();
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("Circular dependency"));
}

#[test]
fn test_fallback_graph_cov_027_release_order_with_cycle() {
    let mut graph = DependencyGraph::new();
    add_test_crate(&mut graph, "x", 1, 0, 0);
    add_test_crate(&mut graph, "y", 1, 0, 0);

    add_normal_dep(&mut graph, "x", "y", "1.0");
    add_normal_dep(&mut graph, "y", "x", "1.0");

    let result = graph.release_order_for("x");
    assert!(result.is_err());
}

#[test]
fn test_fallback_graph_cov_028_deep_chain() {
    // Build a deep chain: a -> b -> c -> d -> e
    let mut graph = DependencyGraph::new();
    for name in &["a", "b", "c", "d", "e"] {
        add_test_crate(&mut graph, name, 1, 0, 0);
    }
    add_normal_dep(&mut graph, "a", "b", "1.0");
    add_normal_dep(&mut graph, "b", "c", "1.0");
    add_normal_dep(&mut graph, "c", "d", "1.0");
    add_normal_dep(&mut graph, "d", "e", "1.0");

    assert!(!graph.has_cycles());
    let order = graph.topological_order().unwrap();
    assert_eq!(order.len(), 5);

    let e_pos = order.iter().position(|n| n == "e").unwrap();
    let a_pos = order.iter().position(|n| n == "a").unwrap();
    assert!(e_pos < a_pos);

    // Transitive deps of a
    let deps = graph.all_dependencies("a");
    assert_eq!(deps.len(), 4);

    // Release order for c should include c, d, e
    let ro = graph.release_order_for("c").unwrap();
    assert_eq!(ro.last().unwrap(), "c");
    assert_contains_crate(&ro, "d");
    assert_contains_crate(&ro, "e");
    assert!(!ro.contains(&"a".to_string()));
    assert!(!ro.contains(&"b".to_string()));
}

#[test]
fn test_fallback_graph_cov_029_dependents_multiple() {
    let mut graph = DependencyGraph::new();
    add_test_crate(&mut graph, "base", 1, 0, 0);
    add_test_crate(&mut graph, "dep1", 1, 0, 0);
    add_test_crate(&mut graph, "dep2", 1, 0, 0);
    add_test_crate(&mut graph, "dep3", 1, 0, 0);

    add_normal_dep(&mut graph, "dep1", "base", "1.0");
    add_normal_dep(&mut graph, "dep2", "base", "1.0");
    add_normal_dep(&mut graph, "dep3", "base", "1.0");

    let dependents = graph.dependents("base");
    assert_eq!(dependents.len(), 3);
}

#[test]
fn test_fallback_graph_cov_030_mixed_dep_types() {
    // Build, normal, and dev deps from one crate
    let mut graph = DependencyGraph::new();
    add_test_crate(&mut graph, "main", 1, 0, 0);
    add_test_crate(&mut graph, "lib-a", 1, 0, 0);
    add_test_crate(&mut graph, "lib-b", 1, 0, 0);
    add_test_crate(&mut graph, "lib-c", 1, 0, 0);

    add_normal_dep(&mut graph, "main", "lib-a", "1.0");
    add_dev_dep(&mut graph, "main", "lib-b", "1.0");
    graph.add_dependency(
        "main",
        "lib-c",
        DependencyEdge {
            version_req: "1.0".to_string(),
            is_path: false,
            kind: DependencyKind::Build,
        },
    );

    // Release graph excludes dev deps
    // main -> lib-a (normal), main -> lib-c (build) remain
    // main -> lib-b (dev) excluded
    assert!(!graph.has_cycles());
}

#[test]
fn test_fallback_graph_cov_031_ensure_node_idempotent() {
    let mut graph = DependencyGraph::new();
    add_test_crate(&mut graph, "node", 1, 0, 0);
    // Adding dependency from same node should not create duplicate
    add_normal_dep(&mut graph, "node", "other", "1.0");
    add_normal_dep(&mut graph, "node", "other", "2.0");

    assert!(graph.node_indices_contains("node"));
    assert!(graph.node_indices_contains("other"));
}

#[test]
fn test_fallback_graph_cov_032_path_dep_issue_fields() {
    let mut graph = DependencyGraph::new();
    add_test_crate(&mut graph, "from", 1, 0, 0);
    add_test_crate(&mut graph, "to", 1, 0, 0);
    add_path_dep(&mut graph, "from", "to");

    let issues = graph.find_path_dependencies();
    assert_eq!(issues.len(), 1);
    assert_eq!(issues[0].crate_name, "from");
    assert_eq!(issues[0].dependency, "to");
    assert!(issues[0].current.contains("path"));
    assert!(issues[0].recommended.is_none());
}

// =========================================================================
// Coverage gap tests: from_workspace (native-only)
// =========================================================================

/// Test from_workspace using the actual batuta project directory.
/// This covers lines 223-310 in graph.rs (the entire from_workspace method).
#[cfg(feature = "native")]
#[test]
fn test_from_workspace_batuta_project() {
    // Use the current project directory (batuta itself)
    let workspace_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let result = DependencyGraph::from_workspace(workspace_path);

    // Should succeed since batuta has a valid Cargo.toml
    assert!(result.is_ok(), "from_workspace failed: {:?}", result.err());

    let graph = result.unwrap();

    // batuta is a PAIML crate, so it should be in the graph
    assert!(graph.get_crate("batuta").is_some(), "batuta should be in the graph");

    // The graph should have at least 1 crate
    assert!(graph.crate_count() >= 1);
}

/// Test from_workspace without Cargo.toml falls back to binary detection.
#[cfg(feature = "native")]
#[test]
fn test_from_workspace_no_cargo_toml_uses_binary_fallback() {
    let result = DependencyGraph::from_workspace(std::path::Path::new(
        "/tmp/nonexistent_workspace_for_test",
    ));
    // Should succeed via binary detection fallback, not fail
    assert!(result.is_ok(), "Expected binary fallback, got: {:?}", result.err());
}

/// Test from_workspace with a minimal temp workspace containing a PAIML crate.
#[cfg(feature = "native")]
#[test]
fn test_from_workspace_minimal_paiml_workspace() {
    use tempfile::TempDir;

    let temp_dir = TempDir::new().unwrap();
    let project_dir = temp_dir.path();

    // Create a minimal workspace with a crate named "batuta" (a PAIML crate name)
    std::fs::create_dir_all(project_dir.join("src")).unwrap();
    std::fs::write(
        project_dir.join("Cargo.toml"),
        r#"[package]
name = "batuta"
version = "0.1.0"
edition = "2021"
"#,
    )
    .unwrap();
    std::fs::write(project_dir.join("src/lib.rs"), "").unwrap();

    let result = DependencyGraph::from_workspace(project_dir);
    assert!(result.is_ok(), "from_workspace failed: {:?}", result.err());

    let graph = result.unwrap();
    assert!(graph.get_crate("batuta").is_some());
    assert_eq!(graph.get_crate("batuta").unwrap().local_version, semver::Version::new(0, 1, 0));
    // No PAIML dependencies in this minimal workspace
    assert!(!graph.has_cycles());
}

/// Test from_workspace with a temp workspace containing two PAIML crates
/// and a dependency edge between them.
#[cfg(feature = "native")]
#[test]
fn test_from_workspace_with_paiml_dependency() {
    use tempfile::TempDir;

    let temp_dir = TempDir::new().unwrap();
    let ws_dir = temp_dir.path();

    // Create workspace root
    std::fs::write(
        ws_dir.join("Cargo.toml"),
        r#"[workspace]
members = ["trueno", "aprender"]
resolver = "2"
"#,
    )
    .unwrap();

    // Create "trueno" crate
    std::fs::create_dir_all(ws_dir.join("trueno/src")).unwrap();
    std::fs::write(
        ws_dir.join("trueno/Cargo.toml"),
        r#"[package]
name = "trueno"
version = "1.0.0"
edition = "2021"
"#,
    )
    .unwrap();
    std::fs::write(ws_dir.join("trueno/src/lib.rs"), "pub fn hello() {}").unwrap();

    // Create "aprender" crate that depends on trueno via path
    std::fs::create_dir_all(ws_dir.join("aprender/src")).unwrap();
    std::fs::write(
        ws_dir.join("aprender/Cargo.toml"),
        r#"[package]
name = "aprender"
version = "0.8.0"
edition = "2021"

[dependencies]
trueno = { path = "../trueno" }
"#,
    )
    .unwrap();
    std::fs::write(ws_dir.join("aprender/src/lib.rs"), "pub use trueno;").unwrap();

    let result = DependencyGraph::from_workspace(ws_dir);
    assert!(result.is_ok(), "from_workspace failed: {:?}", result.err());

    let graph = result.unwrap();

    // Both crates should be in the graph
    assert!(graph.get_crate("trueno").is_some());
    assert!(graph.get_crate("aprender").is_some());
    assert_eq!(graph.crate_count(), 2);

    // aprender should depend on trueno
    let deps = graph.all_dependencies("aprender");
    assert!(
        deps.contains(&"trueno".to_string()),
        "aprender should depend on trueno, got: {:?}",
        deps
    );

    // trueno should have aprender as dependent
    let dependents = graph.dependents("trueno");
    assert!(dependents.contains(&"aprender".to_string()));

    // Should have a path dependency
    let path_deps = graph.find_path_dependencies();
    assert!(
        path_deps.iter().any(|pd| pd.crate_name == "aprender" && pd.dependency == "trueno"),
        "Expected path dependency from aprender to trueno"
    );

    // No cycles
    assert!(!graph.has_cycles());

    // Topological order should work
    let order = graph.topological_order().unwrap();
    let trueno_pos = order.iter().position(|n| n == "trueno").unwrap();
    let aprender_pos = order.iter().position(|n| n == "aprender").unwrap();
    assert!(trueno_pos < aprender_pos);
}

/// Test from_workspace where non-workspace PAIML packages are resolved
/// but their deps should NOT be added (GH-25 fix).
#[cfg(feature = "native")]
#[test]
fn test_from_workspace_non_workspace_paiml_dep_filtered() {
    use tempfile::TempDir;

    let temp_dir = TempDir::new().unwrap();
    let ws_dir = temp_dir.path();

    // Single-crate workspace named "batuta" with no PAIML deps
    std::fs::create_dir_all(ws_dir.join("src")).unwrap();
    std::fs::write(
        ws_dir.join("Cargo.toml"),
        r#"[package]
name = "batuta"
version = "0.6.0"
edition = "2021"
"#,
    )
    .unwrap();
    std::fs::write(ws_dir.join("src/lib.rs"), "").unwrap();

    let result = DependencyGraph::from_workspace(ws_dir);
    assert!(result.is_ok());

    let graph = result.unwrap();
    // Only batuta should be in the graph (no resolved transitive PAIML deps
    // from the crates.io registry)
    assert_eq!(graph.crate_count(), 1);
    assert!(graph.get_crate("batuta").is_some());
}

/// Test from_workspace with dev and build dependencies to cover all DependencyKind
/// match arms (lines 276-279 in graph.rs).
#[cfg(feature = "native")]
#[test]
fn test_from_workspace_with_dev_and_build_deps() {
    use tempfile::TempDir;

    let temp_dir = TempDir::new().unwrap();
    let ws_dir = temp_dir.path();

    // Create workspace with three PAIML crates
    std::fs::write(
        ws_dir.join("Cargo.toml"),
        r#"[workspace]
members = ["trueno", "aprender", "entrenar"]
resolver = "2"
"#,
    )
    .unwrap();

    // Base crate: trueno
    std::fs::create_dir_all(ws_dir.join("trueno/src")).unwrap();
    std::fs::write(
        ws_dir.join("trueno/Cargo.toml"),
        r#"[package]
name = "trueno"
version = "1.0.0"
edition = "2021"
"#,
    )
    .unwrap();
    std::fs::write(ws_dir.join("trueno/src/lib.rs"), "pub fn t() {}").unwrap();

    // Crate with normal dep
    std::fs::create_dir_all(ws_dir.join("aprender/src")).unwrap();
    std::fs::write(
        ws_dir.join("aprender/Cargo.toml"),
        r#"[package]
name = "aprender"
version = "0.8.0"
edition = "2021"

[dependencies]
trueno = { path = "../trueno" }
"#,
    )
    .unwrap();
    std::fs::write(ws_dir.join("aprender/src/lib.rs"), "").unwrap();

    // Crate with dev-dependency on aprender and build-dependency on trueno
    std::fs::create_dir_all(ws_dir.join("entrenar/src")).unwrap();
    std::fs::write(
        ws_dir.join("entrenar/Cargo.toml"),
        r#"[package]
name = "entrenar"
version = "0.5.0"
edition = "2021"

[dev-dependencies]
aprender = { path = "../aprender" }

[build-dependencies]
trueno = { path = "../trueno" }
"#,
    )
    .unwrap();
    std::fs::write(ws_dir.join("entrenar/src/lib.rs"), "").unwrap();
    // Build script needed to justify build-dependency
    std::fs::write(ws_dir.join("entrenar/build.rs"), "fn main() {}").unwrap();

    let result = DependencyGraph::from_workspace(ws_dir);
    assert!(result.is_ok(), "from_workspace failed: {:?}", result.err());

    let graph = result.unwrap();

    // All three crates should be present
    assert_eq!(graph.crate_count(), 3);
    assert!(graph.get_crate("trueno").is_some());
    assert!(graph.get_crate("aprender").is_some());
    assert!(graph.get_crate("entrenar").is_some());

    // entrenar has dev-dep on aprender, so dev-dep cycle won't block cycle detection
    // The release graph should still work (dev deps are excluded from release order)
    assert!(!graph.has_cycles());
}

/// Test from_workspace with a non-path (crates.io) dependency between PAIML crates.
/// This covers the else branch at lines 300-301 (DependencyInfo::new for non-path deps).
#[cfg(feature = "native")]
#[test]
fn test_from_workspace_with_crates_io_dep() {
    use tempfile::TempDir;

    let temp_dir = TempDir::new().unwrap();
    let ws_dir = temp_dir.path();

    // Single crate workspace that depends on a published PAIML crate (trueno)
    std::fs::create_dir_all(ws_dir.join("src")).unwrap();
    std::fs::write(
        ws_dir.join("Cargo.toml"),
        r#"[package]
name = "aprender"
version = "0.25.0"
edition = "2021"

[dependencies]
trueno = "0.14"
"#,
    )
    .unwrap();
    std::fs::write(ws_dir.join("src/lib.rs"), "").unwrap();

    let result = DependencyGraph::from_workspace(ws_dir);
    assert!(result.is_ok(), "from_workspace failed: {:?}", result.err());

    let graph = result.unwrap();

    // aprender is the workspace member
    assert!(graph.get_crate("aprender").is_some());

    // trueno should also be in the graph (resolved from crates.io)
    assert!(graph.get_crate("trueno").is_some());

    // aprender should depend on trueno
    let deps = graph.all_dependencies("aprender");
    assert!(deps.contains(&"trueno".to_string()), "aprender should depend on trueno");

    // The dependency should NOT be a path dependency
    let path_deps = graph.find_path_dependencies();
    assert!(path_deps.is_empty(), "No path dependencies expected for crates.io dep");
}

// =========================================================================
// Coverage gap tests: collect_dependencies visited-node early return
// =========================================================================

/// Test that collect_dependencies properly handles already-visited nodes
/// in a diamond dependency graph where the same leaf is reachable via two paths.
/// This specifically covers the early return at line 438 (visited.contains check).
#[test]
fn test_collect_deps_visited_node_early_return() {
    let mut graph = DependencyGraph::new();
    add_test_crate(&mut graph, "root", 1, 0, 0);
    add_test_crate(&mut graph, "left", 1, 0, 0);
    add_test_crate(&mut graph, "right", 1, 0, 0);
    add_test_crate(&mut graph, "shared", 1, 0, 0);

    // root -> left -> shared
    // root -> right -> shared
    add_normal_dep(&mut graph, "root", "left", "1.0");
    add_normal_dep(&mut graph, "root", "right", "1.0");
    add_normal_dep(&mut graph, "left", "shared", "1.0");
    add_normal_dep(&mut graph, "right", "shared", "1.0");

    // all_dependencies calls collect_dependencies internally
    let deps = graph.all_dependencies("root");

    // "shared" should only appear once despite being reachable via two paths
    assert_eq!(deps.iter().filter(|d| *d == "shared").count(), 1);
    // All 3 deps should be present
    assert_eq!(deps.len(), 3);
    assert_contains_crate(&deps, "left");
    assert_contains_crate(&deps, "right");
    assert_contains_crate(&deps, "shared");
}

/// Test collect_dependencies with a deeper diamond where shared node
/// has its own dependencies, ensuring visited-node early return prevents
/// re-traversal of subtrees.
#[test]
fn test_collect_deps_deep_diamond_with_subtree() {
    let mut graph = DependencyGraph::new();
    for name in &["root", "a", "b", "shared", "deep"] {
        add_test_crate(&mut graph, name, 1, 0, 0);
    }

    // root -> a -> shared -> deep
    // root -> b -> shared -> deep
    add_normal_dep(&mut graph, "root", "a", "1.0");
    add_normal_dep(&mut graph, "root", "b", "1.0");
    add_normal_dep(&mut graph, "a", "shared", "1.0");
    add_normal_dep(&mut graph, "b", "shared", "1.0");
    add_normal_dep(&mut graph, "shared", "deep", "1.0");

    let deps = graph.all_dependencies("root");
    assert_eq!(deps.len(), 4);
    assert_contains_crate(&deps, "a");
    assert_contains_crate(&deps, "b");
    assert_contains_crate(&deps, "shared");
    assert_contains_crate(&deps, "deep");

    // "shared" traversed once, "deep" should also appear only once
    assert_eq!(deps.iter().filter(|d| *d == "deep").count(), 1);
}
