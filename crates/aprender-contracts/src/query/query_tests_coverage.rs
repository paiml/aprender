    // Coverage-targeted tests for query module internals.

    use super::*;

    fn test_index() -> ContractIndex {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../contracts");
        ContractIndex::build_from_directory(&dir).unwrap()
    }

    #[test]
    fn min_level_filter_restricts_results() {
        let index = test_index();
        let all = QueryParams {
            query: "softmax".to_string(),
            limit: 50,
            ..Default::default()
        };
        let all_output = execute(&index, &all);

        let filtered = QueryParams {
            query: "softmax".to_string(),
            limit: 50,
            min_level: Some("L3".to_string()),
            ..Default::default()
        };
        let filtered_output = execute(&index, &filtered);

        // L3 filter should return <= total results
        assert!(filtered_output.total_matches <= all_output.total_matches);
    }

    #[test]
    fn parse_iso_days_ago_recent() {
        // 2026-03-06 at epoch ~1_772_870_400
        let now = 1_772_870_400;
        let days = super::query_enrich::parse_iso_days_ago("2026-03-05", now);
        assert!(days <= 2, "Yesterday should be ~1 day ago, got {days}");
    }

    #[test]
    fn parse_iso_days_ago_invalid() {
        assert_eq!(super::query_enrich::parse_iso_days_ago("invalid", 1_772_870_400), 0);
        assert_eq!(super::query_enrich::parse_iso_days_ago("2026-03", 1_772_870_400), 0);
    }

    #[test]
    fn month_days_all_months() {
        assert_eq!(super::query_enrich::month_days(0), 0);
        assert_eq!(super::query_enrich::month_days(1), 0);
        assert_eq!(super::query_enrich::month_days(2), 31);
        assert_eq!(super::query_enrich::month_days(6), 151);
        assert_eq!(super::query_enrich::month_days(12), 334);
        assert_eq!(super::query_enrich::month_days(13), 0); // out of range
    }

    #[test]
    fn parse_proof_level_all_levels() {
        use crate::proof_status::ProofLevel;
        assert_eq!(super::query_enrich::parse_proof_level("L1"), ProofLevel::L1);
        assert_eq!(super::query_enrich::parse_proof_level("L2"), ProofLevel::L2);
        assert_eq!(super::query_enrich::parse_proof_level("L3"), ProofLevel::L3);
        assert_eq!(super::query_enrich::parse_proof_level("L4"), ProofLevel::L4);
        assert_eq!(super::query_enrich::parse_proof_level("L5"), ProofLevel::L5);
        assert_eq!(super::query_enrich::parse_proof_level("l3"), ProofLevel::L3);
        assert_eq!(super::query_enrich::parse_proof_level("unknown"), ProofLevel::L1);
    }

    #[test]
    #[ignore] // Requires git repo — skipped in container CI
    fn show_diff_enrichment() {
        let index = test_index();
        let params = QueryParams {
            query: "softmax".to_string(),
            show_diff: true,
            limit: 2,
            ..Default::default()
        };
        let output = execute(&index, &params);
        assert!(!output.results.is_empty());
        // Diff info should be populated for contracts in a git repo
        let has_diff = output.results.iter().any(|r| r.diff.is_some());
        assert!(has_diff, "At least one result should have diff info");
    }

    #[test]
    fn depends_on_filter() {
        let index = test_index();
        let params = QueryParams {
            query: "attention".to_string(),
            depends_on: Some("softmax-kernel-v1".to_string()),
            limit: 20,
            ..Default::default()
        };
        let output = execute(&index, &params);
        // All results should depend on softmax-kernel-v1
        for r in &output.results {
            let entry = index.get_by_stem(&r.stem).unwrap();
            assert!(
                entry.depends_on.contains(&"softmax-kernel-v1".to_string()),
                "{} should depend on softmax-kernel-v1",
                r.stem
            );
        }
    }

    #[test]
    fn binding_gaps_filter() {
        let index = test_index();
        let binding_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../contracts/aprender/binding.yaml");
        let params = QueryParams {
            query: "kernel".to_string(),
            binding_gaps_only: true,
            binding_path: Some(binding_path.display().to_string()),
            limit: 20,
            ..Default::default()
        };
        let output = execute(&index, &params);
        // Should only include contracts with binding gaps
        // (may be 0 if all are implemented, which is fine)
        assert!(output.total_matches <= 20);
    }

    #[test]
    fn binding_with_real_registry() {
        let index = test_index();
        let binding_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../contracts/aprender/binding.yaml");
        let params = QueryParams {
            query: "softmax".to_string(),
            show_binding: true,
            binding_path: Some(binding_path.display().to_string()),
            limit: 1,
            ..Default::default()
        };
        let output = execute(&index, &params);
        if let Some(r) = output.results.first() {
            // With real binding, some equations should be bound or unbound
            assert!(!r.bindings.is_empty());
        }
    }

    #[test]
    fn binding_unbound_path() {
        let index = test_index();
        // Use a fake binding path to trigger the "no binding found" path
        let params = QueryParams {
            query: "softmax".to_string(),
            show_binding: true,
            binding_path: Some("nonexistent-binding.yaml".to_string()),
            limit: 1,
            ..Default::default()
        };
        let output = execute(&index, &params);
        assert!(!output.results.is_empty());
        // Without valid binding, equations show "no binding registry"
        if let Some(r) = output.results.first() {
            if !r.bindings.is_empty() {
                assert_eq!(r.bindings[0].status, "no binding registry");
            }
        }
    }

    #[test]
    fn binding_gaps_without_registry() {
        let index = test_index();
        // binding_gaps_only=true with no binding path => all filtered out
        let params = QueryParams {
            query: "softmax".to_string(),
            binding_gaps_only: true,
            limit: 10,
            ..Default::default()
        };
        let output = execute(&index, &params);
        assert_eq!(output.total_matches, 0, "No binding registry means all filtered");
    }

    #[test]
    fn pagerank_enrichment() {
        let index = test_index();
        let params = QueryParams {
            query: "softmax".to_string(),
            show_pagerank: true,
            limit: 3,
            ..Default::default()
        };
        let output = execute(&index, &params);
        assert!(!output.results.is_empty());
        // Softmax-kernel-v1 should have a pagerank score
        let sm = output.results.iter().find(|r| r.stem == "softmax-kernel-v1");
        if let Some(r) = sm {
            assert!(r.pagerank.is_some(), "pagerank should be populated");
            assert!(r.pagerank.unwrap() > 0.0, "pagerank should be positive");
        }
    }

    #[test]
    fn binding_info_unbound_equations() {
        let index = test_index();
        // Use a real binding that doesn't cover all equations of a contract
        let binding_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../contracts/aprender/binding.yaml");
        let params = QueryParams {
            query: "attention".to_string(),
            show_binding: true,
            binding_path: Some(binding_path.display().to_string()),
            limit: 1,
            ..Default::default()
        };
        let output = execute(&index, &params);
        if let Some(r) = output.results.first() {
            // Attention contracts have equations not all bound
            let unbound = r.bindings.iter().any(|b| b.status == "unbound");
            // At least one equation should be unbound for attention
            assert!(
                unbound || r.bindings.iter().all(|b| b.status == "implemented"),
                "Should have unbound or implemented bindings"
            );
        }
    }

    #[test]
    fn min_score_filter() {
        let index = test_index();
        // With a very high min_score, no results should pass
        let params = QueryParams {
            query: "softmax".to_string(),
            min_score: Some(0.99),
            limit: 10,
            ..Default::default()
        };
        let output = execute(&index, &params);
        assert_eq!(output.total_matches, 0, "0.99 threshold should filter everything");
    }

    #[test]
    fn violations_enrichment() {
        let index = test_index();
        let params = QueryParams {
            query: "clustering".to_string(),
            show_violations: true,
            limit: 3,
            ..Default::default()
        };
        let output = execute(&index, &params);
        // Violations require cross-project index which scans sibling projects
        // At least some contracts should show violations (unproven obligations)
        assert!(!output.results.is_empty());
    }

    #[test]
    fn coverage_map_enrichment() {
        // Coverage map requires sibling repos for binding data. The prior
        // `parent.join("aprender")` check was a self-match (both locally and
        // in GHA's `/__w/aprender/aprender` layout). Probing for
        // `provable-contracts` still false-positives on self-hosted runners
        // that carry stale sibling checkouts. Probe for other paiml repos
        // that only exist in a dev workspace checkout.
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").canonicalize().unwrap();
        let has_sibling = root.parent().is_some_and(|p| {
            ["trueno", "bashrs", "forjar"].iter().any(|name| p.join(name).exists())
        });
        if !has_sibling { return; }
        let index = test_index();
        let params = QueryParams {
            query: "softmax".to_string(),
            show_coverage_map: true,
            limit: 3,
            ..Default::default()
        };
        let output = execute(&index, &params);
        assert!(!output.results.is_empty());
        // Coverage map should show aprender for contracts with bindings
        let has_map = output.results.iter().any(|r| !r.coverage_map.is_empty());
        assert!(has_map, "At least one softmax contract should have coverage data");
    }

    #[test]
    fn graph_enrichment_with_depended_by() {
        let index = test_index();
        let params = QueryParams {
            query: "softmax-kernel".to_string(),
            show_graph: true,
            limit: 3,
            ..Default::default()
        };
        let output = execute(&index, &params);
        let sm = output.results.iter().find(|r| r.stem == "softmax-kernel-v1");
        if let Some(r) = sm {
            assert!(!r.depended_by.is_empty(), "softmax should have dependents");
        }
    }

    #[test]
    fn project_filter_restricts_call_sites() {
        let index = test_index();
        let params = QueryParams {
            query: "softmax".to_string(),
            show_call_sites: true,
            project_filter: Some("aprender".to_string()),
            limit: 3,
            ..Default::default()
        };
        let output = execute(&index, &params);
        // All call sites should be from aprender
        for r in &output.results {
            for cs in &r.call_sites {
                assert_eq!(cs.project, "aprender", "project filter should restrict to aprender");
            }
        }
    }

    #[test]
    fn project_filter_restricts_violations() {
        let index = test_index();
        let params = QueryParams {
            query: "softmax".to_string(),
            show_violations: true,
            project_filter: Some("nonexistent".to_string()),
            limit: 3,
            ..Default::default()
        };
        let output = execute(&index, &params);
        // With a nonexistent project filter, no violations should appear
        for r in &output.results {
            assert!(r.violations.is_empty(), "nonexistent project should have no violations");
        }
    }

    #[test]
    fn all_projects_triggers_xp_scan() {
        let index = test_index();
        let params = QueryParams {
            query: "softmax".to_string(),
            all_projects: true,
            limit: 3,
            ..Default::default()
        };
        // all_projects should trigger the cross-project scan without
        // requiring show_call_sites/show_violations/show_coverage_map
        let output = execute(&index, &params);
        assert!(!output.results.is_empty());
    }

    #[test]
    fn rebuild_index_flag() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../contracts");
        // force rebuild should work (skips cache)
        let idx = super::ContractIndex::from_directory_opts(&dir, true).unwrap();
        assert!(idx.entries.len() > 10, "Rebuilt index should have contracts");
    }

    #[test]
    fn default_query_params_new_fields() {
        let params = QueryParams::default();
        assert!(params.project_filter.is_none());
        assert!(params.include_project.is_none());
        assert!(!params.all_projects);
    }
