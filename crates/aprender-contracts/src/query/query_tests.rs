    use super::*;

    fn test_index() -> ContractIndex {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../contracts");
        ContractIndex::build_from_directory(&dir).unwrap()
    }

    #[test]
    fn semantic_query_returns_results() {
        let index = test_index();
        let params = QueryParams {
            query: "softmax numerical stability".to_string(),
            limit: 50, // Increased from default for 405-contract corpus
            ..Default::default()
        };
        let output = execute(&index, &params);
        assert!(!output.results.is_empty());
        assert!(
            output.results.iter().any(|r| r.stem.contains("softmax")),
            "Results should include softmax contract, got: {:?}",
            output.results.iter().map(|r| &r.stem).collect::<Vec<_>>()
        );
    }

    #[test]
    fn literal_query_finds_contracts() {
        let index = test_index();
        let params = QueryParams {
            query: "RMSNorm".to_string(),
            mode: SearchMode::Literal,
            ..Default::default()
        };
        let output = execute(&index, &params);
        assert!(!output.results.is_empty());
    }

    #[test]
    fn regex_query_works() {
        let index = test_index();
        let params = QueryParams {
            query: r"(?i)softmax".to_string(),
            mode: SearchMode::Regex,
            ..Default::default()
        };
        let output = execute(&index, &params);
        assert!(!output.results.is_empty());
    }

    #[test]
    fn obligation_filter_narrows_results() {
        let index = test_index();
        let all = QueryParams {
            query: "kernel".to_string(),
            ..Default::default()
        };
        let filtered = QueryParams {
            query: "kernel".to_string(),
            obligation_filter: Some("invariant".to_string()),
            ..Default::default()
        };
        let all_out = execute(&index, &all);
        let filtered_out = execute(&index, &filtered);
        assert!(filtered_out.total_matches <= all_out.total_matches);
    }

    #[test]
    fn limit_caps_results() {
        let index = test_index();
        let params = QueryParams {
            query: "kernel".to_string(),
            limit: 3,
            ..Default::default()
        };
        let output = execute(&index, &params);
        assert!(output.results.len() <= 3);
    }

    #[test]
    fn show_score_enriches_results() {
        let index = test_index();
        let params = QueryParams {
            query: "softmax".to_string(),
            show_score: true,
            limit: 1,
            ..Default::default()
        };
        let output = execute(&index, &params);
        assert!(!output.results.is_empty());
        assert!(output.results[0].score.is_some());
    }

    #[test]
    fn display_output_is_valid() {
        let index = test_index();
        let params = QueryParams {
            query: "softmax".to_string(),
            limit: 2,
            ..Default::default()
        };
        let output = execute(&index, &params);
        let text = output.to_string();
        assert!(text.contains("[1]"));
        assert!(text.contains("softmax"));
    }

    #[test]
    fn proof_status_enrichment_works() {
        let index = test_index();
        let params = QueryParams {
            query: "softmax".to_string(),
            show_proof_status: true,
            limit: 1,
            ..Default::default()
        };
        let output = execute(&index, &params);
        assert!(!output.results.is_empty());
        let ps = output.results[0].proof_status.as_ref().unwrap();
        assert!(!ps.level.is_empty());
    }

    #[test]
    fn binding_enrichment_without_registry() {
        let index = test_index();
        let params = QueryParams {
            query: "softmax".to_string(),
            show_binding: true,
            limit: 1,
            ..Default::default()
        };
        let output = execute(&index, &params);
        assert!(!output.results.is_empty());
        assert!(!output.results[0].bindings.is_empty());
        assert_eq!(output.results[0].bindings[0].status, "no binding registry");
    }

    #[test]
    fn binding_enrichment_with_registry() {
        let binding_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../contracts/aprender/binding.yaml");
        let index = test_index();
        let params = QueryParams {
            query: "softmax".to_string(),
            show_binding: true,
            binding_path: Some(binding_path.display().to_string()),
            limit: 5,
            ..Default::default()
        };
        let output = execute(&index, &params);
        let has_bound = output.results.iter().any(|r| {
            r.bindings
                .iter()
                .any(|b| b.status != "no binding registry")
        });
        assert!(has_bound, "Should find implemented bindings");
    }

    #[test]
    fn min_score_filter_works() {
        let index = test_index();
        let high = QueryParams {
            query: "kernel".to_string(),
            min_score: Some(0.80),
            ..Default::default()
        };
        let low = QueryParams {
            query: "kernel".to_string(),
            min_score: Some(0.10),
            ..Default::default()
        };
        let high_out = execute(&index, &high);
        let low_out = execute(&index, &low);
        assert!(high_out.total_matches <= low_out.total_matches);
    }

    #[test]
    fn markdown_output_format() {
        let index = test_index();
        let params = QueryParams {
            query: "softmax".to_string(),
            show_score: true,
            show_paper: true,
            limit: 2,
            ..Default::default()
        };
        let output = execute(&index, &params);
        let md = output.to_markdown();
        assert!(md.contains("## Query:"));
        assert!(md.contains("### 1."));
        assert!(md.contains("**Score:**"));
    }

    #[test]
    fn empty_query_returns_empty() {
        let index = test_index();
        let params = QueryParams {
            query: String::new(),
            ..Default::default()
        };
        let output = execute(&index, &params);
        assert!(output.results.is_empty());
    }

    #[test]
    fn display_with_all_enrichment() {
        let index = test_index();
        let binding_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../contracts/aprender/binding.yaml");
        let params = QueryParams {
            query: "softmax".to_string(),
            show_score: true,
            show_proof_status: true,
            show_binding: true,
            show_graph: true,
            show_paper: true,
            binding_path: Some(binding_path.display().to_string()),
            limit: 2,
            ..Default::default()
        };
        let output = execute(&index, &params);
        let text = output.to_string();
        assert!(text.contains("Score:"));
        assert!(text.contains("Proof Level:"));
    }

    #[test]
    fn markdown_with_proof_and_binding() {
        let index = test_index();
        let binding_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../contracts/aprender/binding.yaml");
        let params = QueryParams {
            query: "softmax".to_string(),
            show_score: true,
            show_proof_status: true,
            show_binding: true,
            show_paper: true,
            binding_path: Some(binding_path.display().to_string()),
            limit: 2,
            ..Default::default()
        };
        let output = execute(&index, &params);
        let md = output.to_markdown();
        assert!(md.contains("**Proof Level:**"));
        assert!(md.contains("**Bindings:**"));
        assert!(md.contains("**Papers:**"));
    }

    #[test]
    fn invalid_regex_returns_empty() {
        let index = test_index();
        let params = QueryParams {
            query: r"[invalid(".to_string(),
            mode: SearchMode::Regex,
            ..Default::default()
        };
        let output = execute(&index, &params);
        assert!(output.results.is_empty());
    }

    #[test]
    fn unproven_filter_narrows() {
        let index = test_index();
        let all = QueryParams {
            query: "kernel".to_string(),
            limit: 100,
            ..Default::default()
        };
        let unproven = QueryParams {
            query: "kernel".to_string(),
            unproven_only: true,
            limit: 100,
            ..Default::default()
        };
        let all_out = execute(&index, &all);
        let unproven_out = execute(&index, &unproven);
        assert!(unproven_out.total_matches <= all_out.total_matches);
    }

    #[test]
    fn kind_filter_narrows_to_registries() {
        use crate::schema::ContractKind;
        let index = test_index();
        let all = QueryParams {
            query: ".".to_string(),
            mode: SearchMode::Regex,
            limit: 1000,
            ..Default::default()
        };
        let registries = QueryParams {
            query: ".".to_string(),
            mode: SearchMode::Regex,
            limit: 1000,
            kind_filter: Some(ContractKind::Registry),
            ..Default::default()
        };
        let all_out = execute(&index, &all);
        let registry_out = execute(&index, &registries);
        assert!(registry_out.total_matches < all_out.total_matches);
        assert!(
            registry_out.total_matches > 0,
            "contracts dir should contain at least one registry"
        );
        // Every returned result must be a registry.
        for r in &registry_out.results {
            assert_eq!(
                r.kind,
                ContractKind::Registry,
                "{} returned by --kind registry should be Registry kind, got {:?}",
                r.stem,
                r.kind,
            );
        }
    }

    #[test]
    fn kind_filter_pattern_finds_pattern_contracts() {
        use crate::schema::ContractKind;
        let index = test_index();
        let patterns = QueryParams {
            query: ".".to_string(),
            mode: SearchMode::Regex,
            limit: 100,
            kind_filter: Some(ContractKind::Pattern),
            ..Default::default()
        };
        let output = execute(&index, &patterns);
        // Repo ships 4 pattern contracts under contracts/patterns/.
        assert!(
            output.total_matches >= 4,
            "expected >= 4 pattern contracts, got {}",
            output.total_matches,
        );
        for r in &output.results {
            assert_eq!(r.kind, ContractKind::Pattern);
        }
    }

    #[test]
    fn kind_filter_kernel_excludes_registries() {
        use crate::schema::ContractKind;
        let index = test_index();
        let kernels = QueryParams {
            query: ".".to_string(),
            mode: SearchMode::Regex,
            limit: 1000,
            kind_filter: Some(ContractKind::Kernel),
            ..Default::default()
        };
        let output = execute(&index, &kernels);
        for r in &output.results {
            assert_ne!(
                r.kind,
                ContractKind::Registry,
                "{} kernel result should not be a registry",
                r.stem,
            );
        }
    }

    #[test]
    fn depended_by_filter() {
        let index = test_index();
        let params = QueryParams {
            query: "softmax".to_string(),
            depended_by: Some("attention-kernel-v1".to_string()),
            limit: 100,
            ..Default::default()
        };
        let output = execute(&index, &params);
        for r in &output.results {
            let att = index.get_by_stem("attention-kernel-v1").unwrap();
            assert!(
                att.depends_on.contains(&r.stem),
                "{} should be a dependency of attention-kernel-v1",
                r.stem
            );
        }
    }

    #[test]
    fn case_sensitive_literal() {
        let index = test_index();
        let insensitive = QueryParams {
            query: "rmsnorm".to_string(),
            mode: SearchMode::Literal,
            case_sensitive: false,
            limit: 100,
            ..Default::default()
        };
        let sensitive = QueryParams {
            query: "rmsnorm".to_string(),
            mode: SearchMode::Literal,
            case_sensitive: true,
            limit: 100,
            ..Default::default()
        };
        let insensitive_out = execute(&index, &insensitive);
        let sensitive_out = execute(&index, &sensitive);
        assert!(insensitive_out.total_matches >= sensitive_out.total_matches);
    }

    #[test]
    fn graph_enrichment_shows_depends() {
        let index = test_index();
        let params = QueryParams {
            query: "attention".to_string(),
            show_graph: true,
            limit: 5,
            ..Default::default()
        };
        let output = execute(&index, &params);
        let text = output.to_string();
        let has_deps = output.results.iter().any(|r| !r.depends_on.is_empty());
        if has_deps {
            assert!(text.contains("Depends on:"));
        }
    }

    #[test]
    fn get_by_equation_works() {
        let index = test_index();
        let results = index.get_by_equation("softmax");
        assert!(!results.is_empty());
    }

    #[test]
    fn get_by_obligation_works() {
        let index = test_index();
        let results = index.get_by_obligation("invariant");
        assert!(!results.is_empty());
    }

    #[test]
    fn clean_path_strips_prefix() {
        assert_eq!(
            clean_path("/home/user/src/provable-contracts/contracts/softmax-kernel-v1.yaml"),
            "contracts/softmax-kernel-v1.yaml"
        );
        assert_eq!(
            clean_path("../../contracts/aprender/binding.yaml"),
            "contracts/aprender/binding.yaml"
        );
        assert_eq!(clean_path("no-match.yaml"), "no-match.yaml");
    }

    #[test]
    fn query_result_path_is_clean() {
        let index = test_index();
        let params = QueryParams {
            query: "softmax".to_string(),
            limit: 1,
            ..Default::default()
        };
        let output = execute(&index, &params);
        let path = &output.results[0].path;
        assert!(
            path.starts_with("contracts/"),
            "path should be relative: {path}"
        );
        assert!(!path.contains("/../"), "path should not contain /../: {path}");
    }
