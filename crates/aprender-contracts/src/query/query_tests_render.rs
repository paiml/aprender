    // Rendering tests for query result Display and Markdown output.
    // Split from query_tests_coverage.rs to stay under 500 lines.

    use super::*;
    use crate::schema::ContractKind;

    #[test]
    fn display_with_diff_and_bindings() {
        let r = QueryResult {
            rank: 1,
            stem: "test-v1".to_string(),
            path: "test.yaml".to_string(),
            relevance: 0.9,
            kind: ContractKind::default(),
            description: "test contract".to_string(),
            equations: vec!["eq1".to_string()],
            obligation_count: 3,
            references: vec![],
            depends_on: vec!["dep-v1".to_string()],
            depended_by: vec!["consumer-v1".to_string()],
            score: None,
            proof_status: None,
            bindings: vec![EquationBinding {
                equation: "eq1".to_string(),
                status: "implemented".to_string(),
                module_path: Some("mod::path".to_string()),
            }],
            diff: Some(DiffInfo {
                last_modified: "2026-03-01".to_string(),
                days_ago: 5,
                commit_hash: "abc1234def".to_string(),
            }),
            pagerank: Some(0.042),
            call_sites: vec![],
            violations: vec![],
            coverage_map: vec![],
        };
        let text = format!("{r}");
        assert!(text.contains("Bindings:"));
        assert!(text.contains("eq1: implemented (mod::path)"));
        assert!(text.contains("Last modified: 2026-03-01"));
        assert!(text.contains("Depends on: dep-v1"));
        assert!(text.contains("Depended by: consumer-v1"));
        assert!(text.contains("PageRank: 0.0420"));
    }

    #[test]
    fn markdown_with_diff_and_bindings() {
        let output = QueryOutput {
            query: "test".to_string(),
            total_matches: 1,
            results: vec![QueryResult {
                rank: 1,
                stem: "test-v1".to_string(),
                path: "test.yaml".to_string(),
                relevance: 0.9,
            kind: ContractKind::default(),
                description: "test".to_string(),
                equations: vec![],
                obligation_count: 1,
                references: vec![],
                depends_on: vec!["dep-v1".to_string()],
                depended_by: vec!["consumer-v1".to_string()],
                score: None,
                proof_status: None,
                bindings: vec![EquationBinding {
                    equation: "eq1".to_string(),
                    status: "unbound".to_string(),
                    module_path: None,
                }],
                diff: Some(DiffInfo {
                    last_modified: "2026-03-01".to_string(),
                    days_ago: 5,
                    commit_hash: "abc1234".to_string(),
                }),
                pagerank: None,
                call_sites: vec![],
                violations: vec![],
                coverage_map: vec![],
            }],
        };
        let md = output.to_markdown();
        assert!(md.contains("**Bindings:**"));
        assert!(md.contains("`eq1`: unbound"));
        assert!(md.contains("**Last modified:**"));
        assert!(md.contains("**Depends on:** dep-v1"));
        assert!(md.contains("**Depended by:** consumer-v1"));
    }

    #[test]
    fn display_with_violations_and_coverage() {
        let r = QueryResult {
            rank: 1,
            stem: "test-v1".to_string(),
            path: "test.yaml".to_string(),
            relevance: 0.9,
            kind: ContractKind::default(),
            description: "test".to_string(),
            equations: vec![],
            obligation_count: 3,
            references: vec![],
            depends_on: vec![],
            depended_by: vec![],
            score: None,
            proof_status: None,
            bindings: vec![],
            diff: None,
            pagerank: None,
            call_sites: vec![],
            violations: vec![ViolationInfo {
                project: "aprender".to_string(),
                kind: "binding_gap".to_string(),
                detail: "softmax: not_implemented".to_string(),
            }],
            coverage_map: vec![types::ProjectCoverage {
                project: "aprender".to_string(),
                call_sites: 2,
                binding_refs: 3,
                binding_implemented: 2,
                binding_total: 3,
            }],
        };
        let text = format!("{r}");
        assert!(text.contains("Violations (1):"));
        assert!(text.contains("binding_gap"));
        assert!(text.contains("Coverage map:"));
        assert!(text.contains("aprender"));
    }

    #[test]
    fn markdown_with_violations_and_coverage() {
        let output = QueryOutput {
            query: "test".to_string(),
            total_matches: 1,
            results: vec![QueryResult {
                rank: 1,
                stem: "test-v1".to_string(),
                path: "test.yaml".to_string(),
                relevance: 0.9,
            kind: ContractKind::default(),
                description: "test".to_string(),
                equations: vec![],
                obligation_count: 1,
                references: vec![],
                depends_on: vec![],
                depended_by: vec![],
                score: None,
                proof_status: None,
                bindings: vec![],
                diff: None,
                pagerank: None,
                call_sites: vec![],
                violations: vec![ViolationInfo {
                    project: "trueno".to_string(),
                    kind: "unproven_obligations".to_string(),
                    detail: "2/5 lack Kani".to_string(),
                }],
                coverage_map: vec![types::ProjectCoverage {
                    project: "trueno".to_string(),
                    call_sites: 1,
                    binding_refs: 2,
                    binding_implemented: 1,
                    binding_total: 2,
                }],
            }],
        };
        let md = output.to_markdown();
        assert!(md.contains("**Violations"));
        assert!(md.contains("unproven_obligations"));
        assert!(md.contains("**Coverage map:**"));
        assert!(md.contains("50%"));
    }

    #[test]
    fn coverage_bar_rendering() {
        let full = super::types::ProjectCoverage {
            project: "p".to_string(),
            call_sites: 1,
            binding_refs: 3,
            binding_implemented: 3,
            binding_total: 3,
        };
        let empty = super::types::ProjectCoverage {
            project: "p".to_string(),
            call_sites: 0,
            binding_refs: 0,
            binding_implemented: 0,
            binding_total: 0,
        };
        let r = QueryResult {
            rank: 1,
            stem: "t".to_string(),
            path: "t.yaml".to_string(),
            relevance: 0.5,
            kind: ContractKind::default(),
            description: "t".to_string(),
            equations: vec![],
            obligation_count: 0,
            references: vec![],
            depends_on: vec![],
            depended_by: vec![],
            score: None,
            proof_status: None,
            bindings: vec![],
            diff: None,
            pagerank: None,
            call_sites: vec![],
            violations: vec![],
            coverage_map: vec![full, empty],
        };
        let text = format!("{r}");
        assert!(text.contains("██████"));
        assert!(text.contains("--"));
    }

    #[test]
    fn markdown_all_enrichment_paths() {
        let output = QueryOutput {
            query: "all-enrichment".to_string(),
            total_matches: 1,
            results: vec![QueryResult {
                rank: 1,
                stem: "test-v1".to_string(),
                path: "test.yaml".to_string(),
                relevance: 0.95,
            kind: ContractKind::default(),
                description: "full enrichment test".to_string(),
                equations: vec!["eq1".to_string()],
                obligation_count: 5,
                references: vec!["Paper A".to_string()],
                depends_on: vec!["dep-v1".to_string()],
                depended_by: vec!["cons-v1".to_string()],
                score: Some(ScoreInfo {
                    composite: 0.85,
                    grade: "B".to_string(),
                    spec_depth: 0.9,
                    falsification: 0.8,
                    kani: 0.7,
                    lean: 0.0,
                    binding: 1.0,
                }),
                proof_status: Some(ProofStatusInfo {
                    level: "L3".to_string(),
                    obligations: 5,
                    falsification_tests: 5,
                    kani_harnesses: 3,
                    lean_proved: 0,
                }),
                bindings: vec![EquationBinding {
                    equation: "eq1".to_string(),
                    status: "implemented".to_string(),
                    module_path: Some("mod::path".to_string()),
                }],
                diff: Some(DiffInfo {
                    last_modified: "2026-03-01".to_string(),
                    days_ago: 5,
                    commit_hash: "abc1234def".to_string(),
                }),
                pagerank: Some(0.042),
                call_sites: vec![types::CallSiteInfo {
                    project: "aprender".to_string(),
                    file: "src/nn.rs".to_string(),
                    line: 42,
                    equation: Some("eq1".to_string()),
                }],
                violations: vec![ViolationInfo {
                    project: "aprender".to_string(),
                    kind: "binding_gap".to_string(),
                    detail: "eq2: not_implemented".to_string(),
                }],
                coverage_map: vec![types::ProjectCoverage {
                    project: "aprender".to_string(),
                    call_sites: 1,
                    binding_refs: 2,
                    binding_implemented: 1,
                    binding_total: 2,
                }],
            }],
        };
        let md = output.to_markdown();
        assert!(md.contains("**Score:** 0.85 (Grade B)"));
        assert!(md.contains("**Proof Level:** L3"));
        assert!(md.contains("**Bindings:**"));
        assert!(md.contains("`eq1`: implemented"));
        assert!(md.contains("**Last modified:** 2026-03-01"));
        assert!(md.contains("**PageRank:** 0.0420"));
        assert!(md.contains("**Call sites**"));
        assert!(md.contains("aprender/src/nn.rs:42"));
        assert!(md.contains("**Violations"));
        assert!(md.contains("binding_gap"));
        assert!(md.contains("**Coverage map:**"));
        assert!(md.contains("50%"));
        assert!(md.contains("**Papers:** Paper A"));
        assert!(md.contains("**Depends on:** dep-v1"));
        assert!(md.contains("**Depended by:** cons-v1"));
    }

    #[test]
    fn text_display_with_score_and_proof_status() {
        let r = QueryResult {
            rank: 1,
            stem: "test-v1".to_string(),
            path: "test.yaml".to_string(),
            relevance: 0.9,
            kind: ContractKind::default(),
            description: "test".to_string(),
            equations: vec!["eq1".to_string()],
            obligation_count: 3,
            references: vec!["Ref A".to_string()],
            depends_on: vec![],
            depended_by: vec![],
            score: Some(ScoreInfo {
                composite: 0.75,
                grade: "B".to_string(),
                spec_depth: 0.8,
                falsification: 0.7,
                kani: 0.6,
                lean: 0.0,
                binding: 0.9,
            }),
            proof_status: Some(ProofStatusInfo {
                level: "L4".to_string(),
                obligations: 3,
                falsification_tests: 3,
                kani_harnesses: 3,
                lean_proved: 0,
            }),
            bindings: vec![],
            diff: None,
            pagerank: None,
            call_sites: vec![types::CallSiteInfo {
                project: "trueno".to_string(),
                file: "src/kernel.rs".to_string(),
                line: 10,
                equation: None,
            }],
            violations: vec![],
            coverage_map: vec![],
        };
        let text = format!("{r}");
        assert!(text.contains("Score: 0.75 (Grade B)"));
        assert!(text.contains("Proof Level: L4"));
        assert!(text.contains("Papers: Ref A"));
        assert!(text.contains("trueno/src/kernel.rs:10"));
    }
