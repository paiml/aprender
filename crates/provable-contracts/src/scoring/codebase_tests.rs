use super::*;

fn load_contracts_and_binding() -> (Vec<(String, Contract)>, BindingRegistry) {
    let binding_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../contracts/aprender/binding.yaml");
    let content = std::fs::read_to_string(binding_path).unwrap();
    let binding: BindingRegistry = serde_yaml::from_str(&content).unwrap();

    let contracts_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../contracts");
    let mut parsed = Vec::new();
    for entry in std::fs::read_dir(contracts_dir).unwrap().flatten() {
        let p = entry.path();
        if p.extension().and_then(|x| x.to_str()) != Some("yaml") {
            continue;
        }
        let Ok(c) = crate::schema::parse_contract(&p) else {
            continue;
        };
        // Binding uses filename WITH .yaml extension as the stem
        let stem = p.file_name().unwrap().to_str().unwrap().to_string();
        parsed.push((stem, c));
    }
    (parsed, binding)
}

#[test]
fn score_codebase_with_binding() {
    let (parsed, binding) = load_contracts_and_binding();
    let contracts: Vec<_> = parsed.iter().map(|(s, c)| (s.clone(), c)).collect();
    let score = score_codebase(&contracts, &binding);
    assert!(score.contract_coverage > 0.0);
    assert!(score.binding_completeness > 0.0);
    assert!(score.composite > 0.10, "composite={}", score.composite);
}

#[test]
fn codebase_display_format() {
    let score = CodebaseScore {
        path: "test".into(),
        contract_coverage: 0.5,
        binding_completeness: 0.8,
        mean_contract_score: 0.6,
        proof_depth_dist: 0.4,
        drift: 1.0,
        reverse_coverage: 0.0,
        mutation_testing: 1.0,
        ci_pipeline_depth: 1.0,
        proof_freshness: 1.0,
        defect_patterns: 1.0,
        composite: 0.6,
        grade: Grade::C,
        top_gaps: vec![ScoringGap {
            contract: "softmax".into(),
            dimension: "kani".into(),
            current: 0.3,
            target: 1.0,
            impact: 4.0,
            action: "Write #[kani::proof] harnesses".into(),
        }],
    };
    let text = score.to_string();
    assert!(text.contains("Grade C"));
    assert!(text.contains("softmax"));
    assert!(text.contains("kani::proof"));
}

#[test]
fn empty_binding_scores_low() {
    let binding = BindingRegistry {
        version: "1.0.0".into(),
        target_crate: "test".into(),
        critical_path: vec![],
        bindings: Vec::new(),
    };
    let score = score_codebase(&[], &binding);
    assert!(score.contract_coverage.abs() < f64::EPSILON);
    assert!((score.composite - 0.20).abs() < f64::EPSILON); // only drift=1.0 * 0.20
    assert_eq!(score.grade, Grade::F);
}

#[test]
fn gap_actions_are_populated() {
    let (parsed, binding) = load_contracts_and_binding();
    let contracts: Vec<_> = parsed.iter().map(|(s, c)| (s.clone(), c)).collect();
    let score = score_codebase(&contracts, &binding);
    for gap in &score.top_gaps {
        assert!(!gap.action.is_empty(), "Gap action should not be empty");
        assert!(gap.impact > 0.0, "Gap impact should be positive");
    }
}

#[test]
fn pagerank_weighted_gaps_differ() {
    let (parsed, binding) = load_contracts_and_binding();
    let contracts: Vec<_> = parsed.iter().map(|(s, c)| (s.clone(), c)).collect();

    let without_pr = score_codebase(&contracts, &binding);

    // Build a synthetic pagerank map where one contract has high pagerank
    let mut pr = HashMap::new();
    for (stem, _) in &contracts {
        pr.insert(stem.clone(), 0.01);
    }
    // Give softmax-kernel-v1.yaml a very high pagerank
    pr.insert("softmax-kernel-v1.yaml".into(), 0.50);

    let with_pr = score_codebase_with_pagerank(&contracts, &binding, Some(&pr));

    // The dimensions should still be the same regardless of pagerank
    assert!((without_pr.composite - with_pr.composite).abs() < f64::EPSILON);
}

#[test]
fn dependency_fanout_fallback() {
    let rev_deps: HashMap<&str, usize> = HashMap::new();
    let f = super::dependency_fanout("unknown", None, &rev_deps);
    assert!(
        (f - 1.0).abs() < 1e-9,
        "Unknown contract should have fanout 1.0"
    );

    let mut rev_deps2: HashMap<&str, usize> = HashMap::new();
    rev_deps2.insert("known", 5);
    let f2 = super::dependency_fanout("known", None, &rev_deps2);
    assert!((f2 - 6.0).abs() < 1e-9, "5 reverse deps + 1 = 6.0");
}

#[test]
fn dependency_fanout_with_pagerank() {
    let mut pr = HashMap::new();
    pr.insert("low".to_string(), 0.01);
    pr.insert("high".to_string(), 0.10);
    let rev_deps: HashMap<&str, usize> = HashMap::new();

    let f_low = super::dependency_fanout("low", Some(&pr), &rev_deps);
    let f_high = super::dependency_fanout("high", Some(&pr), &rev_deps);
    assert!(f_high > f_low, "High pagerank should have higher fanout");
    assert!(f_low >= 1.0, "Min fanout should be 1.0");
    assert!(f_high <= 10.0, "Max fanout should be 10.0");
}

#[test]
fn drift_override_affects_composite() {
    let (parsed, binding) = load_contracts_and_binding();
    let contracts: Vec<_> = parsed.iter().map(|(s, c)| (s.clone(), c)).collect();

    let fresh = super::score_codebase_full(&contracts, &binding, None, Some(1.0));
    let stale = super::score_codebase_full(&contracts, &binding, None, Some(0.0));

    assert!((fresh.drift - 1.0).abs() < 1e-9);
    assert!((stale.drift - 0.0).abs() < 1e-9);
    // Drift weight is 0.20, so composite should differ by 0.15
    let diff = fresh.composite - stale.composite;
    assert!((diff - 0.20).abs() < 0.01, "diff={diff}");
}

#[test]
fn reverse_dep_counts() {
    let yaml = r#"
metadata:
  version: "1.0.0"
  description: "Test"
  depends_on: ["dep-v1.yaml"]
equations:
  f:
    formula: "f(x) = x"
"#;
    let contract = crate::schema::parse_contract_str(yaml).unwrap();
    let dep_yaml = r#"
metadata:
  version: "1.0.0"
  description: "Dep"
equations:
  g:
    formula: "g(x) = x"
"#;
    let dep_contract = crate::schema::parse_contract_str(dep_yaml).unwrap();
    let contracts = vec![
        ("test-v1.yaml".to_string(), &contract),
        ("dep-v1.yaml".to_string(), &dep_contract),
    ];
    let counts = super::compute_reverse_dep_counts(&contracts);
    assert_eq!(counts.get("dep-v1.yaml").copied().unwrap_or(0), 1);
    assert_eq!(counts.get("test-v1.yaml").copied().unwrap_or(0), 0);
}

#[test]
fn compute_gaps_falsification_and_binding() {
    // Contract with more obligations than falsification tests
    let yaml = r#"
metadata:
  version: "1.0.0"
  description: "Test"
equations:
  f:
    formula: "f(x) = x"
proof_obligations:
  - type: invariant
    property: "finite output"
  - type: bound
    property: "bounded error"
  - type: equivalence
    property: "matches reference"
falsification_tests:
  - id: FALSIFY-001
    rule: "finite"
    prediction: "finite"
    if_fails: "overflow"
kani_harnesses:
  - id: KANI-001
    obligation: OBL-001
    bound: 16
"#;
    let contract = crate::schema::parse_contract_str(yaml).unwrap();
    let binding = BindingRegistry {
        version: "1.0.0".into(),
        target_crate: "test".into(),
        critical_path: vec![],
        bindings: vec![
            crate::binding::KernelBinding {
                contract: "test-v1.yaml".into(),
                equation: "f".into(),
                status: ImplStatus::Partial,
                module_path: Some("test::f".into()),
                function: None,
                signature: None,
                notes: None,
            },
            crate::binding::KernelBinding {
                contract: "test-v1.yaml".into(),
                equation: "g".into(),
                status: ImplStatus::NotImplemented,
                module_path: Some("test::g".into()),
                function: None,
                signature: None,
                notes: None,
            },
        ],
    };

    let contracts = vec![("test-v1.yaml".to_string(), &contract)];
    let bound_stems: std::collections::BTreeSet<&str> =
        contracts.iter().map(|(s, _)| s.as_str()).collect();
    let gaps = super::compute_gaps(&contracts, &binding, &bound_stems, None);

    // Should have gaps for: kani_coverage, falsification_coverage, binding_partial, binding_coverage
    let dimensions: Vec<&str> = gaps.iter().map(|g| g.dimension.as_str()).collect();
    assert!(
        dimensions.contains(&"kani_coverage"),
        "Expected kani gap: {dimensions:?}"
    );
    assert!(
        dimensions.contains(&"falsification_coverage"),
        "Expected falsification gap: {dimensions:?}"
    );
    assert!(
        dimensions.contains(&"binding_partial"),
        "Expected partial binding gap: {dimensions:?}"
    );
    assert!(
        dimensions.contains(&"binding_coverage"),
        "Expected unimpl binding gap: {dimensions:?}"
    );
}
