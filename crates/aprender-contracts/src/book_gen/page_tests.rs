use super::*;
use crate::graph::dependency_graph;
use crate::schema::parse_contract_str;

fn minimal_contract() -> Contract {
    parse_contract_str(
        r#"
metadata:
  version: "1.0.0"
  description: "Test kernel"
  references: ["Paper A"]
equations:
  f:
    formula: "f(x) = x"
    domain: "x ∈ ℝ^n"
proof_obligations:
  - type: invariant
    property: "output finite"
    formal: "|f(x)| < ∞"
falsification_tests:
  - id: FALSIFY-001
    rule: "finiteness"
    prediction: "output is always finite"
    if_fails: "overflow"
kani_harnesses:
  - id: KANI-001
    obligation: OBL-001
    bound: 16
"#,
    )
    .unwrap()
}

#[test]
fn generates_title_and_equations() {
    let c = minimal_contract();
    let refs = vec![("test-kernel-v1".to_string(), &c)];
    let graph = dependency_graph(&refs);
    let page = generate_contract_page(&c, "test-kernel-v1", &graph);

    assert!(page.contains("# test-kernel-v1"));
    assert!(page.contains("## Equations"));
    assert!(page.contains("$$"));
    assert!(page.contains("## Proof Obligations"));
    assert!(page.contains("## Falsification Tests"));
    assert!(page.contains("## Kani Harnesses"));
}

#[test]
fn latex_rendering_in_equations() {
    let c = parse_contract_str(
        r#"
metadata:
  version: "1.0.0"
  description: "Test"
equations:
  eq1:
    formula: "σ(x)_i = exp(x_i) / Σ_j exp(x_j)"
    domain: "x ∈ ℝ^n"
falsification_tests: []
"#,
    )
    .unwrap();
    let refs = vec![("test".to_string(), &c)];
    let graph = dependency_graph(&refs);
    let page = generate_contract_page(&c, "test", &graph);

    assert!(page.contains("\\sigma"));
    assert!(page.contains("\\exp"));
    assert!(page.contains("\\in"));
    assert!(page.contains("\\mathbb{R}"));
}

#[test]
fn dependency_graph_rendered() {
    let a = parse_contract_str(
        r#"
metadata:
  version: "1.0.0"
  description: "A"
  depends_on: ["b"]
equations:
  f:
    formula: "f(x) = x"
falsification_tests: []
"#,
    )
    .unwrap();
    let b = parse_contract_str(
        r#"
metadata:
  version: "1.0.0"
  description: "B"
equations:
  g:
    formula: "g(x) = x"
falsification_tests: []
"#,
    )
    .unwrap();
    let refs = vec![("a".to_string(), &a), ("b".to_string(), &b)];
    let graph = dependency_graph(&refs);
    let page = generate_contract_page(&a, "a", &graph);

    assert!(page.contains("```mermaid"));
    assert!(page.contains("## Dependencies"));
    assert!(page.contains("[b](b.md)"));
}

#[test]
fn optional_sections_omitted_when_empty() {
    let c = parse_contract_str(
        r#"
metadata:
  version: "1.0.0"
  description: "Minimal"
equations:
  f:
    formula: "f(x) = x"
falsification_tests: []
"#,
    )
    .unwrap();
    let refs = vec![("minimal".to_string(), &c)];
    let graph = dependency_graph(&refs);
    let page = generate_contract_page(&c, "minimal", &graph);

    assert!(!page.contains("## Kernel Phases"));
    assert!(!page.contains("## SIMD Dispatch"));
    assert!(!page.contains("## QA Gate"));
    assert!(!page.contains("## Dependency Graph"));
}

#[test]
fn pipe_escaped_in_tables() {
    let c = parse_contract_str(
        r#"
metadata:
  version: "1.0.0"
  description: "Test"
equations:
  f:
    formula: "f(x) = x"
proof_obligations:
  - type: invariant
    property: "output finite"
    formal: "|f(x)| < 1"
falsification_tests: []
"#,
    )
    .unwrap();
    let refs = vec![("test".to_string(), &c)];
    let graph = dependency_graph(&refs);
    let page = generate_contract_page(&c, "test", &graph);

    // The pipe in formal should be escaped
    assert!(page.contains("\\|"));
}

#[test]
fn codomain_and_invariants_rendered() {
    let c = parse_contract_str(
        r#"
metadata:
  version: "1.0.0"
  description: "Test"
equations:
  f:
    formula: "f(x) = x"
    domain: "x ∈ ℝ^n"
    codomain: "ℝ^n"
    invariants:
      - "f(x) >= 0"
      - "f(0) = 0"
falsification_tests: []
"#,
    )
    .unwrap();
    let refs = vec![("test".to_string(), &c)];
    let graph = dependency_graph(&refs);
    let page = generate_contract_page(&c, "test", &graph);

    assert!(page.contains("**Codomain:**"));
    assert!(page.contains("**Invariants:**"));
}

#[test]
fn kernel_phases_rendered() {
    let c = parse_contract_str(
        r#"
metadata:
  version: "1.0.0"
  description: "Test"
equations:
  f:
    formula: "f(x) = x"
kernel_structure:
  phases:
    - name: "load"
      description: "Load input"
      invariant: "data aligned"
    - name: "compute"
      description: "Run kernel"
falsification_tests: []
"#,
    )
    .unwrap();
    let refs = vec![("test".to_string(), &c)];
    let graph = dependency_graph(&refs);
    let page = generate_contract_page(&c, "test", &graph);

    assert!(page.contains("## Kernel Phases"));
    assert!(page.contains("**load**"));
    assert!(page.contains("**compute**"));
    assert!(page.contains("*data aligned*"));
}

#[test]
fn simd_dispatch_rendered() {
    let c = parse_contract_str(
        r#"
metadata:
  version: "1.0.0"
  description: "Test"
equations:
  f:
    formula: "f(x) = x"
simd_dispatch:
  softmax:
    AVX2: "softmax_avx2"
    Scalar: "softmax_scalar"
falsification_tests: []
"#,
    )
    .unwrap();
    let refs = vec![("test".to_string(), &c)];
    let graph = dependency_graph(&refs);
    let page = generate_contract_page(&c, "test", &graph);

    assert!(page.contains("## SIMD Dispatch"));
    assert!(page.contains("softmax"));
    assert!(page.contains("AVX2"));
}

#[test]
fn qa_gate_rendered() {
    let c = parse_contract_str(
        r#"
metadata:
  version: "1.0.0"
  description: "Test"
equations:
  f:
    formula: "f(x) = x"
qa_gate:
  id: QA-001
  name: "Quality gate"
  description: "Comprehensive testing"
  checks:
    - "unit tests"
    - "integration tests"
  pass_criteria: "all green"
falsification_tests: []
"#,
    )
    .unwrap();
    let refs = vec![("test".to_string(), &c)];
    let graph = dependency_graph(&refs);
    let page = generate_contract_page(&c, "test", &graph);

    assert!(page.contains("## QA Gate"));
    assert!(page.contains("Quality gate"));
    assert!(page.contains("Comprehensive testing"));
    assert!(page.contains("unit tests, integration tests"));
    assert!(page.contains("**Pass criteria:**"));
}

#[test]
fn dependent_graph_rendered() {
    // "b" depends on "a", so "a" page should show "b" as dependent
    let a = parse_contract_str(
        r#"
metadata:
  version: "1.0.0"
  description: "A"
equations:
  f:
    formula: "f(x) = x"
falsification_tests: []
"#,
    )
    .unwrap();
    let b = parse_contract_str(
        r#"
metadata:
  version: "1.0.0"
  description: "B"
  depends_on: ["a"]
equations:
  g:
    formula: "g(x) = x"
falsification_tests: []
"#,
    )
    .unwrap();
    let refs = vec![("a".to_string(), &a), ("b".to_string(), &b)];
    let graph = dependency_graph(&refs);
    let page = generate_contract_page(&a, "a", &graph);

    assert!(page.contains("```mermaid"));
    // b depends on a, so b should appear in a's dependency graph
    assert!(page.contains('b'));
}
