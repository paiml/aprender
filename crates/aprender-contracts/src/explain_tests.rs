#![allow(clippy::all)]
use super::*;
use crate::schema::parse_contract_str;

#[test]
fn explain_minimal_contract() {
    let contract = parse_contract_str(
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
    test: "proptest"
    if_fails: "overflow"
kani_harnesses:
  - id: KANI-001
    obligation: OBL-001
    bound: 16
"#,
    )
    .unwrap();

    let output = explain_contract(&contract, "test-kernel-v1", None);
    assert!(output.contains("test-kernel-v1 (v1.0.0)"));
    assert!(output.contains("Test kernel"));
    assert!(output.contains("Paper A"));
    assert!(output.contains("f(x) = x"));
    assert!(output.contains("[invariant] output finite"));
    assert!(output.contains("FALSIFY-001"));
    assert!(output.contains("KANI-001"));
    assert!(output.contains("Verification ladder"));
}

#[test]
fn explain_with_preconditions() {
    let contract = parse_contract_str(
        r#"
metadata:
  version: "1.0.0"
  description: "Pre/post test"
equations:
  eq:
    formula: "f(x) = x + 1"
    preconditions:
      - "x > 0"
    postconditions:
      - "ret > 1"
falsification_tests: []
"#,
    )
    .unwrap();

    let output = explain_contract(&contract, "prepost-v1", None);
    assert!(output.contains("Preconditions (caller must guarantee)"));
    assert!(output.contains("x > 0"));
    assert!(output.contains("Postconditions (kernel guarantees)"));
    assert!(output.contains("ret > 1"));
}

#[test]
fn explain_with_lean_proof() {
    let contract = parse_contract_str(
        r#"
metadata:
  version: "1.0.0"
  description: "Lean test"
equations:
  f:
    formula: "f(x) = x"
proof_obligations:
  - type: invariant
    property: "test"
    lean:
      theorem: test_theorem
      module: Test.Module
      status: proved
      depends_on: [dep1]
      notes: "Proved over reals"
falsification_tests: []
"#,
    )
    .unwrap();

    let output = explain_contract(&contract, "lean-v1", None);
    assert!(output.contains("Test.Module.test_theorem (proved)"));
    assert!(output.contains("Depends: dep1"));
    assert!(output.contains("Note: Proved over reals"));
}

#[test]
fn explain_markdown_has_headers_and_latex() {
    let contract = parse_contract_str(
        r#"
metadata:
  version: "1.0.0"
  description: "Markdown test"
  references: ["Paper A"]
equations:
  f:
    formula: "σ(x)_i = exp(x_i)"
    domain: "x ∈ ℝ^n"
proof_obligations:
  - type: invariant
    property: "output finite"
falsification_tests: []
"#,
    )
    .unwrap();

    let output = explain_contract_markdown(&contract, "test-v1", None);
    assert!(output.contains("# test-v1"));
    assert!(output.contains("## Equations"));
    assert!(output.contains("$$"));
    assert!(output.contains("\\sigma"));
    assert!(output.contains("## Proof Obligations"));
    assert!(output.contains("| # | Type |"));
}

#[test]
fn explain_json_is_valid() {
    let contract = parse_contract_str(
        r#"
metadata:
  version: "1.0.0"
  description: "JSON test"
  references: ["Paper"]
equations:
  f:
    formula: "f(x) = x"
proof_obligations:
  - type: precondition
    property: "input valid"
  - type: postcondition
    property: "output bounded"
    requires: "PRE-001"
falsification_tests: []
"#,
    )
    .unwrap();

    let output = explain_contract_json(&contract, "json-v1", None);
    let parsed: serde_json::Value = serde_json::from_str(&output).unwrap();
    assert_eq!(parsed["stem"], "json-v1");
    assert_eq!(parsed["version"], "1.0.0");
    assert_eq!(parsed["obligations"].as_array().unwrap().len(), 2);
    assert_eq!(parsed["obligations"][0]["type"], "precondition");
    assert_eq!(parsed["obligations"][1]["requires"], "PRE-001");
}

#[test]
fn explain_renders_dbc_fields() {
    let contract = parse_contract_str(
        r#"
metadata:
  version: "1.0.0"
  description: "DbC fields test"
  references: ["Meyer (1997)"]
  depends_on: ["parent-v1"]
equations:
  f:
    formula: "f(x) = x"
proof_obligations:
  - type: postcondition
    property: "output bounded"
    requires: "PRE-001"
  - type: loop_invariant
    property: "max tracks"
    applies_to_phase: "find_max"
  - type: subcontract
    property: "refines parent"
    parent_contract: "parent-v1"
falsification_tests: []
"#,
    )
    .unwrap();

    let output = explain_contract(&contract, "dbc-v1", None);
    assert!(output.contains("Requires: PRE-001"));
    assert!(output.contains("Phase: find_max"));
    assert!(output.contains("Refines: parent-v1"));
    assert!(output.contains("behavioral subtyping"));
}

#[test]
fn obligation_pattern_coverage() {
    // Verify all obligation types have a pattern
    let types = [
        ObligationType::Invariant,
        ObligationType::Equivalence,
        ObligationType::Bound,
        ObligationType::Monotonicity,
        ObligationType::Idempotency,
        ObligationType::Linearity,
        ObligationType::Symmetry,
        ObligationType::Associativity,
        ObligationType::Conservation,
        ObligationType::Ordering,
        ObligationType::Completeness,
        ObligationType::Soundness,
        ObligationType::Involution,
        ObligationType::Determinism,
        ObligationType::Roundtrip,
        ObligationType::StateMachine,
        ObligationType::Classification,
        ObligationType::Independence,
        ObligationType::Termination,
        ObligationType::Precondition,
        ObligationType::Postcondition,
        ObligationType::Frame,
        ObligationType::LoopInvariant,
        ObligationType::LoopVariant,
        ObligationType::OldState,
        ObligationType::Subcontract,
    ];
    for t in types {
        let pattern = obligation_pattern(t);
        assert!(!pattern.is_empty(), "empty pattern for {t}");
    }
}

// ---------- Additional coverage tests ----------

#[test]
fn strategy_explanation_all_variants() {
    assert!(strategy_explanation("exhaustive").contains("ALL inputs"));
    assert!(strategy_explanation("stub_float").contains("transcendentals"));
    assert!(strategy_explanation("compositional").contains("sub-kernels"));
    assert!(strategy_explanation("bounded_int").contains("integer-only"));
    assert!(strategy_explanation("unknown_strategy").contains("bounded model check"));
    assert!(strategy_explanation("").contains("bounded model check"));
}

#[test]
fn explain_with_no_references() {
    let contract = parse_contract_str(
        r#"
metadata:
  version: "2.0.0"
  description: "No refs"
  references: []
equations:
  f:
    formula: "f(x) = x"
falsification_tests: []
"#,
    )
    .unwrap();

    let output = explain_contract(&contract, "no-refs-v1", None);
    assert!(output.contains("no-refs-v1 (v2.0.0)"));
    assert!(output.contains("This contract specifies No refs."));
    // Should NOT contain "derives from"
    assert!(!output.contains("derives from"));
}

#[test]
fn explain_with_multiple_references() {
    let contract = parse_contract_str(
        r#"
metadata:
  version: "1.0.0"
  description: "Multi ref"
  references: ["Paper A", "Paper B", "Paper C"]
equations:
  g:
    formula: "g(x) = 2x"
falsification_tests: []
"#,
    )
    .unwrap();

    let output = explain_contract(&contract, "multi-ref-v1", None);
    assert!(output.contains("derives from"));
    assert!(output.contains("Paper A"));
    assert!(output.contains("and Paper B"));
    assert!(output.contains("and Paper C"));
}

#[test]
fn explain_with_depends_on() {
    let contract = parse_contract_str(
        r#"
metadata:
  version: "1.0.0"
  description: "Dependent"
  depends_on: ["silu-kernel-v1", "rmsnorm-v1"]
equations:
  h:
    formula: "h(x) = silu(x) + rms(x)"
falsification_tests: []
"#,
    )
    .unwrap();

    let output = explain_contract(&contract, "dep-v1", None);
    assert!(output.contains("Depends on:"));
    assert!(output.contains("silu-kernel-v1"));
    assert!(output.contains("rmsnorm-v1"));
}

#[test]
fn explain_equation_with_codomain() {
    let contract = parse_contract_str(
        r#"
metadata:
  version: "1.0.0"
  description: "Codomain test"
equations:
  f:
    formula: "f(x) = x^2"
    domain: "x ∈ ℝ"
    codomain: "[0, ∞)"
falsification_tests: []
"#,
    )
    .unwrap();

    let output = explain_contract(&contract, "cod-v1", None);
    assert!(output.contains("Domain: x ∈ ℝ"));
    assert!(output.contains("Range:  [0, ∞)"));
}

#[test]
fn explain_equation_with_invariants() {
    let contract = parse_contract_str(
        r#"
metadata:
  version: "1.0.0"
  description: "Invariants test"
equations:
  softmax:
    formula: "σ(x)_i = exp(x_i) / Σexp(x_j)"
    invariants:
      - "sum(output) = 1.0"
      - "output_i >= 0"
falsification_tests: []
"#,
    )
    .unwrap();

    let output = explain_contract(&contract, "inv-v1", None);
    assert!(output.contains("Invariants:"));
    assert!(output.contains("1. sum(output) = 1.0"));
    assert!(output.contains("2. output_i >= 0"));
}

#[test]
fn explain_equation_with_lean_theorem() {
    let contract = parse_contract_str(
        r#"
metadata:
  version: "1.0.0"
  description: "Lean theorem on equation"
equations:
  f:
    formula: "f(x) = x"
    lean_theorem: "Softmax.partition_of_unity"
falsification_tests: []
"#,
    )
    .unwrap();

    let output = explain_contract(&contract, "lean-eq-v1", None);
    assert!(output.contains("Lean theorem: Softmax.partition_of_unity"));
}

#[test]
fn explain_no_equations() {
    let contract = parse_contract_str(
        r#"
metadata:
  version: "1.0.0"
  description: "Registry contract"
  registry: true
falsification_tests: []
"#,
    )
    .unwrap();

    let output = explain_contract(&contract, "registry-v1", None);
    // Should still have header and description but no "Governing equations"
    assert!(output.contains("registry-v1 (v1.0.0)"));
    assert!(!output.contains("Governing equations"));
}

#[test]
fn explain_with_type_invariants() {
    let contract = parse_contract_str(
        r#"
metadata:
  version: "1.0.0"
  description: "Type invariants test"
type_invariants:
  - name: non_empty_dims
    type: ValidatedTensor
    predicate: "!self.dims.is_empty()"
    description: "Tensor dimensions must be non-empty"
  - name: positive_size
    type: ValidatedTensor
    predicate: "self.size > 0"
falsification_tests: []
"#,
    )
    .unwrap();

    let output = explain_contract(&contract, "type-inv-v1", None);
    assert!(output.contains("Type invariants (Meyer's class invariants)"));
    assert!(output.contains("every instance at every stable state"));
    assert!(output.contains("non_empty_dims [ValidatedTensor]"));
    assert!(output.contains("Tensor dimensions must be non-empty"));
    assert!(output.contains("Predicate: !self.dims.is_empty()"));
    assert!(output.contains("positive_size [ValidatedTensor]"));
    assert!(output.contains("Predicate: self.size > 0"));
}

#[test]
fn explain_type_invariant_without_description() {
    let contract = parse_contract_str(
        r#"
metadata:
  version: "1.0.0"
  description: "No desc inv"
type_invariants:
  - name: basic
    type: Foo
    predicate: "self.ok()"
falsification_tests: []
"#,
    )
    .unwrap();

    let output = explain_contract(&contract, "no-desc-inv-v1", None);
    assert!(output.contains("basic [Foo]"));
    // Should NOT have " — " since description is None
    assert!(!output.contains("basic [Foo] —"));
}

#[test]
fn explain_with_coq_spec() {
    let contract = parse_contract_str(
        r#"
metadata:
  version: "1.0.0"
  description: "Coq test"
coq_spec:
  module: SoftmaxSpec
  obligations:
    - links_to: "output finite"
      coq_lemma: softmax_finite
      status: proved
    - links_to: "sum is one"
      coq_lemma: softmax_sum_one
      status: stub
falsification_tests: []
"#,
    )
    .unwrap();

    let output = explain_contract(&contract, "coq-v1", None);
    assert!(output.contains("Coq verification (SoftmaxSpec)"));
    assert!(output.contains("output finite → softmax_finite [proved]"));
    assert!(output.contains("sum is one → softmax_sum_one [stub]"));
}

#[test]
fn explain_coq_spec_no_obligations() {
    let contract = parse_contract_str(
        r#"
metadata:
  version: "1.0.0"
  description: "Coq no obs"
coq_spec:
  module: EmptySpec
  obligations: []
falsification_tests: []
"#,
    )
    .unwrap();

    let output = explain_contract(&contract, "coq-empty-v1", None);
    assert!(output.contains("Coq verification (EmptySpec)"));
    assert!(output.contains("No obligation links defined"));
}

#[test]
fn explain_with_kernel_structure() {
    let contract = parse_contract_str(
        r#"
metadata:
  version: "1.0.0"
  description: "Kernel phases"
kernel_structure:
  phases:
    - name: find_max
      description: "Scan for maximum value"
      invariant: "max >= all seen elements"
    - name: subtract_and_exp
      description: "Compute exp(x - max)"
    - name: normalize
      description: "Divide by sum"
      invariant: "sum(output) = 1.0"
falsification_tests: []
"#,
    )
    .unwrap();

    let output = explain_contract(&contract, "phases-v1", None);
    assert!(output.contains("Kernel phases"));
    assert!(output.contains("1. find_max — Scan for maximum value [max >= all seen elements]"));
    assert!(output.contains("2. subtract_and_exp — Compute exp(x - max)"));
    assert!(output.contains("3. normalize — Divide by sum [sum(output) = 1.0]"));
}

#[test]
fn explain_with_simd_dispatch() {
    let contract = parse_contract_str(
        r#"
metadata:
  version: "1.0.0"
  description: "SIMD dispatch"
simd_dispatch:
  softmax:
    avx2: softmax_avx2
    neon: softmax_neon
    scalar: softmax_scalar
falsification_tests: []
"#,
    )
    .unwrap();

    let output = explain_contract(&contract, "simd-v1", None);
    assert!(output.contains("SIMD dispatch"));
    assert!(output.contains("softmax:"));
    assert!(output.contains("avx2"));
    assert!(output.contains("softmax_avx2"));
}

#[test]
fn explain_with_enforcement() {
    let contract = parse_contract_str(
        r#"
metadata:
  version: "1.0.0"
  description: "Enforcement test"
enforcement:
  no_panic:
    description: "Kernel must never panic"
    check: "cargo clippy"
    severity: "ERROR"
  perf_budget:
    description: "Must complete in 10ms"
falsification_tests: []
"#,
    )
    .unwrap();

    let output = explain_contract(&contract, "enforce-v1", None);
    assert!(output.contains("Enforcement"));
    assert!(output.contains("no_panic — Kernel must never panic (ERROR) → cargo clippy"));
    assert!(output.contains("perf_budget — Must complete in 10ms (ERROR) → -"));
}

#[test]
fn explain_with_qa_gate() {
    let contract = parse_contract_str(
        r#"
metadata:
  version: "1.0.0"
  description: "QA gate test"
qa_gate:
  id: QA-001
  name: "Softmax Gate"
  description: "All softmax properties verified"
  pass_criteria: "All tests pass, coverage > 90%"
  falsification: "Mutant killing rate > 80%"
falsification_tests: []
"#,
    )
    .unwrap();

    let output = explain_contract(&contract, "qa-v1", None);
    assert!(output.contains("Quality gate: QA-001 Softmax Gate"));
    assert!(output.contains("All softmax properties verified"));
    assert!(output.contains("Pass: All tests pass, coverage > 90%"));
    assert!(output.contains("Mutation: Mutant killing rate > 80%"));
}

#[test]
fn explain_qa_gate_minimal() {
    let contract = parse_contract_str(
        r#"
metadata:
  version: "1.0.0"
  description: "Minimal QA"
qa_gate:
  id: QA-002
  name: "Minimal"
falsification_tests: []
"#,
    )
    .unwrap();

    let output = explain_contract(&contract, "qa-min-v1", None);
    assert!(output.contains("Quality gate: QA-002 Minimal"));
    // Should not contain description/pass/mutation lines
    assert!(!output.contains("Pass:"));
    assert!(!output.contains("Mutation:"));
}

#[test]
fn explain_obligation_with_tolerance() {
    let contract = parse_contract_str(
        r#"
metadata:
  version: "1.0.0"
  description: "Tolerance"
proof_obligations:
  - type: equivalence
    property: "agrees with reference"
    formal: "|f(x) - g(x)| < ε"
    tolerance: 0.001
falsification_tests:
  - id: FALSIFY-001
    rule: "agrees with reference"
    prediction: "output matches"
    if_fails: "divergence"
"#,
    )
    .unwrap();

    let output = explain_contract(&contract, "tol-v1", None);
    assert!(output.contains("Tolerance: 1e-3"));
    assert!(output.contains("agrees with reference"));
}

#[test]
fn explain_obligation_cross_ref_to_falsification() {
    let contract = parse_contract_str(
        r#"
metadata:
  version: "1.0.0"
  description: "Cross-ref test"
proof_obligations:
  - type: invariant
    property: "finiteness"
falsification_tests:
  - id: FALSIFY-FIN-001
    rule: "finiteness"
    prediction: "always finite"
    if_fails: "overflow"
"#,
    )
    .unwrap();

    let output = explain_contract(&contract, "xref-v1", None);
    // The obligation "finiteness" should cross-reference FALSIFY-FIN-001
    assert!(output.contains("Verified at: L2 (FALSIFY-FIN-001)"));
}

#[test]
fn explain_obligation_cross_ref_to_kani() {
    let contract = parse_contract_str(
        r#"
metadata:
  version: "1.0.0"
  description: "Kani cross-ref"
proof_obligations:
  - type: bound
    property: "output bounded"
kani_harnesses:
  - id: KANI-BND-001
    obligation: OBL-001
    property: "output bounded"
    bound: 16
falsification_tests:
  - id: FALSIFY-001
    rule: "dummy"
    prediction: "ok"
    if_fails: "fail"
"#,
    )
    .unwrap();

    let output = explain_contract(&contract, "kani-xref-v1", None);
    assert!(output.contains("L4 (KANI-BND-001)"));
}

#[test]
fn explain_with_verification_summary() {
    let contract = parse_contract_str(
        r#"
metadata:
  version: "1.0.0"
  description: "Verification summary"
proof_obligations:
  - type: invariant
    property: "test"
verification_summary:
  total_obligations: 5
  l2_property_tested: 3
  l3_kani_proved: 2
  l4_lean_proved: 1
falsification_tests:
  - id: FALSIFY-001
    rule: "test"
    prediction: "ok"
    if_fails: "fail"
kani_harnesses:
  - id: KANI-001
    obligation: OBL-001
    bound: 8
"#,
    )
    .unwrap();

    let output = explain_contract(&contract, "vs-v1", None);
    // Verification ladder should show L5 lean proved
    assert!(output.contains("L5 (Lean):  1/1 proved (100%)"));
}

#[test]
fn explain_kani_harness_all_strategies() {
    let contract = parse_contract_str(
        r#"
metadata:
  version: "1.0.0"
  description: "Strategy test"
proof_obligations:
  - type: invariant
    property: "test"
kani_harnesses:
  - id: KANI-EXH-001
    obligation: OBL-001
    strategy: exhaustive
    bound: 8
  - id: KANI-SF-001
    obligation: OBL-002
    strategy: stub_float
    bound: 16
  - id: KANI-COMP-001
    obligation: OBL-003
    strategy: compositional
  - id: KANI-BI-001
    obligation: OBL-004
    strategy: bounded_int
    bound: 32
    harness: custom_harness_fn
    property: "integer only check"
falsification_tests:
  - id: FALSIFY-001
    rule: "test"
    prediction: "ok"
    if_fails: "fail"
"#,
    )
    .unwrap();

    let output = explain_contract(&contract, "strat-v1", None);
    assert!(output.contains("Kani bounded model checking"));
    assert!(output.contains("KANI-EXH-001"));
    assert!(output.contains("verify for ALL inputs"));
    assert!(output.contains("KANI-SF-001"));
    assert!(output.contains("transcendentals"));
    assert!(output.contains("KANI-COMP-001"));
    assert!(output.contains("sub-kernels"));
    assert!(output.contains("KANI-BI-001: custom_harness_fn"));
    assert!(output.contains("integer-only"));
    assert!(output.contains("Property: integer only check"));
}

#[test]
fn explain_kani_harness_no_bound_no_strategy() {
    let contract = parse_contract_str(
        r#"
metadata:
  version: "1.0.0"
  description: "Minimal kani"
kani_harnesses:
  - id: KANI-MIN-001
    obligation: OBL-001
falsification_tests: []
"#,
    )
    .unwrap();

    let output = explain_contract(&contract, "kani-min-v1", None);
    assert!(output.contains("KANI-MIN-001"));
    // bound should show as "-" for missing
    assert!(output.contains("bound: -"));
    assert!(output.contains("default"));
}

#[test]
fn explain_verification_ladder_strategies_summary() {
    let contract = parse_contract_str(
        r#"
metadata:
  version: "1.0.0"
  description: "Strats summary"
proof_obligations:
  - type: invariant
    property: "test"
kani_harnesses:
  - id: KANI-001
    obligation: OBL-001
    strategy: exhaustive
    bound: 8
  - id: KANI-002
    obligation: OBL-002
    strategy: exhaustive
    bound: 16
  - id: KANI-003
    obligation: OBL-003
    strategy: bounded_int
    bound: 32
falsification_tests:
  - id: FALSIFY-001
    rule: "test"
    prediction: "ok"
    if_fails: "fail"
"#,
    )
    .unwrap();

    let output = explain_contract(&contract, "strats-v1", None);
    assert!(output.contains("3 harnesses"));
    assert!(output.contains("2\u{d7} exhaustive"));
    assert!(output.contains("1\u{d7} bounded_int"));
}

#[test]
fn explain_markdown_no_equations() {
    let contract = parse_contract_str(
        r#"
metadata:
  version: "1.0.0"
  description: "No equations md"
  registry: true
falsification_tests: []
"#,
    )
    .unwrap();

    let output = explain_contract_markdown(&contract, "no-eq-md-v1", None);
    assert!(output.contains("# no-eq-md-v1"));
    assert!(!output.contains("## Equations"));
}

#[test]
fn explain_markdown_no_references() {
    let contract = parse_contract_str(
        r#"
metadata:
  version: "1.0.0"
  description: "No refs md"
falsification_tests: []
"#,
    )
    .unwrap();

    let output = explain_contract_markdown(&contract, "no-refs-md-v1", None);
    assert!(!output.contains("## References"));
}

#[test]
fn explain_markdown_no_obligations() {
    let contract = parse_contract_str(
        r#"
metadata:
  version: "1.0.0"
  description: "No obligations"
falsification_tests: []
"#,
    )
    .unwrap();

    let output = explain_contract_markdown(&contract, "no-obl-md-v1", None);
    assert!(!output.contains("## Proof Obligations"));
}

#[test]
fn explain_markdown_equation_with_invariants_and_conditions() {
    let contract = parse_contract_str(
        r#"
metadata:
  version: "1.0.0"
  description: "Full equation"
equations:
  f:
    formula: "f(x) = x"
    domain: "x ∈ ℝ"
    codomain: "ℝ"
    invariants:
      - "output finite"
    preconditions:
      - "x > 0"
    postconditions:
      - "ret > 0"
falsification_tests: []
"#,
    )
    .unwrap();

    let output = explain_contract_markdown(&contract, "full-eq-v1", None);
    assert!(output.contains("## Equations"));
    assert!(output.contains("### f"));
    assert!(output.contains("**Domain:**"));
    assert!(output.contains("**Codomain:**"));
    assert!(output.contains("**Invariants:**"));
    assert!(output.contains("**Preconditions:**"));
    assert!(output.contains("**Postconditions:**"));
}

#[test]
fn explain_markdown_with_verification_summary() {
    let contract = parse_contract_str(
        r#"
metadata:
  version: "1.0.0"
  description: "VS markdown"
verification_summary:
  total_obligations: 10
  l4_lean_proved: 7
falsification_tests: []
kani_harnesses:
  - id: KANI-001
    obligation: OBL-001
"#,
    )
    .unwrap();

    let output = explain_contract_markdown(&contract, "vs-md-v1", None);
    assert!(output.contains("## Verification"));
    assert!(output.contains("Lean: 7/10 proved"));
    assert!(output.contains("Kani: 1 harnesses"));
    assert!(output.contains("Tests: 0 falsification"));
}

#[test]
fn explain_markdown_no_verification_summary() {
    let contract = parse_contract_str(
        r#"
metadata:
  version: "1.0.0"
  description: "No VS"
falsification_tests: []
"#,
    )
    .unwrap();

    let output = explain_contract_markdown(&contract, "no-vs-v1", None);
    assert!(output.contains("## Verification"));
    // Should still show kani/test counts
    assert!(output.contains("Kani: 0 harnesses"));
    assert!(output.contains("Tests: 0 falsification"));
    // Should NOT have lean line
    assert!(!output.contains("Lean:"));
}

#[test]
fn explain_markdown_with_binding() {
    let contract = parse_contract_str(
        r#"
metadata:
  version: "1.0.0"
  description: "Binding md"
equations:
  softmax:
    formula: "σ(x) = exp(x) / sum"
  rmsnorm:
    formula: "rms(x) = x / sqrt(mean(x^2))"
falsification_tests: []
"#,
    )
    .unwrap();

    let registry = BindingRegistry {
        version: "1.0.0".to_string(),
        target_crate: "aprender".to_string(),
        critical_path: vec![],
        bindings: vec![
            crate::binding::KernelBinding {
                contract: "bind-md-v1.yaml".to_string(),
                equation: "softmax".to_string(),
                module_path: None,
                function: Some("aprender::nn::softmax".to_string()),
                signature: None,
                status: crate::binding::ImplStatus::Implemented,
                notes: None,
            },
            crate::binding::KernelBinding {
                contract: "bind-md-v1.yaml".to_string(),
                equation: "rmsnorm".to_string(),
                module_path: None,
                function: Some("aprender::nn::rmsnorm".to_string()),
                signature: None,
                status: crate::binding::ImplStatus::Partial,
                notes: None,
            },
        ],
    };

    let output = explain_contract_markdown(&contract, "bind-md-v1", Some(&registry));
    assert!(output.contains("## Bindings (aprender)"));
    assert!(output.contains("| softmax | implemented |"));
    assert!(output.contains("| rmsnorm | partial |"));
}

#[test]
fn explain_markdown_binding_no_match() {
    let contract = parse_contract_str(
        r#"
metadata:
  version: "1.0.0"
  description: "No match binding"
equations:
  f:
    formula: "f(x) = x"
falsification_tests: []
"#,
    )
    .unwrap();

    let registry = BindingRegistry {
        version: "1.0.0".to_string(),
        target_crate: "other".to_string(),
        critical_path: vec![],
        bindings: vec![crate::binding::KernelBinding {
            contract: "different-contract.yaml".to_string(),
            equation: "f".to_string(),
            module_path: None,
            function: None,
            signature: None,
            status: crate::binding::ImplStatus::Implemented,
            notes: None,
        }],
    };

    let output = explain_contract_markdown(&contract, "no-match-v1", Some(&registry));
    // No matching bindings for this contract -> no Bindings section
    assert!(!output.contains("## Bindings"));
}

#[test]
fn explain_json_with_all_obligation_fields() {
    let contract = parse_contract_str(
        r#"
metadata:
  version: "1.0.0"
  description: "Full JSON"
  references: ["Paper A"]
  depends_on: ["parent-v1"]
equations:
  f:
    formula: "f(x) = x"
proof_obligations:
  - type: invariant
    property: "finiteness"
    formal: "|f(x)| < ∞"
    tolerance: 0.0001
    lean:
      theorem: finite_proof
      status: proved
  - type: postcondition
    property: "output bounded"
    requires: "PRE-001"
    applies_to_phase: "compute"
  - type: subcontract
    property: "refines parent"
    parent_contract: "parent-v1"
falsification_tests: []
"#,
    )
    .unwrap();

    let output = explain_contract_json(&contract, "full-json-v1", None);
    let parsed: serde_json::Value = serde_json::from_str(&output).unwrap();

    assert_eq!(parsed["stem"], "full-json-v1");
    assert_eq!(parsed["version"], "1.0.0");
    assert_eq!(parsed["references"][0], "Paper A");
    assert_eq!(parsed["depends_on"][0], "parent-v1");

    let obs = parsed["obligations"].as_array().unwrap();
    assert_eq!(obs.len(), 3);

    // First obligation: invariant with formal, tolerance, lean
    assert_eq!(obs[0]["type"], "invariant");
    assert_eq!(obs[0]["formal"], "|f(x)| < ∞");
    assert_eq!(obs[0]["tolerance"], 0.0001);
    assert_eq!(obs[0]["lean"]["theorem"], "finite_proof");
    assert_eq!(obs[0]["lean"]["status"], "proved");

    // Second: postcondition with requires and applies_to_phase
    assert_eq!(obs[1]["type"], "postcondition");
    assert_eq!(obs[1]["requires"], "PRE-001");
    assert_eq!(obs[1]["applies_to_phase"], "compute");

    // Third: subcontract with parent_contract
    assert_eq!(obs[2]["type"], "subcontract");
    assert_eq!(obs[2]["parent_contract"], "parent-v1");
}

#[test]
fn explain_json_with_binding() {
    let contract = parse_contract_str(
        r#"
metadata:
  version: "1.0.0"
  description: "JSON binding"
equations:
  softmax:
    formula: "σ(x)"
  rmsnorm:
    formula: "rms(x)"
falsification_tests: []
"#,
    )
    .unwrap();

    let registry = BindingRegistry {
        version: "1.0.0".to_string(),
        target_crate: "aprender".to_string(),
        critical_path: vec![],
        bindings: vec![crate::binding::KernelBinding {
            contract: "json-bind-v1.yaml".to_string(),
            equation: "softmax".to_string(),
            module_path: None,
            function: None,
            signature: None,
            status: crate::binding::ImplStatus::Implemented,
            notes: None,
        }],
    };

    let output = explain_contract_json(&contract, "json-bind-v1", Some(&registry));
    let parsed: serde_json::Value = serde_json::from_str(&output).unwrap();

    let binding = &parsed["binding"];
    assert_eq!(binding["target_crate"], "aprender");
    assert_eq!(binding["equations"]["softmax"], "implemented");
    assert_eq!(binding["equations"]["rmsnorm"], "missing");
}

#[test]
fn explain_json_no_binding() {
    let contract = parse_contract_str(
        r#"
metadata:
  version: "1.0.0"
  description: "No binding JSON"
falsification_tests: []
"#,
    )
    .unwrap();

    let output = explain_contract_json(&contract, "no-bind-json-v1", None);
    let parsed: serde_json::Value = serde_json::from_str(&output).unwrap();

    assert!(parsed["binding"].is_null());
    assert_eq!(parsed["falsification_tests"], 0);
    assert_eq!(parsed["kani_harnesses"], 0);
}

#[test]
fn explain_obligation_pattern_specific_values() {
    // Test specific patterns contain expected math symbols
    assert!(obligation_pattern(ObligationType::Invariant).contains("∀x"));
    assert!(obligation_pattern(ObligationType::Equivalence).contains("ε"));
    assert!(obligation_pattern(ObligationType::Bound).contains("≤"));
    assert!(obligation_pattern(ObligationType::Monotonicity).contains("order preserved"));
    assert!(obligation_pattern(ObligationType::Idempotency).contains("f(f(x)) = f(x)"));
    assert!(obligation_pattern(ObligationType::Linearity).contains("homogeneous"));
    assert!(obligation_pattern(ObligationType::Symmetry).contains("permutation"));
    assert!(obligation_pattern(ObligationType::Associativity).contains("grouping"));
    assert!(obligation_pattern(ObligationType::Conservation).contains("conserved"));
    assert!(obligation_pattern(ObligationType::Ordering).contains("order relation"));
    assert!(obligation_pattern(ObligationType::Completeness).contains("completeness"));
    assert!(obligation_pattern(ObligationType::Soundness).contains("soundness"));
    assert!(obligation_pattern(ObligationType::Involution).contains("self-inverse"));
    assert!(obligation_pattern(ObligationType::Determinism).contains("deterministic"));
    assert!(obligation_pattern(ObligationType::Roundtrip).contains("roundtrip"));
    assert!(obligation_pattern(ObligationType::StateMachine).contains("state transitions"));
    assert!(obligation_pattern(ObligationType::Classification).contains("class set"));
    assert!(obligation_pattern(ObligationType::Independence).contains("independence"));
    assert!(obligation_pattern(ObligationType::Termination).contains("terminates"));
    assert!(obligation_pattern(ObligationType::Precondition).contains("caller must guarantee"));
    assert!(obligation_pattern(ObligationType::Postcondition).contains("kernel guarantees"));
    assert!(obligation_pattern(ObligationType::Frame).contains("modifies"));
    assert!(obligation_pattern(ObligationType::LoopInvariant).contains("maintained"));
    assert!(obligation_pattern(ObligationType::LoopVariant).contains("termination witness"));
    assert!(obligation_pattern(ObligationType::OldState).contains("pre to post"));
    assert!(obligation_pattern(ObligationType::Subcontract).contains("behavioral subtyping"));
}

#[test]
fn explain_binding_status_in_plain_text() {
    let contract = parse_contract_str(
        r#"
metadata:
  version: "1.0.0"
  description: "Binding status"
equations:
  f:
    formula: "f(x) = x"
  g:
    formula: "g(x) = 2x"
falsification_tests: []
"#,
    )
    .unwrap();

    let registry = BindingRegistry {
        version: "1.0.0".to_string(),
        target_crate: "testcrate".to_string(),
        critical_path: vec![],
        bindings: vec![
            crate::binding::KernelBinding {
                contract: "bind-status-v1.yaml".to_string(),
                equation: "f".to_string(),
                module_path: None,
                function: None,
                signature: None,
                status: crate::binding::ImplStatus::Implemented,
                notes: None,
            },
            crate::binding::KernelBinding {
                contract: "bind-status-v1.yaml".to_string(),
                equation: "g".to_string(),
                module_path: None,
                function: None,
                signature: None,
                status: crate::binding::ImplStatus::NotImplemented,
                notes: None,
            },
        ],
    };

    let output = explain_contract(&contract, "bind-status-v1", Some(&registry));
    assert!(output.contains("Binding status (testcrate)"));
    assert!(output.contains("f: implemented"));
    assert!(output.contains("g: not_implemented"));
}

#[test]
fn explain_binding_status_missing_equation() {
    let contract = parse_contract_str(
        r#"
metadata:
  version: "1.0.0"
  description: "Missing binding"
equations:
  f:
    formula: "f(x) = x"
falsification_tests: []
"#,
    )
    .unwrap();

    let registry = BindingRegistry {
        version: "1.0.0".to_string(),
        target_crate: "testcrate".to_string(),
        critical_path: vec![],
        bindings: vec![], // No bindings at all
    };

    let output = explain_contract(&contract, "missing-bind-v1", Some(&registry));
    assert!(output.contains("f: missing"));
}

#[test]
fn explain_no_binding_provided() {
    let contract = parse_contract_str(
        r#"
metadata:
  version: "1.0.0"
  description: "No binding"
equations:
  f:
    formula: "f(x) = x"
falsification_tests: []
"#,
    )
    .unwrap();

    let output = explain_contract(&contract, "no-bind-v1", None);
    assert!(!output.contains("Binding status"));
}

#[test]
fn explain_lean_proved_shows_l5_in_cross_ref() {
    let contract = parse_contract_str(
        r#"
metadata:
  version: "1.0.0"
  description: "L5 cross-ref"
proof_obligations:
  - type: invariant
    property: "finiteness"
    lean:
      theorem: finite_thm
      status: proved
falsification_tests:
  - id: FALSIFY-001
    rule: "finiteness"
    prediction: "output finite"
    if_fails: "overflow"
"#,
    )
    .unwrap();

    let output = explain_contract(&contract, "l5-v1", None);
    assert!(output.contains("L5 (Lean)"));
    assert!(output.contains("L2 (FALSIFY-001)"));
}

#[test]
fn explain_enforcement_default_severity() {
    let contract = parse_contract_str(
        r#"
metadata:
  version: "1.0.0"
  description: "Default severity"
enforcement:
  rule1:
    description: "A rule"
falsification_tests: []
"#,
    )
    .unwrap();

    let output = explain_contract(&contract, "def-sev-v1", None);
    // Default severity is "ERROR", default check is "-"
    assert!(output.contains("rule1 — A rule (ERROR) → -"));
}
