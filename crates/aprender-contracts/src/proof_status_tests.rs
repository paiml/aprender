use crate::obligation_matrix::{
    format_obligation_table, obligation_matrix, property_words_match, truncate, L2Status,
};
use crate::proof_status::*;
use crate::schema::{parse_contract_str, Contract};

use crate::proof_status::count_bindings;

/// Build a minimal contract with configurable obligation, test, and harness counts
fn minimal_contract(n_ob: usize, n_ft: usize, n_kani: usize) -> Contract {
    let mut yaml = String::from(
        r#"
metadata:
  version: "1.0.0"
  description: "Test"
  references: ["Paper"]
equations:
  f:
    formula: "f(x) = x"
proof_obligations:
"#,
    );
    for i in 0..n_ob {
        yaml.push_str(&format!(
            "  - type: invariant\n    property: \"prop {i}\"\n"
        ));
    }
    yaml.push_str("falsification_tests:\n");
    for i in 0..n_ft {
        yaml.push_str(&format!(
            "  - id: FT-{i:03}\n    rule: \"r\"\n    prediction: \"p\"\n    if_fails: \"f\"\n"
        ));
    }
    yaml.push_str("kani_harnesses:\n");
    for i in 0..n_kani {
        yaml.push_str(&format!(
            "  - id: KH-{i:03}\n    obligation: OBL-{i:03}\n    bound: 16\n"
        ));
    }
    parse_contract_str(&yaml).unwrap()
}

/// Build a contract with Lean verification summary for L4/L5 level testing
fn contract_with_lean(total: u32, lean_proved: u32) -> Contract {
    let mut c = minimal_contract(total as usize, total as usize, total as usize);
    c.verification_summary = Some(crate::schema::VerificationSummary {
        total_obligations: total,
        l2_property_tested: total,
        l3_kani_proved: total,
        l4_lean_proved: lean_proved,
        l4_sorry_count: total - lean_proved,
        l4_not_applicable: 0,
    });
    c
}

#[test]
fn proof_level_display() {
    assert_eq!(ProofLevel::L1.to_string(), "L1");
    assert_eq!(ProofLevel::L2.to_string(), "L2");
    assert_eq!(ProofLevel::L3.to_string(), "L3");
    assert_eq!(ProofLevel::L4.to_string(), "L4");
    assert_eq!(ProofLevel::L5.to_string(), "L5");
}

#[test]
fn proof_level_ordering() {
    assert!(ProofLevel::L1 < ProofLevel::L2);
    assert!(ProofLevel::L2 < ProofLevel::L3);
    assert!(ProofLevel::L3 < ProofLevel::L4);
    assert!(ProofLevel::L4 < ProofLevel::L5);
}

#[test]
fn level_l1_for_equations_only() {
    let c = minimal_contract(0, 0, 0);
    assert_eq!(compute_proof_level(&c, None), ProofLevel::L1);
}

#[test]
fn level_l2_for_falsification_covered() {
    let c = minimal_contract(3, 3, 0);
    assert_eq!(compute_proof_level(&c, None), ProofLevel::L2);
}

#[test]
fn level_l2_not_enough_tests() {
    let c = minimal_contract(3, 2, 0);
    assert_eq!(compute_proof_level(&c, None), ProofLevel::L1);
}

#[test]
fn level_l3_kani_plus_falsification() {
    let c = minimal_contract(3, 3, 2);
    assert_eq!(compute_proof_level(&c, None), ProofLevel::L3);
}

#[test]
fn level_l3_kani_without_enough_tests() {
    // Has kani but not enough falsification tests
    let c = minimal_contract(3, 2, 2);
    assert_eq!(compute_proof_level(&c, None), ProofLevel::L1);
}

#[test]
fn level_l4_all_lean_proved() {
    let c = contract_with_lean(3, 3);
    // ONT-2a: grounded — the tree has the theorems the equations name.
    assert_eq!(
        compute_proof_level_with_grounding(&c, None, 3),
        ProofLevel::L4
    );
    // …and the same contract with nothing grounding it is the andon's case.
    assert_eq!(
        compute_proof_level_with_grounding(&c, None, 0),
        ProofLevel::L3
    );
}

#[test]
fn level_l4_partial_lean_stays_l3() {
    let c = contract_with_lean(3, 2);
    assert_eq!(
        compute_proof_level_with_grounding(&c, None, 2),
        ProofLevel::L3
    );
}

#[test]
fn level_l5_lean_plus_all_bound() {
    let c = contract_with_lean(3, 3);
    assert_eq!(
        compute_proof_level_with_grounding(&c, Some((1, 1)), 3),
        ProofLevel::L5
    );
}

#[test]
fn level_l4_when_bindings_incomplete() {
    let c = contract_with_lean(3, 3);
    assert_eq!(
        compute_proof_level_with_grounding(&c, Some((0, 1)), 3),
        ProofLevel::L4
    );
}

/// Build a contract with an explicit not-applicable count in its summary.
fn contract_with_lean_na(total: u32, lean_proved: u32, not_applicable: u32) -> Contract {
    let mut c = minimal_contract(total as usize, total as usize, total as usize);
    c.verification_summary = Some(crate::schema::VerificationSummary {
        total_obligations: total,
        l2_property_tested: total,
        l3_kani_proved: total,
        l4_lean_proved: lean_proved,
        l4_sorry_count: total - lean_proved - not_applicable,
        l4_not_applicable: not_applicable,
    });
    c
}

/// STRICT: proved + N/A must cover EVERY obligation. Partial Lean coverage
/// (the old "≥1 resolving ref → L4" over-promotion) now reports L3, even when
/// fully bound (which would previously have inflated it to L5). This is the
/// `lora-algebra`/`attention-kernel` case (4-of-6, 0-of-5, …).
#[test]
fn level_strict_partial_coverage_stays_l3_even_when_bound() {
    let c = contract_with_lean_na(6, 4, 1); // 4 grounded + 1 N/A = 5 < 6
    assert_eq!(
        compute_proof_level_with_grounding(&c, None, 4),
        ProofLevel::L3
    );
    assert_eq!(
        compute_proof_level_with_grounding(&c, Some((3, 3)), 4),
        ProofLevel::L3
    );
}

/// STRICT: a contract whose provable obligations are ALL discharged — some
/// proved, the rest explicitly N/A — is a legitimate L4/L5 (the `softmax-kernel`
/// case: 5 proved + 4 N/A of 9).
#[test]
fn level_strict_full_coverage_with_na_is_l4() {
    let c = contract_with_lean_na(9, 5, 4); // 5 grounded + 4 N/A == 9
    assert_eq!(
        compute_proof_level_with_grounding(&c, None, 5),
        ProofLevel::L4
    );
    assert_eq!(
        compute_proof_level_with_grounding(&c, Some((1, 1)), 5),
        ProofLevel::L5
    );
    // ONT-2a: the SAME coverage, claimed and not grounded, is not L4.
    assert_eq!(
        compute_proof_level_with_grounding(&c, Some((1, 1)), 0),
        ProofLevel::L3
    );
}

/// STRICT: `total` is `proof_obligations.len()`, NOT `verification_summary.
/// total_obligations`, so a summary cannot manufacture L4 by understating the
/// obligation count. Here the contract has 6 real obligations but the summary
/// claims a total of 3 (all "proved") — strict measures 3 proved against 6 real
/// obligations and correctly withholds L4.
#[test]
fn level_strict_summary_cannot_understate_total() {
    let mut c = minimal_contract(6, 6, 6); // 6 real proof_obligations
    c.verification_summary = Some(crate::schema::VerificationSummary {
        total_obligations: 3, // understated
        l2_property_tested: 3,
        l3_kani_proved: 3,
        l4_lean_proved: 3,
        l4_sorry_count: 0,
        l4_not_applicable: 0,
    });
    assert_eq!(
        compute_proof_level_with_grounding(&c, None, 3),
        ProofLevel::L3
    );
}

#[test]
fn report_empty_contracts() {
    let report = proof_status_report(&[], None, false);
    assert_eq!(report.totals.contracts, 0);
    assert_eq!(report.totals.obligations, 0);
    assert!(report.contracts.is_empty());
}

#[test]
fn report_single_contract() {
    let c = minimal_contract(3, 3, 2);
    let report = proof_status_report(&[("test-v1".to_string(), &c)], None, false);
    assert_eq!(report.contracts.len(), 1);
    assert_eq!(report.contracts[0].stem, "test-v1");
    assert_eq!(report.contracts[0].proof_level, ProofLevel::L3);
    assert_eq!(report.totals.obligations, 3);
    assert_eq!(report.totals.falsification_tests, 3);
    assert_eq!(report.totals.kani_harnesses, 2);
}

#[test]
fn report_with_binding() {
    let c = minimal_contract(3, 3, 2);
    let binding = crate::binding::parse_binding_str(
        r#"
version: "1.0.0"
target_crate: test
bindings:
  - contract: test-v1.yaml
    equation: f
    module_path: "test::f"
    function: f
    status: implemented
"#,
    )
    .unwrap();
    let report = proof_status_report(&[("test-v1".to_string(), &c)], Some(&binding), false);
    assert_eq!(report.contracts[0].bindings_implemented, 1);
    assert_eq!(report.contracts[0].bindings_total, 1);
}

#[test]
fn report_with_kernel_classes() {
    let c = minimal_contract(3, 3, 2);
    let report = proof_status_report(&[("softmax-kernel-v1".to_string(), &c)], None, true);
    assert!(!report.kernel_classes.is_empty());
    // Softmax is in classes A, B, C, D, E
    let class_a = report
        .kernel_classes
        .iter()
        .find(|kc| kc.label == "A")
        .unwrap();
    assert!(class_a
        .contract_stems
        .contains(&"softmax-kernel-v1".to_string()));
}

#[test]
fn format_text_produces_output() {
    let c = minimal_contract(3, 3, 2);
    let report = proof_status_report(&[("softmax-kernel-v1".to_string(), &c)], None, true);
    let text = format_text(&report);
    assert!(text.contains("Proof Status"));
    assert!(text.contains("softmax-kernel-v1"));
    assert!(text.contains("Kernel Classes:"));
    assert!(text.contains("Totals:"));
}

#[test]
fn format_text_without_classes() {
    let c = minimal_contract(2, 2, 0);
    let report = proof_status_report(&[("test-v1".to_string(), &c)], None, false);
    let text = format_text(&report);
    assert!(text.contains("test-v1"));
    assert!(!text.contains("Kernel Classes:"));
}

#[test]
fn json_roundtrip() {
    let c = minimal_contract(3, 3, 2);
    let report = proof_status_report(&[("test-v1".to_string(), &c)], None, true);
    let json = serde_json::to_string(&report).unwrap();
    let parsed: ProofStatusReport = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.contracts.len(), 1);
    assert_eq!(parsed.contracts[0].proof_level, ProofLevel::L3);
    assert_eq!(parsed.totals.obligations, 3);
}

#[test]
fn kernel_class_min_level() {
    let c1 = minimal_contract(3, 3, 2); // L3
    let c2 = minimal_contract(3, 3, 0); // L2
    let report = proof_status_report(
        &[
            ("softmax-kernel-v1".to_string(), &c1),
            ("matmul-kernel-v1".to_string(), &c2),
        ],
        None,
        true,
    );
    let class_a = report
        .kernel_classes
        .iter()
        .find(|kc| kc.label == "A")
        .unwrap();
    // min of L3 and L2 is L2
    assert_eq!(class_a.min_proof_level, ProofLevel::L2);
}

#[test]
fn count_bindings_helper() {
    let c = minimal_contract(1, 1, 1);
    let binding = crate::binding::parse_binding_str(
        r#"
version: "1.0.0"
target_crate: test
bindings:
  - contract: test.yaml
    equation: f
    status: implemented
  - contract: other.yaml
    equation: g
    status: implemented
"#,
    )
    .unwrap();
    let (implemented, total) = count_bindings("test.yaml", &c, &binding);
    assert_eq!(implemented, 1);
    assert_eq!(total, 1);
}

#[test]
fn truncate_helper() {
    assert_eq!(truncate("hello", 10), "hello");
    assert_eq!(truncate("hello world", 5), "hello");
}

/// #3338: `pv proof-status contracts/ --table` PANICKED on the real corpus.
///
/// `truncate` sliced by BYTE index, and the width is `min(max byte len, 40)`.
/// Eight contracts hold an obligation property whose byte 40 lands inside a
/// multi-byte char; the first one walked is
/// `contracts/apr-inspect-quantization-v1.yaml`:
///
/// ```text
/// byte index 40 is not a char boundary; it is inside '∈' (bytes 39..42)
/// of `for Q4_K_M Qwen2.5-Coder, quantization ∈ {Q4_K, Q6_K}`
/// ```
///
/// The fixture IS that property, so the cut is inside the same char.
#[test]
fn truncate_cuts_on_a_char_boundary_not_a_byte() {
    let s = "for Q4_K_M Qwen2.5-Coder, quantization ∈ {Q4_K, Q6_K}";
    assert!(
        !s.is_char_boundary(40),
        "fixture must cut INSIDE a multi-byte char, else it proves nothing"
    );
    let t = truncate(s, 40);
    assert!(s.starts_with(t), "truncation must be a prefix");
    assert!(t.len() <= 40, "truncation must not exceed the budget");
    assert_eq!(t, "for Q4_K_M Qwen2.5-Coder, quantization ");
}

/// The panic reached the operator through `format_obligation_table`, so the
/// table path gets its own row rather than only the helper.
#[test]
fn format_obligation_table_survives_a_multibyte_property() {
    let yaml = r#"
metadata:
  version: "1.0.0"
  description: "Multi-byte property at the cut"
  references: ["Paper"]
equations:
  f:
    formula: "f(x) = x"
proof_obligations:
  - type: invariant
    property: "for Q4_K_M Qwen2.5-Coder, quantization ∈ {Q4_K, Q6_K}"
falsification_tests: []
kani_harnesses: []
"#;
    let c = parse_contract_str(yaml).unwrap();
    let matrices = obligation_matrix(&[("multibyte-v1".to_string(), &c)]);
    let text = format_obligation_table(&matrices);
    assert!(text.contains("Contract: multibyte-v1"));
}

#[test]
fn schema_version_present() {
    let report = proof_status_report(&[], None, false);
    assert_eq!(report.schema_version, "1.0.0");
}

#[test]
fn timestamp_is_populated() {
    let report = proof_status_report(&[], None, false);
    assert!(!report.timestamp.is_empty());
    assert!(report.timestamp.ends_with('Z'));
}

// ── Obligation matrix tests ─────────────────────────────────────

#[test]
fn obligation_matrix_empty() {
    let matrices = obligation_matrix(&[]);
    assert!(matrices.is_empty());
}

/// #3347: this test used to be named `obligation_matrix_index_based_l2` and
/// asserted the defect -- 3 obligations and 3 tests, none of them linked to
/// anything, scored L2 across the board because `idx < 3`.
///
/// The contract says nothing about which test covers which obligation, so the
/// honest verdict is Unknown and the level stays L1.
#[test]
fn obligation_matrix_unlinked_tests_are_unknown_not_l2() {
    let c = minimal_contract(3, 3, 0);
    let matrices = obligation_matrix(&[("test-v1".to_string(), &c)]);
    assert_eq!(matrices.len(), 1);
    assert_eq!(matrices[0].obligations.len(), 3);
    for ob in &matrices[0].obligations {
        assert_eq!(ob.l2, L2Status::Unknown);
        assert!(!ob.l2.is_tested());
        assert!(!ob.l3_kani);
        assert!(!ob.l4_lean);
        assert_eq!(ob.max_level, ProofLevel::L1);
    }
}

/// Zero falsification tests is a READING, not an unread window: no test
/// exists, so no test covers this obligation. Untested, not Unknown.
#[test]
fn obligation_matrix_no_tests_is_untested_and_l1() {
    let c = minimal_contract(2, 0, 0);
    let matrices = obligation_matrix(&[("test-v1".to_string(), &c)]);
    assert_eq!(matrices[0].obligations.len(), 2);
    for ob in &matrices[0].obligations {
        assert_eq!(ob.l2, L2Status::Untested);
        assert_eq!(ob.max_level, ProofLevel::L1);
    }
}

/// `proof_obligations[].discharged_by` -- the link written from the
/// obligation's side, and the most-used spelling in `contracts/` (89).
#[test]
fn discharged_by_links_an_obligation_to_its_test() {
    let yaml = r#"
metadata:
  version: "1.0.0"
  description: "discharged_by, both resolvable shapes"
  references: ["Paper"]
equations:
  f:
    formula: "f(x) = x"
proof_obligations:
  - type: invariant
    property: "By index"
    discharged_by: falsification_tests[0]
  - type: invariant
    property: "By test id"
    discharged_by: FT-002
  - type: invariant
    property: "Out of range"
    discharged_by: falsification_tests[9]
  - type: invariant
    property: "Names a kani harness, not a test"
    discharged_by: KANI-X-001
falsification_tests:
  - id: FT-001
    rule: "r"
    prediction: "p"
    if_fails: "f"
  - id: FT-002
    rule: "r"
    prediction: "p"
    if_fails: "f"
kani_harnesses: []
"#;
    let c = parse_contract_str(yaml).unwrap();
    let obs = &obligation_matrix(&[("db-v1".to_string(), &c)])[0].obligations;
    assert_eq!(obs[0].l2, L2Status::Tested, "falsification_tests[0] exists");
    assert_eq!(obs[1].l2, L2Status::Tested, "FT-002 exists");
    assert_eq!(
        obs[2].l2,
        L2Status::Untested,
        "falsification_tests[9] is out of range over 2 tests -- a dangling \
         citation is not a proof"
    );
    assert_eq!(
        obs[3].l2,
        L2Status::Untested,
        "a kani harness id is the L3 column, never an L2 link"
    );
}

/// `binds_to:` is the second spelling of `falsification_tests[].obligation`
/// (38 entries vs 26). It is a serde alias, which is safe ONLY because no
/// entry in `contracts/` carries both keys -- serde would make that a
/// `duplicate field` parse error rather than a silent pick.
#[test]
fn binds_to_is_the_same_link_as_obligation() {
    let yaml = r#"
metadata:
  version: "1.0.0"
  description: "binds_to alias"
  references: ["Paper"]
equations:
  f:
    formula: "f(x) = x"
proof_obligations:
  - id: OB-1
    type: invariant
    property: "Bound by alias"
  - id: OB-2
    type: invariant
    property: "Bound by nothing"
falsification_tests:
  - id: FT-001
    binds_to: OB-1
    rule: "r"
    prediction: "p"
    if_fails: "f"
kani_harnesses: []
"#;
    let c = parse_contract_str(yaml).unwrap();
    let cited: Vec<&str> = c.falsification_tests[0]
        .obligation
        .as_ref()
        .expect("binds_to must land on the obligation field, not be dropped")
        .targets()
        .collect();
    assert_eq!(cited, vec!["OB-1"]);
    let obs = &obligation_matrix(&[("alias-v1".to_string(), &c)])[0].obligations;
    assert_eq!(obs[0].l2, L2Status::Tested);
    assert_eq!(obs[1].l2, L2Status::Untested);
}

/// A contract whose every link DANGLES was not read -- reporting its
/// obligations as Untested would be inventing a finding out of a parse
/// failure. `apr-code-harness-ir-v1` is the real instance: 8 tests cite
/// `OBLIG-IR-N` and that contract's obligations carry no `id` at all.
#[test]
fn links_that_all_dangle_are_unknown_not_untested() {
    let yaml = r#"
metadata:
  version: "1.0.0"
  description: "every citation dangles"
  references: ["Paper"]
equations:
  f:
    formula: "f(x) = x"
proof_obligations:
  - type: invariant
    property: "Alpha"
  - type: invariant
    property: "Beta"
falsification_tests:
  - id: FT-001
    obligation: OBLIG-NOBODY-1
    rule: "r"
    prediction: "p"
    if_fails: "f"
kani_harnesses: []
"#;
    let c = parse_contract_str(yaml).unwrap();
    let obs = &obligation_matrix(&[("dangle-v1".to_string(), &c)])[0].obligations;
    for ob in obs {
        assert_eq!(ob.l2, L2Status::Unknown);
    }
}

/// One test may discharge several obligations in one comma-separated field.
#[test]
fn a_comma_separated_citation_names_several_obligations() {
    let yaml = r#"
metadata:
  version: "1.0.0"
  description: "comma list"
  references: ["Paper"]
equations:
  f:
    formula: "f(x) = x"
proof_obligations:
  - id: OB-1
    type: invariant
    property: "One"
  - id: OB-2
    type: invariant
    property: "Two"
  - id: OB-3
    type: invariant
    property: "Three"
falsification_tests:
  - id: FT-001
    obligation: "OB-1, OB-3"
    rule: "r"
    prediction: "p"
    if_fails: "f"
kani_harnesses: []
"#;
    let c = parse_contract_str(yaml).unwrap();
    let obs = &obligation_matrix(&[("comma-v1".to_string(), &c)])[0].obligations;
    assert_eq!(obs[0].l2, L2Status::Tested);
    assert_eq!(obs[1].l2, L2Status::Untested);
    assert_eq!(obs[2].l2, L2Status::Tested);
}

/// A citation may also name the obligation by its exact `property` text --
/// `apr-mcp-stdio-drain-v1` binds all six of its tests that way.
#[test]
fn a_citation_may_name_the_property_text() {
    let yaml = r#"
metadata:
  version: "1.0.0"
  description: "property-text citation"
  references: ["Paper"]
equations:
  f:
    formula: "f(x) = x"
proof_obligations:
  - type: invariant
    property: "drain-on-every-exit"
  - type: invariant
    property: "no-false-error"
falsification_tests:
  - id: FT-001
    obligation: drain-on-every-exit
    rule: "r"
    prediction: "p"
    if_fails: "f"
kani_harnesses: []
"#;
    let c = parse_contract_str(yaml).unwrap();
    let obs = &obligation_matrix(&[("prop-v1".to_string(), &c)])[0].obligations;
    assert_eq!(obs[0].l2, L2Status::Tested);
    assert_eq!(obs[1].l2, L2Status::Untested);
}

#[test]
fn obligation_matrix_lean_proved() {
    // Build a contract with a Lean-proved obligation
    let yaml = r#"
metadata:
  version: "1.0.0"
  description: "Test lean"
  references: ["Paper"]
equations:
  f:
    formula: "f(x) = x"
proof_obligations:
  - type: invariant
    property: "Output sums to 1"
    lean:
      theorem: Softmax.partition_of_unity
      module: ProvableContracts.Softmax
      status: proved
  - type: bound
    property: "Range is strictly positive"
falsification_tests:
  - id: FT-001
    rule: "normalization"
    prediction: "sums to 1"
    if_fails: "bug"
  - id: FT-002
    rule: "bounded"
    prediction: "in range"
    if_fails: "bug"
kani_harnesses:
  - id: KH-001
    obligation: OB-001
    property: "Output sums to 1"
    bound: 8
"#;
    let c = parse_contract_str(yaml).unwrap();
    let matrices = obligation_matrix(&[("test-v1".to_string(), &c)]);
    assert_eq!(matrices[0].obligations.len(), 2);

    // First obligation: L3 (kani property match "sums"), L4 (lean proved). Its
    // L2 is Unknown -- the two tests name no obligation (#3347) -- which does
    // not disturb a level earned higher up the ladder.
    let ob0 = &matrices[0].obligations[0];
    assert_eq!(ob0.l2, L2Status::Unknown);
    assert!(ob0.l3_kani);
    assert!(ob0.l4_lean);
    assert_eq!(ob0.max_level, ProofLevel::L4);

    // Second obligation "Range is strictly positive": nothing at all.
    let ob1 = &matrices[0].obligations[1];
    assert_eq!(ob1.l2, L2Status::Unknown);
    assert!(!ob1.l3_kani);
    assert!(!ob1.l4_lean);
    assert_eq!(ob1.max_level, ProofLevel::L1);
}

#[test]
fn obligation_matrix_sorry_is_not_l4() {
    let yaml = r#"
metadata:
  version: "1.0.0"
  description: "Test sorry"
  references: ["Paper"]
equations:
  f:
    formula: "f(x) = x"
proof_obligations:
  - type: invariant
    property: "Prop with sorry"
    lean:
      theorem: Foo.bar
      status: sorry
falsification_tests:
  - id: FT-001
    rule: "r"
    prediction: "p"
    if_fails: "f"
kani_harnesses: []
"#;
    let c = parse_contract_str(yaml).unwrap();
    let matrices = obligation_matrix(&[("test-v1".to_string(), &c)]);
    let ob = &matrices[0].obligations[0];
    assert!(!ob.l4_lean);
    // The lone test names no obligation, so L2 is Unknown and the level is L1.
    assert_eq!(ob.l2, L2Status::Unknown);
    assert_eq!(ob.max_level, ProofLevel::L1);
}

/// RED FIRST (#3347). The L2 column ticked on `idx < falsification_tests.len()`
/// — a COUNT, not a link — so every obligation of a contract with enough tests
/// showed ✓ whoever those tests were about.
///
/// This fixture makes the two disagree: TWO obligations, TWO tests, and BOTH
/// tests cite `OB-A`. Nothing anywhere claims `OB-B` is tested. Under the index
/// rule `OB-B` is index 1 < 2 tests, so it ticked.
///
/// The assertion goes through `format_obligation_table` deliberately: the
/// rendered L2 cell is the surface the operator reads, and it is the surface
/// that was lying. It is also API-stable, so this row is the SAME test before
/// and after the fix — it fails on the old code and passes on the new.
#[test]
fn l2_does_not_tick_for_an_obligation_no_test_cites() {
    let yaml = r#"
metadata:
  version: "1.0.0"
  description: "Two obligations, two tests, both tests cite OB-A"
  references: ["Paper"]
equations:
  f:
    formula: "f(x) = x"
proof_obligations:
  - id: OB-A
    type: invariant
    property: "Alpha holds"
  - id: OB-B
    type: invariant
    property: "Beta holds"
falsification_tests:
  - id: FT-001
    obligation: OB-A
    rule: "alpha one"
    prediction: "p"
    if_fails: "f"
  - id: FT-002
    obligation: OB-A
    rule: "alpha two"
    prediction: "p"
    if_fails: "f"
kani_harnesses: []
"#;
    let c = parse_contract_str(yaml).unwrap();
    let matrices = obligation_matrix(&[("cross-cited-v1".to_string(), &c)]);
    let text = format_obligation_table(&matrices);

    let row_a = table_row(&text, "Alpha holds");
    let row_b = table_row(&text, "Beta holds");

    // Discrimination: the contract DOES bind OB-A, so a fix that simply stopped
    // ticking everything would fail here.
    assert!(
        row_a.contains('\u{2713}'),
        "OB-A is cited by two tests and must stay ticked\nrow: {row_a}"
    );
    assert!(
        !row_b.contains('\u{2713}'),
        "OB-B is cited by NO test — the L2 column ticked it from a count, not a link\nrow: {row_b}"
    );
}

/// Pull one obligation's rendered row out of the table by its property text.
fn table_row<'a>(table: &'a str, property: &str) -> &'a str {
    table
        .lines()
        .find(|l| l.contains(property))
        .unwrap_or_else(|| panic!("no table row for property `{property}`:\n{table}"))
}

#[test]
fn property_words_match_basic() {
    assert!(property_words_match(
        "output sums to one",
        "sums to one check"
    ));
    assert!(property_words_match(
        "softmax normalization",
        "verify normalization"
    ));
    assert!(!property_words_match("positivity check", "bounded range"));
}

#[test]
fn property_words_match_ignores_stop_words() {
    // "the" and "for" are stop words, should not cause a match
    assert!(!property_words_match("the output", "the input"));
    // But "output" is not a stop word
    assert!(property_words_match("output range", "output check"));
}

#[test]
fn format_obligation_table_header() {
    let c = minimal_contract(1, 1, 0);
    let matrices = obligation_matrix(&[("test-v1".to_string(), &c)]);
    let text = format_obligation_table(&matrices);
    assert!(text.contains("Obligation Status Matrix"));
    assert!(text.contains("Contract: test-v1"));
    assert!(text.contains("L2 Test"));
    assert!(text.contains("L3 Kani"));
    assert!(text.contains("L4 Lean"));
    assert!(text.contains("Status"));
}

/// All three L2 cells are reachable from the renderer, and they are distinct
/// glyphs. A fix that collapsed Unknown onto either tick or cross would fail
/// here rather than pass quietly.
#[test]
fn format_obligation_table_renders_all_three_l2_cells() {
    // 1 obligation, 1 test, no link => `?`; L3/L4 crossed.
    let unknown = minimal_contract(1, 1, 0);
    let text = format_obligation_table(&obligation_matrix(&[("u-v1".to_string(), &unknown)]));
    assert!(text.contains('?'), "unlinked L2 renders as `?`:\n{text}");
    assert!(
        text.contains('\u{2717}'),
        "L3/L4 render as crosses:\n{text}"
    );

    // 0 tests => a cross in the L2 column, not a `?`.
    let untested = minimal_contract(1, 0, 0);
    let text = format_obligation_table(&obligation_matrix(&[("x-v1".to_string(), &untested)]));
    assert!(
        !text.contains('?'),
        "0 tests is a reading, not Unknown:\n{text}"
    );

    // A linked obligation still ticks.
    let yaml = r#"
metadata:
  version: "1.0.0"
  description: "linked"
  references: ["Paper"]
equations:
  f:
    formula: "f(x) = x"
proof_obligations:
  - id: OB-1
    type: invariant
    property: "Linked prop"
falsification_tests:
  - id: FT-001
    obligation: OB-1
    rule: "r"
    prediction: "p"
    if_fails: "f"
kani_harnesses: []
"#;
    let linked = parse_contract_str(yaml).unwrap();
    let text = format_obligation_table(&obligation_matrix(&[("l-v1".to_string(), &linked)]));
    assert!(
        text.contains('\u{2713}'),
        "a linked obligation ticks:\n{text}"
    );
}

#[test]
fn format_obligation_table_empty_obligations() {
    let c = minimal_contract(0, 0, 0);
    let matrices = obligation_matrix(&[("test-v1".to_string(), &c)]);
    let text = format_obligation_table(&matrices);
    // Should not have "Contract: test-v1" since obligations are empty
    assert!(!text.contains("Contract: test-v1"));
}

#[test]
fn obligation_matrix_multiple_contracts() {
    let c1 = minimal_contract(2, 2, 0);
    let c2 = minimal_contract(1, 0, 0);
    let matrices =
        obligation_matrix(&[("alpha-v1".to_string(), &c1), ("beta-v1".to_string(), &c2)]);
    assert_eq!(matrices.len(), 2);
    assert_eq!(matrices[0].stem, "alpha-v1");
    assert_eq!(matrices[0].obligations.len(), 2);
    assert_eq!(matrices[1].stem, "beta-v1");
    assert_eq!(matrices[1].obligations.len(), 1);
}

#[test]
fn is_fully_bound_edge_cases() {
    let c1 = minimal_contract(1, 1, 1);
    assert_eq!(compute_proof_level(&c1, None), ProofLevel::L3);
    let c2 = contract_with_lean(1, 1);
    assert_eq!(
        compute_proof_level_with_grounding(&c2, Some((0, 0)), 1),
        ProofLevel::L4
    );
    assert_eq!(
        compute_proof_level_with_grounding(&c2, Some((1, 2)), 1),
        ProofLevel::L4
    );
}

#[test]
fn report_multiple_contracts_totals() {
    let c1 = minimal_contract(3, 3, 2);
    let c2 = minimal_contract(5, 5, 1);
    let report = proof_status_report(&[("a-v1".into(), &c1), ("b-v1".into(), &c2)], None, false);
    assert_eq!(report.totals.contracts, 2);
    assert_eq!(report.totals.obligations, 8);
    assert_eq!(report.totals.falsification_tests, 8);
    assert_eq!(report.totals.kani_harnesses, 3);
}

#[test]
fn kernel_class_all_bound_false_when_partial() {
    let c = minimal_contract(3, 3, 2);
    let report = proof_status_report(&[("softmax-kernel-v1".into(), &c)], None, true);
    let class_a = report
        .kernel_classes
        .iter()
        .find(|kc| kc.label == "A")
        .unwrap();
    assert!(!class_a.all_bound);
}

/// CI GUARD (Rank #1): the Lean scan must be self-sufficient from the in-tree
/// staging tree, NOT dependent on a co-located `../provable-contracts` sibling.
///
/// Before the scan-path fix, `proof_status.rs` scanned only
/// `["lean", "../provable-contracts/lean"]`; `./lean` does not exist, so on a
/// fresh clone / CI (no sibling checkout) every Lean-backed contract silently
/// collapsed — the 25 L4 dropped to L3 and both L5 to L3. This test fails if the
/// in-tree tree is removed/emptied or demoted from primary, which is exactly the
/// condition that would make L4/L5 non-reproducible.
#[test]
fn lean_scan_in_tree_is_primary_and_self_sufficient() {
    // (1) The in-tree staging tree must be the FIRST scan base.
    assert_eq!(
        LEAN_THEOREM_BASES[0], "crates/aprender-contracts-staging/lean",
        "the in-tree staging Lean tree must be the primary scan base so proof \
         levels are reproducible without the external ../provable-contracts sibling"
    );

    // (2) That tree must exist and carry a healthy set of sorry-free theorems.
    let theorems = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../crates/aprender-contracts-staging/lean/ProvableContracts/Theorems");
    assert!(
        theorems.is_dir(),
        "in-tree Lean theorems dir missing: {}",
        theorems.display()
    );

    let mut sorry_free = 0usize;
    let mut stack = vec![theorems.clone()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|e| e == "lean") {
                if let Ok(content) = std::fs::read_to_string(&path) {
                    if !content.contains("sorry") {
                        sorry_free += 1;
                    }
                }
            }
        }
    }
    assert!(
        sorry_free >= 40,
        "expected the in-tree Lean tree to carry the corpus of sorry-free \
         theorems (>=40), found {sorry_free} — L4/L5 levels would not be \
         reproducible on CI"
    );
}

// ── ONT-2a (ONT-001 R-4, andon): a claim is not a proof ───────────

/// A summary that claims Lean proof, with no in-tree theorem to ground it, is SELF-DECLARED. It does not
/// reach L4, and bindings cannot promote it to L5 either: `is_lean_proved` now reads the tree, not the
/// contract's opinion of itself.
#[test]
fn ont2a_a_claim_with_nothing_under_it_is_not_l4() {
    let c = contract_with_lean(3, 3); // the summary says 3 of 3 proved; no .lean theorem resolves
    assert!(
        is_l4_self_declared(&c),
        "the claim covers every obligation and nothing in the tree grounds it"
    );
    assert_eq!(compute_proof_level(&c, None), ProofLevel::L3);
    assert_eq!(
        compute_proof_level(&c, Some((1, 1))),
        ProofLevel::L3,
        "bindings must not promote an ungrounded claim to L5"
    );
}

/// The report performs the withdrawal in the open: the contract is flagged, its grounded count is zero,
/// and the report says the L4 total excludes such contracts.
#[test]
fn ont2a_the_report_flags_the_claim_and_says_the_total_excludes_it() {
    let c = contract_with_lean(2, 2);
    let report = proof_status_report(&[("claims-l4".to_string(), &c)], None, false);
    assert!(report.l4_self_declared_excluded);
    assert_eq!(report.totals.l4_self_declared, 1);
    assert_eq!(report.totals.lean_grounded, 0);
    assert!(report.contracts[0].l4_self_declared);
    assert_ne!(report.contracts[0].proof_level, ProofLevel::L4);
}

/// Both words are printed. A withdrawal nobody can see on the line is the silence this row exists to end.
#[test]
fn ont2a_the_text_output_prints_self_declared_and_grounded() {
    let c = contract_with_lean(2, 2);
    let report = proof_status_report(&[("claims-l4".to_string(), &c)], None, false);
    let text = format_text(&report);
    assert!(
        text.contains("self-declared"),
        "the line must say it:\n{text}"
    );
    assert!(
        text.contains("grounded"),
        "and the other column must be named:\n{text}"
    );
}

/// The flag marks an unbacked CLAIM, not the mere absence of a proof: a contract that claims nothing is
/// not self-declared, it is just not L4.
#[test]
fn ont2a_a_contract_that_claims_nothing_is_not_flagged() {
    let c = minimal_contract(2, 2, 2);
    assert!(!is_l4_self_declared(&c));
    let report = proof_status_report(&[("quiet".to_string(), &c)], None, false);
    assert_eq!(report.totals.l4_self_declared, 0);
    assert!(report.l4_self_declared_excluded);
}

// ── PMAT-3091: N/A obligations are counted apart and never raise a level ────

/// `n_ob` ordinary obligations plus `n_na` declared `applies_to: not_applicable`.
fn contract_with_na_obligations(n_ob: usize, n_na: usize, n_ft: usize, n_kani: usize) -> Contract {
    let mut c = minimal_contract(n_ob, n_ft, n_kani);
    for i in 0..n_na {
        let ob: crate::schema::ProofObligation = serde_yaml::from_str(&format!(
            "property: \"na {i}\"\napplies_to: not_applicable\nna_reason: r\nna_owner: o\n"
        ))
        .expect("N/A obligation fixture must parse");
        c.proof_obligations.push(ob);
    }
    c
}

#[test]
fn na_obligations_are_counted_separately_as_k_na() {
    let c = contract_with_na_obligations(3, 2, 5, 0);
    let report = proof_status_report(&[("na-v1".to_string(), &c)], None, false);
    assert_eq!(report.contracts[0].obligations, 5);
    assert_eq!(report.contracts[0].not_applicable, 2);
    assert_eq!(report.totals.not_applicable, 2);
    let text = format_text(&report);
    assert!(text.contains("2 N/A"), "{text}");
}

#[test]
fn na_count_is_zero_without_declarations() {
    let c = minimal_contract(3, 3, 0);
    let report = proof_status_report(&[("plain-v1".to_string(), &c)], None, false);
    assert_eq!(report.contracts[0].not_applicable, 0);
    assert_eq!(report.totals.not_applicable, 0);
}

/// 3 tests cover the 3 ordinary obligations; the 4th is N/A. Were N/A dropped
/// from the denominator, 3 >= 3 would promote the contract to L2 on the
/// strength of a declaration. It stays where an undeclared 4th would leave it.
#[test]
fn na_obligation_does_not_raise_the_level() {
    let with_na = contract_with_na_obligations(3, 1, 3, 0);
    let undeclared = minimal_contract(4, 3, 0);
    assert_eq!(compute_proof_level(&with_na, None), ProofLevel::L1);
    assert_eq!(
        compute_proof_level(&with_na, None),
        compute_proof_level(&undeclared, None)
    );
    let kani = contract_with_na_obligations(3, 1, 3, 3);
    assert_eq!(compute_proof_level(&kani, None), ProofLevel::L1);
}

/// An all-N/A contract has nothing proved, so no L4 credit from the grounding
/// path: the schema N/A is not `verification_summary.l4_not_applicable`.
#[test]
fn na_obligations_grant_no_lean_credit() {
    let c = contract_with_na_obligations(0, 3, 3, 3);
    assert!(!is_lean_proved_with_grounding(&c, 0));
    let one = contract_with_na_obligations(1, 3, 4, 4);
    assert!(!is_lean_proved_with_grounding(&one, 1));
}
