//! SCHEMA-018/019/020: the top-level keys serde used to drop in silence.
//!
//! Every test here is a PAIRED mutation control. A rule that has only ever been
//! seen going green proves nothing, so each case asserts both directions on the
//! same document: the defect present ⇒ the rule fires, the defect removed ⇒ it
//! does not. `legitimate_downstream_keys_are_not_flagged` is the non-vacuity
//! control in the other direction — it pins the 1224-of-1726 contracts that
//! legitimately carry downstream-owned top-level blocks and must stay clean.

use super::*;
use crate::schema::parse_contract_str;

/// A minimal, otherwise-valid registry contract with `extra` spliced in at the
/// top level. Registry kind keeps the kernel-only rules quiet so each test sees
/// only the rule under examination.
fn contract_with_top_level(extra: &str) -> String {
    format!(
        r#"{extra}
metadata:
  version: "1.0.0"
  description: "top-level key fixture"
  registry: true
  references:
    - "Test paper (2026)"
"#
    )
}

fn rules_fired(yaml: &str) -> Vec<String> {
    let contract = parse_contract_str(yaml).expect("fixture must parse");
    validate_contract(&contract)
        .into_iter()
        .map(|v| v.rule)
        .collect()
}

#[test]
fn top_level_kind_is_an_error_but_metadata_kind_is_not() {
    // RED: the key at the top level, where serde drops it.
    let broken = contract_with_top_level("kind: registry");
    assert!(
        rules_fired(&broken).contains(&"SCHEMA-018".to_string()),
        "a top-level `kind:` must raise SCHEMA-018, got {:?}",
        rules_fired(&broken)
    );

    // GREEN: the identical value in the place the schema actually reads.
    let fixed = r#"
metadata:
  version: "1.0.0"
  description: "top-level key fixture"
  kind: registry
  references:
    - "Test paper (2026)"
"#;
    assert!(
        !rules_fired(fixed).contains(&"SCHEMA-018".to_string()),
        "`metadata.kind:` is the correct location and must not raise SCHEMA-018"
    );
}

#[test]
fn top_level_kind_error_names_the_fix() {
    let contract = parse_contract_str(&contract_with_top_level("kind: KernelContract"))
        .expect("fixture must parse");
    let v = validate_contract(&contract)
        .into_iter()
        .find(|v| v.rule == "SCHEMA-018")
        .expect("SCHEMA-018 must fire");
    assert_eq!(v.severity, Severity::Error);
    assert!(
        v.message.contains("metadata.kind"),
        "the message must name where the key belongs, got: {}",
        v.message
    );
}

#[test]
fn near_miss_block_names_are_errors_that_name_the_real_field() {
    // (misspelling actually written, block name that was meant)
    let cases = [
        ("falsification_test", "falsification_tests"),
        ("falsification-tests", "falsification_tests"),
        ("Falsification_Tests", "falsification_tests"),
        ("falsificationTests", "falsification_tests"),
        ("proof_obligation", "proof_obligations"),
        ("proofObligations", "proof_obligations"),
        ("equation", "equations"),
        ("kani_harness", "kani_harnesses"),
        ("qa_gates", "qa_gate"),
        ("simd_dispatches", "simd_dispatch"),
        ("type_invariant", "type_invariants"),
        ("coq_specs", "coq_spec"),
        ("beats", "beat"),
        ("kernel_structures", "kernel_structure"),
        ("meta_data", "metadata"),
    ];
    for (typo, meant) in cases {
        let yaml = contract_with_top_level(&format!("{typo}: []"));
        let contract = parse_contract_str(&yaml).expect("fixture must parse");
        let v = validate_contract(&contract)
            .into_iter()
            .find(|v| v.rule == "SCHEMA-019")
            .unwrap_or_else(|| panic!("`{typo}:` must raise SCHEMA-019"));
        assert_eq!(v.severity, Severity::Error, "{typo}");
        assert!(
            v.message.contains(meant),
            "SCHEMA-019 for `{typo}` must name `{meant}`, got: {}",
            v.message
        );
    }
}

#[test]
fn legitimate_downstream_keys_are_not_flagged() {
    // The non-vacuity control. 1224 of the 1726 contracts `pv lint` walks carry
    // at least one downstream-owned top-level block; if SCHEMA-019 were a
    // blanket unknown-key rule, every name below would red the whole corpus.
    // These are the highest-frequency real ones, plus the near-collisions the
    // plural-stripping normalizer has to get right (`invariants` is NOT
    // `type_invariants`; `gates` is NOT `qa_gate`; `spec` is NOT `coq_spec`).
    for key in [
        "version",
        "name",
        "status",
        "contract",
        "surface",
        "date",
        "pmat_work_tracking",
        "page",
        "sections",
        "scope",
        "description",
        "preconditions",
        "postconditions",
        "invariants",
        "gates",
        "spec",
        "family",
        "architectures",
        "size_variants",
        "inputs",
        "outputs",
        "references",
        "summary",
        "certification",
        "constraints",
        "required_fields",
    ] {
        let yaml = contract_with_top_level(&format!("{key}: \"x\""));
        let fired = rules_fired(&yaml);
        assert!(
            !fired.iter().any(|r| r == "SCHEMA-018" || r == "SCHEMA-019"),
            "top-level `{key}:` is a legitimate downstream block and must not be \
             flagged, got {fired:?}"
        );
    }
}

#[test]
fn legacy_falsification_block_is_captured_not_dropped() {
    // The `contracts/publish-workspace-v1.yaml` shape (#2504): four FALSIFY-PUB-*
    // entries under a top-level `falsification:` key. Before the field existed,
    // serde threw all four away and `pv status` said "Falsification tests: 0"
    // with no further comment.
    let yaml = r#"
metadata:
  version: "1.0.0"
  description: "publish-workspace shape"
  registry: true
  references:
    - "Potvin & Levenberg (2016)"
falsification:
  - name: FALSIFY-PUB-001
    description: "Crate published before its dependency"
    check: "cargo install fails"
  - name: FALSIFY-PUB-002
    description: "Shim crate fails to re-export"
    check: "type mismatch"
"#;
    let contract = parse_contract_str(yaml).expect("fixture must parse");
    assert_eq!(
        contract.legacy_falsification_entries(),
        2,
        "the legacy block must be visible to tooling"
    );
    assert!(
        contract.falsification_tests.is_empty(),
        "the legacy block must NEVER be counted as falsification_tests — that \
         would silently mark an inert contract as enforced"
    );
    // And it is a known key, so it is not reported as a near-miss.
    assert!(
        !rules_fired(yaml).contains(&"SCHEMA-019".to_string()),
        "`falsification:` is a captured legacy block, not a misspelling"
    );
}

#[test]
fn duplicate_mapping_key_is_rejected_and_the_deduplicated_form_is_not() {
    // RED: `contracts/apr-cli-commands-v1.yaml` defined `subcommands:` twice
    // inside one command. The derived deserializer never reads that subtree, so
    // it parsed clean here while every strict reader dropped one of the blocks.
    let broken = r#"
metadata:
  version: "1.0.0"
  description: "duplicate key fixture"
  registry: true
  references:
    - "YAML 1.2 §7.4.2"
commands:
  - name: modelfile
    subcommands: [parse]
    subcommands: [parse, render]
"#;
    let contract = parse_contract_str(broken).expect("the schema does accept it — that is the bug");
    assert!(
        validate_contract(&contract)
            .iter()
            .any(|v| v.rule == "SCHEMA-020"),
        "a duplicate mapping key must raise SCHEMA-020"
    );

    // GREEN: one key, one value.
    let fixed = r#"
metadata:
  version: "1.0.0"
  description: "duplicate key fixture"
  registry: true
  references:
    - "YAML 1.2 §7.4.2"
commands:
  - name: modelfile
    subcommands: [parse, render]
"#;
    assert!(
        !rules_fired(fixed).contains(&"SCHEMA-020".to_string()),
        "the deduplicated document must be clean"
    );
}

#[test]
fn duplicate_key_does_not_blind_the_unknown_key_capture() {
    // The trap this rule was born from: capturing top-level keys through a
    // strict `serde_yaml::Value` returned "no unknown keys" for any document
    // with a nested duplicate, so `apr-cli-commands-v1.yaml`'s top-level
    // `kind: CLICommandContract` went unreported while 118 identical keys in
    // other files were caught. One silent failure hiding another.
    let yaml = r#"
kind: CLICommandContract
metadata:
  version: "1.0.0"
  description: "duplicate key fixture"
  registry: true
  references:
    - "YAML 1.2 §7.4.2"
commands:
  - name: modelfile
    subcommands: [parse]
    subcommands: [parse, render]
"#;
    let fired = rules_fired(yaml);
    assert!(
        fired.contains(&"SCHEMA-018".to_string()),
        "the top-level `kind:` must still be found, got {fired:?}"
    );
    assert!(
        fired.contains(&"SCHEMA-020".to_string()),
        "and the duplicate key must be reported too, got {fired:?}"
    );
}

#[test]
fn allow_list_matches_the_contract_struct() {
    // Pins CONTRACT_TOP_LEVEL_FIELDS to the struct. Add a field to `Contract`
    // without adding it here and the new block becomes "a near-miss of itself":
    // every contract using it would collect a SCHEMA-019 error.
    let value = serde_yaml::to_value(Contract::default()).expect("Contract must serialize");
    let mapping = value.as_mapping().expect("Contract serializes to a mapping");
    let mut serialized: Vec<&str> = mapping.keys().filter_map(serde_yaml::Value::as_str).collect();
    serialized.sort_unstable();
    let mut declared: Vec<&str> = CONTRACT_TOP_LEVEL_FIELDS.to_vec();
    declared.sort_unstable();
    assert_eq!(
        serialized, declared,
        "CONTRACT_TOP_LEVEL_FIELDS has drifted from the Contract struct"
    );
}
