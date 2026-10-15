//! #5291: SCHEMA-024 (`falsification_tests:` written and empty) and SCHEMA-025
//! (a `test_harness:` / `test:` written but blank), as a must-pass / must-fail
//! case table. Every row states how many of each rule it expects, so a row
//! that should stay silent fails as loudly as one that should fire.
//!
//! The headline rows are the #5290 shape: `kind: kernel` plus the legacy
//! `registry: true`, which makes the contract a registry and exempts it from
//! PROVABILITY-001. On pv 0.71.1 both mutants of that contract validated with
//! rc 0 and 0 errors.

use super::*;
use crate::schema::parse_contract_str;

/// The #5290 metadata: declared a kernel, demoted to a registry by the flag.
const REGISTRY_META: &str = r#"
metadata:
  version: "1.0.0"
  description: "registry fixture"
  references: ["Paper (2024)"]
  kind: kernel
  registry: true
"#;

/// A complete kernel contract minus its `falsification_tests:` block.
const KERNEL_META: &str = r#"
metadata:
  version: "1.0.0"
  description: "kernel fixture"
  references: ["Paper (2024)"]
  kind: kernel
equations:
  f:
    formula: "f(x) = x"
proof_obligations:
  - type: invariant
    property: "output is finite"
kani_harnesses:
  - id: KANI-001
    obligation: "output is finite"
    bound: 8
qa_gate:
  id: F-001
  name: "Test Gate"
  checks: ["finiteness"]
  pass_criteria: "All tests pass"
"#;

fn test_entry(id: &str, harness_line: &str) -> String {
    format!(
        "  - id: {id}\n    rule: \"finiteness\"\n    prediction: \"output is always finite\"\n    if_fails: \"overflow\"\n{harness_line}"
    )
}

fn count(violations: &[Violation], rule: &str) -> usize {
    violations
        .iter()
        .filter(|v| v.rule == rule && v.severity == Severity::Error)
        .count()
}

#[test]
fn schema_024_025_case_table() {
    let one_ok = format!(
        "falsification_tests:\n{}",
        test_entry("F-1", "    test_harness: \"cargo test -p x finite\"\n")
    );
    let six_blank: String = (1..=6)
        .map(|i| test_entry(&format!("F-{i}"), "    test_harness: \"\"\n"))
        .collect();
    let six_blank = format!("falsification_tests:\n{six_blank}");

    // (row, metadata+body, block, SCHEMA-024 count, SCHEMA-025 count)
    let rows: Vec<(&str, &str, String, usize, usize)> = vec![
        // must-pass
        ("registry, key absent", REGISTRY_META, String::new(), 0, 0),
        ("registry, one test with a harness", REGISTRY_META, one_ok.clone(), 0, 0),
        (
            "registry, test names no harness at all (not this rule)",
            REGISTRY_META,
            format!("falsification_tests:\n{}", test_entry("F-1", "")),
            0,
            0,
        ),
        ("kernel, one test with a harness", KERNEL_META, one_ok, 0, 0),
        // must-fail
        (
            "registry, written and empty (#5290 mutant 1)",
            REGISTRY_META,
            "falsification_tests: []\n".to_string(),
            1,
            0,
        ),
        (
            "registry, six blank test_harness (#5290 mutant 2)",
            REGISTRY_META,
            six_blank,
            0,
            6,
        ),
        (
            "registry, whitespace-only test_harness",
            REGISTRY_META,
            format!("falsification_tests:\n{}", test_entry("F-1", "    test_harness: \"   \"\n")),
            0,
            1,
        ),
        (
            "registry, blank test:",
            REGISTRY_META,
            format!("falsification_tests:\n{}", test_entry("F-1", "    test: \"\"\n")),
            0,
            1,
        ),
        (
            "registry, blank command: (alias of test:)",
            REGISTRY_META,
            format!("falsification_tests:\n{}", test_entry("F-1", "    command: \"\"\n")),
            0,
            1,
        ),
        (
            "kernel, written and empty",
            KERNEL_META,
            "falsification_tests: []\n".to_string(),
            1,
            0,
        ),
    ];

    for (row, meta, block, want_024, want_025) in rows {
        let yaml = format!("{meta}{block}");
        let contract = parse_contract_str(&yaml).unwrap_or_else(|e| panic!("{row}: parse: {e}"));
        let violations = validate_contract(&contract);
        assert_eq!(count(&violations, "SCHEMA-024"), want_024, "{row}: SCHEMA-024 in {violations:?}");
        assert_eq!(count(&violations, "SCHEMA-025"), want_025, "{row}: SCHEMA-025 in {violations:?}");
    }
}

/// The kernel row above must not be the only thing refusing an empty kernel
/// block: PROVABILITY-001 still fires beside SCHEMA-024. A change that let
/// SCHEMA-024 replace it would weaken the kernel path.
#[test]
fn kernel_empty_block_still_fails_provability() {
    let contract = parse_contract_str(&format!("{KERNEL_META}falsification_tests: []\n")).unwrap();
    let violations = validate_contract(&contract);
    assert!(violations.iter().any(|v| v.rule == "PROVABILITY-001"), "{violations:?}");
}

/// The parse artifact itself: absent and `[]` are the same `Vec` and must
/// not be the same flag.
#[test]
fn falsification_tests_declared_tells_absent_from_empty() {
    let absent = parse_contract_str(REGISTRY_META).unwrap();
    let empty = parse_contract_str(&format!("{REGISTRY_META}falsification_tests: []\n")).unwrap();
    assert!(absent.falsification_tests.is_empty() && !absent.falsification_tests_declared);
    assert!(empty.falsification_tests.is_empty() && empty.falsification_tests_declared);
}
