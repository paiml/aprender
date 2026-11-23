//! Case tables for the stub predicate, plus the two corpus guards that run in
//! `workspace-test` (pass/fail, no report-only row).

use super::*;
use std::path::PathBuf;

const STUB: &str = r#"# Auto-generated from .pmat-work/PMAT-1/contract.json
name: "PMAT-1"
metadata:
  version: "1.0.0"
  kind: schema
  description: "Auto-generated work-contract for PMAT-1"
  references:
    - ".pmat-work/PMAT-1/contract.json"
surface: work-contract
verification_summary:
  target_level: L1
  current_level: L1
  total_obligations: 47
"#;

const TEST_BLOCK: &str = "falsification_tests:\n  - id: F-1\n    rule: \"r\"\n    prediction: \"p\"\n    if_fails: \"f\"\n";

#[test]
fn is_work_stub_case_table() {
    let with_tests = format!("{STUB}{TEST_BLOCK}");
    let with_eq = format!("{STUB}equations:\n  e:\n    formula: \"x\"\n");
    let with_ob = format!("{STUB}proof_obligations: []\n");
    let extra_top = format!("{STUB}entity: x\n");
    let extra_sub = STUB.replace("  kind: schema\n", "  kind: schema\n  owner: x\n");
    let no_header = STUB.replacen("# Auto-generated from .pmat-work/", "# hand-written ", 1);
    let other_surface = STUB.replace("surface: work-contract", "surface: api");
    let rows: [(&str, &str, bool); 8] = [
        ("the pmat template", STUB, true),
        ("carries a falsification test", &with_tests, false),
        ("carries equations", &with_eq, false),
        ("carries proof_obligations", &with_ob, false),
        ("extra top-level key", &extra_top, false),
        ("extra metadata key", &extra_sub, false),
        ("no pmat header", &no_header, false),
        ("another surface", &other_surface, false),
    ];
    for (row, text, stub) in rows {
        assert_eq!(
            is_work_stub(text).is_ok(),
            stub,
            "{row}: {:?}",
            is_work_stub(text)
        );
    }
}

#[test]
fn has_work_marker_case_table() {
    let header_only = STUB.replace("surface: work-contract\n", "");
    let surface_only = STUB.replacen("# Auto-generated from .pmat-work/", "# x ", 1);
    let rows: [(&str, &str, bool); 5] = [
        ("header and surface", STUB, true),
        ("header only", &header_only, true),
        ("surface only", &surface_only, true),
        ("a real contract", "metadata:\n  version: \"1\"\n", false),
        (
            "surface nested, not top-level",
            "metadata:\n  surface: work-contract\n",
            false,
        ),
    ];
    for (row, text, want) in rows {
        assert_eq!(has_work_marker(text), want, "{row}");
    }
    assert_eq!(falsification_test_count(STUB), 0);
    assert_eq!(falsification_test_count(&format!("{STUB}{TEST_BLOCK}")), 1);
}

#[test]
fn guards_fire_on_planted_files() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let tickets = tmp.path().join("tickets");
    let contracts = tmp.path().join("contracts");
    std::fs::create_dir_all(&tickets).expect("mkdir");
    std::fs::create_dir_all(contracts.join("work/quarantine")).expect("mkdir");
    std::fs::write(tickets.join("A.yaml"), STUB).expect("write");
    std::fs::write(contracts.join("work/T.yaml"), format!("{STUB}{TEST_BLOCK}")).expect("write");
    assert_eq!(non_stubs_in(&tickets), (1, vec![]));
    assert!(untested_work_files_in(&contracts).is_empty());
    std::fs::write(
        tickets.join("real-v1.yaml"),
        "metadata:\n  version: \"1\"\n",
    )
    .expect("write");
    std::fs::write(contracts.join("work/quarantine/S.yml"), STUB).expect("write");
    assert_eq!(non_stubs_in(&tickets).1.len(), 1);
    assert_eq!(untested_work_files_in(&contracts).len(), 1);
}

/// GUARD (a): every file in `docs/work-tickets/` is a pmat stub.
#[test]
fn work_tickets_dir_holds_only_stubs() {
    let Some(dir): Option<PathBuf> = crate::tree::workspace_path_or_skip_at(
        "work_tickets_dir_holds_only_stubs",
        Path::new(env!("CARGO_MANIFEST_DIR")),
        WORK_TICKETS_DIR,
    ) else {
        return;
    };
    let (n, bad) = non_stubs_in(&dir);
    assert!(
        n > 0,
        "{} holds no stub: the guard would be vacuous",
        dir.display()
    );
    assert!(
        bad.is_empty(),
        "{WORK_TICKETS_DIR}/ holds only pmat work-ticket stubs; a real contract goes under \
         contracts/ with its falsification tests: {bad:?}"
    );
}

/// GUARD (b): no work-ticket stub under `contracts/`.
#[test]
fn no_untested_work_ticket_under_contracts() {
    let Some(dir): Option<PathBuf> = crate::tree::workspace_path_or_skip_at(
        "no_untested_work_ticket_under_contracts",
        Path::new(env!("CARGO_MANIFEST_DIR")),
        "contracts",
    ) else {
        return;
    };
    let bad = untested_work_files_in(&dir);
    assert!(
        bad.is_empty(),
        "pmat work-ticket stub(s) under contracts/ with no falsification test: {bad:?}. \
         Move each one with `git mv contracts/work/<ID>.yaml {WORK_TICKETS_DIR}/` \
         (byte-for-byte; do not add it to no_test_debt.txt)."
    );
}
