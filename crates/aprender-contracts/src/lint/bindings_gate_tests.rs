//! ONT-3a `bindings` gate. Each test copies `tests/fixtures/ont/code-bound` (a one-member workspace whose registry
//! binds five present items and one ghost, `kern::nn::functional::no_such_function`) and varies one input.

use std::path::{Path, PathBuf};

use super::*;
use crate::lint::{run_named_gate, NamedGateOutcome, NAMED_GATES};

const GHOST: &str = "kern::nn::functional::no_such_function";

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/ont/code-bound")
}

fn copy_dir(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for e in std::fs::read_dir(from).unwrap() {
        let e = e.unwrap();
        let dst = to.join(e.file_name());
        if e.file_type().unwrap().is_dir() {
            copy_dir(&e.path(), &dst);
        } else {
            std::fs::copy(e.path(), dst).unwrap();
        }
    }
}

/// A private copy of the fixture workspace; returns (guard, contract dir).
fn workspace() -> (tempfile::TempDir, PathBuf) {
    let tmp = tempfile::tempdir().unwrap();
    copy_dir(&fixture(), tmp.path());
    let contracts = tmp.path().join("contracts");
    (tmp, contracts)
}

fn allow(contracts: &Path, json: &str) {
    std::fs::write(contracts.join(ALLOWLIST), json).unwrap();
}

fn entry(symbol: &str) -> String {
    format!(r##"{{"symbol": "{symbol}", "reason": "ghost", "ticket": "#1"}}"##)
}

fn ran(outcome: RatchetOutcome) -> (GateResult, Vec<LintFinding>) {
    match outcome {
        RatchetOutcome::Ran { result, findings } => (*result, findings),
        RatchetOutcome::Declined(why) => panic!("declined: {why}"),
    }
}

fn counters(r: &GateResult) -> BindingsCounters {
    match &r.extra {
        Some(GateExtra::Bindings(c)) => (**c).clone(),
        other => panic!("no bindings counters: {other:?}"),
    }
}

fn rules(findings: &[LintFinding]) -> Vec<&str> {
    findings.iter().map(|f| f.rule_id.as_str()).collect()
}

#[test]
fn an_unallowlisted_ghost_is_rejected_by_name() {
    let (_g, c) = workspace();
    let (r, f) = ran(run_bindings_gate(&c));
    assert_eq!(r.verdict, Verdict::Fail);
    assert_eq!(rules(&f), ["PV-ONT-028"]);
    assert!(f[0].message.contains(GHOST), "{}", f[0].message);
    let k = counters(&r);
    assert_eq!(
        (k.checked, k.resolved, k.ghosts, k.allowlisted),
        (6, 5, 1, 0)
    );
    assert_eq!(k.pc_resolver, "fired");
}

#[test]
fn an_allowlisted_ghost_passes_and_is_counted() {
    let (_g, c) = workspace();
    allow(&c, &format!(r#"{{"entries": [{}]}}"#, entry(GHOST)));
    let (r, f) = ran(run_bindings_gate(&c));
    assert!(f.is_empty(), "{f:?}");
    assert_eq!(r.verdict, Verdict::Pass);
    assert_eq!(counters(&r).allowlisted, 1);
}

#[test]
fn an_allowlist_entry_for_a_resolving_symbol_is_stale() {
    let (_g, c) = workspace();
    allow(
        &c,
        &format!(
            r#"{{"entries": [{}, {}]}}"#,
            entry(GHOST),
            entry("kern::helper")
        ),
    );
    let (r, f) = ran(run_bindings_gate(&c));
    assert_eq!(r.verdict, Verdict::Fail);
    assert_eq!(rules(&f), ["PV-ONT-029"]);
    assert!(f[0].message.contains("kern::helper"));
}

#[test]
fn an_allowlist_entry_no_registry_binds_is_stale() {
    let (_g, c) = workspace();
    allow(
        &c,
        &format!(
            r#"{{"entries": [{}, {}]}}"#,
            entry(GHOST),
            entry("kern::never_bound")
        ),
    );
    let (_, f) = ran(run_bindings_gate(&c));
    assert_eq!(rules(&f), ["PV-ONT-029"]);
}

#[test]
fn a_malformed_or_duplicated_allowlist_is_rejected() {
    for bad in [
        "not json".to_string(),
        r#"{"entries": [{"symbol": "x"}]}"#.to_string(),
        format!(r#"{{"entries": [{}], "extra": 1}}"#, entry(GHOST)),
        format!(r#"{{"entries": [{}, {}]}}"#, entry(GHOST), entry(GHOST)),
        r##"{"entries": [{"symbol": "kern::nn::functional::no_such_function", "reason": " ", "ticket": "#1"}]}"##
            .to_string(),
    ] {
        let (_g, c) = workspace();
        allow(&c, &bad);
        let (r, f) = ran(run_bindings_gate(&c));
        assert_eq!(r.verdict, Verdict::Fail, "{bad}");
        assert!(rules(&f).contains(&"PV-ONT-030"), "{bad}: {f:?}");
    }
}

#[test]
fn a_not_implemented_binding_claims_no_code() {
    let (_g, c) = workspace();
    let reg = c.join("binding.yaml");
    let text = std::fs::read_to_string(&reg).unwrap();
    let marked = text.replace(
        "  function: no_such_function\n  status: implemented",
        "  function: no_such_function\n  status: not_implemented",
    );
    assert_ne!(text, marked, "the fixture's ghost row moved");
    std::fs::write(&reg, marked).unwrap();
    let (r, f) = ran(run_bindings_gate(&c));
    assert!(f.is_empty(), "{f:?}");
    assert_eq!(counters(&r).checked, 5);
}

#[test]
fn a_corpus_outside_a_workspace_declines() {
    let (g, c) = workspace();
    std::fs::remove_file(g.path().join("Cargo.toml")).unwrap();
    match run_bindings_gate(&c) {
        RatchetOutcome::Declined(why) => assert!(why.contains("not at a workspace root"), "{why}"),
        RatchetOutcome::Ran { .. } => panic!("measured a corpus with no workspace"),
    }
}

#[test]
fn a_corpus_with_no_implemented_binding_declines() {
    let (_g, c) = workspace();
    let reg = c.join("binding.yaml");
    let text = std::fs::read_to_string(&reg).unwrap();
    std::fs::write(&reg, text.replace("status: implemented", "status: pending")).unwrap();
    assert!(matches!(run_bindings_gate(&c), RatchetOutcome::Declined(_)));
}

#[test]
fn the_gate_runs_alone_under_its_name() {
    assert!(NAMED_GATES.contains(&GATE));
    let (_g, c) = workspace();
    match run_named_gate(&c, GATE) {
        NamedGateOutcome::Ratchet(RatchetOutcome::Ran { result, .. }) => {
            assert_eq!(result.name, GATE);
        }
        _ => panic!("`bindings` did not run as a ratchet gate"),
    }
}

/// The enforcing run: CI's `--lib` job is where this gate binds the repo, since no workflow calls `make contracts`.
/// A new ghost fails here by name, and a fixed ghost fails here until its allowlist entry is removed.
#[test]
fn the_repo_corpus_has_no_unallowlisted_ghost_and_no_stale_entry() {
    let repo = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../contracts");
    let (r, f) = ran(run_bindings_gate(&repo));
    assert!(
        f.is_empty(),
        "{:#?}",
        f.iter().map(|x| &x.message).collect::<Vec<_>>()
    );
    let k = counters(&r);
    assert_eq!((k.ghosts, k.stale_allowlist), (0, 0));
    assert!(
        k.resolved > 200 && k.crates_scanned > 1,
        "the walk measured little: {k:?}"
    );
}

/// Appends one implemented binding `module_path::function` to the copy's registry, and `src` to the named file.
fn bind(c: &Path, module_path: &str, function: &str, file: &str, src: &str) {
    let reg = c.join("binding.yaml");
    let mut text = std::fs::read_to_string(&reg).unwrap();
    text.push_str(&format!(
        "- contract: softmax-kernel-v1.yaml\n  equation: {function}\n  module_path: {module_path}\n  function: {function}\n  status: implemented\n"
    ));
    std::fs::write(&reg, text).unwrap();
    if !src.is_empty() {
        let path = c.join("../crates/kern/src").join(file);
        let mut code = std::fs::read_to_string(&path).unwrap();
        code.push_str(src);
        std::fs::write(&path, code).unwrap();
    }
}

#[test]
fn a_pub_super_fn_resolves() {
    let (_g, c) = workspace();
    allow(&c, &format!(r#"{{"entries": [{}]}}"#, entry(GHOST)));
    bind(
        &c,
        "kern::nn::functional",
        "sup_helper",
        "nn/functional.rs",
        "pub(super) fn sup_helper() {}\n",
    );
    let (r, f) = ran(run_bindings_gate(&c));
    assert!(f.is_empty(), "{f:?}");
    let k = counters(&r);
    assert_eq!((k.checked, k.resolved, k.ghosts), (7, 6, 0));
}

#[test]
fn a_fn_bound_under_the_wrong_module_does_not_resolve() {
    let (_g, c) = workspace();
    allow(&c, &format!(r#"{{"entries": [{}]}}"#, entry(GHOST)));
    // `relu` exists, but in `kern::nn::functional`, not `kern::nn`.
    bind(&c, "kern::nn", "relu", "", "");
    let (r, f) = ran(run_bindings_gate(&c));
    assert_eq!(r.verdict, Verdict::Fail);
    assert_eq!(rules(&f), ["PV-ONT-028"]);
    assert!(f[0].message.contains("kern::nn::relu"), "{}", f[0].message);
    assert_eq!(counters(&r).ghosts, 1);
}

#[test]
fn an_allowlist_entry_without_a_ticket_is_rejected() {
    let (_g, c) = workspace();
    allow(
        &c,
        &format!(r#"{{"entries": [{{"symbol": "{GHOST}", "reason": "ghost"}}]}}"#),
    );
    let (r, f) = ran(run_bindings_gate(&c));
    assert_eq!(r.verdict, Verdict::Fail);
    assert!(rules(&f).contains(&"PV-ONT-030"), "{f:?}");
}

/// ONT-001 row ONT-3a's probe: `jq -e '.pc_extract=="fired" and .unresolved==0 and .crates_scanned>1'` over the
/// gate's JSON. The field names are the spec's, so this reads them off the serialized counters, not the struct.
#[test]
fn the_spec_probe_fields_are_serialized_and_count_only_unallowlisted_ghosts() {
    let json = |c: &Path| serde_json::to_value(counters(&ran(run_bindings_gate(c)).0)).unwrap();
    let (_g, c) = workspace();
    let v = json(&c);
    assert_eq!(
        (v["pc_extract"].as_str(), v["unresolved"].as_u64()),
        (Some("fired"), Some(1))
    );
    allow(&c, &format!(r#"{{"entries": [{}]}}"#, entry(GHOST)));
    assert_eq!(json(&c)["unresolved"].as_u64(), Some(0));
    let repo = json(&Path::new(env!("CARGO_MANIFEST_DIR")).join("../../contracts"));
    assert_eq!(repo["pc_extract"], "fired");
    assert_eq!(repo["unresolved"], 0);
    assert!(repo["crates_scanned"].as_u64() > Some(1), "{repo}");
}

/// PV-ONT-031's matcher. Must-match: the file-path forms the registries held before #4502's rewrite. Must-not-match:
/// real module paths, including ones with a segment that merely contains `src`/`mod`/`rs`.
#[test]
fn the_file_path_form_case_table() {
    for p in [
        "presentar::src::widgets::chart",
        "trueno_db::src::query",
        "batuta::src::oracle::mod",
        "kern::ops::mod",
        "kern::ops::activation.rs",
        "simular::src::engine::rng",
    ] {
        assert!(is_file_path_form(p), "must match: {p}");
    }
    for p in [
        "src",
        "kern::nn::functional",
        "kern::source::modes",
        "kern::srcs::module",
        "kern::rs::modal",
        "aprender::format::converter",
        "kern::ops::activation",
    ] {
        assert!(!is_file_path_form(p), "must not match: {p}");
    }
}

#[test]
fn a_file_path_module_path_is_rejected_even_when_it_would_resolve_elsewhere() {
    let (_g, c) = workspace();
    allow(&c, &format!(r#"{{"entries": [{}]}}"#, entry(GHOST)));
    bind(&c, "kern::src::nn::functional", "softmax", "", "");
    let (r, f) = ran(run_bindings_gate(&c));
    assert_eq!(r.verdict, Verdict::Fail);
    let rs = rules(&f);
    assert!(rs.contains(&"PV-ONT-031"), "{rs:?}");
    assert!(
        f.iter()
            .any(|x| x.rule_id == "PV-ONT-031" && x.message.contains("kern::src::nn::functional")),
        "{f:?}"
    );
    assert_eq!(counters(&r).file_path_form, 1);
}

#[test]
fn an_allowlisted_file_path_ghost_is_exempt_and_counted_by_class() {
    let (_g, c) = workspace();
    let fp = "kern::nn::mod::no_such_function";
    bind(&c, "kern::nn::mod", "no_such_function", "", "");
    allow(
        &c,
        &format!(r#"{{"entries": [{}, {}]}}"#, entry(GHOST), entry(fp)),
    );
    let (r, f) = ran(run_bindings_gate(&c));
    assert!(f.is_empty(), "{f:?}");
    let k = counters(&r);
    assert_eq!(
        (
            k.allowlisted,
            k.allowlisted_absent_leaf,
            k.allowlisted_no_module,
            k.file_path_form
        ),
        (2, 1, 1, 0)
    );
}

/// infra-83's ruling on #4502: no file-path row survives unallowlisted, and every allowlisted ghost is in a named
/// class, so the "counted separately" totals add up.
#[test]
fn the_repo_corpus_has_no_file_path_row_and_every_allowlisted_ghost_has_a_class() {
    let repo = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../contracts");
    let (r, _) = ran(run_bindings_gate(&repo));
    let k = counters(&r);
    assert_eq!(k.file_path_form, 0);
    assert_eq!(
        k.allowlisted,
        k.allowlisted_absent_leaf
            + k.allowlisted_no_module
            + k.allowlisted_not_member
            + k.allowlisted_other,
        "{k:?}"
    );
    assert_eq!(k.allowlisted_other, 0, "{k:?}");
}

/// Kills `+=` -> `*=`/`-=` on `allowlisted_other` in `count_allowlisted`: two unclassified reasons count 2.
#[test]
fn count_allowlisted_counts_each_class_exactly() {
    let mut c = BindingsCounters::default();
    count_allowlisted(&mut c, "something else entirely");
    count_allowlisted(&mut c, "another unclassified reason");
    assert_eq!((c.allowlisted, c.allowlisted_other), (2, 2));
    count_allowlisted(&mut c, "crate x is not a workspace member");
    assert_eq!(
        (c.allowlisted, c.allowlisted_not_member, c.allowlisted_other),
        (3, 1, 2)
    );
}

/// Kills `+=` -> `*=` on `stale_allowlist`, and the deleted `registries`/`files_parsed` counter fields.
#[test]
fn stale_entries_and_scan_counters_are_reported() {
    let (_g, c) = workspace();
    allow(
        &c,
        &format!(
            r#"{{"entries": [{}, {}, {}]}}"#,
            entry(GHOST),
            entry("kern::nn::functional::relu"),
            entry("kern::nn::functional::softmax")
        ),
    );
    let (r, f) = ran(run_bindings_gate(&c));
    let k = counters(&r);
    assert_eq!(k.stale_allowlist, 2, "{f:?}");
    assert!(k.registries >= 1, "{k:?}");
    assert!(k.files_parsed >= 1, "{k:?}");
}
