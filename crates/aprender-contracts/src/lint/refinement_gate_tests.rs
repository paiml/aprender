//! ONT-001 ONT-3b (#4073) — the refinement gate: ghosts, malformed entries, and HEAD against BASE for both
//! counts (unrefined, ORPHANED-ROOT), measured from source by one scanner. The planted falsifiers are
//! `a_new_unproven_module_at_head_is_033_and_named` and `a_bound_root_leaving_the_cone_at_head_is_034_and_named`.

use super::*;

const SM: &str = "ProvableContracts/Theorems/Softmax/Kernel.lean";
const GELU: &str = "ProvableContracts/Theorems/Gelu/Bound.lean";
const SILU: &str = "ProvableContracts/Theorems/Silu/Mono.lean";

/// One side: `<tmp>/lean` (root importing `imports`, the three theorem files always present on disk) and
/// `<tmp>/contracts` (one contract binding the Gelu theorem by label). `formalization` `None` = no file.
fn side(imports: &[&str], formalization: Option<&str>) -> tempfile::TempDir {
    let d = tempfile::tempdir().expect("tempdir");
    let put = |rel: &str, text: &str| {
        let p = d.path().join(rel);
        std::fs::create_dir_all(p.parent().expect("parent")).expect("mkdir");
        std::fs::write(p, text).expect("write");
    };
    let root: String = imports
        .iter()
        .map(|p| format!("import {}\n", p.trim_end_matches(".lean").replace('/', ".")))
        .collect();
    put(
        "lean/ProvableContracts.lean",
        &format!("/- root -/\n{root}"),
    );
    for (path, ns, thm) in [
        (SM, "Softmax", "sm_kernel"),
        (GELU, "Gelu", "gelu_bound"),
        (SILU, "Silu", "silu_mono"),
    ] {
        put(
            &format!("lean/{path}"),
            &format!("namespace ProvableContracts.{ns}\ntheorem {thm} : True := trivial\nend ProvableContracts.{ns}\n"),
        );
    }
    put(
        "contracts/gelu-v1.yaml",
        "equations:\n  e:\n    lean_theorem: Theorems.Gelu\n",
    );
    if let Some(f) = formalization {
        put("lean/formalization.yaml", f);
    }
    d
}

fn resolver(p: &str) -> Result<(), String> {
    if p.ends_with("::real") {
        Ok(())
    } else {
        Err(format!("no item {p}"))
    }
}

fn gate(head: &tempfile::TempDir, base: &tempfile::TempDir) -> RatchetOutcome {
    run_with(
        &head.path().join("lean"),
        &head.path().join("contracts"),
        &base.path().join("lean"),
        &base.path().join("contracts"),
        "base-fixture",
        resolver,
    )
}

fn ran(o: RatchetOutcome) -> (Verdict, Vec<LintFinding>, RefinementCounters) {
    match o {
        RatchetOutcome::Ran { result, findings } => {
            let Some(GateExtra::Refinement(c)) = result.extra else {
                panic!("no counters")
            };
            (result.verdict, findings, *c)
        }
        RatchetOutcome::Declined(why) => panic!("declined: {why}"),
    }
}

fn rules(f: &[LintFinding]) -> Vec<&str> {
    f.iter().map(|f| f.rule_id.as_str()).collect()
}

/// Models for `modules`, each `kind`, `model_of` ending in `a`.
fn models(kind: &str, a: &str, modules: &[&str]) -> String {
    let mut s = String::from("models:\n");
    for m in modules {
        s += &format!("  - {{ module: {m}, model_of: c::m::{a}, relation: {{ kind: {kind}, evidence: e }} }}\n");
    }
    s
}

#[test]
fn equal_head_and_base_pass_and_report_both_counts_for_both_sides() {
    let f = models("extraction", "real", &[SM]);
    let (head, base) = (side(&[SM, GELU], Some(&f)), side(&[SM, GELU], Some(&f)));
    let (v, f, c) = ran(gate(&head, &base));
    assert_eq!((v, f.len()), (Verdict::Pass, 0), "{:?}", rules(&f));
    assert_eq!(
        (c.models, c.resolved, c.l4_models, c.theorem_modules),
        (1, 1, 1, 2)
    );
    assert_eq!(
        (
            c.unrefined,
            c.base_unrefined,
            c.orphaned_roots,
            c.base_orphaned_roots
        ),
        (1, 1, 0, 0)
    );
    assert_eq!(c.base, "base-fixture");
}

#[test]
fn a_new_unproven_module_at_head_is_033_and_named() {
    // The planted falsifier: HEAD imports one more theorem-bearing module and models nothing for it.
    let f = models("extraction", "real", &[SM, GELU]);
    let base = side(&[SM, GELU], Some(&f));
    let head = side(&[SM, GELU, SILU], Some(&f));
    let (v, f, c) = ran(gate(&head, &base));
    assert_eq!((v, rules(&f)), (Verdict::Fail, vec!["PV-ONT-033"]));
    assert!(f[0].message.contains(SILU), "{}", f[0].message);
    assert_eq!((c.unrefined, c.base_unrefined, c.new_unrefined), (1, 0, 1));

    // The same module modelled at HEAD is not new: Pass.
    let fh = models("extraction", "real", &[SM, GELU, SILU]);
    let head = side(&[SM, GELU, SILU], Some(&fh));
    assert_eq!(ran(gate(&head, &base)).0, Verdict::Pass);
}

#[test]
fn a_module_losing_its_model_at_head_is_033() {
    let base = side(&[SM, GELU], Some(&models("extraction", "real", &[SM])));
    let head = side(&[SM, GELU], Some("models: []\n"));
    let (v, f, c) = ran(gate(&head, &base));
    assert_eq!((v, rules(&f)), (Verdict::Fail, vec!["PV-ONT-033"]));
    assert!(f[0].message.contains(SM), "{}", f[0].message);
    assert_eq!((c.unrefined, c.base_unrefined), (2, 1));
}

#[test]
fn a_bound_root_leaving_the_cone_at_head_is_034_and_named() {
    // The planted falsifier: the contract binds Gelu; HEAD drops its import, BASE has it.
    let f = models("extraction", "real", &[SM]);
    let base = side(&[SM, GELU], Some(&f));
    let head = side(&[SM], Some(&f));
    let (v, f, c) = ran(gate(&head, &base));
    assert_eq!((v, rules(&f)), (Verdict::Fail, vec!["PV-ONT-034"]));
    assert!(
        f[0].message.contains("ProvableContracts.Gelu.gelu_bound"),
        "{}",
        f[0].message
    );
    assert_eq!(
        (
            c.orphaned_roots,
            c.base_orphaned_roots,
            c.new_orphaned_roots
        ),
        (1, 0, 1)
    );
    // Fewer unrefined at HEAD (Gelu left the cone) earns nothing against the orphan.
    assert_eq!((c.unrefined, c.base_unrefined), (0, 1));
}

#[test]
fn an_orphan_base_already_had_is_not_new() {
    let f = models("extraction", "real", &[SM]);
    let (head, base) = (side(&[SM], Some(&f)), side(&[SM], Some(&f)));
    let (v, f, c) = ran(gate(&head, &base));
    assert_eq!((v, f.len()), (Verdict::Pass, 0), "{:?}", rules(&f));
    assert_eq!((c.orphaned_roots, c.base_orphaned_roots), (1, 1));
}

#[test]
fn draining_orphans_into_the_cone_trades_one_count_for_the_other() {
    // BASE orphans Gelu; HEAD imports it with no model: 034 is fixed, 033 names Gelu. Both counts, one run.
    let f = models("extraction", "real", &[SM]);
    let base = side(&[SM], Some(&f));
    let head = side(&[SM, GELU], Some(&f));
    let (v, f, c) = ran(gate(&head, &base));
    assert_eq!((v, rules(&f)), (Verdict::Fail, vec!["PV-ONT-033"]));
    assert!(f[0].message.contains(GELU));
    assert_eq!(
        (
            c.orphaned_roots,
            c.base_orphaned_roots,
            c.unrefined,
            c.base_unrefined
        ),
        (0, 1, 1, 0)
    );
}

#[test]
fn a_base_without_formalization_has_every_theorem_module_unrefined() {
    let base = side(&[SM, GELU], None);
    let head = side(&[SM, GELU], Some(&models("extraction", "real", &[SM])));
    let (v, f, c) = ran(gate(&head, &base));
    assert_eq!((v, f.len()), (Verdict::Pass, 0), "{:?}", rules(&f));
    assert_eq!((c.unrefined, c.base_unrefined), (1, 2));
}

#[test]
fn a_ghost_model_of_is_named_and_refines_nothing() {
    let base = side(&[SM, GELU], Some("models: []\n"));
    let head = side(&[SM, GELU], Some(&models("extraction", "ghost", &[SM])));
    let (v, f, c) = ran(gate(&head, &base));
    assert_eq!((v, rules(&f)), (Verdict::Fail, vec!["PV-ONT-031"]));
    assert_eq!((c.ghosts, c.l4_models, c.unrefined), (1, 0, 2));

    let head = side(&[SM, GELU], Some(&models("test_witnessed", "real", &[SM])));
    let (v, _, c) = ran(gate(&head, &base));
    assert_eq!(v, Verdict::Pass);
    assert_eq!(
        (c.l3_models, c.l4_models, c.unrefined),
        (1, 0, 2),
        "test_witnessed refines nothing"
    );
}

#[test]
fn a_malformed_entry_or_a_module_outside_the_cone_is_032() {
    let base = side(&[SM, GELU], Some("models: []\n"));
    let head = side(
        &[SM, GELU],
        Some(&models("extraction", "real", &["X.lean"])),
    );
    assert_eq!(rules(&ran(gate(&head, &base)).1), vec!["PV-ONT-032"]);
    let head = side(&[SM, GELU], Some(&models("extraction", "real", &[SILU])));
    assert_eq!(
        rules(&ran(gate(&head, &base)).1),
        vec!["PV-ONT-032"],
        "Silu is on disk but not imported"
    );
    let head = side(
        &[SM, GELU],
        Some("models:\n  - { module: X.lean, relation: { kind: extraction, evidence: e } }\n"),
    );
    assert_eq!(rules(&ran(gate(&head, &base)).1), vec!["PV-ONT-032"]);
}

#[test]
fn no_head_tree_no_head_formalization_or_no_base_tree_declines() {
    let good = side(&[SM, GELU], Some("models: []\n"));
    let empty = tempfile::tempdir().expect("tempdir");
    let bare = side(&[SM, GELU], None);
    for (head, base, why) in [
        (&empty, &good, "no HEAD tree"),
        (&bare, &good, "no HEAD formalization.yaml"),
        (&good, &empty, "no BASE tree"),
    ] {
        assert!(
            matches!(gate(head, base), RatchetOutcome::Declined(_)),
            "{why}"
        );
    }
}
