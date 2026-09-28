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
            assert_eq!(result.passed, result.verdict == Verdict::Pass);
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
    assert_eq!(c.base_theorem_modules, 2);
    assert_eq!(c.pc_resolver, crate::ontology::witness::FIRED);
}

#[test]
fn named_lists_twelve_then_counts_the_rest() {
    let items: Vec<String> = (0..13).map(|i| format!("m{i}")).collect();
    let refs: Vec<&String> = items.iter().collect();
    assert_eq!(named(&refs[..1]), "m0");
    assert!(!named(&refs[..NAMED]).contains("more"));
    assert!(named(&refs).ends_with("m11 and 1 more"), "{}", named(&refs));
}

/// The real entry point: no formalization.yaml is one decline, and a formalization outside git is another.
#[test]
fn the_entry_point_declines_without_a_formalization_and_without_a_base() {
    let d = tempfile::tempdir().expect("tmp");
    let contracts = d.path().join("contracts");
    std::fs::create_dir_all(&contracts).expect("mkdir");
    let why = |o: RatchetOutcome| match o {
        RatchetOutcome::Declined(w) => w,
        _ => panic!("expected a decline"),
    };
    assert!(why(run_refinement_gate(&contracts)).contains("formalization.yaml beside the corpus"));
    let lean = d.path().join(LEAN_DIR);
    std::fs::create_dir_all(&lean).expect("mkdir");
    std::fs::write(lean.join("formalization.yaml"), "models: []\n").expect("write");
    assert!(why(run_refinement_gate(&contracts)).starts_with("no BASE to compare with"));
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

fn sh_git(dir: &Path, args: &[&str]) {
    let ok = std::process::Command::new("git")
        .arg("-C")
        .arg(dir)
        .args([
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@t",
            "-c",
            "commit.gpgsign=false",
        ])
        .args(args)
        .env_remove("GIT_DIR")
        .env_remove("GIT_WORK_TREE")
        .env_remove("GIT_INDEX_FILE")
        .status()
        .expect("git")
        .success();
    assert!(ok, "git {args:?}");
}

/// A repo whose origin/main holds `formalization.yaml` under LEAN_DIR and one contract.
fn repo() -> tempfile::TempDir {
    let d = tempfile::tempdir().expect("tmp");
    let lean = d.path().join(LEAN_DIR);
    std::fs::create_dir_all(&lean).expect("mkdir");
    std::fs::create_dir_all(d.path().join("contracts")).expect("mkdir");
    std::fs::write(lean.join("formalization.yaml"), "models: []\n").expect("write");
    std::fs::write(d.path().join("contracts/c.yaml"), "x: 1\n").expect("write");
    sh_git(d.path(), &["init", "-q"]);
    sh_git(d.path(), &["add", "."]);
    sh_git(d.path(), &["commit", "-q", "-m", "base"]);
    sh_git(
        d.path(),
        &["update-ref", "refs/remotes/origin/main", "HEAD"],
    );
    d
}

#[test]
fn base_is_extracted_from_git_at_the_merge_base() {
    let d = repo();
    let lean = d.path().join(LEAN_DIR);
    std::fs::write(lean.join("formalization.yaml"), "models: [head-only]\n").expect("write");
    let b = BaseTree::extract(d.path(), &d.path().join("contracts")).expect("extract");
    let head = git(d.path(), &["rev-parse", "HEAD"]).expect("head");
    assert_eq!(
        b.label,
        format!("merge-base(HEAD, origin/main) {}", &head[..12])
    );
    let text = std::fs::read_to_string(b.lean.join("formalization.yaml")).expect("base file");
    assert_eq!(text, "models: []\n");
    assert!(b.contracts.join("c.yaml").is_file());
    let scratch = b.dir.clone();
    drop(b);
    assert!(
        !scratch.exists(),
        "BASE scratch left behind: {}",
        scratch.display()
    );
}

#[test]
#[cfg(unix)]
fn archive_fails_when_tar_cannot_write() {
    let d = repo();
    let commit = git(d.path(), &["rev-parse", "HEAD"]).expect("head");
    let out = tempfile::tempdir().expect("tmp");
    // A target under a regular file: `tar -C` fails with ENOTDIR, git archive does not (one
    // small file fits the pipe). Not chmod 0o555: CI runs as root, and root writes through it.
    let file = out.path().join("not-a-dir");
    std::fs::write(&file, "").expect("file");
    let dir = file.join("base");
    let b = BaseTree {
        lean: dir.join(LEAN_DIR),
        contracts: dir.join("contracts"),
        dir,
        label: "t".into(),
    };
    let r = b.archive(d.path(), &commit, &[LEAN_DIR]);
    assert!(r.is_err(), "{r:?}");
}
