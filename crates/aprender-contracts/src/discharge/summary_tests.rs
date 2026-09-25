//! PVL-001 EV-8a (#4202) — the summary's shape, its derivation rule, and what `modules` lists.

use super::*;
use crate::discharge::comparator::Closure;

fn green() -> Summary {
    Summary {
        tree_sha: Some("t".into()),
        toolchain: Some("leanprover/lean4:v4.29.0-rc4".into()),
        mathlib_rev: Some("m".into()),
        build_exit: Some(0),
        lake_exit: Some(0),
        leanchecker_exit: Some(0),
        axioms_ok: true,
        escapes_ok: true,
        challenges_closed: Some("1/1".into()),
        modules: vec![Module {
            path: "ProvableContracts/Theorems/Gelu/Bound.lean".into(),
            blake3: "b".into(),
            theorems: vec!["ProvableContracts.Gelu.gelu_bound".into()],
        }],
    }
}

#[test]
fn only_a_summary_green_on_every_step_derives() {
    let g = green();
    assert!(g.is_green());
    assert_eq!(g.derived().len(), 1);
    let reds: [fn(&mut Summary); 7] = [
        |s| s.build_exit = Some(2),
        |s| s.lake_exit = Some(1),
        |s| s.lake_exit = None,
        |s| s.leanchecker_exit = Some(124),
        |s| s.leanchecker_exit = None,
        |s| s.axioms_ok = false,
        |s| s.escapes_ok = false,
    ];
    for (i, red) in reds.iter().enumerate() {
        let mut s = green();
        red(&mut s);
        assert!(!s.is_green(), "mutation {i} stayed green");
        assert!(s.derived().is_empty(), "mutation {i} derived something");
    }
}

#[test]
fn the_file_has_the_spec_fields_in_order_and_no_clock() {
    let text = green().render();
    assert!(text.ends_with("}\n"));
    let keys: Vec<&str> = text
        .lines()
        .filter(|l| l.starts_with("  \""))
        .filter_map(|l| l.trim().split('"').nth(1))
        .collect();
    assert_eq!(
        keys,
        [
            "tree_sha",
            "toolchain",
            "mathlib_rev",
            "build_exit",
            "lake_exit",
            "leanchecker_exit",
            "axioms_ok",
            "escapes_ok",
            "challenges_closed",
            "modules"
        ]
    );
    assert_eq!(load_str(&text), green(), "it reads back as written");
    assert_eq!(
        green().render(),
        text,
        "the same run renders the same bytes"
    );
}

fn load_str(text: &str) -> Summary {
    let d = tempfile::tempdir().expect("tempdir");
    let p = d.path().join(SUMMARY_FILE);
    std::fs::write(&p, text).expect("w");
    load(&p).expect("a summary")
}

#[test]
fn a_claim_names_its_theorem_exactly_or_under_the_root_namespace() {
    let g = green();
    let d = g.derived();
    assert!(claim_derived(&d, "ProvableContracts.Gelu.gelu_bound"));
    assert!(claim_derived(&d, "Gelu.gelu_bound"));
    for miss in [
        "gelu_bound",
        "Gelu",
        "Gelu.gelu_bound'",
        "Theorems.Gelu",
        "X.Gelu.gelu_bound",
    ] {
        assert!(!claim_derived(&d, miss), "{miss} derived");
    }
}

#[test]
fn the_summary_sits_beside_the_lean_dir_never_inside_it() {
    assert_eq!(
        summary_path(Path::new("crates/aprender-contracts-staging/lean")),
        Path::new("crates/aprender-contracts-staging/discharge-summary.json")
    );
    assert_eq!(summary_path(Path::new("lean")), Path::new(SUMMARY_FILE));
}

/// Root → Gelu/Bound (one clean theorem, one sorry'd, one private); Orphan is outside the cone.
fn tree() -> (tempfile::TempDir, std::path::PathBuf) {
    let d = tempfile::tempdir().expect("tempdir");
    let lean = d.path().join("lean");
    let w = |rel: &str, text: &str| {
        let p = lean.join(rel);
        std::fs::create_dir_all(p.parent().expect("parent")).expect("mkdir");
        std::fs::write(p, text).expect("w");
    };
    w(
        "ProvableContracts.lean",
        "import ProvableContracts.Theorems.Gelu.Bound\n",
    );
    w(
        "ProvableContracts/Theorems/Gelu/Bound.lean",
        "namespace ProvableContracts.Gelu\ntheorem gelu_bound : True := trivial\n\
         theorem gelu_open : True := by sorry\nprivate theorem helper : True := trivial\nend ProvableContracts.Gelu\n",
    );
    w(
        "ProvableContracts/Theorems/Gelu/Orphan.lean",
        "namespace ProvableContracts.Gelu\ntheorem orphan : True := trivial\nend ProvableContracts.Gelu\n",
    );
    w("lean-toolchain", "leanprover/lean4:v4.29.0-rc4\n");
    w(
        "lake-manifest.json",
        "{\"packages\": [{\"name\": \"aesop\", \"rev\": \"a\"}, {\"name\": \"mathlib\", \"rev\": \"1d04\"}]}",
    );
    (d, lean)
}

#[test]
fn modules_are_the_cone_with_their_escape_free_public_theorems() {
    let (_d, lean) = tree();
    let t = Tree::load(&lean).expect("tree");
    let m = modules(&t);
    let paths: Vec<&str> = m.iter().map(|x| x.path.as_str()).collect();
    assert_eq!(
        paths,
        [
            "ProvableContracts.lean",
            "ProvableContracts/Theorems/Gelu/Bound.lean"
        ],
        "the orphan is outside the cone: nothing checked it"
    );
    assert_eq!(
        m[1].theorems,
        ["ProvableContracts.Gelu.gelu_bound"],
        "sorry'd and private are not listed"
    );
    let bytes = std::fs::read(lean.join(&m[1].path)).expect("r");
    assert_eq!(m[1].blake3, blake3::hash(&bytes).to_hex().to_string());
    assert_eq!(m[1].blake3.len(), 64);
}

#[test]
fn summarize_reads_the_report_and_the_pins() {
    let (_d, lean) = tree();
    let t = Tree::load(&lean).expect("tree");
    let mut r = Report {
        lake_exit: Some(0),
        leanchecker_exit: Some(0),
        escapes_ok: Some(true),
        axioms_fresh: Some(true),
        challenges: Some(Closure {
            closed: 3,
            total: 5,
        }),
        ..Report::default()
    };
    let s = summarize(&r, Some(&t), &lean, Some("abc".into()), Some(0));
    assert!(s.is_green());
    assert_eq!(s.toolchain.as_deref(), Some("leanprover/lean4:v4.29.0-rc4"));
    assert_eq!(s.mathlib_rev.as_deref(), Some("1d04"));
    assert_eq!(s.challenges_closed.as_deref(), Some("3/5"));
    assert_eq!(s.tree_sha.as_deref(), Some("abc"));
    // axioms_ok needs BOTH halves: the file is its regeneration, and it elaborated
    r.lake_exit = Some(1);
    assert!(!summarize(&r, Some(&t), &lean, None, Some(0)).axioms_ok);
    r.lake_exit = Some(0);
    r.axioms_fresh = Some(false);
    assert!(!summarize(&r, Some(&t), &lean, None, Some(0)).axioms_ok);
    r.axioms_fresh = None;
    assert!(
        !summarize(&r, Some(&t), &lean, None, Some(0)).axioms_ok,
        "never compared is not ok"
    );
    r.escapes_ok = None;
    assert!(!summarize(&r, None, &lean, None, Some(0)).escapes_ok);
    assert!(summarize(&r, None, &lean, None, Some(0)).modules.is_empty());
}

// ── EV-8b (#4082): L4 credit from the summary ─────────────────────

#[test]
fn a_green_fresh_closed_summary_grants_its_theorems_by_either_name() {
    let d = discharged(&green(), Some("t"));
    assert_eq!(d.withheld, None);
    assert!(d.grants("ProvableContracts.Gelu.gelu_bound"));
    assert!(
        d.grants("Gelu.gelu_bound"),
        "the YAML may drop the root namespace"
    );
    assert!(
        !d.grants("Gelu.absent"),
        "a theorem the summary does not list earns nothing"
    );
}

/// (mutation of a green summary, the current tree sha, the withheld reason it must produce)
type WithholdCase = (fn(&mut Summary), Option<&'static str>, &'static str);

#[test]
fn a_stale_red_or_unclosed_summary_grants_nothing_and_says_why() {
    let cases: [WithholdCase; 7] = [
        (|_| {}, Some("other"), "stale discharge"),
        (|_| {}, None, "stale discharge"),
        (|s| s.tree_sha = None, Some("t"), "stale discharge"),
        (|s| s.lake_exit = Some(1), Some("t"), "red discharge"),
        (
            |s| s.leanchecker_exit = Some(124),
            Some("t"),
            "red discharge",
        ),
        (
            |s| s.challenges_closed = Some("2/3".into()),
            Some("t"),
            "challenges not closed",
        ),
        (
            |s| s.challenges_closed = None,
            Some("t"),
            "challenges not closed",
        ),
    ];
    for (i, (mutate, current, why)) in cases.iter().enumerate() {
        let mut s = green();
        mutate(&mut s);
        let d = discharged(&s, *current);
        assert!(d.theorems.is_empty(), "case {i} granted {:?}", d.theorems);
        assert!(
            d.withheld.as_deref().is_some_and(|w| w.starts_with(why)),
            "case {i}: {:?}",
            d.withheld
        );
    }
}

#[test]
fn zero_challenges_is_not_all_closed() {
    assert!(!all_challenges_closed(Some("0/0")));
    assert!(!all_challenges_closed(Some("x/1")));
    assert!(all_challenges_closed(Some("3/3")));
}
