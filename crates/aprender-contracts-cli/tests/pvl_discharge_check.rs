//! PVL-001 EV-6a (#4139) — `pv discharge gen-axioms | check`, end to end on a built-in-a-tempdir Lean tree.
//!
//! Every row states the property it defends, and the RED rows carry the spec's mutations verbatim:
//! `axiom pvl_mutation : False` in a Theorem file, and `@[implemented_by]` on a def. The Lean elaboration of
//! `Axioms.lean` is not run here (`--no-lake`: CI has no Mathlib); it was measured on lambda (see #4139).

use std::path::PathBuf;
use std::process::Command;

fn pv_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_pv"))
}

struct Run {
    code: i32,
    stdout: String,
    stderr: String,
}

impl Run {
    fn show(&self) -> String {
        format!(
            "rc {}\n--- stdout\n{}\n--- stderr\n{}",
            self.code, self.stdout, self.stderr
        )
    }
}

const THEOREM_FILE: &str = "lean/ProvableContracts/Theorems/Gelu/Bound.lean";
const CONTRACT: &str = "contracts/gelu-v1.yaml";

/// A one-theorem tree: the root imports `Theorems/Gelu/Bound.lean`, whose `gelu_bound` the contract binds by
/// its domain label `Theorems.Gelu`. The doc comment and the string carry `sorry`/`axiom` text that must NOT scan.
struct Fx {
    dir: tempfile::TempDir,
}

impl Fx {
    fn new() -> Self {
        let fx = Self {
            dir: tempfile::tempdir().expect("tempdir"),
        };
        fx.write(
            "lean/ProvableContracts.lean",
            "import ProvableContracts.Theorems.Gelu.Bound\n",
        );
        fx.write(
            THEOREM_FILE,
            "namespace ProvableContracts.Gelu\n\
             /-- sorry-free: `sorry` and `axiom` in a doc comment are not escapes -/\n\
             theorem gelu_bound : True := trivial\n\
             def s := \"axiom sorry admit\" -- native_decide\n\
             end ProvableContracts.Gelu\n",
        );
        fx.write(
            CONTRACT,
            "equations:\n  e:\n    lean_theorem: Theorems.Gelu\n",
        );
        fx
    }

    fn path(&self, rel: &str) -> PathBuf {
        self.dir.path().join(rel)
    }

    fn write(&self, rel: &str, text: &str) {
        let p = self.path(rel);
        std::fs::create_dir_all(p.parent().expect("parent")).expect("mkdir");
        std::fs::write(p, text).expect("write");
    }

    fn append(&self, rel: &str, text: &str) {
        let old = std::fs::read_to_string(self.path(rel)).expect("read");
        self.write(rel, &format!("{old}{text}"));
    }

    fn pv(&self, args: &[&str]) -> Run {
        let out = Command::new(pv_bin())
            .current_dir(self.dir.path())
            .args(args)
            .output()
            .expect("spawn pv");
        Run {
            code: out.status.code().unwrap_or(-1),
            stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
        }
    }

    fn gen(&self) -> Run {
        self.pv(&["discharge", "gen-axioms", "lean"])
    }

    fn check(&self, extra: &[&str]) -> Run {
        let mut a = vec!["discharge", "check", "lean", "--no-lake"];
        a.extend_from_slice(extra);
        self.pv(&a)
    }

    /// The theorem file with `axiom pvl_mutation : False` planted inside its namespace (the spec's mutation).
    fn plant_axiom(&self) {
        let t = std::fs::read_to_string(self.path(THEOREM_FILE)).expect("read");
        self.write(
            THEOREM_FILE,
            &t.replace(
                "end ProvableContracts.Gelu",
                "axiom pvl_mutation : False\nend ProvableContracts.Gelu",
            ),
        );
    }

    fn allow(&self, entries: &str) {
        self.write("lean/escape-allowlist.yaml", entries);
    }
}

const MUTATION_ENTRY: &str = "- file: ProvableContracts/Theorems/Gelu/Bound.lean\n  decl: ProvableContracts.Gelu.pvl_mutation\n  kind: axiom\n";

fn assert_rc(r: &Run, code: i32, needle: &str) {
    assert_eq!(r.code, code, "{}", r.show());
    assert!(
        r.stdout.contains(needle) || r.stderr.contains(needle),
        "missing {needle:?}\n{}",
        r.show()
    );
}

#[test]
fn a_clean_tree_is_accepted_and_pins_the_bound_theorem_by_its_qualified_name() {
    let fx = Fx::new();
    assert_rc(&fx.gen(), 0, "1 root(s) bound");
    let axioms = std::fs::read_to_string(fx.path("lean/Axioms.lean")).expect("Axioms.lean");
    assert!(
        axioms.contains("run_cmd pvlAxiomsSubset `ProvableContracts.Gelu.gelu_bound pvlPinned"),
        "{axioms}"
    );
    assert!(
        axioms.contains("def pvlPinned : List Name := [`propext, `Classical.choice, `Quot.sound]"),
        "{axioms}"
    );
    let r = fx.check(&[]);
    assert_rc(&r, 0, "ok    discharge lean");
    assert!(
        r.stdout.contains("ROOTS 1 pinned, 0 ORPHANED-ROOT"),
        "{}",
        r.show()
    );
    assert!(
        !r.stdout.contains("ESCAPE"),
        "comment/string text scanned as an escape\n{}",
        r.show()
    );
}

#[test]
fn an_unlisted_axiom_is_red_by_name() {
    let fx = Fx::new();
    fx.gen();
    fx.plant_axiom();
    let r = fx.check(&[]);
    assert_rc(&r, 1, "ESCAPE ProvableContracts/Theorems/Gelu/Bound.lean:5 `axiom` in ProvableContracts.Gelu.pvl_mutation");
    assert!(r.stderr.starts_with("reject:"), "{}", r.show());
}

#[test]
fn implemented_by_on_a_def_is_red() {
    let fx = Fx::new();
    fx.gen();
    fx.append(THEOREM_FILE, "namespace ProvableContracts.Gelu\n@[implemented_by gImpl] def g : Nat := 1\nend ProvableContracts.Gelu\n");
    assert_rc(
        &fx.check(&[]),
        1,
        "`implemented_by` in ProvableContracts.Gelu.g -- not in escape-allowlist.yaml",
    );
}

#[test]
fn a_pending_entry_is_accepted_but_red_under_strict_and_joins_the_pinned_set() {
    let fx = Fx::new();
    fx.plant_axiom();
    fx.allow(&format!(
        "{MUTATION_ENTRY}  reason: drafted\n  ticket: \"#4139\"\n  confirmed_by: pending\n"
    ));
    fx.gen();
    let axioms = std::fs::read_to_string(fx.path("lean/Axioms.lean")).expect("Axioms.lean");
    assert!(
        axioms.contains("`Quot.sound, `ProvableContracts.Gelu.pvl_mutation]"),
        "{axioms}"
    );
    assert_rc(&fx.check(&[]), 0, "PENDING (1)");
    assert_rc(&fx.check(&["--strict"]), 1, "still confirmed_by: pending");
}

#[test]
fn an_entry_without_reason_or_ticket_exempts_nothing() {
    for (drop, needle) in [("reason", "has no reason"), ("ticket", "has no ticket")] {
        let fx = Fx::new();
        fx.plant_axiom();
        let reason = if drop == "reason" {
            ""
        } else {
            "  reason: drafted\n"
        };
        let ticket = if drop == "ticket" {
            ""
        } else {
            "  ticket: \"#4139\"\n"
        };
        fx.allow(&format!(
            "{MUTATION_ENTRY}{reason}{ticket}  confirmed_by: pending\n"
        ));
        fx.gen();
        assert_rc(&fx.check(&[]), 1, needle);
    }
}

#[test]
fn a_stale_entry_is_red() {
    let fx = Fx::new();
    fx.allow(&format!(
        "{MUTATION_ENTRY}  reason: gone\n  ticket: \"#4139\"\n  confirmed_by: pending\n"
    ));
    fx.gen();
    assert_rc(&fx.check(&[]), 1, "STALE escape-allowlist.yaml entry");
}

#[test]
fn an_exact_name_naming_nothing_or_an_axiom_is_missing_root() {
    let fx = Fx::new();
    fx.write(CONTRACT, "equations:\n  e:\n    lean_theorem: Theorems.Gelu\n  f:\n    lean_theorem: ProvableContracts.Gelu.no_such\n");
    fx.gen();
    assert_rc(
        &fx.check(&[]),
        1,
        "MISSING-ROOT contract gelu-v1: ProvableContracts.Gelu.no_such -- no such theorem",
    );
    fx.plant_axiom();
    fx.allow(&format!(
        "{MUTATION_ENTRY}  reason: r\n  ticket: \"#4139\"\n  confirmed_by: pending\n"
    ));
    fx.write(CONTRACT, "equations:\n  e:\n    lean_theorem: Theorems.Gelu\n  f:\n    lean_theorem: ProvableContracts.Gelu.pvl_mutation\n");
    fx.gen();
    assert_rc(
        &fx.check(&[]),
        1,
        "it names an `axiom`, not a proved theorem",
    );
}

#[test]
fn a_new_unresolved_label_fails_by_name() {
    let fx = Fx::new();
    fx.append(CONTRACT, "  f:\n    lean_theorem: Theorems.NoSuchThing\n");
    fx.gen();
    assert_rc(
        &fx.check(&[]),
        1,
        "NEW-UNRESOLVED-LABEL gelu-v1: Theorems.NoSuchThing",
    );
}

#[test]
fn check_never_writes_the_label_set_and_label_ratchet_only_shrinks_it() {
    let fx = Fx::new();
    let set = fx.path("lean/unresolved-labels.json");
    fx.append(CONTRACT, "  f:\n    lean_theorem: Theorems.NoSuchThing\n");
    fx.gen();
    assert_rc(
        &fx.check(&[]),
        1,
        "NEW-UNRESOLVED-LABEL gelu-v1: Theorems.NoSuchThing",
    );
    assert!(!set.exists(), "the gate wrote the label set");
    // no set yet: label-ratchet seeds it from what is measured, and check is then green
    assert_rc(
        &fx.pv(&["discharge", "label-ratchet", "lean"]),
        0,
        "(1 label(s))",
    );
    let seeded = std::fs::read_to_string(&set).expect("seeded");
    assert!(
        seeded.contains("\"label\": \"Theorems.NoSuchThing\""),
        "{seeded}"
    );
    assert!(
        seeded.contains("\"command\": \"make label-ratchet\""),
        "{seeded}"
    );
    assert_rc(&fx.check(&[]), 0, "UNRESOLVED-LABEL (1) (listed 1)");
    assert_eq!(
        std::fs::read_to_string(&set).expect("set"),
        seeded,
        "check rewrote the label set"
    );
    // the label is fixed: still listed is reported, not red; label-ratchet removes it
    fx.write(
        CONTRACT,
        "equations:\n  e:\n    lean_theorem: Theorems.Gelu\n",
    );
    assert_rc(&fx.check(&[]), 0, "RESOLVED-LABEL (1) still listed");
    assert_rc(
        &fx.pv(&["discharge", "label-ratchet", "lean"]),
        0,
        "(0 label(s))",
    );
    // a label that comes back is NOT re-admitted: the ratchet never rises
    fx.append(CONTRACT, "  f:\n    lean_theorem: Theorems.NoSuchThing\n");
    assert_rc(
        &fx.pv(&["discharge", "label-ratchet", "lean"]),
        1,
        "NEW-UNRESOLVED-LABEL gelu-v1: Theorems.NoSuchThing",
    );
    assert!(!std::fs::read_to_string(&set)
        .expect("set")
        .contains("NoSuchThing"));
}

#[test]
fn a_capstone_naming_no_theorem_is_missing_root() {
    let fx = Fx::new();
    fx.write(
        "lean/formalization.yaml",
        "capstones:\n  - ProvableContracts.Gelu.no_such\n",
    );
    fx.gen();
    assert_rc(
        &fx.check(&[]),
        1,
        "MISSING-ROOT capstone ProvableContracts.Gelu.no_such",
    );
    fx.write(
        "lean/formalization.yaml",
        "capstones:\n  - ProvableContracts.Gelu.gelu_bound\n",
    );
    fx.gen();
    let axioms = std::fs::read_to_string(fx.path("lean/Axioms.lean")).expect("Axioms.lean");
    assert!(
        axioms.contains("#guard_msgs in #print axioms ProvableContracts.Gelu.gelu_bound"),
        "{axioms}"
    );
    assert_rc(&fx.check(&[]), 0, "ok    discharge");
}

#[test]
fn a_stale_axioms_file_is_red_and_gen_axioms_check_agrees() {
    let fx = Fx::new();
    fx.gen();
    fx.append("lean/Axioms.lean", "-- hand edit\n");
    assert_rc(&fx.check(&[]), 1, "STALE Axioms.lean");
    assert_rc(
        &fx.pv(&["discharge", "gen-axioms", "lean", "--check"]),
        1,
        "differs from its regeneration",
    );
    fx.gen();
    assert_rc(
        &fx.pv(&["discharge", "gen-axioms", "lean", "--check"]),
        0,
        "is its regeneration",
    );
}

#[test]
fn zero_roots_declines_but_a_failure_outranks_the_decline() {
    let fx = Fx::new();
    fx.write(
        CONTRACT,
        "equations:\n  e:\n    lean_theorem: Theorems.NoSuchThing\n",
    );
    fx.gen();
    fx.pv(&["discharge", "label-ratchet", "lean"]);
    let r = fx.check(&[]);
    assert_rc(
        &r,
        2,
        "decline: 0 contract-bound theorems in the root's import cone",
    );
    std::fs::remove_file(fx.path("lean/Axioms.lean")).expect("rm");
    assert_rc(&fx.check(&[]), 1, "no Axioms.lean");
}

#[test]
fn a_bound_theorem_outside_the_roots_import_cone_is_orphaned_not_pinned() {
    let fx = Fx::new();
    fx.write("lean/ProvableContracts.lean", "-- imports nothing\n");
    fx.gen();
    let axioms = std::fs::read_to_string(fx.path("lean/Axioms.lean")).expect("Axioms.lean");
    assert!(!axioms.contains("gelu_bound"), "{axioms}");
    assert_rc(&fx.check(&[]), 2, "decline: 0 contract-bound theorems");
}

#[test]
fn no_root_file_declines() {
    let fx = Fx::new();
    std::fs::remove_file(fx.path("lean/ProvableContracts.lean")).expect("rm");
    assert_rc(&fx.check(&[]), 2, "decline: no ProvableContracts.lean");
}

/// PVL-001 EV-6a's probe, on the REAL tree: the escapes are all allowlisted, Axioms.lean is its regeneration, no
/// new unresolved label, no MISSING-ROOT — and `check` leaves the label set's bytes alone. Reading the tree is
/// also what puts this target in scripts/tree_reader_tests.txt, i.e. in CI's quick tier, with no workflow edit.
#[test]
fn the_real_lean_tree_passes_the_ev_6a_probe() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let lean = "crates/aprender-contracts-staging/lean";
    let labels = root.join(lean).join("unresolved-labels.json");
    let before = std::fs::read(&labels).expect("the label set is tracked");
    let run = |args: &[&str]| {
        let out = Command::new(pv_bin())
            .current_dir(&root)
            .args(args)
            .output()
            .expect("spawn pv");
        Run {
            code: out.status.code().unwrap_or(-1),
            stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
        }
    };
    assert_rc(
        &run(&["discharge", "gen-axioms", lean, "--check"]),
        0,
        "is its regeneration",
    );
    let r = run(&["discharge", "check", lean, "--no-lake"]);
    assert_rc(&r, 0, "PENDING (7)");
    assert!(r.stdout.contains("ok    discharge"), "{}", r.show());
    // PVL-001 EV-8b probe: the tracked formalization.yaml is consistent with the tree and the summary
    assert_rc(
        &run(&[
            "discharge",
            "check",
            lean,
            "--no-lake",
            "--validate-formalization",
        ]),
        0,
        "FORMALIZATION ok",
    );
    assert_eq!(
        std::fs::read(&labels).expect("label set"),
        before,
        "check wrote the label set"
    );
}

/// PVL-001 EV-8b (#4082): `--validate-formalization` rejects a missing or inconsistent formalization.yaml, and
/// accepts one that agrees with the tree, Axioms.lean and discharge-summary.json.
#[test]
fn validate_formalization_rejects_inconsistency_and_accepts_agreement() {
    let fx = Fx::new();
    let form = "main_results: [ProvableContracts.Gelu.gelu_bound]\nstatus:\n  axioms: [propext, Classical.choice, Quot.sound]\n\
capstones: []\nsorry_count: 0\nscope: all\nreview:\n  status: self-assessed\nautomation:\n  methods: [manual]\n";
    fx.write("lean/formalization.yaml", form);
    fx.write(
        "discharge-summary.json",
        r#"{"tree_sha":"t","toolchain":null,"mathlib_rev":null,"build_exit":0,"lake_exit":0,"leanchecker_exit":0,
"axioms_ok":true,"escapes_ok":true,"challenges_closed":"1/1","modules":[{"path":"ProvableContracts/Theorems/Gelu/Bound.lean",
"blake3":"b","theorems":["ProvableContracts.Gelu.gelu_bound"]}]}"#,
    );
    assert_rc(&fx.gen(), 0, "");
    assert_rc(
        &fx.check(&["--validate-formalization"]),
        0,
        "FORMALIZATION ok",
    );
    assert_rc(&fx.check(&[]), 0, "");
    fx.write(
        "lean/formalization.yaml",
        &form.replace("sorry_count: 0", "sorry_count: 2"),
    );
    assert_rc(
        &fx.check(&["--validate-formalization"]),
        1,
        "FAIL formalization: sorry_count is Some(2); the escape scan measures 0",
    );
    std::fs::remove_file(fx.path("discharge-summary.json")).expect("rm");
    fx.write("lean/formalization.yaml", form);
    assert_rc(
        &fx.check(&["--validate-formalization"]),
        1,
        "no discharge summary",
    );
}
