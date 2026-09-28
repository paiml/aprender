//! PVL-001 EV-7a (#4200) — `pv challenge gen | check`, end to end on a built-in-a-tempdir Lean tree.
//!
//! The RED rows carry the spec's mutation: a hand-edited statement in `Challenge/`. The Lean elaboration of the
//! generated files is not run here (CI has no Mathlib); it was measured on lambda (see #4200).

use std::path::PathBuf;
use std::process::Command;

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

const CHALLENGE: &str = "lean/Challenge/gelu-v1.lean";

struct Fx {
    dir: tempfile::TempDir,
}

impl Fx {
    /// `gelu_bound` is bound by exact name; `unbound` by nothing.
    fn new() -> Self {
        let fx = Self {
            dir: tempfile::tempdir().expect("tempdir"),
        };
        fx.write(
            "lean/ProvableContracts.lean",
            "import ProvableContracts.Theorems.Gelu.Bound\n",
        );
        fx.write(
            "lean/ProvableContracts/Theorems/Gelu/Bound.lean",
            "namespace ProvableContracts.Gelu\n\
             theorem gelu_bound (x : Nat) :\n    x ≤ x + 1 := Nat.le_succ x\n\
             theorem unbound : True := trivial\n\
             end ProvableContracts.Gelu\n",
        );
        fx.write(
            "contracts/gelu-v1.yaml",
            "equations:\n  e:\n    lean_theorem: ProvableContracts.Gelu.gelu_bound\n",
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

    fn read(&self, rel: &str) -> String {
        std::fs::read_to_string(self.path(rel)).expect("read")
    }

    fn pv(&self, action: &str) -> Run {
        let out = Command::new(env!("CARGO_BIN_EXE_pv"))
            .current_dir(self.dir.path())
            .args(["challenge", action, "contracts", "lean"])
            .output()
            .expect("spawn pv");
        Run {
            code: out.status.code().unwrap_or(-1),
            stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
        }
    }

    fn generated(&self) {
        let g = self.pv("gen");
        assert_eq!(g.code, 0, "{}", g.show());
    }
}

#[test]
fn gen_writes_the_statement_with_sorry_and_check_is_then_fresh() {
    let fx = Fx::new();
    fx.generated();
    let text = fx.read(CHALLENGE);
    assert!(
        text.contains(
            "theorem _root_.PvlChallenge.ProvableContracts.Gelu.gelu_bound (x : Nat) : x ≤ x + 1 := sorry\n"
        ),
        "{text}"
    );
    assert!(
        text.contains("namespace ProvableContracts.Gelu\n"),
        "{text}"
    );
    assert!(!text.contains("unbound"), "{text}");
    let c = fx.pv("check");
    assert_eq!(c.code, 0, "{}", c.show());
    assert!(
        c.stdout.contains("challenge-fresh: 1 file(s)"),
        "{}",
        c.show()
    );
}

#[test]
fn gen_is_deterministic() {
    let fx = Fx::new();
    fx.generated();
    let first = fx.read(CHALLENGE);
    fx.generated();
    assert_eq!(fx.read(CHALLENGE), first);
}

/// EV-7a's mutation: hand-edit a statement → RED, naming the file and the line.
#[test]
fn a_hand_edited_statement_is_red() {
    let fx = Fx::new();
    fx.generated();
    let text = fx.read(CHALLENGE);
    fx.write(CHALLENGE, &text.replace("x ≤ x + 1", "x ≤ x + 2"));
    let c = fx.pv("check");
    assert_eq!(c.code, 1, "{}", c.show());
    assert!(
        c.stdout
            .contains("FAIL STALE differs  Challenge/gelu-v1.lean:"),
        "{}",
        c.show()
    );
}

/// The solution's statement changing (a weakened theorem) leaves the committed challenge stale → RED.
#[test]
fn a_changed_solution_statement_is_red_until_regenerated() {
    let fx = Fx::new();
    fx.generated();
    let src = "lean/ProvableContracts/Theorems/Gelu/Bound.lean";
    let text = fx.read(src);
    fx.write(src, &text.replace("x ≤ x + 1", "0 ≤ x"));
    assert_eq!(fx.pv("check").code, 1);
    fx.generated();
    assert_eq!(fx.pv("check").code, 0);
}

#[test]
fn an_extra_or_missing_challenge_file_is_red() {
    let fx = Fx::new();
    fx.generated();
    fx.write("lean/Challenge/Sub/stray.lean", "");
    let c = fx.pv("check");
    assert_eq!(c.code, 1, "{}", c.show());
    assert!(
        c.stdout.contains("extra    Challenge/Sub/stray.lean"),
        "{}",
        c.show()
    );
    fx.generated();
    assert!(!fx.path("lean/Challenge/Sub/stray.lean").exists());
    std::fs::remove_file(fx.path(CHALLENGE)).expect("rm");
    let c = fx.pv("check");
    assert_eq!(c.code, 1, "{}", c.show());
    assert!(
        c.stdout.contains("missing  Challenge/gelu-v1.lean"),
        "{}",
        c.show()
    );
}

/// Zero challenges is a decline (rc 2), never "fresh": nothing is pinned.
#[test]
fn zero_challenges_declines() {
    let fx = Fx::new();
    fx.write(
        "contracts/gelu-v1.yaml",
        "equations:\n  e:\n    formula: x\n",
    );
    for action in ["gen", "check"] {
        let r = fx.pv(action);
        assert_eq!(r.code, 2, "{action}: {}", r.show());
        assert!(
            r.stderr.contains("zero challenges"),
            "{action}: {}",
            r.show()
        );
    }
}

#[test]
fn an_unloadable_tree_declines() {
    let fx = Fx::new();
    std::fs::remove_file(fx.path("lean/ProvableContracts.lean")).expect("rm");
    assert_eq!(fx.pv("check").code, 2);
}
