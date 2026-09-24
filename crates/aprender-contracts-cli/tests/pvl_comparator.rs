//! PVL-001 EV-7b (#4201) — `pv discharge check --comparator`, end to end with a stub `lake` on PATH.
//!
//! The spec's RED: a solution that proves a WEAKER statement than its Challenge pins → rc 1 (MISMATCH), and a
//! solution on `sorry` → rc 1. No Challenge file, or zero rows → rc 2, never a pass. The stub replays rows the
//! committed `scripts/Comparator.lean` produced on a real Lean tree; the real run needs the built tree and
//! Mathlib, and is measured on lambda (see #4201). The last test pins the committed script to the row shape the
//! parser reads, so the two cannot drift apart silently.

use std::path::{Path, PathBuf};
use std::process::Command;

use provable_contracts::discharge::comparator::{CHALLENGE_NS, COMPARATOR, SORRY_AXIOM};

fn pv_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_pv"))
}

/// sha256 hashes `scripts/Comparator.lean` printed for `gelu_pos (x : Nat) (h : 0 < x) : 0 < g x`, and for the
/// same theorem with its hypothesis strengthened to `1 < x` (measured with Lean v4.29.0-rc4, 2026-09-24).
const PINNED: &str = "902d0aa1835a8af24db87013bf140f155ab57b9002d3665f37294f2a3623b6fd";
const WEAKER: &str = "66afc7f44f64333e7cc3a4c495e91ce8d3e376d8e813accc9542f2a9f5397224";
const NAME: &str = "ProvableContracts.Gelu.gelu_pos";

struct Fx {
    dir: tempfile::TempDir,
}

impl Fx {
    /// A one-theorem tree with `Axioms.lean` generated, a Challenge file and the comparator script in place.
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
            "namespace ProvableContracts.Gelu\ntheorem gelu_bound : True := trivial\nend ProvableContracts.Gelu\n",
        );
        fx.write(
            "contracts/gelu-v1.yaml",
            "equations:\n  e:\n    lean_theorem: Theorems.Gelu\n",
        );
        fx.write("lean/Challenge/gelu-v1.lean", "-- EV-7a writes this\n");
        fx.write(
            &format!("lean/{COMPARATOR}"),
            "-- the real one is committed\n",
        );
        let g = fx.pv(&["discharge", "gen-axioms", "lean"]);
        assert_eq!(g.0, 0, "gen-axioms: {}", g.1);
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

    /// `bin/lake` on the test's PATH: `env lean --run <script> <files>` prints `rows` and exits `rc`; every other
    /// `env lean` (Axioms.lean) passes.
    fn stub_lake(&self, rc: i32, rows: &str) {
        self.write("rows.ndjson", rows);
        self.write(
            "bin/lake",
            &format!(
                "#!/bin/sh\nif [ \"$3\" = --run ]; then cat '{}'; echo 'stub comparator rc {rc}' >&2; exit {rc}; fi\nexit 0\n",
                self.path("rows.ndjson").display()
            ),
        );
        let p = self.path("bin/lake");
        let mut perm = std::fs::metadata(&p).expect("meta").permissions();
        std::os::unix::fs::PermissionsExt::set_mode(&mut perm, 0o755);
        std::fs::set_permissions(&p, perm).expect("chmod");
    }

    /// Runs pv with PATH = the fixture's `bin/` (the stub lake, if any), then the system dirs — no real `lake`.
    fn pv(&self, args: &[&str]) -> (i32, String) {
        let path = format!("{}:/usr/bin:/bin", self.path("bin").display());
        let out = Command::new(pv_bin())
            .current_dir(self.dir.path())
            .env("PATH", path)
            .args(args)
            .output()
            .expect("spawn pv");
        (
            out.status.code().unwrap_or(-1),
            format!(
                "{}{}",
                String::from_utf8_lossy(&out.stdout),
                String::from_utf8_lossy(&out.stderr)
            ),
        )
    }

    fn check(&self) -> (i32, String) {
        self.pv(&["discharge", "check", "lean", "--comparator"])
    }
}

fn row(sol: &str, axioms: &str) -> String {
    format!(
        "{{\"name\": \"{NAME}\", \"challenge_type_hash\": \"{PINNED}\", \"solution_type_hash\": {sol}, \"axioms\": {axioms}}}\n"
    )
}

fn assert_rc(r: &(i32, String), code: i32, needle: &str) {
    assert_eq!(r.0, code, "{}", r.1);
    assert!(r.1.contains(needle), "missing {needle:?}\n{}", r.1);
}

#[test]
fn a_solution_of_the_pinned_statement_accepts() {
    let fx = Fx::new();
    fx.stub_lake(0, &row(&format!("\"{PINNED}\""), "[\"propext\"]"));
    assert_rc(&fx.check(), 0, "COMPARATOR 1/1 challenge(s) closed");
}

#[test]
fn a_solution_of_a_weaker_statement_rejects() {
    let fx = Fx::new();
    fx.stub_lake(0, &row(&format!("\"{WEAKER}\""), "[]"));
    let r = fx.check();
    assert_rc(&r, 1, &format!("FAIL  MISMATCH {NAME}"));
    assert!(r.1.contains(WEAKER) && r.1.contains(PINNED), "{}", r.1);
}

#[test]
fn a_sorry_solution_rejects() {
    let fx = Fx::new();
    fx.stub_lake(0, &row(&format!("\"{PINNED}\""), "[\"sorryAx\"]"));
    assert_rc(&fx.check(), 1, &format!("FAIL  SORRY {NAME}"));
}

#[test]
fn a_missing_solution_rejects() {
    let fx = Fx::new();
    fx.stub_lake(0, &row("null", "null"));
    assert_rc(&fx.check(), 1, &format!("FAIL  MISSING-ROOT {NAME}"));
}

#[test]
fn a_challenge_that_does_not_elaborate_rejects() {
    let fx = Fx::new();
    fx.stub_lake(1, "");
    let r = fx.check();
    assert_rc(&r, 1, "did not elaborate");
    assert!(r.1.contains("stub comparator rc 1"), "{}", r.1);
}

#[test]
fn zero_rows_declines() {
    let fx = Fx::new();
    fx.stub_lake(0, "");
    assert_rc(&fx.check(), 2, "0 challenge rows");
}

#[test]
fn no_challenge_file_declines() {
    let fx = Fx::new();
    fx.stub_lake(0, &row(&format!("\"{PINNED}\""), "[]"));
    std::fs::remove_dir_all(fx.path("lean/Challenge")).expect("rm");
    assert_rc(&fx.check(), 2, "no Challenge/*.lean");
}

#[test]
fn no_lake_declines() {
    let fx = Fx::new();
    assert_rc(&fx.check(), 2, "decline:");
}

#[test]
fn comparator_and_no_lake_conflict() {
    let fx = Fx::new();
    let r = fx.pv(&["discharge", "check", "lean", "--no-lake", "--comparator"]);
    assert_eq!(r.0, 2, "clap refuses the pair: {}", r.1);
    assert!(r.1.contains("cannot be used with"), "{}", r.1);
}

/// The committed script emits exactly the fields `Row` parses, under the namespace and sorry axiom the judge
/// uses, hashes with sha256 (cop ruling on #4201: not a 64-bit hash) and carries the FIPS self-test.
#[test]
fn the_committed_comparator_script_matches_the_row_shape() {
    let f = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../aprender-contracts-staging/lean")
        .join(COMPARATOR);
    let text = std::fs::read_to_string(&f).expect("scripts/Comparator.lean is committed");
    for needle in [
        "\\\"name\\\": ",
        "\\\"challenge_type_hash\\\": ",
        "\\\"solution_type_hash\\\": ",
        "\\\"axioms\\\": ",
        &format!("`{CHALLENGE_NS}"),
        "def sha256 ",
        "--self-test",
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "autoImplicit",
    ] {
        assert!(text.contains(needle), "Comparator.lean lacks {needle:?}");
    }
    assert!(
        text.contains("def typeHash (e : Expr) : String := sha256 (canon e).toUTF8")
            && !text.contains("hash (canon"),
        "the statement hash must be sha256 over canon, not Lean's 64-bit hash"
    );
    assert_eq!(SORRY_AXIOM, "sorryAx");
}
