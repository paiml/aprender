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
/// same theorem with its hypothesis strengthened to `1 < x` (measured with Lean v4.29.0-rc4, 2026-09-24;
/// re-measured 2026-09-25 after #4241's canon change, the old script reproducing the old pair on the same file).
const PINNED: &str = "2f264bb0afcc74e21bfc40220d9dbca9cbf2d2b34fdd5729aa3a74d3a36ce1f2";
const WEAKER: &str = "2e6864e79530a3bd6bd657ba470b62ace36875c8cc65ef9c584a62297967862c";
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
        fx.write(
            "lean/Challenge/gelu-v1.lean",
            &format!("-- EV-7a writes this\ntheorem _root_.{CHALLENGE_NS}.{NAME} (x : Nat) (h : 0 < x) : 0 < g x := sorry\n"),
        );
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

    /// `bin/lake` on the test's PATH: `env lean --run <script> --self-test` passes the 3 FIPS vectors (the
    /// real script's own check; pv runs it before trusting a hash), `env lean --run <script> <files>` prints
    /// `rows` and exits `rc`; every other `env lean` (Axioms.lean) passes.
    fn stub_lake(&self, rc: i32, rows: &str) {
        self.write("rows.ndjson", rows);
        self.write(
            "bin/lake",
            &format!(
                "#!/bin/sh\nif [ \"$5\" = --self-test ]; then printf 'ok    sha256 \"\" = e3\\nok    sha256 \"abc\" = ba\\nok    sha256 \"abcdbcde\" = 24\\n'; exit 0; fi\nif [ \"$3\" = --run ]; then cat '{}'; echo 'stub comparator rc {rc}' >&2; exit {rc}; fi\nexit 0\n",
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

/// #4237 (cop ruling): differing hashes close iff the comparator measured the types defeq at `.instances`.
#[test]
fn differing_hashes_close_only_when_defeq_at_instances() {
    for (d, rc, needle) in [
        ("true", 0, format!("MATCH(instances) {NAME}")),
        ("false", 1, format!("FAIL  MISMATCH {NAME}")),
    ] {
        let fx = Fx::new();
        let r = row(&format!("\"{WEAKER}\""), "[\"propext\"]").replace(
            ", \"axioms\"",
            &format!(", \"defeq_instances\": {d}, \"axioms\""),
        );
        fx.stub_lake(0, &r);
        assert_rc(&fx.check(), rc, &needle);
    }
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

/// Zero rows AND zero declared roots: nothing was compared, a decline.
#[test]
fn zero_rows_declines() {
    let fx = Fx::new();
    fx.write("lean/Challenge/gelu-v1.lean", "-- EV-7a writes this\n");
    fx.stub_lake(0, "");
    assert_rc(&fx.check(), 2, "0 challenge rows");
}

/// #4240: `rowsOf` drops an `isInternal` root and it emits no row — the declared root is counted, rc 1.
#[test]
fn a_declared_root_with_no_row_rejects_as_missing_row() {
    let fx = Fx::new();
    let dropped = "ProvableContracts.Gelu._private_lemma";
    fx.write(
        "lean/Challenge/gelu-v1.lean",
        &format!(
            "theorem _root_.{CHALLENGE_NS}.{NAME} (x : Nat) (h : 0 < x) : 0 < g x := sorry\n\
             theorem _root_.{CHALLENGE_NS}.{dropped} : True := sorry\n"
        ),
    );
    fx.stub_lake(0, &row(&format!("\"{PINNED}\""), "[\"propext\"]"));
    let r = fx.check();
    assert_rc(&r, 1, &format!("FAIL  MISSING-ROW {dropped}"));
    assert!(
        r.1.contains("COMPARATOR 1/1"),
        "judge_rows still sees one row: {}",
        r.1
    );
    assert!(r.1.contains("1 row(s), 2 declared root(s)"), "{}", r.1);
}

/// Zero rows against a declared root is a missing row (rc 1), never the vacuity decline.
#[test]
fn zero_rows_against_a_declared_root_rejects() {
    let fx = Fx::new();
    fx.stub_lake(0, "");
    assert_rc(&fx.check(), 1, &format!("FAIL  MISSING-ROW {NAME}"));
}

#[test]
fn a_row_no_challenge_file_declares_rejects() {
    let fx = Fx::new();
    let ghost = row(&format!("\"{PINNED}\""), "[]").replace(NAME, "ProvableContracts.Gelu.ghost");
    fx.stub_lake(
        0,
        &format!("{}{ghost}", row(&format!("\"{PINNED}\""), "[]")),
    );
    assert_rc(
        &fx.check(),
        1,
        "FAIL  UNEXPECTED-ROW ProvableContracts.Gelu.ghost",
    );
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
        "\\\"defeq_instances\\\": ",
        "withTransparency .instances (Meta.isDefEq a b)",
        "if c == s then pure none else some <$> defeqInstances",
        "(`inst_a, `hyp, false), (`inst_a, `rhs, false),",
        "(`inst_a, `dflt, false)] do",
        "is not in the environment",
        "\\\"axioms\\\": ",
        &format!("`{CHALLENGE_NS}"),
        "def sha256 ",
        "--self-test",
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "autoImplicit",
        // #4241: names length-prefixed, universe params by position; both pinned by self-test controls.
        "def canonName (n : Name) : String",
        "| some i => s!\"u{i}\"",
        "for (a, b, want) in [(`ua, `ub, true), (`uc, `ud, false)] do",
        "bad := bad + (← selfTestCanon)",
    ] {
        assert!(text.contains(needle), "Comparator.lean lacks {needle:?}");
    }
    assert!(
        text.contains(
            "def typeHash (ci : ConstantInfo) : String := sha256 (canon ci.levelParams ci.type).toUTF8"
        )
            && !text.contains("hash (canon"),
        "the statement hash must be sha256 over canon, not Lean's 64-bit hash"
    );
    // A failed import is the root cause and `processCommands` drops the header's messages: measured on the
    // lambda run of EV-7a's 39 files, 19 withheld files printed only `Unknown identifier ℝ`, never the missing
    // `.olean`. The header's errors are printed and the file withheld before any command runs.
    assert!(
        text.contains("let hdr := messages.toList.filter (·.severity == .error)"),
        "Comparator.lean must print processHeader's errors (a missing .olean) and withhold the file"
    );
    assert_eq!(SORRY_AXIOM, "sorryAx");
}
