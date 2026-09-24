//! PVL-001 EV-10 (paiml/aprender#4198) — `pv obligations --gate` replaces pmat's
//! `scripts/pv-obligation-gate.py`.
//!
//! GOLDEN: `tests/fixtures/pvl/obligations/<fixture>.stdout` and `.rc` were recorded by running
//! the Python script itself over each fixture (the command is in that directory's README). This
//! file holds `pv obligations --gate` to the same stdout, byte for byte, and the same exit code:
//!
//! - `pmat/` — pmat's 35 contracts, verbatim, and the 10 `src/` files their `applies_to` names
//!   resolve to: GREEN, `0 problem(s) over 35 contracts`, exit 0.
//! - `broken/` — one planted defect per check, each on its own output line, beside a control
//!   for every rule that must NOT fire: RED, 6 problems, exit 1.
//!
//! DISCRIMINATION: the two goldens differ, so a pv that printed either for both fails one; the
//! `pmat` denominator is counted from the fixture, not trusted from the golden; and without
//! `--gate` the same report exits 0, so exit 1 is the flag's doing and not a crash.

use std::path::{Path, PathBuf};
use std::process::Command;

fn pv_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_pv"))
}

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/pvl/obligations")
}

struct Run {
    code: i32,
    stdout: String,
    stderr: String,
}

/// `pv obligations <root> [--gate]` from a scratch cwd, so nothing resolves against the cwd.
fn obligations(root: &Path, gate: bool) -> Run {
    let scratch = tempfile::tempdir().expect("scratch cwd is creatable");
    let mut cmd = Command::new(pv_bin());
    cmd.arg("obligations").arg(root).current_dir(scratch.path());
    if gate {
        cmd.arg("--gate");
    }
    let out = cmd.output().expect("pv runs");
    Run {
        code: out.status.code().expect("pv exits, not signalled"),
        stdout: String::from_utf8(out.stdout).expect("utf-8 stdout"),
        stderr: String::from_utf8(out.stderr).expect("utf-8 stderr"),
    }
}

fn golden(name: &str) -> (String, i32) {
    let dir = fixtures();
    let stdout =
        std::fs::read_to_string(dir.join(format!("{name}.stdout"))).expect("golden stdout");
    let rc = std::fs::read_to_string(dir.join(format!("{name}.rc"))).expect("golden rc");
    (stdout, rc.trim().parse().expect("golden rc is an integer"))
}

fn assert_matches_script(name: &str) -> Run {
    let (want_stdout, want_rc) = golden(name);
    let run = obligations(&fixtures().join(name), true);
    assert_eq!(
        run.stdout, want_stdout,
        "{name}: stdout differs from the script's\nstderr: {}",
        run.stderr
    );
    assert_eq!(
        run.code, want_rc,
        "{name}: exit code differs from the script's\nstderr: {}",
        run.stderr
    );
    run
}

#[test]
fn pmat_corpus_is_green_exactly_as_the_script_reports_it() {
    let run = assert_matches_script("pmat");
    assert_eq!(run.code, 0);
    assert!(run.stderr.is_empty(), "{}", run.stderr);
}

#[test]
fn pmat_golden_counts_every_contract_in_the_fixture() {
    let n = std::fs::read_dir(fixtures().join("pmat/contracts"))
        .expect("pmat/contracts")
        .filter_map(Result::ok)
        .filter_map(|e| e.file_name().into_string().ok())
        .filter(|n| n.ends_with(".yaml") && !n.ends_with("binding.yaml"))
        .count();
    assert_eq!(n, 35, "the fixture is pmat's 35 contracts");
    let (stdout, _) = golden("pmat");
    assert_eq!(
        stdout,
        format!("pv obligation gate: 0 problem(s) over {n} contracts\n")
    );
}

#[test]
fn broken_fixture_is_red_exactly_as_the_script_reports_it() {
    let run = assert_matches_script("broken");
    assert_eq!(run.code, 1);
    assert_eq!(
        run.stderr,
        "reject: obligation gate failed (6 problem(s) over 4 contracts)\n"
    );
}

#[test]
fn broken_fixture_names_each_planted_defect_once() {
    let (stdout, _) = golden("broken");
    for (needle, why) in [
        ("contracts/a-invalid-v1.yaml: pv validate failed", "check 1"),
        (
            "contracts/b-hidden-v1.yaml: 2 test-bearing obligation(s)",
            "check 2 (the action-only entry is not counted)",
        ),
        ("applies_to 'ghost_fn_xyz' names neither", "check 3a"),
        (
            "applies_to 'bounded' names neither",
            "check 3a: `fn bounded_v2` is not `fn bounded`",
        ),
        (
            "applies_to 'grade_gate' is proved against 'Grade', but none of ['src/gate.rs.txt']",
            "check 3b: GradeTable is not Grade",
        ),
        (
            "applies_to 'grade_prefixed' is proved against 'Grade', but none of ['src/prefixed.rs.txt']",
            "check 3b: TdgGrade is not Grade",
        ),
    ] {
        assert_eq!(
            stdout.matches(needle).count(),
            1,
            "{why}: {needle}\n{stdout}"
        );
    }
    for control in [
        "real_fn",
        "grade_ok",
        "'relu'",
        "'all'",
        "z-binding",
        "nested",
        ".hidden-v1",
        "notes.yml",
    ] {
        assert!(
            !stdout.contains(control),
            "control {control} must not be reported:\n{stdout}"
        );
    }
}

#[test]
fn the_two_goldens_discriminate() {
    assert_ne!(golden("pmat"), golden("broken"));
}

#[test]
fn without_gate_the_same_report_exits_zero() {
    let (want, _) = golden("broken");
    let run = obligations(&fixtures().join("broken"), false);
    assert_eq!(run.stdout, want);
    assert_eq!(run.code, 0, "{}", run.stderr);
}

#[test]
fn zero_contracts_is_a_decline_not_a_pass() {
    let empty = tempfile::tempdir().expect("tempdir");
    let run = obligations(empty.path(), true);
    assert_eq!(run.code, 2, "{}", run.stderr);
    assert!(
        run.stderr.starts_with("decline: 0 contracts under "),
        "{}",
        run.stderr
    );
    assert!(run.stdout.is_empty(), "{}", run.stdout);
}
