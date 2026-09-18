//! ONT-6 (PMAT-3451) — `pv lint` reports into the one verdict lattice, arms per repo, and maps its exit.
//!
//! ONT-001 v4.3 §5 ONT-6 RED, verbatim: "`armed_gates` shorter than committed → exit 3 `error: armed_gates
//! shrank`; unarmed gate → `Unknown{NotArmed}`; `Unknown` → exit 2 + `decline:`". Operator rulings
//! (2026-09-17): the default armed set is the 8 gates minus `reverse-coverage`; `Fail` prints `reject:` at
//! exit 1; only the shrink error exits 3. ONT R-2: an explicitly empty armed set declines.
//!
//! DISCRIMINATION: the control corpus (one real contract, `softmax-kernel-v1.yaml`, the PVL-1 control) must
//! PASS at exit 0 with the armed meet printed — so a build that exits 2 or 3 for everything fails it.

use std::path::{Path, PathBuf};
use std::process::Command;

fn pv_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_pv"))
}

struct Run {
    code: i32,
    stdout: String,
    stderr: String,
}

/// Run `pv` from a scratch cwd: `pv lint` writes `.pv/` into the current directory.
fn pv(args: &[&str]) -> Run {
    let scratch = tempfile::tempdir().expect("scratch cwd is creatable");
    let out = Command::new(pv_bin())
        .current_dir(scratch.path())
        .args(args)
        .output()
        .expect("failed to spawn pv");
    Run {
        code: out.status.code().unwrap_or(-1),
        stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
    }
}

fn show(r: &Run) -> String {
    format!(
        "exit {}\n--- stdout\n{}\n--- stderr\n{}",
        r.code, r.stdout, r.stderr
    )
}

fn repo_contracts() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../contracts")
}

/// A corpus directory holding the PVL-1 control contract, and optionally a `lint-baseline.json`.
fn corpus(baseline: Option<&str>) -> tempfile::TempDir {
    let d = tempfile::tempdir().expect("corpus dir is creatable");
    std::fs::copy(
        repo_contracts().join("softmax-kernel-v1.yaml"),
        d.path().join("softmax-kernel-v1.yaml"),
    )
    .expect("control contract copies");
    if let Some(b) = baseline {
        std::fs::write(d.path().join("lint-baseline.json"), b).expect("baseline writes");
    }
    d
}

fn s(p: &Path) -> &str {
    p.to_str().expect("utf-8 path")
}

const EIGHT: [&str; 8] = [
    "validate",
    "audit",
    "score",
    "verify",
    "enforce",
    "enforcement-level",
    "duplicate-stems",
    "composition",
];

fn baseline_json(names: &[&str]) -> String {
    let list = names
        .iter()
        .map(|n| format!("\"{n}\""))
        .collect::<Vec<_>>()
        .join(", ");
    format!("{{\n  \"armed_gates\": [{list}]\n}}\n")
}

#[test]
fn control_default_arming_passes_and_prints_the_armed_meet() {
    let d = corpus(None);
    let r = pv(&["lint", s(d.path())]);
    assert_eq!(r.code, 0, "{}", show(&r));
    assert!(r.stdout.contains("armed meet: Pass"), "{}", show(&r));
    assert!(
        r.stdout
            .contains("armed_gates monotone: NOT CHECKED (no comparand)"),
        "a corpus outside any git repository has no comparand, and says so\n{}",
        show(&r)
    );
}

#[test]
fn explicit_empty_armed_set_declines_r2() {
    let d = corpus(Some("{\n  \"armed_gates\": []\n}\n"));
    let r = pv(&["lint", s(d.path())]);
    assert_eq!(r.code, 2, "{}", show(&r));
    assert!(r.stderr.contains("decline: NotArmed"), "{}", show(&r));
}

#[test]
fn a_failing_armed_gate_rejects_at_exit_1() {
    let d = tempfile::tempdir().expect("corpus dir");
    std::fs::copy(
        repo_contracts().join("softmax-kernel-v1.yaml"),
        d.path().join("softmax-kernel-v1.yaml"),
    )
    .expect("control copies");
    std::fs::write(
        d.path().join("broken-v1.yaml"),
        "metadata:\n  version: not-a-contract\n",
    )
    .expect("fixture");
    let r = pv(&["lint", s(d.path())]);
    assert_eq!(r.code, 1, "{}", show(&r));
    assert!(
        r.stderr.lines().any(|l| l.starts_with("reject: ")),
        "a measured failure is `reject:`, not `error:`\n{}",
        show(&r)
    );
}

#[test]
fn dropping_a_committed_armed_gate_is_exit_3() {
    let repo = tempfile::tempdir().expect("repo dir");
    let contracts = repo.path().join("contracts");
    std::fs::create_dir_all(&contracts).expect("contracts dir");
    std::fs::copy(
        repo_contracts().join("softmax-kernel-v1.yaml"),
        contracts.join("softmax-kernel-v1.yaml"),
    )
    .expect("control copies");
    std::fs::write(contracts.join("lint-baseline.json"), baseline_json(&EIGHT)).expect("baseline");
    git_init_commit(repo.path());

    // Positive control: the committed eight, unchanged, pass the monotone check against HEAD.
    let same = pv(&["lint", s(&contracts), "--armed-baseline-ref", "HEAD"]);
    assert_eq!(same.code, 0, "{}", show(&same));
    assert!(
        same.stdout
            .contains("armed_gates monotone: OK against HEAD"),
        "{}",
        show(&same)
    );

    std::fs::write(
        contracts.join("lint-baseline.json"),
        baseline_json(&EIGHT[..7]),
    )
    .expect("shrunk baseline");
    let r = pv(&["lint", s(&contracts), "--armed-baseline-ref", "HEAD"]);
    assert_eq!(r.code, 3, "{}", show(&r));
    assert!(
        r.stderr.contains("error: armed_gates shrank: composition"),
        "{}",
        show(&r)
    );
}

/// `git init` + commit everything under `dir`. `GIT_DIR`/`GIT_WORK_TREE`/`GIT_INDEX_FILE` are dropped: under a
/// git hook they are exported, and inherited they would aim `git init` at the hook's own repository.
fn git_init_commit(dir: &Path) {
    let git = |args: &[&str]| {
        let st = Command::new("git")
            .current_dir(dir)
            .env_remove("GIT_DIR")
            .env_remove("GIT_WORK_TREE")
            .env_remove("GIT_INDEX_FILE")
            .args([
                "-c",
                "user.email=t@t",
                "-c",
                "user.name=t",
                "-c",
                "commit.gpgsign=false",
            ])
            .args(args)
            .output()
            .expect("git runs");
        assert!(
            st.status.success(),
            "git {args:?}: {}",
            String::from_utf8_lossy(&st.stderr)
        );
    };
    git(&["init", "-q"]);
    git(&["add", "-A"]);
    git(&["commit", "-q", "-m", "baseline"]);
}

#[test]
fn an_explicit_comparand_that_cannot_be_read_is_an_error_not_a_skip() {
    // Not a git work tree: the caller asked for a check that cannot run.
    let d = corpus(None);
    let r = pv(&["lint", s(d.path()), "--armed-baseline-ref", "HEAD"]);
    assert_eq!(r.code, 1, "{}", show(&r));
    assert!(
        r.stderr.contains("error: --armed-baseline-ref HEAD:"),
        "{}",
        show(&r)
    );
    assert!(!r.stdout.contains("NOT CHECKED"), "{}", show(&r));

    // A work tree, but the ref names no commit.
    let repo = tempfile::tempdir().expect("repo dir");
    let contracts = repo.path().join("contracts");
    std::fs::create_dir_all(&contracts).expect("contracts dir");
    std::fs::copy(
        repo_contracts().join("softmax-kernel-v1.yaml"),
        contracts.join("softmax-kernel-v1.yaml"),
    )
    .expect("control copies");
    git_init_commit(repo.path());
    let r = pv(&["lint", s(&contracts), "--armed-baseline-ref", "no-such-ref"]);
    assert_eq!(r.code, 1, "{}", show(&r));
    assert!(
        r.stderr
            .contains("error: --armed-baseline-ref no-such-ref: not a commit"),
        "{}",
        show(&r)
    );

    // A committed baseline that does not parse is an error, not the default set.
    std::fs::write(contracts.join("lint-baseline.json"), "{ not json").expect("bad baseline");
    git_init_commit(repo.path());
    std::fs::write(contracts.join("lint-baseline.json"), baseline_json(&EIGHT)).expect("fixed");
    let r = pv(&["lint", s(&contracts), "--armed-baseline-ref", "HEAD"]);
    assert_eq!(r.code, 1, "{}", show(&r));
    assert!(
        r.stderr.contains("error: armed_gates comparand HEAD"),
        "{}",
        show(&r)
    );
}

#[test]
fn json_report_carries_the_lattice() {
    let d = corpus(None);
    let r = pv(&["lint", s(d.path()), "--format", "json"]);
    assert_eq!(r.code, 0, "{}", show(&r));
    let v: serde_json::Value = serde_json::from_str(&r.stdout)
        .unwrap_or_else(|e| panic!("stdout is JSON: {e}\n{}", show(&r)));
    assert_eq!(v["verdict"], "Pass", "{}", show(&r));
    let armed = v["armed_gates"]
        .as_array()
        .unwrap_or_else(|| panic!("armed_gates array\n{}", show(&r)));
    assert_eq!(armed.len(), 8, "{}", show(&r));
    assert_eq!(
        v["not_armed"],
        serde_json::Value::Array(vec![
            serde_json::Value::String("reverse-coverage".into()),
            // ONT-2b's gate runs everywhere (R-8) and this corpus declares no `armed_gates`, so it is
            // reported and excluded — the DEFAULT set is still the eight ONT-6 ruled.
            serde_json::Value::String("sigma".into()),
        ]),
        "{}",
        show(&r)
    );
    let gates = v["gates"]
        .as_array()
        .unwrap_or_else(|| panic!("gates array\n{}", show(&r)));
    assert!(!gates.is_empty());
    for g in gates {
        assert!(
            g.get("verdict").is_some(),
            "every gate emits a verdict: {g}"
        );
    }
    let rc = gates
        .iter()
        .find(|g| g["name"] == "reverse-coverage")
        .expect("reverse-coverage gate present");
    assert_eq!(rc["verdict"], "Unknown(Skip)", "{rc}");
}

/// ONT-6 ruled the DEFAULT set at eight. A repo may arm more: ONT-2b appended `sigma` to aprender's own
/// declaration, which is what §3.9 monotonicity protects — the eight are still armed, and one more is.
#[test]
fn repo_baseline_arms_the_eight_ruled_gates_and_every_later_row_that_armed_one() {
    let text = std::fs::read_to_string(repo_contracts().join("lint-baseline.json"))
        .expect("repo baseline reads");
    let v: serde_json::Value = serde_json::from_str(&text).expect("repo baseline is JSON");
    let names: Vec<&str> = v["armed_gates"]
        .as_array()
        .expect("armed_gates array")
        .iter()
        .filter_map(|x| x.as_str())
        .collect();
    for ruled in EIGHT {
        assert!(
            names.contains(&ruled),
            "the ruled 8 stay armed; `{ruled}` is missing from contracts/lint-baseline.json"
        );
    }
    assert_eq!(
        names,
        [
            "validate",
            "audit",
            "score",
            "verify",
            "enforce",
            "enforcement-level",
            "duplicate-stems",
            "composition",
            "sigma"
        ]
        .to_vec(),
        "the ruled 8 plus the gates later rows armed (ONT-2b: sigma)"
    );
}
