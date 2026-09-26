//! PVL-001 EV-11 (PMAT-4166) — `pv lint --gate theorem-pairing --gate depends-on-present`: two shrink-only
//! ratchets read from `contracts/lint-baseline.json`, which the gates never write.
//!
//! The row's probe, verbatim (paiml/infra PVL-001 @00553b0b): `json_object contracts/lint-baseline.json && jq -e
//! '(.unpaired_theorem_modules|numbers) and (.contracts_without_depends_on|numbers) and (.command|strings)'
//! contracts/lint-baseline.json && "$PV" lint contracts/ --gate theorem-pairing --gate depends-on-present`, and the
//! accept adds `git diff --exit-code contracts/lint-baseline.json` after it. The first test is that probe on the
//! real corpus; the rest run on a throwaway repo built in a tempdir, because the gates read the repo ROOT
//! (`lean/`, `book/`) as well as the contract dir.
//!
//! | case | expected |
//! |---|---|
//! | real corpus, both gates | exit 0, both Pass, the baseline byte-identical after |
//! | tempdir at the baseline | exit 0 |
//! | the spec's mutation: add an unpaired Theorem module | exit 1, PV-RAT-001 — the meet rejects |
//! | a kernel contract with no `depends_on` above the baseline | exit 1, PV-RAT-002 |
//! | no baseline key | exit 2, `Unknown(Report)` with the count printed — reported, never a pass |
//! | no Lean base | exit 2, decline naming what is missing |
//! | an unknown name among several | exit 1 before anything runs |

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

fn s(p: &Path) -> &str {
    p.to_str().expect("utf-8 path")
}

fn repo_contracts() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../contracts")
}

fn write(root: &Path, rel: &str, text: &str) {
    let p = root.join(rel);
    std::fs::create_dir_all(p.parent().expect("has a parent")).expect("mkdir");
    std::fs::write(p, text).expect("write");
}

/// `pv lint --gate X` prints one JSON object per gate, pretty-printed and back to back.
fn reports(stdout: &str) -> Vec<serde_json::Value> {
    serde_json::Deserializer::from_str(stdout)
        .into_iter::<serde_json::Value>()
        .map(|v| v.expect("each gate report is JSON"))
        .collect()
}

const BOTH: [&str; 4] = ["--gate", "theorem-pairing", "--gate", "depends-on-present"];

/// A repo with one paired theorem module and one kernel contract (the PVL-1 control, which has no
/// `depends_on`), at baselines 0 and 1.
fn repo() -> tempfile::TempDir {
    let t = tempfile::tempdir().expect("tempdir");
    write(
        t.path(),
        "lean/ProvableContracts/Theorems/S/A.lean",
        "theorem a : True := trivial\n",
    );
    write(
        t.path(),
        "book/a.md",
        "see ProvableContracts.Theorems.S.A\n",
    );
    std::fs::create_dir_all(t.path().join("contracts")).expect("mkdir");
    std::fs::copy(
        repo_contracts().join("softmax-kernel-v1.yaml"),
        t.path().join("contracts/softmax-kernel-v1.yaml"),
    )
    .expect("control contract copies");
    write(
        t.path(),
        "contracts/lint-baseline.json",
        "{\n  \"unpaired_theorem_modules\": 0,\n  \"contracts_without_depends_on\": 1\n}\n",
    );
    t
}

fn lint(root: &Path, gates: &[&str]) -> Run {
    let dir = root.join("contracts");
    let mut args = vec!["lint", s(&dir)];
    args.extend_from_slice(gates);
    pv(&args)
}

#[test]
fn the_probe_passes_on_the_real_corpus_and_writes_nothing() {
    let baseline = repo_contracts().join("lint-baseline.json");
    let before = std::fs::read(&baseline).expect("baseline readable");
    let doc: serde_json::Value = serde_json::from_slice(&before).expect("baseline is JSON");
    assert!(doc.is_object());
    assert!(doc["unpaired_theorem_modules"].is_u64(), "{doc}");
    assert!(doc["contracts_without_depends_on"].is_u64(), "{doc}");
    assert_eq!(doc["command"], "make lint-ratchet");

    let dir = repo_contracts();
    let mut args = vec!["lint", s(&dir)];
    args.extend_from_slice(&BOTH);
    let r = pv(&args);
    assert_eq!(r.code, 0, "{}", show(&r));
    let got = reports(&r.stdout);
    assert_eq!(got.len(), 2, "{}", show(&r));
    for g in &got {
        assert_eq!(g["verdict"], "Pass", "{g}");
    }
    assert_eq!(
        got[0]["unpaired_theorem_modules"], doc["unpaired_theorem_modules"],
        "the recorded baseline is the measured count (`make lint-ratchet` wrote it)"
    );
    assert_eq!(
        got[1]["contracts_without_depends_on"],
        doc["contracts_without_depends_on"]
    );
    assert_eq!(
        std::fs::read(&baseline).expect("baseline readable"),
        before,
        "a gate wrote the baseline"
    );
}

#[test]
fn at_the_baseline_both_gates_pass() {
    let t = repo();
    let r = lint(t.path(), &BOTH);
    assert_eq!(r.code, 0, "{}", show(&r));
}

/// PVL-001 EV-11's mutation, verbatim: "add an unpaired Theorem module → RED".
#[test]
fn adding_an_unpaired_theorem_module_is_red() {
    let t = repo();
    write(
        t.path(),
        "lean/ProvableContracts/Theorems/S/B.lean",
        "theorem b : True := trivial\n",
    );
    let r = lint(t.path(), &BOTH);
    assert_eq!(r.code, 1, "{}", show(&r));
    assert!(r.stdout.contains("PV-RAT-001"), "{}", show(&r));
    assert!(!r.stdout.contains("PV-RAT-002"), "{}", show(&r));
    assert!(
        r.stderr.contains("1/2 armed gates passed"),
        "the meet names the one that held: {}",
        show(&r)
    );
}

#[test]
fn a_kernel_contract_without_depends_on_above_the_baseline_is_red() {
    let t = repo();
    let text = std::fs::read_to_string(repo_contracts().join("softmax-kernel-v1.yaml"))
        .expect("control contract readable");
    write(t.path(), "contracts/softmax-kernel-copy-v1.yaml", &text);
    let r = lint(t.path(), &["--gate", "depends-on-present"]);
    assert_eq!(r.code, 1, "{}", show(&r));
    assert!(r.stdout.contains("PV-RAT-002"), "{}", show(&r));
}

#[test]
fn no_baseline_reports_the_count_and_is_never_a_pass() {
    let t = repo();
    write(t.path(), "contracts/lint-baseline.json", "{}\n");
    let r = lint(t.path(), &BOTH);
    assert_eq!(r.code, 2, "{}", show(&r));
    let got = reports(&r.stdout);
    assert_eq!(got.len(), 2, "{}", show(&r));
    for g in &got {
        assert_eq!(g["verdict"], "Unknown(Report)", "{g}");
    }
    // the count `make lint-ratchet` records as the first baseline
    assert_eq!(got[0]["unpaired_theorem_modules"], 0);
    assert_eq!(got[1]["contracts_without_depends_on"], 1);
}

#[test]
fn no_lean_base_is_a_decline_naming_what_is_missing() {
    let t = repo();
    std::fs::remove_dir_all(t.path().join("lean")).expect("rm lean");
    let r = lint(t.path(), &["--gate", "theorem-pairing"]);
    assert_eq!(r.code, 2, "{}", show(&r));
    assert!(r.stderr.contains("no Lean theorem base"), "{}", show(&r));
}

#[test]
fn an_unknown_name_among_several_is_refused_before_anything_runs() {
    let t = repo();
    let r = lint(t.path(), &["--gate", "theorem-pairing", "--gate", "bogus"]);
    assert_eq!(r.code, 1, "{}", show(&r));
    assert!(r.stderr.contains("--gate bogus"), "{}", show(&r));
    assert!(r.stdout.is_empty(), "nothing ran: {}", show(&r));
}
