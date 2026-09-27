//! PVL-001 EV-11 (PMAT-4166) — `pv lint --gate theorem-pairing --gate depends-on-present`: two shrink-only
//! ratchets. Since #3569 each baseline is the same gate measured over the comparand (merge-base) tree that
//! `PV_LINT_COMPARAND` names, in the same run — never a number stored in `contracts/lint-baseline.json`, which a
//! PR could restamp. The first test is the probe on the real corpus (a hold against itself); the rest run on a
//! throwaway repo built in a tempdir, because the gates read the repo ROOT (`lean/`, `book/`) as well as the
//! contract dir.
//!
//! | case | expected |
//! |---|---|
//! | real corpus against itself, both gates | exit 0, both Pass, and no retired key stored |
//! | tempdir unchanged against its snapshot | exit 0 |
//! | the spec's mutation: add an unpaired Theorem module | exit 1, PV-RAT-001 — the meet rejects |
//! | a kernel contract with no `depends_on` added | exit 1, PV-RAT-002; the same pair reversed passes |
//! | a restamped stored count | moves nothing, in either direction |
//! | no comparand named | exit 2, `Unknown(Report)` with the count printed — reported, never a pass |
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

const COMPARAND: &str = "PV_LINT_COMPARAND";

fn pv(args: &[&str]) -> Run {
    pv_against(args, None)
}

/// `pv` with `comparand` named as the tree the baselines are measured over; `None` names none, whatever the
/// caller's environment exports.
fn pv_against(args: &[&str], comparand: Option<&Path>) -> Run {
    let scratch = tempfile::tempdir().expect("scratch cwd is creatable");
    let mut cmd = Command::new(pv_bin());
    cmd.current_dir(scratch.path())
        .args(args)
        .env_remove(COMPARAND);
    if let Some(dir) = comparand {
        cmd.env(COMPARAND, dir);
    }
    let out = cmd.output().expect("failed to spawn pv");
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
/// `depends_on`): 0 unpaired, 1 without depends_on.
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
    t
}

/// A copy of `root` as it stands now: the comparand (merge-base) tree later edits are judged against.
fn snapshot(root: &Path) -> tempfile::TempDir {
    fn copy(from: &Path, to: &Path) {
        std::fs::create_dir_all(to).expect("mkdir");
        for entry in std::fs::read_dir(from).expect("read_dir") {
            let entry = entry.expect("entry");
            let dest = to.join(entry.file_name());
            if entry.file_type().expect("file type").is_dir() {
                copy(&entry.path(), &dest);
            } else {
                std::fs::copy(entry.path(), dest).expect("copy");
            }
        }
    }
    let t = tempfile::tempdir().expect("tempdir");
    copy(root, t.path());
    t
}

fn lint(root: &Path, gates: &[&str]) -> Run {
    lint_against(root, None, gates)
}

/// `pv lint root/contracts` with `base/contracts` named as the comparand.
fn lint_against(root: &Path, base: Option<&Path>, gates: &[&str]) -> Run {
    let dir = root.join("contracts");
    let mut args = vec!["lint", s(&dir)];
    args.extend_from_slice(gates);
    let comparand = base.map(|b| b.join("contracts"));
    pv_against(&args, comparand.as_deref())
}

/// The keys #3569 retired from the stored file: a comparand is measured, never stored.
const RETIRED: [&str; 2] = ["unpaired_theorem_modules", "contracts_without_depends_on"];

#[test]
fn the_probe_passes_on_the_real_corpus_against_itself_and_stores_nothing() {
    let baseline = repo_contracts().join("lint-baseline.json");
    let before = std::fs::read(&baseline).expect("baseline readable");
    let doc: serde_json::Value = serde_json::from_slice(&before).expect("baseline is JSON");
    for key in RETIRED {
        assert!(doc.get(key).is_none(), "{key} is measured, not stored: {doc}");
    }

    let dir = repo_contracts();
    let mut args = vec!["lint", s(&dir)];
    args.extend_from_slice(&BOTH);
    let r = pv_against(&args, Some(&dir));
    assert_eq!(r.code, 0, "{}", show(&r));
    let got = reports(&r.stdout);
    assert_eq!(got.len(), 2, "{}", show(&r));
    for g in &got {
        assert_eq!(g["verdict"], "Pass", "{g}");
    }
    assert_eq!(
        got[0]["baseline"], got[0]["unpaired_theorem_modules"],
        "the baseline is the comparand measured: {}",
        got[0]
    );
    assert_eq!(
        std::fs::read(&baseline).expect("baseline readable"),
        before,
        "a gate wrote the stored file"
    );
}

#[test]
fn unchanged_against_its_snapshot_both_gates_pass() {
    let t = repo();
    let base = snapshot(t.path());
    let r = lint_against(t.path(), Some(base.path()), &BOTH);
    assert_eq!(r.code, 0, "{}", show(&r));
}

/// PVL-001 EV-11's mutation, verbatim: "add an unpaired Theorem module → RED".
#[test]
fn adding_an_unpaired_theorem_module_is_red() {
    let t = repo();
    let base = snapshot(t.path());
    write(
        t.path(),
        "lean/ProvableContracts/Theorems/S/B.lean",
        "theorem b : True := trivial\n",
    );
    let r = lint_against(t.path(), Some(base.path()), &BOTH);
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
fn a_kernel_contract_without_depends_on_added_is_red_and_removed_is_green() {
    let t = repo();
    let base = snapshot(t.path());
    let text = std::fs::read_to_string(repo_contracts().join("softmax-kernel-v1.yaml"))
        .expect("control contract readable");
    write(t.path(), "contracts/softmax-kernel-copy-v1.yaml", &text);
    let gate = ["--gate", "depends-on-present"];
    let r = lint_against(t.path(), Some(base.path()), &gate);
    assert_eq!(r.code, 1, "{}", show(&r));
    assert!(r.stdout.contains("PV-RAT-002"), "{}", show(&r));
    // the same pair the other way round: head 1, comparand 2 — a fall
    let r = lint_against(base.path(), Some(t.path()), &gate);
    assert_eq!(r.code, 0, "{}", show(&r));
}

#[test]
fn a_restamped_stored_count_moves_nothing() {
    let t = repo();
    let base = snapshot(t.path());
    write(
        t.path(),
        "lean/ProvableContracts/Theorems/S/B.lean",
        "theorem b : True := trivial\n",
    );
    // the pre-#3569 dodge: restamp the stored number to cover the rise
    write(
        t.path(),
        "contracts/lint-baseline.json",
        "{\n  \"unpaired_theorem_modules\": 99,\n  \"contracts_without_depends_on\": 99\n}\n",
    );
    let r = lint_against(t.path(), Some(base.path()), &BOTH);
    assert_eq!(r.code, 1, "a restamp does not hide a measured rise: {}", show(&r));
    assert!(r.stdout.contains("PV-RAT-001"), "{}", show(&r));
    // and with no comparand the stored number is not a baseline either
    let r = lint(t.path(), &BOTH);
    assert_eq!(r.code, 2, "{}", show(&r));
}

#[test]
fn no_comparand_reports_the_count_and_is_never_a_pass() {
    let t = repo();
    let r = lint(t.path(), &BOTH);
    assert_eq!(r.code, 2, "{}", show(&r));
    let got = reports(&r.stdout);
    assert_eq!(got.len(), 2, "{}", show(&r));
    for g in &got {
        assert_eq!(g["verdict"], "Unknown(Report)", "{g}");
    }
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
