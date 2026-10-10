//! #4455 / PMAT-4455 — a findings line under contract (`contracts/findings-ledger-v1.yaml`), on the CLI.
//!
//! The case table for `scripts/findings_ledger.sh`, which has pv judge a ledger on the REAL contract.
//! Every refused fixture is a conforming line 1 plus one defective line 2, so a refusal must name
//! line 2 and the one constraint that fired, and line 1 must pass beside it.
//!
//! | fixture | line 2 | expected |
//! |---|---|---|
//! | `ok` | a second conforming line, `suspected_epic: unknown` | **Pass** (exit 0) |
//! | `missing-sha` | no `found_at_sha` | Fail, `minCount` |
//! | `extra-key` | an eighth key, `session` | Fail, `closed` |
//! | `epic-tbd` | `suspected_epic: tbd` | Fail, `pattern`: the arm that matters |
//! | `epic-bare` | `suspected_epic: 4433` | Fail, `pattern`: a shape that permits two spellings gets both |
//! | `severity-high` | `severity: high` | Fail, `in` |
//! | `short-sha` | a 9-hex `found_at_sha` | Fail, `pattern` |
//! | `bad-id` | `id: F-1` | Fail, `pattern` |
//! | `short-title` | `title: broken` | Fail, `minLength` |
//! | `torn` | half a line | exit 2: not judged, not a pass |
//! | `empty` | *(no lines)* | exit 3: not a ledger, not a pass |
//!
//! The fixtures carry no copy of the contract: the script reads `contracts/findings-ledger-v1.yaml`
//! itself, so widening the shape there turns this table red.

use std::path::{Path, PathBuf};
use std::process::Command;

const SHA: &str = "4e27b64015549396d729f1b688e456b5772a1365";

fn pv_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_pv"))
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn fixture(name: &str) -> PathBuf {
    repo_root()
        .join("tests/fixtures/ont/findings-ledger")
        .join(format!("{name}.jsonl"))
}

struct Run {
    code: i32,
    stdout: String,
    stderr: String,
}

impl Run {
    fn all(&self) -> String {
        format!(
            "exit {}\n--- stdout\n{}\n--- stderr\n{}",
            self.code, self.stdout, self.stderr
        )
    }

    fn report(&self) -> serde_json::Value {
        serde_json::from_str(&self.stdout)
            .unwrap_or_else(|e| panic!("pv's report is JSON ({e})\n{}", self.all()))
    }

    /// Every `message` in the report, with its severity.
    fn messages(&self) -> Vec<(String, String)> {
        fn walk(v: &serde_json::Value, out: &mut Vec<(String, String)>) {
            match v {
                serde_json::Value::Object(m) => {
                    if let Some(msg) = m.get("message").and_then(|x| x.as_str()) {
                        let sev = m.get("severity").and_then(|x| x.as_str()).unwrap_or("");
                        out.push((sev.to_owned(), msg.to_owned()));
                    }
                    m.values().for_each(|x| walk(x, out));
                }
                serde_json::Value::Array(a) => a.iter().for_each(|x| walk(x, out)),
                _ => {}
            }
        }
        let mut out = Vec::new();
        walk(&self.report(), &mut out);
        out
    }
}

fn ledger_sh(args: &[&str]) -> Run {
    let out = Command::new("bash")
        .arg(repo_root().join("scripts/findings_ledger.sh"))
        .args(args)
        .env("PV", pv_bin())
        .output()
        .expect("failed to spawn bash");
    Run {
        code: out.status.code().unwrap_or(-1),
        stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
    }
}

fn check(name: &str) -> Run {
    let path = fixture(name);
    ledger_sh(&["check", path.to_str().expect("utf-8 path")])
}

#[test]
fn a_ledger_of_conforming_lines_passes_including_the_unknown_epic() {
    let r = check("ok");
    assert_eq!(r.code, 0, "{}", r.all());
    let v = r.report();
    assert_eq!(v["verdict"].as_str(), Some("Pass"), "{}", r.all());
    assert_eq!(v["extra"]["violations"].as_u64(), Some(0), "{}", r.all());
    assert_eq!(
        v["extra"]["focus_nodes_n"].as_u64(),
        Some(2),
        "both lines must be focus nodes\n{}",
        r.all()
    );
    assert!(
        v["extra"]["armed_shapes"]
            .as_array()
            .expect("armed_shapes")
            .iter()
            .any(|s| s.as_str() == Some("findings-ledger-v1")),
        "the script must arm the shape, or the pass is vacuous\n{}",
        r.all()
    );
}

#[test]
fn each_defective_line_is_refused_naming_its_line_and_its_constraint() {
    for (name, component) in [
        ("missing-sha", "minCount"),
        ("extra-key", "closed"),
        ("epic-tbd", "pattern"),
        ("epic-bare", "pattern"),
        ("severity-high", "in"),
        ("short-sha", "pattern"),
        ("bad-id", "pattern"),
        ("short-title", "minLength"),
    ] {
        let r = check(name);
        assert_eq!(r.code, 1, "{name} was not refused\n{}", r.all());
        let errors: Vec<String> = r
            .messages()
            .into_iter()
            .filter(|(sev, _)| sev == "Error")
            .map(|(_, m)| m)
            .collect();
        assert_eq!(
            errors.len(),
            1,
            "{name}: exactly line 2 is defective, so exactly one violation\n{}",
            r.all()
        );
        assert!(
            errors[0].starts_with(
                "ont:finding/findings-ledger-v1.2 violates shape `findings-ledger-v1`"
            ) && errors[0].contains(&format!("({component})")),
            "{name}: the violation must name line 2 and ({component}), got {:?}",
            errors[0]
        );
    }
}

#[test]
fn a_torn_line_is_not_a_pass() {
    let r = check("torn");
    assert_eq!(
        r.code,
        2,
        "a line pv cannot read must leave the verdict open\n{}",
        r.all()
    );
    assert_eq!(
        r.report()["verdict"].as_str(),
        Some("Unknown(Warn)"),
        "{}",
        r.all()
    );
}

#[test]
fn an_empty_or_absent_ledger_is_not_a_pass() {
    let r = check("empty");
    assert_eq!(r.code, 3, "{}", r.all());
    let absent = fixture("no-such-ledger");
    let r = ledger_sh(&["check", absent.to_str().expect("utf-8 path")]);
    assert_eq!(r.code, 3, "{}", r.all());
}

fn add(ledger: &Path, extra: &[&str]) -> Run {
    let mut args = vec![
        "add",
        "--ledger",
        ledger.to_str().expect("utf-8 path"),
        "--id",
        "FND-20261009-writer-case",
        "--title",
        "a line the writer must append",
        "--evidence",
        "it was measured \"here\" \\ there",
        "--repro",
        "none",
        "--severity",
        "P2",
        "--sha",
        SHA,
    ];
    args.extend_from_slice(extra);
    ledger_sh(&args)
}

#[test]
fn the_writer_appends_a_line_pv_passed_and_nothing_else() {
    let dir = tempfile::tempdir().expect("tempdir");
    let ledger = dir.path().join("findings.jsonl");

    let r = add(&ledger, &["--epic", "unknown"]);
    assert_eq!(r.code, 0, "{}", r.all());
    let after_one = std::fs::read(&ledger).expect("the writer created the ledger");
    let line: serde_json::Value = serde_json::from_slice(&after_one).expect("one JSON line");
    assert_eq!(
        line["evidence"].as_str(),
        Some("it was measured \"here\" \\ there")
    );
    assert_eq!(line["found_at_sha"].as_str(), Some(SHA));
    assert_eq!(
        line.as_object().expect("an object").len(),
        7,
        "the seven fields and no other"
    );

    // Refused by pv: the plausible but undeclared sentinel, and a field left out.
    for extra in [&["--epic", "tbd"][..], &[][..]] {
        let r = add(&ledger, extra);
        assert_eq!(r.code, 1, "{extra:?} was appended\n{}", r.all());
        assert_eq!(
            std::fs::read(&ledger).expect("ledger"),
            after_one,
            "a refused line must leave the ledger byte-identical ({extra:?})"
        );
    }

    // A control character JSON would need \u for is a usage error, before pv.
    let r = add(&ledger, &["--epic", "unknown", "--repro", "a\u{1}b"]);
    assert_eq!(r.code, 64, "{}", r.all());
    assert_eq!(std::fs::read(&ledger).expect("ledger"), after_one);

    // A second conforming line, and the whole ledger then passes.
    assert_eq!(add(&ledger, &["--epic", "#4645"]).code, 0);
    let r = ledger_sh(&["check", ledger.to_str().expect("utf-8 path")]);
    assert_eq!(r.code, 0, "{}", r.all());
    assert_eq!(
        r.report()["extra"]["focus_nodes_n"].as_u64(),
        Some(2),
        "{}",
        r.all()
    );
}

#[test]
fn the_writer_never_glues_a_line_onto_a_torn_last_line() {
    let dir = tempfile::tempdir().expect("tempdir");
    let ledger = dir.path().join("findings.jsonl");
    std::fs::write(&ledger, b"{\"id\":\"no newline\"}").expect("write");
    let before = std::fs::read(&ledger).expect("ledger");
    let r = add(&ledger, &["--epic", "unknown"]);
    assert_eq!(r.code, 3, "{}", r.all());
    assert_eq!(std::fs::read(&ledger).expect("ledger"), before);
}

#[test]
fn the_contract_stays_out_of_the_corpus_gate() {
    // It binds lines written after it landed, and no PR or release step reads docs/findings/. An
    // `entity:` here, or an entry in `armed_shapes`, would put the older lines on the corpus gate.
    let contract = std::fs::read_to_string(repo_root().join("contracts/findings-ledger-v1.yaml"))
        .expect("the contract is in the tree");
    assert!(
        !contract.lines().any(|l| l.starts_with("entity:")),
        "findings-ledger-v1 must not declare an entity"
    );
    let baseline: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(repo_root().join("contracts/lint-baseline.json"))
            .expect("the lint baseline is in the tree"),
    )
    .expect("the lint baseline is JSON");
    let armed = baseline["armed_shapes"].as_array().expect("armed_shapes[]");
    assert!(
        !armed.is_empty(),
        "an empty armed_shapes would make this check vacuous"
    );
    assert!(
        !armed
            .iter()
            .any(|s| s.as_str() == Some("findings-ledger-v1")),
        "findings-ledger-v1 must not be armed in the corpus"
    );
}
