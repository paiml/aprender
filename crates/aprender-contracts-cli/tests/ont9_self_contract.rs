//! ONT-9 (#4078) — the ontology has a contract on itself: `contracts/ont-self-v1.yaml`, `entity.type: pv-contract`,
//! `ref: contracts/`.
//!
//! ONT-001 §5 ONT-9 probe, minus the ledger's `merged`, in `the_spec_probe_holds_on_the_repo`. Around it: the
//! contract names a real test for every falsifier it lists (a contract about contracts that cites a ghost is the
//! defect it exists to catch); it does not claim the L3 rung before `cargo kani` has run; and the corpus-level
//! mutation the row names — introduce a cycle — turns the `relations` gate RED on the CLI, including the mixed
//! `depends_on ∪ supersedes` cycle the per-role sweep used to pass.

use std::path::{Path, PathBuf};
use std::process::Command;

struct Run {
    code: i32,
    stdout: String,
    stderr: String,
}

fn pv(args: &[&str]) -> Run {
    let out = Command::new(env!("CARGO_BIN_EXE_pv"))
        .args(args)
        .output()
        .expect("spawn pv");
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

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn s(p: &Path) -> &str {
    p.to_str().expect("utf-8 path")
}

const CONTRACT: &str = "contracts/ont-self-v1.yaml";
const WITNESS_RS: &str = "crates/aprender-contracts/src/ontology/witness.rs";

fn contract() -> serde_yaml::Value {
    let text = std::fs::read_to_string(repo().join(CONTRACT))
        .unwrap_or_else(|e| panic!("{CONTRACT} is readable: {e}"));
    serde_yaml::from_str(&text).unwrap_or_else(|e| panic!("{CONTRACT} parses: {e}"))
}

fn text(rel: &str) -> String {
    std::fs::read_to_string(repo().join(rel)).unwrap_or_else(|e| panic!("{rel}: {e}"))
}

/// §5 ONT-9's probe: `tracked contracts/ont-self-v1.yaml && rc_is 0 "$PV" validate contracts/ont-self-v1.yaml &&
/// present 'fn ont_planted' …/witness.rs && present 'KANI-ONT-9-1' …/witness.rs` (`merged ONT-9` is the ledger's).
#[test]
fn the_spec_probe_holds_on_the_repo() {
    // CI runs this in a container whose uid does not own the checkout, and actions/checkout marked it safe only in
    // the host's git config — git refuses the repo ("dubious ownership") and the probe read "not tracked". git 2.34
    // honours `safe.directory` only from global/system config (not `-c`, not GIT_CONFIG_COUNT), so hand it one.
    let cfg = tempfile::NamedTempFile::new().expect("temp gitconfig");
    std::fs::write(cfg.path(), "[safe]\n\tdirectory = *\n").expect("write temp gitconfig");
    let tracked = Command::new("git")
        .args(["ls-files", "--error-unmatch", CONTRACT])
        .env("GIT_CONFIG_GLOBAL", cfg.path())
        .current_dir(repo())
        .output()
        .expect("spawn git");
    assert!(
        tracked.status.success(),
        "{CONTRACT} is not tracked: {}",
        String::from_utf8_lossy(&tracked.stderr)
    );
    let r = pv(&["validate", s(&repo().join(CONTRACT))]);
    assert_eq!(r.code, 0, "{}", show(&r));
    let w = text(WITNESS_RS);
    assert!(
        w.contains("fn ont_planted"),
        "no `fn ont_planted` in {WITNESS_RS}"
    );
    assert!(
        w.contains("KANI-ONT-9-1"),
        "no `KANI-ONT-9-1` in {WITNESS_RS}"
    );
}

/// The contract is about the corpus itself (§5 ONT-9, B.16), and binds the checker and the gates it governs.
#[test]
fn the_contract_is_anchored_on_the_corpus_and_depends_on_what_it_governs() {
    let c = contract();
    assert_eq!(c["entity"]["type"].as_str(), Some("pv-contract"));
    assert_eq!(c["entity"]["ref"].as_str(), Some("contracts/"));
    let deps: Vec<&str> = c["relations"]["depends_on"]
        .as_sequence()
        .expect("relations.depends_on is a list")
        .iter()
        .filter_map(serde_yaml::Value::as_str)
        .collect();
    for d in [
        "ont-consistency-v1",
        "ont-relations-v1",
        "ont-verdict-lattice-v1",
    ] {
        assert!(
            deps.contains(&d),
            "ont-self-v1 must depend_on {d}: {deps:?}"
        );
    }
}

/// Every falsifier names a test that exists. `cargo test … <path>::<name>` whose `<name>` is no `fn` anywhere is
/// a filter that matches zero tests and exits 0 — a ghost binding, the defect a contract on contracts must not
/// carry itself.
#[test]
fn every_falsifier_names_a_test_function_that_exists() {
    let c = contract();
    let tests = c["falsification_tests"]
        .as_sequence()
        .expect("falsification_tests is a list");
    assert!(tests.len() >= 7, "one falsifier per obligation at least");
    let src = crate_sources();
    for t in tests {
        let cmd = t["test"].as_str().expect("test is a string");
        let last = cmd.split_whitespace().last().expect("non-empty");
        if last.starts_with("--") || !cmd.contains("--lib") {
            // `--test <target>` or `--bin` — the target is the binding; check the file exists.
            let target = cmd.split_whitespace().skip_while(|w| *w != "--test").nth(1);
            if let Some(target) = target {
                assert!(
                    src.iter()
                        .any(|(p, _)| p.ends_with(&format!("tests/{target}.rs"))),
                    "{cmd}: no tests/{target}.rs"
                );
                if let Some(name) = cmd.split_whitespace().nth(6) {
                    assert!(
                        src.iter()
                            .any(|(_, body)| body.contains(&format!("fn {name}("))),
                        "{cmd}: no `fn {name}(`"
                    );
                }
            }
            continue;
        }
        let name = last.rsplit("::").next().expect("a path");
        assert!(
            src.iter()
                .any(|(_, body)| body.contains(&format!("fn {name}("))),
            "{cmd}: no `fn {name}(` in aprender-contracts{{,-cli}} — a ghost binding"
        );
    }
}

fn crate_sources() -> Vec<(String, String)> {
    let mut out = Vec::new();
    let mut stack = vec![
        repo().join("crates/aprender-contracts/src"),
        repo().join("crates/aprender-contracts-cli/src"),
        repo().join("crates/aprender-contracts-cli/tests"),
    ];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).into_iter().flatten().flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.extension().is_some_and(|x| x == "rs") {
                let body = std::fs::read_to_string(&p).unwrap_or_default();
                out.push((p.to_string_lossy().into_owned(), body));
            }
        }
    }
    out
}

/// The L3 rung is not claimed before it ran. `KANI-ONT-9-1` is a real `#[kani::proof]` in witness.rs and is
/// listed, and while `proof.status` is `declared` the summary counts zero Kani-proved obligations. Flipping one
/// without the other — a proof claimed on a run nobody made — is RED here.
#[test]
fn the_kani_rung_is_declared_not_claimed() {
    let c = contract();
    let h = c["kani_harnesses"]
        .as_sequence()
        .expect("kani_harnesses is a list");
    assert!(
        h.iter().any(|x| x["id"].as_str() == Some("KANI-ONT-9-1")
            && x["harness"].as_str() == Some("kani_ont_9_1")),
        "KANI-ONT-9-1 → kani_ont_9_1 is listed"
    );
    let w = text(WITNESS_RS);
    assert!(
        w.contains("#[kani::proof]") && w.contains("fn kani_ont_9_1()"),
        "the harness is a real #[kani::proof] fn"
    );
    let status = c["proof"]["status"].as_str().expect("proof.status");
    let l3 = c["verification_summary"]["l3_kani_proved"]
        .as_u64()
        .expect("l3_kani_proved");
    match status {
        "declared" => assert_eq!(l3, 0, "declared, so nothing is Kani-proved yet"),
        "proved" => assert!(l3 >= 1, "proved, so the summary counts it"),
        other => panic!("proof.status `{other}` is neither declared nor proved"),
    }
}

/// Mutation "introduce a cycle → RED", at the CLI: the repo's relations (ont-self-v1 included) are acyclic and
/// exit 0; a per-role cycle and a `depends_on ∪ supersedes` cycle each exit 1 with PV-ONT-009.
#[test]
fn a_cycle_in_the_corpus_turns_the_relations_gate_red() {
    let live = pv(&["lint", s(&repo().join("contracts")), "--gate", "relations"]);
    assert_eq!(live.code, 0, "{}", show(&live));
    for (fixture, path) in [
        ("relations-cycle", "a -> b -> c -> a"),
        ("relations-mixed-cycle", "a -> b -> a"),
    ] {
        let r = pv(&[
            "lint",
            s(&repo().join("tests/fixtures/ont").join(fixture)),
            "--gate",
            "relations",
        ]);
        let all = format!("{}{}", r.stdout, r.stderr);
        assert_eq!(r.code, 1, "{fixture}: {}", show(&r));
        assert!(
            all.contains("PV-ONT-009") && all.contains(path),
            "{fixture}: {}",
            show(&r)
        );
    }
}
