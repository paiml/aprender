//! ONT-4f (aprender#4330) — GitHub repos, issues, pull requests and milestones as focus nodes, on the CLI.
//!
//! Each fixture is the real Σ, the real `contracts/github-entities-v1.yaml`, and committed snapshots under
//! `evidence/github/<type>/`:
//!
//! | fixture | exit | why |
//! |---|---|---|
//! | `github-green` | 0 | one tracked snapshot per type; `pr:baseRepo` and `issue:milestone` resolve to IRIs |
//! | `github-merged-no-mergedat` | 1 | `state: MERGED` with no `mergedAt` — refused by the extractor, named |
//! | `github-untracked-milestone` | 1 | an issue naming a milestone no snapshot tracks — FAIL CLOSED, never Unknown |
//! | `github-sha-mismatch` | 1 | a repo whose `sha` disagrees with its `ref` — refused naming both shas |
//!
//! DISCRIMINATION, in both directions: `github-green` must PASS, so a build that refuses every snapshot fails
//! here; the three mutation fixtures must FAIL, so a build that accepts anything fails here too. The fixture copies
//! of the contract are asserted byte-identical to the real one, so weakening the real shapes without the fixtures
//! is caught by the drift test instead. No test here touches the network: the snapshots are the whole input.

use std::path::{Path, PathBuf};
use std::process::Command;

fn pv_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_pv"))
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn fixture(name: &str) -> PathBuf {
    repo_root().join("tests/fixtures/ont").join(name)
}

const FIXTURES: [&str; 4] = [
    "github-green",
    "github-merged-no-mergedat",
    "github-untracked-milestone",
    "github-sha-mismatch",
];

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

    fn json(&self) -> serde_json::Value {
        serde_json::from_str(&self.stdout).expect("json report")
    }
}

fn shapes_on(contracts: &Path, cwd: Option<&Path>) -> Run {
    let mut cmd = Command::new(pv_bin());
    cmd.args([
        "lint",
        contracts.to_str().expect("utf-8 path"),
        "--gate",
        "shapes",
        "--format",
        "json",
    ]);
    if let Some(dir) = cwd {
        cmd.current_dir(dir);
    }
    let out = cmd.output().expect("failed to spawn pv");
    Run {
        code: out.status.code().unwrap_or(-1),
        stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
    }
}

fn on_fixture(name: &str) -> Run {
    shapes_on(&fixture(name).join("contracts"), None)
}

fn violations(r: &Run) -> u64 {
    r.json()["extra"]["violations"].as_u64().unwrap_or(0)
}

#[test]
fn one_tracked_snapshot_per_type_passes_and_every_type_is_counted() {
    let r = on_fixture("github-green");
    assert_eq!(r.code, 0, "{}", r.all());
    assert_eq!(violations(&r), 0, "{}", r.all());
    let by = &r.json()["extra"]["by_entity_type"];
    for t in ["repo", "issue", "pull-request", "milestone"] {
        assert_eq!(by[t], 1, "by_entity_type[{t}]\n{}", r.all());
    }
}

#[test]
fn mutation_a_merged_pull_request_with_no_merged_at_fails_naming_the_field() {
    let r = on_fixture("github-merged-no-mergedat");
    assert_eq!(r.code, 1, "{}", r.all());
    assert!(r.stdout.contains("PV-ONT-012"), "{}", r.all());
    assert!(r.stdout.contains("`mergedAt` is absent"), "{}", r.all());
    assert!(
        r.stdout.contains("paiml__aprender__3706.json"),
        "{}",
        r.all()
    );
}

#[test]
fn mutation_an_issue_naming_an_untracked_milestone_fails_closed_never_unknown() {
    let r = on_fixture("github-untracked-milestone");
    assert_eq!(r.code, 1, "Fail, not a decline:\n{}", r.all());
    assert_eq!(violations(&r), 1, "{}", r.all());
    assert!(r.stdout.contains("github-issue"), "{}", r.all());
    assert!(r.stdout.contains("milestoneUnresolved"), "{}", r.all());
    assert!(r.stdout.contains("paiml/aprender#3"), "{}", r.all());
}

#[test]
fn mutation_a_repo_whose_sha_disagrees_with_its_ref_fails_naming_both() {
    let r = on_fixture("github-sha-mismatch");
    assert_eq!(r.code, 1, "{}", r.all());
    assert!(r.stdout.contains("PV-ONT-012"), "{}", r.all());
    assert!(
        r.stdout
            .contains("aa7c6ef03ee7b7f8d8dc09dc393e97619952d95f")
            && r.stdout
                .contains("0000000000000000000000000000000000000000"),
        "both shas are named\n{}",
        r.all()
    );
}

#[test]
fn the_real_corpus_counts_every_github_type_and_fires_every_control() {
    let r = shapes_on(Path::new("contracts"), Some(&repo_root()));
    let v = r.json();
    for t in ["repo", "issue", "pull-request", "milestone"] {
        assert!(
            v["extra"]["by_entity_type"][t].as_u64().unwrap_or(0) >= 1,
            "by_entity_type[{t}] is not >= 1\n{}",
            r.all()
        );
        assert_eq!(
            v["pc_extract"][t], "fired",
            "pc_extract[{t}]: {}",
            v["pc_extract"]
        );
    }
}

#[test]
fn every_fixture_carries_the_real_contract_byte_for_byte() {
    let real = std::fs::read(repo_root().join("contracts/github-entities-v1.yaml"))
        .expect("the real contract is in the tree");
    for name in FIXTURES {
        let copy = std::fs::read(fixture(name).join("contracts/github-entities-v1.yaml"))
            .unwrap_or_else(|e| panic!("{name} carries the contract: {e}"));
        assert_eq!(
            copy, real,
            "{name}'s copy of github-entities-v1.yaml has drifted from contracts/"
        );
    }
}
