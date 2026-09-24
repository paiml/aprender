//! ONT-4c (PMAT-3847, #3847) — README.md, CLAUDE.md and a CSV as graded documents, on the CLI.
//!
//! ONT-001 v4.16 §5 ONT-4c RED, on `pv lint --gate shapes`: `tests/fixtures/ont/docs-green/` is a whole repo in
//! miniature (a merge-path workflow, a README and a CLAUDE.md with claim fences, a CSV and its producer, and a
//! baseline carrying the measured sets). Every other case is ONE edit to a copy of it, so each RED is caused by
//! that edit and nothing else:
//!
//! - a claim no merge-path step runs is refused by name (README and CLAUDE.md), whatever the fence's spelling of
//!   `bash`/`sh`/`shell`, and a step under `if: false` runs nothing;
//! - a path CLAUDE.md names that does not exist is refused by name;
//! - a ragged CSV row is refused naming the file and the row;
//! - F-33: a committed `verified_commands[]` that disagrees with the live set is RED naming the command;
//! - F-34 (git): a verified claim relabelled away with no new `withdrawn[]` entry is RED naming it, and passes
//!   with one; a `withdrawn[]` entry for a claim still live is RED; a baseline whose keys are absent at the
//!   comparand is checked against ∅ (bootstrap), not exempt; a work tree with no comparand ref is RED, and a
//!   corpus outside any work tree is `not-checked`.
//!
//! DISCRIMINATION: the green fixture must PASS with every doc type counted, so a build that refuses everything
//! fails this file; each RED asserts the refusal TEXT, so a build that fails for another reason fails it too.

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

fn pv_in(cwd: &Path, args: &[&str]) -> Run {
    let out = Command::new(pv_bin())
        .current_dir(cwd)
        .args(args)
        .env_remove("GIT_DIR")
        .env_remove("GIT_WORK_TREE")
        .env_remove("GIT_INDEX_FILE")
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

fn json_of(r: &Run) -> serde_json::Value {
    serde_json::from_str(&r.stdout).unwrap_or_else(|e| panic!("stdout is JSON: {e}\n{}", show(r)))
}

fn fixture_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/ont/docs-green")
}

fn copy_tree(src: &Path, dst: &Path) {
    std::fs::create_dir_all(dst).expect("mkdir");
    for e in std::fs::read_dir(src).expect("read fixture dir") {
        let p = e.expect("entry").path();
        let to = dst.join(p.file_name().expect("name"));
        if p.is_dir() {
            copy_tree(&p, &to);
        } else {
            std::fs::copy(&p, &to).expect("copy");
        }
    }
}

/// A fresh copy of the green fixture, OUTSIDE any git work tree (tempdir), so F-34 reads `not-checked`.
fn copy() -> tempfile::TempDir {
    let tmp = tempfile::tempdir().expect("tempdir");
    copy_tree(&fixture_root(), tmp.path());
    tmp
}

fn gate_at(repo: &Path, extra: &[&str]) -> Run {
    let mut args = vec!["lint", "contracts", "--gate", "shapes", "--format", "json"];
    args.extend_from_slice(extra);
    pv_in(repo, &args)
}

fn append(p: &Path, text: &str) {
    let mut s = std::fs::read_to_string(p).expect("read");
    s.push_str(text);
    std::fs::write(p, s).expect("write");
}

fn replace(p: &Path, from: &str, to: &str) {
    let s = std::fs::read_to_string(p).expect("read");
    assert_eq!(
        s.matches(from).count(),
        1,
        "exactly one `{from}` in {}",
        p.display()
    );
    std::fs::write(p, s.replacen(from, to, 1)).expect("write");
}

fn baseline(repo: &Path, readme: &[&str], readme_w: &[&str], claude: &[&str], claude_w: &[&str]) {
    let arr = |xs: &[&str]| serde_json::to_string(xs).expect("json");
    std::fs::write(
        repo.join("contracts/lint-baseline.json"),
        format!(
            "{{\n  \"armed_gates\": [\"shapes\"],\n  \"armed_shapes\": [\"shape-holder\", \"readme-doc\", \"claude-doc\", \"data-csv\"],\n  \"readme\": {{\"verified_commands\":{},\"withdrawn\":{}}},\n  \"claude_md\": {{\"verified_commands\":{},\"withdrawn\":{}}}\n}}\n",
            arr(readme),
            arr(readme_w),
            arr(claude),
            arr(claude_w)
        ),
    )
    .expect("baseline");
}

fn git(repo: &Path, args: &[&str]) {
    let out = Command::new("git")
        .arg("-C")
        .arg(repo)
        .args(args)
        .env_remove("GIT_DIR")
        .env_remove("GIT_WORK_TREE")
        .env_remove("GIT_INDEX_FILE")
        .output()
        .expect("git runs");
    assert!(
        out.status.success(),
        "git {args:?}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

/// A copy of the green fixture committed as the comparand.
fn committed_copy() -> tempfile::TempDir {
    let tmp = copy();
    let repo = tmp.path();
    git(repo, &["init", "-q", "-b", "main"]);
    git(repo, &["config", "user.email", "t@example.com"]);
    git(repo, &["config", "user.name", "t"]);
    git(repo, &["add", "-A"]);
    git(repo, &["commit", "-q", "-m", "comparand"]);
    tmp
}

fn strs(v: &serde_json::Value) -> Vec<&str> {
    v.as_array()
        .map(|a| a.iter().filter_map(serde_json::Value::as_str).collect())
        .unwrap_or_default()
}

const CHECK: &str = "bash scripts/check.sh";
const TEST: &str = "cargo test --lib";

#[test]
fn the_green_fixture_passes_with_every_doc_type_counted_and_the_sets_reported() {
    let t = copy();
    let r = gate_at(t.path(), &[]);
    assert_eq!(r.code, 0, "{}", show(&r));
    let v = json_of(&r);
    assert_eq!(v["verdict"], "Pass", "{}", show(&r));
    for ty in ["readme", "llm-context", "csv"] {
        assert_eq!(v["by_entity_type"][ty], 1, "{ty}\n{}", show(&r));
        assert_eq!(v["pc_extract"][ty], "fired", "{ty}\n{}", show(&r));
    }
    assert_eq!(strs(&v["readme"]["verified_commands"]), [CHECK, TEST]);
    assert_eq!(strs(&v["claude_md"]["verified_commands"]), [TEST]);
    assert_eq!(
        v["ratchets"]["measured_sets"],
        "not-checked",
        "outside a git work tree there is nothing to compare\n{}",
        show(&r)
    );
}

#[test]
fn readme_bad_command_is_refused_by_name() {
    let t = copy();
    append(&t.path().join("README.md"), "\n```bash\nmake ghost\n```\n");
    let r = gate_at(t.path(), &[]);
    assert_eq!(r.code, 1, "{}", show(&r));
    assert!(
        r.stdout
            .contains("readme:verifiedCommand unresolved: make ghost"),
        "{}",
        show(&r)
    );
}

#[test]
fn readme_tag_variants_are_claims_and_console_is_not() {
    for (open, close, claim) in [
        ("```{.Bash}", "```", true),
        ("~~~ shell", "~~~", true),
        ("```SH title=x", "```", true),
        ("```console", "```", false),
        ("```text", "```", false),
    ] {
        let t = copy();
        append(
            &t.path().join("README.md"),
            &format!("\n{open}\nmake ghost\n{close}\n"),
        );
        let r = gate_at(t.path(), &[]);
        let refused = r
            .stdout
            .contains("readme:verifiedCommand unresolved: make ghost");
        assert_eq!(refused, claim, "{open}\n{}", show(&r));
        assert_eq!(r.code, if claim { 1 } else { 0 }, "{open}\n{}", show(&r));
    }
}

#[test]
fn a_step_under_if_false_runs_nothing_so_its_command_is_not_a_verified_claim() {
    let t = copy();
    append(
        &t.path().join("README.md"),
        "\n```bash\nmake disabled-step\n```\n",
    );
    let r = gate_at(t.path(), &[]);
    assert_eq!(r.code, 1, "{}", show(&r));
    assert!(
        r.stdout
            .contains("readme:verifiedCommand unresolved: make disabled-step"),
        "{}",
        show(&r)
    );
}

#[test]
fn claude_md_bad_command_and_bad_path_are_refused_by_name() {
    let t = copy();
    append(
        &t.path().join("CLAUDE.md"),
        "\nSee `no/such/dir/`.\n\n```shell\nmake nope\n```\n",
    );
    let r = gate_at(t.path(), &[]);
    assert_eq!(r.code, 1, "{}", show(&r));
    assert!(
        r.stdout.contains("llm:command unresolved: make nope"),
        "{}",
        show(&r)
    );
    assert!(
        r.stdout
            .contains("llm:referencedPath unresolved: no/such/dir/"),
        "{}",
        show(&r)
    );
}

#[test]
fn claude_md_without_its_purpose_section_fails_the_shape() {
    let t = copy();
    replace(
        &t.path().join("CLAUDE.md"),
        "## Project Overview",
        "## Overview",
    );
    let r = gate_at(t.path(), &[]);
    assert_eq!(r.code, 1, "{}", show(&r));
    assert!(r.stdout.contains("purposeSection"), "{}", show(&r));
}

#[test]
fn csv_column_mismatch_is_refused_naming_file_and_row() {
    let t = copy();
    append(&t.path().join("data/train.csv"), "1,2,3,4\n");
    let r = gate_at(t.path(), &[]);
    assert_eq!(r.code, 1, "{}", show(&r));
    assert!(r.stdout.contains("data/train.csv"), "{}", show(&r));
    assert!(
        r.stdout.contains("row 4 has 4 column(s), the header has 3"),
        "{}",
        show(&r)
    );
}

#[test]
fn f33_a_committed_set_that_disagrees_with_the_live_one_is_red_naming_the_command() {
    let t = copy();
    baseline(t.path(), &[TEST], &[], &[TEST], &[]);
    let r = gate_at(t.path(), &[]);
    assert_eq!(r.code, 1, "{}", show(&r));
    assert!(r.stdout.contains("PV-ONT-013"), "{}", show(&r));
    assert!(
        r.stdout
            .contains(&format!("lacks the live claim `{CHECK}`")),
        "{}",
        show(&r)
    );
}

#[test]
fn f34_relabel_drops_claim_without_withdrawn_is_red_and_with_it_passes() {
    let t = committed_copy();
    let repo = t.path();
    replace(
        &repo.join("README.md"),
        "```bash\nbash scripts/check.sh",
        "```text\nbash scripts/check.sh",
    );
    // restamped verified_commands (F-33 green), but no withdrawal
    baseline(repo, &[TEST], &[], &[TEST], &[]);
    let r = gate_at(repo, &["--armed-baseline-ref", "HEAD"]);
    assert_eq!(r.code, 1, "{}", show(&r));
    let v = json_of(&r);
    assert_eq!(v["ratchets"]["measured_sets"], "checked", "{}", show(&r));
    assert!(r.stdout.contains("PV-ONT-014"), "{}", show(&r));
    assert!(
        r.stdout
            .contains(&format!("verified claim `{CHECK}` was dropped")),
        "{}",
        show(&r)
    );
    // the same drop, recorded
    baseline(repo, &[TEST], &[CHECK], &[TEST], &[]);
    let r = gate_at(repo, &["--armed-baseline-ref", "HEAD"]);
    assert_eq!(r.code, 0, "{}", show(&r));
    assert_eq!(json_of(&r)["ratchets"]["measured_sets"], "checked");
}

#[test]
fn f34_withdrawal_without_drop_is_red_naming_it() {
    let t = committed_copy();
    let repo = t.path();
    baseline(repo, &[CHECK, TEST], &[TEST], &[TEST], &[]);
    let r = gate_at(repo, &["--armed-baseline-ref", "HEAD"]);
    assert_eq!(r.code, 1, "{}", show(&r));
    assert!(
        r.stdout.contains(&format!(
            "withdrawn[] names `{TEST}`, which was not dropped"
        )),
        "{}",
        show(&r)
    );
}

#[test]
fn f34_ratchet_bootstrap_key_absent_at_the_comparand_is_checked_against_the_empty_set() {
    let t = copy();
    let repo = t.path();
    // the comparand predates the keys
    std::fs::write(
        repo.join("contracts/lint-baseline.json"),
        "{\n  \"armed_gates\": [\"shapes\"],\n  \"armed_shapes\": [\"shape-holder\", \"readme-doc\", \"claude-doc\", \"data-csv\"]\n}\n",
    )
    .expect("baseline");
    git(repo, &["init", "-q", "-b", "main"]);
    git(repo, &["config", "user.email", "t@example.com"]);
    git(repo, &["config", "user.name", "t"]);
    git(repo, &["add", "-A"]);
    git(repo, &["commit", "-q", "-m", "comparand without the keys"]);
    baseline(repo, &[CHECK, TEST], &[], &[TEST], &[]);
    let r = gate_at(repo, &["--armed-baseline-ref", "HEAD"]);
    assert_eq!(r.code, 0, "{}", show(&r));
    assert_eq!(json_of(&r)["ratchets"]["measured_sets"], "checked");
    // a withdrawal at bootstrap names a claim that was never in set(comparand) = ∅
    baseline(repo, &[CHECK, TEST], &["make gone"], &[TEST], &[]);
    let r = gate_at(repo, &["--armed-baseline-ref", "HEAD"]);
    assert_eq!(r.code, 1, "{}", show(&r));
    assert!(
        r.stdout
            .contains("withdrawn[] names `make gone`, which was not dropped"),
        "{}",
        show(&r)
    );
}

#[test]
fn f34_a_work_tree_with_no_comparand_ref_is_red_not_unchecked() {
    let t = committed_copy();
    // no origin/main, no explicit ref
    let r = gate_at(t.path(), &[]);
    assert_eq!(r.code, 1, "{}", show(&r));
    assert_eq!(json_of(&r)["ratchets"]["measured_sets"], "checked");
    assert!(
        r.stdout.contains("no comparand in a git work tree"),
        "{}",
        show(&r)
    );
}
