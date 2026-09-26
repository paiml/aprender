//! #4110 — a crates.io install of apr names its commit, never `+no-git`. Since #4219 every
//! workspace [[bin]] stamps its SHA through this crate, so the table runs against it.
//!
//! The 0.69.1 post-publish install printed `apr 0.69.1 (v0.69.1+no-git)` (0.68.1 did too) although
//! the published tarball carried the commit: `cargo package` writes `.cargo_vcs_info.json` into
//! every crate. These tests compile the SHIPPED `lib.rs` plus a `fn main() { emit() }` with rustc and run it the way cargo runs a
//! build script (CARGO_MANIFEST_DIR, CARGO_PKG_VERSION, cwd = the package dir) in a temp package
//! dir with no `.git` of ours above it. The vcs file is byte-exact from the real apr-cli-0.69.1 crate.
//! The release binaries printed `+no-git` too: binary-release.yml builds as root in a container over
//! the runner-owned checkout, and git refuses it ("dubious ownership"); row `foreign-owner`.
//! `planted_regressions_turn_red` re-runs the table on three planted copies of `lib.rs` — the vcs
//! rung removed (the pre-#4110 script), the vcs rung behind `git rev-parse`, and the safe.directory
//! retry removed — and requires each to fail, so the table cannot pass vacuously.

use std::path::{Path, PathBuf};
use std::process::Command;

const LIB_RS: &str = include_str!("lib.rs");

/// What a package's `build.rs` compiles to: this crate's code, entered through [`super::emit`].
fn shipped() -> String {
    format!("{LIB_RS}\nfn main() {{\n    emit();\n}}\n")
}
const VCS_CALL: &str = "    if let Some(sha) = vcs_info_sha() {";

/// Byte-exact `.cargo_vcs_info.json` of apr-cli-0.69.1 as downloaded from crates.io.
const VCS_REAL: &str = "{\n  \"git\": {\n    \"sha1\": \"d8a6df53a6c7104ef4eabfd2284908f93f1d6a3f\"\n  },\n  \"path_in_vcs\": \"crates/apr-cli\"\n}";
const VCS_DIRTY: &str = "{\n  \"git\": {\n    \"sha1\": \"d8a6df53a6c7104ef4eabfd2284908f93f1d6a3f\",\n    \"dirty\": true\n  },\n  \"path_in_vcs\": \"crates/apr-cli\"\n}";
const VCS_BAD: &str =
    "{\n  \"git\": {\n    \"sha1\": \"not-a-sha\"\n  },\n  \"path_in_vcs\": \"crates/apr-cli\"\n}";

fn compile(source: &str, dir: &Path, name: &str) -> PathBuf {
    let src = dir.join(format!("{name}.rs"));
    let bin = dir.join(name);
    std::fs::write(&src, source).expect("write build script copy");
    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".to_string());
    let out = Command::new(rustc)
        .args([
            "--edition",
            "2021",
            "--crate-name",
            "build_script_build",
            "-o",
        ])
        .arg(&bin)
        .arg(&src)
        .output()
        .expect("rustc must be runnable: this test compiles the shipped build-sha source");
    assert!(
        out.status.success(),
        "rustc rejected {name}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    bin
}

fn git(dir: &Path, args: &[&str]) -> String {
    let out = Command::new("git")
        .args([
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "user.email=f@x",
            "-c",
            "user.name=f",
        ])
        .args(args)
        .current_dir(dir)
        .output()
        .expect("git must be runnable");
    assert!(
        out.status.success(),
        "git {args:?}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).trim().to_string()
}

/// A fresh package dir under `root`; the vcs file written when given; optionally its own repo.
fn package(root: &Path, name: &str, vcs: Option<&str>) -> PathBuf {
    let dir = root.join(name);
    std::fs::create_dir_all(&dir).expect("mkdir package");
    if let Some(v) = vcs {
        std::fs::write(dir.join(".cargo_vcs_info.json"), v).expect("write vcs file");
    }
    dir
}

/// Runs the compiled build script like cargo does; returns the APR_GIT_SHA it emits.
/// `ceiling` stops git's upward search there (None: search freely, as a real install would).
fn run(bin: &Path, pkg: &Path, ceiling: Option<&Path>, env: &[(&str, &str)]) -> String {
    let out_dir = pkg.join("target-out");
    std::fs::create_dir_all(&out_dir).expect("mkdir OUT_DIR");
    let mut cmd = Command::new(bin);
    cmd.current_dir(pkg)
        .env("CARGO_MANIFEST_DIR", pkg)
        .env("OUT_DIR", &out_dir)
        .env("CARGO_PKG_VERSION", "9.9.9")
        .env_remove("APR_GIT_SHA_OVERRIDE")
        .env_remove("GIT_DIR")
        .env_remove("GIT_WORK_TREE")
        .env(
            "GIT_CEILING_DIRECTORIES",
            ceiling.unwrap_or_else(|| Path::new("/nonexistent")),
        )
        .envs(env.iter().copied());
    let out = cmd.output().expect("run build script");
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .find_map(|l| l.strip_prefix("cargo:rustc-env=APR_GIT_SHA="))
        .unwrap_or("<none>")
        .to_string()
}

/// The case table. Returns the names of the rows that landed wrong (empty = all as expected).
fn table(build_rs: &str, label: &str) -> Vec<String> {
    let tmp = tempfile::tempdir().expect("tempdir");
    let root = tmp.path();
    let bin = compile(build_rs, root, label);
    let mut wrong = Vec::new();
    let mut expect = |row: &str, got: String, want: &str| {
        if got != want {
            wrong.push(format!("{row}: got {got:?}, want {want:?}"));
        }
    };
    let pkgs = root.join("pkgs");
    let p = package(&pkgs, "packaged", Some(VCS_REAL));
    expect("packaged", run(&bin, &p, Some(&pkgs), &[]), "d8a6df53a");
    // an unrelated repository ABOVE the unpacked crate, with a commit, and no ceiling between them
    let foreign = package(&pkgs, "foreign", None);
    git(&foreign, &["init", "-q"]);
    git(
        &foreign,
        &["commit", "-q", "--allow-empty", "-m", "foreign"],
    );
    let p = package(&foreign, "crate", Some(VCS_REAL));
    expect("foreign-repo", run(&bin, &p, None, &[]), "d8a6df53a");
    let p = package(&pkgs, "dirty", Some(VCS_DIRTY));
    expect("dirty", run(&bin, &p, Some(&pkgs), &[]), "d8a6df53a-dirty");
    let p = package(&pkgs, "malformed", Some(VCS_BAD));
    expect(
        "malformed",
        run(&bin, &p, Some(&pkgs), &[]),
        "v9.9.9+no-git",
    );
    let p = package(&pkgs, "no-vcs", None);
    expect("no-vcs", run(&bin, &p, Some(&pkgs), &[]), "v9.9.9+no-git");
    let p = package(&pkgs, "override", Some(VCS_REAL));
    expect(
        "override",
        run(
            &bin,
            &p,
            Some(&pkgs),
            &[("APR_GIT_SHA_OVERRIDE", "release123")],
        ),
        "release123",
    );
    // a dev checkout: no vcs file, its own repo -> `git rev-parse --short HEAD` (worktree freshness)
    let dev = package(&pkgs, "dev", None);
    git(&dev, &["init", "-q"]);
    git(&dev, &["commit", "-q", "--allow-empty", "-m", "dev"]);
    let head = git(&dev, &["rev-parse", "--short", "HEAD"]);
    expect("dev-checkout", run(&bin, &dev, None, &[]), &head);
    // the release lane: root in a container over the runner-owned checkout. git's own test knob
    // makes it see another owner; the precondition proves plain git really refuses the repo here
    let other_owner = [("GIT_TEST_ASSUME_DIFFERENT_OWNER", "1")];
    let refused = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(&dev)
        .envs(other_owner)
        .output();
    assert!(
        !refused.expect("git").status.success(),
        "cannot simulate a checkout owned by another user: this git ignores GIT_TEST_ASSUME_DIFFERENT_OWNER"
    );
    expect("foreign-owner", run(&bin, &dev, None, &other_owner), &head);
    wrong
}

#[test]
fn packaged_build_rs_names_its_commit() {
    let wrong = table(&shipped(), "shipped");
    assert!(
        wrong.is_empty(),
        "shipped build-sha rows landed wrong: {wrong:#?}"
    );
}

#[test]
fn planted_regressions_turn_red() {
    let shipped = shipped();
    assert_eq!(
        shipped.matches(VCS_CALL).count(),
        1,
        "the vcs rung call site moved; re-anchor the plants"
    );
    let no_rung = shipped.replace(VCS_CALL, "    if let Some(sha) = None::<String> {");
    let git_first = shipped.replace(
        VCS_CALL,
        "    if let Some(sha) = run_git(&[\"rev-parse\", \"--short\", \"HEAD\"]).or_else(vcs_info_sha) {",
    );
    let w1 = table(&no_rung, "plant_no_rung");
    assert!(
        w1.iter().any(|r| r.starts_with("packaged:")),
        "no-vcs-rung plant survived: {w1:#?}"
    );
    // #4254 split the chain across lines and appended the `.git` read after the retry.
    let retry = "\n        .or_else(|| trusted_git_retry(&head))";
    assert_eq!(
        shipped.matches(retry).count(),
        1,
        "the git retry call site moved; re-anchor the plant"
    );
    let w3 = table(&shipped.replace(retry, ""), "plant_no_retry");
    assert!(
        w3.iter().any(|r| r.starts_with("foreign-owner:")),
        "no-retry plant survived: {w3:#?}"
    );
    let w2 = table(&git_first, "plant_git_first");
    assert!(
        w2.iter().any(|r| r.starts_with("foreign-repo:")),
        "git-first plant survived: {w2:#?}"
    );
    assert!(
        !w2.iter().any(|r| r.starts_with("packaged:")),
        "git-first plant should still pass packaged: {w2:#?}"
    );
}
