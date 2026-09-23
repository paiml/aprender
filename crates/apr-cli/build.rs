// Contract: apr-version-traceability-v1 F-VERSION-001..004 (paiml/aprender#597, #1862).
// Resolve APR_GIT_SHA with a multi-source fallback hierarchy:
//   1. APR_GIT_SHA_OVERRIDE env var (CI/release hook)
//   2. .cargo_vcs_info.json `git.sha1` — written by `cargo package` into every published
//      tarball, so a crates.io install knows its commit (#4110). Ahead of `git rev-parse`
//      because an unpacked crate may sit inside SOME OTHER repository whose HEAD is not ours
//   3. git rev-parse --short HEAD (dev builds from worktree or primary checkout; retried with
//      the enclosing checkout marked safe.directory when git refuses a checkout owned by another
//      user — binary-release.yml builds as root in a container over the runner's checkout)
//   4. committed .git-sha file
//   5. "v{CARGO_PKG_VERSION}+no-git" (informative fallback — never bare "unknown")
fn main() {
    let sha = resolve_git_sha();
    println!("cargo:rustc-env=APR_GIT_SHA={sha}");

    // Rerun build when HEAD moves. In a primary checkout `.git` is a directory;
    // in a worktree `.git` is a file pointer to `<common-dir>/worktrees/<name>/`.
    // Resolve the actual git directory(ies) via `git rev-parse` so both layouts
    // are watched correctly (#1862).
    register_git_rerun_triggers();

    println!("cargo:rerun-if-env-changed=APR_GIT_SHA_OVERRIDE");
    println!("cargo:rerun-if-changed=.git-sha");
    println!("cargo:rerun-if-changed=.cargo_vcs_info.json");
}

fn register_git_rerun_triggers() {
    // Worktree-local HEAD: <git-dir>/HEAD. In a worktree this is the per-worktree
    // HEAD pointer; in a primary checkout it's the same as <common-dir>/HEAD.
    if let Some(git_dir) = run_git(&["rev-parse", "--git-dir"]) {
        let head = format!("{git_dir}/HEAD");
        if std::path::Path::new(&head).exists() {
            println!("cargo:rerun-if-changed={head}");
        }
    }
    // Shared refs live under <common-dir>/refs/heads/ regardless of layout.
    if let Some(common_dir) = run_git(&["rev-parse", "--git-common-dir"]) {
        let refs = format!("{common_dir}/refs/heads");
        if std::path::Path::new(&refs).exists() {
            println!("cargo:rerun-if-changed={refs}");
        }
    }
}

fn run_git(args: &[&str]) -> Option<String> {
    let out = std::process::Command::new("git").args(args).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if s.is_empty() {
        None
    } else {
        Some(s)
    }
}

fn resolve_git_sha() -> String {
    // 1. Explicit override from CI/release
    if let Ok(s) = std::env::var("APR_GIT_SHA_OVERRIDE") {
        let s = s.trim();
        if !s.is_empty() {
            return s.to_string();
        }
    }

    // 2. The commit `cargo package` recorded (crates.io / packaged builds, no .git)
    if let Some(sha) = vcs_info_sha() {
        return sha;
    }

    // 3. Live git hash from worktree (works for both primary checkouts and worktrees).
    //    `trusted_git_retry`: the release binaries are built as root in a container over the
    //    runner-owned checkout, where git refuses the repo as "dubious ownership" (#4110)
    let head = ["rev-parse", "--short", "HEAD"];
    if let Some(sha) = run_git(&head).or_else(|| trusted_git_retry(&head)) {
        return sha;
    }

    // 4. Committed .git-sha fallback (updated by release script at publish time)
    if let Ok(contents) = std::fs::read_to_string(".git-sha") {
        let s = contents.trim().to_string();
        if !s.is_empty() {
            return s;
        }
    }

    // 5. Informative fallback — never bare "unknown"
    let version = std::env::var("CARGO_PKG_VERSION").unwrap_or_else(|_| "0.0.0".to_string());
    format!("v{version}+no-git")
}

/// `git.sha1` from `.cargo_vcs_info.json`, shortened to 9 hex digits (what `git rev-parse
/// --short` prints in this repository), with `-dirty` when cargo packaged a dirty tree.
/// Parsed by hand: a build script has no serde, and the file is cargo's own fixed shape.
fn vcs_info_sha() -> Option<String> {
    let dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
    let text = std::fs::read_to_string(format!("{dir}/.cargo_vcs_info.json")).ok()?;
    let after = &text[text.find("\"sha1\"")? + "\"sha1\"".len()..];
    let open = after.find('"')?;
    let value = &after[open + 1..];
    let sha = &value[..value.find('"')?];
    if sha.len() != 40 || !sha.bytes().all(|b| b.is_ascii_hexdigit()) {
        return None;
    }
    let dirty = text.contains("\"dirty\": true") || text.contains("\"dirty\":true");
    Some(format!(
        "{}{}",
        &sha[..9],
        if dirty { "-dirty" } else { "" }
    ))
}

/// `git <args>` again, with the checkout that encloses this package marked `safe.directory`.
///
/// git refuses a repository owned by another user ("detected dubious ownership"), and the release
/// lane builds as root in a container over the runner-owned checkout, so every published binary
/// printed `+no-git` (#4110). The exception goes in a private GLOBAL config under OUT_DIR:
/// git ignores `safe.directory` given by `-c` or GIT_CONFIG_* before 2.39 (measured, 2.34.1).
/// Only the checkout found by walking up from CARGO_MANIFEST_DIR is trusted, never `*`, and
/// `rev-parse` runs no hooks. This build is already compiling that checkout's code.
fn trusted_git_retry(args: &[&str]) -> Option<String> {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").ok()?;
    let top = std::path::Path::new(&manifest)
        .ancestors()
        .find(|d| d.join(".git").exists())?;
    let config =
        std::path::Path::new(&std::env::var("OUT_DIR").ok()?).join("apr-safe-directory.gitconfig");
    std::fs::write(
        &config,
        format!("[safe]\n\tdirectory = {}\n", top.display()),
    )
    .ok()?;
    let out = std::process::Command::new("git")
        .args(args)
        .env("GIT_CONFIG_GLOBAL", &config)
        .output()
        .ok()?;
    let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
    (out.status.success() && !s.is_empty()).then_some(s)
}
