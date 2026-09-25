// Contract: apr-version-traceability-v1 F-VERSION-001..004 (paiml/aprender#597, #1862).
// Resolve APR_GIT_SHA with a multi-source fallback hierarchy:
//   1. APR_GIT_SHA_OVERRIDE env var (CI/release hook)
//   2. git rev-parse --short HEAD (dev builds from worktree or primary checkout)
//   3. committed .git-sha file (crates.io installs)
//   4. "v{CARGO_PKG_VERSION}+no-git" (informative fallback — never bare "unknown")
fn main() {
    let sha = resolve_git_sha();
    println!("cargo:rustc-env=APR_GIT_SHA={sha}");
    // EXT-001 I-5: a dirty or unidentifiable engine is refused at start_run.
    println!("cargo:rustc-env=APR_GIT_DIRTY={}", resolve_git_dirty(&sha));

    // Rerun build when HEAD moves. In a primary checkout `.git` is a directory;
    // in a worktree `.git` is a file pointer to `<common-dir>/worktrees/<name>/`.
    // Resolve the actual git directory(ies) via `git rev-parse` so both layouts
    // are watched correctly (#1862).
    register_git_rerun_triggers();

    println!("cargo:rerun-if-env-changed=APR_GIT_SHA_OVERRIDE");
    // Tracked edits anywhere in the workspace change the dirty flag.
    println!("cargo:rerun-if-changed=..");
    println!("cargo:rerun-if-changed=.git-sha");
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

    // 2. Live git hash from worktree (works for both primary checkouts and worktrees)
    if let Some(sha) = run_git(&["rev-parse", "--short", "HEAD"]) {
        return sha;
    }

    // 3. Committed .git-sha fallback (updated by release script at publish time)
    if let Ok(contents) = std::fs::read_to_string(".git-sha") {
        let s = contents.trim().to_string();
        if !s.is_empty() {
            return s;
        }
    }

    // 4. Informative fallback — never bare "unknown"
    let version = std::env::var("CARGO_PKG_VERSION").unwrap_or_else(|_| "0.0.0".to_string());
    format!("v{version}+no-git")
}

/// `"1"` when tracked files differ from HEAD, `"0"` when they do not or the
/// sha came from a release (override or committed `.git-sha`), `"unknown"`
/// when neither git nor a release sha identifies the build (EXT-001 I-5).
fn resolve_git_dirty(sha: &str) -> &'static str {
    if sha.ends_with("+no-git") {
        return "unknown";
    }
    let from_release = std::env::var("APR_GIT_SHA_OVERRIDE").is_ok_and(|s| !s.trim().is_empty());
    if from_release {
        return "0";
    }
    let Ok(out) = std::process::Command::new("git")
        .args(["status", "--porcelain", "--untracked-files=no"])
        .output()
    else {
        return "0"; // no git: the sha came from the committed .git-sha
    };
    if !out.status.success() {
        return "0";
    }
    if out.stdout.is_empty() {
        "0"
    } else {
        "1"
    }
}
