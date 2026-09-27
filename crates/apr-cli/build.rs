// Contract: apr-version-traceability-v1 F-VERSION-001..004 (paiml/aprender#597, #1862).
// Resolve APR_GIT_SHA with a multi-source fallback hierarchy:
//   1. APR_GIT_SHA_OVERRIDE env var (CI/release hook)
//   2. git rev-parse --short HEAD (dev builds from worktree or primary checkout)
//   3. committed .git-sha file (crates.io installs)
//   4. "v{CARGO_PKG_VERSION}+no-git" (informative fallback — never bare "unknown")
//
// The resolution and its rerun-if-changed triggers live in the shared
// `aprender-build-sha` crate (#4219) so every workspace [[bin]] stamps the same SHA.
fn main() {
    build_sha::emit();

    // EXT-001 I-5: a dirty or unidentifiable engine is refused at start_run.
    // Same resolution order as `emit`, so the flag describes the stamped SHA.
    let override_env = std::env::var("APR_GIT_SHA_OVERRIDE").ok();
    let git_head = run_git(&["rev-parse", "--short", "HEAD"]);
    let sha_file = std::fs::read_to_string(".git-sha").ok();
    let version = std::env::var("CARGO_PKG_VERSION").ok();
    let sha = build_sha::resolve(
        override_env.as_deref(),
        git_head.as_deref(),
        sha_file.as_deref(),
        version.as_deref(),
    );
    println!("cargo:rustc-env=APR_GIT_DIRTY={}", resolve_git_dirty(&sha));
    // Tracked edits anywhere in the workspace change the dirty flag.
    println!("cargo:rerun-if-changed=..");
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
