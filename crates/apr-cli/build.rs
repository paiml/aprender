// Contract: apr-version-traceability-v1 F-VERSION-001..004 (paiml/aprender#597, #1862).
// Resolve APR_GIT_SHA with a multi-source fallback hierarchy:
//   1. APR_GIT_SHA_OVERRIDE env var (CI/release hook)
//   2. .cargo_vcs_info.json `git.sha1` (crates.io / packaged builds, #4110)
//   3. git rev-parse --short HEAD, retried with the checkout marked safe.directory (#4110)
//   4. committed .git-sha file
//   5. "v{CARGO_PKG_VERSION}+no-git" (informative fallback — never bare "unknown")
//
// The resolution and its rerun-if-changed triggers live in the shared
// `aprender-build-sha` crate (#4219) so every workspace [[bin]] stamps the same SHA.
fn main() {
    let sha = build_sha::emit();

    // EXT-001 I-5: a dirty or unidentifiable engine is refused at start_run.
    // Derived from the SHA `emit` stamped, so the flag and the SHA never disagree.
    println!("cargo:rustc-env=APR_GIT_DIRTY={}", resolve_git_dirty(&sha));
    // Tracked edits under crates/ change the dirty flag. Edits elsewhere in the
    // repo (contracts/, docs/) do not rerun this script.
    println!("cargo:rerun-if-changed=..");
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
