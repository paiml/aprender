// G0.1 (contracts/bin-aprender-contracts-cli--pv-v1.yaml, #4476): `pv --version`
// names the commit it was built from — the first 9 hex of the git sha — so a
// stale pv can no longer read as HEAD. The semver is a WORKSPACE version shared
// by every worktree; without the sha two trees print byte-identical lines.
//
// PV_GIT_SHA, same fallback order as apr-cli's APR_GIT_SHA:
//   1. APR_GIT_SHA_OVERRIDE (the CI/release hook both binaries share)
//   2. git rev-parse --short=9 HEAD (dev builds, primary checkout or worktree)
//   3. committed .git-sha (crates.io installs)
//   4. "no-git" — never a made-up sha
fn main() {
    println!("cargo:rustc-env=PV_GIT_SHA={}", resolve_git_sha());
    register_git_rerun_triggers();
    println!("cargo:rerun-if-env-changed=APR_GIT_SHA_OVERRIDE");
    println!("cargo:rerun-if-changed=.git-sha");
}

fn register_git_rerun_triggers() {
    // In a worktree `.git` is a file pointer, so ask git where HEAD and refs live.
    if let Some(git_dir) = run_git(&["rev-parse", "--git-dir"]) {
        let head = format!("{git_dir}/HEAD");
        if std::path::Path::new(&head).exists() {
            println!("cargo:rerun-if-changed={head}");
        }
    }
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
    (!s.is_empty()).then_some(s)
}

fn short9(s: &str) -> String {
    s.chars().take(9).collect()
}

fn resolve_git_sha() -> String {
    if let Ok(s) = std::env::var("APR_GIT_SHA_OVERRIDE") {
        let s = s.trim();
        if !s.is_empty() {
            return short9(s);
        }
    }
    if let Some(sha) = run_git(&["rev-parse", "--short=9", "HEAD"]) {
        return short9(&sha);
    }
    if let Ok(contents) = std::fs::read_to_string(".git-sha") {
        let s = contents.trim();
        if !s.is_empty() {
            return short9(s);
        }
    }
    "no-git".to_string()
}
