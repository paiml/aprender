// Contract: apr-version-traceability-v1 F-VERSION-001 (paiml/aprender#597).
// Resolve APR_GIT_SHA with a multi-source fallback hierarchy:
//   1. APR_GIT_SHA_OVERRIDE env var (CI/release hook)
//   2. git rev-parse --short HEAD (dev builds from worktree)
//   3. committed .git-sha file (crates.io installs)
//   4. "v{CARGO_PKG_VERSION}+no-git" (informative fallback — never bare "unknown")
fn main() {
    let sha = resolve_git_sha();
    println!("cargo:rustc-env=APR_GIT_SHA={sha}");

    // Only set rerun-if-changed when .git exists (avoids rebuild-every-time
    // on crates.io installs where no .git directory is present)
    let git_head = std::path::Path::new("../../.git/HEAD");
    if git_head.exists() {
        println!("cargo:rerun-if-changed=../../.git/HEAD");
        println!("cargo:rerun-if-changed=../../.git/refs/heads/");
    }
    println!("cargo:rerun-if-env-changed=APR_GIT_SHA_OVERRIDE");
    println!("cargo:rerun-if-changed=.git-sha");
}

fn resolve_git_sha() -> String {
    // 1. Explicit override from CI/release
    if let Ok(s) = std::env::var("APR_GIT_SHA_OVERRIDE") {
        let s = s.trim();
        if !s.is_empty() {
            return s.to_string();
        }
    }

    // 2. Live git hash from worktree
    let output = std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output();
    if let Ok(o) = output {
        if o.status.success() {
            let s = String::from_utf8_lossy(&o.stdout).trim().to_string();
            if !s.is_empty() {
                return s;
            }
        }
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
