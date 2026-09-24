// Contract: apr-version-traceability-v1 F-VERSION-001..004 (paiml/aprender#597, #1862).
// Resolve APR_GIT_SHA with a multi-source fallback hierarchy:
//   1. APR_GIT_SHA_OVERRIDE env var (CI/release hook)
//   2. git rev-parse --short HEAD (dev builds from worktree or primary checkout)
//   2b. <git-dir>/HEAD read as files (a checkout git refuses to open, #4254)
//   3. committed .git-sha file (crates.io installs)
//   4. "v{CARGO_PKG_VERSION}+no-git" (informative fallback — never bare "unknown")
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

    // 2b. #4254: the checkout's own HEAD, read as files. The release assets are
    // built as root in a container over a workspace another uid owns, where git
    // refuses the repository ("dubious ownership") and step 2 returns nothing —
    // every v0.69.1 asset reported "+no-git". Reading HEAD needs no git binary.
    if let Some(sha) = read_head_sha() {
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

/// The commit `HEAD` names, abbreviated as `rev-parse --short` abbreviates it in
/// this repository (9 hex digits), read from the checkout's files: `.git` found
/// upward from the manifest, a worktree's `gitdir:` pointer followed, a symbolic
/// ref resolved through loose refs and then `packed-refs`. `None` when any step
/// is missing or the result is not a full hex object id.
fn read_head_sha() -> Option<String> {
    use std::path::{Path, PathBuf};

    let start = PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR")?);
    let dot_git = start
        .ancestors()
        .map(|d| d.join(".git"))
        .find(|p| p.exists())?;
    let git_dir = if dot_git.is_dir() {
        dot_git
    } else {
        let pointer = std::fs::read_to_string(&dot_git).ok()?;
        let target = PathBuf::from(pointer.trim().strip_prefix("gitdir:")?.trim());
        if target.is_absolute() {
            target
        } else {
            dot_git.parent()?.join(target)
        }
    };
    let common_dir = match std::fs::read_to_string(git_dir.join("commondir")) {
        Ok(rel) => git_dir.join(rel.trim()),
        Err(_) => git_dir.clone(),
    };
    let is_oid = |s: &str| s.len() == 40 && s.bytes().all(|b| b.is_ascii_hexdigit());
    let head = std::fs::read_to_string(git_dir.join("HEAD")).ok()?;
    let head = head.trim();
    let oid = match head.strip_prefix("ref:") {
        None => head.to_string(),
        Some(name) => {
            let name = name.trim();
            let loose = |dir: &Path| std::fs::read_to_string(dir.join(name)).ok();
            match loose(&git_dir).or_else(|| loose(&common_dir)) {
                Some(oid) => oid.trim().to_string(),
                None => std::fs::read_to_string(common_dir.join("packed-refs"))
                    .ok()?
                    .lines()
                    .filter_map(|l| l.split_once(' '))
                    .find(|(_, r)| *r == name)
                    .map(|(oid, _)| oid.to_string())?,
            }
        }
    };
    is_oid(&oid).then(|| oid[..9].to_string())
}
