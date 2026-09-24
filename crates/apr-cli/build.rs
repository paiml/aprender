// Contract: apr-version-traceability-v1 F-VERSION-001..004 (paiml/aprender#597, #1862).
// Resolve APR_GIT_SHA with a multi-source fallback hierarchy:
//   1. APR_GIT_SHA_OVERRIDE env var (CI/release hook)
//   2. git rev-parse --short HEAD (dev builds from worktree or primary checkout)
//   2b. the SHA read straight from `.git` — no git binary (#4254: the release asset
//       said "v0.69.1+no-git" because `git` fails inside the sibling build container
//       over a bind-mounted checkout, while the checkout's `.git` is right there)
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

    // 2b. Read the checkout's own `.git` when the git binary cannot answer (#4254)
    if let Some(sha) = read_head_sha_from_dot_git() {
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

/// #4254: the commit HEAD names, read from `.git` without the git binary.
///
/// Walks up from `CARGO_MANIFEST_DIR` to the first `.git`, which is either the git
/// directory or, in a worktree, a file `gitdir: <path>` (whose `commondir` holds the
/// shared refs). HEAD is a bare SHA when detached (a tag checkout) or `ref: <name>`,
/// resolved as a loose ref, then through `packed-refs`. Short form, 9 hex digits.
fn read_head_sha_from_dot_git() -> Option<String> {
    use std::path::{Path, PathBuf};
    let start = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").ok()?);
    let dot_git = start
        .ancestors()
        .map(|d| d.join(".git"))
        .find(|p| p.exists())?;
    let git_dir: PathBuf = if dot_git.is_file() {
        let text = std::fs::read_to_string(&dot_git).ok()?;
        let rel = text.trim().strip_prefix("gitdir:")?.trim();
        let p = Path::new(rel);
        if p.is_absolute() {
            p.to_path_buf()
        } else {
            dot_git.parent()?.join(p)
        }
    } else {
        dot_git
    };
    let common_dir = match std::fs::read_to_string(git_dir.join("commondir")) {
        Ok(c) => {
            let p = Path::new(c.trim());
            if p.is_absolute() {
                p.to_path_buf()
            } else {
                git_dir.join(p)
            }
        }
        Err(_) => git_dir.clone(),
    };
    let head = std::fs::read_to_string(git_dir.join("HEAD")).ok()?;
    let head = head.trim();
    let full = match head.strip_prefix("ref:") {
        None => head.to_string(),
        Some(name) => {
            let name = name.trim();
            [&git_dir, &common_dir]
                .iter()
                .find_map(|d| std::fs::read_to_string(d.join(name)).ok())
                .map(|s| s.trim().to_string())
                .or_else(|| {
                    let packed = std::fs::read_to_string(common_dir.join("packed-refs")).ok()?;
                    packed.lines().find_map(|l| {
                        let (sha, r) = l.split_once(' ')?;
                        (r.trim() == name).then(|| sha.to_string())
                    })
                })?
        }
    };
    let is_sha = full.len() >= 40 && full.bytes().all(|b| b.is_ascii_hexdigit());
    is_sha.then(|| full[..9].to_string())
}
