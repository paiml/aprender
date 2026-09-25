//! Stamp `APR_GIT_SHA` into a binary at build time (#4219).
//!
//! Call [`emit`] from a package's `build.rs`; the binary then reads the value
//! with `env!("APR_GIT_SHA")` and prints `<name> <version> (<sha>)` from
//! `--version`, the shape `apr` has printed since #597.
//!
//! Contract: apr-version-traceability-v1 F-VERSION-001..004 (paiml/aprender#597,
//! #1862). `APR_GIT_SHA` is resolved with a multi-source fallback hierarchy:
//!   1. `APR_GIT_SHA_OVERRIDE` env var (CI/release hook)
//!   2. `git rev-parse --short HEAD` (dev builds from worktree or primary checkout)
//!   3. committed `.git-sha` file (crates.io installs)
//!   4. `v{CARGO_PKG_VERSION}+no-git` (informative fallback — never bare "unknown")
//!
//! This logic was moved out of `crates/apr-cli/build.rs` unchanged; only the
//! pure resolution order was factored into [`resolve`] so it can be tested
//! without running git.

/// Resolve `APR_GIT_SHA`, print it as `cargo:rustc-env`, and register the
/// rerun-if-changed triggers. Call once from `build.rs`.
pub fn emit() {
    let sha = resolve_git_sha();
    println!("cargo:rustc-env=APR_GIT_SHA={sha}");

    // Rerun build when HEAD moves. In a primary checkout `.git` is a directory;
    // in a worktree `.git` is a file pointer to `<common-dir>/worktrees/<name>/`.
    // Resolve the actual git directory(ies) via `git rev-parse` so both layouts
    // are watched correctly (#1862).
    for path in git_rerun_triggers() {
        println!("cargo:rerun-if-changed={path}");
    }

    println!("cargo:rerun-if-env-changed=APR_GIT_SHA_OVERRIDE");
    println!("cargo:rerun-if-changed=.git-sha");
}

/// The pure resolution order. Each argument is the raw result of one source,
/// `None` when that source was unavailable. The first non-empty (after trim)
/// source wins; `v{version}+no-git` is the fallback.
#[must_use]
pub fn resolve(
    override_env: Option<&str>,
    git_head: Option<&str>,
    sha_file: Option<&str>,
    version: Option<&str>,
) -> String {
    // 1. Explicit override from CI/release
    if let Some(s) = nonblank(override_env) {
        return s.to_string();
    }
    // 2. Live git hash from worktree (works for both primary checkouts and worktrees)
    if let Some(s) = nonblank(git_head) {
        return s.to_string();
    }
    // 3. Committed .git-sha fallback (updated by release script at publish time)
    if let Some(s) = nonblank(sha_file) {
        return s.to_string();
    }
    // 4. Informative fallback — never bare "unknown"
    format!("v{}+no-git", version.unwrap_or("0.0.0"))
}

fn nonblank(s: Option<&str>) -> Option<&str> {
    s.map(str::trim).filter(|s| !s.is_empty())
}

/// The git paths whose change must rerun the build script. Returned rather
/// than printed so a test can read them.
fn git_rerun_triggers() -> Vec<String> {
    let mut paths = Vec::new();
    // Worktree-local HEAD: <git-dir>/HEAD. In a worktree this is the per-worktree
    // HEAD pointer; in a primary checkout it's the same as <common-dir>/HEAD.
    if let Some(git_dir) = run_git(&["rev-parse", "--git-dir"]) {
        let head = format!("{git_dir}/HEAD");
        if std::path::Path::new(&head).exists() {
            paths.push(head);
        }
    }
    // Shared refs live under <common-dir>/refs/heads/ regardless of layout.
    if let Some(common_dir) = run_git(&["rev-parse", "--git-common-dir"]) {
        let refs = format!("{common_dir}/refs/heads");
        if std::path::Path::new(&refs).exists() {
            paths.push(refs);
        }
    }
    paths
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
    resolve_git_sha_with(
        |k| std::env::var(k).ok(),
        run_git,
        |p| std::fs::read_to_string(p).ok(),
    )
}

/// [`resolve_git_sha`] over injected sources, so a test can drive every branch
/// without touching the process environment.
fn resolve_git_sha_with(
    env: impl Fn(&str) -> Option<String>,
    git: impl Fn(&[&str]) -> Option<String>,
    read: impl Fn(&str) -> Option<String>,
) -> String {
    // Sources are consulted lazily and in order, as the original apr-cli
    // build.rs did: git is not run when the override is set, and `.git-sha`
    // is not read when git answers.
    let override_env = env("APR_GIT_SHA_OVERRIDE");
    if let Some(s) = nonblank(override_env.as_deref()) {
        return s.to_string();
    }
    let git_head = git(&["rev-parse", "--short", "HEAD"]);
    let sha_file = match git_head {
        Some(_) => None,
        None => read(".git-sha"),
    };
    let version = env("CARGO_PKG_VERSION");
    resolve(
        None,
        git_head.as_deref(),
        sha_file.as_deref(),
        version.as_deref(),
    )
}

#[cfg(test)]
mod tests {
    use super::{git_rerun_triggers, resolve, resolve_git_sha, resolve_git_sha_with, run_git};

    #[test]
    fn override_wins_over_everything() {
        assert_eq!(
            resolve(
                Some(" abc123456 \n"),
                Some("def456789"),
                Some("0123456789"),
                Some("1.2.3")
            ),
            "abc123456"
        );
    }

    #[test]
    fn blank_override_falls_through_to_git() {
        assert_eq!(
            resolve(
                Some("  "),
                Some("def456789"),
                Some("0123456789"),
                Some("1.2.3")
            ),
            "def456789"
        );
        assert_eq!(resolve(None, Some("def456789"), None, None), "def456789");
    }

    #[test]
    fn sha_file_used_only_without_git() {
        assert_eq!(
            resolve(None, None, Some("0123456789\n"), Some("1.2.3")),
            "0123456789"
        );
    }

    #[test]
    fn no_git_fallback_names_the_version_never_unknown() {
        assert_eq!(resolve(None, None, None, Some("0.69.0")), "v0.69.0+no-git");
        assert_eq!(
            resolve(Some(""), None, Some(" \n"), Some("0.69.0")),
            "v0.69.0+no-git"
        );
        assert_eq!(resolve(None, None, None, None), "v0.0.0+no-git");
    }

    // The git-running half: these kill the mutants that the pure `resolve`
    // tests cannot reach (#4315 mutants job, 8 MISSED).

    #[test]
    fn run_git_returns_trimmed_stdout_on_success() {
        let v = run_git(&["--version"]).expect("git is on PATH in every build env");
        assert!(v.starts_with("git version "), "{v:?}");
        assert_eq!(v, v.trim());
    }

    #[test]
    fn run_git_is_none_on_failure_and_on_empty_stdout() {
        assert_eq!(run_git(&["definitely-not-a-git-subcommand"]), None);
        // `git rev-parse` with no arguments succeeds and prints nothing.
        assert_eq!(run_git(&["rev-parse"]), None);
    }

    #[test]
    fn rerun_triggers_watch_head_and_refs_when_in_a_checkout() {
        let triggers = git_rerun_triggers();
        if run_git(&["rev-parse", "--git-dir"]).is_none() {
            assert!(triggers.is_empty(), "{triggers:?}");
            return;
        }
        assert!(
            triggers.iter().any(|p| p.ends_with("/HEAD")),
            "no HEAD trigger: {triggers:?}"
        );
        assert!(
            triggers.iter().any(|p| p.ends_with("/refs/heads")),
            "no refs/heads trigger: {triggers:?}"
        );
        for p in &triggers {
            assert!(std::path::Path::new(p).exists(), "{p}");
        }
    }

    /// A fake source: records whether it was consulted, answers from a table.
    fn fake<'a>(
        table: &'a [(&'a str, &'a str)],
        hit: &'a std::cell::Cell<bool>,
    ) -> impl Fn(&str) -> Option<String> + 'a {
        move |k| {
            hit.set(true);
            table
                .iter()
                .find(|(key, _)| *key == k)
                .map(|(_, v)| (*v).to_string())
        }
    }

    #[test]
    fn override_wins_and_neither_git_nor_the_file_is_consulted() {
        let (g, f, e) = Default::default();
        let git_hit: &std::cell::Cell<bool> = &g;
        let got = resolve_git_sha_with(
            fake(&[("APR_GIT_SHA_OVERRIDE", " rel123 \n")], &e),
            |_: &[&str]| {
                git_hit.set(true);
                Some("gitsha".into())
            },
            fake(&[(".git-sha", "filesha")], &f),
        );
        assert_eq!(got, "rel123");
        assert!(!g.get(), "git ran although the override was set");
        assert!(!f.get(), ".git-sha was read although the override was set");
    }

    #[test]
    fn git_answers_and_the_file_is_not_read() {
        let (f, e) = Default::default();
        let got = resolve_git_sha_with(
            fake(&[("APR_GIT_SHA_OVERRIDE", "  ")], &e),
            |a: &[&str]| (a == ["rev-parse", "--short", "HEAD"]).then(|| "abc1234".to_string()),
            fake(&[(".git-sha", "filesha")], &f),
        );
        assert_eq!(got, "abc1234");
        assert!(!f.get(), ".git-sha was read although git answered");
    }

    #[test]
    fn without_git_the_file_then_the_version_is_used() {
        let (f, e) = Default::default();
        let no_git = |_: &[&str]| None;
        let got = resolve_git_sha_with(
            fake(&[("CARGO_PKG_VERSION", "1.2.3")], &e),
            no_git,
            fake(&[(".git-sha", "0123456789\n")], &f),
        );
        assert_eq!(got, "0123456789");
        assert!(f.get());
        let (f2, e2) = Default::default();
        let got = resolve_git_sha_with(
            fake(&[("CARGO_PKG_VERSION", "1.2.3")], &e2),
            no_git,
            fake(&[], &f2),
        );
        assert_eq!(got, "v1.2.3+no-git");
    }

    #[test]
    fn resolve_git_sha_reads_the_live_sources() {
        // An oracle independent of `resolve`: in a checkout with no override set,
        // the live value is exactly git's short HEAD.
        let got = resolve_git_sha();
        assert!(!got.is_empty());
        if let Some(o) = std::env::var("APR_GIT_SHA_OVERRIDE")
            .ok()
            .filter(|s| !s.trim().is_empty())
        {
            assert_eq!(got, o.trim());
        } else if let Some(head) = run_git(&["rev-parse", "--short", "HEAD"]) {
            assert_eq!(got, head);
        } else {
            assert!(
                got.ends_with("+no-git") || std::path::Path::new(".git-sha").exists(),
                "{got}"
            );
        }
    }
}
