//! Per-project auto-memory loader (PMAT-CODE-MEMORY-AUTO-001).
//!
//! Mirrors Claude Code's auto-memory mechanism: every project has an
//! associated directory under `~/.config/apr/projects/<slug>/memory/`
//! containing freely-named markdown files. When `apr code` starts
//! against a given working directory, every `*.md` in that directory
//! is concatenated into the system prompt under a `## Auto-memory`
//! section. This lets the user (or apr itself) accumulate
//! per-project notes — facts about the user, project state,
//! feedback, references — without ever editing a single CLAUDE.md.
//!
//! ## Path slug
//!
//! Claude Code uses a hyphenated path slug like
//! `-home-noah-src-aprender` (leading dash + every `/` → `-`). We
//! keep the same slug for cross-tool compatibility: a user moving
//! from Claude Code to apr keeps their auto-memory by symlinking
//! `~/.claude/projects/` ↔ `~/.config/apr/projects/`.
//!
//! ## File ordering
//!
//! Files are loaded in lexicographic order so `MEMORY.md` lands first
//! (uppercase `M` < lowercase `f`/`p`/`r`/`u` ASCII) and the rest
//! sort by name. A file named `MEMORY.md` is therefore the canonical
//! "top-of-memory index" matching Claude's convention.
//!
//! ## Pure-function design
//!
//! No terminal I/O; warnings are returned via `&mut Vec<String>` for
//! the caller to route. No filesystem mutation — read-only loader.

use std::path::{Path, PathBuf};

/// Path-slug Claude Code uses for the project directory under
/// `~/.config/apr/projects/`. Format: leading `-`, then every path
/// separator replaced with `-`. Non-absolute paths get prefixed with
/// the host's HOME (Claude's behavior) so a relative `cwd` still
/// produces a stable slug.
pub fn project_slug(cwd: &Path) -> String {
    let abs = if cwd.is_absolute() {
        cwd.to_path_buf()
    } else {
        // Best-effort canonicalization; fall back to verbatim if cwd
        // can't be resolved (e.g. the path doesn't exist).
        std::fs::canonicalize(cwd).unwrap_or_else(|_| cwd.to_path_buf())
    };
    let s = abs.to_string_lossy();
    // Each `/` (or `\` on Windows) → `-`. For absolute paths, the
    // leading `/` becomes the leading `-` automatically; relative
    // paths get an explicit leading `-` prefix so the slug always
    // starts with `-` (matches Claude Code's emit). We avoid
    // double-dashing for absolute paths.
    let mut out = String::with_capacity(s.len() + 1);
    let starts_with_sep = s.starts_with('/') || s.starts_with('\\');
    if !starts_with_sep {
        out.push('-');
    }
    for ch in s.chars() {
        if ch == '/' || ch == '\\' {
            out.push('-');
        } else {
            out.push(ch);
        }
    }
    out
}

/// Auto-memory root directory: `$APR_CONFIG/projects` if the env var
/// is set (override; exclusive — same Poka-Yoke as the settings
/// ladder), otherwise `~/.config/apr/projects`.
pub fn auto_memory_root() -> Option<PathBuf> {
    if let Ok(custom) = std::env::var("APR_CONFIG") {
        if !custom.is_empty() {
            return Some(PathBuf::from(custom).join("projects"));
        }
    }
    dirs::config_dir().map(|d| d.join("apr").join("projects"))
}

/// Per-project memory directory: `<root>/<slug>/memory/`.
pub fn project_memory_dir(cwd: &Path) -> Option<PathBuf> {
    auto_memory_root().map(|r| r.join(project_slug(cwd)).join("memory"))
}

/// Load every `*.md` file in the project's memory directory in
/// lexicographic order. Returns `None` when the directory doesn't
/// exist or has no markdown files (so the caller can omit the
/// `## Auto-memory` heading entirely instead of emitting an empty
/// section).
///
/// Per-file read errors emit a warning and are skipped; one bad
/// file doesn't take down the whole load.
pub fn load_auto_memory(cwd: &Path, warnings: &mut Vec<String>) -> Option<String> {
    let dir = project_memory_dir(cwd)?;
    if !dir.is_dir() {
        return None;
    }
    let mut entries: Vec<PathBuf> = match std::fs::read_dir(&dir) {
        Ok(rd) => rd
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.is_file() && p.extension().is_some_and(|e| e == "md"))
            .collect(),
        Err(e) => {
            warnings.push(format!("auto-memory: read_dir({}) failed: {e}", dir.display()));
            return None;
        }
    };
    entries.sort();
    if entries.is_empty() {
        return None;
    }
    let mut out = String::new();
    for path in &entries {
        match std::fs::read_to_string(path) {
            Ok(body) => {
                let name =
                    path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                if !out.is_empty() && !out.ends_with("\n\n") {
                    out.push('\n');
                }
                out.push_str(&format!("### {name}\n\n"));
                out.push_str(&body);
                if !out.ends_with('\n') {
                    out.push('\n');
                }
            }
            Err(e) => {
                warnings.push(format!("auto-memory: read({}) failed: {e}", path.display()));
            }
        }
    }
    if out.is_empty() {
        None
    } else {
        Some(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // PMAT-876: tests below mutate the process-wide `APR_CONFIG` env var.
    // `std::env` is process-global and `cargo test` runs tests in parallel,
    // so they all acquire ONE crate-wide lock (env_test_support::env_lock)
    // — shared across auto_memory + settings + instructions, all of which
    // touch the SAME variable — and use ScopedEnv to restore the prior
    // value on drop so the env is never left mutated for another test.
    use crate::agent::env_test_support::{env_lock, ScopedEnv};
    use std::fs;
    use std::path::Path;

    fn write(path: &Path, body: &str) {
        if let Some(p) = path.parent() {
            fs::create_dir_all(p).expect("mkdir");
        }
        fs::write(path, body).expect("write");
    }

    // ── project_slug ───────────────────────────────────────────────

    #[test]
    fn slug_for_absolute_path() {
        let s = project_slug(Path::new("/home/noah/src/aprender"));
        assert_eq!(s, "-home-noah-src-aprender");
    }

    #[test]
    fn slug_for_root() {
        let s = project_slug(Path::new("/"));
        // "/" → "-" (single slash)
        assert_eq!(s, "-");
    }

    #[test]
    fn slug_with_dots_preserved() {
        // Claude doesn't sanitize dots; only `/` becomes `-`.
        let s = project_slug(Path::new("/tmp/a.b.c"));
        assert_eq!(s, "-tmp-a.b.c");
    }

    #[test]
    fn slug_strips_trailing_slash() {
        // canonicalize won't add it, but if cwd ends in `/`, the
        // string conversion preserves it. We accept the `-` at the
        // tail since matching Claude's exact emit is more important
        // than aesthetics.
        let s = project_slug(Path::new("/tmp/x/"));
        assert!(s == "-tmp-x-" || s == "-tmp-x", "got {s:?}");
    }

    // ── auto_memory_root ───────────────────────────────────────────

    #[test]
    fn root_honors_apr_config_env() {
        let _guard = env_lock();
        let dir = tempfile::tempdir().expect("tempdir");
        let _env = ScopedEnv::set("APR_CONFIG", dir.path());
        let r = auto_memory_root().expect("root resolved");
        assert_eq!(r, dir.path().join("projects"));
    }

    #[test]
    fn root_uses_config_dir_when_env_unset() {
        let _guard = env_lock();
        let _env = ScopedEnv::remove("APR_CONFIG");
        let r = auto_memory_root().expect("root resolved on supported platform");
        // Should end in apr/projects regardless of host home.
        assert!(r.ends_with("apr/projects"), "unexpected root: {r:?}");
    }

    // ── project_memory_dir ─────────────────────────────────────────

    #[test]
    fn project_memory_dir_layout() {
        let _guard = env_lock();
        let cfg = tempfile::tempdir().expect("cfg");
        let _env = ScopedEnv::set("APR_CONFIG", cfg.path());
        let dir = project_memory_dir(Path::new("/tmp/myproj")).expect("dir");
        assert_eq!(dir, cfg.path().join("projects").join("-tmp-myproj").join("memory"));
    }

    // ── load_auto_memory ───────────────────────────────────────────

    #[test]
    fn load_returns_none_when_no_dir() {
        let _guard = env_lock();
        let cfg = tempfile::tempdir().expect("cfg");
        let _env = ScopedEnv::set("APR_CONFIG", cfg.path());
        let mut warns = Vec::new();
        // Project memory dir doesn't exist (no `projects/<slug>/memory/`).
        let out = load_auto_memory(Path::new("/tmp/never"), &mut warns);
        assert!(out.is_none());
        assert!(warns.is_empty());
    }

    #[test]
    fn load_returns_none_when_dir_empty() {
        let _guard = env_lock();
        let cfg = tempfile::tempdir().expect("cfg");
        let mem_dir = cfg.path().join("projects").join("-tmp-x").join("memory");
        fs::create_dir_all(&mem_dir).expect("mkdir");
        let _env = ScopedEnv::set("APR_CONFIG", cfg.path());
        let mut warns = Vec::new();
        let out = load_auto_memory(Path::new("/tmp/x"), &mut warns);
        assert!(out.is_none(), "empty memory dir → None, got: {out:?}");
        assert!(warns.is_empty());
    }

    #[test]
    fn load_concatenates_md_files_in_lex_order() {
        let _guard = env_lock();
        let cfg = tempfile::tempdir().expect("cfg");
        let mem_dir = cfg.path().join("projects").join("-tmp-y").join("memory");
        write(&mem_dir.join("MEMORY.md"), "# Top-of-memory index\n");
        write(&mem_dir.join("zzz_user.md"), "User notes\n");
        write(&mem_dir.join("feedback_x.md"), "Feedback X\n");
        let _env = ScopedEnv::set("APR_CONFIG", cfg.path());
        let mut warns = Vec::new();
        let out = load_auto_memory(Path::new("/tmp/y"), &mut warns).expect("loaded");
        assert!(warns.is_empty());
        // MEMORY.md (uppercase M) sorts before lowercase f/z.
        let memory_idx = out.find("Top-of-memory index").expect("MEMORY present");
        let feedback_idx = out.find("Feedback X").expect("feedback present");
        let user_idx = out.find("User notes").expect("user present");
        assert!(memory_idx < feedback_idx, "MEMORY.md must come first");
        assert!(feedback_idx < user_idx, "feedback < user lexicographically");
        // Each file gets a `### <name>` heading.
        assert!(out.contains("### MEMORY.md"));
        assert!(out.contains("### feedback_x.md"));
        assert!(out.contains("### zzz_user.md"));
    }

    #[test]
    fn load_skips_non_md_files() {
        let _guard = env_lock();
        let cfg = tempfile::tempdir().expect("cfg");
        let mem_dir = cfg.path().join("projects").join("-tmp-skip").join("memory");
        write(&mem_dir.join("note.md"), "kept\n");
        write(&mem_dir.join("note.txt"), "skipped\n");
        write(&mem_dir.join("note.json"), "skipped\n");
        let _env = ScopedEnv::set("APR_CONFIG", cfg.path());
        let mut warns = Vec::new();
        let out = load_auto_memory(Path::new("/tmp/skip"), &mut warns).expect("loaded");
        assert!(out.contains("kept"));
        assert!(!out.contains("skipped"), "non-md files must NOT be loaded");
    }

    #[test]
    fn load_skips_subdirectories() {
        let _guard = env_lock();
        let cfg = tempfile::tempdir().expect("cfg");
        let mem_dir = cfg.path().join("projects").join("-tmp-sub").join("memory");
        write(&mem_dir.join("ok.md"), "ok-content\n");
        // Subdirectory under memory/ — must not recurse.
        fs::create_dir_all(mem_dir.join("nested")).expect("mkdir nested");
        write(&mem_dir.join("nested").join("hidden.md"), "hidden-content\n");
        let _env = ScopedEnv::set("APR_CONFIG", cfg.path());
        let mut warns = Vec::new();
        let out = load_auto_memory(Path::new("/tmp/sub"), &mut warns).expect("loaded");
        assert!(out.contains("ok-content"));
        assert!(!out.contains("hidden-content"), "must not recurse into subdirs");
    }
}
