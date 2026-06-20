//! Project-instruction loading with `@import` syntax + user-level fallback
//! (PMAT-CODE-MEMORY-PARITY-001).
//!
//! Mirrors Claude Code's CLAUDE.md memory mechanism. Two extensions over
//! the legacy single-file project-only loader:
//!
//! 1. **`@<path>` inline imports** inside CLAUDE.md / APR.md content.
//!    A line starting with `@` followed by a relative or absolute path
//!    is replaced with that file's contents (transitively, with depth
//!    limit). Mirrors Claude Code's `@./CONVENTIONS.md` syntax. Missing
//!    or unreadable imports leave the line verbatim and emit a stderr
//!    warning (Poka-Yoke — no silent partial expansion).
//!
//! 2. **User-level fallback**: when no project-level CLAUDE.md / APR.md
//!    is found in `cwd`, load from user-level locations in this order:
//!    `$APR_CONFIG/CLAUDE.md` → `~/.config/apr/CLAUDE.md` →
//!    `~/.claude/CLAUDE.md` (Claude-Code cross-compat). Same fallback
//!    applies to APR.md (apr-native filename takes precedence over
//!    Claude-Code-flavored filename in either layer).
//!
//! Pure-function design: no terminal I/O, no caller-state mutation. Any
//! warnings are returned via an `&mut Vec<String>` so the caller can
//! decide where they go (REPL stderr vs CCPA trace).

use std::path::{Path, PathBuf};

/// Maximum recursive `@import` depth. A flat document with one level of
/// `@CONVENTIONS.md` is the common case; 4 hops handles a chain like
/// CLAUDE.md → conventions.md → security.md → boilerplate.md without
/// pathological loops eating context budget.
pub const MAX_IMPORT_DEPTH: usize = 4;

/// Filenames considered as project-level instructions, in priority order.
/// `APR.md` is the apr-native; `CLAUDE.md` is the cross-compat name.
pub const PROJECT_FILENAMES: &[&str] = &["APR.md", "CLAUDE.md"];

/// Find the first existing project-level instructions file under `cwd`.
/// Honors [`PROJECT_FILENAMES`] priority order.
pub fn find_project_instructions(cwd: &Path) -> Option<PathBuf> {
    PROJECT_FILENAMES.iter().map(|f| cwd.join(f)).find(|p| p.is_file())
}

/// Find the first existing user-global instructions file. Search order:
/// `$APR_CONFIG/<name>` → `~/.config/apr/<name>` → `~/.claude/<name>`,
/// trying each `name` from [`PROJECT_FILENAMES`] within each layer
/// before moving to the next layer.
pub fn find_user_global_instructions() -> Option<PathBuf> {
    for layer in user_global_search_dirs() {
        for fname in PROJECT_FILENAMES {
            let p = layer.join(fname);
            if p.is_file() {
                return Some(p);
            }
        }
    }
    None
}

/// User-global instruction search directories, in priority order.
///
/// * If `$APR_CONFIG` is set, that's the **only** location consulted —
///   setting the env var is treated as an explicit opt-out of the
///   default lookup chain (Poka-Yoke; tests + sandboxed runs need this
///   so host-level CLAUDE.md can't leak in).
/// * Otherwise, search XDG `~/.config/apr/` first, then `~/.claude/`
///   (Claude-Code cross-compat).
fn user_global_search_dirs() -> Vec<PathBuf> {
    if let Ok(custom) = std::env::var("APR_CONFIG") {
        if !custom.is_empty() {
            return vec![PathBuf::from(custom)];
        }
    }
    let mut out = Vec::new();
    if let Some(cfg) = dirs::config_dir() {
        out.push(cfg.join("apr"));
    }
    if let Some(home) = dirs::home_dir() {
        out.push(home.join(".claude"));
    }
    out
}

/// Expand `@<path>` import lines in `content` recursively. Each
/// imported file's contents replace the import line (relative paths
/// resolved against `base_dir`).
///
/// Behavior:
/// * Only lines whose **trimmed start** is `@` followed by a non-
///   whitespace path token are imports. This means a line like
///   `Talk to noah@paiml.com` is NOT an import — `@` must lead.
/// * Imports are resolved relative to the importing file's directory,
///   not `cwd`, matching Claude Code's `@./conventions.md` semantics.
/// * Recursion depth is capped at [`MAX_IMPORT_DEPTH`]. Cycles or
///   over-deep chains leave the import line verbatim and emit a
///   warning. So does any I/O failure (Poka-Yoke).
pub fn expand_imports(content: &str, base_dir: &Path, warnings: &mut Vec<String>) -> String {
    expand_imports_inner(content, base_dir, 0, warnings)
}

fn expand_imports_inner(
    content: &str,
    base_dir: &Path,
    depth: usize,
    warnings: &mut Vec<String>,
) -> String {
    let mut out = String::with_capacity(content.len());
    for line in content.lines() {
        if let Some(import_path) = parse_import_line(line) {
            if depth >= MAX_IMPORT_DEPTH {
                warnings.push(format!(
                    "@{import_path}: import depth limit ({MAX_IMPORT_DEPTH}) exceeded; line kept verbatim"
                ));
                out.push_str(line);
                out.push('\n');
                continue;
            }
            let resolved = resolve_import_path(import_path, base_dir);
            match std::fs::read_to_string(&resolved) {
                Ok(body) => {
                    let next_base = resolved
                        .parent()
                        .map(Path::to_path_buf)
                        .unwrap_or_else(|| base_dir.to_path_buf());
                    let expanded = expand_imports_inner(&body, &next_base, depth + 1, warnings);
                    out.push_str(&expanded);
                    if !out.ends_with('\n') {
                        out.push('\n');
                    }
                }
                Err(e) => {
                    warnings.push(format!("@{import_path}: {e}"));
                    out.push_str(line);
                    out.push('\n');
                }
            }
        } else {
            out.push_str(line);
            out.push('\n');
        }
    }
    out
}

/// If `line` is an `@<path>` import directive, return the path. The
/// `@` must be at the start of the trimmed line (not mid-line) so an
/// inline mention like `email noah@paiml.com` is preserved.
///
/// The path token runs until the first whitespace, so paths with
/// spaces are NOT supported (matches Claude Code's grammar).
fn parse_import_line(line: &str) -> Option<&str> {
    let t = line.trim_start();
    let after_at = t.strip_prefix('@')?;
    let path = after_at.split_whitespace().next()?;
    if path.is_empty() {
        None
    } else {
        Some(path)
    }
}

/// Resolve `import_path` relative to `base_dir`. Absolute paths and
/// `~`-prefixed paths are expanded; otherwise the result is
/// `base_dir.join(import_path)`.
fn resolve_import_path(import_path: &str, base_dir: &Path) -> PathBuf {
    if let Some(rest) = import_path.strip_prefix("~/") {
        if let Some(home) = dirs::home_dir() {
            return home.join(rest);
        }
    }
    let p = Path::new(import_path);
    if p.is_absolute() {
        p.to_path_buf()
    } else {
        base_dir.join(p)
    }
}

/// Truncate `content` to `max_bytes` on a UTF-8 char boundary,
/// appending an `(truncated from N bytes)` annotation. `max_bytes==0`
/// returns `None` (caller skips loading entirely).
pub fn truncate_to_budget(content: String, max_bytes: usize) -> Option<String> {
    if max_bytes == 0 {
        return None;
    }
    if content.len() <= max_bytes {
        return Some(content);
    }
    let end = content
        .char_indices()
        .take_while(|(i, _)| *i < max_bytes)
        .last()
        .map(|(i, c)| i + c.len_utf8())
        .unwrap_or(max_bytes.min(content.len()));
    Some(format!("{}...\n(truncated from {} bytes)", &content[..end], content.len()))
}

/// Load layered project + user-global instructions with `@import`
/// expansion. Returns `None` if `max_bytes` is 0 or no file exists at
/// any layer.
///
/// Layering: when both a user-global file AND a project file exist,
/// the user-global content is concatenated FIRST (under a
/// `## User-global instructions` heading) and the project content
/// appears AFTER (matching Claude Code's "later layer wins context-
/// wise but earlier layers still inform the model"). Either layer
/// can be missing.
pub fn load_layered_instructions(
    cwd: &Path,
    max_bytes: usize,
    warnings: &mut Vec<String>,
) -> Option<String> {
    if max_bytes == 0 {
        return None;
    }
    let mut accumulated = String::new();

    if let Some(user_path) = find_user_global_instructions() {
        if let Ok(body) = std::fs::read_to_string(&user_path) {
            let user_dir = user_path.parent().unwrap_or(Path::new("."));
            let expanded = expand_imports(&body, user_dir, warnings);
            accumulated.push_str("## User-global instructions (");
            accumulated.push_str(&user_path.display().to_string());
            accumulated.push_str(")\n\n");
            accumulated.push_str(&expanded);
            if !accumulated.ends_with("\n\n") {
                accumulated.push('\n');
            }
        }
    }

    if let Some(project_path) = find_project_instructions(cwd) {
        if let Ok(body) = std::fs::read_to_string(&project_path) {
            let project_dir = project_path.parent().unwrap_or(cwd);
            let expanded = expand_imports(&body, project_dir, warnings);
            if !accumulated.is_empty() {
                accumulated.push_str("\n## Project instructions (");
                accumulated.push_str(&project_path.display().to_string());
                accumulated.push_str(")\n\n");
            }
            accumulated.push_str(&expanded);
        }
    }

    if accumulated.is_empty() {
        return None;
    }
    truncate_to_budget(accumulated, max_bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    // PMAT-876: shared crate-wide env lock + save/restore guard, shared
    // with auto_memory + settings tests (all touch the same global
    // `APR_CONFIG`). `std::env` is process-global and tests run in
    // parallel, so one lock serializes them ALL.
    use crate::agent::env_test_support::{env_lock, ScopedEnv};
    use std::fs;

    fn write(path: &Path, body: &str) {
        if let Some(p) = path.parent() {
            fs::create_dir_all(p).expect("mkdir");
        }
        fs::write(path, body).expect("write");
    }

    // ── parse_import_line ──────────────────────────────────────────

    #[test]
    fn import_line_simple() {
        assert_eq!(parse_import_line("@./CONVENTIONS.md"), Some("./CONVENTIONS.md"));
    }

    #[test]
    fn import_line_strips_indent() {
        assert_eq!(parse_import_line("  @abs/path.md"), Some("abs/path.md"));
    }

    #[test]
    fn import_line_stops_at_whitespace() {
        // anything after the path is ignored — Claude Code grammar
        assert_eq!(parse_import_line("@./README.md trailing comment"), Some("./README.md"));
    }

    #[test]
    fn import_line_email_not_an_import() {
        assert_eq!(parse_import_line("Email noah@paiml.com"), None);
    }

    #[test]
    fn import_line_bare_at_is_not() {
        assert_eq!(parse_import_line("@"), None);
        assert_eq!(parse_import_line("@   "), None);
    }

    #[test]
    fn import_line_inline_at_ignored() {
        // Claude Code only considers leading `@`. An inline `@` mid-line
        // is just text.
        assert_eq!(parse_import_line("see @./foo.md inline"), None);
    }

    // ── resolve_import_path ────────────────────────────────────────

    #[test]
    fn resolve_relative_against_base() {
        let p = resolve_import_path("./conventions.md", Path::new("/tmp/proj"));
        assert_eq!(p, Path::new("/tmp/proj/./conventions.md"));
    }

    #[test]
    fn resolve_absolute_passes_through() {
        let p = resolve_import_path("/abs/file.md", Path::new("/tmp/proj"));
        assert_eq!(p, Path::new("/abs/file.md"));
    }

    #[test]
    fn resolve_tilde_expands_home() {
        let p = resolve_import_path("~/CONVENTIONS.md", Path::new("/tmp/proj"));
        if let Some(home) = dirs::home_dir() {
            assert_eq!(p, home.join("CONVENTIONS.md"));
        }
    }

    // ── expand_imports ─────────────────────────────────────────────

    #[test]
    fn expand_no_imports_returns_unchanged_modulo_newlines() {
        let mut warns = Vec::new();
        let out = expand_imports("hello\nworld\n", Path::new("/tmp"), &mut warns);
        assert_eq!(out, "hello\nworld\n");
        assert!(warns.is_empty());
    }

    #[test]
    fn expand_single_import() {
        let dir = tempfile::tempdir().expect("tempdir");
        let imp = dir.path().join("conv.md");
        write(&imp, "## Conventions\n- camelCase\n");
        let body = format!("Top-level\n@{}\nBottom\n", imp.display());
        let mut warns = Vec::new();
        let out = expand_imports(&body, dir.path(), &mut warns);
        assert!(out.contains("Top-level"));
        assert!(out.contains("## Conventions"));
        assert!(out.contains("camelCase"));
        assert!(out.contains("Bottom"));
        assert!(warns.is_empty());
    }

    #[test]
    fn expand_relative_import_against_base() {
        let dir = tempfile::tempdir().expect("tempdir");
        let imp = dir.path().join("conv.md");
        write(&imp, "imported-body");
        let body = "@./conv.md\n";
        let mut warns = Vec::new();
        let out = expand_imports(body, dir.path(), &mut warns);
        assert!(out.contains("imported-body"));
    }

    #[test]
    fn expand_missing_import_keeps_line_and_warns() {
        let dir = tempfile::tempdir().expect("tempdir");
        let body = "@./not-there.md\n";
        let mut warns = Vec::new();
        let out = expand_imports(body, dir.path(), &mut warns);
        assert!(out.contains("@./not-there.md"));
        assert_eq!(warns.len(), 1);
        assert!(warns[0].contains("not-there.md"));
    }

    #[test]
    fn expand_recursive_imports() {
        // a.md @-imports b.md which @-imports c.md (3 levels).
        let dir = tempfile::tempdir().expect("tempdir");
        let a = dir.path().join("a.md");
        let b = dir.path().join("b.md");
        let c = dir.path().join("c.md");
        write(&a, "AAA\n@./b.md\n");
        write(&b, "BBB\n@./c.md\n");
        write(&c, "CCC\n");
        let mut warns = Vec::new();
        let out = expand_imports(&fs::read_to_string(&a).unwrap(), dir.path(), &mut warns);
        assert!(out.contains("AAA"));
        assert!(out.contains("BBB"));
        assert!(out.contains("CCC"));
        assert!(warns.is_empty());
    }

    #[test]
    fn expand_recursive_path_resolves_against_importing_file() {
        // Sub-dir import: imported file lives elsewhere; its own
        // imports must resolve against ITS directory, not the
        // top-level one.
        let dir = tempfile::tempdir().expect("tempdir");
        let sub = dir.path().join("sub");
        let outer = dir.path().join("outer.md");
        let mid = sub.join("mid.md");
        let leaf = sub.join("leaf.md");
        write(&outer, "TOP\n@./sub/mid.md\n");
        write(&mid, "MID\n@./leaf.md\n"); // relative to sub/, not to dir/
        write(&leaf, "LEAF\n");
        let mut warns = Vec::new();
        let out = expand_imports(&fs::read_to_string(&outer).unwrap(), dir.path(), &mut warns);
        assert!(out.contains("TOP"));
        assert!(out.contains("MID"));
        assert!(out.contains("LEAF"), "leaf.md should resolve relative to sub/, got: {out:?}");
    }

    #[test]
    fn expand_depth_limit_prevents_cycle_blowup() {
        // a.md @-imports b.md, b.md @-imports a.md → cycle. The depth
        // limit caps the recursion; the line that would trigger the
        // (depth+1)-th expansion is kept verbatim with a warning.
        let dir = tempfile::tempdir().expect("tempdir");
        let a = dir.path().join("a.md");
        let b = dir.path().join("b.md");
        write(&a, "@./b.md\n");
        write(&b, "@./a.md\n");
        let mut warns = Vec::new();
        let out = expand_imports(&fs::read_to_string(&a).unwrap(), dir.path(), &mut warns);
        // Stays bounded — output is non-empty but doesn't blow up.
        assert!(out.len() < 100_000);
        // The depth-limit warning fires.
        assert!(warns.iter().any(|w| w.contains("depth limit")), "warns: {warns:?}");
    }

    // ── truncate_to_budget ─────────────────────────────────────────

    #[test]
    fn truncate_zero_budget_yields_none() {
        assert!(truncate_to_budget("xxx".into(), 0).is_none());
    }

    #[test]
    fn truncate_under_budget_passthrough() {
        let s = truncate_to_budget("short".into(), 100).expect("kept");
        assert_eq!(s, "short");
    }

    #[test]
    fn truncate_over_budget_appends_annotation() {
        let big = "x".repeat(500);
        let s = truncate_to_budget(big, 100).expect("truncated");
        assert!(s.starts_with("x"));
        assert!(s.contains("truncated from 500 bytes"));
    }

    #[test]
    fn truncate_respects_utf8_boundary() {
        // 3-byte char at byte 99 must not be split.
        let s = format!("{}é", "a".repeat(99));
        let truncated = truncate_to_budget(s, 100).expect("truncated");
        // No char-boundary panics; just verify it parses as valid UTF-8
        // and contains the truncation annotation.
        assert!(truncated.contains("truncated from"));
    }

    // ── find_user_global_instructions / load_layered_instructions ──
    //
    // PMAT-876: tests below mutate the process-wide `APR_CONFIG` env var.
    // `std::env` is process-global and cargo test runs `#[test]` fns in
    // parallel by default, so they all acquire the ONE crate-wide
    // `env_test_support::env_lock` (shared with auto_memory + settings,
    // which touch the same variable) and use ScopedEnv to restore the
    // prior value on drop.

    #[test]
    fn user_global_honors_apr_config_env_first() {
        let _guard = env_lock();
        let dir = tempfile::tempdir().expect("tempdir");
        write(&dir.path().join("CLAUDE.md"), "user-global-content");
        let _env = ScopedEnv::set("APR_CONFIG", dir.path());
        let p = find_user_global_instructions().expect("found");
        assert_eq!(p, dir.path().join("CLAUDE.md"));
    }

    #[test]
    fn user_global_prefers_apr_md_over_claude_md_within_layer() {
        let _guard = env_lock();
        let dir = tempfile::tempdir().expect("tempdir");
        write(&dir.path().join("APR.md"), "apr-version");
        write(&dir.path().join("CLAUDE.md"), "claude-version");
        let _env = ScopedEnv::set("APR_CONFIG", dir.path());
        let p = find_user_global_instructions().expect("found");
        assert_eq!(p, dir.path().join("APR.md"), "APR.md wins over CLAUDE.md within a layer");
    }

    #[test]
    fn load_layered_returns_none_when_nothing_to_load() {
        let _guard = env_lock();
        let cfg = tempfile::tempdir().expect("cfg");
        let proj = tempfile::tempdir().expect("proj");
        let _env = ScopedEnv::set("APR_CONFIG", cfg.path());
        let mut warns = Vec::new();
        let out = load_layered_instructions(proj.path(), 4096, &mut warns);
        assert!(out.is_none());
    }

    #[test]
    fn load_layered_concatenates_user_global_then_project() {
        let _guard = env_lock();
        let cfg = tempfile::tempdir().expect("cfg");
        let proj = tempfile::tempdir().expect("proj");
        write(&cfg.path().join("CLAUDE.md"), "USER-GLOBAL-BODY\n");
        write(&proj.path().join("CLAUDE.md"), "PROJECT-BODY\n");
        let _env = ScopedEnv::set("APR_CONFIG", cfg.path());
        let mut warns = Vec::new();
        let out = load_layered_instructions(proj.path(), 65536, &mut warns).expect("loaded");
        let user_idx = out.find("USER-GLOBAL-BODY").expect("user-global present");
        let proj_idx = out.find("PROJECT-BODY").expect("project present");
        assert!(
            user_idx < proj_idx,
            "user-global must come before project so project wins context-wise"
        );
        assert!(out.contains("User-global instructions"));
        assert!(out.contains("Project instructions"));
    }

    #[test]
    fn load_layered_resolves_imports_in_each_layer() {
        let _guard = env_lock();
        let cfg = tempfile::tempdir().expect("cfg");
        let proj = tempfile::tempdir().expect("proj");
        // user-global: imports a sibling file
        write(&cfg.path().join("CLAUDE.md"), "USER\n@./shared.md\n");
        write(&cfg.path().join("shared.md"), "USER-SHARED\n");
        // project: imports a sibling file
        write(&proj.path().join("CLAUDE.md"), "PROJ\n@./conv.md\n");
        write(&proj.path().join("conv.md"), "PROJ-CONV\n");
        let _env = ScopedEnv::set("APR_CONFIG", cfg.path());
        let mut warns = Vec::new();
        let out = load_layered_instructions(proj.path(), 65536, &mut warns).expect("loaded");
        assert!(out.contains("USER-SHARED"));
        assert!(out.contains("PROJ-CONV"));
        assert!(warns.is_empty(), "no warnings expected, got: {warns:?}");
    }
}
