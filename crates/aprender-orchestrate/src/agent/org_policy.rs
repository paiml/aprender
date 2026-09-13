//! Managed organisation policy loader (Claude-Code parity).
//!
//! PMAT-CODE-ORG-POLICY-001: Claude Code honours a highest-tier
//! "enforced settings" file that an organisation admin drops in
//! `/etc/claude-code/CLAUDE.md`. Its content is treated as
//! instructions whose precedence sits ABOVE project and user
//! scopes, so corporate security/compliance rules cannot be
//! overridden by a developer-level `~/.claude/CLAUDE.md` or a
//! repo-level `CLAUDE.md`.
//!
//! This module ships the pure file-load primitive that the REPL
//! prompt builder layers on top of the existing
//! `load_project_instructions` / user-scope ladder. Canonical
//! paths are injected by the caller so tests can use
//! `tempfile::tempdir()` without touching `/etc`.
//!
//! # Claude-Code compatibility
//!
//! Two canonical paths are consulted in this order, first-wins:
//! 1. `/etc/apr-code/CLAUDE.md` — native path
//! 2. `/etc/claude-code/CLAUDE.md` — Claude-Code cross-compat
//!
//! The tier wins over project and user scopes. Missing files are
//! not an error — the policy is simply absent.
//!
//! # Example
//!
//! ```rust,ignore
//! use batuta::agent::org_policy::load_org_policy;
//! use std::path::Path;
//!
//! let roots = [
//!     Path::new("/etc/apr-code"),
//!     Path::new("/etc/claude-code"),
//! ];
//! if let Some(policy) = load_org_policy(&roots, "CLAUDE.md", 65_536) {
//!     // prepend to prompt at the enforced tier
//! }
//! ```

use std::path::{Path, PathBuf};

/// Precedence tier at which the loaded policy content belongs.
///
/// Managed org policy always wins — this enum exists so the
/// prompt builder can tag each instruction block with its tier
/// and the merge step is total-ordered.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PolicyTier {
    /// Loaded from `/etc/...` (or the injected admin root) —
    /// highest precedence, overrides project and user scopes.
    Enforced,
}

/// A loaded org-policy document and its source.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OrgPolicy {
    /// Absolute path the content was read from.
    pub source: PathBuf,
    /// Policy markdown contents (possibly truncated).
    pub content: String,
    /// Precedence tier — always [`PolicyTier::Enforced`] today,
    /// reserved for a future Standard tier.
    pub tier: PolicyTier,
}

/// Canonical system-wide roots in Claude-Code-parity precedence.
///
/// Returned as owned `PathBuf`s so callers can freely pass them
/// into [`load_org_policy`] or test-time shadows. The order is
/// `apr-code` first (native), `claude-code` second (cross-compat)
/// so a site that installs both sees the `apr-code` file win.
pub fn canonical_system_roots() -> [PathBuf; 2] {
    [PathBuf::from("/etc/apr-code"), PathBuf::from("/etc/claude-code")]
}

/// Load the first existing `<root>/<filename>` from `roots`.
///
/// - First-wins: `roots` is consulted in order and the first file
///   that exists and reads successfully is returned.
/// - `max_bytes == 0` disables the loader (returns `None`).
/// - `max_bytes > 0` truncates on a UTF-8 char boundary and
///   annotates with a `(truncated from N bytes)` tail so the
///   prompt builder can surface budget pressure to the user.
/// - Missing file or I/O error on a given root is silently
///   skipped — the loader does not panic or propagate errors so
///   a world-readable `/etc` cannot ransom the REPL on boot.
///
/// Callers that want the canonical system roots can pass
/// [`canonical_system_roots`] directly; tests inject `tempdir()`
/// paths so no global state is touched.
pub fn load_org_policy<P>(roots: &[P], filename: &str, max_bytes: usize) -> Option<OrgPolicy>
where
    P: AsRef<Path>,
{
    if max_bytes == 0 {
        return None;
    }
    for root in roots {
        let path = root.as_ref().join(filename);
        if !path.is_file() {
            continue;
        }
        let Ok(content) = std::fs::read_to_string(&path) else { continue };
        let truncated = truncate_on_char_boundary(&content, max_bytes);
        return Some(OrgPolicy { source: path, content: truncated, tier: PolicyTier::Enforced });
    }
    None
}

/// Truncate `content` on a UTF-8 char boundary, appending a
/// `(truncated from N bytes)` tail when the original exceeded
/// `max_bytes`. Used by [`load_org_policy`] so the budget check
/// is deterministic even in the face of multi-byte characters.
fn truncate_on_char_boundary(content: &str, max_bytes: usize) -> String {
    if content.len() <= max_bytes {
        return content.to_string();
    }
    let end = content
        .char_indices()
        .take_while(|(i, _)| *i < max_bytes)
        .last()
        .map(|(i, c)| i + c.len_utf8())
        .unwrap_or(max_bytes.min(content.len()));
    format!("{}...\n(truncated from {} bytes)", &content[..end], content.len())
}

#[cfg(test)]
mod tests;
