//! REPL status line renderer (Claude-Code parity).
//!
//! PMAT-CODE-STATUS-LINE-001: Claude Code renders a one-line status
//! strip at the bottom of the REPL showing model name, permission
//! mode, session cost, git branch, and cwd-relative directory. This
//! module ships the data struct + pure render function so TUI/REPL
//! call sites don't duplicate the formatting rules.
//!
//! # Example
//!
//! ```rust
//! use batuta::agent::status_line::StatusLine;
//!
//! let line = StatusLine {
//!     model: "opus-4-7".into(),
//!     mode: "bypassPermissions".into(),
//!     cost_usd: 0.0234,
//!     branch: Some("main".into()),
//!     cwd_short: Some("~/src/aprender".into()),
//! };
//! let rendered = line.render();
//! assert!(rendered.contains("opus-4-7"));
//! assert!(rendered.contains("bypassPermissions"));
//! assert!(rendered.contains("$0.02"));
//! assert!(rendered.contains("main"));
//! ```

use std::path::Path;

use super::permission::PermissionMode;

/// Data captured by the REPL and rendered as a one-line status strip.
///
/// All fields are owned strings so the renderer is pure and
/// trivially testable without borrowing from the live session.
#[derive(Debug, Clone, PartialEq)]
pub struct StatusLine {
    /// Model identifier (e.g. `opus-4-7`, `qwen2.5-coder-7b`).
    pub model: String,
    /// Permission-mode canonical identifier
    /// (`default`/`plan`/`acceptEdits`/`bypassPermissions`).
    pub mode: String,
    /// Session cost in USD. Formatted to 2 decimal places in the
    /// output.
    pub cost_usd: f64,
    /// Git branch, if the REPL is inside a repo.
    pub branch: Option<String>,
    /// Home-relative cwd display (e.g. `~/src/aprender`).
    pub cwd_short: Option<String>,
}

impl StatusLine {
    /// Render as a single-line string using the Claude-Code column
    /// order: `model | [mode] | $cost | branch | cwd`.
    ///
    /// Missing optional fields are elided (no empty cells).
    pub fn render(&self) -> String {
        let mut parts: Vec<String> = Vec::with_capacity(5);
        parts.push(self.model.clone());
        parts.push(format!("[{}]", self.mode));
        parts.push(format!("${:.2}", self.cost_usd));
        if let Some(branch) = self.branch.as_deref() {
            parts.push(branch.to_string());
        }
        if let Some(cwd) = self.cwd_short.as_deref() {
            parts.push(cwd.to_string());
        }
        parts.join(" | ")
    }

    /// Build a [`StatusLine`] from the live session primitives.
    ///
    /// Pure — caller passes in already-resolved values so this is
    /// trivially testable without touching git or the filesystem.
    pub fn build(
        model: impl Into<String>,
        mode: PermissionMode,
        cost_usd: f64,
        branch: Option<String>,
        cwd_short: Option<String>,
    ) -> Self {
        Self { model: model.into(), mode: mode.to_string(), cost_usd, branch, cwd_short }
    }
}

/// Collapse `$HOME` prefix in `path` to `~/` for display.
///
/// Returns `path.to_string_lossy()` when no home prefix matches.
pub fn short_cwd(path: &Path, home: Option<&Path>) -> String {
    if let Some(home) = home {
        if let Ok(stripped) = path.strip_prefix(home) {
            if stripped.as_os_str().is_empty() {
                return "~".to_string();
            }
            return format!("~/{}", stripped.display());
        }
    }
    path.display().to_string()
}

#[cfg(test)]
mod tests;
