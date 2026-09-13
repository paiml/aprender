//! Permission modes for `apr code` (Claude-Code parity).
//!
//! PMAT-CODE-PERMISSIONS-001: Claude Code cycles through four
//! permission modes via `Shift+Tab` (and advertises them in the
//! status line):
//!
//! | Mode                | Reads | Writes/Edits | Shell   |
//! |---------------------|-------|--------------|---------|
//! | `default`           | ✓     | ask          | ask     |
//! | `plan`              | ✓     | blocked      | blocked |
//! | `acceptEdits`       | ✓     | auto-allow   | ask     |
//! | `bypassPermissions` | ✓     | auto-allow   | auto-allow |
//!
//! This module ships the enum + per-mode policy so call sites can
//! query `policy.verdict(capability)` instead of hard-coding the
//! matrix at each prompt site. Runtime wiring into the REPL prompt
//! loop (the actual Shift+Tab / status-line UX) is follow-up
//! PMAT-CODE-PERMISSIONS-RUNTIME-001 (P2).
//!
//! # Example
//!
//! ```rust,ignore
//! use batuta::agent::permission::{PermissionMode, PermissionVerdict};
//! use batuta::agent::capability::Capability;
//!
//! let mode = PermissionMode::Plan;
//! assert_eq!(
//!     mode.verdict(&Capability::FileRead { allowed_paths: vec!["*".into()] }),
//!     PermissionVerdict::Allow
//! );
//! assert_eq!(
//!     mode.verdict(&Capability::FileWrite { allowed_paths: vec!["*".into()] }),
//!     PermissionVerdict::Block
//! );
//! ```

use serde::{Deserialize, Serialize};

use super::capability::Capability;

/// One of the four Claude-Code permission modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum PermissionMode {
    /// Prompt per action (read auto-allowed). Claude Code's launch
    /// default.
    #[default]
    Default,
    /// Read-only exploration. Writes, edits, and shell invocations
    /// are blocked outright.
    Plan,
    /// Writes and edits auto-allowed; shell still prompts.
    AcceptEdits,
    /// No prompts — every capability auto-allowed. The "YOLO" mode.
    BypassPermissions,
}

/// What the policy says about a single capability request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PermissionVerdict {
    /// Run without prompting.
    Allow,
    /// Pause and ask the user (the REPL-runtime layer handles UX).
    Ask,
    /// Refuse outright; surface a blocked-by-policy error.
    Block,
}

impl PermissionMode {
    /// Parse from a CLI / TOML string. Accepts both `camelCase`
    /// (canonical, matches Claude Code) and `kebab-case` spellings.
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim() {
            "default" => Some(Self::Default),
            "plan" => Some(Self::Plan),
            "acceptEdits" | "accept-edits" | "accept_edits" => Some(Self::AcceptEdits),
            "bypassPermissions" | "bypass-permissions" | "bypass_permissions" => {
                Some(Self::BypassPermissions)
            }
            _ => None,
        }
    }

    /// Canonical camelCase identifier (matches Claude Code + our
    /// serde rename).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::Plan => "plan",
            Self::AcceptEdits => "acceptEdits",
            Self::BypassPermissions => "bypassPermissions",
        }
    }

    /// Next mode in the Shift+Tab cycle (wraps around).
    ///
    /// Order mirrors Claude Code's cycle: default → plan →
    /// acceptEdits → bypassPermissions → default.
    pub fn next(self) -> Self {
        match self {
            Self::Default => Self::Plan,
            Self::Plan => Self::AcceptEdits,
            Self::AcceptEdits => Self::BypassPermissions,
            Self::BypassPermissions => Self::Default,
        }
    }

    /// Verdict on a single capability request.
    pub fn verdict(self, capability: &Capability) -> PermissionVerdict {
        match self {
            Self::BypassPermissions => PermissionVerdict::Allow,
            Self::Default => match capability {
                Capability::FileRead { .. } | Capability::Memory | Capability::Rag => {
                    PermissionVerdict::Allow
                }
                _ => PermissionVerdict::Ask,
            },
            Self::Plan => match capability {
                Capability::FileRead { .. } | Capability::Memory | Capability::Rag => {
                    PermissionVerdict::Allow
                }
                _ => PermissionVerdict::Block,
            },
            Self::AcceptEdits => match capability {
                Capability::FileRead { .. }
                | Capability::FileWrite { .. }
                | Capability::Memory
                | Capability::Rag => PermissionVerdict::Allow,
                _ => PermissionVerdict::Ask,
            },
        }
    }

    /// True when a capability request would run without user
    /// interaction under this mode (helpful for non-interactive
    /// `apr code -p` batch jobs).
    pub fn would_run_unattended(self, capability: &Capability) -> bool {
        matches!(self.verdict(capability), PermissionVerdict::Allow)
    }
}

impl std::fmt::Display for PermissionMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[cfg(test)]
mod tests;
