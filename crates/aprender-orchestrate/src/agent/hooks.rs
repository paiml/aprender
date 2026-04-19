//! Hook system for `apr code` — mirrors Claude Code's hook events.
//!
//! Hooks let a user intercept the agent loop at well-defined moments and
//! run an external command. The exit code of the command decides what
//! happens next:
//!
//! * `0` → allow (hook did nothing interesting, or approved the action)
//! * `1` → warn (hook emitted a message; agent continues)
//! * `2` → block (hook vetoed the action; agent aborts that step)
//!
//! This matches Claude Code's `settings.json` → `[[hooks]]` surface 1:1.
//!
//! PMAT-CODE-HOOKS-001. This file introduces the event enum, the config
//! struct that deserializes from `manifest.toml`'s `[[hooks]]` table,
//! and the registry + runner used by the agent loop. Runtime integration
//! ships incrementally:
//!
//! * SessionStart — wired from `cmd_code` (see `agent/code.rs`)
//! * PreToolUse / PostToolUse / UserPromptSubmit / Stop / SubagentStop —
//!   surface ships now; call sites land in PMAT-CODE-HOOKS-RUNTIME-001.

use std::collections::HashMap;
use std::path::Path;
use std::process::Command;

use serde::{Deserialize, Serialize};

/// Canonical hook events mirroring Claude Code.
///
/// Claude's official list (as of 2026-04) is exactly the six below. Any
/// additional events APR-specific work wants to surface should land as
/// a separate enum variant + status_history entry in the parity contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub enum HookEvent {
    /// Fires once when the agent session starts.
    SessionStart,
    /// Fires before every tool call; exit code 2 vetoes the call.
    PreToolUse,
    /// Fires after every tool call regardless of success.
    PostToolUse,
    /// Fires whenever the user submits a prompt.
    UserPromptSubmit,
    /// Fires when the top-level agent is about to terminate.
    Stop,
    /// Fires when a sub-agent (spawned via the Task tool) terminates.
    SubagentStop,
}

/// A single hook entry deserialized from `manifest.hooks[]`.
///
/// ```toml
/// [[hooks]]
/// event = "PreToolUse"
/// matcher = "shell"         # optional — tool name glob/regex
/// command = "echo 'pre-tool' >> ~/.apr/hooks.log"
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HookConfig {
    /// Which event triggers this hook.
    pub event: HookEvent,
    /// Optional matcher (tool name, prompt text, etc.). When `None` the
    /// hook always fires for its event.
    #[serde(default)]
    pub matcher: Option<String>,
    /// Shell command to execute. Ran via `sh -c`.
    pub command: String,
    /// Optional timeout in seconds (default 30).
    #[serde(default = "default_timeout_secs")]
    pub timeout_secs: u64,
}

fn default_timeout_secs() -> u64 {
    30
}

/// Decision returned by a hook, derived from the command's exit code.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HookDecision {
    /// Exit 0 — agent continues as if no hook ran.
    Allow,
    /// Exit 1 — agent continues but surfaces the hook's stderr to the user.
    Warn(String),
    /// Exit 2 — agent aborts the current step; stderr becomes the reason.
    Block(String),
}

impl HookDecision {
    pub fn from_exit_code(code: i32, stderr: String) -> Self {
        match code {
            0 => Self::Allow,
            1 => Self::Warn(stderr),
            _ => Self::Block(stderr),
        }
    }

    pub fn is_blocking(&self) -> bool {
        matches!(self, Self::Block(_))
    }
}

/// Indexed collection of hooks keyed by event.
#[derive(Debug, Default)]
pub struct HookRegistry {
    by_event: HashMap<HookEvent, Vec<HookConfig>>,
}

impl HookRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Build a registry from a raw list (typically `manifest.hooks`).
    pub fn from_configs(configs: impl IntoIterator<Item = HookConfig>) -> Self {
        let mut reg = Self::new();
        for cfg in configs {
            reg.register(cfg);
        }
        reg
    }

    pub fn register(&mut self, cfg: HookConfig) {
        self.by_event.entry(cfg.event).or_default().push(cfg);
    }

    pub fn hooks_for(&self, event: HookEvent) -> &[HookConfig] {
        self.by_event.get(&event).map_or(&[], |v| v.as_slice())
    }

    pub fn len(&self) -> usize {
        self.by_event.values().map(Vec::len).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Fire every hook registered for `event` whose `matcher` (if any) is
    /// contained in `context`. Returns the first blocking decision if
    /// any, else the collected warnings, else Allow.
    pub fn run(&self, event: HookEvent, context: &str, cwd: &Path) -> HookDecision {
        let mut warnings = Vec::new();
        for cfg in self.hooks_for(event) {
            if let Some(m) = cfg.matcher.as_deref() {
                if !context.contains(m) {
                    continue;
                }
            }
            match run_single(cfg, cwd) {
                HookDecision::Allow => {}
                HookDecision::Warn(msg) => warnings.push(msg),
                block @ HookDecision::Block(_) => return block,
            }
        }
        if warnings.is_empty() {
            HookDecision::Allow
        } else {
            HookDecision::Warn(warnings.join("\n"))
        }
    }
}

fn run_single(cfg: &HookConfig, cwd: &Path) -> HookDecision {
    let output = match Command::new("sh")
        .arg("-c")
        .arg(&cfg.command)
        .current_dir(cwd)
        .output()
    {
        Ok(o) => o,
        Err(e) => {
            return HookDecision::Warn(format!(
                "hook '{}' failed to spawn: {e}",
                cfg.command
            ));
        }
    };
    let code = output.status.code().unwrap_or(0);
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
    HookDecision::from_exit_code(code, stderr)
}

#[cfg(test)]
mod tests;
