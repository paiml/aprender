//! Claude-Code-parity settings ladder for `apr code`
//! (PMAT-CODE-CONFIG-LADDER-001).
//!
//! Implements the same user-global → project-local → CLI-override
//! precedence ladder Claude Code uses for `~/.claude/settings.json`,
//! adapted to the apr-code surface:
//!
//! | Layer | Path | Notes |
//! |-------|------|-------|
//! | User-global | `$APR_CONFIG/settings.json` (override) or `~/.config/apr/settings.json` | machine-wide defaults |
//! | Project-local | `<project_root>/.apr/settings.json` | repo-specific overrides |
//! | CLI flags | `--model`, `--max-turns`, `--manifest` | always wins |
//!
//! ## Precedence
//!
//! Latter layers override earlier ones field-by-field. Missing files at any
//! layer are non-errors (the layer just contributes no fields). Malformed
//! JSON is a hard error so the operator notices the broken file rather than
//! silently running on partial config (Poka-Yoke).
//!
//! ## Why JSON, not TOML
//!
//! `apr code`'s legacy `--manifest` flag accepts TOML
//! ([`AgentManifest`](super::manifest::AgentManifest)). The settings ladder
//! is *additive*: `settings.json` carries the small set of fields a typical
//! user wants to set machine-wide (model, max_turns, system prompt extras),
//! and matches Claude Code's `settings.json` format so users coming from
//! Claude Code can copy their settings file with minimal edits.
//!
//! For full agent specification (capabilities, hooks, MCP servers,
//! resource quotas), use `--manifest path/to/manifest.toml` — it
//! short-circuits the ladder.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Claude-Code-parity settings layer (`apr code`).
///
/// Every field is `Option<_>` so we can tell "explicitly set" apart from
/// "use the next layer's default" during merge. After merge, unset fields
/// fall back to [`super::manifest::ModelConfig::default()`] /
/// build_default_manifest values.
#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AprSettings {
    /// Path to local model file OR HuggingFace repo (e.g. `qwen3:1.7b-q4k`).
    /// Mirrors Claude Code's `model: "claude-3-5-sonnet-20241022"`.
    pub model: Option<String>,

    /// Maximum REPL/agent turns before stopping. Claude Code uses no
    /// equivalent (it caps via budget); this is an apr-code-specific knob.
    pub max_turns: Option<u32>,

    /// Extra text appended to the agent's system prompt. Mirrors
    /// Claude Code's `customApiKeyResponses` / `extraSystemPrompt`.
    pub extra_system_prompt: Option<String>,

    /// Default working-directory project root override. Resolved relative
    /// to the settings file's directory; absolute paths pass through.
    pub project: Option<PathBuf>,

    /// Permission mode for the agent's tool dispatch (PMAT-CODE-CONFIG-LADDER-FIELDS-001).
    /// Mirrors Claude Code's `permissionMode` field. Accepts the camelCase /
    /// kebab-case / snake_case aliases that
    /// [`crate::agent::permission::PermissionMode::parse`] honors:
    /// `"default" | "plan" | "acceptEdits" | "bypassPermissions"` (and
    /// case-insensitive equivalents). Unknown strings produce a settings-load
    /// error from `apply_settings_to_manifest` (Poka-Yoke).
    ///
    /// Stored as `Option<String>` rather than `Option<PermissionMode>` so the
    /// settings type stays JSON-trivial and so unknown values surface a
    /// clear `apr code` error message at apply time rather than a generic
    /// serde error at parse time.
    #[serde(rename = "permissionMode", alias = "permission_mode")]
    pub permission_mode: Option<String>,

    /// Hostnames the agent's `NetworkTool` / `BrowserTool` may reach
    /// (PMAT-CODE-CONFIG-LADDER-FIELDS-001). Mirrors `AgentManifest.allowed_hosts`.
    /// Sovereign privacy tier always blocks network tools regardless of this
    /// list (Poka-Yoke; tier wins over config). Empty list = no network
    /// tools registered.
    #[serde(rename = "allowedHosts", alias = "allowed_hosts")]
    pub allowed_hosts: Option<Vec<String>>,
}

impl AprSettings {
    /// Field-by-field merge: `other` wins over `self` for any `Some(_)` field.
    /// Used to fold project-local over user-global, then CLI over that.
    pub fn merge(&mut self, other: &AprSettings) {
        if other.model.is_some() {
            self.model = other.model.clone();
        }
        if other.max_turns.is_some() {
            self.max_turns = other.max_turns;
        }
        if other.extra_system_prompt.is_some() {
            self.extra_system_prompt = other.extra_system_prompt.clone();
        }
        if other.project.is_some() {
            self.project = other.project.clone();
        }
        if other.permission_mode.is_some() {
            self.permission_mode = other.permission_mode.clone();
        }
        if other.allowed_hosts.is_some() {
            self.allowed_hosts = other.allowed_hosts.clone();
        }
    }

    /// Parse JSON text into a settings layer. Empty/whitespace-only text
    /// is treated as "no settings here" (returns Default), matching the
    /// missing-file convention. Malformed JSON returns a hard error so
    /// the operator notices instead of silently running on partial config.
    pub fn from_json_str(buf: &str) -> anyhow::Result<Self> {
        let trimmed = buf.trim();
        if trimmed.is_empty() {
            return Ok(Self::default());
        }
        serde_json::from_str::<Self>(trimmed)
            .map_err(|e| anyhow::anyhow!("invalid settings JSON: {e}"))
    }

    /// Read a settings file. Missing files return `Default` (non-error).
    /// Malformed JSON or non-readable files are hard errors.
    pub fn read_from_path(path: &Path) -> anyhow::Result<Self> {
        if !path.exists() {
            return Ok(Self::default());
        }
        let buf = std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("cannot read {}: {e}", path.display()))?;
        Self::from_json_str(&buf).map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))
    }

    /// User-global settings path: `$APR_CONFIG/settings.json` if set,
    /// else `~/.config/apr/settings.json` (XDG-style).
    pub fn user_global_path() -> Option<PathBuf> {
        if let Ok(custom) = std::env::var("APR_CONFIG") {
            if !custom.is_empty() {
                return Some(PathBuf::from(custom).join("settings.json"));
            }
        }
        // dirs::config_dir() returns ~/.config on Linux, equivalent on macOS/Windows.
        dirs::config_dir().map(|d| d.join("apr").join("settings.json"))
    }

    /// Project-local settings path: `<project_root>/.apr/settings.json`.
    pub fn project_local_path(project_root: &Path) -> PathBuf {
        project_root.join(".apr").join("settings.json")
    }

    /// Load and merge the user-global → project-local layers.
    /// CLI overrides happen at the call site (after this returns).
    ///
    /// Field precedence: project-local > user-global > defaults.
    pub fn load_layered(project_root: &Path) -> anyhow::Result<Self> {
        let mut merged = Self::default();
        if let Some(p) = Self::user_global_path() {
            merged.merge(&Self::read_from_path(&p)?);
        }
        merged.merge(&Self::read_from_path(&Self::project_local_path(project_root))?);
        Ok(merged)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // PMAT-876: shared crate-wide env lock + save/restore guard. All
    // env-mutating tests across auto_memory + settings + instructions
    // serialize on this ONE lock because they touch the same global
    // `APR_CONFIG` variable.
    use crate::agent::env_test_support::{env_lock, ScopedEnv};
    use std::fs;
    use std::path::Path;

    fn write(path: &Path, body: &str) {
        if let Some(p) = path.parent() {
            fs::create_dir_all(p).expect("mkdir -p");
        }
        fs::write(path, body).expect("write");
    }

    #[test]
    fn default_is_all_none() {
        let s = AprSettings::default();
        assert!(s.model.is_none());
        assert!(s.max_turns.is_none());
        assert!(s.extra_system_prompt.is_none());
        assert!(s.project.is_none());
    }

    #[test]
    fn from_json_parses_minimal() {
        let s = AprSettings::from_json_str(r#"{"model":"qwen3:1.7b-q4k"}"#).expect("parse");
        assert_eq!(s.model.as_deref(), Some("qwen3:1.7b-q4k"));
        assert!(s.max_turns.is_none());
    }

    #[test]
    fn from_json_parses_full() {
        let s = AprSettings::from_json_str(
            r#"{"model":"qwen3:1.7b-q4k","max_turns":25,"extra_system_prompt":"Be terse","project":"/tmp/proj"}"#,
        )
        .expect("parse");
        assert_eq!(s.model.as_deref(), Some("qwen3:1.7b-q4k"));
        assert_eq!(s.max_turns, Some(25));
        assert_eq!(s.extra_system_prompt.as_deref(), Some("Be terse"));
        assert_eq!(s.project.as_deref(), Some(Path::new("/tmp/proj")));
    }

    #[test]
    fn from_json_empty_is_default() {
        let s = AprSettings::from_json_str("").expect("empty");
        assert_eq!(s, AprSettings::default());
        let s = AprSettings::from_json_str("   \n\t  ").expect("whitespace");
        assert_eq!(s, AprSettings::default());
    }

    #[test]
    fn from_json_malformed_errs_loudly() {
        let err = AprSettings::from_json_str("{not json").expect_err("must err");
        assert!(format!("{err}").contains("invalid settings JSON"));
    }

    #[test]
    fn from_json_unknown_field_is_rejected() {
        // Poka-Yoke: typo in field name shouldn't silently no-op.
        let err = AprSettings::from_json_str(r#"{"modle":"foo"}"#).expect_err("must reject typo");
        assert!(format!("{err}").contains("invalid settings JSON"));
    }

    // PMAT-CODE-CONFIG-LADDER-FIELDS-001 — permission_mode + allowed_hosts

    #[test]
    fn from_json_parses_permission_mode_camel() {
        // Claude Code's `permissionMode` shape (camelCase wire form).
        let s = AprSettings::from_json_str(r#"{"permissionMode":"acceptEdits"}"#).expect("parse");
        assert_eq!(s.permission_mode.as_deref(), Some("acceptEdits"));
    }

    #[test]
    fn from_json_parses_permission_mode_snake_alias() {
        // Operator-friendly snake_case alias also accepted.
        let s = AprSettings::from_json_str(r#"{"permission_mode":"plan"}"#).expect("parse");
        assert_eq!(s.permission_mode.as_deref(), Some("plan"));
    }

    #[test]
    fn from_json_parses_allowed_hosts_camel() {
        let s =
            AprSettings::from_json_str(r#"{"allowedHosts":["docs.anthropic.com","crates.io"]}"#)
                .expect("parse");
        assert_eq!(
            s.allowed_hosts.as_deref(),
            Some(&["docs.anthropic.com".to_string(), "crates.io".to_string()][..])
        );
    }

    #[test]
    fn from_json_parses_allowed_hosts_snake_alias() {
        let s = AprSettings::from_json_str(r#"{"allowed_hosts":["github.com"]}"#).expect("parse");
        assert_eq!(s.allowed_hosts.as_deref(), Some(&["github.com".to_string()][..]));
    }

    #[test]
    fn merge_permission_mode_other_wins() {
        let mut base =
            AprSettings { permission_mode: Some("default".into()), ..Default::default() };
        let over = AprSettings { permission_mode: Some("plan".into()), ..Default::default() };
        base.merge(&over);
        assert_eq!(base.permission_mode.as_deref(), Some("plan"));
    }

    #[test]
    fn merge_allowed_hosts_other_wins_replaces_not_unions() {
        // Settings ladder semantics: project-local fully replaces user-global
        // for any field with `Some(_)`. We do NOT do list-union — operator
        // who wants both must list both in the project file.
        let mut base = AprSettings {
            allowed_hosts: Some(vec!["a.com".into(), "b.com".into()]),
            ..Default::default()
        };
        let over = AprSettings { allowed_hosts: Some(vec!["c.com".into()]), ..Default::default() };
        base.merge(&over);
        assert_eq!(base.allowed_hosts.as_deref(), Some(&["c.com".to_string()][..]));
    }

    #[test]
    fn merge_other_wins() {
        let mut base =
            AprSettings { model: Some("a".into()), max_turns: Some(10), ..Default::default() };
        let over = AprSettings { model: Some("b".into()), ..Default::default() };
        base.merge(&over);
        assert_eq!(base.model.as_deref(), Some("b"));
        assert_eq!(base.max_turns, Some(10), "untouched fields keep base value");
    }

    #[test]
    fn merge_none_keeps_base() {
        let mut base = AprSettings { model: Some("a".into()), ..Default::default() };
        let over = AprSettings::default();
        base.merge(&over);
        assert_eq!(base.model.as_deref(), Some("a"));
    }

    #[test]
    fn read_missing_path_returns_default() {
        let p = std::env::temp_dir().join("does-not-exist-aprcfg.json");
        let _ = std::fs::remove_file(&p);
        let s = AprSettings::read_from_path(&p).expect("missing is ok");
        assert_eq!(s, AprSettings::default());
    }

    #[test]
    fn read_malformed_path_errs_loudly() {
        let dir = tempfile::tempdir().expect("tempdir");
        let p = dir.path().join("settings.json");
        write(&p, "{not json");
        let err = AprSettings::read_from_path(&p).expect_err("must err");
        let msg = format!("{err}");
        assert!(msg.contains("invalid settings JSON") || msg.contains("settings.json"));
    }

    #[test]
    fn user_global_honors_apr_config_env() {
        let _guard = env_lock();
        let dir = tempfile::tempdir().expect("tempdir");
        let _env = ScopedEnv::set("APR_CONFIG", dir.path());
        let p = AprSettings::user_global_path().expect("path resolved");
        assert_eq!(p, dir.path().join("settings.json"));
    }

    #[test]
    fn project_local_path_under_project() {
        let p = AprSettings::project_local_path(Path::new("/tmp/myproj"));
        assert_eq!(p, Path::new("/tmp/myproj/.apr/settings.json"));
    }

    #[test]
    fn load_layered_project_overrides_user_global() {
        let _guard = env_lock();
        // Set up a temp APR_CONFIG with model="user" and a temp project with model="project".
        // Project must win.
        let cfg_dir = tempfile::tempdir().expect("cfg tempdir");
        let proj_dir = tempfile::tempdir().expect("proj tempdir");
        write(&cfg_dir.path().join("settings.json"), r#"{"model":"user-global","max_turns":5}"#);
        write(&proj_dir.path().join(".apr").join("settings.json"), r#"{"model":"project-local"}"#);
        let _env = ScopedEnv::set("APR_CONFIG", cfg_dir.path());
        let s = AprSettings::load_layered(proj_dir.path()).expect("load");

        assert_eq!(s.model.as_deref(), Some("project-local"), "project must win");
        assert_eq!(s.max_turns, Some(5), "user-global field passes through when project is silent");
    }

    #[test]
    fn load_layered_no_files_returns_default() {
        let _guard = env_lock();
        let cfg_dir = tempfile::tempdir().expect("cfg tempdir");
        let proj_dir = tempfile::tempdir().expect("proj tempdir");
        // No settings.json written anywhere.
        let _env = ScopedEnv::set("APR_CONFIG", cfg_dir.path());
        let s = AprSettings::load_layered(proj_dir.path()).expect("load");
        assert_eq!(s, AprSettings::default());
    }
}
