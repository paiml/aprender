//! Project-root `.mcp.json` loader (PMAT-CODE-MCP-JSON-LOADER-001).
//!
//! Mirrors Claude Code's `<project_root>/.mcp.json` shape:
//!
//! ```json
//! {
//!   "mcpServers": {
//!     "filesystem": {
//!       "command": "npx",
//!       "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
//!       "env": {}
//!     }
//!   }
//! }
//! ```
//!
//! When `apr code` starts, this loader reads the file (if present) and
//! merges its servers into [`AgentManifest::mcp_servers`] before MCP
//! discovery fires. Manifest-declared servers always win on name
//! collision (CLI/manifest beats project-root config — same pattern as
//! the settings-ladder). Missing file is a non-error; malformed JSON
//! is a hard error so the operator notices.
//!
//! Pure-function design: the parser + merge live here as pure
//! functions; the only caller-visible side effect is mutation of an
//! AgentManifest passed by `&mut`.

#![cfg(feature = "agents-mcp")]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::agent::manifest::{AgentManifest, McpServerConfig, McpTransport};

/// Raw `.mcp.json` shape — Claude Code's wire format.
#[derive(Debug, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct McpJson {
    /// Map of server-name → config. `mcpServers` matches Claude
    /// Code's spelling exactly so an existing file copies in unchanged.
    #[serde(default, rename = "mcpServers")]
    pub mcp_servers: BTreeMap<String, McpServerEntry>,
}

/// One entry in the `mcpServers` map. Either `command` (stdio
/// transport) or `url` (sse / websocket) must be present — enforced
/// at merge time, not parse time, so missing-both produces a clear
/// error message.
#[derive(Debug, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct McpServerEntry {
    /// stdio: command to launch (e.g. `"npx"` or `"/usr/local/bin/server"`).
    #[serde(default)]
    pub command: Option<String>,

    /// stdio: argv tail (without command).
    #[serde(default)]
    pub args: Vec<String>,

    /// SSE / WebSocket URL.
    #[serde(default)]
    pub url: Option<String>,

    /// Optional explicit transport. Defaults to `stdio` if `command`
    /// present, `sse` if `url` present.
    #[serde(default)]
    pub transport: Option<String>,

    /// Capability allowlist; `["*"]` = all. Empty = all (Claude
    /// default).
    #[serde(default)]
    pub capabilities: Vec<String>,

    /// Process env (stdio only). Currently parsed but not yet
    /// threaded through to the spawn — tracked in
    /// `PMAT-CODE-MCP-ENV-001` (P2).
    #[serde(default)]
    pub env: BTreeMap<String, String>,
}

/// Project-root `.mcp.json` path: `<project>/.mcp.json`.
pub fn project_mcp_json_path(project_root: &Path) -> PathBuf {
    project_root.join(".mcp.json")
}

/// Parse JSON text. Empty/whitespace-only text returns `Default`
/// (matches the missing-file convention so a `.mcp.json` containing
/// just `{}` or whitespace is a no-op rather than an error).
pub fn from_json_str(buf: &str) -> anyhow::Result<McpJson> {
    let trimmed = buf.trim();
    if trimmed.is_empty() {
        return Ok(McpJson::default());
    }
    serde_json::from_str(trimmed).map_err(|e| anyhow::anyhow!("invalid .mcp.json: {e}"))
}

/// Read `.mcp.json` from `path`. Missing file returns `Default`
/// (non-error); other I/O failures and malformed JSON are hard errors
/// so the operator notices instead of silently running on partial
/// config (Poka-Yoke).
pub fn read_from_path(path: &Path) -> anyhow::Result<McpJson> {
    if !path.exists() {
        return Ok(McpJson::default());
    }
    let buf = std::fs::read_to_string(path)
        .map_err(|e| anyhow::anyhow!("cannot read {}: {e}", path.display()))?;
    from_json_str(&buf).map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))
}

/// Convert a parsed `.mcp.json` into [`McpServerConfig`] entries
/// (the AgentManifest shape). Returns one error per malformed entry
/// rather than the first; callers see all problems at once.
pub fn to_manifest_entries(parsed: &McpJson) -> Result<Vec<McpServerConfig>, Vec<String>> {
    let mut out = Vec::with_capacity(parsed.mcp_servers.len());
    let mut errs = Vec::new();
    for (name, entry) in &parsed.mcp_servers {
        match entry_to_config(name, entry) {
            Ok(cfg) => out.push(cfg),
            Err(e) => errs.push(e),
        }
    }
    if errs.is_empty() {
        Ok(out)
    } else {
        Err(errs)
    }
}

fn entry_to_config(name: &str, entry: &McpServerEntry) -> Result<McpServerConfig, String> {
    if name.is_empty() {
        return Err("mcpServers entry has empty name".to_owned());
    }
    let transport = resolve_transport(name, entry)?;
    let mut command = Vec::new();
    if let Some(ref c) = entry.command {
        command.push(c.clone());
        command.extend(entry.args.iter().cloned());
    }
    Ok(McpServerConfig {
        name: name.to_owned(),
        transport,
        command,
        url: entry.url.clone(),
        capabilities: entry.capabilities.clone(),
    })
}

fn resolve_transport(name: &str, entry: &McpServerEntry) -> Result<McpTransport, String> {
    if let Some(ref t) = entry.transport {
        return match t.as_str() {
            "stdio" => Ok(McpTransport::Stdio),
            "sse" => Ok(McpTransport::Sse),
            "websocket" | "ws" => Ok(McpTransport::WebSocket),
            other => Err(format!(
                "mcpServers[\"{name}\"]: unknown transport \"{other}\" (expected stdio|sse|websocket)"
            )),
        };
    }
    // Heuristic: command → stdio, url → sse, neither → error.
    match (entry.command.is_some(), entry.url.is_some()) {
        (true, _) => Ok(McpTransport::Stdio),
        (false, true) => Ok(McpTransport::Sse),
        (false, false) => Err(format!(
            "mcpServers[\"{name}\"]: must specify either `command` (stdio) or `url` (sse/ws)"
        )),
    }
}

/// Merge `.mcp.json`-derived servers into `manifest.mcp_servers`.
/// Manifest-declared servers (already present by name) always win —
/// the project-root `.mcp.json` is treated as a default that the
/// operator can override via the TOML manifest.
///
/// Returns the count of servers added (so the caller can log a
/// one-line summary). Errors per malformed entry are returned via
/// the `Err` arm; the manifest is left unchanged in that case.
pub fn merge_into_manifest(
    manifest: &mut AgentManifest,
    parsed: &McpJson,
) -> Result<usize, Vec<String>> {
    let entries = to_manifest_entries(parsed)?;
    let existing: std::collections::HashSet<String> =
        manifest.mcp_servers.iter().map(|s| s.name.clone()).collect();
    let mut added = 0;
    for cfg in entries {
        if existing.contains(&cfg.name) {
            continue; // manifest wins
        }
        manifest.mcp_servers.push(cfg);
        added += 1;
    }
    Ok(added)
}

/// One-shot: load `.mcp.json` from the project root (if present),
/// merge into `manifest`. Errors propagate up. Returns the count of
/// servers added.
pub fn load_and_merge(manifest: &mut AgentManifest, project_root: &Path) -> anyhow::Result<usize> {
    let path = project_mcp_json_path(project_root);
    let parsed = read_from_path(&path)?;
    if parsed.mcp_servers.is_empty() {
        return Ok(0);
    }
    merge_into_manifest(manifest, &parsed).map_err(|errs| anyhow::anyhow!(errs.join("; ")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn write(path: &Path, body: &str) {
        if let Some(p) = path.parent() {
            fs::create_dir_all(p).expect("mkdir");
        }
        fs::write(path, body).expect("write");
    }

    // ── from_json_str ──────────────────────────────────────────────

    #[test]
    fn parse_empty_yields_default() {
        let p = from_json_str("").expect("empty ok");
        assert!(p.mcp_servers.is_empty());
        let p = from_json_str("   \n\t   ").expect("whitespace ok");
        assert!(p.mcp_servers.is_empty());
    }

    #[test]
    fn parse_empty_object_yields_default() {
        let p = from_json_str("{}").expect("empty obj ok");
        assert!(p.mcp_servers.is_empty());
    }

    #[test]
    fn parse_minimal_stdio_entry() {
        let s = r#"{
            "mcpServers": {
                "filesystem": {
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-filesystem"]
                }
            }
        }"#;
        let p = from_json_str(s).expect("parse");
        let fs_entry = p.mcp_servers.get("filesystem").expect("entry");
        assert_eq!(fs_entry.command.as_deref(), Some("npx"));
        assert_eq!(fs_entry.args, vec!["-y", "@modelcontextprotocol/server-filesystem"]);
    }

    #[test]
    fn parse_sse_entry() {
        let s = r#"{
            "mcpServers": {
                "remote": {"url": "https://example.com/sse"}
            }
        }"#;
        let p = from_json_str(s).expect("parse");
        assert_eq!(p.mcp_servers["remote"].url.as_deref(), Some("https://example.com/sse"));
    }

    #[test]
    fn parse_unknown_top_level_field_rejected() {
        let s = r#"{"badKey": 1}"#;
        let err = from_json_str(s).expect_err("must reject");
        assert!(format!("{err}").contains("invalid .mcp.json"));
    }

    #[test]
    fn parse_unknown_entry_field_rejected() {
        // Poka-Yoke: typo in entry shouldn't silently no-op.
        let s = r#"{"mcpServers": {"x": {"comand": "npx"}}}"#;
        let err = from_json_str(s).expect_err("must reject typo");
        assert!(format!("{err}").contains("invalid .mcp.json"));
    }

    #[test]
    fn parse_malformed_json_errs_loudly() {
        let err = from_json_str("{not json").expect_err("must err");
        assert!(format!("{err}").contains("invalid .mcp.json"));
    }

    // ── read_from_path ─────────────────────────────────────────────

    #[test]
    fn read_missing_path_returns_default() {
        let p = std::env::temp_dir().join("does-not-exist-mcpjson.json");
        let _ = std::fs::remove_file(&p);
        let parsed = read_from_path(&p).expect("missing ok");
        assert!(parsed.mcp_servers.is_empty());
    }

    #[test]
    fn read_malformed_path_errs_loudly() {
        let dir = tempfile::tempdir().expect("tempdir");
        let p = dir.path().join(".mcp.json");
        write(&p, "{not json");
        let err = read_from_path(&p).expect_err("must err");
        let msg = format!("{err}");
        assert!(msg.contains(".mcp.json"));
    }

    // ── resolve_transport ──────────────────────────────────────────

    #[test]
    fn resolve_transport_stdio_when_command_set() {
        let e = McpServerEntry { command: Some("npx".into()), ..Default::default() };
        let t = resolve_transport("x", &e).unwrap();
        assert!(matches!(t, McpTransport::Stdio));
    }

    #[test]
    fn resolve_transport_sse_when_url_set() {
        let e = McpServerEntry { url: Some("https://x".into()), ..Default::default() };
        let t = resolve_transport("x", &e).unwrap();
        assert!(matches!(t, McpTransport::Sse));
    }

    #[test]
    fn resolve_transport_explicit_overrides_heuristic() {
        let e = McpServerEntry {
            command: Some("npx".into()),
            transport: Some("websocket".into()),
            ..Default::default()
        };
        let t = resolve_transport("x", &e).unwrap();
        assert!(matches!(t, McpTransport::WebSocket));
    }

    #[test]
    fn resolve_transport_unknown_explicit_errs() {
        let e = McpServerEntry { transport: Some("magic".into()), ..Default::default() };
        let err = resolve_transport("x", &e).unwrap_err();
        assert!(err.contains("magic"));
    }

    #[test]
    fn resolve_transport_neither_command_nor_url_errs() {
        let e = McpServerEntry::default();
        let err = resolve_transport("x", &e).unwrap_err();
        assert!(err.contains("must specify"));
    }

    // ── to_manifest_entries ────────────────────────────────────────

    #[test]
    fn to_manifest_entries_collects_errors() {
        let s = r#"{
            "mcpServers": {
                "good": {"command": "npx", "args": ["-y"]},
                "bad":  {}
            }
        }"#;
        let p = from_json_str(s).expect("parse");
        let errs = to_manifest_entries(&p).expect_err("must err on bad entry");
        assert_eq!(errs.len(), 1);
        assert!(errs[0].contains("\"bad\""));
    }

    #[test]
    fn to_manifest_entries_preserves_command_args() {
        let s = r#"{
            "mcpServers": {
                "fs": {"command": "/usr/bin/server", "args": ["--root", "/tmp"]}
            }
        }"#;
        let p = from_json_str(s).expect("parse");
        let cfgs = to_manifest_entries(&p).expect("ok");
        assert_eq!(cfgs.len(), 1);
        assert_eq!(cfgs[0].command, vec!["/usr/bin/server", "--root", "/tmp"]);
    }

    // ── merge_into_manifest ────────────────────────────────────────

    #[test]
    fn merge_adds_new_entries() {
        let mut m = AgentManifest::default();
        let s = r#"{"mcpServers": {"fs": {"command": "npx"}}}"#;
        let parsed = from_json_str(s).expect("parse");
        let added = merge_into_manifest(&mut m, &parsed).expect("ok");
        assert_eq!(added, 1);
        assert_eq!(m.mcp_servers.len(), 1);
        assert_eq!(m.mcp_servers[0].name, "fs");
    }

    #[test]
    fn merge_manifest_wins_on_name_collision() {
        // Manifest already has "fs" — `.mcp.json` "fs" must not overwrite.
        let mut m = AgentManifest::default();
        m.mcp_servers.push(McpServerConfig {
            name: "fs".into(),
            transport: McpTransport::Stdio,
            command: vec!["manifest-cmd".into()],
            url: None,
            capabilities: vec![],
        });
        let s = r#"{"mcpServers": {"fs": {"command": "json-cmd"}}}"#;
        let parsed = from_json_str(s).expect("parse");
        let added = merge_into_manifest(&mut m, &parsed).expect("ok");
        assert_eq!(added, 0, "manifest must win over .mcp.json on name collision");
        assert_eq!(m.mcp_servers.len(), 1);
        assert_eq!(m.mcp_servers[0].command[0], "manifest-cmd");
    }

    #[test]
    fn merge_with_no_entries_is_noop() {
        let mut m = AgentManifest::default();
        let parsed = McpJson::default();
        let added = merge_into_manifest(&mut m, &parsed).expect("ok");
        assert_eq!(added, 0);
        assert!(m.mcp_servers.is_empty());
    }

    // ── load_and_merge ─────────────────────────────────────────────

    #[test]
    fn load_and_merge_missing_file_is_noop() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut m = AgentManifest::default();
        let added = load_and_merge(&mut m, dir.path()).expect("ok");
        assert_eq!(added, 0);
        assert!(m.mcp_servers.is_empty());
    }

    #[test]
    fn load_and_merge_real_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mcp = dir.path().join(".mcp.json");
        write(&mcp, r#"{"mcpServers": {"fs": {"command": "npx", "args": ["-y", "fs-server"]}}}"#);
        let mut m = AgentManifest::default();
        let added = load_and_merge(&mut m, dir.path()).expect("ok");
        assert_eq!(added, 1);
        assert_eq!(m.mcp_servers[0].name, "fs");
        assert_eq!(m.mcp_servers[0].command, vec!["npx", "-y", "fs-server"]);
    }

    #[test]
    fn load_and_merge_malformed_file_errors() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mcp = dir.path().join(".mcp.json");
        write(&mcp, "{not json");
        let mut m = AgentManifest::default();
        let err = load_and_merge(&mut m, dir.path()).expect_err("must err");
        let msg = format!("{err}");
        assert!(msg.contains(".mcp.json"));
    }
}
