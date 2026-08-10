//! `apr.serve` — fire-and-forget subprocess wrapper over `apr serve`.
//!
//! Unlike every other Phase-1 tool, this one does NOT wait for the subprocess
//! to exit: `apr serve` is a long-running HTTP daemon. We spawn it, capture
//! the OS pid, and return `{pid, url}` so the MCP client can reach the daemon.
//! The caller is responsible for killing the pid out-of-band.
//!
//! M3 shipped `notifications/cancelled` → SIGTERM → SIGKILL for `apr.run`
//! only (see `server.rs::CancelHandle` docs: "Only `apr.run` currently
//! honours cancellation"). A lifecycle-tracked registry for `apr.serve` —
//! cancel token → SIGTERM the captured pid with 30s grace → SIGKILL — is a
//! post-M3 follow-up targeted at M5 alongside the pmcp dispatcher port (see
//! `docs/specifications/apr-mcp-server-spec.md` § Milestones → M5).
//! Until then, dropping the `Child` leaves a zombie on Unix until the OS
//! parent reaps it.

#![allow(clippy::disallowed_methods)] // serde_json::json! macro expands to .unwrap() internally

use crate::types::{ContentBlock, InputSchema, ToolCallResult, ToolDefinition};
use std::process::{Command, Stdio};

/// Tool name registered with MCP clients.
pub const NAME: &str = "apr.serve";

/// Default HTTP port when the caller omits `port`.
const DEFAULT_PORT: u16 = 8080;

/// Return the MCP tool definition for `apr.serve`.
///
/// FALSIFY-MCP-008: the `inputSchema` is parsed from the build-time codegen
/// constant `crate::schemas::APR_SERVE_SCHEMA`, which `build.rs` emits from
/// `contracts/apr-mcp-tool-schemas-v1.yaml`. The contract is the single
/// source of truth — the live `tools/list` response and the YAML must agree
/// byte-for-byte after JSON canonicalization (asserted by
/// `tests/falsify_mcp_008.rs`).
#[must_use]
pub fn serve_tool_definition() -> ToolDefinition {
    let input_schema: InputSchema = serde_json::from_str(crate::schemas::APR_SERVE_SCHEMA).expect(
        "FALSIFY-MCP-008: apr.serve codegen constant must parse as InputSchema; \
             regenerate by editing contracts/apr-mcp-tool-schemas-v1.yaml and rebuilding",
    );
    ToolDefinition {
        name: NAME.to_string(),
        description: crate::schemas::APR_SERVE_DESCRIPTION.to_string(),
        input_schema,
    }
}

/// Execute `apr.serve` by spawning `apr serve <model_path> --port <port>`.
///
/// Fire-and-forget: the `Child` handle is dropped and the subprocess continues
/// running. On Unix this leaves a zombie until the OS parent reaps it. A
/// lifecycle-tracked registry is a post-M3 follow-up (M5 alongside the pmcp
/// dispatcher port); see this module's header for context.
#[must_use]
pub fn call(args: &serde_json::Value) -> ToolCallResult {
    let Some(model_path) = args.get("model_path").and_then(|v| v.as_str()) else {
        return ToolCallResult::error("Missing required argument: model_path");
    };

    let port: u16 = match args.get("port") {
        None => DEFAULT_PORT,
        Some(v) => match v.as_u64().and_then(|n| u16::try_from(n).ok()) {
            Some(n) => n,
            None => {
                return ToolCallResult::error(format!(
                    "Invalid port: expected integer 0..=65535, got {v}"
                ));
            }
        },
    };

    // #2384: spawn the running `apr` executable, never a `$PATH` lookup —
    // otherwise `apr mcp` from 0.63.0 starts whatever older `apr` is first on
    // `$PATH`, and fails outright when the install dir is not on `$PATH`.
    let spawn_result = Command::new(crate::apr_bin::apr_binary())
        .arg("serve")
        .arg(model_path)
        .arg("--port")
        .arg(port.to_string())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn();

    let child = match spawn_result {
        Ok(c) => c,
        Err(e) => {
            return ToolCallResult::error(format!("failed to spawn apr serve: {e}"));
        }
    };

    let pid: u32 = child.id();
    // Fire-and-forget: drop the Child handle. See module doc-comment for M3
    // follow-up on proper lifecycle tracking.
    drop(child);

    let payload = serde_json::json!({
        "pid": pid,
        "url": format!("http://localhost:{port}"),
        "note": "fire-and-forget: kill pid via OS to stop",
    });
    let text = serde_json::to_string(&payload)
        .unwrap_or_else(|_| format!("{{\"pid\":{pid},\"url\":\"http://localhost:{port}\"}}"));

    ToolCallResult {
        content: vec![ContentBlock::text(text)],
        is_error: None,
    }
}

/// HELIX-IDEA-002 — unified-signature shim for the inventory dispatcher.
pub fn dispatch(
    args: &serde_json::Value,
    _cancel: &std::sync::mpsc::Receiver<()>,
    _sink: Option<&crate::server::NotificationSink>,
    _token: Option<serde_json::Value>,
) -> ToolCallResult {
    call(args)
}

crate::register_mcp_tool!(
    name: NAME,
    definition: serve_tool_definition,
    dispatch: dispatch,
);

#[cfg(test)]
#[allow(clippy::disallowed_methods)] // serde_json::json! expands to code that hits unwrap()
mod tests {
    use super::*;

    #[test]
    fn definition_has_correct_name_and_required_field() {
        let def = serve_tool_definition();
        assert_eq!(def.name, "apr.serve");
        assert_eq!(def.input_schema.schema_type, "object");
        assert_eq!(def.input_schema.required, vec!["model_path".to_string()]);
        for field in ["model_path", "port"] {
            assert!(
                def.input_schema.properties.contains_key(field),
                "{field} property present"
            );
        }
    }

    /// Missing `model_path` must return `isError: true` with the offending
    /// field name — mirrors FALSIFY-MCP-VALIDATE-001. This is the only unit
    /// test we can run without spawning a real `apr serve` daemon.
    #[test]
    fn missing_model_path_returns_error() {
        let result = call(&serde_json::json!({}));
        assert_eq!(result.is_error, Some(true));
        assert!(
            result.content[0].text.contains("model_path"),
            "error message must mention model_path, got: {}",
            result.content[0].text
        );
    }

    #[test]
    fn nonstring_model_path_returns_error() {
        let result = call(&serde_json::json!({ "model_path": 42 }));
        assert_eq!(result.is_error, Some(true));
    }

    #[test]
    fn out_of_range_port_returns_error() {
        let result = call(&serde_json::json!({
            "model_path": "/tmp/x.apr",
            "port": 99999
        }));
        assert_eq!(result.is_error, Some(true));
        assert!(result.content[0].text.contains("port"));
    }
}
