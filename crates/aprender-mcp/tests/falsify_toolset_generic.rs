//! FALSIFY-MCP-TOOLSET-001 (PMAT-3954, paiml/aprender#3954) — a caller-supplied tool set.
//!
//! aprender-mcp served apr's tools only: the dispatcher read a global index built from
//! `inventory` at link time and `initialize` answered the const `SERVER_NAME`. A downstream
//! daemon (arbiter, paiml/infra#917) must serve ITS OWN tools on this server. This test drives
//! the real stdio path (`serve_stream`) with a one-tool index and holds:
//!   - `initialize` names the CALLER's server, not "aprender-mcp";
//!   - `tools/list` lists exactly the caller's tools — no apr tool leaks in;
//!   - `tools/call` reaches the caller's dispatch function, through the worker thread;
//!   - an unknown tool is an error, never a silent success;
//!   - two tools with one name are refused when the index is built.
//!
//! It uses nothing apr-specific, so it compiles and runs with `--no-default-features`.

use std::io::Cursor;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};

use aprender_mcp::tools::{DispatchFn, ToolIndex};
use aprender_mcp::{AprMcpServer, InputSchema, NotificationSink, ToolCallResult, ToolDefinition};

fn echo(
    args: &serde_json::Value,
    _cancel: &mpsc::Receiver<()>,
    _sink: Option<&NotificationSink>,
    _token: Option<serde_json::Value>,
) -> ToolCallResult {
    ToolCallResult::success(format!("echo:{}", args["word"].as_str().unwrap_or("?")))
}

fn def(name: &str) -> ToolDefinition {
    ToolDefinition {
        name: name.to_string(),
        description: "fixture tool".to_string(),
        input_schema: InputSchema {
            schema_type: "object".to_string(),
            properties: Default::default(),
            required: vec![],
        },
    }
}

fn index() -> ToolIndex {
    ToolIndex::from_entries(vec![(def("fx.echo"), echo as DispatchFn)]).expect("one tool, one name")
}

fn drive(input: &str) -> Vec<serde_json::Value> {
    let mut server = AprMcpServer::with_tools("fixture-server", "9.9.9", Arc::new(index()));
    let out = Arc::new(Mutex::new(Vec::<u8>::new()));
    server
        .serve_stream(Cursor::new(input.as_bytes().to_vec()), Arc::clone(&out))
        .expect("serve_stream returns Ok at EOF");
    let bytes = out.lock().expect("output mutex").clone();
    String::from_utf8(bytes)
        .expect("utf-8 output")
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).expect("every output line is JSON"))
        .collect()
}

fn by_id(responses: &[serde_json::Value], id: i64) -> &serde_json::Value {
    responses
        .iter()
        .find(|r| r["id"] == id)
        .unwrap_or_else(|| panic!("no response with id {id}: {responses:?}"))
}

const INIT: &str = r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"t","version":"0"}}}"#;
const LIST: &str = r#"{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}"#;
const CALL: &str = r#"{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"fx.echo","arguments":{"word":"hi"}}}"#;
const MISS: &str = r#"{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"apr.version","arguments":{}}}"#;

#[test]
fn initialize_names_the_callers_server() {
    let r = drive(&format!("{INIT}\n"));
    let info = &by_id(&r, 1)["result"]["serverInfo"];
    assert_eq!(info["name"], "fixture-server", "serverInfo must be the caller's: {info}");
    assert_eq!(info["version"], "9.9.9", "serverInfo version must be the caller's: {info}");
}

#[test]
fn tools_list_is_exactly_the_callers_tools() {
    let r = drive(&format!("{INIT}\n{LIST}\n"));
    let names: Vec<&str> = by_id(&r, 2)["result"]["tools"]
        .as_array()
        .expect("tools array")
        .iter()
        .map(|t| t["name"].as_str().expect("tool name"))
        .collect();
    assert_eq!(names, vec!["fx.echo"], "no apr tool may leak into a caller's list");
}

#[test]
fn tools_call_reaches_the_callers_dispatch_through_the_worker() {
    let r = drive(&format!("{INIT}\n{CALL}\n"));
    let call = by_id(&r, 3);
    assert!(call.get("error").is_none(), "tools/call must succeed: {call}");
    assert_eq!(call["result"]["content"][0]["text"], "echo:hi", "the caller's function answered: {call}");
}

#[test]
fn a_tool_the_caller_did_not_register_is_an_error() {
    let r = drive(&format!("{INIT}\n{MISS}\n"));
    let call = by_id(&r, 4);
    let is_error = call.get("error").is_some() || call["result"]["isError"] == true;
    assert!(is_error, "apr.version is not in the caller's set and must not answer: {call}");
}

#[test]
fn two_tools_with_one_name_are_refused() {
    let dup = ToolIndex::from_entries(vec![
        (def("fx.echo"), echo as DispatchFn),
        (def("fx.echo"), echo as DispatchFn),
    ]);
    assert!(dup.is_err(), "a duplicate tool name must be refused when the index is built");
}
