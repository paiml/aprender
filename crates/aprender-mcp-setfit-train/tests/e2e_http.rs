//! E2E-SETFIT-TRAIN-HTTP-001 — a REAL SetFit training run as an MCP Task, over
//! live **streamable HTTP**, against the server as a REMOTE process.
//!
//! The stdio E2E (`e2e_stdio.rs`) proves the lifecycle over the transport a
//! local client spawns. This one proves it over the transport this org actually
//! deploys, and the two differ in ways that can break independently:
//!
//! - the server is reached over a socket, not inherited pipes, so the bound
//!   address has to be discovered rather than assumed;
//! - requests carry `Mcp-Session-Id` and an `Accept` header the transport
//!   validates, neither of which exists on stdio;
//! - the SDK serves every session through ONE `Arc<Mutex<Server>>`, which is
//!   what keeps the mint handoff's single slot correct here — a claim only a
//!   live HTTP run can exercise.
//!
//! Env-gated on `APR_MCP_E2E_SETFIT_TRAIN_BIN` exactly like the stdio E2E:
//! unset means a visible SKIP, armed means every other precondition FAILS
//! loudly rather than skipping.

#![allow(clippy::disallowed_methods)] // serde_json::json! expands to .unwrap() internally

use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

const ENV_APR_BIN: &str = "APR_MCP_E2E_SETFIT_TRAIN_BIN";
const ENV_MODEL_DIR: &str = "APR_MCP_E2E_SETFIT_TRAIN_MODEL_DIR";

/// Submit runs the CLI's `--dry-run` pre-flight synchronously.
const CALL_TIMEOUT: Duration = Duration::from_secs(120);
/// The whole training run: 127 s measured, with debug-host and CI margin.
const TRAIN_BUDGET: Duration = Duration::from_secs(600);
const POLL_INTERVAL: Duration = Duration::from_secs(3);
/// How long to wait for the server to bind and announce its address.
const STARTUP_TIMEOUT: Duration = Duration::from_secs(30);

/// Kill the server when the test unwinds — a panicking assertion must not leave
/// a listener bound, and must not orphan the training child it owns.
struct KillOnDrop(std::process::Child);

impl Drop for KillOnDrop {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

/// Remove a test's artifact directory even when an assertion panics: a leaked
/// run leaves an 86 MiB `.apr` behind.
struct RemoveOnDrop(PathBuf);

impl Drop for RemoveOnDrop {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// One MCP request over streamable HTTP, with the session the server assigned.
struct Client {
    http: reqwest::blocking::Client,
    url: String,
    session: Option<String>,
    next_id: u64,
}

impl Client {
    fn new(base: &str) -> Self {
        Self {
            http: reqwest::blocking::Client::builder()
                .timeout(CALL_TIMEOUT)
                .build()
                .expect("build http client"),
            url: base.to_string(),
            session: None,
            next_id: 0,
        }
    }

    /// POST one JSON-RPC request and return the parsed response.
    ///
    /// `Accept` carries BOTH types the transport accepts, and the session id is
    /// echoed back on every request after `initialize` assigns one — the two
    /// pieces of the streamable-HTTP contract stdio has no equivalent for.
    fn call(&mut self, method: &str, params: serde_json::Value, what: &str) -> serde_json::Value {
        self.next_id += 1;
        let body = serde_json::json!({
            "jsonrpc": "2.0",
            "id": self.next_id,
            "method": method,
            "params": params,
        });
        let mut req = self
            .http
            .post(&self.url)
            .header("content-type", "application/json")
            .header("accept", "application/json, text/event-stream");
        if let Some(session) = self.session.as_deref() {
            req = req.header("mcp-session-id", session);
        }
        let response = req
            .body(body.to_string())
            .send()
            .unwrap_or_else(|e| panic!("{what}: transport error: {e}"));
        let status = response.status();
        if let Some(assigned) = response
            .headers()
            .get("mcp-session-id")
            .and_then(|v| v.to_str().ok())
        {
            self.session = Some(assigned.to_string());
        }
        let text = response
            .text()
            .unwrap_or_else(|e| panic!("{what}: body read failed: {e}"));
        assert!(status.is_success(), "{what}: HTTP {status}\n{text}");
        // A streamable-HTTP server may answer JSON or a one-event SSE stream;
        // accept either rather than pinning the framing this server happens to
        // choose today.
        let payload = text
            .lines()
            .find_map(|line| line.strip_prefix("data: "))
            .unwrap_or(&text);
        serde_json::from_str(payload)
            .unwrap_or_else(|e| panic!("{what}: non-JSON body: {e}\n{text}"))
    }
}

fn is_refusal(response: &serde_json::Value) -> bool {
    response.get("error").is_some() || response["result"]["isError"] == serde_json::json!(true)
}

fn tool_payload(response: &serde_json::Value) -> serde_json::Value {
    let text = response["result"]["content"][0]["text"]
        .as_str()
        .unwrap_or_else(|| panic!("expected one text content: {response:?}"));
    serde_json::from_str(text).unwrap_or_else(|e| panic!("payload is not JSON: {e}\n{text}"))
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("repo root resolves")
}

#[test]
#[allow(clippy::too_many_lines)] // one linear protocol transcript, deliberately unfragmented
fn a_training_run_completes_as_an_mcp_task_over_streamable_http() {
    let Some(apr_bin) = std::env::var_os(ENV_APR_BIN) else {
        println!(
            "E2E-SETFIT-TRAIN-HTTP-001 SKIP: {ENV_APR_BIN} unset — export it as the pinned \
             setfit-featured apr (scripts/apr_bin.sh) to arm this test"
        );
        return;
    };
    let apr_bin = PathBuf::from(apr_bin);
    assert!(
        apr_bin.is_file(),
        "{ENV_APR_BIN} points at {}, which is not a file",
        apr_bin.display()
    );

    let root = repo_root();
    let data = root.join("data/tweet-eval-stance");
    let selection = data.join("selection-manifest.json");
    let model_dir = std::env::var_os(ENV_MODEL_DIR).map_or_else(
        || {
            PathBuf::from(std::env::var_os("HOME").expect("HOME is set"))
                .join(".cache/aprender/minilm-l6-v2-1110a243")
        },
        PathBuf::from,
    );
    let config_path = root.join("crates/apr-cli/tests/fixtures/setfit/train-config.json");
    for (what, path, is_dir) in [
        ("data dir (apr data tweet-eval-stance)", &data, true),
        ("selection manifest (apr data select)", &selection, false),
        ("encoder checkout", &model_dir, true),
        ("reference train config fixture", &config_path, false),
    ] {
        let ok = if is_dir {
            path.is_dir()
        } else {
            path.is_file()
        };
        assert!(ok, "armed E2E but {what} is missing at {}", path.display());
    }
    let config: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(&config_path).expect("read train config fixture"),
    )
    .expect("train config fixture parses");

    let output_dir = std::env::temp_dir().join(format!(
        "aprender-mcp-setfit-train-http-e2e-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&output_dir).expect("create output dir");
    let _cleanup = RemoveOnDrop(output_dir.clone());

    // Port 0: the OS picks, so a busy CI box cannot collide. The server prints
    // where it landed, which is the only way this test can know.
    let mut child = KillOnDrop(
        Command::new(env!("CARGO_BIN_EXE_aprender-mcp-setfit-train"))
            .arg("--apr-bin")
            .arg(&apr_bin)
            .arg("--data")
            .arg(&data)
            .arg("--selection")
            .arg(&selection)
            .arg("--model-dir")
            .arg(&model_dir)
            .arg("--output-dir")
            .arg(&output_dir)
            .arg("--addr")
            .arg("127.0.0.1:0")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn aprender-mcp-setfit-train"),
    );
    let stderr = child.0.stderr.take().expect("piped stderr");
    let (tx, rx) = mpsc::channel::<String>();
    thread::spawn(move || {
        for line in BufReader::new(stderr).lines() {
            let Ok(line) = line else { break };
            eprintln!("server: {line}");
            if tx.send(line).is_err() {
                break;
            }
        }
    });

    let deadline = Instant::now() + STARTUP_TIMEOUT;
    let base = loop {
        assert!(
            Instant::now() < deadline,
            "the server did not announce a bound address within {STARTUP_TIMEOUT:?}"
        );
        let line = rx
            .recv_timeout(STARTUP_TIMEOUT)
            .expect("server stderr closed before it announced an address");
        if let Some(url) = line.split("listening on ").nth(1) {
            break url.trim().to_string();
        }
    };
    let mut client = Client::new(&base);

    // 1) initialize — the tasks capability must be advertised, and the
    // transport must assign a session.
    let init = client.call(
        "initialize",
        serde_json::json!({
            "protocolVersion": "2025-11-25",
            "capabilities": {},
            "clientInfo": { "name": "e2e-http", "version": "0" }
        }),
        "initialize",
    );
    assert!(init.get("error").is_none(), "initialize failed: {init:?}");
    assert!(
        !init["result"]["capabilities"]["tasks"].is_null(),
        "the tasks capability must be advertised over HTTP too: {init:?}"
    );
    assert!(
        client.session.is_some(),
        "the streamable-HTTP transport must assign an Mcp-Session-Id"
    );

    // 2) the same two tools, over the remote transport.
    let list = client.call("tools/list", serde_json::json!({}), "tools/list");
    let tools = list["result"]["tools"].as_array().expect("tools array");
    let mut names: Vec<&str> = tools.iter().filter_map(|t| t["name"].as_str()).collect();
    names.sort_unstable();
    let mut expected = vec![
        aprender_mcp_setfit_train::TOOL_TRAIN,
        aprender_mcp_setfit_train::TOOL_STATUS,
        aprender_mcp_setfit_train::TOOL_UPLOAD,
    ];
    expected.sort_unstable();
    assert_eq!(names, expected);

    // 3) task-augmented train (v1 trigger: the `task` field).
    // `dataset_uri` names the SAME directory the server defaults to, so the
    // run is the reference run — what changes is that the paths reach the CLI
    // through the per-run override rather than the packaged default. On the
    // local dispatcher a dataset URI is a directory on this machine.
    let submit = serde_json::json!({
        "name": aprender_mcp_setfit_train::TOOL_TRAIN,
        "arguments": { "config": config, "dataset_uri": data.display().to_string() },
        "task": {}
    });
    let created = client.call("tools/call", submit.clone(), "task-augmented train");
    assert!(created.get("error").is_none(), "train errored: {created:?}");
    let task_id = created["result"]["task"]["taskId"]
        .as_str()
        .unwrap_or_else(|| panic!("create envelope must nest a store-minted task: {created:?}"))
        .to_string();

    // 4) single-flight, over HTTP.
    let second = client.call("tools/call", submit, "second (refused) train");
    assert!(
        is_refusal(&second),
        "a second submit while one trains must be refused: {second:?}"
    );

    // 5) poll tasks/get to terminal. Each poll is its own HTTP request, which
    // is the property that makes this shape survive a 30-second gateway.
    let deadline = Instant::now() + TRAIN_BUDGET;
    let final_status = loop {
        assert!(
            Instant::now() < deadline,
            "training task {task_id} did not finish within {TRAIN_BUDGET:?}"
        );
        thread::sleep(POLL_INTERVAL);
        let got = client.call(
            "tasks/get",
            serde_json::json!({ "taskId": task_id }),
            "tasks/get",
        );
        let status = got["result"]["task"]["status"]
            .as_str()
            .unwrap_or_else(|| panic!("tasks/get must nest the task: {got:?}"))
            .to_string();
        if status != "working" {
            break status;
        }
    };
    assert_eq!(
        final_status, "completed",
        "the reference train must complete (a failure reports status `failed`)"
    );

    // 6) tasks/result carries the trainer's own report, and the artifact exists.
    let result = client.call(
        "tasks/result",
        serde_json::json!({ "taskId": task_id }),
        "tasks/result",
    );
    assert!(!is_refusal(&result), "completed run errored: {result:?}");
    let payload = tool_payload(&result);
    assert_eq!(payload["schema_version"], 1);
    assert_eq!(payload["phase"], "completed");
    assert_eq!(payload["task_id"], serde_json::json!(task_id));
    let sha = payload["report"]["artifact_sha256"]
        .as_str()
        .expect("the trainer's --json report rides in the payload");
    assert_eq!(sha.len(), 64);
    let artifact = PathBuf::from(
        payload["artifact_path"]
            .as_str()
            .expect("artifact_path in payload"),
    );
    assert!(
        artifact.is_file(),
        "the payload names {} but nothing is there",
        artifact.display()
    );

    // 7) the polling road, asked about THIS task, agrees byte for byte.
    let status = client.call(
        "tools/call",
        serde_json::json!({
            "name": aprender_mcp_setfit_train::TOOL_STATUS,
            "arguments": { "task_id": task_id }
        }),
        "train_status",
    );
    assert_eq!(
        tool_payload(&status),
        payload,
        "both roads must serve one shape for one task, over HTTP as over stdio"
    );

    // 8) strictness survives the transport.
    let strict = client.call(
        "tools/call",
        serde_json::json!({
            "name": aprender_mcp_setfit_train::TOOL_TRAIN,
            "arguments": { "config": {}, "shots": 8 }
        }),
        "unknown-key refusal",
    );
    assert!(
        is_refusal(&strict),
        "an unknown argument key must be refused end-to-end: {strict:?}"
    );

    drop(child);
}
