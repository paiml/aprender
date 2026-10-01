//! E2E-SETFIT-TRAIN-PMCP-001 — a REAL SetFit training run as an MCP Task,
//! over live stdio, end to end.
//!
//! Env-gated on `APR_MCP_E2E_SETFIT_TRAIN_BIN` (the pinned, setfit-featured
//! `apr` — the `$APR` that `scripts/apr_bin.sh` exports): unset means a
//! println! SKIP and an early return. Once armed, every other precondition
//! FAILS loudly rather than skipping — a half-armed gate that skips is how a
//! broken setup reads as green.
//!
//! What one armed run proves, in order:
//! 1. the server advertises the `tasks` capability (the store is wired);
//! 2. `tools/call train` with the v1 `task` trigger returns a STORE-minted
//!    task id in the frozen `{task: …}` create envelope;
//! 3. single-flight: a second submit while the first trains is refused as a
//!    tool error naming the running job;
//! 4. `tasks/get` polls `working → completed` — which only happens if the
//!    `TrainingTaskStore` decorator observed the create and the waiter wrote
//!    the terminal state against the id and owner dispatch actually used;
//! 5. `tasks/result` serves the status payload with the trainer's own
//!    `--json` report, and the artifact it names EXISTS on disk;
//! 6. the payload's `task_id` matches the polled task, and `train_status`
//!    called with THAT id serves a byte-identical payload — the correlation
//!    that makes "both roads agree" a claim about one run rather than about
//!    whatever happened to run last;
//! 7. an unknown argument key is refused end-to-end (`deny_unknown_fields`).
//!
//! Not covered here because it would cost a second 127-second run: a PLAIN
//! (non-task-augmented) submit must leave no binding for a later task to
//! adopt. That is the unit test `a_plain_call_leaves_no_binding_for_a_later_-
//! task_to_adopt`, and it is the regression that retired the old pairing
//! heuristic.
//!
//! The training child is the REAL recipe (TweetEval stance-abortion, 8 shots,
//! seed 17 — the reference config fixture): measured 127 s wall on an
//! M-series host, hence the generous poll budget.

#![allow(clippy::disallowed_methods)] // serde_json::json! expands to .unwrap() internally

use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

/// The pinned `apr` binary. Unset ⇒ visible SKIP.
const ENV_APR_BIN: &str = "APR_MCP_E2E_SETFIT_TRAIN_BIN";
/// Optional override for the encoder checkout (default: the documented cache).
///
/// This is the repo's ONE name for "where the MiniLM checkout is" — the same
/// variable `production_checkout_dir()` reads in the core conformance suite,
/// the apr-cli lifecycle suite and the train evidence suite. A bespoke
/// `APR_MCP_E2E_SETFIT_TRAIN_MODEL_DIR` would have been a sixth notion of the
/// checkout, so a host that relocated the 86.7 MB directory and exported the
/// shared variable would move four suites and leave this one probing a stale
/// default.
const ENV_MODEL_DIR: &str = "APRENDER_MINILM_DIR";

/// Protocol responses are instant — except `train`'s submit, which runs the
/// CLI's `--dry-run` pre-flight synchronously (manifest replay, seconds).
const CALL_TIMEOUT: Duration = Duration::from_secs(120);
/// The whole training run: 127 s measured, debug-host and CI margin on top.
const TRAIN_BUDGET: Duration = Duration::from_secs(600);
const POLL_INTERVAL: Duration = Duration::from_secs(3);

/// Kill the server when the test unwinds.
///
/// A panicking assertion must not orphan a child that holds inherited pipes
/// open — on the predict sibling that orphan turned one 10-second failure into
/// a 600-second harness timeout. Here the stake is higher: the orphan would own
/// a training child of its own, saturating the CPU for minutes after the test
/// gave up.
struct KillOnDrop(std::process::Child);

impl Drop for KillOnDrop {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

/// Remove the run's output directory however the test leaves — the trailing
/// `remove_dir_all` only ran on the success path, so any of the ~15 assertions
/// that fire AFTER the artifact is written leaked an 86.6 MiB `.apr` (plus its
/// config) into TMPDIR, under a fresh pid-named directory nothing ever reuses.
/// Same reasoning as [`KillOnDrop`], applied to what the child produced.
struct RemoveOnDrop(PathBuf);

impl Drop for RemoveOnDrop {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn request(id: u64, method: &str, params: serde_json::Value) -> String {
    serde_json::json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params })
        .to_string()
}

fn send(stdin: &mut impl Write, line: &str) {
    stdin.write_all(line.as_bytes()).expect("write request");
    stdin.write_all(b"\n").expect("write newline");
    stdin.flush().expect("flush stdin");
}

fn recv_json(rx: &mpsc::Receiver<String>, timeout: Duration, what: &str) -> serde_json::Value {
    let line = rx
        .recv_timeout(timeout)
        .unwrap_or_else(|e| panic!("timed out waiting for {what}: {e}"));
    serde_json::from_str(&line)
        .unwrap_or_else(|e| panic!("non-JSON line while waiting for {what}: {e}\n{line}"))
}

/// One request/response exchange. Owning the id here is what keeps a forgotten
/// increment from being possible, and the assertion below is what makes the id
/// mean something: `recv_json` takes the next line off one FIFO, so a single
/// server-originated frame (`notifications/message` from a log sink, a
/// `notifications/progress` for a request carrying a `progressToken`) would
/// otherwise be consumed AS this call's response and shift every later
/// assertion by one — reported as a task-store defect when nothing was wrong
/// with the run. Frames with no `id` are notifications and are skipped.
fn call(
    stdin: &mut impl Write,
    rx: &mpsc::Receiver<String>,
    next_id: &mut u64,
    method: &str,
    params: serde_json::Value,
    what: &str,
) -> serde_json::Value {
    *next_id += 1;
    send(stdin, &request(*next_id, method, params));
    loop {
        let frame = recv_json(rx, CALL_TIMEOUT, what);
        if frame.get("id").is_none() {
            continue; // a notification, not our response
        }
        assert_eq!(
            frame["id"],
            serde_json::json!(*next_id),
            "response out of order for {what}: {frame:?}"
        );
        return frame;
    }
}

/// A refusal is either a JSON-RPC error or an in-band `isError` result.
fn is_refusal(response: &serde_json::Value) -> bool {
    response.get("error").is_some() || response["result"]["isError"] == serde_json::json!(true)
}

/// The JSON document a tool result carries as its single text content.
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

fn default_model_dir() -> PathBuf {
    let home = std::env::var_os("HOME").expect("HOME is set");
    PathBuf::from(home).join(".cache/aprender/minilm-l6-v2-1110a243")
}

#[test]
#[allow(clippy::too_many_lines)] // one linear protocol transcript, deliberately unfragmented
fn a_training_run_completes_as_an_mcp_task_over_live_stdio() {
    let Some(apr_bin) = std::env::var_os(ENV_APR_BIN) else {
        println!(
            "E2E-SETFIT-TRAIN-PMCP-001 SKIP: {ENV_APR_BIN} unset — export it as the pinned \
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
    let model_dir = std::env::var_os(ENV_MODEL_DIR).map_or_else(default_model_dir, PathBuf::from);
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
        "aprender-mcp-setfit-train-e2e-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&output_dir).expect("create output dir");
    let _output_dir_guard = RemoveOnDrop(output_dir.clone());

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
            // REQUIRED now: streamable HTTP is the default transport, so a
            // spawn without this would bind a socket and never answer a byte
            // on the pipes this test writes to.
            .arg("--stdio")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .expect("spawn aprender-mcp-setfit-train"),
    );
    let mut stdin = child.0.stdin.take().expect("piped stdin");
    let stdout = child.0.stdout.take().expect("piped stdout");

    let (tx, rx) = mpsc::channel::<String>();
    let reader = thread::spawn(move || {
        for line in BufReader::new(stdout).lines() {
            let Ok(line) = line else { break };
            if line.trim().is_empty() {
                continue;
            }
            if tx.send(line).is_err() {
                break;
            }
        }
    });
    let id = &mut 0_u64;

    // 1) initialize — the tasks capability must be advertised (store wired).
    let init = call(
        &mut stdin,
        &rx,
        id,
        "initialize",
        serde_json::json!({
            "protocolVersion": "2025-11-25",
            "capabilities": {},
            "clientInfo": { "name": "e2e", "version": "0" }
        }),
        "initialize response",
    );
    assert!(init.get("error").is_none(), "initialize failed: {init:?}");
    assert!(
        !init["result"]["capabilities"]["tasks"].is_null(),
        "the tasks capability must be auto-advertised when a store is configured: {init:?}"
    );

    // 2) tools/list — exactly the two tools, strict schemas visible.
    let list = call(
        &mut stdin,
        &rx,
        id,
        "tools/list",
        serde_json::json!({}),
        "tools/list response",
    );
    let tools = list["result"]["tools"].as_array().expect("tools array");
    let mut names: Vec<&str> = tools.iter().filter_map(|t| t["name"].as_str()).collect();
    names.sort_unstable();
    // BOTH sides sorted. Comparing a sorted vec against a hand-ordered literal
    // held only by the accident that "train" < "train_status"; renaming
    // TOOL_STATUS to anything sorting earlier turned this red on a correct
    // server. The invariant is "exactly this SET", so sort both.
    let mut expected = vec![
        aprender_mcp_setfit_train::TOOL_TRAIN,
        aprender_mcp_setfit_train::TOOL_STATUS,
        aprender_mcp_setfit_train::TOOL_UPLOAD,
    ];
    expected.sort_unstable();
    assert_eq!(
        names, expected,
        "a thin training server advertises train + train_status and nothing else"
    );
    for tool in tools {
        assert_eq!(
            tool["inputSchema"]["additionalProperties"],
            serde_json::json!(false),
            "deny_unknown_fields must be visible to clients: {tool:?}"
        );
    }

    // 3) tools/call train, task-augmented (v1 trigger: the `task` field).
    let submit = serde_json::json!({
        "name": aprender_mcp_setfit_train::TOOL_TRAIN,
        "arguments": { "config": config },
        "task": {}
    });
    let created = call(
        &mut stdin,
        &rx,
        id,
        "tools/call",
        submit.clone(),
        "task-augmented train response",
    );
    assert!(
        created.get("error").is_none(),
        "train submit errored: {created:?}"
    );
    let task_id = created["result"]["task"]["taskId"]
        .as_str()
        .unwrap_or_else(|| panic!("create envelope must nest a store-minted task: {created:?}"))
        .to_string();
    assert_ne!(
        task_id, "handler-fabricated-discarded",
        "the store-minted id must replace the handler's fabricated one"
    );

    // 4) single-flight: a second submit while the first trains is refused.
    let second = call(
        &mut stdin,
        &rx,
        id,
        "tools/call",
        submit,
        "second (refused) train response",
    );
    assert!(
        is_refusal(&second),
        "a second submit while one trains must be refused: {second:?}"
    );
    // `is_refusal` alone cannot tell single-flight from a spawn error or a
    // config-write failure, so the module doc's claim 3 ("refused as a tool
    // error NAMING the running job") was unproven. Pin the message.
    assert!(
        second.to_string().contains("already running"),
        "the refusal must be single-flight's, not any other error: {second:?}"
    );

    // 5) poll tasks/get until terminal.
    let deadline = Instant::now() + TRAIN_BUDGET;
    let final_status = loop {
        assert!(
            Instant::now() < deadline,
            "training task {task_id} did not reach a terminal status within {TRAIN_BUDGET:?}"
        );
        thread::sleep(POLL_INTERVAL);
        let got = call(
            &mut stdin,
            &rx,
            id,
            "tasks/get",
            serde_json::json!({ "taskId": task_id }),
            "tasks/get response",
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
        "the reference train must complete (a failure completes with an isError result)"
    );

    // 6) tasks/result — the one status payload, carrying the trainer's report.
    let result = call(
        &mut stdin,
        &rx,
        id,
        "tasks/result",
        serde_json::json!({ "taskId": task_id }),
        "tasks/result response",
    );
    assert!(
        !is_refusal(&result),
        "a completed reference train must not be an error result: {result:?}"
    );
    let payload = tool_payload(&result);
    assert_eq!(payload["schema_version"], 1);
    assert_eq!(payload["phase"], "completed");
    let sha = payload["report"]["artifact_sha256"]
        .as_str()
        .expect("the trainer's --json report rides in the payload");
    assert_eq!(sha.len(), 64, "artifact_sha256 must be a full sha256 hex");
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

    // 7) The correlation: this payload names the task we polled, and its
    // that same id selects the SAME run through the polling road. Asserting
    // train_status's *latest* job would pass even if the task had been paired
    // with someone else's run — which is exactly the defect the decorator
    // replaced.
    assert_eq!(
        payload["task_id"],
        serde_json::json!(task_id),
        "the status payload must name the task it was published against"
    );
    // One identity: the task id IS the run id, so a task client and a polling
    // client cannot be talking about different things.
    assert_eq!(payload["task_id"], serde_json::json!(task_id));
    let status = call(
        &mut stdin,
        &rx,
        id,
        "tools/call",
        serde_json::json!({
            "name": aprender_mcp_setfit_train::TOOL_STATUS,
            "arguments": { "task_id": task_id }
        }),
        "train_status response",
    );
    let status_payload = tool_payload(&status);
    assert_eq!(
        status_payload, payload,
        "both roads must serve one shape for one job, byte for byte"
    );

    // 8) strictness probe through the live server.
    let strict = call(
        &mut stdin,
        &rx,
        id,
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
    // `is_refusal` alone made this probe VACUOUS: delete `deny_unknown_fields`
    // and `{"config": {}}` deserializes fine, admits a job, and is then refused
    // by the CLI's own pre-flight for the missing knobs — still a refusal, test
    // still green, regression undetected. Serde's unknown-field error names the
    // offending key; a pre-flight refusal does not.
    assert!(
        strict.to_string().contains("shots"),
        "the refusal must be deny_unknown_fields naming `shots`, not a downstream \
         pre-flight failure: {strict:?}"
    );

    // Shutdown is the client's kill (pmcp's run_stdio does not exit on EOF —
    // measured on the predict sibling); KillOnDrop reaps.
    drop(stdin);
    drop(child);
    reader.join().expect("stdout reader thread panicked");
    // `output_dir` is reaped by `_output_dir_guard`, on this path and on every
    // panicking one.
}
