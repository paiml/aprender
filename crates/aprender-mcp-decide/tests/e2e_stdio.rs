//! E2E-DECIDE-PMCP-001 — the thin decide server classifies over live stdio MCP.
//!
//! The first leg ALWAYS runs: it packs the checked-in tiny Laya fixture
//! (`crates/aprender-decide/tests/fixtures/laya_tiny`, plan 08-02) into a real `.apr` on
//! disk with the production packer, spawns THIS crate's binary on it, and holds the
//! answers to Laya's own oracle. The second leg runs only when
//! `APR_MCP_E2E_DECIDE_MODEL` names a real decide `.apr` (plan 08-09 packs one); it
//! otherwise prints `SKIP` and returns — never `#[ignore]`.
//!
//! The binary is pinned by `env!("CARGO_BIN_EXE_...")`: no PATH, no shadowed artifact.

#![allow(clippy::disallowed_methods)] // serde_json::json! expands to .unwrap() internally

use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use sha2::{Digest, Sha256};

const ENV_MODEL: &str = "APR_MCP_E2E_DECIDE_MODEL";
/// The initialize response arrives only after the model loads. The tiny fixture loads
/// in well under a second; a real ~850 MB artifact spends about a minute on the load
/// ladder in a DEBUG build (08-05 measured 64 s), so the budget is generous.
const INIT_TIMEOUT: Duration = Duration::from_secs(300);
/// A debug-build forward pass over a few full-window rows of the real model.
const CALL_TIMEOUT: Duration = Duration::from_secs(300);

/// Kill the server when the test unwinds — a panicking assertion must not orphan a
/// child that holds inherited pipes open.
struct KillOnDrop(Child);

impl Drop for KillOnDrop {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

/// A line-oriented JSON-RPC client over the child's stdio.
struct Client {
    stdin: ChildStdin,
    rx: mpsc::Receiver<String>,
    next_id: u64,
    // Declared last so the pipes above close before the child is killed and reaped.
    _child: KillOnDrop,
}

impl Client {
    fn spawn(model: &Path) -> Self {
        let mut child = KillOnDrop(
            Command::new(env!("CARGO_BIN_EXE_aprender-mcp-decide"))
                .arg("--model")
                .arg(model)
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::inherit())
                .spawn()
                .expect("spawn aprender-mcp-decide"),
        );
        let stdin = child.0.stdin.take().expect("piped stdin");
        let stdout = child.0.stdout.take().expect("piped stdout");
        let (tx, rx) = mpsc::channel::<String>();
        thread::spawn(move || {
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
        let mut client = Self {
            stdin,
            rx,
            next_id: 1,
            _child: child,
        };
        let init = client.request(
            "initialize",
            serde_json::json!({
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": { "name": "e2e", "version": "0" }
            }),
            INIT_TIMEOUT,
        );
        assert!(init.get("error").is_none(), "initialize failed: {init:?}");
        client
    }

    /// Write one request frame WITHOUT waiting for its reply; returns its id.
    fn send(&mut self, method: &str, params: serde_json::Value) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        let line =
            serde_json::json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params })
                .to_string();
        self.stdin
            .write_all(line.as_bytes())
            .expect("write request");
        self.stdin.write_all(b"\n").expect("write newline");
        self.stdin.flush().expect("flush");
        id
    }

    /// Read the next reply line, whichever request it answers.
    fn recv(&mut self, what: &str, timeout: Duration) -> serde_json::Value {
        let reply = self
            .rx
            .recv_timeout(timeout)
            .unwrap_or_else(|e| panic!("timed out waiting for {what}: {e}"));
        serde_json::from_str(&reply)
            .unwrap_or_else(|e| panic!("non-JSON line for {what}: {e}\n{reply}"))
    }

    fn request(
        &mut self,
        method: &str,
        params: serde_json::Value,
        timeout: Duration,
    ) -> serde_json::Value {
        let id = self.send(method, params);
        let value = self.recv(&format!("{method} #{id}"), timeout);
        assert_eq!(value["id"], id, "reply id for {method}");
        value
    }

    fn list_tools(&mut self) -> Vec<serde_json::Value> {
        let list = self.request("tools/list", serde_json::json!({}), INIT_TIMEOUT);
        list["result"]["tools"]
            .as_array()
            .expect("tools array")
            .clone()
    }

    fn call(&mut self, arguments: serde_json::Value) -> serde_json::Value {
        self.request(
            "tools/call",
            serde_json::json!({ "name": aprender_mcp_decide::TOOL_NAME, "arguments": arguments }),
            CALL_TIMEOUT,
        )
    }
}

/// The classify payload of a successful call (pmcp returns a Value tool's output as
/// JSON text content).
fn payload(call: &serde_json::Value) -> serde_json::Value {
    assert!(call.get("error").is_none(), "tools/call errored: {call:?}");
    let result = &call["result"];
    assert_ne!(
        result["isError"],
        serde_json::json!(true),
        "classify must not be an in-band error: {result:?}"
    );
    let text = result["content"][0]["text"]
        .as_str()
        .expect("content[0].text must be a string");
    serde_json::from_str(text).expect("classify payload is JSON")
}

/// The refusal message of a BOUND refusal, which must arrive in the wire shape
/// `refusal_names_bound` states (plan 08-28, B-iserror): a SUCCESSFUL JSON-RPC response —
/// no `error` member, so never -32603 — whose `result` is a tool result with
/// `isError: true`, exactly one text content item carrying the message, and no
/// `structuredContent` (the refusal has no second, structured copy that could drift).
fn refusal(call: &serde_json::Value) -> String {
    assert!(
        call.get("error").is_none(),
        "a bound refusal must not be a JSON-RPC error: {call}"
    );
    let result = &call["result"];
    assert_eq!(
        result["isError"],
        serde_json::json!(true),
        "a bound refusal is an isError tool result: {call}"
    );
    assert!(
        result.get("structuredContent").is_none(),
        "no structured copy: {call}"
    );
    let content = result["content"].as_array().expect("content array");
    assert_eq!(content.len(), 1, "one content item: {call}");
    assert_eq!(content[0]["type"], "text", "text content: {call}");
    content[0]["text"]
        .as_str()
        .expect("refusal text")
        .to_string()
}

fn sha256_file(path: &Path) -> String {
    let bytes = std::fs::read(path).expect("read served file");
    Sha256::digest(&bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../aprender-decide/tests/fixtures/laya_tiny")
}

/// Pack the tiny fixture with the production packer into a real `.apr` on disk. The
/// tempdir is returned so the file outlives the server that serves it.
fn packed_tiny() -> (tempfile::TempDir, PathBuf) {
    let dir = fixture_dir();
    let bytes = aprender_decide::pack_run_dir(&dir, &dir.join("data")).expect("pack laya_tiny");
    let tmp = tempfile::tempdir().expect("tempdir");
    let model = tmp.path().join("laya_tiny.apr");
    std::fs::write(&model, &bytes).expect("write packed .apr");
    (tmp, model)
}

fn probs_abs() -> f64 {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../contracts/laya-parity-v1.yaml");
    let text = std::fs::read_to_string(&path).expect("read laya-parity-v1");
    let doc: serde_yaml::Value = serde_yaml::from_str(&text).expect("parse laya-parity-v1");
    doc["equations"]["probs_abs"]["float_tolerance"]
        .as_f64()
        .expect("laya-parity-v1 equations.probs_abs.float_tolerance")
}

fn f64s(v: &serde_json::Value) -> Vec<f64> {
    v.as_array()
        .expect("number array")
        .iter()
        .map(|x| x.as_f64().expect("number"))
        .collect()
}

/// Every label must appear in `description`, first occurrences in `labels` order.
fn assert_labels_in_order(description: &str, labels: &[&str]) {
    let mut last = 0usize;
    for label in labels {
        let at = description[last..]
            .find(label)
            .unwrap_or_else(|| panic!("description lacks label {label:?} after byte {last}"));
        last += at + label.len();
    }
}

#[test]
fn the_tiny_decide_server_classifies_over_live_stdio() {
    let dir = fixture_dir();
    let (_tmp, model) = packed_tiny();

    let mut client = Client::spawn(&model);

    // Exactly ONE tool, strict, whose description is the artifact's own task.
    let tools = client.list_tools();
    assert_eq!(tools.len(), 1, "one model, one tool: {tools:?}");
    assert_eq!(tools[0]["name"], aprender_mcp_decide::TOOL_NAME);
    assert_eq!(
        tools[0]["inputSchema"]["additionalProperties"],
        serde_json::json!(false),
        "deny_unknown_fields must be visible to clients"
    );
    let description = tools[0]["description"].as_str().expect("description");
    assert!(
        description.contains("Which team should handle this customer message?"),
        "the description states the artifact's question: {description}"
    );
    let task_labels = ["shipping", "billing", "account"];
    assert_labels_in_order(description, &task_labels);

    // Classify the oracle's task rows (one of them over the 64-token window).
    let oracle: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("oracle.json")).expect("read oracle.json"))
            .expect("parse oracle.json");
    let rows: Vec<&serde_json::Value> = oracle["rows"]
        .as_array()
        .expect("oracle rows")
        .iter()
        .filter(|r| r["qid"] == "team")
        .collect();
    assert!(rows.len() >= 2, "the oracle has task rows");
    let texts: Vec<&str> = rows
        .iter()
        .map(|r| r["state"].as_str().expect("state"))
        .collect();
    // The SERVED count bound is tier policy (decide-tool-boundary-v1: 8 texts at 10 240 MB),
    // so the rows go in contract-sized batches; each fixture row is at most 64 tokens, so a
    // full batch fits the budget. The results are re-joined in input order.
    let batch = aprender_mcp_decide::ClassifyLimits::CONTRACTED.max_texts;
    let outs: Vec<serde_json::Value> = texts
        .chunks(batch)
        .map(|chunk| payload(&client.call(serde_json::json!({ "texts": chunk }))))
        .collect();
    let mut out = outs[0].clone();
    for later in &outs[1..] {
        assert_eq!(later["model"], out["model"], "one identity across batches");
        assert_eq!(
            later["labels"], out["labels"],
            "one label order across batches"
        );
    }
    out["results"] = serde_json::Value::Array(
        outs.iter()
            .flat_map(|o| o["results"].as_array().expect("results").clone())
            .collect(),
    );

    assert_eq!(
        out["model"]["artifact_sha256"].as_str(),
        Some(sha256_file(&model).as_str()),
        "the served identity is the sha256 of the file on disk (D-11)"
    );
    assert_eq!(
        out["model"]["recipe_id"].as_str(),
        Some(sha256_file(&dir.join("recipe.json")).as_str()),
        "recipe_id is the sha256 of recipe.json"
    );
    assert_eq!(out["model"]["method"], "laya");
    assert_eq!(out["model"]["base"], "laya-tiny-synthetic@fixtures");
    assert_eq!(out["labels"], serde_json::json!(task_labels), "task order");

    let bar = probs_abs();
    let results = out["results"].as_array().expect("results");
    assert_eq!(results.len(), rows.len(), "one result per text, in order");
    let mut worst = 0.0f64;
    let mut saw_truncated = false;
    for (i, (row, r)) in rows.iter().zip(results).enumerate() {
        let got = f64s(&r["probabilities"]);
        let want = f64s(&row["probabilities"]);
        assert_eq!(got.len(), task_labels.len(), "row {i}: K probabilities");
        for (g, w) in got.iter().zip(&want) {
            let d = (g - w).abs();
            assert!(d <= bar, "row {i}: |{g} - {w}| = {d} > probs_abs {bar}");
            worst = worst.max(d);
        }
        let argmax = usize::try_from(row["argmax"].as_u64().expect("argmax")).expect("fits");
        assert_eq!(r["label"], task_labels[argmax], "row {i}: argmax label");
        let ids = row["ids"].as_array().expect("ids").len();
        assert_eq!(r["tokens"].as_u64(), Some(ids as u64), "row {i}: tokens");
        assert_eq!(
            r["truncated"], row["truncated"],
            "row {i}: truncated (D-12)"
        );
        saw_truncated |= r["truncated"] == serde_json::json!(true);
    }
    assert!(
        saw_truncated,
        "the oracle's over-window row must come back truncated"
    );
    println!(
        "E2E-DECIDE-PMCP-001: {} rows over live stdio, max|d| {worst:.3e} (probs_abs {bar:e})",
        rows.len()
    );

    // The SERVED bounds are the contracted ones: 9 texts is refused naming the key.
    let nine = vec!["hello"; 9];
    let message = refusal(&client.call(serde_json::json!({ "texts": nine })));
    assert!(
        message.contains("classify_max_texts") && message.contains("decide-tool-boundary-v1"),
        "the live server enforces the contracted count bound: {message}"
    );

    // The count is checked on the JSON array BEFORE any element becomes a String
    // (classify_count_bound, V5-c): classify_max_texts + 1 NON-strings are refused by their
    // count over the real transport. Had the elements been deserialized first, the refusal
    // would be the shape message, which does not name classify_max_texts.
    let over = aprender_mcp_decide::ClassifyLimits::CONTRACTED.max_texts + 1;
    let numbers: Vec<u64> = (0..over as u64).collect();
    let message = refusal(&client.call(serde_json::json!({ "texts": numbers })));
    assert!(
        message.contains("classify_max_texts")
            && message.contains("decide-tool-boundary-v1")
            && !message.contains("deny_unknown_fields"),
        "the live server refuses an oversized list by its count, before its shape: {message}"
    );

    // Caller-supplied labels are a rejection, not a silently ignored knob (D-09).
    let strict = refusal(&client.call(serde_json::json!({ "texts": ["a"], "labels": ["x", "y"] })));
    assert!(
        strict.contains("deny_unknown_fields"),
        "an unknown key must be refused end-to-end: {strict}"
    );

    // A malformed argument is refused WITHOUT echoing caller text (refusal_names_bound, ASVS
    // V7): pmcp's typed-tool path would have returned serde's message, which quotes it.
    const SECRET: &str = "ZQXJ-e2e-caller-document-4111";
    let mut unknown_key = serde_json::Map::new();
    unknown_key.insert("texts".to_string(), serde_json::json!(["a"]));
    unknown_key.insert(SECRET.to_string(), serde_json::json!(1));
    for bad in [
        serde_json::json!({ "texts": SECRET }),
        serde_json::Value::Object(unknown_key),
    ] {
        let message = refusal(&client.call(bad));
        assert!(
            message.contains("decide-tool-boundary-v1") && !message.contains(SECRET),
            "a malformed argument must be refused without echo: {message}"
        );
    }
}

/// `classify_admission` over the SHIPPED transport (plan 08-28, V5-b).
///
/// pmcp 2.19.3's stdio server hands every inbound request to ONE sequential worker over an
/// unbounded channel (`Server::run` -> `spawn_request_worker`), so `ClassifyService::call`
/// is never concurrent through this transport and the `Busy` refusal is not reachable from
/// it. `classify_max_pending + 1` calls written back to back, before any reply is read, must
/// therefore ALL be classified, in request order — none refused naming
/// `classify_max_pending`. Each text is one word longer than the last, so every reply is tied
/// to its request by content (its built-row length), not only by its JSON-RPC id.
#[test]
fn pipelined_calls_are_serialized_not_refused() {
    let (_tmp, model) = packed_tiny();
    let mut client = Client::spawn(&model);

    let limits = aprender_mcp_decide::ClassifyLimits::CONTRACTED;
    let n = limits.max_pending + 1;
    let texts: Vec<String> = (1..=n).map(|k| vec!["parcel"; k].join(" ")).collect();
    let ids: Vec<u64> = texts
        .iter()
        .map(|text| {
            client.send(
                "tools/call",
                serde_json::json!({
                    "name": aprender_mcp_decide::TOOL_NAME,
                    "arguments": { "texts": [text] }
                }),
            )
        })
        .collect();

    let mut last_tokens = 0u64;
    for (i, id) in ids.iter().enumerate() {
        let reply = client.recv(&format!("pipelined call {i}"), CALL_TIMEOUT);
        // A Busy refusal is a bound refusal: an isError tool result since plan 08-28
        // (B-iserror); a JSON-RPC error is read too, so neither shape can hide one.
        let refused = reply["error"]["message"].as_str().or_else(|| {
            (reply["result"]["isError"] == serde_json::json!(true))
                .then(|| reply["result"]["content"][0]["text"].as_str().unwrap_or(""))
        });
        if let Some(message) = refused {
            panic!(
                "pipelined call {i} of {n} was refused (names classify_max_pending: {}): {message}",
                message.contains("classify_max_pending")
            );
        }
        assert_eq!(reply["id"], *id, "reply {i} arrives in request order");
        let out = payload(&reply);
        let results = out["results"].as_array().expect("results");
        assert_eq!(results.len(), 1, "reply {i}: one result for its one text");
        assert_eq!(
            results[0]["truncated"],
            serde_json::json!(false),
            "reply {i}: the texts stay inside the tiny window, so tokens order them"
        );
        let tokens = results[0]["tokens"].as_u64().expect("tokens");
        assert!(
            tokens > last_tokens,
            "reply {i} answers text {i}: its built row ({tokens}) is longer than the previous \
             reply's ({last_tokens})"
        );
        last_tokens = tokens;
    }
    println!(
        "classify_admission over live stdio: {n} pipelined calls (classify_max_pending {} + 1) \
         all classified in request order",
        limits.max_pending
    );
}

/// `refusal_names_bound` over the SHIPPED transport (plan 08-28, B-iserror): every bound
/// refusal the live stdio server can produce — both count bounds, the argument shape, the
/// unknown key, the per-text byte bound and, where the fixture can reach it, the token
/// budget — arrives as a successful `tools/call` result with `isError: true` and its message
/// as the one text content (the shape [`refusal`] asserts), never as JSON-RPC -32603. Each
/// message names the contract and its key and carries none of the caller's 12-character
/// distinctive text. A classify on the same connection afterwards still succeeds: an in-band
/// refusal leaves the server serving.
///
/// The token budget is reachable over the served binary only when `max_texts` full-window
/// rows of the tiny fixture (64 tokens each) exceed `classify_max_total_tokens`. At the
/// contracted 10 240 MB tier (decide-tool-boundary-v1 7.0.0: 8 texts, 800 tokens) they do
/// not (8 x 64 = 512), so instead of sending a request no legal count can build, the test
/// proves the unreachability: `max_texts` full-window texts are CLASSIFIED. The budget
/// refusal is the same `tool_rejected` value (`error_taxonomy_budget_rejected_model_internal`)
/// whose wire mapping the other five cases prove.
#[test]
fn bound_refusals_are_iserror_results_over_live_stdio() {
    const SECRET: &str = "ZQXJ-e2e-4kq";
    let (_tmp, model) = packed_tiny();
    let mut client = Client::spawn(&model);
    let limits = aprender_mcp_decide::ClassifyLimits::CONTRACTED;

    let over_count = vec![SECRET; limits.max_texts + 1];
    let over_bytes = format!("{SECRET} {}", "y".repeat(limits.max_text_bytes));
    // A text whose built row is the tiny fixture's full 64-token window.
    let window_filler = format!("{SECRET} {}", "parcel ".repeat(200));
    let window = 64;
    let budget_reachable = limits.max_texts * window > limits.max_total_tokens;
    let mut unknown_key = serde_json::Map::new();
    unknown_key.insert("texts".to_string(), serde_json::json!([SECRET]));
    unknown_key.insert(SECRET.to_string(), serde_json::json!(1));
    let cases = [
        (
            "classify_max_texts",
            serde_json::json!({ "texts": over_count }),
        ),
        ("classify_min_texts", serde_json::json!({ "texts": [] })),
        (
            "deny_unknown_fields",
            serde_json::json!({ "texts": SECRET }),
        ),
        (
            "deny_unknown_fields",
            serde_json::Value::Object(unknown_key),
        ),
        (
            "classify_max_text_bytes",
            serde_json::json!({ "texts": [over_bytes] }),
        ),
    ];
    let mut cases = cases.to_vec();
    if budget_reachable {
        cases.push((
            "classify_max_total_tokens",
            serde_json::json!({ "texts": vec![window_filler.as_str(); limits.max_texts] }),
        ));
    } else {
        // Unreachable at this tier, and measured so: the most a legal count can build is
        // classified, every row at the full window and truncated.
        let full =
            payload(&client.call(
                serde_json::json!({ "texts": vec![window_filler.as_str(); limits.max_texts] }),
            ));
        let results = full["results"].as_array().expect("results");
        assert_eq!(results.len(), limits.max_texts);
        for r in results {
            assert_eq!(r["tokens"], window, "a full-window row: {r}");
            assert_eq!(r["truncated"], true, "{r}");
        }
    }
    let refusals = cases.len();
    for (key, arguments) in cases {
        let message = refusal(&client.call(arguments));
        assert!(
            message.contains(key) && message.contains("decide-tool-boundary-v1"),
            "{key}: the in-band refusal names its bound: {message}"
        );
        assert!(
            !message.contains(SECRET),
            "{key} echoed the text: {message}"
        );
    }
    let out = payload(&client.call(serde_json::json!({ "texts": ["Where is my parcel?"] })));
    assert_eq!(out["results"].as_array().map(Vec::len), Some(1));
    println!(
        "refusal_names_bound over live stdio: {refusals} bound refusals, each an isError result \
         (token budget reachable with the tiny fixture: {budget_reachable})"
    );
}

#[test]
fn a_real_decide_model_classifies_over_live_stdio() {
    let Some(model) = std::env::var_os(ENV_MODEL) else {
        println!("SKIP: {ENV_MODEL} not set");
        return;
    };
    let model = PathBuf::from(model);
    assert!(
        model.is_file(),
        "{ENV_MODEL} points at {}, which is not a file",
        model.display()
    );
    let mut client = Client::spawn(&model);
    let tools = client.list_tools();
    assert_eq!(tools.len(), 1, "one model, one tool: {tools:?}");
    assert_eq!(tools[0]["name"], aprender_mcp_decide::TOOL_NAME);

    // One text per call: this leg proves the served model, not the batch bound (at the
    // superseded 3 008 MB tier two real sentences, about 70 built tokens each, exceeded the
    // 120-token budget together).
    let mut out = payload(&client.call(serde_json::json!({
        "texts": ["I think this policy is a terrible idea and should be scrapped."]
    })));
    let second = payload(&client.call(serde_json::json!({
        "texts": ["The weather in Lisbon was lovely this weekend."]
    })));
    assert_eq!(second["model"], out["model"], "one identity across calls");
    let mut results = out["results"].as_array().expect("results").clone();
    results.extend(
        second["results"]
            .as_array()
            .expect("results")
            .iter()
            .cloned(),
    );
    out["results"] = serde_json::Value::Array(results);
    assert_eq!(
        out["model"]["artifact_sha256"].as_str(),
        Some(sha256_file(&model).as_str()),
        "the served identity is the sha256 of the file on disk (D-11)"
    );
    let labels = out["labels"].as_array().expect("labels");
    assert!(labels.len() >= 2, "a decision has at least two labels");
    for (i, r) in out["results"]
        .as_array()
        .expect("results")
        .iter()
        .enumerate()
    {
        let probs = f64s(&r["probabilities"]);
        assert_eq!(probs.len(), labels.len(), "result {i}: K probabilities");
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "result {i}: sums to {sum}");
        assert!(
            labels.contains(&r["label"]),
            "result {i}: label is one of the task's"
        );
    }
    println!(
        "E2E-DECIDE-PMCP-001 (real): {} identity {}",
        model.display(),
        out["model"]["artifact_sha256"]
    );
}
