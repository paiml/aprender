//! E2E-SETFIT-MCP-001 — the full SetFit loop over MCP, against a REAL artifact.
//!
//! Every other `apr.predict` test in this crate proves the tool against a mock
//! shim or a fixture that predict's routing refuses (deliberately: F-10
//! guarantees no fixture pipeline can produce a SetFit-tagged `.apr`, so none
//! is checked in). This test is the one place the whole stack meets a real
//! trained classifier: the live `apr mcp` stdio server, the real `apr predict
//! --input` subprocess, core's verified SetFit graph, real probabilities out.
//!
//! # Env-gated, not `#[ignore]`d
//!
//! Same pattern as `APR_MCP_E2E_MODEL` in the spec's M4 real-model gates: when
//! `APR_MCP_E2E_SETFIT_MODEL` is unset the test prints a visible SKIP line and
//! returns — a grep for `ok` is never mistaken for a run. Point the variable
//! at a `setfit-apr-v1` artifact to arm it. One reproduction recipe
//! (first proven 2026-08-18, TweetEval stance-abortion, 8 shots, seed 17):
//!
//! ```bash
//! apr data tweet-eval-stance --output data/tweet-eval-stance
//! apr data select --data data/tweet-eval-stance --shots 8 --seed 17
//! apr setfit train \
//!   --config crates/apr-cli/tests/fixtures/setfit/train-config.json \
//!   --data data/tweet-eval-stance \
//!   --selection data/tweet-eval-stance/selection-manifest.json \
//!   --model-dir ~/.cache/aprender/minilm-l6-v2-1110a243 \
//!   --output models/setfit-abortion-s17x8.apr
//! APR_MCP_E2E_SETFIT_MODEL=$PWD/models/setfit-abortion-s17x8.apr \
//!   cargo test -p aprender-mcp --test e2e_setfit_predict
//! ```
//!
//! Optionally set `APR_BIN` to a pinned `setfit`-featured binary (the `$APR`
//! that `scripts/apr_bin.sh` exports) to skip the in-test cargo build.
//!
//! # The binary must carry the `setfit` feature
//!
//! SetFit routing inside `apr predict` is feature-gated. A default-features
//! `apr` refuses the artifact at routing with "not a classifier" phrasing —
//! this test detects that refusal and fails with the remediation spelled out,
//! because "rebuild with --features setfit" must never be diagnosed as a
//! model defect. The spawned server is pinned to the same binary via
//! `APR_BIN`, so the server and the predict subprocess cannot diverge.

#![allow(clippy::disallowed_methods)] // serde_json::json! expands to .unwrap() internally

mod common;

use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

/// Absolute path to a trained `setfit-apr-v1` artifact. Unset ⇒ visible SKIP.
const ENV_MODEL: &str = "APR_MCP_E2E_SETFIT_MODEL";

/// initialize is pure protocol — instant.
const INIT_TIMEOUT: Duration = Duration::from_secs(10);

/// tools/call loads an ~87 MB encoder and runs a real forward pass on CPU.
const CALL_TIMEOUT: Duration = Duration::from_secs(180);

/// Locate a `setfit`-featured `apr`: `APR_BIN` if the operator pinned one,
/// else build it with `--features setfit` and resolve via `cargo_bin`.
///
/// Same shape as `falsify_mcp_dogfood_001::build_apr_binary`, with two
/// deliberate differences. The feature flag is added because prediction
/// routing requires it (a mock never did). And there is NO trust-an-existing-
/// binary early return: `cargo_bin` would happily hand back a stale
/// default-features debug build whose routing refusal reads like a model
/// defect — the shadowed-artifact failure mode this repo has been burned by.
/// The build is incremental, so when the featured binary already exists this
/// costs one no-op cargo invocation.
fn setfit_apr_binary() -> PathBuf {
    if let Some(pinned) = std::env::var_os("APR_BIN") {
        let pinned = PathBuf::from(pinned);
        assert!(
            pinned.is_file(),
            "APR_BIN points at {}, which is not a file",
            pinned.display()
        );
        return pinned;
    }
    // The shared builder, not `assert_cmd::cargo::cargo_bin`: that guesses the
    // path from the profile directory, and `.cargo/config.toml` redirects the
    // target dir in this repo, so the guess names a file that is not there.
    // `apr_binary_with_features` takes the path cargo REPORTS for the build it
    // just ran, and `setfit` is required — a default-features `apr predict`
    // refuses to route a `setfit-apr-v1` artifact.
    common::apr_binary_with_features("setfit")
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

#[test]
fn a_trained_setfit_artifact_classifies_over_the_live_stdio_server() {
    let Some(model) = std::env::var_os(ENV_MODEL) else {
        println!(
            "E2E-SETFIT-MCP-001 SKIP: {ENV_MODEL} unset — train a setfit-apr-v1 \
             artifact (recipe in this file's header) and export the var to arm this gate"
        );
        return;
    };
    let model = PathBuf::from(model);
    assert!(
        model.is_file(),
        "{ENV_MODEL} points at {}, which is not a file",
        model.display()
    );
    let model_arg = model
        .to_str()
        .expect("model path must be UTF-8")
        .to_string();

    let bin_path = setfit_apr_binary();
    let mut child = Command::new(&bin_path)
        .arg("mcp")
        // Pin the predict subprocess to the very binary serving MCP, so a
        // stale PATH `apr` can never answer for the one under test.
        .env("APR_BIN", &bin_path)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("spawn `apr mcp`");
    let mut stdin = child.stdin.take().expect("piped stdin");
    let stdout = child.stdout.take().expect("piped stdout");

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

    send(
        &mut stdin,
        &request(
            1,
            "initialize",
            serde_json::json!({ "protocolVersion": "2024-11-05" }),
        ),
    );
    let init = recv_json(&rx, INIT_TIMEOUT, "initialize response");
    assert_eq!(init["id"], 1);
    assert!(init.get("error").is_none(), "initialize failed: {init:?}");

    // Two texts of clearly different stance-y register; the assertion below is
    // structural (this test must hold for ANY setfit-apr-v1 the env names),
    // but two distinct inputs prove per-row results are per-row.
    let texts = serde_json::json!([
        "Every woman deserves the right to make her own healthcare decisions.",
        "The weather in Lisbon was lovely this weekend."
    ]);
    send(
        &mut stdin,
        &request(
            2,
            "tools/call",
            serde_json::json!({
                "name": "apr.predict",
                "arguments": { "model_path": model_arg, "texts": texts }
            }),
        ),
    );
    let call = recv_json(&rx, CALL_TIMEOUT, "apr.predict tools/call response");
    assert_eq!(call["id"], 2);
    assert!(
        call.get("error").is_none(),
        "tools/call must not be a protocol error: {call:?}"
    );
    let result = &call["result"];
    let text = result["content"][0]["text"]
        .as_str()
        .expect("content[0].text must be a string");
    if result["isError"].as_bool() == Some(true) {
        assert!(
            !text.contains("not a classifier") && !text.contains("model_type"),
            "predict refused ROUTING — the apr binary under test was built without \
             `--features setfit`. Rebuild it (`cargo build --bin apr --features setfit`) \
             and re-run; this is a build-configuration defect, not a model defect.\n{text}"
        );
        panic!("apr.predict returned isError=true against a real artifact:\n{text}");
    }

    // The tool forwards `apr predict --json` stdout verbatim — parse and hold
    // it to the CLI's own response contract.
    let payload: serde_json::Value =
        serde_json::from_str(text).expect("content[0].text must be the CLI's JSON");
    let sha = payload["artifact_sha256"]
        .as_str()
        .expect("artifact_sha256 must be present");
    assert_eq!(sha.len(), 64, "artifact_sha256 must be a sha256 hex digest");
    assert!(
        sha.chars().all(|c| c.is_ascii_hexdigit()),
        "artifact_sha256 must be hex, got {sha}"
    );
    let results = payload["results"].as_array().expect("results array");
    assert_eq!(results.len(), 2, "one result per input text, in order");
    for (i, row) in results.iter().enumerate() {
        let label = row["label"].as_str().unwrap_or_default();
        assert!(!label.is_empty(), "results[{i}].label must be non-empty");
        let probs = row["probabilities"]
            .as_array()
            .unwrap_or_else(|| panic!("results[{i}].probabilities must be an array: {row}"));
        assert!(
            probs.len() >= 2,
            "a classifier has at least two classes, results[{i}] has {}",
            probs.len()
        );
        let sum: f64 = probs.iter().filter_map(serde_json::Value::as_f64).sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "results[{i}].probabilities must sum to 1.0, got {sum}"
        );
    }

    // EOF on stdin is the server's documented shutdown signal.
    drop(stdin);
    let status = child.wait().expect("server must exit after stdin EOF");
    assert!(status.success(), "apr mcp exited non-zero: {status:?}");
    reader.join().expect("stdout reader thread panicked");
}
