// Integration tests: unwrap()/expect()/panic!() are idiomatic; strict workspace lints relaxed here.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::disallowed_methods
)]
//! FALSIFY-4089: `apr run` and `apr serve` take the SAME default backend.
//!
//! #4089 measured it on yoga (RTX 4060, a `--features cuda` build, Qwen3.5-4B): with NO backend
//! flag, `apr run` printed "Backend: GPU" while `apr serve run` resolved `backend=cpu` and its
//! `/v1/completions` answered `"used_gpu": false`. A user switching verbs silently lost the GPU.
//!
//! This asserts the equality through each verb's PROVENANCE, never by intent: `apr run --json`
//! reports `used_gpu`, and the served completion reports `used_gpu`. Both verbs start with no
//! backend flag, on the same binary and the same model file.
//!
//! It needs a real model and a pinned binary, so it runs when both are named:
//!   APR_BIN=<path to a cuda apr> APR_FALSIFY_MODEL=<a .gguf> cargo test -p apr-cli \
//!       --test falsify_serve_run_default_backend_4089 -- --nocapture
//! Without them it prints a SKIP line naming what is missing. That is loud, not a silent pass.
//! On a host where CUDA is present, the two verbs must also BOTH report `used_gpu: true`: equal
//! but both CPU on a GPU host would be the bug the other way round.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

struct KillOnDrop(Child);
impl Drop for KillOnDrop {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

fn used_gpu_of(json: &serde_json::Value) -> Option<bool> {
    // Both surfaces carry `used_gpu`; run's may nest it under a result object.
    json.get("used_gpu")
        .and_then(serde_json::Value::as_bool)
        .or_else(|| {
            json.pointer("/result/used_gpu")
                .and_then(serde_json::Value::as_bool)
        })
}

/// `apr run --json` provenance: (`used_gpu`, `backend.fell_back`).
fn run_provenance(bin: &str, model: &str) -> (bool, Option<bool>) {
    let out = Command::new(bin)
        .args([
            "run",
            model,
            "--prompt",
            "Hi",
            "--max-tokens",
            "4",
            "--json",
        ])
        .output()
        .expect("spawn apr run");
    let stdout = String::from_utf8_lossy(&out.stdout);
    // The object is pretty-printed over many lines: take it from its first `{`.
    let json = stdout.find('{').map(|i| &stdout[i..]).unwrap_or_else(|| {
        panic!(
            "apr run --json printed no JSON (rc {:?}):\n{stdout}",
            out.status.code()
        )
    });
    let v: serde_json::Value = serde_json::from_str(json.trim_end())
        .unwrap_or_else(|e| panic!("apr run --json does not parse ({e}):\n{stdout}"));
    let used = used_gpu_of(&v).unwrap_or_else(|| panic!("apr run --json carries no used_gpu: {v}"));
    (
        used,
        v.pointer("/backend/fell_back")
            .and_then(serde_json::Value::as_bool),
    )
}

fn http_post(port: u16, path: &str, body: &str) -> Option<String> {
    let mut s = TcpStream::connect(("127.0.0.1", port)).ok()?;
    s.set_read_timeout(Some(Duration::from_secs(120))).ok()?;
    write!(
        s,
        "POST {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )
    .ok()?;
    let mut resp = String::new();
    s.read_to_string(&mut resp).ok()?;
    resp.split("\r\n\r\n").nth(1).map(str::to_string)
}

fn serve_used_gpu(bin: &str, model: &str) -> bool {
    let port = free_port();
    let child = Command::new(bin)
        .args(["serve", "run", model, "--port", &port.to_string()])
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("spawn apr serve run");
    let mut guard = KillOnDrop(child);
    // Echo the server's startup log so the `gpu-layers: requested=...` line is in the receipt.
    let stdout = guard.0.stdout.take().unwrap();
    std::thread::spawn(move || {
        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
            println!("[serve] {line}");
        }
    });
    let completion = r#"{"model":"default","prompt":"Hi","max_tokens":4,"temperature":0}"#;
    let chat = r#"{"model":"default","messages":[{"role":"user","content":"Hi"}],"max_tokens":4,"temperature":0}"#;
    let deadline = Instant::now() + Duration::from_secs(300);
    let first = loop {
        if let Some(status) = guard.0.try_wait().expect("poll apr serve") {
            panic!("apr serve exited ({status}) before answering; its log is above");
        }
        if let Some(resp) = http_post(port, "/v1/completions", completion) {
            break resp;
        }
        assert!(
            Instant::now() < deadline,
            "apr serve never answered /v1/completions within 300 s"
        );
        std::thread::sleep(Duration::from_millis(500));
    };
    // Read provenance where the server reports it: /v1/completions, else the chat route.
    for (path, resp) in [
        ("/v1/completions", Some(first)),
        (
            "/v1/chat/completions",
            http_post(port, "/v1/chat/completions", chat),
        ),
    ] {
        let resp = resp.unwrap_or_else(|| panic!("{path}: no response"));
        let v: serde_json::Value = serde_json::from_str(&resp)
            .unwrap_or_else(|e| panic!("{path} is not JSON ({e}): {resp}"));
        println!("[serve] {path} used_gpu={:?}", used_gpu_of(&v));
        if let Some(used) = used_gpu_of(&v) {
            return used;
        }
    }
    panic!("apr serve reports used_gpu on neither /v1/completions nor /v1/chat/completions: no provenance to assert");
}

#[test]
fn falsify_4089_serve_and_run_take_the_same_default_backend() {
    let (Ok(bin), Ok(model)) = (std::env::var("APR_BIN"), std::env::var("APR_FALSIFY_MODEL"))
    else {
        eprintln!("SKIP FALSIFY-4089: set APR_BIN and APR_FALSIFY_MODEL to run it; this check did NOT run");
        return;
    };
    let (run, run_fell_back) = run_provenance(&bin, &model);
    let serve = serve_used_gpu(&bin, &model);
    println!(
        "FALSIFY-4089: apr run used_gpu={run} fell_back={run_fell_back:?}  apr serve used_gpu={serve}  \
         (no backend flag, {model})"
    );
    assert_eq!(
        run, serve,
        "#4089: with no backend flag, `apr run` used_gpu={run} but `apr serve` used_gpu={serve} \
         on the same binary and model"
    );
    if std::env::var("APR_EXPECT_GPU").as_deref() == Ok("1") {
        assert!(
            run && serve,
            "APR_EXPECT_GPU=1: a CUDA host must put BOTH verbs on the GPU by default"
        );
        assert_eq!(
            run_fell_back,
            Some(false),
            "APR_EXPECT_GPU=1: `apr run` must not have fallen back"
        );
    }
}
