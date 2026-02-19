#![allow(clippy::disallowed_methods)]
//! QA Example: apr serve Falsification Suite (PMAT-QA-RUST-001)
//!
//! Popperian falsification tests for `apr serve` HTTP endpoints.
//! The most comprehensive QA example, testing OpenAI API compatibility.
//!
//! # Tests (35 Points)
//!
//! | ID | Test | Points | Criterion |
//! |----|------|--------|-----------|
//! | P017 | Health endpoint | 2 | `/health` returns 200 |
//! | P018 | Compute mode | 2 | Response contains cpu/gpu/cuda |
//! | P019 | Valid JSON | 2 | Response parses as JSON |
//! | P020 | OpenAI structure | 3 | `choices[0].message.content` exists |
//! | P021 | Non-empty content | 2 | Content length > 0 |
//! | P022 | No token artifacts | 2 | No raw tokens in output |
//! | P023 | No BPE artifacts | 2 | No Ġ/Ċ in output |
//! | P024 | SSE streaming | 3 | `data: {` prefix present |
//! | P025 | Stream termination | 2 | `[DONE]` marker present |
//! | P026 | Determinism T=0 | 3 | Same request → same response |
//! | P027 | Malformed JSON | 2 | Returns 400 on bad input |
//! | P028 | Coherency | 2 | Output is intelligible |
//! | P029 | No multi-turn loop | 3 | No fake Human:/Assistant: |
//! | P030 | Trace brick level | 1 | `brick_trace` in response |
//! | P031 | Trace step level | 1 | `step_trace` in response |
//! | P032 | Trace layer level | 1 | `layer_trace` in response |
//! | P033 | Default suppression | 2 | No trace fields without header |
//!
//! # Usage
//!
//! ```bash
//! cargo run --example qa_serve
//! cargo run --example qa_serve -- --model path/to/model.gguf --port 8080
//! cargo run --example qa_serve -- --all-models
//! ```

use std::env;
use std::io::{BufRead, BufReader};
use std::net::TcpStream;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

// Colors
const RED: &str = "\x1b[0;31m";
const GREEN: &str = "\x1b[0;32m";
const YELLOW: &str = "\x1b[0;33m";
const BLUE: &str = "\x1b[0;34m";
const CYAN: &str = "\x1b[0;36m";
const NC: &str = "\x1b[0m";

struct TestResult {
    id: &'static str,
    name: &'static str,
    passed: bool,
    details: Option<String>,
    points: u32,
}

impl TestResult {
    fn pass(id: &'static str, name: &'static str, points: u32) -> Self {
        Self {
            id,
            name,
            passed: true,
            details: None,
            points,
        }
    }
    fn pass_with_details(
        id: &'static str,
        name: &'static str,
        points: u32,
        details: String,
    ) -> Self {
        Self {
            id,
            name,
            passed: true,
            details: Some(details),
            points,
        }
    }
    fn fail(id: &'static str, name: &'static str, points: u32, details: String) -> Self {
        Self {
            id,
            name,
            passed: false,
            details: Some(details),
            points,
        }
    }
    fn skip(id: &'static str, name: &'static str, _points: u32, reason: String) -> Self {
        Self {
            id,
            name,
            passed: true,
            details: Some(format!("SKIP: {}", reason)),
            points: 0,
        }
    }
    fn print(&self) {
        let status = if self.passed {
            format!("{}[PASS]{}", GREEN, NC)
        } else {
            format!("{}[FAIL]{}", RED, NC)
        };
        println!("{} {}: {}", status, self.id, self.name);
        if let Some(ref d) = self.details {
            println!("       {}", d);
        }
    }
}

struct QaConfig {
    model_path: Option<PathBuf>,
    apr_binary: PathBuf,
    port: u16,
    #[allow(dead_code)]
    all_models: bool,
    verbose: bool,
}

impl Default for QaConfig {
    fn default() -> Self {
        Self {
            model_path: None,
            apr_binary: find_apr_binary(),
            port: 18080, // Use high port to avoid conflicts with common services
            all_models: false,
            verbose: false,
        }
    }
}

fn find_apr_binary() -> PathBuf {
    // Check custom target directory FIRST (common dev setup)
    for p in [
        "/mnt/nvme-raid0/targets/aprender/release/apr",
        "/mnt/nvme-raid0/targets/aprender/debug/apr",
        "target/release/apr",
        "target/debug/apr",
    ] {
        let path = PathBuf::from(p);
        if path.exists() {
            return path;
        }
    }
    PathBuf::from("cargo")
}

fn find_default_model() -> Option<PathBuf> {
    let home = env::var("HOME").unwrap_or_default();
    for p in [
        format!("{home}/.cache/pacha/models/d4c4d9763127153c.gguf"),
        format!("{home}/.cache/huggingface/models/qwen2.5-coder-0.5b-gguf/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf"),
        "models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf".to_string(),
    ] {
        if PathBuf::from(&p).exists() { return Some(PathBuf::from(p)); }
    }
    None
}

/// Check if something is already listening on the port
fn check_port_in_use(port: u16) -> bool {
    TcpStream::connect(format!("127.0.0.1:{}", port)).is_ok()
}

/// Verify server is OUR server (responds with valid APR health check)
fn verify_server_health(port: u16) -> bool {
    match http_get("127.0.0.1", port, "/health") {
        Ok((status, body)) => {
            // Must be 200 AND contain expected fields
            status == 200 && body.contains("status") && body.contains("healthy")
        }
        Err(_) => false,
    }
}

/// Start the apr serve process
fn start_server(config: &QaConfig, model: &PathBuf) -> Option<Child> {
    // Check if port is already in use BEFORE starting
    if check_port_in_use(config.port) {
        eprintln!(
            "{}ERROR: Port {} already in use. Cannot start server.{}",
            RED, config.port, NC
        );
        eprintln!(
            "{}Try: lsof -i :{} to find what's using it{}",
            YELLOW, config.port, NC
        );
        return None;
    }

    let port_str = config.port.to_string();
    let args = vec![
        "serve",
        model.to_str().unwrap_or(""),
        "--port",
        &port_str,
        "--gpu",
    ];

    let mut cmd = if config.apr_binary.to_string_lossy() == "cargo" {
        let mut c = Command::new("cargo");
        c.args([
            "run",
            "-p",
            "apr-cli",
            "--release",
            "--features",
            "inference",
            "--",
        ]);
        c.args(&args);
        c
    } else {
        let mut c = Command::new(&config.apr_binary);
        c.args(&args);
        c
    };

    cmd.stdout(Stdio::piped()).stderr(Stdio::piped());

    if config.verbose {
        eprintln!("{}DEBUG: Starting server {:?}{}", CYAN, cmd, NC);
    }

    match cmd.spawn() {
        Ok(child) => {
            // Wait for server to be ready - verify with health check, not just TCP
            for i in 0..60 {
                thread::sleep(Duration::from_secs(1));
                if verify_server_health(config.port) {
                    if config.verbose {
                        eprintln!("{}Server health check passed after {}s{}", GREEN, i + 1, NC);
                    }
                    return Some(child);
                }
                if config.verbose && i % 10 == 9 {
                    eprintln!("{}Waiting for server... {}s{}", YELLOW, i + 1, NC);
                }
            }
            eprintln!(
                "{}Server failed to respond to health check within 60s{}",
                RED, NC
            );
            None
        }
        Err(e) => {
            eprintln!("{}Failed to spawn server: {}{}", RED, e, NC);
            None
        }
    }
}

/// HTTP GET request (minimal, no external deps)
fn http_get(host: &str, port: u16, path: &str) -> Result<(u16, String), String> {
    let addr = format!("{}:{}", host, port);
    let mut stream = TcpStream::connect(&addr).map_err(|e| e.to_string())?;
    stream.set_read_timeout(Some(Duration::from_secs(30))).ok();

    use std::io::Write;
    let request = format!(
        "GET {} HTTP/1.1\r\nHost: {}:{}\r\nConnection: close\r\n\r\n",
        path, host, port
    );
    stream
        .write_all(request.as_bytes())
        .map_err(|e| e.to_string())?;

    let mut reader = BufReader::new(stream);
    let mut status_line = String::new();
    reader
        .read_line(&mut status_line)
        .map_err(|e| e.to_string())?;

    let status: u16 = status_line
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);

    // Skip headers
    loop {
        let mut line = String::new();
        reader.read_line(&mut line).ok();
        if line.trim().is_empty() {
            break;
        }
    }

    let mut body = String::new();
    for line in reader.lines().map_while(Result::ok) {
        body.push_str(&line);
        body.push('\n');
    }

    Ok((status, body))
}

/// HTTP POST request with JSON body
fn http_post(
    host: &str,
    port: u16,
    path: &str,
    body: &str,
    headers: &[(&str, &str)],
) -> Result<(u16, String), String> {
    let addr = format!("{}:{}", host, port);
    let mut stream = TcpStream::connect(&addr).map_err(|e| e.to_string())?;
    stream.set_read_timeout(Some(Duration::from_secs(60))).ok();

    use std::io::Write;
    let mut header_str = String::new();
    for (k, v) in headers {
        header_str.push_str(&format!("{}: {}\r\n", k, v));
    }

    let request = format!(
        "POST {} HTTP/1.1\r\nHost: {}:{}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n{}Connection: close\r\n\r\n{}",
        path, host, port, body.len(), header_str, body
    );
    stream
        .write_all(request.as_bytes())
        .map_err(|e| e.to_string())?;

    let mut reader = BufReader::new(stream);
    let mut status_line = String::new();
    reader
        .read_line(&mut status_line)
        .map_err(|e| e.to_string())?;

    let status: u16 = status_line
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);

    // Skip headers
    loop {
        let mut line = String::new();
        reader.read_line(&mut line).ok();
        if line.trim().is_empty() {
            break;
        }
    }

    let mut resp_body = String::new();
    for line in reader.lines().map_while(Result::ok) {
        resp_body.push_str(&line);
        resp_body.push('\n');
    }

    Ok((status, resp_body))
}

/// Extract JSON field (minimal parser)
fn extract_json_content(json: &str) -> Option<String> {
    // Look for "content": "..." pattern
    if let Some(start) = json.find("\"content\":") {
        let rest = &json[start + 10..];
        if let Some(quote_start) = rest.find('"') {
            let content_start = quote_start + 1;
            let mut end = content_start;
            let chars: Vec<char> = rest.chars().collect();
            while end < chars.len() {
                if chars[end] == '"' && (end == 0 || chars[end - 1] != '\\') {
                    break;
                }
                end += 1;
            }
            return Some(chars[content_start..end].iter().collect());
        }
    }
    None
}

// === TEST FUNCTIONS ===

fn test_health(config: &QaConfig) -> TestResult {
    match http_get("127.0.0.1", config.port, "/health") {
        Ok((status, _)) if status == 200 => TestResult::pass("P017", "Health Endpoint", 2),
        Ok((status, _)) => {
            TestResult::fail("P017", "Health Endpoint", 2, format!("Status {}", status))
        }
        Err(e) => TestResult::fail("P017", "Health Endpoint", 2, e),
    }
}

fn test_compute_mode(config: &QaConfig) -> TestResult {
    match http_get("127.0.0.1", config.port, "/health") {
        Ok((_, body)) => {
            if body.contains("cpu") || body.contains("gpu") || body.contains("cuda") {
                TestResult::pass_with_details("P018", "Compute Mode", 2, "Mode found".to_string())
            } else {
                TestResult::fail("P018", "Compute Mode", 2, "No mode in response".to_string())
            }
        }
        Err(e) => TestResult::fail("P018", "Compute Mode", 2, e),
    }
}

fn test_valid_json(config: &QaConfig) -> TestResult {
    let body = r#"{"model":"default","messages":[{"role":"user","content":"Hi"}],"max_tokens":5}"#;
    match http_post("127.0.0.1", config.port, "/v1/chat/completions", body, &[]) {
        Ok((200, resp)) => {
            if resp.contains('{') && resp.contains('}') && resp.contains("choices") {
                TestResult::pass("P019", "Valid JSON Response", 2)
            } else {
                TestResult::fail("P019", "Valid JSON Response", 2, "Invalid JSON".to_string())
            }
        }
        Ok((status, _)) => TestResult::fail(
            "P019",
            "Valid JSON Response",
            2,
            format!("Status {}", status),
        ),
        Err(e) => TestResult::fail("P019", "Valid JSON Response", 2, e),
    }
}

fn test_openai_structure(config: &QaConfig) -> TestResult {
    let body =
        r#"{"model":"default","messages":[{"role":"user","content":"Say hi"}],"max_tokens":10}"#;
    match http_post("127.0.0.1", config.port, "/v1/chat/completions", body, &[]) {
        Ok((200, resp)) => {
            if resp.contains("choices") && resp.contains("message") && resp.contains("content") {
                TestResult::pass("P020", "OpenAI Structure", 3)
            } else {
                TestResult::fail("P020", "OpenAI Structure", 3, "Missing fields".to_string())
            }
        }
        Ok((s, _)) => TestResult::fail("P020", "OpenAI Structure", 3, format!("Status {}", s)),
        Err(e) => TestResult::fail("P020", "OpenAI Structure", 3, e),
    }
}

fn test_non_empty_content(config: &QaConfig) -> TestResult {
    let body =
        r#"{"model":"default","messages":[{"role":"user","content":"Say hello"}],"max_tokens":10}"#;
    match http_post("127.0.0.1", config.port, "/v1/chat/completions", body, &[]) {
        Ok((200, resp)) => {
            if let Some(content) = extract_json_content(&resp) {
                if !content.is_empty() {
                    TestResult::pass_with_details(
                        "P021",
                        "Non-Empty Content",
                        2,
                        format!("Len: {}", content.len()),
                    )
                } else {
                    TestResult::fail("P021", "Non-Empty Content", 2, "Empty content".to_string())
                }
            } else {
                TestResult::fail(
                    "P021",
                    "Non-Empty Content",
                    2,
                    "No content field".to_string(),
                )
            }
        }
        Ok((s, _)) => TestResult::fail("P021", "Non-Empty Content", 2, format!("Status {}", s)),
        Err(e) => TestResult::fail("P021", "Non-Empty Content", 2, e),
    }
}


include!("includes/qa_serve_include_01.rs");
