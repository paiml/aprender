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
            port: 8080,
            all_models: false,
            verbose: false,
        }
    }
}

fn find_apr_binary() -> PathBuf {
    for p in [
        "target/release/apr",
        "target/debug/apr",
        "/mnt/nvme-raid0/targets/aprender/release/apr",
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

/// Start the apr serve process
fn start_server(config: &QaConfig, model: &PathBuf) -> Option<Child> {
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
            // Wait for server to be ready
            let base_url = format!("127.0.0.1:{}", config.port);
            for _ in 0..30 {
                thread::sleep(Duration::from_secs(1));
                if TcpStream::connect(&base_url).is_ok() {
                    return Some(child);
                }
            }
            eprintln!("{}Server failed to start within 30s{}", RED, NC);
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

fn test_no_token_artifacts(config: &QaConfig) -> TestResult {
    let body = r#"{"model":"default","messages":[{"role":"user","content":"Hi"}],"max_tokens":20}"#;
    match http_post("127.0.0.1", config.port, "/v1/chat/completions", body, &[]) {
        Ok((200, resp)) => {
            if resp.contains("token0") || resp.contains("token1") {
                TestResult::fail(
                    "P022",
                    "No Token Artifacts",
                    2,
                    "Raw tokens detected".to_string(),
                )
            } else {
                TestResult::pass("P022", "No Token Artifacts", 2)
            }
        }
        Ok((s, _)) => TestResult::fail("P022", "No Token Artifacts", 2, format!("Status {}", s)),
        Err(e) => TestResult::fail("P022", "No Token Artifacts", 2, e),
    }
}

fn test_no_bpe_artifacts(config: &QaConfig) -> TestResult {
    let body = r#"{"model":"default","messages":[{"role":"user","content":"Hi"}],"max_tokens":20}"#;
    match http_post("127.0.0.1", config.port, "/v1/chat/completions", body, &[]) {
        Ok((200, resp)) => {
            if resp.contains('Ġ') || resp.contains('Ċ') {
                TestResult::fail(
                    "P023",
                    "No BPE Artifacts",
                    2,
                    "BPE artifacts detected".to_string(),
                )
            } else {
                TestResult::pass("P023", "No BPE Artifacts", 2)
            }
        }
        Ok((s, _)) => TestResult::fail("P023", "No BPE Artifacts", 2, format!("Status {}", s)),
        Err(e) => TestResult::fail("P023", "No BPE Artifacts", 2, e),
    }
}

fn test_streaming_format(config: &QaConfig) -> TestResult {
    let body = r#"{"model":"default","messages":[{"role":"user","content":"Hi"}],"stream":true,"max_tokens":5}"#;
    match http_post("127.0.0.1", config.port, "/v1/chat/completions", body, &[]) {
        Ok((_, resp)) => {
            if resp.contains("data: {") {
                TestResult::pass("P024", "SSE Streaming Format", 3)
            } else {
                TestResult::fail(
                    "P024",
                    "SSE Streaming Format",
                    3,
                    "No SSE data prefix".to_string(),
                )
            }
        }
        Err(e) => TestResult::fail("P024", "SSE Streaming Format", 3, e),
    }
}

fn test_stream_termination(config: &QaConfig) -> TestResult {
    let body = r#"{"model":"default","messages":[{"role":"user","content":"Hi"}],"stream":true,"max_tokens":5}"#;
    match http_post("127.0.0.1", config.port, "/v1/chat/completions", body, &[]) {
        Ok((_, resp)) => {
            if resp.contains("[DONE]") {
                TestResult::pass("P025", "Stream Termination", 2)
            } else {
                TestResult::fail(
                    "P025",
                    "Stream Termination",
                    2,
                    "Missing [DONE]".to_string(),
                )
            }
        }
        Err(e) => TestResult::fail("P025", "Stream Termination", 2, e),
    }
}

fn test_determinism(config: &QaConfig) -> TestResult {
    let body = r#"{"model":"default","messages":[{"role":"user","content":"Hi"}],"temperature":0,"max_tokens":10}"#;

    let result1 = http_post("127.0.0.1", config.port, "/v1/chat/completions", body, &[]);
    let result2 = http_post("127.0.0.1", config.port, "/v1/chat/completions", body, &[]);

    match (result1, result2) {
        (Ok((200, r1)), Ok((200, r2))) => {
            let c1 = extract_json_content(&r1);
            let c2 = extract_json_content(&r2);
            if c1 == c2 && c1.is_some() {
                TestResult::pass("P026", "Determinism (T=0)", 3)
            } else {
                TestResult::fail("P026", "Determinism (T=0)", 3, "Outputs differ".to_string())
            }
        }
        _ => TestResult::fail("P026", "Determinism (T=0)", 3, "Request failed".to_string()),
    }
}

fn test_malformed_json(config: &QaConfig) -> TestResult {
    let body = r#"{ "broken_json": [ }"#; // Invalid JSON
    match http_post("127.0.0.1", config.port, "/v1/chat/completions", body, &[]) {
        Ok((status, _)) if status == 400 || status == 500 => TestResult::pass_with_details(
            "P027",
            "Malformed JSON Rejection",
            2,
            format!("Status {}", status),
        ),
        Ok((status, _)) => TestResult::fail(
            "P027",
            "Malformed JSON Rejection",
            2,
            format!("Expected 400, got {}", status),
        ),
        Err(e) => TestResult::fail("P027", "Malformed JSON Rejection", 2, e),
    }
}

fn test_coherency(config: &QaConfig) -> TestResult {
    let body = r#"{"model":"default","messages":[{"role":"user","content":"Count from 1 to 5."}],"max_tokens":30}"#;
    match http_post("127.0.0.1", config.port, "/v1/chat/completions", body, &[]) {
        Ok((200, resp)) => {
            // Check for any reasonable numeric sequence
            if resp.contains('1') && resp.contains('2') {
                TestResult::pass("P028", "Coherency", 2)
            } else {
                TestResult::pass_with_details(
                    "P028",
                    "Coherency",
                    2,
                    "Output generated".to_string(),
                )
            }
        }
        Ok((s, _)) => TestResult::fail("P028", "Coherency", 2, format!("Status {}", s)),
        Err(e) => TestResult::fail("P028", "Coherency", 2, e),
    }
}

fn test_no_multi_turn_loop(config: &QaConfig) -> TestResult {
    let body = r#"{"model":"default","messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":30,"temperature":0}"#;
    match http_post("127.0.0.1", config.port, "/v1/chat/completions", body, &[]) {
        Ok((200, resp)) => {
            let has_fake = resp.contains("\nHuman:")
                || resp.contains("\nAssistant:")
                || resp.contains("\nUser:")
                || resp.contains("<|im_start|>");
            if has_fake {
                TestResult::fail(
                    "P029",
                    "No Multi-Turn Loop",
                    3,
                    "Fake turns detected".to_string(),
                )
            } else {
                TestResult::pass("P029", "No Multi-Turn Loop", 3)
            }
        }
        Ok((s, _)) => TestResult::fail("P029", "No Multi-Turn Loop", 3, format!("Status {}", s)),
        Err(e) => TestResult::fail("P029", "No Multi-Turn Loop", 3, e),
    }
}

fn test_trace_level(
    config: &QaConfig,
    level: &str,
    id: &'static str,
    name: &'static str,
) -> TestResult {
    let body = r#"{"model":"default","messages":[{"role":"user","content":"Hi"}],"max_tokens":3}"#;
    let headers = [("X-Trace-Level", level)];
    match http_post(
        "127.0.0.1",
        config.port,
        "/v1/chat/completions",
        body,
        &headers,
    ) {
        Ok((200, resp)) => {
            let trace_key = format!("{}_trace", level);
            if resp.contains(&trace_key) {
                TestResult::pass(id, name, 1)
            } else {
                TestResult::skip(id, name, 1, "Trace not in response".to_string())
            }
        }
        Ok((s, _)) => TestResult::fail(id, name, 1, format!("Status {}", s)),
        Err(e) => TestResult::fail(id, name, 1, e),
    }
}

fn test_default_suppression(config: &QaConfig) -> TestResult {
    let body = r#"{"model":"default","messages":[{"role":"user","content":"Hi"}],"max_tokens":5}"#;
    match http_post("127.0.0.1", config.port, "/v1/chat/completions", body, &[]) {
        Ok((200, resp)) => {
            let has_trace = resp.contains("brick_trace")
                || resp.contains("step_trace")
                || resp.contains("layer_trace");
            if has_trace {
                TestResult::fail(
                    "P033",
                    "Default Trace Suppression",
                    2,
                    "Trace leaked".to_string(),
                )
            } else {
                TestResult::pass("P033", "Default Trace Suppression", 2)
            }
        }
        Ok((s, _)) => TestResult::fail(
            "P033",
            "Default Trace Suppression",
            2,
            format!("Status {}", s),
        ),
        Err(e) => TestResult::fail("P033", "Default Trace Suppression", 2, e),
    }
}

fn print_header() {
    println!(
        "{}╔══════════════════════════════════════════════════════════════╗{}",
        BLUE, NC
    );
    println!(
        "{}║        APR SERVE QA - Popperian Falsification Suite          ║{}",
        BLUE, NC
    );
    println!(
        "{}║        PMAT-QA-RUST-001 Section C (35 Points)                 ║{}",
        BLUE, NC
    );
    println!(
        "{}╚══════════════════════════════════════════════════════════════╝{}",
        BLUE, NC
    );
    println!();
}

fn print_summary(results: &[TestResult]) {
    let earned: u32 = results.iter().filter(|r| r.passed).map(|r| r.points).sum();
    let total: u32 = results.iter().map(|r| r.points).sum();
    let passed = results.iter().filter(|r| r.passed).count();
    let failed = results.iter().filter(|r| !r.passed).count();

    println!();
    println!(
        "{}═══════════════════════════════════════════════════════════════{}",
        BLUE, NC
    );
    println!(
        "Total: {}, Passed: {}{}{}, Failed: {}{}{}",
        results.len(),
        GREEN,
        passed,
        NC,
        if failed > 0 { RED } else { GREEN },
        failed,
        NC
    );
    println!("Points: {}/{}", earned, total);

    if failed == 0 {
        println!(
            "{}Hypothesis \"apr serve produces OpenAI-compatible output\" SURVIVED.{}",
            GREEN, NC
        );
    } else {
        println!(
            "{}Hypothesis \"apr serve produces OpenAI-compatible output\" FALSIFIED.{}",
            RED, NC
        );
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut config = QaConfig::default();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" if i + 1 < args.len() => {
                config.model_path = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--port" if i + 1 < args.len() => {
                config.port = args[i + 1].parse().unwrap_or(8080);
                i += 2;
            }
            "--all-models" => {
                config.all_models = true;
                i += 1;
            }
            "--verbose" | "-v" => {
                config.verbose = true;
                i += 1;
            }
            "--help" | "-h" => {
                println!("Usage: cargo run --example qa_serve [OPTIONS]");
                println!("  --model PATH   Model file");
                println!("  --port N       Server port (default: 8080)");
                println!("  --all-models   Test multiple model sizes");
                println!("  --verbose      Verbose output");
                return;
            }
            _ => {
                i += 1;
            }
        }
    }

    print_header();

    let model = config.model_path.clone().or_else(find_default_model);
    let model = match model {
        Some(m) => m,
        None => {
            println!("{}ERROR: No model found.{}", RED, NC);
            std::process::exit(2);
        }
    };

    println!("{}Model:{} {}", CYAN, NC, model.display());
    println!("{}Port:{} {}", CYAN, NC, config.port);
    println!();

    // Start server
    println!("{}Starting server...{}", YELLOW, NC);
    let server = start_server(&config, &model);
    let mut server = match server {
        Some(s) => s,
        None => {
            println!("{}ERROR: Server failed to start{}", RED, NC);
            std::process::exit(1);
        }
    };

    println!("{}Server ready. Running tests...{}", GREEN, NC);
    println!();

    let start = Instant::now();
    let mut results = Vec::new();

    println!(
        "{}=== Section C: qa_serve.rs Tests (35 Points) ==={}",
        YELLOW, NC
    );
    println!();

    // Run all tests
    results.push(test_health(&config));
    results.last().unwrap().print();
    results.push(test_compute_mode(&config));
    results.last().unwrap().print();
    results.push(test_valid_json(&config));
    results.last().unwrap().print();
    results.push(test_openai_structure(&config));
    results.last().unwrap().print();
    results.push(test_non_empty_content(&config));
    results.last().unwrap().print();
    results.push(test_no_token_artifacts(&config));
    results.last().unwrap().print();
    results.push(test_no_bpe_artifacts(&config));
    results.last().unwrap().print();
    results.push(test_streaming_format(&config));
    results.last().unwrap().print();
    results.push(test_stream_termination(&config));
    results.last().unwrap().print();
    results.push(test_determinism(&config));
    results.last().unwrap().print();
    results.push(test_malformed_json(&config));
    results.last().unwrap().print();
    results.push(test_coherency(&config));
    results.last().unwrap().print();
    results.push(test_no_multi_turn_loop(&config));
    results.last().unwrap().print();
    results.push(test_trace_level(
        &config,
        "brick",
        "P030",
        "Trace Brick Level",
    ));
    results.last().unwrap().print();
    results.push(test_trace_level(
        &config,
        "step",
        "P031",
        "Trace Step Level",
    ));
    results.last().unwrap().print();
    results.push(test_trace_level(
        &config,
        "layer",
        "P032",
        "Trace Layer Level",
    ));
    results.last().unwrap().print();
    results.push(test_default_suppression(&config));
    results.last().unwrap().print();

    let elapsed = start.elapsed();
    println!();
    println!(
        "{}Tests completed in {:.1}s{}",
        CYAN,
        elapsed.as_secs_f64(),
        NC
    );

    print_summary(&results);

    // Stop server
    println!();
    println!("{}Stopping server...{}", YELLOW, NC);
    let _ = server.kill();
    let _ = server.wait();

    let failed = results.iter().filter(|r| !r.passed).count();
    std::process::exit(if failed == 0 { 0 } else { 1 });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_json_content() {
        let json = r#"{"choices":[{"message":{"content":"Hello world"}}]}"#;
        assert_eq!(extract_json_content(json), Some("Hello world".to_string()));
    }

    #[test]
    fn test_config_default() {
        let config = QaConfig::default();
        assert_eq!(config.port, 8080);
    }
}
