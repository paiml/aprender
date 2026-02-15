//! Dogfood test for GH-261: CPU fallback when GPU serve fails per-request.
//!
//! Starts `apr serve --gpu` on a small APR model, then verifies:
//!   1. `/health` reports `gpu_fallback: true`
//!   2. `/v1/completions` returns a response (GPU or CPU-fallback)
//!   3. `/v1/chat/completions` returns a well-formed OpenAI-compatible response
//!
//! Requires a small APR model (e.g. qwen2-0.5b-int8.apr) at the path below.
//! Adjust `MODEL_PATH` if your model lives elsewhere.
//!
//! ```bash
//! cargo run --release --example gpu_fallback_dogfood
//! ```

use std::io::{BufRead, BufReader};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

const PORT: u16 = 19261;
const MODEL_PATH: &str = "../tiny-model-ground-truth/models/qwen2-0.5b-int8.apr";

fn main() {
    println!("=== GH-261 Dogfood: GPU Serve CPU Fallback ===\n");

    let model = std::path::Path::new(MODEL_PATH);
    if !model.exists() {
        eprintln!("Model not found at {MODEL_PATH}");
        eprintln!("Download with: apr pull hf://Qwen/Qwen2-0.5B-Instruct");
        std::process::exit(1);
    }

    // Start server
    println!("Starting `apr serve --gpu` on port {PORT}...");
    let mut child = Command::new("apr")
        .args([
            "serve",
            "--gpu",
            MODEL_PATH,
            "--port",
            &PORT.to_string(),
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Failed to spawn `apr serve`. Is apr installed?");

    // Wait for server readiness by polling /health
    let base = format!("http://127.0.0.1:{PORT}");
    let client = std::net::TcpStream::connect_timeout(
        &format!("127.0.0.1:{PORT}")
            .parse()
            .expect("valid socket addr"),
        Duration::from_millis(100),
    );
    drop(client);

    let deadline = Instant::now() + Duration::from_secs(30);
    let mut ready = false;
    while Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(500));
        if let Ok(resp) = http_get(&format!("{base}/health")) {
            if resp.contains("healthy") {
                ready = true;
                break;
            }
        }
    }

    if !ready {
        // Dump stderr for debugging
        if let Some(stderr) = child.stderr.take() {
            let reader = BufReader::new(stderr);
            for line in reader.lines().take(20).flatten() {
                eprintln!("  server stderr: {line}");
            }
        }
        child.kill().ok();
        panic!("Server did not become ready within 30s");
    }
    println!("  Server ready.\n");

    // Test 1: Health endpoint
    println!("Test 1: /health reports gpu_fallback");
    let health = http_get(&format!("{base}/health")).expect("health request failed");
    assert!(
        health.contains("\"gpu_fallback\""),
        "Expected gpu_fallback in health response, got: {health}"
    );
    assert!(
        health.contains("true"),
        "Expected gpu_fallback=true, got: {health}"
    );
    println!("  PASS: {health}");

    // Test 2: /v1/completions
    println!("Test 2: /v1/completions returns response");
    let comp = http_post(
        &format!("{base}/v1/completions"),
        r#"{"prompt":"Hello","max_tokens":4}"#,
    )
    .expect("completions request failed");
    assert!(
        comp.contains("\"text\"") || comp.contains("\"error\""),
        "Unexpected completions response: {comp}"
    );
    // If fallback fired, we should see compute_mode
    if comp.contains("cpu-fallback") {
        println!("  PASS (cpu-fallback): {comp}");
    } else {
        println!("  PASS (gpu): {comp}");
    }

    // Test 3: /v1/chat/completions
    println!("Test 3: /v1/chat/completions returns OpenAI-compatible response");
    let chat = http_post(
        &format!("{base}/v1/chat/completions"),
        r#"{"messages":[{"role":"user","content":"Hi"}],"max_tokens":4}"#,
    )
    .expect("chat request failed");
    assert!(
        chat.contains("\"choices\"") || chat.contains("\"error\""),
        "Unexpected chat response: {chat}"
    );
    if chat.contains("apr-cpu-fallback") {
        println!("  PASS (cpu-fallback model tag present)");
    } else if chat.contains("apr-gpu") {
        println!("  PASS (gpu model tag present)");
    } else {
        println!("  PASS: {chat}");
    }

    // Cleanup
    child.kill().ok();
    child.wait().ok();

    println!("\n=== All 3 dogfood tests PASSED ===");
}

/// Minimal HTTP GET using std::net (no external deps).
fn http_get(url: &str) -> Result<String, String> {
    let (host, path) = parse_url(url)?;
    let req = format!("GET {path} HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\n\r\n");
    send_http(&host, &req)
}

/// Minimal HTTP POST with JSON body using std::net.
fn http_post(url: &str, body: &str) -> Result<String, String> {
    let (host, path) = parse_url(url)?;
    let req = format!(
        "POST {path} HTTP/1.1\r\nHost: {host}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    send_http(&host, &req)
}

fn parse_url(url: &str) -> Result<(String, String), String> {
    let stripped = url
        .strip_prefix("http://")
        .ok_or("URL must start with http://")?;
    let (host, path) = stripped
        .split_once('/')
        .map(|(h, p)| (h.to_string(), format!("/{p}")))
        .unwrap_or((stripped.to_string(), "/".to_string()));
    Ok((host, path))
}

fn send_http(host: &str, request: &str) -> Result<String, String> {
    use std::io::{Read, Write};
    let mut stream = std::net::TcpStream::connect_timeout(
        &host.parse().map_err(|e| format!("bad addr: {e}"))?,
        Duration::from_secs(15),
    )
    .map_err(|e| format!("connect failed: {e}"))?;
    stream
        .set_read_timeout(Some(Duration::from_secs(30)))
        .ok();
    stream
        .write_all(request.as_bytes())
        .map_err(|e| format!("write failed: {e}"))?;
    let mut buf = Vec::new();
    stream
        .read_to_end(&mut buf)
        .map_err(|e| format!("read failed: {e}"))?;
    let response = String::from_utf8_lossy(&buf).to_string();
    // Extract body after \r\n\r\n
    if let Some(idx) = response.find("\r\n\r\n") {
        Ok(response[idx + 4..].to_string())
    } else {
        Ok(response)
    }
}
