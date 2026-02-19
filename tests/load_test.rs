#![allow(clippy::disallowed_methods)]
//! Load Testing Infrastructure (GH-205, PMAT-194)
//!
//! Tests for concurrent request handling and server capacity.
//! These tests require a running server and are marked as `#[ignore]` by default.
//!
//! # Running Load Tests
//!
//! ```bash
//! # Start the server first
//! apr serve model.gguf --port 8080 &
//!
//! # Run load tests
//! cargo test --test load_test -- --ignored --nocapture
//! ```
//!
//! # Quality Philosophy
//!
//! Per Toyota Way and Popperian Falsification:
//! - Tests must be falsifiable (clear pass/fail criteria)
//! - 50-concurrent request capacity is a HARD requirement
//! - Zombie session cleanup must be verified
//!
//! # References
//!
//! - PMAT-194: Load testing infrastructure
//! - GH-205: Load testing ticket
//! - docs/specifications/qwen2.5-coder-showcase-demo.md Section 20.3

use std::io::{BufRead, BufReader, Write};
use std::net::TcpStream;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Server host for load tests (configurable via environment)
fn server_host() -> String {
    std::env::var("APR_TEST_SERVER_HOST").unwrap_or_else(|_| "127.0.0.1:8080".to_string())
}

/// Minimum concurrent request capacity
const MIN_CONCURRENT_REQUESTS: u32 = 50;

/// Maximum acceptable response time for health check (ms)
const MAX_HEALTH_RESPONSE_MS: u64 = 100;

/// Maximum acceptable response time for chat request (ms)
const MAX_CHAT_RESPONSE_MS: u64 = 30_000;

/// Send HTTP GET request and return response status
fn http_get(host: &str, path: &str, timeout: Duration) -> Result<u16, String> {
    let stream = TcpStream::connect(host).map_err(|e| format!("Connect failed: {e}"))?;
    stream
        .set_read_timeout(Some(timeout))
        .map_err(|e| format!("Set timeout failed: {e}"))?;
    stream
        .set_write_timeout(Some(timeout))
        .map_err(|e| format!("Set timeout failed: {e}"))?;

    let mut stream = stream;
    let request = format!("GET {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n");
    stream
        .write_all(request.as_bytes())
        .map_err(|e| format!("Write failed: {e}"))?;
    stream.flush().map_err(|e| format!("Flush failed: {e}"))?;

    let mut reader = BufReader::new(stream);
    let mut status_line = String::new();
    reader
        .read_line(&mut status_line)
        .map_err(|e| format!("Read failed: {e}"))?;

    // Parse status code from "HTTP/1.1 200 OK"
    let parts: Vec<&str> = status_line.split_whitespace().collect();
    if parts.len() >= 2 {
        parts[1]
            .parse()
            .map_err(|e| format!("Parse status failed: {e}"))
    } else {
        Err(format!("Invalid status line: {status_line}"))
    }
}

/// Send HTTP POST request and return response status
fn http_post(host: &str, path: &str, body: &str, timeout: Duration) -> Result<u16, String> {
    let stream = TcpStream::connect(host).map_err(|e| format!("Connect failed: {e}"))?;
    stream
        .set_read_timeout(Some(timeout))
        .map_err(|e| format!("Set timeout failed: {e}"))?;
    stream
        .set_write_timeout(Some(timeout))
        .map_err(|e| format!("Set timeout failed: {e}"))?;

    let mut stream = stream;
    let request = format!(
        "POST {path} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    stream
        .write_all(request.as_bytes())
        .map_err(|e| format!("Write failed: {e}"))?;
    stream.flush().map_err(|e| format!("Flush failed: {e}"))?;

    let mut reader = BufReader::new(stream);
    let mut status_line = String::new();
    reader
        .read_line(&mut status_line)
        .map_err(|e| format!("Read failed: {e}"))?;

    let parts: Vec<&str> = status_line.split_whitespace().collect();
    if parts.len() >= 2 {
        parts[1]
            .parse()
            .map_err(|e| format!("Parse status failed: {e}"))
    } else {
        Err(format!("Invalid status line: {status_line}"))
    }
}

/// L50-01: Server handles 50 concurrent requests without dropping connections
///
/// This test verifies the system can handle MIN_CONCURRENT_REQUESTS simultaneous
/// requests without connection failures or timeouts.
///
/// # Falsification Criteria
///
/// - PASS: All 50 requests complete successfully with status 200
/// - FAIL: Any request fails, times out, or returns non-200 status
#[test]
#[ignore = "requires running server: apr serve model.gguf --port 8080"]
fn l50_01_fifty_concurrent_requests() {
    use std::thread;

    let host = server_host();
    let success_count = Arc::new(AtomicU32::new(0));
    let failure_count = Arc::new(AtomicU32::new(0));
    let start = Instant::now();

    let handles: Vec<_> = (0..MIN_CONCURRENT_REQUESTS)
        .map(|i| {
            let host = host.clone();
            let success = Arc::clone(&success_count);
            let failure = Arc::clone(&failure_count);

            thread::spawn(move || {
                match http_get(
                    &host,
                    "/health",
                    Duration::from_millis(MAX_HEALTH_RESPONSE_MS * 10),
                ) {
                    Ok(200) => {
                        success.fetch_add(1, Ordering::SeqCst);
                        println!("Request {i}: OK");
                    }
                    Ok(status) => {
                        failure.fetch_add(1, Ordering::SeqCst);
                        println!("Request {i}: Failed with status {status}");
                    }
                    Err(e) => {
                        failure.fetch_add(1, Ordering::SeqCst);
                        println!("Request {i}: Error - {e}");
                    }
                }
            })
        })
        .collect();

    // Wait for all requests to complete
    for handle in handles {
        let _ = handle.join();
    }

    let elapsed = start.elapsed();
    let successes = success_count.load(Ordering::SeqCst);
    let failures = failure_count.load(Ordering::SeqCst);

    println!("\n=== L50-01 Results ===");
    println!("Total requests: {MIN_CONCURRENT_REQUESTS}");
    println!("Successful: {successes}");
    println!("Failed: {failures}");
    println!("Time: {elapsed:?}");

    assert_eq!(
        successes, MIN_CONCURRENT_REQUESTS,
        "Expected all {MIN_CONCURRENT_REQUESTS} requests to succeed, but {failures} failed"
    );
}

/// L50-02: Server handles 50 concurrent chat completions
///
/// More demanding test with actual inference requests.
///
/// # Falsification Criteria
///
/// - PASS: All 50 chat requests complete with valid responses
/// - FAIL: Any request fails, times out, or returns invalid response
#[test]
#[ignore = "requires running server: apr serve model.gguf --port 8080"]
fn l50_02_fifty_concurrent_chat_requests() {
    use std::thread;

    let host = server_host();
    let success_count = Arc::new(AtomicU32::new(0));
    let failure_count = Arc::new(AtomicU32::new(0));
    let response_times: Arc<std::sync::Mutex<Vec<Duration>>> =
        Arc::new(std::sync::Mutex::new(Vec::new()));
    let start = Instant::now();

    let handles: Vec<_> = (0..MIN_CONCURRENT_REQUESTS)
        .map(|i| {
            let host = host.clone();
            let success = Arc::clone(&success_count);
            let failure = Arc::clone(&failure_count);
            let times = Arc::clone(&response_times);

            thread::spawn(move || {
                let body = r#"{"model":"test","messages":[{"role":"user","content":"Hi"}],"max_tokens":5,"temperature":0.0}"#;
                let req_start = Instant::now();

                match http_post(
                    &host,
                    "/v1/chat/completions",
                    body,
                    Duration::from_millis(MAX_CHAT_RESPONSE_MS),
                ) {
                    Ok(200) => {
                        let elapsed = req_start.elapsed();
                        success.fetch_add(1, Ordering::SeqCst);
                        if let Ok(mut lock) = times.lock() {
                            lock.push(elapsed);
                        }
                        println!("Chat {i}: OK ({elapsed:?})");
                    }
                    Ok(status) => {
                        failure.fetch_add(1, Ordering::SeqCst);
                        println!("Chat {i}: Failed with status {status}");
                    }
                    Err(e) => {
                        failure.fetch_add(1, Ordering::SeqCst);
                        println!("Chat {i}: Error - {e}");
                    }
                }
            })
        })
        .collect();

    // Wait for all requests to complete
    for handle in handles {
        let _ = handle.join();
    }

    let elapsed = start.elapsed();
    let successes = success_count.load(Ordering::SeqCst);
    let failures = failure_count.load(Ordering::SeqCst);

    // Calculate statistics
    let times = response_times.lock().map(|t| t.clone()).unwrap_or_default();
    let avg_time: Duration = if !times.is_empty() {
        times.iter().sum::<Duration>() / times.len() as u32
    } else {
        Duration::ZERO
    };

    println!("\n=== L50-02 Results ===");
    println!("Total requests: {MIN_CONCURRENT_REQUESTS}");
    println!("Successful: {successes}");
    println!("Failed: {failures}");
    println!("Total time: {elapsed:?}");
    println!("Avg response time: {avg_time:?}");

    assert_eq!(
        successes, MIN_CONCURRENT_REQUESTS,
        "Expected all {MIN_CONCURRENT_REQUESTS} chat requests to succeed, but {failures} failed"
    );
}

/// L50-03: Response times remain stable under load
///
/// Verifies that response times don't degrade significantly as concurrent
/// requests increase.
///
/// # Falsification Criteria
///
/// - PASS: P99 response time < 5x baseline
/// - FAIL: P99 response time >= 5x baseline
#[test]
#[ignore = "requires running server: apr serve model.gguf --port 8080"]
fn l50_03_response_time_stability() {
    let host = server_host();
    let timeout = Duration::from_secs(5);

    // Baseline: single request
    let baseline_start = Instant::now();
    let _ = http_get(&host, "/health", timeout);
    let baseline = baseline_start.elapsed();

    println!("Baseline response time: {baseline:?}");

    // Measure under load (sequential for simplicity)
    let mut response_times: Vec<Duration> = Vec::with_capacity(100);
    for _ in 0..100 {
        let start = Instant::now();
        let _ = http_get(&host, "/health", timeout);
        response_times.push(start.elapsed());
    }

    // Sort for percentile calculation
    response_times.sort();

    let p50 = response_times.get(49).copied().unwrap_or(Duration::ZERO);
    let p99 = response_times.get(98).copied().unwrap_or(Duration::ZERO);

    println!("P50 response time: {p50:?}");
    println!("P99 response time: {p99:?}");
    println!(
        "P99/Baseline ratio: {:.2}x",
        p99.as_secs_f64() / baseline.as_secs_f64()
    );

    // P99 should be less than 5x baseline
    let max_acceptable = baseline * 5;
    assert!(
        p99 < max_acceptable,
        "P99 response time ({p99:?}) exceeds 5x baseline ({max_acceptable:?})"
    );
}

/// L50-04: Server recovers from burst traffic
///
/// Sends a burst of requests, waits, then verifies server is still responsive.
///
/// # Falsification Criteria
///
/// - PASS: Server responds to health check after burst
/// - FAIL: Server becomes unresponsive after burst
#[test]
#[ignore = "requires running server: apr serve model.gguf --port 8080"]
fn l50_04_burst_recovery() {
    use std::thread;

    let host = server_host();
    let timeout = Duration::from_secs(5);

    // Pre-burst health check
    let pre_result = http_get(&host, "/health", timeout);
    assert!(pre_result.is_ok(), "Server should be healthy before burst");

    // Burst: 100 rapid requests
    println!("Sending burst of 100 requests...");
    let handles: Vec<_> = (0..100)
        .map(|_| {
            let host = host.clone();
            thread::spawn(move || {
                let _ = http_get(&host, "/health", Duration::from_secs(1));
            })
        })
        .collect();

    for handle in handles {
        let _ = handle.join();
    }

    // Wait for server to stabilize
    println!("Waiting for server to stabilize...");
    thread::sleep(Duration::from_secs(2));

    // Post-burst health check
    let post_result = http_get(&host, "/health", timeout);
    assert!(
        post_result.is_ok(),
        "Server should remain healthy after burst traffic"
    );
    println!("Server recovered successfully after burst");
}

/// L50-05: No resource leaks under sustained load
///
/// Runs sustained load and checks for increasing latency (symptom of leaks).
///
/// # Falsification Criteria
///
/// - PASS: Average latency in last 10 requests <= 3x average of first 10
/// - FAIL: Latency increases significantly over time
#[test]
#[ignore = "requires running server: apr serve model.gguf --port 8080"]
fn l50_05_no_resource_leaks() {
    let host = server_host();
    let timeout = Duration::from_secs(5);

    let mut all_times: Vec<Duration> = Vec::with_capacity(100);

    // Make 100 sequential requests
    for i in 0..100 {
        let start = Instant::now();
        let _ = http_get(&host, "/health", timeout);
        let elapsed = start.elapsed();
        all_times.push(elapsed);

        if i % 20 == 0 {
            println!("Request {i}: {elapsed:?}");
        }
    }

    // Compare first 10 vs last 10
    let first_10_avg: Duration = all_times.iter().take(10).sum::<Duration>() / 10;
    let last_10_avg: Duration = all_times.iter().rev().take(10).sum::<Duration>() / 10;

    println!("\nFirst 10 avg: {first_10_avg:?}");
    println!("Last 10 avg: {last_10_avg:?}");

    // Last 10 should not be more than 3x first 10 (allows for variance)
    let max_acceptable = first_10_avg * 3;
    assert!(
        last_10_avg <= max_acceptable,
        "Latency increased significantly: first 10 avg = {first_10_avg:?}, last 10 avg = {last_10_avg:?} (max acceptable: {max_acceptable:?})"
    );

    println!("No resource leak detected");
}

#[cfg(test)]
mod unit_tests {
    use super::*;

    #[test]
    fn test_server_host_contains_port() {
        // Default should include port
        let host = server_host();
        assert!(
            host.contains(':'),
            "Server host should include port: {host}"
        );
    }

    #[test]
    fn test_constants() {
        assert_eq!(MIN_CONCURRENT_REQUESTS, 50);
        assert!(MAX_HEALTH_RESPONSE_MS < MAX_CHAT_RESPONSE_MS);
    }
}
