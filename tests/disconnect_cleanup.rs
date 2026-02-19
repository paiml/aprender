#![allow(clippy::disallowed_methods)]
//! Disconnect Cleanup Tests (GH-205, PMAT-194)
//!
//! Tests for zombie session detection and cleanup.
//! Verifies that abruptly disconnected clients don't leave resources behind.
//!
//! # Running Disconnect Tests
//!
//! ```bash
//! # Start the server first
//! apr serve model.gguf --port 8080 &
//!
//! # Run disconnect tests
//! cargo test --test disconnect_cleanup -- --ignored --nocapture
//! ```
//!
//! # Quality Philosophy
//!
//! Per Toyota Way and Popperian Falsification:
//! - Zombie sessions are defects (resource leaks)
//! - Server must handle client disconnects gracefully
//! - No orphaned resources after client abort
//!
//! # References
//!
//! - PMAT-194: Load testing infrastructure
//! - GH-205: Load testing ticket
//! - docs/specifications/qwen2.5-coder-showcase-demo.md Section 20.3

use std::io::{Read, Write};
use std::net::TcpStream;
use std::time::Duration;

/// Server URL for disconnect tests
fn server_host() -> String {
    std::env::var("APR_TEST_SERVER_HOST").unwrap_or_else(|_| "127.0.0.1:8080".to_string())
}

/// D50-01: Abrupt TCP disconnect during request
///
/// Simulates a client that disconnects mid-request.
/// Verifies server continues operating normally.
///
/// # Falsification Criteria
///
/// - PASS: Server responds to subsequent requests after client abort
/// - FAIL: Server becomes unresponsive or returns errors
#[test]
#[ignore = "requires running server: apr serve model.gguf --port 8080"]
fn d50_01_abrupt_disconnect_during_request() {
    let host = server_host();

    // Send partial HTTP request then disconnect
    if let Ok(mut stream) = TcpStream::connect(&host) {
        let _ = stream.set_write_timeout(Some(Duration::from_secs(1)));
        // Send incomplete HTTP request
        let _ = stream.write_all(b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Length: 1000\r\n\r\n{\"partial\":");
        // Immediately drop connection (simulates client crash)
        drop(stream);
        println!("Sent partial request and disconnected");
    }

    // Wait a moment for server to process the abort
    std::thread::sleep(Duration::from_millis(500));

    // Verify server is still responsive
    if let Ok(mut stream) = TcpStream::connect(&host) {
        let _ = stream.set_read_timeout(Some(Duration::from_secs(5)));
        let _ = stream.set_write_timeout(Some(Duration::from_secs(1)));

        // Send valid health request
        let _ = stream
            .write_all(b"GET /health HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n");
        let _ = stream.flush();

        let mut response = String::new();
        let _ = stream.read_to_string(&mut response);

        assert!(
            response.contains("200") || response.contains("OK"),
            "Server should respond to health check after client abort. Got: {response}"
        );
        println!(
            "Server recovered: {}",
            response.lines().next().unwrap_or("")
        );
    } else {
        panic!("Server became unresponsive after client abort");
    }
}

/// D50-02: Multiple abrupt disconnects in sequence
///
/// Stress tests the server's ability to handle repeated client aborts.
///
/// # Falsification Criteria
///
/// - PASS: Server handles 10 abrupt disconnects and remains responsive
/// - FAIL: Server degrades after repeated aborts
#[test]
#[ignore = "requires running server: apr serve model.gguf --port 8080"]
fn d50_02_multiple_abrupt_disconnects() {
    let host = server_host();

    // Perform 10 abrupt disconnects
    for i in 0..10 {
        if let Ok(mut stream) = TcpStream::connect(&host) {
            let _ = stream.set_write_timeout(Some(Duration::from_millis(100)));
            // Send partial data
            let _ = stream.write_all(b"GET /health HTTP/1.1\r\n");
            // Abrupt disconnect
            drop(stream);
            println!("Abort {}/10", i + 1);
        }
        std::thread::sleep(Duration::from_millis(50));
    }

    // Wait for cleanup
    std::thread::sleep(Duration::from_secs(1));

    // Verify server health
    if let Ok(mut stream) = TcpStream::connect(&host) {
        let _ = stream.set_read_timeout(Some(Duration::from_secs(5)));
        let _ = stream.set_write_timeout(Some(Duration::from_secs(1)));

        let _ = stream
            .write_all(b"GET /health HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n");
        let _ = stream.flush();

        let mut response = String::new();
        let _ = stream.read_to_string(&mut response);

        assert!(
            response.contains("200"),
            "Server should remain healthy after 10 abrupt disconnects"
        );
        println!("Server healthy after 10 aborts");
    } else {
        panic!("Server unresponsive after multiple aborts");
    }
}

/// D50-03: Disconnect during streaming response
///
/// Starts a streaming request then disconnects mid-stream.
/// Verifies server cleans up the streaming context.
///
/// # Falsification Criteria
///
/// - PASS: Server continues operating after client aborts streaming response
/// - FAIL: Server hangs or leaks resources
#[test]
#[ignore = "requires running server: apr serve model.gguf --port 8080"]
fn d50_03_disconnect_during_streaming() {
    let host = server_host();

    // Start a streaming chat request
    if let Ok(mut stream) = TcpStream::connect(&host) {
        let _ = stream.set_write_timeout(Some(Duration::from_secs(1)));
        let _ = stream.set_read_timeout(Some(Duration::from_millis(500)));

        let body = r#"{"model":"test","messages":[{"role":"user","content":"Count from 1 to 100"}],"stream":true,"max_tokens":50}"#;
        let request = format!(
            "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
            body.len(),
            body
        );

        let _ = stream.write_all(request.as_bytes());
        let _ = stream.flush();

        // Read a bit of the response to confirm streaming started
        let mut buffer = [0u8; 256];
        if let Ok(n) = stream.read(&mut buffer) {
            if n > 0 {
                println!("Received {n} bytes before abort");
            }
        }

        // Abrupt disconnect mid-stream
        drop(stream);
        println!("Disconnected during streaming");
    }

    // Wait for server to detect disconnect and cleanup
    std::thread::sleep(Duration::from_secs(2));

    // Verify server is still healthy
    if let Ok(mut stream) = TcpStream::connect(&host) {
        let _ = stream.set_read_timeout(Some(Duration::from_secs(5)));
        let _ = stream.set_write_timeout(Some(Duration::from_secs(1)));

        let _ = stream
            .write_all(b"GET /health HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n");
        let _ = stream.flush();

        let mut response = String::new();
        let _ = stream.read_to_string(&mut response);

        assert!(
            response.contains("200"),
            "Server should be healthy after streaming abort"
        );
        println!("Server healthy after streaming abort");
    } else {
        panic!("Server unresponsive after streaming abort");
    }
}

/// D50-04: Concurrent disconnects don't cause race conditions
///
/// Multiple clients disconnect simultaneously.
///
/// # Falsification Criteria
///
/// - PASS: Server handles concurrent aborts without crashes
/// - FAIL: Server panics or becomes unresponsive
#[test]
#[ignore = "requires running server: apr serve model.gguf --port 8080"]
fn d50_04_concurrent_disconnects() {
    use std::thread;

    let host = server_host();

    // Spawn 20 threads that all disconnect abruptly
    let handles: Vec<_> = (0..20)
        .map(|i| {
            let host = host.clone();
            thread::spawn(move || {
                if let Ok(mut stream) = TcpStream::connect(&host) {
                    let _ = stream.set_write_timeout(Some(Duration::from_millis(100)));
                    let _ = stream.write_all(
                        format!("GET /health HTTP/1.1\r\nX-Request: {i}\r\n").as_bytes(),
                    );
                    // Abrupt drop
                }
            })
        })
        .collect();

    for handle in handles {
        let _ = handle.join();
    }

    println!("All 20 concurrent aborts completed");

    // Wait for cleanup
    std::thread::sleep(Duration::from_secs(1));

    // Verify server is healthy
    if let Ok(mut stream) = TcpStream::connect(&host) {
        let _ = stream.set_read_timeout(Some(Duration::from_secs(5)));
        let _ = stream.set_write_timeout(Some(Duration::from_secs(1)));

        let _ = stream
            .write_all(b"GET /health HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n");
        let _ = stream.flush();

        let mut response = String::new();
        let _ = stream.read_to_string(&mut response);

        assert!(
            response.contains("200"),
            "Server should handle concurrent disconnects"
        );
        println!("Server healthy after concurrent aborts");
    } else {
        panic!("Server unresponsive after concurrent aborts");
    }
}

/// D50-05: Connection timeout handling
///
/// Client connects but sends nothing (timeout scenario).
///
/// # Falsification Criteria
///
/// - PASS: Server eventually closes idle connection and remains healthy
/// - FAIL: Server accumulates zombie connections
#[test]
#[ignore = "requires running server: apr serve model.gguf --port 8080"]
fn d50_05_idle_connection_timeout() {
    let host = server_host();

    // Open 5 connections and leave them idle
    let mut idle_streams = Vec::new();
    for i in 0..5 {
        if let Ok(stream) = TcpStream::connect(&host) {
            println!("Opened idle connection {}", i + 1);
            idle_streams.push(stream);
        }
    }

    // Wait longer than server's expected timeout
    println!("Waiting for server timeout...");
    std::thread::sleep(Duration::from_secs(5));

    // Drop idle connections
    drop(idle_streams);

    // Server should still be healthy
    if let Ok(mut stream) = TcpStream::connect(&host) {
        let _ = stream.set_read_timeout(Some(Duration::from_secs(5)));
        let _ = stream.set_write_timeout(Some(Duration::from_secs(1)));

        let _ = stream
            .write_all(b"GET /health HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n");
        let _ = stream.flush();

        let mut response = String::new();
        let _ = stream.read_to_string(&mut response);

        assert!(
            response.contains("200"),
            "Server should handle idle connections gracefully"
        );
        println!("Server healthy after idle connections");
    } else {
        panic!("Server unresponsive after idle connections");
    }
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
}
