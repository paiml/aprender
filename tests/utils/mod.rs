// Sprint 34: Integration Test Utilities
//
// Helper functions for testing with actual Jaeger backend

#![allow(dead_code)] // Test utility functions may not all be used in every test module

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::thread;
use std::time::Duration;

/// Jaeger API trace response
#[derive(Debug, Deserialize, Serialize)]
pub struct JaegerTracesResponse {
    pub data: Vec<JaegerTrace>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct JaegerTrace {
    #[serde(rename = "traceID")]
    pub trace_id: String,
    pub spans: Vec<JaegerSpan>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct JaegerSpan {
    #[serde(rename = "traceID")]
    pub trace_id: String,
    #[serde(rename = "spanID")]
    pub span_id: String,
    #[serde(rename = "operationName")]
    pub operation_name: String,
    pub references: Vec<JaegerReference>,
    pub tags: Vec<JaegerTag>,
    #[serde(rename = "startTime")]
    pub start_time: u64,
    pub duration: u64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct JaegerReference {
    #[serde(rename = "refType")]
    pub ref_type: String,
    #[serde(rename = "traceID")]
    pub trace_id: String,
    #[serde(rename = "spanID")]
    pub span_id: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct JaegerTag {
    pub key: String,
    #[serde(rename = "type")]
    pub tag_type: String,
    pub value: serde_json::Value,
}

/// Wait for Jaeger to be ready (up to 30 seconds)
pub fn wait_for_jaeger_ready(endpoint: &str) -> Result<()> {
    for i in 0..30 {
        match reqwest::blocking::get(format!("{}/api/services", endpoint)) {
            Ok(response) if response.status().is_success() => {
                eprintln!("[test-utils] Jaeger ready after {} seconds", i);
                return Ok(());
            }
            _ => {
                if i == 0 {
                    eprintln!("[test-utils] Waiting for Jaeger at {}...", endpoint);
                }
                thread::sleep(Duration::from_secs(1));
            }
        }
    }
    Err(anyhow!("Jaeger not ready after 30 seconds"))
}

/// Query Jaeger API for traces by service name
pub fn query_jaeger_traces(
    jaeger_url: &str,
    service: &str,
    trace_id: Option<&str>,
) -> Result<Vec<JaegerTrace>> {
    let mut url = format!("{}/api/traces?service={}&limit=100", jaeger_url, service);

    if let Some(tid) = trace_id {
        url = format!("{}&traceID={}", url, tid);
    }

    eprintln!("[test-utils] Querying Jaeger: {}", url);

    let response =
        reqwest::blocking::get(&url).map_err(|e| anyhow!("Failed to query Jaeger: {}", e))?;

    if !response.status().is_success() {
        return Err(anyhow!("Jaeger query failed: {}", response.status()));
    }

    let traces_response: JaegerTracesResponse =
        response.json().map_err(|e| anyhow!("Failed to parse Jaeger response: {}", e))?;

    Ok(traces_response.data)
}

/// Wait for trace to appear in Jaeger (with retries)
pub fn wait_for_trace(jaeger_url: &str, service: &str, timeout_secs: u64) -> Result<JaegerTrace> {
    for i in 0..timeout_secs {
        let traces = query_jaeger_traces(jaeger_url, service, None)?;
        if !traces.is_empty() {
            eprintln!("[test-utils] Found trace after {} seconds: {}", i, traces[0].trace_id);
            return Ok(traces[0].clone());
        }
        thread::sleep(Duration::from_secs(1));
    }
    Err(anyhow!("No traces found for service '{}' after {} seconds", service, timeout_secs))
}

/// Verify span exists in trace with expected attributes
pub fn verify_span_exists<'a>(
    trace: &'a JaegerTrace,
    span_name: &str,
    expected_attributes: &HashMap<String, String>,
) -> Result<&'a JaegerSpan> {
    for span in &trace.spans {
        if span.operation_name == span_name {
            // Verify all expected attributes
            for (key, expected_value) in expected_attributes {
                let tag = span
                    .tags
                    .iter()
                    .find(|t| t.key == *key)
                    .ok_or_else(|| anyhow!("Missing attribute: {}", key))?;

                let actual_value = tag.value.to_string().trim_matches('"').to_string();

                if actual_value != *expected_value {
                    return Err(anyhow!(
                        "Attribute mismatch for '{}': expected '{}', got '{}'",
                        key,
                        expected_value,
                        actual_value
                    ));
                }
            }

            eprintln!("[test-utils] ✓ Span '{}' verified", span_name);
            return Ok(span);
        }
    }

    Err(anyhow!("Span '{}' not found in trace", span_name))
}

/// Verify parent-child relationship between two spans
pub fn verify_parent_child(
    trace: &JaegerTrace,
    parent_span_name: &str,
    child_span_name: &str,
) -> Result<()> {
    // Find parent span
    let parent_span = trace
        .spans
        .iter()
        .find(|s| s.operation_name == parent_span_name)
        .ok_or_else(|| anyhow!("Parent span '{}' not found", parent_span_name))?;

    // Find child span
    let child_span = trace
        .spans
        .iter()
        .find(|s| s.operation_name == child_span_name)
        .ok_or_else(|| anyhow!("Child span '{}' not found", child_span_name))?;

    // Verify child has reference to parent
    let has_parent_ref = child_span
        .references
        .iter()
        .any(|r| r.ref_type == "CHILD_OF" && r.span_id == parent_span.span_id);

    if !has_parent_ref {
        return Err(anyhow!("Span '{}' is not a child of '{}'", child_span_name, parent_span_name));
    }

    eprintln!(
        "[test-utils] ✓ Parent-child relationship verified: {} → {}",
        parent_span_name, child_span_name
    );

    Ok(())
}

/// Extract trace-id from Renacer stderr output
pub fn extract_trace_id_from_stderr(_stderr: &str) -> Option<String> {
    // Look for OTLP initialization message or trace context
    // Example patterns:
    // - "[renacer: OTLP export enabled to ...]"
    // - "[renacer: Distributed tracing enabled...]"

    // For now, we'll query Jaeger for the latest trace
    // In practice, we'd parse stderr or use a deterministic trace-id
    None
}

/// Count spans matching a predicate
pub fn count_spans<F>(trace: &JaegerTrace, predicate: F) -> usize
where
    F: Fn(&JaegerSpan) -> bool,
{
    trace.spans.iter().filter(|s| predicate(s)).count()
}

/// Get span attribute value
pub fn get_span_attribute(span: &JaegerSpan, key: &str) -> Option<String> {
    span.tags
        .iter()
        .find(|t| t.key == key)
        .map(|t| t.value.to_string().trim_matches('"').to_string())
}

/// Start Jaeger container (if not already running)
pub fn ensure_jaeger_running() -> Result<()> {
    // Check if Jaeger is already running
    if reqwest::blocking::get("http://localhost:16686/api/services").is_ok() {
        eprintln!("[test-utils] Jaeger already running");
        return Ok(());
    }

    eprintln!("[test-utils] Starting Jaeger container...");

    // Start Jaeger via docker-compose
    std::process::Command::new("docker-compose")
        .args(["-f", "docker-compose-test.yml", "up", "-d", "jaeger"])
        .output()
        .map_err(|e| anyhow!("Failed to start Jaeger: {}", e))?;

    // Wait for Jaeger to be ready
    wait_for_jaeger_ready("http://localhost:16686")?;

    Ok(())
}

/// Stop Jaeger container
pub fn stop_jaeger() -> Result<()> {
    eprintln!("[test-utils] Stopping Jaeger container...");

    std::process::Command::new("docker-compose")
        .args(["-f", "docker-compose-test.yml", "down"])
        .output()
        .map_err(|e| anyhow!("Failed to stop Jaeger: {}", e))?;

    Ok(())
}

/// Clear Jaeger data (restart container)
pub fn clear_jaeger_data() -> Result<()> {
    eprintln!("[test-utils] Clearing Jaeger data...");

    stop_jaeger()?;
    thread::sleep(Duration::from_secs(2));
    ensure_jaeger_running()?;

    Ok(())
}
