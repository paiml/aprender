//! Example: Network Interception (Feature 7)
//!
//! Demonstrates: Intercepting and mocking HTTP requests
//!
//! Run with: `cargo run --example network_intercept`
//!
//! Toyota Way: Poka-Yoke (Mistake-Proofing) - Type-safe request handling
//!
//! PMAT-006: Added abort functionality and wait_for_response

use jugar_probar::prelude::*;
use std::collections::HashMap;

fn main() -> ProbarResult<()> {
    println!("=== Network Interception Example ===\n");

    // 1. Create network interception
    println!("1. Creating network interceptor...");
    let mut interceptor = NetworkInterceptionBuilder::new().build();

    println!("   Interceptor created");

    // 2. Add URL patterns
    println!("\n2. Creating URL patterns...");
    let exact_pattern = UrlPattern::Exact("https://api.example.com/users".into());
    let glob_pattern = UrlPattern::Glob("https://api.example.com/*".into());
    let regex_pattern = UrlPattern::Regex(r"https://.*\.example\.com/.*".into());

    println!("   Exact: {:?}", exact_pattern);
    println!("   Glob: {:?}", glob_pattern);
    println!("   Regex: {:?}", regex_pattern);

    // 3. Create mock response
    println!("\n3. Creating mock response...");
    let mock_response = MockResponse::new()
        .with_status(200)
        .with_header("Content-Type", "application/json")
        .with_body(br#"{"id": 1, "name": "Test User"}"#.to_vec());

    println!("   Status: {}", mock_response.status);
    println!("   Headers: {:?}", mock_response.headers);
    println!("   Body length: {} bytes", mock_response.body.len());

    // 4. Add route with mock
    println!("\n4. Adding route with mock response...");
    let route = Route::new(exact_pattern, HttpMethod::Get, mock_response);

    interceptor.route(route);
    println!("   Route added");
    println!("   Route count: {}", interceptor.route_count());

    // 5. Simulate captured requests
    println!("\n5. Simulating captured requests...");
    let mut request = CapturedRequest::new("https://api.example.com/users", HttpMethod::Get, 0);
    let _ = request
        .headers
        .insert("Accept".to_string(), "application/json".to_string());

    println!("   URL: {}", request.url);
    println!("   Method: {:?}", request.method);
    println!("   Headers: {:?}", request.headers);

    // 6. HTTP methods
    println!("\n6. HTTP methods...");
    let methods = [
        HttpMethod::Get,
        HttpMethod::Post,
        HttpMethod::Put,
        HttpMethod::Delete,
        HttpMethod::Patch,
        HttpMethod::Head,
        HttpMethod::Options,
    ];

    for method in &methods {
        println!("   {:?}", method);
    }

    // 7. Response status codes
    println!("\n7. Common response status codes...");
    let statuses = [
        (200, "OK"),
        (201, "Created"),
        (400, "Bad Request"),
        (401, "Unauthorized"),
        (404, "Not Found"),
        (500, "Internal Server Error"),
    ];

    for (code, text) in &statuses {
        println!("   {} {}", code, text);
    }

    // 8. PMAT-006: Request Abort functionality
    println!("\n8. Request Abort (PMAT-006)...");
    demo_request_abort()?;

    // 9. PMAT-006: Wait for request/response
    println!("\n9. Wait for Request/Response (PMAT-006)...");
    demo_wait_for_request()?;

    // 10. Clear routes
    println!("\n10. Clearing routes...");
    interceptor.clear_routes();
    println!("    Routes cleared. Count: {}", interceptor.route_count());

    println!("\n✅ Network interception example completed!");
    Ok(())
}

/// PMAT-006: Demonstrate request abort functionality
fn demo_request_abort() -> ProbarResult<()> {
    println!("   --- Request Abort Demo ---");

    // Create interceptor
    let mut interceptor = NetworkInterception::new();

    // Abort all requests to tracking endpoints
    println!("   Blocking tracking endpoints...");
    interceptor.abort("/analytics", AbortReason::BlockedByClient);
    interceptor.abort("/tracking", AbortReason::BlockedByClient);
    interceptor.abort("/ads", AbortReason::BlockedByClient);

    // Abort with different reasons
    println!("   Simulating network failures...");
    interceptor.abort_pattern(
        UrlPattern::Contains("unreachable.com".to_string()),
        AbortReason::ConnectionFailed,
    );
    interceptor.abort_pattern(
        UrlPattern::Contains("timeout.com".to_string()),
        AbortReason::TimedOut,
    );

    // Start interception
    interceptor.start();

    // Test blocked request
    let response = interceptor.handle_request(
        "https://example.com/analytics/event",
        HttpMethod::Post,
        HashMap::new(),
        None,
    );

    if let Some(resp) = response {
        println!("   Blocked request response:");
        println!("     Status: {} (0 = aborted)", resp.status);
        println!("     Body: {}", resp.body_string());
    }

    // Show all abort reasons
    println!("\n   Available AbortReason variants:");
    let reasons = [
        AbortReason::Failed,
        AbortReason::Aborted,
        AbortReason::TimedOut,
        AbortReason::AccessDenied,
        AbortReason::ConnectionClosed,
        AbortReason::ConnectionFailed,
        AbortReason::ConnectionRefused,
        AbortReason::ConnectionReset,
        AbortReason::InternetDisconnected,
        AbortReason::NameNotResolved,
        AbortReason::BlockedByClient,
    ];

    for reason in &reasons {
        println!("     {:?}: {}", reason, reason.message());
    }

    Ok(())
}

/// PMAT-006: Demonstrate wait for request/response
fn demo_wait_for_request() -> ProbarResult<()> {
    println!("   --- Wait for Request Demo ---");

    // Create interceptor with capture_all
    let mut interceptor = NetworkInterception::new().capture_all();
    interceptor.get(
        "/api/users",
        MockResponse::json(&serde_json::json!({
            "users": [{"id": 1, "name": "Alice"}]
        }))?,
    );
    interceptor.start();

    // Simulate some requests
    interceptor.handle_request(
        "https://example.com/api/users",
        HttpMethod::Get,
        HashMap::new(),
        None,
    );

    interceptor.handle_request(
        "https://example.com/api/posts",
        HttpMethod::Get,
        HashMap::new(),
        None,
    );

    // Find request matching pattern
    let pattern = UrlPattern::Contains("users".to_string());
    let found = interceptor.find_request(&pattern);

    println!("   find_request(Contains(\"users\")):");
    if let Some(req) = found {
        println!("     Found: {}", req.url);
        println!("     Method: {:?}", req.method);
    } else {
        println!("     Not found");
    }

    // Find response for pattern
    let response = interceptor.find_response_for(&pattern);
    println!("\n   find_response_for(Contains(\"users\")):");
    if let Some(resp) = response {
        println!("     Status: {}", resp.status);
        println!("     Body: {}", resp.body_string());
    } else {
        println!("     Not found");
    }

    // Get all captured responses
    let responses = interceptor.captured_responses();
    println!("\n   captured_responses():");
    println!("     Count: {}", responses.len());
    for (i, resp) in responses.iter().enumerate() {
        println!("     [{}] Status: {}", i, resp.status);
    }

    Ok(())
}
