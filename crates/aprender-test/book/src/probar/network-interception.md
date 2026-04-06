# Network Interception

> **Toyota Way**: Poka-Yoke (Mistake-Proofing) - Type-safe request handling

Intercept and mock HTTP requests for isolated testing.

## Running the Example

```bash
cargo run --example network_intercept
```

## Basic Usage

```rust
use probar::prelude::*;

// Create network interceptor
let mut interceptor = NetworkInterceptionBuilder::new()
    .capture_all()          // Capture all requests
    .block_unmatched()      // Block unmatched requests
    .build();

// Add mock routes
interceptor.get("/api/users", MockResponse::json(&serde_json::json!({
    "users": [{"id": 1, "name": "Alice"}]
}))?);

interceptor.post("/api/users", MockResponse::new().with_status(201));

// Start interception
interceptor.start();
```

## URL Patterns

```rust
use probar::network::UrlPattern;

// Exact match
let exact = UrlPattern::Exact("https://api.example.com/users".into());

// Prefix match
let prefix = UrlPattern::Prefix("https://api.example.com/".into());

// Contains substring
let contains = UrlPattern::Contains("/api/".into());

// Glob pattern
let glob = UrlPattern::Glob("https://api.example.com/*".into());

// Regex pattern
let regex = UrlPattern::Regex(r"https://.*\.example\.com/.*".into());

// Match any
let any = UrlPattern::Any;
```

## Mock Responses

```rust
use probar::network::MockResponse;

// Simple text response
let text = MockResponse::text("Hello, World!");

// JSON response
let json = MockResponse::json(&serde_json::json!({
    "status": "success",
    "data": {"id": 123}
}))?;

// Error response
let error = MockResponse::error(404, "Not Found");

// Custom response with builder
let custom = MockResponse::new()
    .with_status(200)
    .with_header("Content-Type", "application/json")
    .with_header("X-Custom", "value")
    .with_body(br#"{"key": "value"}"#.to_vec())
    .with_delay(100);  // 100ms delay
```

## Request Abort (PMAT-006)

Block requests with specific error reasons (Playwright parity):

```rust
use probar::network::{NetworkInterception, AbortReason, UrlPattern};

let mut interceptor = NetworkInterception::new();

// Block tracking and ads
interceptor.abort("/analytics", AbortReason::BlockedByClient);
interceptor.abort("/tracking", AbortReason::BlockedByClient);
interceptor.abort("/ads", AbortReason::BlockedByClient);

// Simulate network failures
interceptor.abort_pattern(
    UrlPattern::Contains("unreachable.com".into()),
    AbortReason::ConnectionFailed,
);

interceptor.abort_pattern(
    UrlPattern::Contains("timeout.com".into()),
    AbortReason::TimedOut,
);

interceptor.start();
```

### Abort Reasons

| Reason | Error Code | Description |
|--------|------------|-------------|
| `Failed` | `net::ERR_FAILED` | Generic failure |
| `Aborted` | `net::ERR_ABORTED` | Request aborted |
| `TimedOut` | `net::ERR_TIMED_OUT` | Request timed out |
| `AccessDenied` | `net::ERR_ACCESS_DENIED` | Access denied |
| `ConnectionClosed` | `net::ERR_CONNECTION_CLOSED` | Connection closed |
| `ConnectionFailed` | `net::ERR_CONNECTION_FAILED` | Connection failed |
| `ConnectionRefused` | `net::ERR_CONNECTION_REFUSED` | Connection refused |
| `ConnectionReset` | `net::ERR_CONNECTION_RESET` | Connection reset |
| `InternetDisconnected` | `net::ERR_INTERNET_DISCONNECTED` | No internet |
| `NameNotResolved` | `net::ERR_NAME_NOT_RESOLVED` | DNS failure |
| `BlockedByClient` | `net::ERR_BLOCKED_BY_CLIENT` | Blocked by client |

## Wait for Request/Response (PMAT-006)

```rust
use probar::network::{NetworkInterception, UrlPattern};

let mut interceptor = NetworkInterception::new().capture_all();
interceptor.start();

// ... trigger some network activity ...

// Find captured request
let pattern = UrlPattern::Contains("api/users".into());
if let Some(request) = interceptor.find_request(&pattern) {
    println!("Found request: {}", request.url);
    println!("Method: {:?}", request.method);
}

// Find response for pattern
if let Some(response) = interceptor.find_response_for(&pattern) {
    println!("Status: {}", response.status);
    println!("Body: {}", response.body_string());
}

// Get all captured responses
let responses = interceptor.captured_responses();
println!("Total responses: {}", responses.len());
```

## Assertions

```rust
use probar::network::{NetworkInterception, UrlPattern};

let mut interceptor = NetworkInterception::new().capture_all();
interceptor.start();

// ... trigger network activity ...

// Assert request was made
interceptor.assert_requested(&UrlPattern::Contains("/api/users".into()))?;

// Assert request count
interceptor.assert_requested_times(&UrlPattern::Contains("/api/".into()), 3)?;

// Assert request was NOT made
interceptor.assert_not_requested(&UrlPattern::Contains("/admin".into()))?;
```

## Route Management

```rust
use probar::network::{NetworkInterception, Route, UrlPattern, HttpMethod, MockResponse};

let mut interceptor = NetworkInterception::new();

// Add route directly
let route = Route::new(
    UrlPattern::Contains("/api/users".into()),
    HttpMethod::Get,
    MockResponse::text("users data"),
).times(2);  // Only match twice

interceptor.route(route);

// Check route count
println!("Routes: {}", interceptor.route_count());

// Clear all routes
interceptor.clear_routes();

// Clear captured requests
interceptor.clear_captured();
```

## HTTP Methods

```rust
use probar::network::HttpMethod;

// Available methods
let get = HttpMethod::Get;
let post = HttpMethod::Post;
let put = HttpMethod::Put;
let delete = HttpMethod::Delete;
let patch = HttpMethod::Patch;
let head = HttpMethod::Head;
let options = HttpMethod::Options;
let any = HttpMethod::Any;  // Matches any method

// Parse from string
let method = HttpMethod::from_str("POST");

// Convert to string
let s = method.as_str();  // "POST"

// Check if methods match
assert!(HttpMethod::Any.matches(&HttpMethod::Get));
```

## Example: Testing API Calls

```rust
use probar::prelude::*;

fn test_user_api() -> ProbarResult<()> {
    let mut interceptor = NetworkInterceptionBuilder::new()
        .capture_all()
        .build();

    // Mock API responses
    interceptor.get("/api/users", MockResponse::json(&serde_json::json!({
        "users": [
            {"id": 1, "name": "Alice"},
            {"id": 2, "name": "Bob"}
        ]
    }))?);

    interceptor.post("/api/users", MockResponse::new()
        .with_status(201)
        .with_json(&serde_json::json!({"id": 3, "name": "Charlie"}))?);

    interceptor.delete("/api/users/1", MockResponse::new().with_status(204));

    // Block external tracking
    interceptor.abort("/analytics", AbortReason::BlockedByClient);

    interceptor.start();

    // ... run your tests ...

    // Verify API calls
    interceptor.assert_requested(&UrlPattern::Contains("/api/users".into()))?;

    Ok(())
}
```
