# Execution Tracing

> **Toyota Way**: Genchi Genbutsu (Go and See) - See actual execution flow

Generate comprehensive execution traces for debugging with detailed span tracking, event capture, and trace archives.

## Running the Example

```bash
cargo run --example execution_trace
```

## Quick Start

```rust
use probar::{ExecutionTracer, TracingConfig};

// Create a tracer
let tracer = ExecutionTracer::new("my_test");

// Start a span
let span_id = tracer.start_span("test_login");

// Do some work...

// End the span
tracer.end_span(&span_id);

// Get trace data
let events = tracer.events();
println!("Captured {} events", events.len());
```

## Tracing Configuration

```rust
use probar::TracingConfig;

// Default configuration
let config = TracingConfig::default();

// Custom configuration
let config = TracingConfig::new()
    .capture_all()  // Enable all capture options
    .with_max_events(50000);

// Minimal configuration
let minimal = TracingConfig::new()
    .capture_none()  // Disable all capture
    .with_max_events(1000);

// Check what's captured
println!("Screenshots: {}", config.capture_screenshots);
println!("Network: {}", config.capture_network);
println!("Console: {}", config.capture_console);
println!("Performance: {}", config.capture_performance);
println!("Max events: {}", config.max_events);
```

## Traced Spans

Spans represent named sections of execution:

```rust
use probar::{TracedSpan, SpanStatus};

// Create a span
let mut span = TracedSpan::new("login_flow", 0);

// Add attributes for context
span.add_attribute("user", "test@example.com");
span.add_attribute("method", "oauth2");

// Check span state
assert_eq!(span.status, SpanStatus::Running);

// Complete the span
span.end(150);  // End at 150ms
assert_eq!(span.duration_ms, Some(150));
assert_eq!(span.status, SpanStatus::Ok);

// Or mark as error
let mut error_span = TracedSpan::new("failed_request", 0);
error_span.mark_error("Connection timeout");
assert_eq!(error_span.status, SpanStatus::Error);
```

## Nested Spans

```rust
use probar::ExecutionTracer;

let tracer = ExecutionTracer::new("test");

// Parent span
let parent_id = tracer.start_span("test_checkout");

// Child spans
let cart_id = tracer.start_span_with_parent("load_cart", &parent_id);
tracer.end_span(&cart_id);

let payment_id = tracer.start_span_with_parent("process_payment", &parent_id);
tracer.end_span(&payment_id);

tracer.end_span(&parent_id);

// Spans form a tree structure for visualization
```

## Traced Events

```rust
use probar::{TracedEvent, EventCategory, EventLevel};

// Event categories
let categories = [
    EventCategory::Test,        // Test lifecycle events
    EventCategory::Assertion,   // Assertion results
    EventCategory::Interaction, // User interactions
    EventCategory::Network,     // Network requests
    EventCategory::Console,     // Console output
];

// Create events
let event = TracedEvent::new("button_click", EventCategory::Interaction)
    .with_level(EventLevel::Info)
    .with_data("selector", "#submit-btn")
    .with_data("coordinates", "100,200");

println!("Event: {} [{:?}]", event.name, event.category);
```

## Network Events

```rust
use probar::{NetworkEvent, HttpMethod};

// Capture network activity
let request = NetworkEvent::request(
    HttpMethod::Post,
    "https://api.example.com/login",
)
.with_header("Content-Type", "application/json")
.with_body(r#"{"username": "test"}"#);

let response = NetworkEvent::response(200)
    .with_header("Content-Type", "application/json")
    .with_body(r#"{"token": "xyz"}"#)
    .with_duration_ms(150);

println!("Request: {} {}", request.method, request.url);
println!("Response: {} ({}ms)", response.status, response.duration_ms);
```

## Console Messages

```rust
use probar::{ConsoleMessage, ConsoleLevel};

// Capture console output
let log = ConsoleMessage::new(ConsoleLevel::Log, "User logged in");
let warning = ConsoleMessage::new(ConsoleLevel::Warn, "Session expiring soon");
let error = ConsoleMessage::new(ConsoleLevel::Error, "Failed to save");

// With stack trace
let error_with_trace = ConsoleMessage::new(ConsoleLevel::Error, "Exception")
    .with_stack_trace("Error at line 42\n  at login.js:42");
```

## Execution Tracer

```rust
use probar::{ExecutionTracer, TracingConfig};

// Create tracer with custom config
let config = TracingConfig::default()
    .capture_all()
    .with_max_events(10000);

let mut tracer = ExecutionTracer::with_config("my_test", config);

// Record spans
let span_id = tracer.start_span("test_case");

// Record events
tracer.record_event("click", "button.submit");
tracer.record_network_start("GET", "/api/data");
tracer.record_console("log", "Loading data...");

tracer.end_span(&span_id);

// Get trace summary
let summary = tracer.summary();
println!("Spans: {}", summary.span_count);
println!("Events: {}", summary.event_count);
println!("Duration: {}ms", summary.total_duration_ms);
```

## Trace Archives

Save and load traces for later analysis:

```rust
use probar::{ExecutionTracer, TraceArchive};

// Create and populate tracer
let tracer = ExecutionTracer::new("test");
// ... run tests ...

// Save trace to file
tracer.save_to_file("traces/test_run.json")?;

// Load trace later
let archive = TraceArchive::load_from_file("traces/test_run.json")?;

println!("Test: {}", archive.metadata.test_name);
println!("Started: {}", archive.metadata.start_time);
println!("Spans: {}", archive.spans.len());
println!("Events: {}", archive.events.len());
```

## Trace Metadata

```rust
use probar::TraceMetadata;

// Metadata is automatically captured
let metadata = TraceMetadata::new("integration_test")
    .with_environment("ci")
    .with_version("1.0.0")
    .with_tag("smoke-test");

println!("Test: {}", metadata.test_name);
println!("Environment: {:?}", metadata.environment);
```

## Filtering Events

```rust
use probar::{ExecutionTracer, EventCategory};

let tracer = ExecutionTracer::new("test");
// ... record events ...

// Get events by category
let network_events = tracer.events_by_category(EventCategory::Network);
let console_events = tracer.events_by_category(EventCategory::Console);

// Get events in time range
let early_events = tracer.events_in_range(0, 1000);  // First second
```

## Span Status

```rust
use probar::SpanStatus;

// Span status values
let statuses = [
    SpanStatus::Running,   // Span in progress
    SpanStatus::Ok,        // Completed successfully
    SpanStatus::Error,     // Failed with error
    SpanStatus::Cancelled, // Cancelled before completion
];

// Check status
fn handle_span(status: SpanStatus) {
    match status {
        SpanStatus::Ok => println!("Success"),
        SpanStatus::Error => println!("Failed - check attributes"),
        SpanStatus::Running => println!("Still running"),
        SpanStatus::Cancelled => println!("Was cancelled"),
    }
}
```

## Integration with Test Framework

```rust
use probar::{ExecutionTracer, TestHarness, TestSuite};

fn run_traced_tests(suite: &TestSuite) {
    let tracer = ExecutionTracer::new(&suite.name);

    for test in &suite.tests {
        let span_id = tracer.start_span(&test.name);

        // Run test
        // let result = test.run();

        // Record result
        // if result.passed {
        //     tracer.record_event("pass", &test.name);
        // } else {
        //     tracer.record_event("fail", result.error);
        // }

        tracer.end_span(&span_id);
    }

    // Save trace for CI
    let _ = tracer.save_to_file("traces/test_run.json");
}
```

## Best Practices

1. **Meaningful Spans**: Name spans after logical operations, not implementation details
2. **Add Context**: Use attributes to capture relevant debugging information
3. **Limit Events**: Set appropriate `max_events` to avoid memory issues
4. **Archive Failures**: Save traces when tests fail for debugging
5. **Structured Data**: Use consistent attribute names across spans
6. **Parent-Child**: Use nested spans to show call hierarchy
