//! Example: Execution Tracing (Feature 9)
//!
//! Demonstrates: Comprehensive tracing of test execution
//!
//! Run with: `cargo run --example execution_trace`
//!
//! Toyota Way: Genchi Genbutsu (Go and See) - See actual execution flow

use jugar_probar::prelude::*;

fn main() -> ProbarResult<()> {
    println!("=== Execution Tracing Example ===\n");

    // 1. Create tracing config
    println!("1. Creating tracing configuration...");
    let config = TracingConfig::new();

    println!("   Config created with defaults");

    // 2. Create execution tracer
    println!("\n2. Creating execution tracer...");
    let mut tracer = ExecutionTracer::new("test_example", config);

    println!("   Tracer created for 'test_example'");

    // 3. Start tracing and create spans
    println!("\n3. Recording spans...");
    tracer.start();

    let span_id = tracer.start_span("login_flow");
    println!("   Started span: {}", span_id);

    // Record some events
    tracer.info("login", "User initiated login");
    tracer.warn("validation", "Password strength warning");

    tracer.end_span(&span_id);
    println!("   Ended span");

    // 4. Event categories
    println!("\n4. Event categories...");
    let categories = [
        "Test",
        "Assertion",
        "Interaction",
        "Network",
        "Console",
        "Screenshot",
    ];

    for category in &categories {
        println!("   {}", category);
    }

    // 5. Create traced event
    println!("\n5. Creating traced events...");
    let event = TracedEvent::new("button_click", EventCategory::Test, tracer.elapsed_ms())
        .with_message("User clicked submit button");

    println!("   Event: {} at {}ms", event.name, event.timestamp_ms);

    tracer.record_event(event);

    // 6. Console messages
    println!("\n6. Console message levels...");
    let level_names = ["Log", "Info", "Warn", "Error", "Debug"];

    for level in &level_names {
        println!("   {}", level);
    }

    // Record some console messages
    let msg = ConsoleMessage {
        level: ConsoleLevel::Info,
        text: "Test started".into(),
        timestamp_ms: tracer.elapsed_ms(),
        source: None,
        line: None,
    };
    tracer.record_console(msg);

    // 7. Network events
    println!("\n7. Recording network events...");
    let mut network_event =
        NetworkEvent::new("https://api.example.com/users", "GET", tracer.elapsed_ms());
    network_event.complete(200, 150);

    println!("   URL: {}", network_event.url);
    println!("   Method: {}", network_event.method);
    println!("   Status: {:?}", network_event.status);

    tracer.record_network(network_event);

    // 8. Trace metadata
    println!("\n8. Trace metadata...");
    let metadata = TraceMetadata::new("integration_test");

    println!("   Test name: {}", metadata.test_name);

    // 9. Stop and get archive
    println!("\n9. Stopping tracer and getting archive...");
    let archive = tracer.stop();

    println!("   Spans recorded: {}", archive.spans.len());
    println!("   Events recorded: {}", archive.events.len());
    println!("   Network events: {}", archive.network_events.len());
    println!("   Console messages: {}", archive.console_messages.len());

    // 10. Span status
    println!("\n10. Span status values...");
    let statuses = ["Running", "Ok", "Error", "Cancelled"];

    for status in &statuses {
        println!("   {}", status);
    }

    println!("\n✅ Execution tracing example completed!");
    Ok(())
}
