//! web_sys_gen Demo
//!
//! Demonstrates the web_sys_gen module which provides generated abstractions
//! that replace hand-written web_sys calls.
//!
//! Run with: cargo run --example web_sys_gen_demo -p jugar-probar

use jugar_probar::brick::web_sys_gen::{
    BlobUrl, CustomEventDispatcher, EventDetail, FetchClient, PerformanceTiming,
    GENERATION_METADATA,
};

#[allow(clippy::items_after_statements)]
#[allow(clippy::needless_collect)]
fn main() {
    println!("=== web_sys_gen Demo ===\n");

    // Demo 1: Performance Timing
    println!("1. Performance Timing");
    println!("   -------------------");

    let start = PerformanceTiming::now();
    println!("   Current timestamp: {:.3}ms", start);

    // Measure an operation
    let (result, duration) = PerformanceTiming::measure(|| {
        // Simulate some work
        let mut sum = 0u64;
        for i in 0..1_000_000 {
            sum = sum.wrapping_add(i);
        }
        sum
    });

    println!("   Measured operation:");
    println!("     Result: {}", result);
    println!("     Duration: {:.3}ms\n", duration);

    // Demo 2: Event Detail Types
    println!("2. Event Detail Types");
    println!("   -------------------");

    let string_detail = EventDetail::string("Hello, World!");
    println!("   String detail: {:?}", string_detail);

    let number_detail = EventDetail::number(42.5);
    println!("   Number detail: {:?}", number_detail);

    #[derive(serde::Serialize)]
    struct TranscriptData {
        text: String,
        confidence: f32,
        is_final: bool,
    }

    let transcript = TranscriptData {
        text: "Testing speech recognition".into(),
        confidence: 0.95,
        is_final: true,
    };

    let json_detail = EventDetail::json(&transcript);
    println!("   JSON detail: {:?}\n", json_detail);

    // Demo 3: Custom Event Dispatcher
    println!("3. Custom Event Dispatcher");
    println!("   ------------------------");

    let dispatcher = CustomEventDispatcher::new("transcription-complete");
    println!("   Created dispatcher for: transcription-complete");

    // On native, dispatch is a no-op but returns Ok
    match dispatcher.dispatch() {
        Ok(dispatched) => println!("   Dispatch result: {} (native fallback)", dispatched),
        Err(e) => println!("   Dispatch error: {}", e),
    }

    match dispatcher.dispatch_with_detail(EventDetail::string("test")) {
        Ok(dispatched) => {
            println!(
                "   Dispatch with detail result: {} (native fallback)\n",
                dispatched
            );
        }
        Err(e) => println!("   Dispatch with detail error: {}\n", e),
    }

    // Demo 4: Blob URL (native fallback)
    println!("4. Blob URL Creation (native fallback)");
    println!("   ------------------------------------");

    let js_code = r#"
        self.onmessage = (e) => {
            const result = e.data * 2;
            self.postMessage(result);
        };
    "#;

    match BlobUrl::from_js_code(js_code) {
        Ok(url) => println!("   Created blob URL: {}", url),
        Err(e) => println!("   Expected error (native): {}", e),
    }

    // Revoke always succeeds on native
    match BlobUrl::revoke("blob:test") {
        Ok(()) => println!("   Revoke succeeded (native no-op)\n"),
        Err(e) => println!("   Revoke error: {}\n", e),
    }

    // Demo 5: Fetch Client (native fallback)
    println!("5. Fetch Client (native fallback)");
    println!("   -------------------------------");

    let client = FetchClient::new();
    println!("   Created FetchClient");
    println!("   Note: fetch_bytes() is async and returns NotInBrowser on native\n");

    // We can't actually call the async method in this sync example,
    // but we demonstrate the API exists
    let _ = client;

    // Demo 6: Base URL
    println!("6. Base URL");
    println!("   ---------");

    match jugar_probar::brick::web_sys_gen::get_base_url() {
        Some(url) => println!("   Base URL: {}", url),
        None => println!("   No base URL available"),
    }

    // Demo 7: Generation Metadata
    println!("\n7. Generation Metadata");
    println!("   ---------------------");
    println!("   Spec: {}", GENERATION_METADATA.spec);
    println!("   Ticket: {}", GENERATION_METADATA.ticket);
    println!("   Method: {}", GENERATION_METADATA.method);

    // Demo 8: Timing multiple operations
    println!("\n8. Timing Multiple Operations");
    println!("   ----------------------------");

    // Vector allocation
    let (result, duration) = PerformanceTiming::measure(|| {
        let v: Vec<i32> = (0..10_000).collect();
        v.len()
    });
    println!(
        "   Vector allocation: result={}, duration={:.4}ms",
        result, duration
    );

    // String concatenation
    let (result, duration) = PerformanceTiming::measure(|| {
        let mut s = String::new();
        for i in 0..1000 {
            s.push_str(&i.to_string());
        }
        s.len()
    });
    println!(
        "   String concatenation: result={}, duration={:.4}ms",
        result, duration
    );

    // Hash computation
    let (result, duration) = PerformanceTiming::measure(|| {
        use std::collections::HashMap;
        let mut map = HashMap::new();
        for i in 0..1000 {
            map.insert(i, i * 2);
        }
        map.len()
    });
    println!(
        "   Hash computation: result={}, duration={:.4}ms",
        result, duration
    );

    println!("\n=== Demo Complete ===");
}
