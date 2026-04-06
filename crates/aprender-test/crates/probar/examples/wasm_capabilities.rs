//! WASM Thread Capabilities Demo
//!
//! Demonstrates WasmThreadCapabilities from PROBAR-SPEC-011:
//! - SharedArrayBuffer availability detection
//! - COOP/COEP header verification
//! - Thread capability querying
//! - Feature detection for WASM threading
//!
//! Run with: cargo run --example wasm_capabilities -p jugar-probar

use jugar_probar::capabilities::{CapabilityStatus, RequiredHeaders, WasmThreadCapabilities};

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║      WASM Thread Capabilities Demo (PROBAR-SPEC-011)         ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    demo_capability_status();
    demo_required_headers();
    demo_full_support();
    demo_no_support();
    demo_threading_assertion();
}

/// Demonstrate capability status enum
fn demo_capability_status() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  1. Capability Status Types");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let statuses: Vec<(CapabilityStatus, &str)> = vec![
        (
            CapabilityStatus::Available,
            "Feature is available and working",
        ),
        (
            CapabilityStatus::Unavailable("Missing COOP/COEP headers".to_string()),
            "Feature is not available with reason",
        ),
        (
            CapabilityStatus::Unknown,
            "Capability status could not be determined",
        ),
    ];

    println!("  Capability status meanings:\n");
    for (status, description) in &statuses {
        println!("    {:?}", status);
        println!("      {}\n", description);
    }
}

/// Demonstrate required headers
fn demo_required_headers() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  2. Required HTTP Headers for Threading");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    println!("  For SharedArrayBuffer to work, your server must send:\n");
    println!("    Cross-Origin-Opener-Policy: {}", RequiredHeaders::COOP);
    println!(
        "    Cross-Origin-Embedder-Policy: {}",
        RequiredHeaders::COEP
    );

    println!("\n  Alternative COEP values:");
    println!("    - require-corp (strict, recommended)");
    println!("    - credentialless (more permissive)");

    println!("\n  Example server configuration (nginx):");
    println!("    ─────────────────────────────────────");
    println!("    add_header Cross-Origin-Opener-Policy same-origin;");
    println!("    add_header Cross-Origin-Embedder-Policy require-corp;");
    println!();
}

/// Demonstrate full threading support
fn demo_full_support() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  3. Full Threading Support Configuration");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let caps = WasmThreadCapabilities::full_support();

    println!("  WasmThreadCapabilities::full_support():");
    println!(
        "    ├─ cross_origin_isolated: {}",
        caps.cross_origin_isolated
    );
    println!("    ├─ shared_array_buffer: {}", caps.shared_array_buffer);
    println!("    ├─ atomics: {}", caps.atomics);
    println!("    ├─ hardware_concurrency: {}", caps.hardware_concurrency);
    println!(
        "    ├─ coop_header: {:?}",
        caps.coop_header.as_deref().unwrap_or("(none)")
    );
    println!(
        "    ├─ coep_header: {:?}",
        caps.coep_header.as_deref().unwrap_or("(none)")
    );
    println!("    └─ is_secure_context: {}", caps.is_secure_context);

    println!("\n  Threading assertion: ");
    match caps.assert_threading_ready() {
        Ok(()) => println!("    ✓ All requirements met for multi-threaded WASM"),
        Err(e) => println!("    ✗ Error: {}", e),
    }
    println!();
}

/// Demonstrate no threading support
fn demo_no_support() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  4. No Threading Support Configuration");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let caps = WasmThreadCapabilities::no_support();

    println!("  WasmThreadCapabilities::no_support():");
    println!(
        "    ├─ cross_origin_isolated: {}",
        caps.cross_origin_isolated
    );
    println!("    ├─ shared_array_buffer: {}", caps.shared_array_buffer);
    println!("    ├─ atomics: {}", caps.atomics);
    println!("    ├─ hardware_concurrency: {}", caps.hardware_concurrency);
    println!(
        "    ├─ coop_header: {}",
        caps.coop_header.as_deref().unwrap_or("(none)")
    );
    println!(
        "    ├─ coep_header: {}",
        caps.coep_header.as_deref().unwrap_or("(none)")
    );
    println!("    └─ is_secure_context: {}", caps.is_secure_context);

    println!("\n  Threading assertion: ");
    match caps.assert_threading_ready() {
        Ok(()) => println!("    ✓ All requirements met"),
        Err(e) => println!("    ✗ Error: {}", e),
    }
    println!();
}

/// Demonstrate threading assertion with custom config
fn demo_threading_assertion() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  5. Custom Threading Configuration");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Partial support - SAB available but no atomics
    let partial = WasmThreadCapabilities {
        cross_origin_isolated: true,
        shared_array_buffer: true,
        atomics: false, // Missing atomics
        hardware_concurrency: 4,
        coop_header: Some("same-origin".to_string()),
        coep_header: Some("require-corp".to_string()),
        is_secure_context: true,
        errors: vec![],
    };

    println!("  Partial support (SharedArrayBuffer but no Atomics):");
    println!(
        "    ├─ shared_array_buffer: {}",
        partial.shared_array_buffer
    );
    println!("    └─ atomics: {}", partial.atomics);

    println!("\n  Threading assertion: ");
    match partial.assert_threading_ready() {
        Ok(()) => println!("    ✓ All requirements met"),
        Err(e) => println!("    ✗ {}", e),
    }

    // Missing headers
    let no_headers = WasmThreadCapabilities {
        cross_origin_isolated: false,
        shared_array_buffer: false,
        atomics: false,
        hardware_concurrency: 4,
        coop_header: None,
        coep_header: None,
        is_secure_context: true,
        errors: vec![],
    };

    println!("\n  Missing COOP/COEP headers:");
    println!("    ├─ coop_header: (none)");
    println!("    └─ coep_header: (none)");

    println!("\n  Threading assertion: ");
    match no_headers.assert_threading_ready() {
        Ok(()) => println!("    ✓ All requirements met"),
        Err(e) => println!("    ✗ {}", e),
    }

    println!();
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  Demo complete!");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
}
