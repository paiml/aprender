//! Docker Cross-Browser WASM Testing Demo
//!
//! Demonstrates DockerTestRunner from PROBAR-SPEC-014:
//! - Container lifecycle management (start/stop)
//! - Multi-browser testing (Chrome, Firefox, WebKit)
//! - COOP/COEP header configuration for SharedArrayBuffer
//! - Parallel cross-browser test execution
//! - Result aggregation and reporting
//!
//! Run with: cargo run --example docker_demo -p jugar-probar --features docker

#[cfg(feature = "docker")]
use jugar_probar::docker::{
    check_shared_array_buffer_support, validate_coop_coep_headers, Browser, ContainerConfig,
    ContainerState, CoopCoepConfig, DockerTestRunner, ParallelRunner,
};
#[cfg(feature = "docker")]
use std::collections::HashMap;
#[cfg(feature = "docker")]
use std::time::Duration;

#[cfg(feature = "docker")]
fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║     Docker Cross-Browser WASM Testing (PROBAR-SPEC-014)      ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    demo_browser_configuration();
    demo_coop_coep_headers();
    demo_single_browser_lifecycle();
    demo_parallel_cross_browser();
    demo_header_validation();
    demo_container_config();
}

#[cfg(feature = "docker")]
fn demo_browser_configuration() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  1. Browser Configuration");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    println!("  Supported browsers for Docker-based testing:\n");

    for browser in Browser::all() {
        println!("    Browser: {}", browser);
        println!("      ├─ CDP Port: {}", browser.default_cdp_port());
        println!("      ├─ Image: {}", browser.image_name());
        println!(
            "      └─ Container Prefix: {}\n",
            browser.container_prefix()
        );
    }

    println!("  Browser parsing examples:");
    let test_cases = ["chrome", "FIREFOX", "webkit", "safari", "ff", "chromium"];
    for name in test_cases {
        match Browser::from_str(name) {
            Some(b) => println!("    '{}' -> {:?}", name, b),
            None => println!("    '{}' -> (not recognized)", name),
        }
    }
    println!();
}

#[cfg(feature = "docker")]
fn demo_coop_coep_headers() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  2. COOP/COEP Header Configuration");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    println!("  Required headers for SharedArrayBuffer support:\n");

    let config = CoopCoepConfig::default();
    println!("    CoopCoepConfig::default():");
    println!("      ├─ COOP: {}", config.coop);
    println!("      ├─ COEP: {}", config.coep);
    println!("      ├─ CORP: {}", config.corp);
    println!("      ├─ Enabled: {}", config.enabled);
    println!(
        "      └─ SharedArrayBuffer Available: {}\n",
        config.shared_array_buffer_available()
    );

    let disabled = CoopCoepConfig::disabled();
    println!("    CoopCoepConfig::disabled():");
    println!("      ├─ Enabled: {}", disabled.enabled);
    println!(
        "      └─ SharedArrayBuffer Available: {}\n",
        disabled.shared_array_buffer_available()
    );

    println!("  Why COOP/COEP matters:");
    println!("    Cross-Origin-Opener-Policy: same-origin");
    println!("      → Prevents other origins from opening window reference");
    println!("    Cross-Origin-Embedder-Policy: require-corp");
    println!("      → Requires explicit CORS or CORP on all subresources");
    println!("    Together: Enables cross-origin isolation for SharedArrayBuffer\n");
}

#[cfg(feature = "docker")]
fn demo_single_browser_lifecycle() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  3. Single Browser Container Lifecycle");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    // Create a runner for Chrome
    let mut runner = DockerTestRunner::builder()
        .browser(Browser::Chrome)
        .with_coop_coep(true)
        .timeout(Duration::from_secs(60))
        .cleanup(true)
        .capture_logs(true)
        .build()
        .expect("Failed to build runner");

    println!("    Created DockerTestRunner for Chrome");
    println!("      ├─ CDP URL: {}", runner.cdp_url());
    println!("      ├─ Timeout: {:?}", runner.config().timeout);
    println!("      └─ Initial State: {}\n", runner.state());

    // Simulate container lifecycle
    println!("    Container lifecycle simulation:");

    print!("      1. Starting container... ");
    runner.simulate_start().expect("Failed to start");
    println!("✓ State: {}", runner.state());

    if let Some(id) = runner.container_id() {
        println!("         Container ID: {}", id);
    }

    print!("      2. Running tests... ");
    let results = runner
        .simulate_run_tests(&[
            "tests/worker_lifecycle.rs",
            "tests/shared_memory.rs",
            "tests/ring_buffer.rs",
        ])
        .expect("Failed to run tests");
    println!("✓");
    println!(
        "         Passed: {}, Failed: {}",
        results.passed, results.failed
    );
    println!("         Duration: {:?}", results.total_duration);

    print!("      3. Stopping container... ");
    runner.simulate_stop().expect("Failed to stop");
    println!("✓ State: {}", runner.state());

    println!("\n    Captured logs:");
    for log in runner.logs() {
        println!("      │ {}", log);
    }
    println!();
}

#[cfg(feature = "docker")]
fn demo_parallel_cross_browser() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  4. Parallel Cross-Browser Testing");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    let mut runner = ParallelRunner::builder()
        .browsers(&Browser::all())
        .tests(&[
            "tests/e2e_worker.rs",
            "tests/e2e_atomics.rs",
            "tests/e2e_audio_worklet.rs",
        ])
        .timeout(Duration::from_secs(120))
        .build()
        .expect("Failed to build parallel runner");

    println!("    ParallelRunner configuration:");
    println!("      ├─ Browsers: {:?}", runner.browsers());
    println!("      └─ Tests: {} test files\n", runner.tests().len());

    print!("    Running tests across all browsers... ");
    runner.simulate_run().expect("Failed to run");
    println!("✓\n");

    println!("    Results by browser:");
    for (browser, result) in runner.results_by_browser() {
        let status = if result.all_passed() { "✓" } else { "✗" };
        println!(
            "      {} {}: {} passed, {} failed ({:.1}%)",
            status,
            browser,
            result.passed,
            result.failed,
            result.pass_rate()
        );
    }

    let (total_passed, total_failed, total_duration) = runner.aggregate_stats();
    println!("\n    Aggregate stats:");
    println!("      ├─ Total passed: {}", total_passed);
    println!("      ├─ Total failed: {}", total_failed);
    println!("      ├─ Total duration: {:?}", total_duration);
    println!(
        "      └─ All browsers passed: {}\n",
        if runner.all_passed() {
            "✓ Yes"
        } else {
            "✗ No"
        }
    );
}

#[cfg(feature = "docker")]
fn demo_header_validation() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  5. HTTP Header Validation");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    println!("  Validating server response headers:\n");

    // Valid headers
    let mut valid_headers = HashMap::new();
    valid_headers.insert(
        "Cross-Origin-Opener-Policy".to_string(),
        "same-origin".to_string(),
    );
    valid_headers.insert(
        "Cross-Origin-Embedder-Policy".to_string(),
        "require-corp".to_string(),
    );

    print!("    Valid headers: ");
    match validate_coop_coep_headers(&valid_headers) {
        Ok(true) => println!("✓ SharedArrayBuffer enabled"),
        Ok(false) => println!("✗ Headers present but invalid"),
        Err(e) => println!("✗ Error: {}", e),
    }

    // Missing COOP
    let mut missing_coop = HashMap::new();
    missing_coop.insert(
        "Cross-Origin-Embedder-Policy".to_string(),
        "require-corp".to_string(),
    );

    print!("    Missing COOP: ");
    match validate_coop_coep_headers(&missing_coop) {
        Ok(_) => println!("✓ Valid"),
        Err(e) => println!("✗ {}", e),
    }

    // Wrong values
    let mut wrong_values = HashMap::new();
    wrong_values.insert(
        "cross-origin-opener-policy".to_string(),
        "unsafe-none".to_string(),
    );
    wrong_values.insert(
        "cross-origin-embedder-policy".to_string(),
        "unsafe-none".to_string(),
    );

    print!("    Wrong values: ");
    match validate_coop_coep_headers(&wrong_values) {
        Ok(_) => println!("✓ Valid"),
        Err(e) => println!("✗ {}", e),
    }

    // Check SharedArrayBuffer support helper
    println!("\n  SharedArrayBuffer support check:");
    let config = CoopCoepConfig::default();
    let supported = check_shared_array_buffer_support(&config);
    println!(
        "    Default config: {}",
        if supported {
            "✓ Supported"
        } else {
            "✗ Not supported"
        }
    );

    let disabled = CoopCoepConfig::disabled();
    let supported = check_shared_array_buffer_support(&disabled);
    println!(
        "    Disabled config: {}",
        if supported {
            "✓ Supported"
        } else {
            "✗ Not supported"
        }
    );
    println!();
}

#[cfg(feature = "docker")]
fn demo_container_config() {
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  6. Container Configuration");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    println!("  Container states:");
    let states = [
        ContainerState::NotCreated,
        ContainerState::Creating,
        ContainerState::Starting,
        ContainerState::Running,
        ContainerState::HealthChecking,
        ContainerState::Stopping,
        ContainerState::Stopped,
        ContainerState::Error,
    ];
    for state in states {
        println!("    - {}", state);
    }

    println!("\n  Browser-specific container configs:");
    for browser in Browser::all() {
        let config = ContainerConfig::for_browser(browser);
        println!("\n    {}:", browser);
        println!("      ├─ Image: {}", config.image);
        println!("      ├─ Ports: {:?}", config.ports);
        println!(
            "      ├─ Memory limit: {} MB",
            config.memory_limit.unwrap_or(0) / 1024 / 1024
        );
        println!("      └─ CPU limit: {:?}", config.cpu_limit);
    }

    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  Demo complete!");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
}

#[cfg(not(feature = "docker"))]
fn main() {
    eprintln!("This example requires the 'docker' feature.");
    eprintln!("Run with: cargo run --example docker_demo -p jugar-probar --features docker");
}
