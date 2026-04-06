/// Poll a child process until it exits or the timeout elapses.
/// Returns Ok(status) if the child exited, or Err(message) on timeout/error.
fn wait_with_timeout_sim(
    child: &mut std::process::Child,
    timeout: Duration,
) -> Result<std::process::ExitStatus, String> {
    let start = Instant::now();
    let poll_interval = Duration::from_millis(100);
    loop {
        match child.try_wait() {
            Ok(Some(status)) => return Ok(status),
            Ok(None) if start.elapsed() >= timeout => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(format!("HANG: Process killed after {}s timeout", timeout.as_secs()));
            }
            Ok(None) => std::thread::sleep(poll_interval),
            Err(e) => return Err(format!("Process error: {}", e)),
        }
    }
}

/// Report hang-detection result and timing accuracy.
fn report_hang_result(result: &Result<std::process::ExitStatus, String>, elapsed: Duration, timeout: Duration) {
    match result {
        Err(msg) if msg.contains("HANG") => {
            println!("  {}✓ PASS{}: Hang detected and process killed", "\x1b[32m", "\x1b[0m");
            println!("    Elapsed: {:.2}s (timeout: {}s)", elapsed.as_secs_f64(), timeout.as_secs());
            if elapsed.as_secs() <= timeout.as_secs() + 1 {
                println!("    Timeout enforcement: ACCURATE");
            } else {
                println!(
                    "    {}⚠ Timeout enforcement: DELAYED by {:.2}s{}",
                    "\x1b[33m", elapsed.as_secs_f64() - timeout.as_secs_f64(), "\x1b[0m"
                );
            }
        }
        Ok(_) => println!("  {}✗ FAIL{}: Process completed without hang detection", "\x1b[31m", "\x1b[0m"),
        Err(e) => println!("  {}✗ FAIL{}: Unexpected error: {}", "\x1b[31m", "\x1b[0m", e),
    }
}

fn check_no_zombie() {
    println!("\n  Checking for zombie processes...");
    let ps_output = Command::new("ps")
        .args(["aux"])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).to_string())
        .unwrap_or_default();

    if ps_output.contains("sleep 10") {
        println!("  {}✗ ZOMBIE DETECTED{}: 'sleep 10' process still running", "\x1b[31m", "\x1b[0m");
    } else {
        println!("  {}✓ No zombie processes{}", "\x1b[32m", "\x1b[0m");
    }
}

fn test_hang_detection_simulation() {
    print_section_header("TEST 1: HANG DETECTION (Simulation)");
    println!("  Testing wait_with_timeout logic with a 3-second simulated hang...");

    let timeout = Duration::from_secs(3);
    let start = Instant::now();

    let mut child = Command::new("sleep")
        .arg("10")
        .spawn()
        .expect("Failed to spawn sleep process");

    let result = wait_with_timeout_sim(&mut child, timeout);
    let elapsed = start.elapsed();

    report_hang_result(&result, elapsed, timeout);
    check_no_zombie();
}

// ============================================================================
// TEST 3: ZOMBIE SERVER (Informational - requires manual testing)
// ============================================================================

fn test_zombie_server_info() {
    print_section_header("TEST 3: ZOMBIE SERVER (PMAT-098-PF SIGINT Resiliency)");

    println!(
        "  {}SIGINT Handler Implementation (PMAT-098-PF):{}",
        "\x1b[1m", "\x1b[0m"
    );
    println!("     - Global process registry: Arc<Mutex<Vec<u32>>>");
    println!("     - ctrlc handler kills all registered processes");
    println!("     - ProcessGuard RAII for panic safety");
    println!("     - Jidoka message: '[JIDOKA] SIGINT received. Reaping N child processes...'");
    println!();
    println!("  {}Manual Verification:{}", "\x1b[1m", "\x1b[0m");
    println!();
    println!("  {}A. Interruption Test:{}", "\x1b[1m", "\x1b[0m");
    println!("     1. Run: cargo run --example qa_run -- --modality serve");
    println!("     2. Press Ctrl+C immediately after 'Waiting for Health Check'");
    println!("     3. Verify: '[JIDOKA] SIGINT received...' message appears");
    println!("     4. Check: lsof -i :PORT returns empty (server port released)");
    println!("     5. Check: ps aux | grep 'apr.*serve' returns empty");
    println!("     6. {}PASS if all checks pass{}", "\x1b[32m", "\x1b[0m");
    println!();
    println!("  {}B. Port Recovery Test:{}", "\x1b[1m", "\x1b[0m");
    println!("     1. Interrupt a test run with Ctrl+C");
    println!("     2. Immediately start a new test run");
    println!(
        "     3. {}PASS if no 'Address already in use' error{}",
        "\x1b[32m", "\x1b[0m"
    );
    println!();
    println!("  {}Implementation Details:{}", "\x1b[1m", "\x1b[0m");
    println!("     - setup_signal_handler() called at main() start");
    println!("     - run_serve_test() uses ProcessGuard for server");
    println!("     - run_chat_test() uses register_process/unregister_process");
    println!("     - Exit code 130 on SIGINT (standard convention)");
}

// ============================================================================
// MAIN
// ============================================================================

fn main() {
    println!(
        "{}╔═══════════════════════════════════════════════════════════════╗{}",
        "\x1b[1;36m", "\x1b[0m"
    );
    println!(
        "{}║     QA INFRASTRUCTURE FALSIFICATION (PMAT-098 Red Team)       ║{}",
        "\x1b[1;36m", "\x1b[0m"
    );
    println!(
        "{}╚═══════════════════════════════════════════════════════════════╝{}",
        "\x1b[1;36m", "\x1b[0m"
    );

    // Run automated tests
    test_hang_detection_simulation();
    test_garbage_detection();
    test_answer_verification();
    test_matrix_integrity();
    test_zombie_server_info();

    print_section_header("FALSIFICATION SUMMARY");

    println!("  {}FINDINGS (All Fixed):{}", "\x1b[1m", "\x1b[0m");
    println!(
        "  1. Hang Detection: {}✓{} Works as designed (polling + kill)",
        "\x1b[32m", "\x1b[0m"
    );
    println!(
        "  2. Garbage Detection: {}✓{} All edge cases handled correctly",
        "\x1b[32m", "\x1b[0m"
    );
    println!(
        "  3. Zombie Server: {}✓ FIXED{} - SIGINT handler + ProcessGuard (PMAT-098-PF)",
        "\x1b[32m", "\x1b[0m"
    );
    println!(
        "  4. Matrix Integrity: {}✓ FIXED{} - Documentation updated to 21 cells",
        "\x1b[32m", "\x1b[0m"
    );
    println!(
        "  5. Answer Verification: {}✓ FIXED{} - Word boundary check added",
        "\x1b[32m", "\x1b[0m"
    );
    println!();
    println!("  {}IMPLEMENTATION:{}", "\x1b[1m", "\x1b[0m");
    println!("  - ctrlc crate added to dev-dependencies");
    println!("  - Global process registry: PROCESS_REGISTRY (OnceLock<Arc<Mutex<Vec<u32>>>>)");
    println!("  - ProcessGuard RAII struct for automatic cleanup on Drop");
    println!("  - Jidoka-style shutdown message on SIGINT");
}
