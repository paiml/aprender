//! TUI Simulation Tests for GEMV Scheduler Flow
//!
//! These tests visualize data flow through the CudaScheduler to catch
//! state accumulation bugs that unit tests miss.
//!
//! Run with: cargo test --test probar_tui_simulation --features cuda -- --nocapture

#[cfg(feature = "cuda")]
use realizar::gpu::CudaScheduler;

/// Simple value tracker for TUI simulation
#[allow(dead_code)]
struct ValueTracker {
    name: String,
    values: Vec<f32>,
}

#[allow(dead_code)]
impl ValueTracker {
    fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            values: Vec::new(),
        }
    }

    fn record(&mut self, value: f32) {
        self.values.push(value);
    }

    fn values(&self) -> &[f32] {
        &self.values
    }
}

/// TUI Simulation: Watch data flow through CudaScheduler
/// Catches state accumulation bugs that unit tests miss
#[test]
#[cfg(feature = "cuda")]
fn test_scheduler_parity_tui_simulation() {
    println!();
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║  TUI SIMULATION: GEMV Data Flow Through CudaScheduler                ║");
    println!("║  PARITY-119: Visual testing per decoder-throughput spec §12.3        ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝");

    let mut scheduler = CudaScheduler::new().expect("CUDA init");
    let mut tracker = ValueTracker::new("output[0]");

    // Test data: small dimensions to catch the early exit bug
    let k = 16usize;
    let n = 8usize;
    let x: Vec<f32> = vec![1.0; k];
    let a: Vec<f32> = vec![1.0; k * n];
    let expected = k as f32; // sum of k ones

    // Step 1: First execution
    println!();
    println!("┌─ STEP 1: First Execution ────────────────────────────────────────────┐");
    println!("│  Input: x = [1.0; {}], A = [1.0; {}×{}]", k, k, n);
    let r1 = scheduler.matmul(&x, &a, 1, k, n).expect("matmul 1");
    tracker.record(r1[0]);
    let s1 = if (r1[0] - expected).abs() < 0.001 {
        "✓"
    } else {
        "✗"
    };
    println!(
        "│  Result[0] = {:.1} (expected: {:.1}) {}",
        r1[0], expected, s1
    );
    println!("└──────────────────────────────────────────────────────────────────────┘");

    // Step 2: Second execution (same inputs - MUST be identical)
    println!();
    println!("┌─ STEP 2: Second Execution (State Isolation Check) ────────────────────┐");
    println!("│  Same inputs as Step 1");
    let r2 = scheduler.matmul(&x, &a, 1, k, n).expect("matmul 2");
    tracker.record(r2[0]);
    let s2 = if (r2[0] - expected).abs() < 0.001 {
        "✓"
    } else {
        "✗"
    };
    println!(
        "│  Result[0] = {:.1} (expected: {:.1}) {}",
        r2[0], expected, s2
    );

    if (r1[0] - r2[0]).abs() > 0.001 {
        println!(
            "│  ⚠️  STATE LEAK DETECTED: r1={:.1}, r2={:.1}",
            r1[0], r2[0]
        );
        println!(
            "│  Ratio: {:.2}x (common patterns: 2x=half iters, 4x=tile, 8x=accum)",
            r2[0] / r1[0]
        );
    } else {
        println!("│  ✓ No state leak: results identical");
    }
    println!("└──────────────────────────────────────────────────────────────────────┘");

    // Step 3: Different input - verify isolation
    println!();
    println!("┌─ STEP 3: Different Input ──────────────────────────────────────────────┐");
    let x2: Vec<f32> = vec![2.0; k];
    let expected2 = k as f32 * 2.0;
    println!("│  Input: x = [2.0; {}], A = [1.0; {}×{}]", k, k, n);
    let r3 = scheduler.matmul(&x2, &a, 1, k, n).expect("matmul 3");
    tracker.record(r3[0]);
    let s3 = if (r3[0] - expected2).abs() < 0.001 {
        "✓"
    } else {
        "✗"
    };
    println!(
        "│  Result[0] = {:.1} (expected: {:.1}) {}",
        r3[0], expected2, s3
    );
    println!("└──────────────────────────────────────────────────────────────────────┘");

    // Step 4: Return to original - verify no contamination
    println!();
    println!("┌─ STEP 4: Return to Original Input ─────────────────────────────────────┐");
    println!("│  Same inputs as Step 1 (after Step 3 modified them)");
    let r4 = scheduler.matmul(&x, &a, 1, k, n).expect("matmul 4");
    tracker.record(r4[0]);
    let s4 = if (r4[0] - expected).abs() < 0.001 {
        "✓"
    } else {
        "✗"
    };
    println!(
        "│  Result[0] = {:.1} (expected: {:.1}) {}",
        r4[0], expected, s4
    );

    if (r1[0] - r4[0]).abs() > 0.001 {
        println!("│  ⚠️  STATE CONTAMINATION: Step 3 affected Step 4!");
    } else {
        println!("│  ✓ No contamination from Step 3");
    }
    println!("└──────────────────────────────────────────────────────────────────────┘");

    // Final analysis
    println!();
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║  ANALYSIS                                                            ║");
    println!("╠══════════════════════════════════════════════════════════════════════╣");
    println!("║  Values tracked: {:?}", tracker.values());
    let isolation = (r1[0] - r2[0]).abs() < 0.001;
    let correctness = (r1[0] - expected).abs() < 0.001;
    let no_contamination = (r1[0] - r4[0]).abs() < 0.001;
    println!(
        "║  T1 State isolation: {}",
        if isolation { "PASS ✓" } else { "FAIL ✗" }
    );
    println!(
        "║  T2 Correctness:     {}",
        if correctness { "PASS ✓" } else { "FAIL ✗" }
    );
    println!(
        "║  T3 No contamination: {}",
        if no_contamination {
            "PASS ✓"
        } else {
            "FAIL ✗"
        }
    );
    println!("╚══════════════════════════════════════════════════════════════════════╝");

    // Assertions for CI
    assert!(
        (r1[0] - expected).abs() < 0.001,
        "T2 Correctness failed: expected {}, got {}",
        expected,
        r1[0]
    );
    assert!(
        (r1[0] - r2[0]).abs() < 0.001,
        "T1 State isolation failed: r1={}, r2={}",
        r1[0],
        r2[0]
    );
    assert!(
        (r3[0] - expected2).abs() < 0.001,
        "Different input failed: expected {}, got {}",
        expected2,
        r3[0]
    );
    assert!(
        (r1[0] - r4[0]).abs() < 0.001,
        "State contamination: r1={}, r4={}",
        r1[0],
        r4[0]
    );
}

/// TUI Simulation: Boundary condition testing
/// Tests edge cases that often reveal kernel bugs
#[test]
#[cfg(feature = "cuda")]
#[ignore = "requires CUDA runtime library access"]
fn test_tui_boundary_simulation() {
    println!();
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║  TUI SIMULATION: Boundary Conditions                                 ║");
    println!("║  PARITY-119: Tests edge cases per decoder-throughput spec §12.3      ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝");

    let mut scheduler = CudaScheduler::new().expect("CUDA init");

    let test_cases = vec![
        ("n < TILE_SIZE (caught bug)", 256, 8),
        ("k < TILE_SIZE", 16, 256),
        ("Both < TILE_SIZE", 16, 8),
        ("Exact TILE_SIZE", 256, 256),
        ("2× TILE_SIZE", 512, 512),
        ("Large (4K)", 4096, 4096),
        ("Non-power-of-two", 127, 63),
        ("Prime dimensions", 127, 131),
    ];

    let mut all_passed = true;

    for (name, k, n) in test_cases {
        println!();
        println!(
            "┌─ {} (k={}, n={}) ─────────────────────────────────",
            name, k, n
        );
        let x: Vec<f32> = vec![1.0; k];
        let a: Vec<f32> = vec![1.0; k * n];
        let expected = k as f32;

        let result = scheduler.matmul(&x, &a, 1, k, n).expect("matmul");
        let passed = (result[0] - expected).abs() < 0.01;
        let status = if passed { "✓ PASS" } else { "✗ FAIL" };

        println!(
            "│  Expected: {:.1}, Got: {:.1} → {}",
            expected, result[0], status
        );

        if !passed {
            let ratio = result[0] / expected;
            println!("│  Ratio: {:.4}x", ratio);
            all_passed = false;
        }
        println!("└──────────────────────────────────────────────────────────────");
    }

    println!();
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║  SUMMARY: T3 Boundary Conditions                                     ║");
    println!("╠══════════════════════════════════════════════════════════════════════╣");
    println!(
        "║  Result: {}",
        if all_passed {
            "ALL PASSED ✓"
        } else {
            "SOME FAILED ✗"
        }
    );
    println!("╚══════════════════════════════════════════════════════════════════════╝");

    assert!(all_passed, "Some boundary condition tests failed");
}

/// TUI Simulation: Large dimension stress test
#[test]
#[cfg(feature = "cuda")]
#[ignore = "requires CUDA runtime library access"]
fn test_tui_large_dimension_simulation() {
    println!();
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║  TUI SIMULATION: Large Dimension Stress Test                         ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝");

    let mut scheduler = CudaScheduler::new().expect("CUDA init");

    let k = 4096usize;
    let n = 4096usize;

    // Use varying input values for more robust testing
    let x: Vec<f32> = (0..k).map(|i| (i % 10) as f32 * 0.1).collect();
    let a: Vec<f32> = (0..k * n).map(|i| ((i % 7) as f32) * 0.01).collect();

    // CPU reference computation
    println!();
    println!("┌─ Computing CPU reference ──────────────────────────────────────────────┐");
    let mut expected = vec![0.0f32; n];
    for j in 0..n {
        let mut sum = 0.0f32;
        for i in 0..k {
            sum += x[i] * a[i * n + j];
        }
        expected[j] = sum;
    }
    println!(
        "│  CPU reference computed for {}×{} = {} elements",
        k,
        n,
        k * n
    );
    println!("│  Expected[0] = {:.6}", expected[0]);
    println!("│  Expected[n-1] = {:.6}", expected[n - 1]);
    println!("└──────────────────────────────────────────────────────────────────────┘");

    // GPU computation
    println!();
    println!("┌─ Computing GPU result ─────────────────────────────────────────────────┐");
    let result = scheduler.matmul(&x, &a, 1, k, n).expect("matmul");
    println!("│  GPU result computed");
    println!("│  Result[0] = {:.6}", result[0]);
    println!("│  Result[n-1] = {:.6}", result[n - 1]);
    println!("└──────────────────────────────────────────────────────────────────────┘");

    // Comparison
    println!();
    println!("┌─ Comparing Results ────────────────────────────────────────────────────┐");
    let mut max_error = 0.0f32;
    let mut max_error_idx = 0;
    let mut error_count = 0;

    for (i, (e, a)) in expected.iter().zip(result.iter()).enumerate() {
        let error = (e - a).abs();
        if error > max_error {
            max_error = error;
            max_error_idx = i;
        }
        if error > 0.01 {
            error_count += 1;
        }
    }

    println!("│  Max error: {:.6} at index {}", max_error, max_error_idx);
    println!("│  Elements with error > 0.01: {}/{}", error_count, n);
    let passed = max_error < 0.1;
    println!("│  Result: {}", if passed { "PASS ✓" } else { "FAIL ✗" });
    println!("└──────────────────────────────────────────────────────────────────────┘");

    assert!(
        passed,
        "Large dimension test failed: max_error = {}",
        max_error
    );
}

/// TUI Simulation: Sequential operations verify no state leakage
#[test]
#[cfg(feature = "cuda")]
#[ignore = "requires CUDA runtime library access"]
fn test_tui_sequential_operations() {
    println!();
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║  TUI SIMULATION: Sequential Operations (State Leakage Detection)    ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝");

    let mut scheduler = CudaScheduler::new().expect("CUDA init");

    let configs = vec![(64, 64), (128, 128), (256, 256), (512, 512), (1024, 1024)];

    println!();
    println!("┌─ Running sequential operations with varying sizes ────────────────────┐");

    let mut all_passed = true;

    for (k, n) in &configs {
        let x: Vec<f32> = vec![1.0; *k];
        let a: Vec<f32> = vec![1.0; k * n];
        let expected = *k as f32;

        // Run twice, compare
        let r1 = scheduler.matmul(&x, &a, 1, *k, *n).expect("matmul 1");
        let r2 = scheduler.matmul(&x, &a, 1, *k, *n).expect("matmul 2");

        let diff = (r1[0] - r2[0]).abs();
        let passed = diff < 0.001 && (r1[0] - expected).abs() < 0.01;
        let status = if passed { "✓" } else { "✗" };

        println!(
            "│  {}×{}: r1={:.1}, r2={:.1}, diff={:.4} {}",
            k, n, r1[0], r2[0], diff, status
        );

        if !passed {
            all_passed = false;
        }
    }

    println!("└──────────────────────────────────────────────────────────────────────┘");

    println!();
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!(
        "║  Result: {}",
        if all_passed {
            "NO STATE LEAKAGE DETECTED ✓"
        } else {
            "STATE LEAKAGE DETECTED ✗"
        }
    );
    println!("╚══════════════════════════════════════════════════════════════════════╝");

    assert!(
        all_passed,
        "State leakage detected in sequential operations"
    );
}
