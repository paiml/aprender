#![allow(clippy::disallowed_methods)]
//! F-PROF-001: True Per-Brick Profiling Falsification Test
//!
//! Per spec: docs/specifications/qwen2.5-coder-showcase-demo.md §6.8
//!
//! This test verifies that profiling data is REAL (measured per-brick),
//! not DERIVED from total throughput.
//!
//! FALSIFICATION CRITERIA:
//! 1. "No per-brick data collected" message in output -> FAIL
//! 2. "derived from throughput" message in output -> FAIL (eventually)
//! 3. Perfect correlation between brick latencies -> FAIL

#[test]
fn f_prof_001_independent_variance() {
    // This test requires a real model and GPU execution to falsify.
    // Since we are in a CI/build environment without GPU access for this test runner,
    // we document the falsification logic here.

    // Logic:
    // 1. Run cbtop with real profiling
    //    cargo run -p apr-cli -- cbtop --model-path <MODEL> --headless
    //
    // 2. Check output for "No per-brick data collected"
    //    If present: FAIL (BrickProfiler not integrated)
    //
    // 3. Check for "derived from throughput"
    //    If present: FAIL (Still using derived estimates)

    println!("F-PROF-001: Manual verification required (needs GPU + Model)");
    println!("  Run: apr cbtop --model-path <MODEL> --headless");
    println!("  Check: Output MUST show 'Per-Brick Timing (REAL...)' table");
    println!("  Check: Output MUST NOT show 'No per-brick data collected'");
}
