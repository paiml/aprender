#![allow(clippy::disallowed_methods)]
//! B4 CPU Performance Issue Simulation
//!
//! This example simulates the "B4 CPU Performance" issue from E.9.1,
//! demonstrating how high page fault counts can be detected and diagnosed.
//!
//! Run with: cargo run --release --example b4_fault_simulation

use std::time::{Duration, Instant};
use trueno::brick::{cpu_cycles, get_page_faults, init_time_service};

const PAGE_SIZE: usize = 4096;
const FAULT_STORM_SIZE: usize = 50 * 1024 * 1024; // 50MB to trigger faults

fn main() {
    println!("=== B4 CPU Performance Issue Simulation ===\n");
    println!("This simulates the scenario where high page fault counts");
    println!("cause unexpected CPU performance degradation.\n");

    // Initialize timing infrastructure
    init_time_service();
    std::thread::sleep(Duration::from_millis(50));

    // Warm-up phase (establish baseline)
    println!("Phase 1: Baseline measurement (warm cache)");
    let mut baseline_data: Vec<u8> = vec![0u8; 1024 * 1024]; // 1MB warm
    let baseline_start = Instant::now();
    let baseline_cycles_start = cpu_cycles();

    let baseline_len = baseline_data.len();
    for i in 0..1_000_000 {
        baseline_data[i % baseline_len] = (i & 0xFF) as u8;
    }

    let baseline_cycles = cpu_cycles() - baseline_cycles_start;
    let baseline_time = baseline_start.elapsed();
    let baseline_ipc = 1_000_000.0 / baseline_cycles as f64;

    println!("  Cycles: {}", baseline_cycles);
    println!("  Time: {:?}", baseline_time);
    println!("  Estimated IPC: {:.3}", baseline_ipc);

    // Fault storm phase
    println!("\nPhase 2: Page fault storm simulation");
    let (minor_before, major_before) = get_page_faults();
    let fault_start = Instant::now();
    let fault_cycles_start = cpu_cycles();

    // Allocate fresh memory and touch it to trigger page faults
    let mut fault_data: Vec<u8> = Vec::with_capacity(FAULT_STORM_SIZE);
    for i in 0..FAULT_STORM_SIZE {
        fault_data.push((i & 0xFF) as u8);
    }

    // Do some work to measure IPC during fault storm
    let work_cycles_start = cpu_cycles();
    let mut work_sum = 0u64;
    for chunk in fault_data.chunks(PAGE_SIZE) {
        for &byte in chunk {
            work_sum = work_sum.wrapping_add(byte as u64);
        }
    }
    let work_cycles = cpu_cycles() - work_cycles_start;

    let fault_cycles = cpu_cycles() - fault_cycles_start;
    let fault_time = fault_start.elapsed();
    let (minor_after, major_after) = get_page_faults();

    let minor_faults = minor_after.saturating_sub(minor_before);
    let major_faults = major_after.saturating_sub(major_before);
    let fault_ipc = (FAULT_STORM_SIZE as f64) / work_cycles as f64;

    println!("  Allocated: {} MB", FAULT_STORM_SIZE / (1024 * 1024));
    println!("  Minor page faults: {}", minor_faults);
    println!("  Major page faults: {}", major_faults);
    println!("  Cycles: {}", fault_cycles);
    println!("  Work cycles: {}", work_cycles);
    println!("  Time: {:?}", fault_time);
    println!("  Estimated IPC: {:.3}", fault_ipc);
    println!("  Work sum (anti-optimize): {}", work_sum);

    // Detection and warning
    println!("\n=== Diagnostics ===");

    let fault_rate = minor_faults as f64 / fault_time.as_secs_f64();
    let expected_faults = FAULT_STORM_SIZE / PAGE_SIZE;
    let fault_ratio = minor_faults as f64 / expected_faults as f64;

    if minor_faults > 1000 {
        println!("[WARN] High page fault count detected: minor_faults={}", minor_faults);
    }

    if fault_ipc < 0.5 {
        println!("[WARN] Low IPC detected: {:.3} (threshold: 0.5)", fault_ipc);
        println!("       This indicates CPU is stalled waiting for memory.");
    }

    println!("\nPage fault analysis:");
    println!("  Expected faults (pages touched): {}", expected_faults);
    println!("  Actual minor faults: {}", minor_faults);
    println!("  Fault coverage ratio: {:.2}%", fault_ratio * 100.0);
    println!("  Fault rate: {:.0} faults/sec", fault_rate);

    // IPC comparison
    println!("\nIPC comparison:");
    println!("  Baseline IPC: {:.3}", baseline_ipc);
    println!("  Fault storm IPC: {:.3}", fault_ipc);
    let ipc_degradation = (baseline_ipc - fault_ipc) / baseline_ipc * 100.0;
    if ipc_degradation > 0.0 {
        println!("  IPC degradation: {:.1}%", ipc_degradation);
    }

    // Recommendation
    println!("\n=== Recommendation ===");
    if minor_faults > (expected_faults / 2) as u64 {
        println!("The high page fault count suggests memory is being accessed");
        println!("in a pattern that causes many TLB misses and page faults.");
        println!("\nMitigation strategies:");
        println!("  1. Use madvise(MADV_WILLNEED) to prefetch memory");
        println!("  2. Use huge pages (2MB/1GB) to reduce TLB pressure");
        println!("  3. Improve memory access locality");
        println!("  4. Pre-fault memory with memset before use");
    }

    drop(baseline_data);
    drop(fault_data);

    println!("\n✓ B4 fault simulation completed");
}
