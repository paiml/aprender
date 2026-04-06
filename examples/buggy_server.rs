//! Buggy server simulation for anomaly detection demo
//!
//! This creates syscall patterns with intentional latency spikes
//! to demonstrate renacer's anomaly detection capabilities.

use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::thread;
use std::time::Duration;

fn main() {
    println!("Buggy Server v0.1 - Simulating production workload...");
    println!("Watch for anomalies in the TUI!");

    let mut iteration = 0u64;

    loop {
        iteration += 1;

        // Normal fast operations (99% of the time)
        for _ in 0..100 {
            fast_operation();
        }

        // THE BUG: Every 50 iterations, simulate a pathological case
        // This could be a cache miss, lock contention, or disk flush
        if iteration % 50 == 0 {
            println!("[ANOMALY] Iteration {}: Triggering slow path...", iteration);
            slow_buggy_operation();
        }

        // Every 200 iterations, simulate an even worse scenario
        if iteration % 200 == 0 {
            println!("[CRITICAL] Iteration {}: Synchronous disk flush!", iteration);
            catastrophic_flush();
        }

        thread::sleep(Duration::from_millis(10));
    }
}

/// Fast path - typical microsecond-level syscalls
fn fast_operation() {
    // Quick file stat - should be ~1-5μs
    let _ = std::fs::metadata("/tmp");

    // Quick read from /dev/null
    if let Ok(mut f) = File::open("/dev/null") {
        let mut buf = [0u8; 64];
        let _ = f.read(&mut buf);
    }
}

/// THE BUG: Simulates a slow path that shouldn't happen
/// In real code this might be: missing index, lock contention, GC pause
fn slow_buggy_operation() {
    // Simulate: accidentally reading a large file synchronously
    // This creates a 10-50ms latency spike
    let path = "/tmp/renacer_demo_largefile";

    // Create a "large" file if it doesn't exist
    if let Ok(mut f) = OpenOptions::new().write(true).create(true).truncate(true).open(path) {
        // Write 1MB of data
        let data = vec![0xABu8; 1024 * 1024];
        let _ = f.write_all(&data);
        let _ = f.sync_all(); // Force to disk - this is slow!
    }

    // Now read it back synchronously
    if let Ok(mut f) = File::open(path) {
        let mut buf = Vec::new();
        let _ = f.read_to_end(&mut buf);
    }
}

/// CATASTROPHIC: Full fsync - simulates the worst case
fn catastrophic_flush() {
    // This forces ALL pending writes to disk
    // In production this might be an accidental sync_all() in hot path

    let path = "/tmp/renacer_demo_sync";
    if let Ok(mut f) = OpenOptions::new().write(true).create(true).truncate(true).open(path) {
        // Write data
        let data = vec![0xFFu8; 4 * 1024 * 1024]; // 4MB
        let _ = f.write_all(&data);

        // THE KILLER: sync_all() on 4MB
        // This can take 50-200ms on spinning disk, 5-20ms on SSD
        let _ = f.sync_all();
    }
}
