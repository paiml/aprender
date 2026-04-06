#![allow(clippy::disallowed_methods)]
//! Incremental Profile Snapshots Demo
//!
//! Demonstrates delta-compressed profile storage.
//!
//! Run with: cargo run --example incremental_snapshot_demo -p cbtop

use cbtop::{IncrementalSnapshotStore, MetricData, ProfileSnapshot, SnapshotConfig, SnapshotQuery};

fn main() {
    println!("=== Incremental Profile Snapshots Demo ===\n");

    // Create snapshot store with keyframe every 5 snapshots
    let config =
        SnapshotConfig { keyframe_interval: 5, verify_checksums: true, ..Default::default() };
    let mut store = IncrementalSnapshotStore::new(config);

    // Create 20 sequential snapshots
    println!("Creating 20 profile snapshots...");
    for i in 0..20 {
        let mut snapshot = ProfileSnapshot::new(i);
        snapshot.set_fingerprint(format!("workload_{}", i % 3));
        snapshot.timestamp_ns = (1000000000 + i * 1000000) as u64;

        // Add metrics with stable values (good for delta compression)
        for m in 0..5 {
            let mut metric = MetricData::new(format!("metric_{}", m));
            for v in 0..100 {
                // Stable base values with small drift
                let value = (m * 100 + v) as f64 + (i % 2) as f64 * 0.1;
                metric.add(value, snapshot.timestamp_ns + v as u64);
            }
            snapshot.add_metric(metric);
        }

        store.append(snapshot).unwrap();
    }

    // Show storage statistics
    println!("\n=== Storage Statistics ===");
    println!("Snapshots stored: {}", store.count());
    println!("Raw size: {} bytes", store.total_raw_size());
    println!("Compressed size: {} bytes", store.total_compressed_size());
    println!("Compression ratio: {:.1}%", store.compression_ratio() * 100.0);

    // Query by fingerprint
    println!("\n=== Query by Workload Fingerprint ===");
    let query = SnapshotQuery::new().fingerprint("workload_0");
    let results = store.query(&query).unwrap();
    println!("Snapshots with fingerprint 'workload_0': {}", results.len());

    // Query by time range
    println!("\n=== Query by Time Range ===");
    let query = SnapshotQuery::new().time_range(1000005000000, 1000015000000);
    let results = store.query(&query).unwrap();
    println!(
        "Snapshots in time range: {} ({} - {})",
        results.len(),
        results.first().map(|s| s.index).unwrap_or(0),
        results.last().map(|s| s.index).unwrap_or(0)
    );

    // Demonstrate snapshot reconstruction
    println!("\n=== Snapshot Reconstruction ===");
    for idx in [0, 3, 7, 15] {
        let snapshot = store.get(idx).unwrap();
        let metric = snapshot.metrics.get("metric_0").unwrap();
        println!(
            "Snapshot {}: {} metrics, {} values in metric_0, fingerprint={}",
            idx,
            snapshot.metrics.len(),
            metric.values.len(),
            snapshot.workload_fingerprint
        );
    }

    // Verify checksum integrity
    println!("\n=== Checksum Verification ===");
    let snapshot = store.get(10).unwrap();
    let verified = snapshot.verify_checksum();
    println!("Snapshot 10 checksum valid: {}", verified);

    println!("\n✅ Incremental snapshots demo complete!");
}
