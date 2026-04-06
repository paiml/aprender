//! Tests for trace collectors

use super::hash_chain::HashChainCollector;
use super::ring::RingCollector;
use super::stream::{StreamCollector, StreamFormat};
use super::traits::TraceCollector;
use crate::monitor::inference::path::LinearPath;
use crate::monitor::inference::trace::DecisionTrace;
use std::io::Write;

fn make_test_trace(seq: u64) -> DecisionTrace<LinearPath> {
    let path = LinearPath::new(vec![0.5, -0.3], 0.1, 0.5, 0.87);
    DecisionTrace::new(seq * 1000, seq, seq, path, 0.87, 100)
}

// ==========================================================================
// RingCollector tests
// ==========================================================================

#[test]
fn test_ring_collector_new() {
    let collector = RingCollector::<LinearPath, 64>::new();
    assert_eq!(collector.len(), 0);
    assert!(collector.is_empty());
    assert_eq!(collector.capacity(), 64);
}

#[test]
fn test_ring_collector_record() {
    let mut collector = RingCollector::<LinearPath, 64>::new();
    collector.record(make_test_trace(0));
    assert_eq!(collector.len(), 1);
    assert!(!collector.is_empty());
}

#[test]
fn test_ring_collector_wraparound() {
    let mut collector = RingCollector::<LinearPath, 4>::new();

    // Fill beyond capacity
    for i in 0..6 {
        collector.record(make_test_trace(i));
    }

    assert_eq!(collector.len(), 4);

    // Check we have the last 4 entries
    let all = collector.all();
    assert_eq!(all.len(), 4);
    assert_eq!(all[0].sequence, 2);
    assert_eq!(all[1].sequence, 3);
    assert_eq!(all[2].sequence, 4);
    assert_eq!(all[3].sequence, 5);
}

#[test]
fn test_ring_collector_recent() {
    let mut collector = RingCollector::<LinearPath, 8>::new();

    for i in 0..5 {
        collector.record(make_test_trace(i));
    }

    let recent = collector.recent(3);
    assert_eq!(recent.len(), 3);
    // Most recent first
    assert_eq!(recent[0].sequence, 4);
    assert_eq!(recent[1].sequence, 3);
    assert_eq!(recent[2].sequence, 2);
}

#[test]
fn test_ring_collector_last() {
    let mut collector = RingCollector::<LinearPath, 4>::new();

    assert!(collector.last().is_none());

    collector.record(make_test_trace(42));
    assert_eq!(collector.last().expect("collection should not be empty").sequence, 42);

    collector.record(make_test_trace(43));
    assert_eq!(collector.last().expect("collection should not be empty").sequence, 43);
}

#[test]
fn test_ring_collector_clear() {
    let mut collector = RingCollector::<LinearPath, 4>::new();

    for i in 0..3 {
        collector.record(make_test_trace(i));
    }

    collector.clear();
    assert_eq!(collector.len(), 0);
    assert!(collector.is_empty());
}

#[test]
fn test_ring_collector_flush() {
    let mut collector = RingCollector::<LinearPath, 4>::new();
    assert!(collector.flush().is_ok());
}

// ==========================================================================
// StreamCollector tests
// ==========================================================================

#[test]
fn test_stream_collector_json_lines() {
    let mut buffer = Vec::new();
    {
        let mut collector =
            StreamCollector::<LinearPath, _>::new(&mut buffer, StreamFormat::JsonLines)
                .with_flush_threshold(10);

        collector.record(make_test_trace(0));
        collector.record(make_test_trace(1));
        collector.flush().expect("operation should succeed");
    }

    let output = String::from_utf8(buffer).expect("operation should succeed");
    let lines: Vec<&str> = output.lines().collect();
    assert_eq!(lines.len(), 2);

    // Each line should be valid JSON
    for line in lines {
        assert!(serde_json::from_str::<serde_json::Value>(line).is_ok());
    }
}

#[test]
fn test_stream_collector_binary() {
    let mut buffer = Vec::new();
    {
        let mut collector =
            StreamCollector::<LinearPath, _>::new(&mut buffer, StreamFormat::Binary);

        collector.record(make_test_trace(0));
        collector.flush().expect("operation should succeed");
    }

    // Should have length prefix + trace bytes
    assert!(buffer.len() > 4);

    // Verify length prefix
    let len = u32::from_le_bytes([buffer[0], buffer[1], buffer[2], buffer[3]]) as usize;
    assert_eq!(buffer.len(), 4 + len);
}

#[test]
fn test_stream_collector_auto_flush() {
    use std::sync::{Arc, Mutex};

    let buffer = Arc::new(Mutex::new(Vec::new()));
    {
        let buf_clone = Arc::clone(&buffer);
        let mut collector =
            StreamCollector::<LinearPath, _>::new(SyncWriter(buf_clone), StreamFormat::JsonLines)
                .with_flush_threshold(2);

        collector.record(make_test_trace(0));
        // Should not have written yet (buffered)
        assert!(buffer.lock().unwrap_or_else(std::sync::PoisonError::into_inner).is_empty());

        collector.record(make_test_trace(1));
        // Should have auto-flushed at threshold
    }

    // After drop, should have data
    assert!(!buffer.lock().unwrap_or_else(std::sync::PoisonError::into_inner).is_empty());
}

/// Wrapper to make Arc<Mutex<Vec<u8>>> implement Write + Sync
struct SyncWriter(std::sync::Arc<std::sync::Mutex<Vec<u8>>>);

impl Write for SyncWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap_or_else(std::sync::PoisonError::into_inner).write(buf)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.0.lock().unwrap_or_else(std::sync::PoisonError::into_inner).flush()
    }
}

#[test]
fn test_stream_collector_len() {
    let buffer = Vec::new();
    let mut collector = StreamCollector::<LinearPath, _>::new(buffer, StreamFormat::JsonLines);

    assert_eq!(collector.len(), 0);
    collector.record(make_test_trace(0));
    assert_eq!(collector.len(), 1);
    collector.record(make_test_trace(1));
    assert_eq!(collector.len(), 2);
}

// ==========================================================================
// HashChainCollector tests
// ==========================================================================

#[test]
fn test_hash_chain_new() {
    let collector = HashChainCollector::<LinearPath>::new();
    assert_eq!(collector.len(), 0);
    assert!(collector.is_empty());
    assert_eq!(collector.latest_hash(), [0u8; 32]);
}

#[test]
fn test_hash_chain_record() {
    let mut collector = HashChainCollector::<LinearPath>::new();

    collector.record(make_test_trace(0));
    assert_eq!(collector.len(), 1);
    assert_ne!(collector.latest_hash(), [0u8; 32]);

    let entry = collector.get(0).expect("key should exist");
    assert_eq!(entry.sequence, 0);
    assert_eq!(entry.prev_hash, [0u8; 32]); // Genesis
}

#[test]
fn test_hash_chain_linking() {
    let mut collector = HashChainCollector::<LinearPath>::new();

    collector.record(make_test_trace(0));
    let first_hash = collector.latest_hash();

    collector.record(make_test_trace(1));

    let second_entry = collector.get(1).expect("key should exist");
    assert_eq!(second_entry.prev_hash, first_hash);
}

#[test]
fn test_hash_chain_verify_empty() {
    let collector = HashChainCollector::<LinearPath>::new();
    let verification = collector.verify_chain();
    assert!(verification.valid);
    assert_eq!(verification.entries_verified, 0);
}

#[test]
fn test_hash_chain_verify_valid() {
    let mut collector = HashChainCollector::<LinearPath>::new();

    for i in 0..10 {
        collector.record(make_test_trace(i));
    }

    let verification = collector.verify_chain();
    assert!(verification.valid);
    assert_eq!(verification.entries_verified, 10);
    assert!(verification.first_break.is_none());
}

#[test]
fn test_hash_chain_detect_tampering() {
    let mut collector = HashChainCollector::<LinearPath>::new();

    for i in 0..5 {
        collector.record(make_test_trace(i));
    }

    // Tamper with an entry
    if let Some(entry) = collector.entries.get_mut(2) {
        entry.trace.output = 999.0; // Change the output
    }

    let verification = collector.verify_chain();
    assert!(!verification.valid);
    assert_eq!(verification.first_break, Some(2));
    assert!(verification.error.is_some());
}

#[test]
fn test_hash_chain_detect_sequence_tampering() {
    let mut collector = HashChainCollector::<LinearPath>::new();

    for i in 0..3 {
        collector.record(make_test_trace(i));
    }

    // Tamper with sequence
    if let Some(entry) = collector.entries.get_mut(1) {
        entry.sequence = 999;
    }

    let verification = collector.verify_chain();
    assert!(!verification.valid);
    assert_eq!(verification.first_break, Some(1));
}

#[test]
fn test_hash_chain_to_json() {
    let mut collector = HashChainCollector::<LinearPath>::new();
    collector.record(make_test_trace(0));

    let json = collector.to_json().expect("operation should succeed");
    assert!(json.contains("sequence"));
    assert!(json.contains("prev_hash"));
    assert!(json.contains("hash"));
}

#[test]
fn test_hash_chain_flush() {
    let mut collector = HashChainCollector::<LinearPath>::new();
    assert!(collector.flush().is_ok());
}

#[test]
fn test_hash_chain_entries() {
    let mut collector = HashChainCollector::<LinearPath>::new();

    for i in 0..3 {
        collector.record(make_test_trace(i));
    }

    let entries = collector.entries();
    assert_eq!(entries.len(), 3);
    assert_eq!(entries[0].sequence, 0);
    assert_eq!(entries[1].sequence, 1);
    assert_eq!(entries[2].sequence, 2);
}
