use crate::http_client::*;
// ================================================================================

/// Thread safety verification result
#[derive(Debug)]
pub struct ThreadSafetyResult {
    pub num_threads: usize,
    pub num_requests: usize,
    pub successful_requests: usize,
    pub failed_requests: usize,
    pub data_races_detected: bool,
    pub meets_qa028: bool,
}

impl ThreadSafetyResult {
    pub fn success(threads: usize, requests: usize) -> Self {
        Self {
            num_threads: threads,
            num_requests: requests,
            successful_requests: requests,
            failed_requests: 0,
            data_races_detected: false,
            meets_qa028: true,
        }
    }

    pub fn with_failures(threads: usize, total: usize, failed: usize) -> Self {
        Self {
            num_threads: threads,
            num_requests: total,
            successful_requests: total - failed,
            failed_requests: failed,
            data_races_detected: false,
            meets_qa028: failed == 0,
        }
    }

    pub fn data_race_detected(threads: usize, requests: usize) -> Self {
        Self {
            num_threads: threads,
            num_requests: requests,
            successful_requests: 0,
            failed_requests: requests,
            data_races_detected: true,
            meets_qa028: false,
        }
    }
}

/// Thread-safe request counter for testing
pub struct AtomicRequestCounter {
    pub successful: std::sync::atomic::AtomicUsize,
    pub failed: std::sync::atomic::AtomicUsize,
}

impl AtomicRequestCounter {
    pub fn new() -> Self {
        Self {
            successful: std::sync::atomic::AtomicUsize::new(0),
            failed: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    pub fn record_success(&self) {
        self.successful
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    }

    pub fn record_failure(&self) {
        self.failed
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    }

    pub fn get_result(&self, threads: usize) -> ThreadSafetyResult {
        let successful = self.successful.load(std::sync::atomic::Ordering::SeqCst);
        let failed = self.failed.load(std::sync::atomic::Ordering::SeqCst);
        ThreadSafetyResult::with_failures(threads, successful + failed, failed)
    }
}

impl Default for AtomicRequestCounter {
    fn default() -> Self {
        Self::new()
    }
}

/// IMP-181a: Test thread safety result structure
#[test]
fn test_imp_181a_thread_safety_result() {
    let success = ThreadSafetyResult::success(4, 100);
    assert!(
        success.meets_qa028,
        "IMP-181a: Successful result should meet QA-028"
    );
    assert_eq!(
        success.successful_requests, 100,
        "IMP-181a: All requests should succeed"
    );

    let with_failures = ThreadSafetyResult::with_failures(4, 100, 5);
    assert!(
        !with_failures.meets_qa028,
        "IMP-181a: Failures should not meet QA-028"
    );
    assert_eq!(
        with_failures.failed_requests, 5,
        "IMP-181a: Should track 5 failures"
    );

    let data_race = ThreadSafetyResult::data_race_detected(4, 100);
    assert!(
        !data_race.meets_qa028,
        "IMP-181a: Data race should not meet QA-028"
    );
    assert!(
        data_race.data_races_detected,
        "IMP-181a: Should detect data race"
    );

    println!("\nIMP-181a: Thread Safety Results:");
    println!(
        "  Success: {}/{} -> meets_qa028={}",
        success.successful_requests, success.num_requests, success.meets_qa028
    );
    println!(
        "  Failures: {}/{} -> meets_qa028={}",
        with_failures.successful_requests, with_failures.num_requests, with_failures.meets_qa028
    );
    println!(
        "  Data race: detected={} -> meets_qa028={}",
        data_race.data_races_detected, data_race.meets_qa028
    );
}

/// IMP-181b: Test atomic request counter
#[test]
fn test_imp_181b_atomic_counter() {
    let counter = AtomicRequestCounter::new();

    // Simulate concurrent access
    for _ in 0..10 {
        counter.record_success();
    }
    for _ in 0..2 {
        counter.record_failure();
    }

    let result = counter.get_result(4);
    assert_eq!(
        result.successful_requests, 10,
        "IMP-181b: Should count 10 successes"
    );
    assert_eq!(
        result.failed_requests, 2,
        "IMP-181b: Should count 2 failures"
    );
    assert!(
        !result.meets_qa028,
        "IMP-181b: Failures should not meet QA-028"
    );

    println!("\nIMP-181b: Atomic Counter:");
    println!("  Successful: {}", result.successful_requests);
    println!("  Failed: {}", result.failed_requests);
    println!(
        "  QA-028: {}",
        if result.meets_qa028 { "PASS" } else { "FAIL" }
    );
}

/// IMP-181c: Test concurrent counter updates
#[test]
fn test_imp_181c_concurrent_updates() {
    use std::sync::Arc;
    use std::thread;

    let counter = Arc::new(AtomicRequestCounter::new());
    let num_threads = 4;
    let ops_per_thread = 100;

    let handles: Vec<_> = (0..num_threads)
        .map(|_| {
            let c = Arc::clone(&counter);
            thread::spawn(move || {
                for _ in 0..ops_per_thread {
                    c.record_success();
                }
            })
        })
        .collect();

    for h in handles {
        h.join().expect("Thread panicked");
    }

    let result = counter.get_result(num_threads);
    assert_eq!(
        result.successful_requests,
        num_threads * ops_per_thread,
        "IMP-181c: Should count all {} operations",
        num_threads * ops_per_thread
    );
    assert!(result.meets_qa028, "IMP-181c: No failures = QA-028 pass");

    println!("\nIMP-181c: Concurrent Updates:");
    println!(
        "  {} threads x {} ops = {} total",
        num_threads, ops_per_thread, result.successful_requests
    );
    println!("  Data races: {}", result.data_races_detected);
    println!(
        "  QA-028: {}",
        if result.meets_qa028 { "PASS" } else { "FAIL" }
    );
}

/// IMP-181d: Real-world concurrent requests
#[test]
#[ignore = "Requires running llama.cpp server on port 8082"]
fn test_imp_181d_realworld_concurrent() {
    use std::sync::Arc;
    use std::thread;

    let counter = Arc::new(AtomicRequestCounter::new());
    let num_threads = 4;

    let handles: Vec<_> = (0..num_threads)
        .map(|i| {
            let c = Arc::clone(&counter);
            thread::spawn(move || {
                let client = ModelHttpClient::with_timeout(30);
                let request = CompletionRequest {
                    model: "default".to_string(),
                    prompt: format!("Thread {}: Say hello", i),
                    max_tokens: 5,
                    temperature: Some(0.0),
                    stream: false,
                };

                match client.llamacpp_completion("http://127.0.0.1:8082", &request) {
                    Ok(_) => c.record_success(),
                    Err(_) => c.record_failure(),
                }
            })
        })
        .collect();

    for h in handles {
        h.join().expect("Thread panicked");
    }

    let result = counter.get_result(num_threads);

    println!("\nIMP-181d: Real-World Concurrent:");
    println!("  Threads: {}", num_threads);
    println!("  Successful: {}", result.successful_requests);
    println!("  Failed: {}", result.failed_requests);
    println!(
        "  QA-028: {}",
        if result.meets_qa028 { "PASS" } else { "FAIL" }
    );
}

// ================================================================================
// IMP-182: Deterministic Output (QA-029)
// Verify deterministic output with fixed seed
// ================================================================================

/// Determinism verification result
#[derive(Debug)]
pub struct DeterminismResult {
    pub seed: u64,
    pub num_runs: usize,
    pub outputs_identical: bool,
    pub hash_matches: bool,
    pub meets_qa029: bool,
}

impl DeterminismResult {
    pub fn deterministic(seed: u64, runs: usize) -> Self {
        Self {
            seed,
            num_runs: runs,
            outputs_identical: true,
            hash_matches: true,
            meets_qa029: true,
        }
    }

    pub fn non_deterministic(seed: u64, runs: usize, reason: &str) -> Self {
        let _ = reason;
        Self {
            seed,
            num_runs: runs,
            outputs_identical: false,
            hash_matches: false,
            meets_qa029: false,
        }
    }
}

/// Simple hash for determinism checking
pub fn simple_hash(s: &str) -> u64 {
    let mut hash: u64 = 5381;
    for b in s.bytes() {
        hash = hash.wrapping_mul(33).wrapping_add(u64::from(b));
    }
    hash
}

/// IMP-182a: Test determinism result structure
#[test]
fn test_imp_182a_determinism_result() {
    let det = DeterminismResult::deterministic(42, 5);
    assert!(
        det.outputs_identical,
        "IMP-182a: Deterministic should have identical outputs"
    );
    assert!(
        det.meets_qa029,
        "IMP-182a: Deterministic should meet QA-029"
    );

    let non_det = DeterminismResult::non_deterministic(42, 5, "Outputs differ");
    assert!(
        !non_det.outputs_identical,
        "IMP-182a: Non-deterministic should have different outputs"
    );
    assert!(
        !non_det.meets_qa029,
        "IMP-182a: Non-deterministic should not meet QA-029"
    );

    println!("\nIMP-182a: Determinism Results:");
    println!(
        "  Deterministic: seed={}, runs={}, meets_qa029={}",
        det.seed, det.num_runs, det.meets_qa029
    );
    println!(
        "  Non-deterministic: seed={}, runs={}, meets_qa029={}",
        non_det.seed, non_det.num_runs, non_det.meets_qa029
    );
}

/// IMP-182b: Test simple hash function
#[test]
fn test_imp_182b_simple_hash() {
    let s1 = "Hello, World!";
    let s2 = "Hello, World!";
    let s3 = "Hello, World?";

    let h1 = simple_hash(s1);
    let h2 = simple_hash(s2);
    let h3 = simple_hash(s3);

    assert_eq!(h1, h2, "IMP-182b: Identical strings should have same hash");
    assert_ne!(
        h1, h3,
        "IMP-182b: Different strings should have different hash"
    );

    println!("\nIMP-182b: Simple Hash:");
    println!("  '{}' -> {}", s1, h1);
    println!("  '{}' -> {}", s2, h2);
    println!("  '{}' -> {}", s3, h3);
}

/// IMP-182c: Test determinism verification
#[test]
fn test_imp_182c_determinism_check() {
    let outputs = vec![
        "The answer is 42".to_string(),
        "The answer is 42".to_string(),
        "The answer is 42".to_string(),
    ];

    let hashes: Vec<u64> = outputs.iter().map(|s| simple_hash(s)).collect();
    let all_same = hashes.windows(2).all(|w| w[0] == w[1]);

    let result = if all_same {
        DeterminismResult::deterministic(42, outputs.len())
    } else {
        DeterminismResult::non_deterministic(42, outputs.len(), "Hashes differ")
    };

    assert!(
        result.meets_qa029,
        "IMP-182c: Identical outputs should be deterministic"
    );

    // Test with different outputs
    let varied_outputs = vec![
        "The answer is 42".to_string(),
        "The answer is 43".to_string(),
    ];
    let varied_hashes: Vec<u64> = varied_outputs.iter().map(|s| simple_hash(s)).collect();
    let varied_same = varied_hashes.windows(2).all(|w| w[0] == w[1]);

    let varied_result = if varied_same {
        DeterminismResult::deterministic(42, varied_outputs.len())
    } else {
        DeterminismResult::non_deterministic(42, varied_outputs.len(), "Hashes differ")
    };

    assert!(
        !varied_result.meets_qa029,
        "IMP-182c: Different outputs should not be deterministic"
    );

    println!("\nIMP-182c: Determinism Check:");
    println!("  Same outputs: meets_qa029={}", result.meets_qa029);
    println!(
        "  Different outputs: meets_qa029={}",
        varied_result.meets_qa029
    );
}

include!("imp_182d.rs");
include!("imp_184d.rs");
include!("outliers_outlier.rs");
include!("imp_189d.rs");
