//! Falsification Framework for 10X Performance Claims
//!
//! Implements Popperian falsification protocol from the specification.
//! Every optimization claim must be falsifiable with specific, measurable criteria.
//!
//! ## Falsification Protocol (from spec)
//!
//! ```text
//! def falsify_optimization(name: str, impl: Callable) -> bool:
//!     baseline = benchmark(current_impl, iterations=1000)
//!     optimized = benchmark(impl, iterations=1000)
//!
//!     # Statistical significance: p < 0.01
//!     p_value = mann_whitney_u(baseline, optimized)
//!     if p_value > 0.01:
//!         return FALSIFIED("Not statistically significant")
//!
//!     # Minimum improvement: 10%
//!     speedup = median(baseline) / median(optimized)
//!     if speedup < 1.10:
//!         return FALSIFIED(f"Speedup {speedup:.2f}x < 1.10x threshold")
//!
//!     # Regression check: no metric worse by >5%
//!     for metric in [latency_p99, memory_rss, cpu_usage]:
//!         if metric(optimized) > metric(baseline) * 1.05:
//!             return FALSIFIED(f"{metric.name} regressed >5%")
//!
//!     return VERIFIED(speedup)
//! ```

use std::time::{Duration, Instant};

/// Result of a falsification test
#[derive(Debug, Clone)]
pub enum FalsificationResult {
    /// Claim verified with measured speedup
    Verified { speedup: f64, p_value: f64, samples: usize },
    /// Claim falsified with reason
    Falsified { reason: String, speedup: f64, p_value: f64 },
    /// Test could not be run
    Skipped { reason: String },
}

impl FalsificationResult {
    /// Check if the result is verified
    pub fn is_verified(&self) -> bool {
        matches!(self, FalsificationResult::Verified { .. })
    }

    /// Check if the result is falsified
    pub fn is_falsified(&self) -> bool {
        matches!(self, FalsificationResult::Falsified { .. })
    }

    /// Get the speedup if available
    pub fn speedup(&self) -> Option<f64> {
        match self {
            FalsificationResult::Verified { speedup, .. } => Some(*speedup),
            FalsificationResult::Falsified { speedup, .. } => Some(*speedup),
            FalsificationResult::Skipped { .. } => None,
        }
    }
}

/// A falsification test for a specific optimization
#[derive(Debug, Clone)]
pub struct FalsificationTest {
    /// Test ID (e.g., "PERF-005-21")
    pub id: String,
    /// Human-readable claim
    pub claim: String,
    /// Method to verify
    pub method: String,
    /// Pass threshold
    pub pass_threshold: String,
    /// Current status
    pub status: FalsificationResult,
}

impl FalsificationTest {
    /// Create a new pending test
    pub fn new(
        id: impl Into<String>,
        claim: impl Into<String>,
        method: impl Into<String>,
        pass_threshold: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            claim: claim.into(),
            method: method.into(),
            pass_threshold: pass_threshold.into(),
            status: FalsificationResult::Skipped { reason: "Not yet run".into() },
        }
    }

    /// Mark as verified
    pub fn verify(&mut self, speedup: f64, p_value: f64, samples: usize) {
        self.status = FalsificationResult::Verified { speedup, p_value, samples };
    }

    /// Mark as falsified
    pub fn falsify(&mut self, reason: impl Into<String>, speedup: f64, p_value: f64) {
        self.status = FalsificationResult::Falsified { reason: reason.into(), speedup, p_value };
    }
}

/// Falsification engine for running benchmark comparisons
pub struct Falsifier {
    /// Minimum number of iterations for statistical significance
    pub min_iterations: usize,
    /// Maximum p-value for statistical significance
    pub max_p_value: f64,
    /// Minimum speedup to claim improvement
    pub min_speedup: f64,
    /// Maximum regression allowed in any metric
    pub max_regression: f64,
}

impl Default for Falsifier {
    fn default() -> Self {
        Self {
            min_iterations: 1000,
            max_p_value: 0.01,
            min_speedup: 1.10,    // 10% improvement minimum
            max_regression: 0.05, // 5% regression maximum
        }
    }
}

impl Falsifier {
    /// Create a new falsifier with custom parameters
    pub fn new(
        min_iterations: usize,
        max_p_value: f64,
        min_speedup: f64,
        max_regression: f64,
    ) -> Self {
        Self { min_iterations, max_p_value, min_speedup, max_regression }
    }

    /// Run a falsification test comparing baseline to optimized
    pub fn falsify<F, G>(&self, baseline: F, optimized: G) -> FalsificationResult
    where
        F: Fn() -> Duration,
        G: Fn() -> Duration,
    {
        // Collect samples
        let mut baseline_samples: Vec<Duration> = Vec::with_capacity(self.min_iterations);
        let mut optimized_samples: Vec<Duration> = Vec::with_capacity(self.min_iterations);

        // Warmup
        for _ in 0..10 {
            let _ = baseline();
            let _ = optimized();
        }

        // Collect baseline
        for _ in 0..self.min_iterations {
            baseline_samples.push(baseline());
        }

        // Collect optimized
        for _ in 0..self.min_iterations {
            optimized_samples.push(optimized());
        }

        // Calculate medians
        let baseline_median = Self::median(&baseline_samples);
        let optimized_median = Self::median(&optimized_samples);

        // Calculate speedup
        let speedup = baseline_median.as_nanos() as f64 / optimized_median.as_nanos() as f64;

        // Calculate p-value using Mann-Whitney U test
        let p_value = self.mann_whitney_u(&baseline_samples, &optimized_samples);

        // Apply falsification criteria
        if p_value > self.max_p_value {
            return FalsificationResult::Falsified {
                reason: format!(
                    "Not statistically significant: p={:.4} > {:.4}",
                    p_value, self.max_p_value
                ),
                speedup,
                p_value,
            };
        }

        if speedup < self.min_speedup {
            return FalsificationResult::Falsified {
                reason: format!("Speedup {:.2}x < {:.2}x threshold", speedup, self.min_speedup),
                speedup,
                p_value,
            };
        }

        FalsificationResult::Verified { speedup, p_value, samples: self.min_iterations }
    }

    /// Calculate median of durations
    fn median(samples: &[Duration]) -> Duration {
        let mut sorted: Vec<Duration> = samples.to_vec();
        sorted.sort();
        let mid = sorted.len() / 2;
        if sorted.len() % 2 == 0 {
            (sorted[mid - 1] + sorted[mid]) / 2
        } else {
            sorted[mid]
        }
    }

    /// Mann-Whitney U test for two samples
    /// Returns approximate p-value
    fn mann_whitney_u(&self, a: &[Duration], b: &[Duration]) -> f64 {
        let n1 = a.len();
        let n2 = b.len();

        // Convert to nanoseconds for comparison
        let mut combined: Vec<(u128, bool)> = Vec::with_capacity(n1 + n2);
        for d in a {
            combined.push((d.as_nanos(), true)); // true = from a
        }
        for d in b {
            combined.push((d.as_nanos(), false)); // false = from b
        }

        // Rank all values
        combined.sort_by_key(|x| x.0);

        // Calculate rank sum for a
        let mut rank_sum_a: f64 = 0.0;
        for (rank, (_, is_a)) in combined.iter().enumerate() {
            if *is_a {
                rank_sum_a += (rank + 1) as f64;
            }
        }

        // Calculate U statistic
        let u1 = rank_sum_a - (n1 * (n1 + 1)) as f64 / 2.0;
        let u2 = (n1 * n2) as f64 - u1;
        let u = u1.min(u2);

        // Calculate mean and standard deviation under null hypothesis
        let mean_u = (n1 * n2) as f64 / 2.0;
        let std_u = ((n1 * n2 * (n1 + n2 + 1)) as f64 / 12.0).sqrt();

        // Calculate z-score
        let z = (u - mean_u) / std_u;

        // Convert to approximate p-value (two-tailed)
        // Using normal approximation for large samples
        Self::normal_cdf(-z.abs()) * 2.0
    }

    /// Standard normal CDF approximation
    fn normal_cdf(x: f64) -> f64 {
        // Abramowitz and Stegun approximation
        let t = 1.0 / (1.0 + 0.2316419 * x.abs());
        let d = 0.3989422804014327; // 1/sqrt(2*pi)
        let p = d
            * (-x * x / 2.0).exp()
            * (0.319381530 * t - 0.356563782 * t * t + 1.781477937 * t * t * t
                - 1.821255978 * t * t * t * t
                + 1.330274429 * t * t * t * t * t);
        if x > 0.0 {
            1.0 - p
        } else {
            p
        }
    }

    /// Run a quick benchmark and return duration
    pub fn benchmark<F>(mut func: F, iterations: usize) -> Duration
    where
        F: FnMut(),
    {
        let start = Instant::now();
        for _ in 0..iterations {
            func();
        }
        start.elapsed() / iterations as u32
    }
}

/// 100-Point Falsification Matrix
///
/// Tracks all verification points from the specification.
pub struct FalsificationMatrix {
    tests: Vec<FalsificationTest>,
}

impl FalsificationMatrix {
    /// Create a new matrix with all 100 test points
    pub fn new() -> Self {
        let tests = vec![
            // Section A: Baseline Measurements (Points 1-20)
            FalsificationTest::new(
                "A.01",
                "Baseline IOPS measured",
                "fio randread 4K QD=32",
                "Documented: 286K",
            ),
            FalsificationTest::new(
                "A.02",
                "Baseline throughput measured",
                "fio seqread 1M",
                "Documented: 2.1 GB/s",
            ),
            FalsificationTest::new(
                "A.03",
                "Baseline latency p99 measured",
                "fio latency",
                "Documented",
            ),
            FalsificationTest::new(
                "A.04",
                "Kernel zram IOPS measured",
                "fio on /dev/zram0",
                "~1.5M IOPS",
            ),
            FalsificationTest::new(
                "A.05",
                "Context switches counted",
                "perf stat -e context-switches",
                "Documented",
            ),
            FalsificationTest::new(
                "A.06",
                "Syscalls per I/O counted",
                "strace -c",
                "Documented: 1",
            ),
            FalsificationTest::new(
                "A.07",
                "Memory copies counted",
                "perf record memcpy",
                "Documented: 2-3",
            ),
            FalsificationTest::new(
                "A.08",
                "TLB misses measured",
                "perf stat -e dTLB-load-misses",
                "Documented",
            ),
            FalsificationTest::new(
                "A.09",
                "Cache misses measured",
                "perf stat -e LLC-load-misses",
                "Documented",
            ),
            FalsificationTest::new("A.10", "NUMA locality verified", "numastat", "Documented"),
            FalsificationTest::new("A.11", "CPU utilization profiled", "perf top", "Documented"),
            FalsificationTest::new("A.12", "Lock contention profiled", "perf lock", "Documented"),
            FalsificationTest::new("A.13", "io_uring submission rate", "bpftrace", "Documented"),
            FalsificationTest::new("A.14", "io_uring completion rate", "bpftrace", "Documented"),
            FalsificationTest::new(
                "A.15",
                "Compression CPU cycles",
                "perf stat -e cycles",
                "Documented",
            ),
            FalsificationTest::new(
                "A.16",
                "I/O path CPU cycles",
                "perf stat -e cycles",
                "Documented",
            ),
            FalsificationTest::new("A.17", "Memory bandwidth used", "pcm-memory", "Documented"),
            FalsificationTest::new("A.18", "PCIe bandwidth (if GPU)", "nvidia-smi", "N/A"),
            FalsificationTest::new("A.19", "Kernel CPU time", "/proc/stat", "Documented"),
            FalsificationTest::new("A.20", "Userspace CPU time", "/proc/stat", "Documented"),
            // Section B: PERF-005 Registered Buffers (Points 21-30)
            FalsificationTest::new("B.21", "Buffer registration succeeds", "Unit test", "No error"),
            FalsificationTest::new(
                "B.22",
                "Registered buffer used in SQE",
                "strace analysis",
                "IOSQE_BUFFER_SELECT set",
            ),
            FalsificationTest::new(
                "B.23",
                "Per-I/O buffer setup eliminated",
                "perf record",
                "No mmap per I/O",
            ),
            FalsificationTest::new("B.24", "TLB misses reduced", "perf stat", ">50% reduction"),
            FalsificationTest::new("B.25", "IOPS improved", "fio benchmark", ">1.5x baseline"),
            FalsificationTest::new(
                "B.26",
                "Latency p99 not regressed",
                "fio benchmark",
                "<1.1x baseline",
            ),
            FalsificationTest::new(
                "B.27",
                "Memory usage not increased",
                "/proc/meminfo",
                "<1.1x baseline",
            ),
            FalsificationTest::new("B.28", "Buffer reuse verified", "Custom tracing", "100% reuse"),
            FalsificationTest::new("B.29", "No buffer leaks", "Valgrind", "0 leaks"),
            FalsificationTest::new(
                "B.30",
                "Correctness maintained",
                "Data integrity test",
                "100% match",
            ),
            // Section C: PERF-006 Zero-Copy (Points 31-40)
            FalsificationTest::new(
                "C.31",
                "UBLK_F_SUPPORT_ZERO_COPY enabled",
                "Kernel check",
                "Flag set",
            ),
            FalsificationTest::new(
                "C.32",
                "Kernel buffer mapped",
                "/proc/pid/maps",
                "Mapping exists",
            ),
            FalsificationTest::new(
                "C.33",
                "memcpy eliminated",
                "perf record",
                "0 memcpy in hot path",
            ),
            FalsificationTest::new(
                "C.34",
                "In-place compression works",
                "Unit test",
                "Correct output",
            ),
            FalsificationTest::new("C.35", "Throughput improved", "fio benchmark", ">2x baseline"),
            FalsificationTest::new("C.36", "CPU usage reduced", "top", ">30% reduction"),
            FalsificationTest::new(
                "C.37",
                "Memory bandwidth reduced",
                "pcm-memory",
                ">40% reduction",
            ),
            FalsificationTest::new("C.38", "No data corruption", "Checksum test", "100% match"),
            FalsificationTest::new("C.39", "Concurrent access safe", "Stress test", "No races"),
            FalsificationTest::new(
                "C.40",
                "Error handling correct",
                "Fault injection",
                "Graceful recovery",
            ),
            // Section D: PERF-007 SQPOLL (Points 41-50)
            FalsificationTest::new("D.41", "SQPOLL thread created", "ps aux", "kworker visible"),
            FalsificationTest::new(
                "D.42",
                "Syscalls eliminated",
                "strace -c",
                "0 syscalls in steady state",
            ),
            FalsificationTest::new(
                "D.43",
                "Kernel thread CPU pinned",
                "/proc/pid/status",
                "Correct affinity",
            ),
            FalsificationTest::new(
                "D.44",
                "SQPOLL idle timeout works",
                "Power measurement",
                "Thread sleeps when idle",
            ),
            FalsificationTest::new(
                "D.45",
                "IOPS improved",
                "fio benchmark",
                ">1.5x over registered buffers",
            ),
            FalsificationTest::new(
                "D.46",
                "Latency improved",
                "fio benchmark",
                ">30% p99 reduction",
            ),
            FalsificationTest::new("D.47", "CPU efficiency improved", "IOPS/CPU-cycle", ">1.5x"),
            FalsificationTest::new(
                "D.48",
                "No starvation",
                "Long-running test",
                "Consistent throughput",
            ),
            FalsificationTest::new("D.49", "Graceful degradation", "Overload test", "No crash"),
            FalsificationTest::new("D.50", "Shutdown clean", "Resource check", "All released"),
            // Section E: PERF-008 Fixed Files (Points 51-60)
            FalsificationTest::new("E.51", "Files registered", "io_uring API", "Success return"),
            FalsificationTest::new(
                "E.52",
                "Fixed index used",
                "SQE inspection",
                "IOSQE_FIXED_FILE set",
            ),
            FalsificationTest::new(
                "E.53",
                "fd lookup eliminated",
                "perf record",
                "No fd table access",
            ),
            FalsificationTest::new("E.54", "IOPS improved", "fio benchmark", ">10% over SQPOLL"),
            FalsificationTest::new(
                "E.55",
                "Combined with SQPOLL",
                "Integration test",
                "Both active",
            ),
            FalsificationTest::new(
                "E.56",
                "Hot path optimized",
                "Flame graph",
                "<5% in fd handling",
            ),
            FalsificationTest::new("E.57", "Error on bad index", "Fault test", "Graceful error"),
            FalsificationTest::new("E.58", "Unregister works", "Cleanup test", "No leaks"),
            FalsificationTest::new("E.59", "Re-register works", "Restart test", "Correct behavior"),
            FalsificationTest::new("E.60", "Concurrent safe", "Stress test", "No races"),
            // Section F: PERF-009 Huge Pages (Points 61-70)
            FalsificationTest::new(
                "F.61",
                "Huge pages allocated",
                "/proc/meminfo",
                "HugePages_Free reduced",
            ),
            FalsificationTest::new(
                "F.62",
                "2MB pages used",
                "/proc/pid/smaps",
                "AnonHugePages > 0",
            ),
            FalsificationTest::new("F.63", "TLB misses reduced", "perf stat", ">90% reduction"),
            FalsificationTest::new("F.64", "Page faults reduced", "perf stat", ">50% reduction"),
            FalsificationTest::new("F.65", "Throughput improved", "fio benchmark", ">1.5x"),
            FalsificationTest::new(
                "F.66",
                "Memory overhead acceptable",
                "RSS measurement",
                "<1.1x",
            ),
            FalsificationTest::new("F.67", "Fallback to 4KB works", "Low-memory test", "Graceful"),
            FalsificationTest::new(
                "F.68",
                "Fragmentation handled",
                "Long-running test",
                "Stable performance",
            ),
            FalsificationTest::new("F.69", "NUMA-aware allocation", "numastat", "Local allocation"),
            FalsificationTest::new(
                "F.70",
                "Transparent HP disabled",
                "System check",
                "Explicit control",
            ),
            // Section G: PERF-010 NUMA Optimization (Points 71-80)
            FalsificationTest::new("G.71", "NUMA topology detected", "numactl -H", "Correct nodes"),
            FalsificationTest::new("G.72", "Memory bound to node", "numastat -p", ">99% local"),
            FalsificationTest::new("G.73", "Thread pinned to CPU", "taskset -p", "Correct mask"),
            FalsificationTest::new(
                "G.74",
                "Cross-NUMA eliminated",
                "perf stat numa",
                "0 remote access",
            ),
            FalsificationTest::new("G.75", "Latency improved", "fio benchmark", ">20% reduction"),
            FalsificationTest::new(
                "G.76",
                "Multi-socket scaling",
                "2-socket test",
                ">1.8x speedup",
            ),
            FalsificationTest::new("G.77", "Memory bandwidth local", "pcm-memory", ">95% local"),
            FalsificationTest::new(
                "G.78",
                "Interrupt affinity set",
                "/proc/interrupts",
                "Correct CPU",
            ),
            FalsificationTest::new("G.79", "Migration disabled", "perf sched", "No migrations"),
            FalsificationTest::new(
                "G.80",
                "Graceful on single-node",
                "Unit test",
                "Works correctly",
            ),
            // Section H: PERF-011 Lock-Free Multi-Queue (Points 81-90)
            FalsificationTest::new(
                "H.81",
                "Lock-free data structure",
                "Code review",
                "No mutexes in hot path",
            ),
            FalsificationTest::new(
                "H.82",
                "CAS operations used",
                "Assembly inspection",
                "lock cmpxchg",
            ),
            FalsificationTest::new("H.83", "ABA problem handled", "Stress test", "No corruption"),
            FalsificationTest::new(
                "H.84",
                "Memory ordering correct",
                "ThreadSanitizer",
                "No data races",
            ),
            FalsificationTest::new(
                "H.85",
                "Scalability linear",
                "1-8 queue test",
                ">0.9 efficiency",
            ),
            FalsificationTest::new("H.86", "IOPS @ 8 queues", "fio benchmark", ">2M"),
            FalsificationTest::new("H.87", "Contention eliminated", "perf lock", "0 contended"),
            FalsificationTest::new("H.88", "Cache line padding", "sizeof check", "64-byte aligned"),
            FalsificationTest::new(
                "H.89",
                "False sharing eliminated",
                "perf c2c",
                "No false sharing",
            ),
            FalsificationTest::new(
                "H.90",
                "Graceful single-queue",
                "Fallback test",
                "Works correctly",
            ),
            // Section I: PERF-012 Adaptive Batching (Points 91-100)
            FalsificationTest::new(
                "I.91",
                "Batch size adapts",
                "Logging",
                "Size changes with load",
            ),
            FalsificationTest::new("I.92", "Latency target met", "fio benchmark", "p99 < 50us"),
            FalsificationTest::new(
                "I.93",
                "Throughput maintained",
                "fio benchmark",
                ">90% of fixed batch",
            ),
            FalsificationTest::new(
                "I.94",
                "EMA calculation correct",
                "Unit test",
                "Mathematical correctness",
            ),
            FalsificationTest::new(
                "I.95",
                "Convergence fast",
                "Step response test",
                "<100ms to adapt",
            ),
            FalsificationTest::new(
                "I.96",
                "Stability achieved",
                "Long-running test",
                "No oscillation",
            ),
            FalsificationTest::new(
                "I.97",
                "Min batch respected",
                "Edge test",
                "Never below minimum",
            ),
            FalsificationTest::new(
                "I.98",
                "Max batch respected",
                "Edge test",
                "Never above maximum",
            ),
            FalsificationTest::new(
                "I.99",
                "Mixed workload handled",
                "Realistic benchmark",
                "Good for both",
            ),
            FalsificationTest::new(
                "I.100",
                "10X ACHIEVED",
                "Full benchmark suite",
                "10X kernel zram",
            ),
        ];

        Self { tests }
    }

    /// Get all tests
    pub fn tests(&self) -> &[FalsificationTest] {
        &self.tests
    }

    /// Get mutable reference to a test by ID
    pub fn get_mut(&mut self, id: &str) -> Option<&mut FalsificationTest> {
        self.tests.iter_mut().find(|t| t.id == id)
    }

    /// Count verified tests
    pub fn verified_count(&self) -> usize {
        self.tests.iter().filter(|t| t.status.is_verified()).count()
    }

    /// Count falsified tests
    pub fn falsified_count(&self) -> usize {
        self.tests.iter().filter(|t| t.status.is_falsified()).count()
    }

    /// Get percentage complete
    pub fn completion_percentage(&self) -> f64 {
        let total = self.tests.len();
        let verified = self.verified_count();
        (verified as f64 / total as f64) * 100.0
    }

    /// Print summary report
    pub fn print_summary(&self) {
        println!("=== 100-Point Falsification Matrix ===");
        println!("Verified: {}/100", self.verified_count());
        println!("Falsified: {}/100", self.falsified_count());
        println!("Pending: {}/100", 100 - self.verified_count() - self.falsified_count());
        println!("Completion: {:.1}%", self.completion_percentage());
    }
}

impl Default for FalsificationMatrix {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_falsification_result_verified() {
        let result = FalsificationResult::Verified { speedup: 2.5, p_value: 0.001, samples: 1000 };
        assert!(result.is_verified());
        assert!(!result.is_falsified());
        assert_eq!(result.speedup(), Some(2.5));
    }

    #[test]
    fn test_falsification_result_falsified() {
        let result = FalsificationResult::Falsified {
            reason: "Too slow".into(),
            speedup: 1.05,
            p_value: 0.02,
        };
        assert!(!result.is_verified());
        assert!(result.is_falsified());
        assert_eq!(result.speedup(), Some(1.05));
    }

    #[test]
    fn test_falsification_result_skipped() {
        let result = FalsificationResult::Skipped { reason: "Not yet run".into() };
        assert!(!result.is_verified());
        assert!(!result.is_falsified());
        assert_eq!(result.speedup(), None);
    }

    #[test]
    fn test_falsifier_default() {
        let f = Falsifier::default();
        assert_eq!(f.min_iterations, 1000);
        assert_eq!(f.max_p_value, 0.01);
        assert_eq!(f.min_speedup, 1.10);
        assert_eq!(f.max_regression, 0.05);
    }

    #[test]
    fn test_falsifier_median_odd() {
        let samples =
            vec![Duration::from_nanos(100), Duration::from_nanos(200), Duration::from_nanos(300)];
        let median = Falsifier::median(&samples);
        assert_eq!(median, Duration::from_nanos(200));
    }

    #[test]
    fn test_falsifier_median_even() {
        let samples = vec![
            Duration::from_nanos(100),
            Duration::from_nanos(200),
            Duration::from_nanos(300),
            Duration::from_nanos(400),
        ];
        let median = Falsifier::median(&samples);
        assert_eq!(median, Duration::from_nanos(250));
    }

    #[test]
    fn test_falsifier_benchmark() {
        let mut counter = 0u64;
        let duration = Falsifier::benchmark(
            || {
                counter += 1;
            },
            100,
        );
        // Should complete quickly
        assert!(duration < Duration::from_millis(1));
    }

    #[test]
    fn test_falsification_matrix_has_100_tests() {
        let matrix = FalsificationMatrix::new();
        assert_eq!(matrix.tests().len(), 100);
    }

    #[test]
    fn test_falsification_matrix_initial_state() {
        let matrix = FalsificationMatrix::new();
        assert_eq!(matrix.verified_count(), 0);
        assert_eq!(matrix.falsified_count(), 0);
        assert_eq!(matrix.completion_percentage(), 0.0);
    }

    #[test]
    fn test_falsification_matrix_get_mut() {
        let mut matrix = FalsificationMatrix::new();
        let test = matrix.get_mut("B.21").unwrap();
        assert_eq!(test.claim, "Buffer registration succeeds");
        test.verify(1.5, 0.001, 1000);
        assert!(test.status.is_verified());
    }

    #[test]
    fn test_falsification_matrix_verified_count() {
        let mut matrix = FalsificationMatrix::new();
        matrix.get_mut("B.21").unwrap().verify(1.5, 0.001, 1000);
        matrix.get_mut("B.22").unwrap().verify(1.6, 0.002, 1000);
        assert_eq!(matrix.verified_count(), 2);
        assert_eq!(matrix.completion_percentage(), 2.0);
    }

    #[test]
    fn test_falsification_test_new() {
        let test = FalsificationTest::new("TEST.01", "Test claim", "test method", "threshold");
        assert_eq!(test.id, "TEST.01");
        assert_eq!(test.claim, "Test claim");
        assert!(!test.status.is_verified());
    }

    #[test]
    fn test_normal_cdf_at_zero() {
        // CDF at 0 should be 0.5
        let cdf = Falsifier::normal_cdf(0.0);
        assert!((cdf - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_normal_cdf_symmetry() {
        // CDF(-x) + CDF(x) should equal 1
        let x = 1.5;
        let sum = Falsifier::normal_cdf(-x) + Falsifier::normal_cdf(x);
        assert!((sum - 1.0).abs() < 0.01);
    }

    // ========================================================================
    // Falsification Protocol Verification
    // ========================================================================

    #[test]
    fn test_falsify_detects_no_improvement() {
        let falsifier = Falsifier::new(100, 0.05, 1.10, 0.05);

        // Both functions take same time - should be falsified
        let result =
            falsifier.falsify(|| Duration::from_nanos(1000), || Duration::from_nanos(1000));

        // Should be falsified because speedup < 1.10
        assert!(result.is_falsified());
        if let FalsificationResult::Falsified { speedup, .. } = result {
            assert!((speedup - 1.0).abs() < 0.1);
        }
    }

    #[test]
    fn test_falsify_detects_clear_improvement() {
        let falsifier = Falsifier::new(100, 0.05, 1.10, 0.05);

        // Optimized is 2x faster - should be verified
        let result =
            falsifier.falsify(|| Duration::from_nanos(2000), || Duration::from_nanos(1000));

        assert!(result.is_verified());
        if let FalsificationResult::Verified { speedup, .. } = result {
            assert!(speedup > 1.5);
        }
    }
}
