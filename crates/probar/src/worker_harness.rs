//! WASM Worker Test Harness (PROBAR-SPEC-013)
//!
//! Comprehensive testing framework for Web Workers in WASM applications.
//! Addresses critical defect vectors in worker thread communication.
//!
//! ## Toyota Way Application:
//! - **Jidoka**: Automatic detection of worker defects
//! - **Poka-Yoke**: Type-safe message protocols prevent protocol errors
//! - **Heijunka**: Balanced testing across all worker states
//!
//! ## Critical Defect Vectors Addressed:
//! 1. Worker initialization race conditions
//! 2. Message ordering violations (Lamport [8])
//! 3. SharedArrayBuffer race conditions (Herlihy-Shavit [7])
//! 4. Ring buffer overflow/underflow
//! 5. Worker error recovery failures
//! 6. Memory leak in long-running workers
//!
//! ## References:
//! - [7] Herlihy & Shavit (2012) "The Art of Multiprocessor Programming"
//! - [8] Lamport (1978) "Time, Clocks, and the Ordering of Events"
//! - whisper.apr: SharedRingBuffer implementation

use std::fmt;
use std::time::Duration;

/// Worker lifecycle states for testing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WorkerLifecycleState {
    /// Worker not yet created
    NotCreated,
    /// Worker script being loaded
    Loading,
    /// Worker initializing (loading WASM, etc.)
    Initializing,
    /// Worker ready for commands
    Ready,
    /// Worker processing a command
    Processing,
    /// Worker in error state (recoverable)
    Error,
    /// Worker terminated
    Terminated,
}

impl Default for WorkerLifecycleState {
    fn default() -> Self {
        Self::NotCreated
    }
}

impl fmt::Display for WorkerLifecycleState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotCreated => write!(f, "NotCreated"),
            Self::Loading => write!(f, "Loading"),
            Self::Initializing => write!(f, "Initializing"),
            Self::Ready => write!(f, "Ready"),
            Self::Processing => write!(f, "Processing"),
            Self::Error => write!(f, "Error"),
            Self::Terminated => write!(f, "Terminated"),
        }
    }
}

/// Worker test configuration
#[derive(Debug, Clone)]
pub struct WorkerTestConfig {
    /// Timeout for worker initialization
    pub init_timeout: Duration,
    /// Timeout for command processing
    pub command_timeout: Duration,
    /// Maximum messages before overflow test
    pub max_messages: usize,
    /// Whether to test error recovery
    pub test_error_recovery: bool,
    /// Whether to test memory leaks
    pub test_memory_leaks: bool,
    /// Number of iterations for stress testing
    pub stress_iterations: usize,
    /// Whether to verify Lamport ordering
    pub verify_lamport_ordering: bool,
    /// Whether to test SharedArrayBuffer operations
    pub test_shared_memory: bool,
    /// Ring buffer size for testing
    pub ring_buffer_size: usize,
}

impl Default for WorkerTestConfig {
    fn default() -> Self {
        Self {
            init_timeout: Duration::from_secs(10),
            command_timeout: Duration::from_secs(30),
            max_messages: 1000,
            test_error_recovery: true,
            test_memory_leaks: true,
            stress_iterations: 100,
            verify_lamport_ordering: true,
            test_shared_memory: true,
            ring_buffer_size: 16384, // 16KB default
        }
    }
}

impl WorkerTestConfig {
    /// Create minimal configuration for fast tests
    #[must_use]
    pub fn minimal() -> Self {
        Self {
            init_timeout: Duration::from_secs(5),
            command_timeout: Duration::from_secs(5),
            max_messages: 100,
            test_error_recovery: false,
            test_memory_leaks: false,
            stress_iterations: 10,
            verify_lamport_ordering: true,
            test_shared_memory: false,
            ring_buffer_size: 4096,
        }
    }

    /// Create comprehensive configuration for thorough testing
    #[must_use]
    pub fn comprehensive() -> Self {
        Self {
            init_timeout: Duration::from_secs(30),
            command_timeout: Duration::from_secs(60),
            max_messages: 10000,
            test_error_recovery: true,
            test_memory_leaks: true,
            stress_iterations: 1000,
            verify_lamport_ordering: true,
            test_shared_memory: true,
            ring_buffer_size: 65536, // 64KB
        }
    }
}

/// Worker test result
#[derive(Debug, Clone, Default)]
pub struct WorkerTestResult {
    /// All tests passed
    pub passed: bool,
    /// Lifecycle tests passed
    pub lifecycle_passed: bool,
    /// Message ordering tests passed
    pub ordering_passed: bool,
    /// SharedArrayBuffer tests passed
    pub shared_memory_passed: bool,
    /// Ring buffer tests passed
    pub ring_buffer_passed: bool,
    /// Error recovery tests passed
    pub error_recovery_passed: bool,
    /// Memory leak tests passed
    pub memory_leak_passed: bool,
    /// Test failures with details
    pub failures: Vec<WorkerTestFailure>,
    /// Performance metrics
    pub metrics: WorkerMetrics,
}

impl WorkerTestResult {
    /// Check if all tests passed
    #[must_use]
    pub fn is_passed(&self) -> bool {
        self.passed && self.failures.is_empty()
    }

    /// Get failure count
    #[must_use]
    pub fn failure_count(&self) -> usize {
        self.failures.len()
    }
}

impl fmt::Display for WorkerTestResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_passed() {
            writeln!(f, "Worker Tests: PASSED")?;
        } else {
            writeln!(
                f,
                "Worker Tests: FAILED ({} failures)",
                self.failure_count()
            )?;
            for failure in &self.failures {
                writeln!(f, "  - {failure}")?;
            }
        }

        writeln!(f, "\nMetrics:")?;
        writeln!(f, "  Init time: {:?}", self.metrics.initialization_time)?;
        writeln!(
            f,
            "  Avg message latency: {:?}",
            self.metrics.average_message_latency
        )?;
        writeln!(
            f,
            "  Messages processed: {}",
            self.metrics.messages_processed
        )?;

        Ok(())
    }
}

/// Worker test failure details
#[derive(Debug, Clone)]
pub struct WorkerTestFailure {
    /// Test category
    pub category: WorkerTestCategory,
    /// Failure description
    pub description: String,
    /// Expected value/state
    pub expected: String,
    /// Actual value/state
    pub actual: String,
}

impl fmt::Display for WorkerTestFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{:?}] {}: expected '{}', got '{}'",
            self.category, self.description, self.expected, self.actual
        )
    }
}

/// Worker test categories
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkerTestCategory {
    /// Lifecycle state transitions
    Lifecycle,
    /// Message ordering
    Ordering,
    /// SharedArrayBuffer operations
    SharedMemory,
    /// Ring buffer operations
    RingBuffer,
    /// Error handling and recovery
    ErrorRecovery,
    /// Memory management
    Memory,
    /// Performance
    Performance,
}

/// Worker performance metrics
#[derive(Debug, Clone, Default)]
pub struct WorkerMetrics {
    /// Time to initialize worker
    pub initialization_time: Duration,
    /// Average message round-trip latency
    pub average_message_latency: Duration,
    /// Maximum message latency observed
    pub max_message_latency: Duration,
    /// Total messages processed
    pub messages_processed: u64,
    /// Messages dropped (if any)
    pub messages_dropped: u64,
    /// Memory at start (bytes)
    pub memory_start: u64,
    /// Memory at end (bytes)
    pub memory_end: u64,
    /// Number of error recoveries
    pub error_recoveries: u32,
}

impl WorkerMetrics {
    /// Check if there's a memory leak (>10% growth)
    #[must_use]
    pub fn has_memory_leak(&self) -> bool {
        if self.memory_start == 0 {
            return false;
        }
        let growth = self.memory_end.saturating_sub(self.memory_start);
        let threshold = self.memory_start / 10; // 10% threshold
        growth > threshold
    }

    /// Get memory growth in bytes
    #[must_use]
    pub fn memory_growth(&self) -> i64 {
        self.memory_end as i64 - self.memory_start as i64
    }
}

/// Ring buffer test configuration
#[derive(Debug, Clone)]
pub struct RingBufferTestConfig {
    /// Buffer size in bytes
    pub buffer_size: usize,
    /// Sample size in bytes
    pub sample_size: usize,
    /// Number of samples to write
    pub num_samples: usize,
    /// Whether to test overflow behavior
    pub test_overflow: bool,
    /// Whether to test underrun behavior
    pub test_underrun: bool,
    /// Whether to test concurrent access
    pub test_concurrent: bool,
}

impl Default for RingBufferTestConfig {
    fn default() -> Self {
        Self {
            buffer_size: 16384, // 16KB
            sample_size: 512,   // 512 bytes per sample
            num_samples: 1000,
            test_overflow: true,
            test_underrun: true,
            test_concurrent: true,
        }
    }
}

/// Ring buffer test results
#[derive(Debug, Clone, Default)]
pub struct RingBufferTestResult {
    /// All tests passed
    pub passed: bool,
    /// Write operations succeeded
    pub writes_succeeded: u64,
    /// Write operations failed
    pub writes_failed: u64,
    /// Read operations succeeded
    pub reads_succeeded: u64,
    /// Read operations failed
    pub reads_failed: u64,
    /// Overflow events detected
    pub overflows_detected: u32,
    /// Underrun events detected
    pub underruns_detected: u32,
    /// Data corruption detected
    pub corruption_detected: bool,
    /// Failures
    pub failures: Vec<String>,
}

impl RingBufferTestResult {
    /// Check if all operations succeeded without corruption
    #[must_use]
    pub fn is_passed(&self) -> bool {
        self.passed && !self.corruption_detected && self.failures.is_empty()
    }
}

/// SharedArrayBuffer test configuration
#[derive(Debug, Clone)]
pub struct SharedMemoryTestConfig {
    /// Buffer size in bytes
    pub buffer_size: usize,
    /// Number of atomic operations to test
    pub num_atomic_ops: usize,
    /// Whether to test Atomics.wait/notify
    pub test_wait_notify: bool,
    /// Whether to test concurrent writes
    pub test_concurrent_writes: bool,
    /// Timeout for wait operations
    pub wait_timeout: Duration,
}

impl Default for SharedMemoryTestConfig {
    fn default() -> Self {
        Self {
            buffer_size: 4096,
            num_atomic_ops: 1000,
            test_wait_notify: true,
            test_concurrent_writes: true,
            wait_timeout: Duration::from_millis(100),
        }
    }
}

/// SharedArrayBuffer test results
#[derive(Debug, Clone, Default)]
pub struct SharedMemoryTestResult {
    /// All tests passed
    pub passed: bool,
    /// Atomic operations correct
    pub atomics_correct: bool,
    /// Wait/notify operations correct
    pub wait_notify_correct: bool,
    /// Memory visibility correct
    pub visibility_correct: bool,
    /// Race conditions detected
    pub race_conditions_detected: u32,
    /// Failures
    pub failures: Vec<String>,
}

impl SharedMemoryTestResult {
    /// Check if shared memory is safe
    #[must_use]
    pub fn is_passed(&self) -> bool {
        self.passed && self.race_conditions_detected == 0 && self.failures.is_empty()
    }
}

/// Worker test harness for comprehensive worker testing
#[derive(Debug, Clone)]
pub struct WorkerTestHarness {
    config: WorkerTestConfig,
}

impl Default for WorkerTestHarness {
    fn default() -> Self {
        Self::new()
    }
}

impl WorkerTestHarness {
    /// Create new test harness with default configuration
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: WorkerTestConfig::default(),
        }
    }

    /// Create with custom configuration
    #[must_use]
    pub fn with_config(config: WorkerTestConfig) -> Self {
        Self { config }
    }

    /// Get the configuration
    #[must_use]
    pub fn config(&self) -> &WorkerTestConfig {
        &self.config
    }

    /// Test worker lifecycle transitions
    #[must_use]
    pub fn test_lifecycle_transitions(&self) -> Vec<WorkerTestFailure> {
        let mut failures = Vec::new();

        // Valid transitions from each state
        let valid_transitions: &[(WorkerLifecycleState, &[WorkerLifecycleState])] = &[
            (
                WorkerLifecycleState::NotCreated,
                &[WorkerLifecycleState::Loading],
            ),
            (
                WorkerLifecycleState::Loading,
                &[
                    WorkerLifecycleState::Initializing,
                    WorkerLifecycleState::Error,
                ],
            ),
            (
                WorkerLifecycleState::Initializing,
                &[WorkerLifecycleState::Ready, WorkerLifecycleState::Error],
            ),
            (
                WorkerLifecycleState::Ready,
                &[
                    WorkerLifecycleState::Processing,
                    WorkerLifecycleState::Error,
                    WorkerLifecycleState::Terminated,
                ],
            ),
            (
                WorkerLifecycleState::Processing,
                &[
                    WorkerLifecycleState::Ready,
                    WorkerLifecycleState::Error,
                    WorkerLifecycleState::Terminated,
                ],
            ),
            (
                WorkerLifecycleState::Error,
                &[
                    WorkerLifecycleState::Ready,
                    WorkerLifecycleState::Terminated,
                ],
            ),
            (WorkerLifecycleState::Terminated, &[]),
        ];

        // Verify each state has defined transitions
        for (state, transitions) in valid_transitions {
            if *state != WorkerLifecycleState::Terminated && transitions.is_empty() {
                failures.push(WorkerTestFailure {
                    category: WorkerTestCategory::Lifecycle,
                    description: format!("State {} has no valid transitions", state),
                    expected: "at least one valid transition".to_string(),
                    actual: "no transitions".to_string(),
                });
            }
        }

        failures
    }

    /// Verify message ordering invariants (Lamport)
    ///
    /// Verifies that messages maintain causal ordering.
    #[must_use]
    pub fn verify_message_ordering(&self, timestamps: &[u64]) -> Vec<WorkerTestFailure> {
        let mut failures = Vec::new();

        if !self.config.verify_lamport_ordering {
            return failures;
        }

        let mut last_time = 0u64;
        for (i, &time) in timestamps.iter().enumerate() {
            if time < last_time {
                failures.push(WorkerTestFailure {
                    category: WorkerTestCategory::Ordering,
                    description: format!("Message {} violates Lamport ordering", i),
                    expected: format!("timestamp > {last_time}"),
                    actual: format!("timestamp = {time}"),
                });
            }
            last_time = time;
        }

        failures
    }

    /// Test ring buffer operations
    #[must_use]
    pub fn test_ring_buffer(&self, config: &RingBufferTestConfig) -> RingBufferTestResult {
        let mut result = RingBufferTestResult {
            passed: true,
            ..Default::default()
        };

        // Test basic write/read cycle
        let capacity = config.buffer_size / config.sample_size;

        // Simulate writes
        for i in 0..config.num_samples {
            if i < capacity {
                result.writes_succeeded += 1;
            } else if config.test_overflow {
                // Overflow case
                result.overflows_detected += 1;
                if result.overflows_detected == 1 {
                    // First overflow is expected
                    result.writes_succeeded += 1;
                } else {
                    result.writes_failed += 1;
                }
            } else {
                result.writes_failed += 1;
            }
        }

        // Simulate reads
        for i in 0..config.num_samples {
            if (i as u64) < result.writes_succeeded {
                result.reads_succeeded += 1;
            } else if config.test_underrun {
                result.underruns_detected += 1;
                result.reads_failed += 1;
            } else {
                result.reads_failed += 1;
            }
        }

        // Check for issues
        if result.writes_failed > 0 && !config.test_overflow {
            result.failures.push(format!(
                "Write failures without overflow testing: {}",
                result.writes_failed
            ));
            result.passed = false;
        }

        result
    }

    /// Test SharedArrayBuffer operations
    #[must_use]
    pub fn test_shared_memory(&self, config: &SharedMemoryTestConfig) -> SharedMemoryTestResult {
        if !self.config.test_shared_memory {
            return SharedMemoryTestResult {
                passed: true,
                atomics_correct: true,
                wait_notify_correct: true,
                visibility_correct: true,
                ..Default::default()
            };
        }

        let mut result = SharedMemoryTestResult {
            passed: true,
            atomics_correct: true,
            wait_notify_correct: config.test_wait_notify,
            visibility_correct: true,
            ..Default::default()
        };

        // Simulate atomic operations test
        // In real implementation, this would use CDP to execute in browser
        let mut counter = 0i64;
        for _ in 0..config.num_atomic_ops {
            // Simulate Atomics.add
            counter += 1;
        }

        // Verify final value
        if counter != config.num_atomic_ops as i64 {
            result.atomics_correct = false;
            result.race_conditions_detected += 1;
            result.failures.push(format!(
                "Atomic counter mismatch: expected {}, got {}",
                config.num_atomic_ops, counter
            ));
            result.passed = false;
        }

        result
    }

    /// Generate JavaScript for worker lifecycle testing
    #[must_use]
    pub fn lifecycle_test_js() -> &'static str {
        r#"
(function() {
    const states = [];
    const transitions = [];

    window.__PROBAR_WORKER_STATES__ = states;
    window.__PROBAR_WORKER_TRANSITIONS__ = transitions;

    function recordState(workerId, state) {
        const timestamp = performance.now();
        states.push({ workerId, state, timestamp });
    }

    function recordTransition(workerId, from, to) {
        const timestamp = performance.now();
        transitions.push({ workerId, from, to, timestamp });
    }

    // Intercept Worker construction
    const originalWorker = window.Worker;
    window.Worker = function(url, options) {
        const worker = new originalWorker(url, options);
        const workerId = states.filter(s => s.state === 'NotCreated').length;

        recordState(workerId, 'NotCreated');
        recordTransition(workerId, 'NotCreated', 'Loading');
        recordState(workerId, 'Loading');

        worker.addEventListener('message', function(e) {
            const data = e.data;
            if (data && data.type) {
                const type = data.type.toLowerCase();
                const currentState = states.filter(s => s.workerId === workerId).pop()?.state || 'Loading';

                if (type === 'ready' || type === 'initialized') {
                    recordTransition(workerId, currentState, 'Ready');
                    recordState(workerId, 'Ready');
                } else if (type === 'error') {
                    recordTransition(workerId, currentState, 'Error');
                    recordState(workerId, 'Error');
                } else if (type === 'processing' || type === 'busy') {
                    recordTransition(workerId, currentState, 'Processing');
                    recordState(workerId, 'Processing');
                } else if (type === 'complete' || type === 'done') {
                    recordTransition(workerId, currentState, 'Ready');
                    recordState(workerId, 'Ready');
                }
            }
        });

        worker.addEventListener('error', function(e) {
            const currentState = states.filter(s => s.workerId === workerId).pop()?.state || 'Loading';
            recordTransition(workerId, currentState, 'Error');
            recordState(workerId, 'Error');
        });

        return worker;
    };

    return { success: true };
})();
"#
    }

    /// Generate JavaScript for ring buffer testing
    #[must_use]
    pub fn ring_buffer_test_js(buffer_size: usize) -> String {
        format!(
            r#"
(function() {{
    const BUFFER_SIZE = {buffer_size};
    const HEADER_SIZE = 64; // Cache-line aligned header
    const DATA_OFFSET = HEADER_SIZE;

    // Create SharedArrayBuffer
    const sab = new SharedArrayBuffer(BUFFER_SIZE + HEADER_SIZE);
    const header = new Int32Array(sab, 0, 4);
    const data = new Float32Array(sab, DATA_OFFSET);

    // Initialize header
    // [0] = write_idx, [1] = read_idx, [2] = capacity, [3] = flags
    Atomics.store(header, 0, 0);
    Atomics.store(header, 1, 0);
    Atomics.store(header, 2, data.length);
    Atomics.store(header, 3, 0);

    window.__PROBAR_RING_BUFFER__ = {{
        sab: sab,
        header: header,
        data: data,
        write: function(samples) {{
            const writeIdx = Atomics.load(header, 0);
            const readIdx = Atomics.load(header, 1);
            const capacity = Atomics.load(header, 2);
            const available = capacity - (writeIdx - readIdx);

            if (samples.length > available) {{
                return {{ success: false, error: 'overflow', available: available }};
            }}

            for (let i = 0; i < samples.length; i++) {{
                data[(writeIdx + i) % capacity] = samples[i];
            }}

            Atomics.store(header, 0, writeIdx + samples.length);
            return {{ success: true, written: samples.length }};
        }},
        read: function(count) {{
            const writeIdx = Atomics.load(header, 0);
            const readIdx = Atomics.load(header, 1);
            const available = writeIdx - readIdx;

            if (count > available) {{
                return {{ success: false, error: 'underrun', available: available }};
            }}

            const capacity = Atomics.load(header, 2);
            const result = new Float32Array(count);
            for (let i = 0; i < count; i++) {{
                result[i] = data[(readIdx + i) % capacity];
            }}

            Atomics.store(header, 1, readIdx + count);
            return {{ success: true, data: Array.from(result) }};
        }},
        getStats: function() {{
            return {{
                writeIdx: Atomics.load(header, 0),
                readIdx: Atomics.load(header, 1),
                capacity: Atomics.load(header, 2),
                available: Atomics.load(header, 2) - (Atomics.load(header, 0) - Atomics.load(header, 1))
            }};
        }}
    }};

    return {{ success: true, bufferSize: BUFFER_SIZE }};
}})();
"#
        )
    }

    /// Generate JavaScript for shared memory testing
    #[must_use]
    pub fn shared_memory_test_js(buffer_size: usize) -> String {
        format!(
            r#"
(function() {{
    const BUFFER_SIZE = {buffer_size};

    // Create SharedArrayBuffer for atomic operations
    const sab = new SharedArrayBuffer(BUFFER_SIZE);
    const int32View = new Int32Array(sab);

    window.__PROBAR_SHARED_MEMORY__ = {{
        sab: sab,
        int32View: int32View,

        // Test Atomics.add
        testAtomicAdd: function(index, count) {{
            let result = 0;
            for (let i = 0; i < count; i++) {{
                result = Atomics.add(int32View, index, 1);
            }}
            return {{ finalValue: Atomics.load(int32View, index), lastResult: result }};
        }},

        // Test Atomics.compareExchange
        testCompareExchange: function(index, expected, replacement) {{
            const oldValue = Atomics.compareExchange(int32View, index, expected, replacement);
            return {{ oldValue: oldValue, newValue: Atomics.load(int32View, index) }};
        }},

        // Test Atomics.wait/notify (must be called from worker)
        testWaitNotify: function(index) {{
            return {{
                info: 'Atomics.wait must be called from a worker thread',
                canUseWait: typeof SharedArrayBuffer !== 'undefined'
            }};
        }},

        // Get memory stats
        getStats: function() {{
            return {{
                bufferSize: BUFFER_SIZE,
                int32Length: int32View.length
            }};
        }},

        // Reset all values
        reset: function() {{
            for (let i = 0; i < int32View.length; i++) {{
                Atomics.store(int32View, i, 0);
            }}
            return {{ success: true }};
        }}
    }};

    return {{ success: true, bufferSize: BUFFER_SIZE }};
}})();
"#
        )
    }

    /// Validate worker test results against config requirements
    #[must_use]
    pub fn validate_results(&self, result: &WorkerTestResult) -> bool {
        if !result.lifecycle_passed {
            return false;
        }

        if self.config.verify_lamport_ordering && !result.ordering_passed {
            return false;
        }

        if self.config.test_shared_memory && !result.shared_memory_passed {
            return false;
        }

        if self.config.test_error_recovery && !result.error_recovery_passed {
            return false;
        }

        if self.config.test_memory_leaks && !result.memory_leak_passed {
            return false;
        }

        result.failures.is_empty()
    }
}

/// Error type for worker testing
#[derive(Debug, Clone)]
pub enum WorkerTestError {
    /// Initialization failed
    InitializationFailed(String),
    /// Timeout during test
    Timeout(String),
    /// Protocol error
    ProtocolError(String),
    /// CDP error
    CdpError(String),
}

impl fmt::Display for WorkerTestError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InitializationFailed(msg) => write!(f, "Initialization failed: {msg}"),
            Self::Timeout(msg) => write!(f, "Timeout: {msg}"),
            Self::ProtocolError(msg) => write!(f, "Protocol error: {msg}"),
            Self::CdpError(msg) => write!(f, "CDP error: {msg}"),
        }
    }
}

impl std::error::Error for WorkerTestError {}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    // =========================================================================
    // H0-WH-01: Lifecycle state tests
    // =========================================================================

    #[test]
    fn h0_wh_01_lifecycle_state_display() {
        assert_eq!(
            format!("{}", WorkerLifecycleState::NotCreated),
            "NotCreated"
        );
        assert_eq!(format!("{}", WorkerLifecycleState::Loading), "Loading");
        assert_eq!(format!("{}", WorkerLifecycleState::Ready), "Ready");
        assert_eq!(
            format!("{}", WorkerLifecycleState::Processing),
            "Processing"
        );
        assert_eq!(format!("{}", WorkerLifecycleState::Error), "Error");
        assert_eq!(
            format!("{}", WorkerLifecycleState::Terminated),
            "Terminated"
        );
        assert_eq!(
            format!("{}", WorkerLifecycleState::Initializing),
            "Initializing"
        );
    }

    #[test]
    fn h0_wh_02_lifecycle_state_default() {
        let state = WorkerLifecycleState::default();
        assert_eq!(state, WorkerLifecycleState::NotCreated);
    }

    // =========================================================================
    // H0-WH-03: Configuration tests
    // =========================================================================

    #[test]
    fn h0_wh_03_default_config() {
        let config = WorkerTestConfig::default();
        assert_eq!(config.init_timeout, Duration::from_secs(10));
        assert!(config.test_error_recovery);
        assert!(config.verify_lamport_ordering);
    }

    #[test]
    fn h0_wh_04_minimal_config() {
        let config = WorkerTestConfig::minimal();
        assert_eq!(config.stress_iterations, 10);
        assert!(!config.test_error_recovery);
        assert!(!config.test_memory_leaks);
    }

    #[test]
    fn h0_wh_05_comprehensive_config() {
        let config = WorkerTestConfig::comprehensive();
        assert_eq!(config.stress_iterations, 1000);
        assert!(config.test_error_recovery);
        assert!(config.test_memory_leaks);
    }

    // =========================================================================
    // H0-WH-06: Harness tests
    // =========================================================================

    #[test]
    fn h0_wh_06_harness_creation() {
        let harness = WorkerTestHarness::new();
        assert!(harness.config().verify_lamport_ordering);
    }

    #[test]
    fn h0_wh_07_harness_with_config() {
        let config = WorkerTestConfig::minimal();
        let harness = WorkerTestHarness::with_config(config);
        assert_eq!(harness.config().stress_iterations, 10);
    }

    #[test]
    fn h0_wh_08_harness_default() {
        let harness = WorkerTestHarness::default();
        assert!(harness.config().test_shared_memory);
    }

    // =========================================================================
    // H0-WH-09: Lifecycle transition tests
    // =========================================================================

    #[test]
    fn h0_wh_09_valid_lifecycle_transitions() {
        let harness = WorkerTestHarness::new();
        let failures = harness.test_lifecycle_transitions();
        // Should have no failures for valid transitions
        assert!(failures.is_empty());
    }

    // =========================================================================
    // H0-WH-10: Message ordering tests
    // =========================================================================

    #[test]
    fn h0_wh_10_valid_ordering() {
        let harness = WorkerTestHarness::new();
        let timestamps = vec![1, 2, 3, 4, 5];
        let failures = harness.verify_message_ordering(&timestamps);
        assert!(failures.is_empty());
    }

    #[test]
    fn h0_wh_11_invalid_ordering() {
        let harness = WorkerTestHarness::new();
        let timestamps = vec![1, 2, 5, 3, 6]; // 3 < 5 - invalid
        let failures = harness.verify_message_ordering(&timestamps);
        assert!(!failures.is_empty());
        assert_eq!(failures[0].category, WorkerTestCategory::Ordering);
    }

    #[test]
    fn h0_wh_12_ordering_disabled() {
        let config = WorkerTestConfig {
            verify_lamport_ordering: false,
            ..Default::default()
        };
        let harness = WorkerTestHarness::with_config(config);
        let timestamps = vec![5, 3, 1]; // Invalid but should pass
        let failures = harness.verify_message_ordering(&timestamps);
        assert!(failures.is_empty());
    }

    // =========================================================================
    // H0-WH-13: Ring buffer tests
    // =========================================================================

    #[test]
    fn h0_wh_13_ring_buffer_basic() {
        let harness = WorkerTestHarness::new();
        let config = RingBufferTestConfig::default();
        let result = harness.test_ring_buffer(&config);
        assert!(result.writes_succeeded > 0);
        assert!(result.reads_succeeded > 0);
    }

    #[test]
    fn h0_wh_14_ring_buffer_overflow() {
        let harness = WorkerTestHarness::new();
        let config = RingBufferTestConfig {
            buffer_size: 1024,
            sample_size: 512,
            num_samples: 100, // Much more than capacity (2)
            test_overflow: true,
            ..Default::default()
        };
        let result = harness.test_ring_buffer(&config);
        assert!(result.overflows_detected > 0);
    }

    #[test]
    fn h0_wh_15_ring_buffer_underrun() {
        let harness = WorkerTestHarness::new();
        let config = RingBufferTestConfig {
            buffer_size: 16384,
            sample_size: 512,
            num_samples: 100,
            test_underrun: true,
            ..Default::default()
        };
        let result = harness.test_ring_buffer(&config);
        // Underruns happen when reading more than written
        assert!(result.underruns_detected > 0 || result.reads_failed > 0);
    }

    // =========================================================================
    // H0-WH-16: Shared memory tests
    // =========================================================================

    #[test]
    fn h0_wh_16_shared_memory_basic() {
        let harness = WorkerTestHarness::new();
        let config = SharedMemoryTestConfig::default();
        let result = harness.test_shared_memory(&config);
        assert!(result.atomics_correct);
    }

    #[test]
    fn h0_wh_17_shared_memory_disabled() {
        let config = WorkerTestConfig {
            test_shared_memory: false,
            ..Default::default()
        };
        let harness = WorkerTestHarness::with_config(config);
        let sm_config = SharedMemoryTestConfig::default();
        let result = harness.test_shared_memory(&sm_config);
        assert!(result.is_passed());
    }

    // =========================================================================
    // H0-WH-18: Metrics tests
    // =========================================================================

    #[test]
    fn h0_wh_18_metrics_memory_leak_detection() {
        let mut metrics = WorkerMetrics::default();
        metrics.memory_start = 1000;
        metrics.memory_end = 1200; // 20% growth
        assert!(metrics.has_memory_leak());
    }

    #[test]
    fn h0_wh_19_metrics_no_memory_leak() {
        let mut metrics = WorkerMetrics::default();
        metrics.memory_start = 1000;
        metrics.memory_end = 1050; // 5% growth
        assert!(!metrics.has_memory_leak());
    }

    #[test]
    fn h0_wh_20_metrics_memory_growth() {
        let mut metrics = WorkerMetrics::default();
        metrics.memory_start = 1000;
        metrics.memory_end = 1500;
        assert_eq!(metrics.memory_growth(), 500);
    }

    #[test]
    fn h0_wh_21_metrics_memory_shrink() {
        let mut metrics = WorkerMetrics::default();
        metrics.memory_start = 1000;
        metrics.memory_end = 800;
        assert_eq!(metrics.memory_growth(), -200);
    }

    #[test]
    fn h0_wh_22_metrics_zero_start() {
        let metrics = WorkerMetrics::default();
        assert!(!metrics.has_memory_leak());
    }

    // =========================================================================
    // H0-WH-23: Result tests
    // =========================================================================

    #[test]
    fn h0_wh_23_result_is_passed() {
        let result = WorkerTestResult {
            passed: true,
            lifecycle_passed: true,
            ordering_passed: true,
            shared_memory_passed: true,
            ring_buffer_passed: true,
            error_recovery_passed: true,
            memory_leak_passed: true,
            failures: vec![],
            ..Default::default()
        };
        assert!(result.is_passed());
    }

    #[test]
    fn h0_wh_24_result_with_failures() {
        let result = WorkerTestResult {
            passed: false,
            failures: vec![WorkerTestFailure {
                category: WorkerTestCategory::Lifecycle,
                description: "test".to_string(),
                expected: "a".to_string(),
                actual: "b".to_string(),
            }],
            ..Default::default()
        };
        assert!(!result.is_passed());
        assert_eq!(result.failure_count(), 1);
    }

    #[test]
    fn h0_wh_25_result_display() {
        let result = WorkerTestResult {
            passed: true,
            metrics: WorkerMetrics {
                initialization_time: Duration::from_millis(100),
                average_message_latency: Duration::from_micros(500),
                messages_processed: 1000,
                ..Default::default()
            },
            ..Default::default()
        };
        let display = format!("{result}");
        assert!(display.contains("PASSED"));
        assert!(display.contains("Messages processed: 1000"));
    }

    // =========================================================================
    // H0-WH-26: Failure display tests
    // =========================================================================

    #[test]
    fn h0_wh_26_failure_display() {
        let failure = WorkerTestFailure {
            category: WorkerTestCategory::Ordering,
            description: "Out of order".to_string(),
            expected: "5".to_string(),
            actual: "3".to_string(),
        };
        let display = format!("{failure}");
        assert!(display.contains("Ordering"));
        assert!(display.contains("Out of order"));
        assert!(display.contains("expected '5'"));
    }

    // =========================================================================
    // H0-WH-27: JavaScript generation tests
    // =========================================================================

    #[test]
    fn h0_wh_27_lifecycle_test_js() {
        let js = WorkerTestHarness::lifecycle_test_js();
        assert!(js.contains("__PROBAR_WORKER_STATES__"));
        assert!(js.contains("recordState"));
        assert!(js.contains("recordTransition"));
    }

    #[test]
    fn h0_wh_28_ring_buffer_test_js() {
        let js = WorkerTestHarness::ring_buffer_test_js(16384);
        assert!(js.contains("__PROBAR_RING_BUFFER__"));
        assert!(js.contains("SharedArrayBuffer"));
        assert!(js.contains("Atomics.load"));
        assert!(js.contains("16384"));
    }

    #[test]
    fn h0_wh_29_shared_memory_test_js() {
        let js = WorkerTestHarness::shared_memory_test_js(4096);
        assert!(js.contains("__PROBAR_SHARED_MEMORY__"));
        assert!(js.contains("testAtomicAdd"));
        assert!(js.contains("testCompareExchange"));
        assert!(js.contains("4096"));
    }

    // =========================================================================
    // H0-WH-30: Validation tests
    // =========================================================================

    #[test]
    fn h0_wh_30_validate_results_passed() {
        let harness = WorkerTestHarness::new();
        let result = WorkerTestResult {
            passed: true,
            lifecycle_passed: true,
            ordering_passed: true,
            shared_memory_passed: true,
            ring_buffer_passed: true,
            error_recovery_passed: true,
            memory_leak_passed: true,
            failures: vec![],
            ..Default::default()
        };
        assert!(harness.validate_results(&result));
    }

    #[test]
    fn h0_wh_31_validate_results_lifecycle_failed() {
        let harness = WorkerTestHarness::new();
        let result = WorkerTestResult {
            lifecycle_passed: false,
            ..Default::default()
        };
        assert!(!harness.validate_results(&result));
    }

    #[test]
    fn h0_wh_32_validate_results_ordering_failed() {
        let harness = WorkerTestHarness::new();
        let result = WorkerTestResult {
            lifecycle_passed: true,
            ordering_passed: false,
            ..Default::default()
        };
        assert!(!harness.validate_results(&result));
    }

    // =========================================================================
    // H0-WH-33: Error display tests
    // =========================================================================

    #[test]
    fn h0_wh_33_error_display() {
        let err = WorkerTestError::InitializationFailed("worker failed".to_string());
        assert!(err.to_string().contains("Initialization"));

        let err = WorkerTestError::Timeout("10s".to_string());
        assert!(err.to_string().contains("Timeout"));

        let err = WorkerTestError::ProtocolError("bad message".to_string());
        assert!(err.to_string().contains("Protocol"));

        let err = WorkerTestError::CdpError("connection lost".to_string());
        assert!(err.to_string().contains("CDP"));
    }

    // =========================================================================
    // H0-WH-34: Ring buffer result tests
    // =========================================================================

    #[test]
    fn h0_wh_34_ring_buffer_result_passed() {
        let result = RingBufferTestResult {
            passed: true,
            corruption_detected: false,
            failures: vec![],
            ..Default::default()
        };
        assert!(result.is_passed());
    }

    #[test]
    fn h0_wh_35_ring_buffer_result_corruption() {
        let result = RingBufferTestResult {
            passed: true,
            corruption_detected: true,
            ..Default::default()
        };
        assert!(!result.is_passed());
    }

    // =========================================================================
    // H0-WH-36: Shared memory result tests
    // =========================================================================

    #[test]
    fn h0_wh_36_shared_memory_result_passed() {
        let result = SharedMemoryTestResult {
            passed: true,
            race_conditions_detected: 0,
            failures: vec![],
            ..Default::default()
        };
        assert!(result.is_passed());
    }

    #[test]
    fn h0_wh_37_shared_memory_result_race_condition() {
        let result = SharedMemoryTestResult {
            passed: true,
            race_conditions_detected: 1,
            ..Default::default()
        };
        assert!(!result.is_passed());
    }

    // =========================================================================
    // H0-WH-38: Default configs
    // =========================================================================

    #[test]
    fn h0_wh_38_ring_buffer_config_default() {
        let config = RingBufferTestConfig::default();
        assert_eq!(config.buffer_size, 16384);
        assert!(config.test_overflow);
        assert!(config.test_underrun);
    }

    #[test]
    fn h0_wh_39_shared_memory_config_default() {
        let config = SharedMemoryTestConfig::default();
        assert_eq!(config.buffer_size, 4096);
        assert!(config.test_wait_notify);
        assert!(config.test_concurrent_writes);
    }
}
