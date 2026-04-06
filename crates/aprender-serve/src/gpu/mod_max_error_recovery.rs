
impl ErrorRecoveryStrategy {
    /// Create new error recovery strategy
    #[must_use]
    pub fn new() -> Self {
        Self {
            max_retries: 3,
            base_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(10),
            jitter: 0.1,
        }
    }

    /// Set maximum retries
    #[must_use]
    pub fn with_max_retries(mut self, max_retries: u32) -> Self {
        self.max_retries = max_retries;
        self
    }

    /// Set base delay
    #[must_use]
    pub fn with_base_delay(mut self, base_delay: Duration) -> Self {
        self.base_delay = base_delay;
        self
    }

    /// Set maximum delay
    #[must_use]
    pub fn with_max_delay(mut self, max_delay: Duration) -> Self {
        self.max_delay = max_delay;
        self
    }

    /// Set jitter factor (0.0 - 1.0)
    #[must_use]
    pub fn with_jitter(mut self, jitter: f64) -> Self {
        self.jitter = jitter.clamp(0.0, 1.0);
        self
    }

    /// Get max retries
    #[must_use]
    pub fn max_retries(&self) -> u32 {
        self.max_retries
    }

    /// Classify an error
    #[must_use]
    pub fn classify_error(&self, error: &std::io::Error) -> ErrorClassification {
        match error.kind() {
            std::io::ErrorKind::TimedOut
            | std::io::ErrorKind::ConnectionReset
            | std::io::ErrorKind::ConnectionAborted
            | std::io::ErrorKind::Interrupted
            | std::io::ErrorKind::WouldBlock => ErrorClassification::Transient,

            std::io::ErrorKind::Other => {
                let msg = error.to_string().to_lowercase();
                if msg.contains("gpu") || msg.contains("cuda") || msg.contains("wgpu") {
                    ErrorClassification::GpuFailure
                } else {
                    ErrorClassification::Transient
                }
            },

            _ => ErrorClassification::Fatal,
        }
    }

    /// Determine recovery action
    #[must_use]
    pub fn determine_action(&self, error: &std::io::Error, attempt: u32) -> RecoveryAction {
        if attempt >= self.max_retries {
            return RecoveryAction::Fail;
        }

        match self.classify_error(error) {
            ErrorClassification::Transient => RecoveryAction::Retry {
                delay: self.calculate_delay(attempt),
            },
            ErrorClassification::GpuFailure => RecoveryAction::FallbackToCpu,
            ErrorClassification::Fatal => RecoveryAction::Fail,
        }
    }

    /// Determine action with explicit GPU fallback
    #[must_use]
    pub fn determine_action_with_fallback(
        &self,
        error: &std::io::Error,
        attempt: u32,
    ) -> RecoveryAction {
        let msg = error.to_string().to_lowercase();
        if msg.contains("gpu") || msg.contains("unavailable") {
            RecoveryAction::FallbackToCpu
        } else {
            self.determine_action(error, attempt)
        }
    }

    /// Calculate delay for retry attempt with exponential backoff
    #[must_use]
    pub fn calculate_delay(&self, attempt: u32) -> Duration {
        let base_ms = self.base_delay.as_millis() as f64;
        let exp_delay = base_ms * 2.0_f64.powi(attempt as i32);
        let capped_delay = exp_delay.min(self.max_delay.as_millis() as f64);

        // Add jitter
        let jitter_range = capped_delay * self.jitter;
        let jittered = capped_delay + (jitter_range * 0.5); // Simplified jitter

        Duration::from_millis(jittered as u64)
    }
}

impl Default for ErrorRecoveryStrategy {
    fn default() -> Self {
        Self::new()
    }
}

/// Degradation mode for system state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DegradationMode {
    /// Normal operation
    Normal,
    /// Running on CPU fallback
    CpuFallback,
    /// Memory pressure - reduced capacity
    MemoryPressure,
    /// Low latency priority mode
    LowLatency,
    /// High throughput priority mode
    HighThroughput,
}

/// System load metrics
#[derive(Debug, Clone, Copy)]
pub struct SystemLoad {
    /// CPU utilization percentage
    pub cpu_percent: f64,
    /// Memory utilization percentage
    pub memory_percent: f64,
    /// Current queue depth
    pub queue_depth: u32,
}

/// Graceful degradation manager
pub struct DegradationManager {
    gpu_available: bool,
    memory_pressure: f64,
    system_load: Option<SystemLoad>,
    latency_priority: bool,
    mode: DegradationMode,
}

impl DegradationManager {
    /// Create new degradation manager
    #[must_use]
    pub fn new() -> Self {
        Self {
            gpu_available: true,
            memory_pressure: 0.0,
            system_load: None,
            latency_priority: false,
            mode: DegradationMode::Normal,
        }
    }

    /// Get current degradation mode
    #[must_use]
    pub fn current_mode(&self) -> DegradationMode {
        self.mode
    }

    /// Set GPU availability
    pub fn set_gpu_available(&mut self, available: bool) {
        self.gpu_available = available;
        self.update_mode();
    }

    /// Update memory pressure (0.0 - 1.0)
    pub fn update_memory_pressure(&mut self, pressure: f64) {
        self.memory_pressure = pressure.clamp(0.0, 1.0);
        self.update_mode();
    }

    /// Update system load
    pub fn update_system_load(&mut self, load: SystemLoad) {
        self.system_load = Some(load);
        self.update_mode();
    }

    /// Set latency priority mode
    pub fn set_latency_priority(&mut self, priority: bool) {
        self.latency_priority = priority;
        self.update_mode();
    }

    /// Get recommended batch size based on system state
    #[must_use]
    pub fn recommended_batch_size(&self, requested: usize) -> usize {
        if self.memory_pressure > 0.8 {
            // Reduce batch size under memory pressure
            (requested as f64 * (1.0 - self.memory_pressure)).max(1.0) as usize
        } else {
            requested
        }
    }

    /// Get recommended max context length based on system state
    #[must_use]
    pub fn recommended_max_context(&self, requested: usize) -> usize {
        if let Some(load) = &self.system_load {
            if load.cpu_percent > 90.0 || load.memory_percent > 80.0 || load.queue_depth > 50 {
                // Reduce context length under high load
                (requested as f64 * 0.75).max(256.0) as usize
            } else {
                requested
            }
        } else {
            requested
        }
    }

    fn update_mode(&mut self) {
        self.mode = if !self.gpu_available {
            DegradationMode::CpuFallback
        } else if self.latency_priority {
            DegradationMode::LowLatency
        } else if self.memory_pressure > 0.8 {
            DegradationMode::MemoryPressure
        } else if let Some(load) = &self.system_load {
            if load.cpu_percent > 90.0 || load.memory_percent > 80.0 {
                DegradationMode::MemoryPressure
            } else {
                DegradationMode::Normal
            }
        } else {
            DegradationMode::Normal
        };
    }
}

impl Default for DegradationManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Request outcome for failure tracking
#[derive(Debug, Clone)]
pub enum RequestOutcome {
    /// Request completed successfully
    Success,
    /// Request failed with error message
    Failed(String),
}

/// Failure isolator with circuit breaker
pub struct FailureIsolator {
    active_requests: std::sync::atomic::AtomicU64,
    success_count: std::sync::atomic::AtomicU64,
    failure_count: std::sync::atomic::AtomicU64,
    consecutive_failures: std::sync::atomic::AtomicU32,
    circuit_open: std::sync::atomic::AtomicBool,
    next_request_id: std::sync::atomic::AtomicU64,
    failure_threshold: u32,
    cleanups: std::sync::Mutex<HashMap<u64, Box<dyn FnOnce() + Send>>>,
}

impl FailureIsolator {
    /// Create new failure isolator
    #[must_use]
    pub fn new() -> Self {
        Self {
            active_requests: std::sync::atomic::AtomicU64::new(0),
            success_count: std::sync::atomic::AtomicU64::new(0),
            failure_count: std::sync::atomic::AtomicU64::new(0),
            consecutive_failures: std::sync::atomic::AtomicU32::new(0),
            circuit_open: std::sync::atomic::AtomicBool::new(false),
            next_request_id: std::sync::atomic::AtomicU64::new(0),
            failure_threshold: 5,
            cleanups: std::sync::Mutex::new(HashMap::new()),
        }
    }

    /// Get number of active requests
    #[must_use]
    pub fn active_requests(&self) -> u64 {
        self.active_requests
            .load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Get success count
    #[must_use]
    pub fn success_count(&self) -> u64 {
        self.success_count.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Get failure count
    #[must_use]
    pub fn failure_count(&self) -> u64 {
        self.failure_count.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Check if circuit is open
    #[must_use]
    pub fn is_circuit_open(&self) -> bool {
        self.circuit_open.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Start a new isolated request
    #[must_use]
    pub fn start_request(&self) -> u64 {
        self.active_requests
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        self.next_request_id
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
    }

    /// Try to start a request (fails if circuit is open)
    ///
    /// # Errors
    /// Returns error if circuit breaker is open.
    pub fn try_start_request(&self) -> std::result::Result<u64, &'static str> {
        if self.is_circuit_open() {
            Err("Circuit breaker is open")
        } else {
            Ok(self.start_request())
        }
    }

    /// Register cleanup handler for a request
    pub fn register_cleanup<F: FnOnce() + Send + 'static>(&self, request_id: u64, cleanup: F) {
        if let Ok(mut cleanups) = self.cleanups.lock() {
            cleanups.insert(request_id, Box::new(cleanup));
        }
    }

    /// Complete a request with outcome
    pub fn complete_request(&self, request_id: u64, outcome: &RequestOutcome) {
        self.active_requests
            .fetch_sub(1, std::sync::atomic::Ordering::SeqCst);

        match outcome {
            RequestOutcome::Success => {
                self.success_count
                    .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                self.consecutive_failures
                    .store(0, std::sync::atomic::Ordering::SeqCst);
            },
            RequestOutcome::Failed(_) => {
                self.failure_count
                    .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                let failures = self
                    .consecutive_failures
                    .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
                    + 1;

                // Open circuit if threshold exceeded
                if failures >= self.failure_threshold {
                    self.circuit_open
                        .store(true, std::sync::atomic::Ordering::SeqCst);
                }

                // Run cleanup handler
                if let Ok(mut cleanups) = self.cleanups.lock() {
                    if let Some(cleanup) = cleanups.remove(&request_id) {
                        cleanup();
                    }
                }
            },
        }
    }

    /// Reset circuit breaker
    pub fn reset_circuit(&self) {
        self.circuit_open
            .store(false, std::sync::atomic::Ordering::SeqCst);
        self.consecutive_failures
            .store(0, std::sync::atomic::Ordering::SeqCst);
    }
}

impl Default for FailureIsolator {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// M30: Connection Pooling & Resource Limits (IMP-073, IMP-074, IMP-075)
// ============================================================================

/// Connection pool configuration (IMP-073)
#[derive(Debug, Clone)]
pub struct ConnectionConfig {
    max_connections: usize,
    min_connections: usize,
    idle_timeout: Duration,
}

impl ConnectionConfig {
    /// Create new connection config with defaults
    #[must_use]
    pub fn new() -> Self {
        Self {
            max_connections: 10,
            min_connections: 1,
            idle_timeout: Duration::from_secs(300),
        }
    }

    /// Set maximum connections
    #[must_use]
    pub fn with_max_connections(mut self, max: usize) -> Self {
        self.max_connections = max;
        self
    }

    /// Set minimum connections
    #[must_use]
    pub fn with_min_connections(mut self, min: usize) -> Self {
        self.min_connections = min;
        self
    }

    /// Set idle timeout
    #[must_use]
    pub fn with_idle_timeout(mut self, timeout: Duration) -> Self {
        self.idle_timeout = timeout;
        self
    }
}

impl Default for ConnectionConfig {
    fn default() -> Self {
        Self::new()
    }
}
