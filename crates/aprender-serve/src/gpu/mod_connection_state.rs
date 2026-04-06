
/// Connection state for health checking
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConnectionState {
    /// Connection is healthy
    Healthy,
    /// Connection is stale and needs recycling
    Stale,
    /// Connection is broken
    Broken,
}

/// Connection handle
#[derive(Debug)]
pub struct Connection {
    #[allow(dead_code)]
    id: u64,
    created_at: std::time::Instant,
}

/// Connection pool with bounded capacity (IMP-073)
pub struct ConnectionPool {
    config: ConnectionConfig,
    active: std::sync::atomic::AtomicUsize,
    idle: std::sync::Mutex<Vec<Connection>>,
    next_id: std::sync::atomic::AtomicU64,
}

impl ConnectionPool {
    /// Create new connection pool
    #[must_use]
    pub fn new(config: ConnectionConfig) -> Self {
        Self {
            config,
            active: std::sync::atomic::AtomicUsize::new(0),
            idle: std::sync::Mutex::new(Vec::new()),
            next_id: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Get max connections
    #[must_use]
    pub fn max_connections(&self) -> usize {
        self.config.max_connections
    }

    /// Get min connections
    #[must_use]
    pub fn min_connections(&self) -> usize {
        self.config.min_connections
    }

    /// Get active connection count
    #[must_use]
    pub fn active_connections(&self) -> usize {
        self.active.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Get idle connection count
    #[must_use]
    pub fn idle_connections(&self) -> usize {
        self.idle.lock().expect("mutex poisoned").len()
    }

    /// Acquire a connection (blocking)
    ///
    /// # Errors
    /// Returns error if pool is exhausted.
    pub fn acquire(&self) -> std::result::Result<Connection, &'static str> {
        // Try to get from idle pool first
        {
            let mut idle = self.idle.lock().expect("mutex poisoned");
            if let Some(conn) = idle.pop() {
                self.active
                    .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                return Ok(conn);
            }
        }

        // Create new if under limit
        let current = self.active.load(std::sync::atomic::Ordering::SeqCst);
        if current < self.config.max_connections {
            self.active
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let id = self
                .next_id
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            return Ok(Connection {
                id,
                created_at: std::time::Instant::now(),
            });
        }

        Err("Pool exhausted")
    }

    /// Try to acquire a connection (non-blocking)
    ///
    /// # Errors
    /// Returns error if pool is exhausted.
    pub fn try_acquire(&self) -> std::result::Result<Connection, &'static str> {
        self.acquire()
    }

    /// Release a connection back to pool
    pub fn release(&self, conn: Connection) {
        self.active
            .fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
        let mut idle = self.idle.lock().expect("mutex poisoned");
        idle.push(conn);
    }

    /// Check connection health
    #[must_use]
    pub fn check_health(&self, conn: &Connection) -> ConnectionState {
        let age = conn.created_at.elapsed();
        if age > self.config.idle_timeout {
            ConnectionState::Stale
        } else {
            ConnectionState::Healthy
        }
    }

    /// Warm the pool to min_connections
    pub fn warm(&self) {
        let current_idle = self.idle_connections();
        let need = self.config.min_connections.saturating_sub(current_idle);

        let mut idle = self.idle.lock().expect("mutex poisoned");
        for _ in 0..need {
            let id = self
                .next_id
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            idle.push(Connection {
                id,
                created_at: std::time::Instant::now(),
            });
        }
    }
}

/// Resource configuration (IMP-074)
#[derive(Debug, Clone)]
#[allow(clippy::struct_field_names)]
pub struct ResourceConfig {
    max_memory_per_request: u64,
    max_total_memory: u64,
    max_compute_time: Duration,
    max_queue_depth: usize,
}

impl ResourceConfig {
    /// Create new resource config with defaults
    #[must_use]
    pub fn new() -> Self {
        Self {
            max_memory_per_request: 512 * 1024 * 1024, // 512MB
            max_total_memory: 4 * 1024 * 1024 * 1024,  // 4GB
            max_compute_time: Duration::from_secs(30),
            max_queue_depth: 100,
        }
    }

    /// Set max memory per request
    #[must_use]
    pub fn with_max_memory_per_request(mut self, bytes: u64) -> Self {
        self.max_memory_per_request = bytes;
        self
    }

    /// Set max total memory
    #[must_use]
    pub fn with_max_total_memory(mut self, bytes: u64) -> Self {
        self.max_total_memory = bytes;
        self
    }

    /// Set max compute time
    #[must_use]
    pub fn with_max_compute_time(mut self, time: Duration) -> Self {
        self.max_compute_time = time;
        self
    }

    /// Set max queue depth
    #[must_use]
    pub fn with_max_queue_depth(mut self, depth: usize) -> Self {
        self.max_queue_depth = depth;
        self
    }
}

impl Default for ResourceConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of limit check
#[derive(Debug, Clone)]
pub enum LimitResult {
    /// Request is allowed
    Allowed,
    /// Request is denied with reason
    Denied {
        /// Reason for denial
        reason: String,
    },
    /// Backpressure should be applied
    Backpressure,
}

/// Resource limiter (IMP-074)
pub struct ResourceLimiter {
    config: ResourceConfig,
    current_memory: std::sync::atomic::AtomicU64,
    queue_depth: std::sync::atomic::AtomicUsize,
}

impl ResourceLimiter {
    /// Create new resource limiter
    #[must_use]
    pub fn new(config: ResourceConfig) -> Self {
        Self {
            config,
            current_memory: std::sync::atomic::AtomicU64::new(0),
            queue_depth: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// Check if memory request is within limits
    #[must_use]
    pub fn check_memory(&self, bytes: u64) -> LimitResult {
        if bytes > self.config.max_memory_per_request {
            return LimitResult::Denied {
                reason: format!(
                    "Request {} bytes exceeds per-request limit {} bytes",
                    bytes, self.config.max_memory_per_request
                ),
            };
        }

        let current = self
            .current_memory
            .load(std::sync::atomic::Ordering::SeqCst);
        if current + bytes > self.config.max_total_memory {
            return LimitResult::Denied {
                reason: format!(
                    "Total memory {} + {} would exceed limit {}",
                    current, bytes, self.config.max_total_memory
                ),
            };
        }

        LimitResult::Allowed
    }

    /// Allocate memory
    ///
    /// # Errors
    /// Returns error if memory limit exceeded.
    pub fn allocate(&self, bytes: u64) -> std::result::Result<(), &'static str> {
        if let LimitResult::Denied { .. } = self.check_memory(bytes) {
            return Err("Memory limit exceeded");
        }
        self.current_memory
            .fetch_add(bytes, std::sync::atomic::Ordering::SeqCst);
        Ok(())
    }

    /// Deallocate memory
    pub fn deallocate(&self, bytes: u64) {
        self.current_memory
            .fetch_sub(bytes, std::sync::atomic::Ordering::SeqCst);
    }

    /// Get current memory usage
    #[must_use]
    pub fn current_memory(&self) -> u64 {
        self.current_memory
            .load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Enqueue a request
    pub fn enqueue(&self) -> LimitResult {
        let current = self
            .queue_depth
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        if current >= self.config.max_queue_depth {
            self.queue_depth
                .fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
            LimitResult::Backpressure
        } else {
            LimitResult::Allowed
        }
    }

    /// Try to enqueue (returns backpressure if full)
    #[must_use]
    pub fn try_enqueue(&self) -> LimitResult {
        let current = self.queue_depth.load(std::sync::atomic::Ordering::SeqCst);
        if current >= self.config.max_queue_depth {
            LimitResult::Backpressure
        } else {
            self.enqueue()
        }
    }

    /// Dequeue a request
    pub fn dequeue(&self) {
        let current = self.queue_depth.load(std::sync::atomic::Ordering::SeqCst);
        if current > 0 {
            self.queue_depth
                .fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
        }
    }

    /// Start compute timer
    #[must_use]
    pub fn start_compute(&self) -> std::time::Instant {
        std::time::Instant::now()
    }
}

/// Resource metrics snapshot (IMP-075)
#[derive(Debug, Clone)]
pub struct ResourceMetrics {
    /// Current memory usage in bytes
    pub memory_bytes: u64,
    /// GPU utilization percentage (0-100)
    pub gpu_utilization: f64,
    /// Current queue depth
    pub queue_depth: usize,
    /// Last recorded latency in milliseconds
    pub last_latency_ms: u64,
}

/// Latency statistics
#[derive(Debug, Clone)]
pub struct LatencyStats {
    /// Minimum latency in ms
    pub min_ms: u64,
    /// Maximum latency in ms
    pub max_ms: u64,
    /// Average latency in ms
    pub avg_ms: u64,
}

/// Resource monitor snapshot
#[derive(Debug, Clone)]
pub struct ResourceSnapshot {
    /// Unix timestamp
    pub timestamp: u64,
    /// Memory in bytes
    pub memory_bytes: u64,
    /// GPU utilization
    pub gpu_utilization: f64,
    /// Queue depth
    pub queue_depth: usize,
}

/// Resource monitor (IMP-075)
pub struct ResourceMonitor {
    memory_bytes: std::sync::atomic::AtomicU64,
    gpu_utilization: std::sync::Mutex<f64>,
    queue_depth: std::sync::atomic::AtomicUsize,
    latencies: std::sync::Mutex<Vec<u64>>,
    last_latency_ms: std::sync::atomic::AtomicU64,
}
