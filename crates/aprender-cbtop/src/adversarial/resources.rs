//! Resource limiting, bit-flip injection, cancellation, and recovery.

use std::time::{Duration, Instant};

use super::{AdversarialError, AdversarialResult, ResourceUsage};

/// Bit-flip injector for testing corruption handling
#[derive(Debug, Clone)]
pub struct BitFlipInjector {
    /// Seed for reproducible bit flips
    pub seed: u64,
    /// Number of bits to flip
    pub flip_count: usize,
}

impl Default for BitFlipInjector {
    fn default() -> Self {
        Self {
            seed: 42,
            flip_count: 1,
        }
    }
}

impl BitFlipInjector {
    /// Create a new bit-flip injector
    pub fn new(seed: u64, flip_count: usize) -> Self {
        Self { seed, flip_count }
    }

    /// Inject bit flips into data (returns modified copy)
    pub fn inject(&self, data: &[u8]) -> Vec<u8> {
        let mut result = data.to_vec();
        if result.is_empty() {
            return result;
        }

        // Simple LCG for reproducible "random" positions
        let mut rng_state = self.seed;
        for _ in 0..self.flip_count {
            // LCG: next = (a * state + c) mod m
            rng_state = rng_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);

            let byte_idx = (rng_state as usize) % result.len();
            let bit_idx = ((rng_state >> 32) as usize) % 8;

            result[byte_idx] ^= 1 << bit_idx;
        }

        result
    }

    /// Inject bit flips into float data
    pub fn inject_floats(&self, data: &[f32]) -> Vec<f32> {
        // Convert to bytes, flip bits, convert back
        let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();

        let corrupted = self.inject(&bytes);

        corrupted
            .chunks_exact(4)
            .map(|chunk| {
                let arr: [u8; 4] = chunk.try_into().unwrap_or([0; 4]);
                f32::from_le_bytes(arr)
            })
            .collect()
    }
}

/// Resource limiter for bounded operations (F1016, F1017, F1018)
#[derive(Debug, Clone)]
pub struct ResourceLimiter {
    /// Maximum stack depth for recursive operations
    pub max_stack_depth: usize,
    /// Maximum memory allocation in bytes
    pub max_memory_bytes: usize,
    /// Timeout for operations
    pub timeout: Duration,
    /// Current stack depth
    current_depth: usize,
    /// Current allocated memory
    current_memory: usize,
    /// Operation start time
    start_time: Option<Instant>,
}

impl Default for ResourceLimiter {
    fn default() -> Self {
        Self {
            max_stack_depth: 1000,
            max_memory_bytes: 1024 * 1024 * 1024, // 1GB
            timeout: Duration::from_secs(60),
            current_depth: 0,
            current_memory: 0,
            start_time: None,
        }
    }
}

impl ResourceLimiter {
    /// Create a new resource limiter
    pub fn new() -> Self {
        Self::default()
    }

    /// Set maximum stack depth
    pub fn with_max_depth(mut self, max_depth: usize) -> Self {
        self.max_stack_depth = max_depth;
        self
    }

    /// Set maximum memory
    pub fn with_max_memory(mut self, max_bytes: usize) -> Self {
        self.max_memory_bytes = max_bytes;
        self
    }

    /// Set timeout
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Start a timed operation
    pub fn start_operation(&mut self) {
        self.start_time = Some(Instant::now());
    }

    /// Check if operation has timed out (F1018)
    pub fn check_timeout(&self, operation: &str) -> AdversarialResult<()> {
        if let Some(start) = self.start_time {
            let elapsed = start.elapsed();
            if elapsed > self.timeout {
                return Err(AdversarialError::Timeout {
                    operation: operation.to_string(),
                    elapsed,
                    limit: self.timeout,
                });
            }
        }
        Ok(())
    }

    /// Enter a recursive call (F1016)
    pub fn enter_recursion(&mut self) -> AdversarialResult<()> {
        self.current_depth += 1;
        if self.current_depth > self.max_stack_depth {
            return Err(AdversarialError::StackOverflow {
                depth: self.current_depth,
                max_depth: self.max_stack_depth,
            });
        }
        Ok(())
    }

    /// Exit a recursive call
    pub fn exit_recursion(&mut self) {
        if self.current_depth > 0 {
            self.current_depth -= 1;
        }
    }

    /// Request memory allocation (F1017)
    pub fn request_memory(&mut self, bytes: usize) -> AdversarialResult<()> {
        let new_total = self.current_memory.saturating_add(bytes);
        if new_total > self.max_memory_bytes {
            return Err(AdversarialError::ResourceExhausted {
                resource: format!(
                    "memory: requested {bytes} bytes, would exceed limit of {} bytes",
                    self.max_memory_bytes
                ),
            });
        }
        self.current_memory = new_total;
        Ok(())
    }

    /// Release memory
    pub fn release_memory(&mut self, bytes: usize) {
        self.current_memory = self.current_memory.saturating_sub(bytes);
    }

    /// Get current resource usage
    pub fn usage(&self) -> ResourceUsage {
        ResourceUsage {
            stack_depth: self.current_depth,
            memory_bytes: self.current_memory,
            elapsed: self.start_time.map(|s| s.elapsed()),
        }
    }

    /// Reset limiter state
    pub fn reset(&mut self) {
        self.current_depth = 0;
        self.current_memory = 0;
        self.start_time = None;
    }
}

/// Cancellation token for cooperative cancellation (F1019)
#[derive(Debug, Clone)]
pub struct CancellationToken {
    cancelled: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

impl Default for CancellationToken {
    fn default() -> Self {
        Self::new()
    }
}

impl CancellationToken {
    /// Create a new cancellation token
    pub fn new() -> Self {
        Self {
            cancelled: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
        }
    }

    /// Request cancellation
    pub fn cancel(&self) {
        self.cancelled
            .store(true, std::sync::atomic::Ordering::SeqCst);
    }

    /// Check if cancelled
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Return error if cancelled
    pub fn check(&self, operation: &str) -> AdversarialResult<()> {
        if self.is_cancelled() {
            return Err(AdversarialError::Cancelled {
                operation: operation.to_string(),
            });
        }
        Ok(())
    }

    /// Clone the token (same underlying state)
    pub fn clone_token(&self) -> Self {
        Self {
            cancelled: std::sync::Arc::clone(&self.cancelled),
        }
    }
}

/// Recovery handler for error recovery testing (F1020)
#[derive(Debug, Clone)]
pub struct RecoveryHandler<S: Clone> {
    /// Stored checkpoint state
    checkpoint: Option<S>,
}

impl<S: Clone> Default for RecoveryHandler<S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S: Clone> RecoveryHandler<S> {
    /// Create a new recovery handler
    pub fn new() -> Self {
        Self { checkpoint: None }
    }

    /// Save a checkpoint
    pub fn checkpoint(&mut self, state: S) {
        self.checkpoint = Some(state);
    }

    /// Recover from error using checkpoint
    pub fn recover(&self) -> AdversarialResult<S> {
        self.checkpoint
            .clone()
            .ok_or_else(|| AdversarialError::RecoveryFailed {
                original_error: "unknown".to_string(),
                recovery_error: "no checkpoint available".to_string(),
            })
    }

    /// Try operation with automatic recovery on failure
    pub fn try_with_recovery<F, T, E>(&self, operation: F) -> AdversarialResult<T>
    where
        F: FnOnce() -> Result<T, E>,
        E: std::fmt::Display,
    {
        match operation() {
            Ok(result) => Ok(result),
            Err(e) => Err(AdversarialError::RecoveryFailed {
                original_error: e.to_string(),
                recovery_error: if self.checkpoint.is_some() {
                    "operation failed, checkpoint available".to_string()
                } else {
                    "operation failed, no checkpoint".to_string()
                },
            }),
        }
    }

    /// Check if checkpoint exists
    pub fn has_checkpoint(&self) -> bool {
        self.checkpoint.is_some()
    }
}
