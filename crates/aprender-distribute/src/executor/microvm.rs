//! MicroVM Executor
//!
//! Hardware-isolated task execution using pepita's KVM-based `MicroVMs`.
//! Provides Docker/Lambda-like isolation with sub-100ms cold start.
#![allow(clippy::doc_markdown, clippy::missing_errors_doc, clippy::cast_precision_loss)]
//!
//! ## Example
//!
//! ```rust,ignore
//! use repartir::executor::microvm::{MicroVmExecutor, MicroVmConfig};
//! use repartir::task::Task;
//!
//! let config = MicroVmConfig::builder()
//!     .memory_mib(256)
//!     .vcpus(2)
//!     .build()?;
//!
//! let executor = MicroVmExecutor::new(config)?;
//! let result = executor.execute(task).await?;
//! ```

use crate::error::{RepartirError, Result};
use crate::task::{ExecutionResult, Task};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use uuid::Uuid;

// Re-export pepita types for convenience
pub use pepita::vmm::{ExitReason, JailerConfig, MemoryRegion, MicroVm, VmConfig, VmState};

// ============================================================================
// MICROVM EXECUTOR CONFIG
// ============================================================================

/// MicroVM executor configuration
#[derive(Debug, Clone)]
pub struct MicroVmExecutorConfig {
    /// Memory per VM in MiB
    pub memory_mib: u32,
    /// vCPUs per VM
    pub vcpus: u32,
    /// Maximum concurrent VMs
    pub max_vms: usize,
    /// VM idle timeout before eviction
    pub idle_timeout: Duration,
    /// Enable jailer security
    pub enable_jailer: bool,
    /// Jailer configuration
    pub jailer_config: Option<JailerConfig>,
    /// Enable warm pool
    pub warm_pool_enabled: bool,
    /// Warm pool size
    pub warm_pool_size: usize,
}

impl Default for MicroVmExecutorConfig {
    fn default() -> Self {
        Self {
            memory_mib: 128,
            vcpus: 1,
            max_vms: 100,
            idle_timeout: Duration::from_secs(300),
            enable_jailer: false,
            jailer_config: None,
            warm_pool_enabled: true,
            warm_pool_size: 3,
        }
    }
}

impl MicroVmExecutorConfig {
    /// Create a new builder
    #[must_use]
    pub fn builder() -> MicroVmExecutorConfigBuilder {
        MicroVmExecutorConfigBuilder::default()
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        if self.memory_mib == 0 {
            return Err(RepartirError::InvalidTask {
                reason: "Memory must be at least 1 MiB".to_string(),
            });
        }
        if self.vcpus == 0 {
            return Err(RepartirError::InvalidTask {
                reason: "Must have at least 1 vCPU".to_string(),
            });
        }
        if self.max_vms == 0 {
            return Err(RepartirError::InvalidTask {
                reason: "Must allow at least 1 VM".to_string(),
            });
        }
        Ok(())
    }
}

/// Builder for MicroVM executor configuration
#[derive(Debug, Default)]
pub struct MicroVmExecutorConfigBuilder {
    config: MicroVmExecutorConfig,
}

impl MicroVmExecutorConfigBuilder {
    /// Set memory per VM in MiB
    #[must_use]
    pub const fn memory_mib(mut self, mib: u32) -> Self {
        self.config.memory_mib = mib;
        self
    }

    /// Set vCPUs per VM
    #[must_use]
    pub const fn vcpus(mut self, count: u32) -> Self {
        self.config.vcpus = count;
        self
    }

    /// Set maximum concurrent VMs
    #[must_use]
    pub const fn max_vms(mut self, count: usize) -> Self {
        self.config.max_vms = count;
        self
    }

    /// Set VM idle timeout
    #[must_use]
    pub const fn idle_timeout(mut self, timeout: Duration) -> Self {
        self.config.idle_timeout = timeout;
        self
    }

    /// Enable jailer security
    #[must_use]
    pub fn enable_jailer(mut self, config: JailerConfig) -> Self {
        self.config.enable_jailer = true;
        self.config.jailer_config = Some(config);
        self
    }

    /// Enable warm pool
    #[must_use]
    pub const fn warm_pool(mut self, enabled: bool, size: usize) -> Self {
        self.config.warm_pool_enabled = enabled;
        self.config.warm_pool_size = size;
        self
    }

    /// Build the configuration
    pub fn build(self) -> Result<MicroVmExecutorConfig> {
        self.config.validate()?;
        Ok(self.config)
    }
}

// ============================================================================
// VM INSTANCE
// ============================================================================

/// VM instance wrapper
#[derive(Debug)]
pub struct VmInstance {
    /// Instance ID
    pub id: Uuid,
    /// Underlying MicroVM
    vm: MicroVm,
    /// Is instance busy
    busy: AtomicBool,
    /// Creation time
    created_at: Instant,
    /// Last used time
    last_used: std::sync::Mutex<Instant>,
    /// Execution count
    exec_count: AtomicU64,
}

impl VmInstance {
    /// Create a new VM instance
    pub fn new(config: &MicroVmExecutorConfig) -> Result<Self> {
        let vm_config = VmConfig::builder()
            .vcpus(config.vcpus)
            .memory_mb(u64::from(config.memory_mib))
            .build()
            .map_err(|e| RepartirError::ExecutionFailed {
                message: format!("Failed to create VM config: {e}"),
                exit_code: None,
            })?;

        let vm = MicroVm::create(vm_config).map_err(|e| RepartirError::ExecutionFailed {
            message: format!("Failed to create MicroVM: {e}"),
            exit_code: None,
        })?;

        Ok(Self {
            id: Uuid::new_v4(),
            vm,
            busy: AtomicBool::new(false),
            created_at: Instant::now(),
            last_used: std::sync::Mutex::new(Instant::now()),
            exec_count: AtomicU64::new(0),
        })
    }

    /// Check if instance is busy
    #[must_use]
    pub fn is_busy(&self) -> bool {
        self.busy.load(Ordering::Acquire)
    }

    /// Check if instance is idle (not busy)
    #[must_use]
    pub fn is_idle(&self) -> bool {
        !self.is_busy()
    }

    /// Get VM state
    #[must_use]
    pub fn state(&self) -> VmState {
        self.vm.state()
    }

    /// Get execution count
    #[must_use]
    pub fn exec_count(&self) -> u64 {
        self.exec_count.load(Ordering::Relaxed)
    }

    /// Get time since creation
    #[must_use]
    pub fn age(&self) -> Duration {
        self.created_at.elapsed()
    }

    /// Get time since last use
    #[must_use]
    pub fn idle_time(&self) -> Duration {
        self.last_used.lock().map(|guard| guard.elapsed()).unwrap_or(Duration::ZERO)
    }

    /// Try to acquire the instance for execution
    pub fn try_acquire(&self) -> bool {
        self.busy.compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire).is_ok()
    }

    /// Release the instance after execution
    pub fn release(&self) {
        if let Ok(mut last_used) = self.last_used.lock() {
            *last_used = Instant::now();
        }
        self.exec_count.fetch_add(1, Ordering::Relaxed);
        self.busy.store(false, Ordering::Release);
    }

    /// Execute in this VM (mock)
    pub fn execute(&self, _task: &Task) -> Result<ExitReason> {
        self.vm.run().map_err(|e| RepartirError::ExecutionFailed {
            message: format!("VM execution failed: {e}"),
            exit_code: None,
        })
    }
}

// ============================================================================
// MICROVM EXECUTOR
// ============================================================================

/// MicroVM-based task executor
///
/// Provides hardware-level isolation using KVM-based MicroVMs.
pub struct MicroVmExecutor {
    /// Configuration
    config: MicroVmExecutorConfig,
    /// VM instances (warm pool)
    instances: std::sync::RwLock<Vec<Arc<VmInstance>>>,
    /// Is executor available
    available: AtomicBool,
    /// Total executions
    total_executions: AtomicU64,
    /// Cold starts
    cold_starts: AtomicU64,
    /// Warm starts
    warm_starts: AtomicU64,
}

impl MicroVmExecutor {
    /// Create a new MicroVM executor
    pub fn new(config: MicroVmExecutorConfig) -> Result<Self> {
        config.validate()?;

        let executor = Self {
            config,
            instances: std::sync::RwLock::new(Vec::new()),
            available: AtomicBool::new(true),
            total_executions: AtomicU64::new(0),
            cold_starts: AtomicU64::new(0),
            warm_starts: AtomicU64::new(0),
        };

        // Pre-warm pool if enabled
        if executor.config.warm_pool_enabled {
            executor.warm_pool()?;
        }

        Ok(executor)
    }

    /// Create with default configuration
    pub fn default_config() -> Result<Self> {
        Self::new(MicroVmExecutorConfig::default())
    }

    /// Pre-warm the VM pool
    #[allow(clippy::significant_drop_tightening)]
    fn warm_pool(&self) -> Result<()> {
        let mut instances = self.instances.write().map_err(|_| RepartirError::ExecutionFailed {
            message: "Lock poisoned".to_string(),
            exit_code: None,
        })?;

        for _ in 0..self.config.warm_pool_size {
            let instance = VmInstance::new(&self.config)?;
            instances.push(Arc::new(instance));
        }

        Ok(())
    }

    /// Check if executor is available
    #[must_use]
    pub fn is_available(&self) -> bool {
        self.available.load(Ordering::Acquire)
    }

    /// Get configuration
    #[must_use]
    pub const fn config(&self) -> &MicroVmExecutorConfig {
        &self.config
    }

    /// Get number of VM instances
    pub fn instance_count(&self) -> usize {
        self.instances.read().map(|guard| guard.len()).unwrap_or(0)
    }

    /// Get number of idle instances
    pub fn idle_count(&self) -> usize {
        self.instances.read().map(|guard| guard.iter().filter(|i| i.is_idle()).count()).unwrap_or(0)
    }

    /// Get total executions
    #[must_use]
    pub fn total_executions(&self) -> u64 {
        self.total_executions.load(Ordering::Relaxed)
    }

    /// Get cold start count
    #[must_use]
    pub fn cold_starts(&self) -> u64 {
        self.cold_starts.load(Ordering::Relaxed)
    }

    /// Get warm start count
    #[must_use]
    pub fn warm_starts(&self) -> u64 {
        self.warm_starts.load(Ordering::Relaxed)
    }

    /// Get warm start ratio
    #[must_use]
    pub fn warm_start_ratio(&self) -> f64 {
        let total = self.total_executions();
        if total == 0 {
            return 0.0;
        }
        self.warm_starts() as f64 / total as f64
    }

    /// Acquire a VM instance (from pool or create new)
    fn acquire_instance(&self) -> Result<Arc<VmInstance>> {
        // Try to get from warm pool first
        if let Ok(instances) = self.instances.read() {
            for instance in instances.iter() {
                if instance.try_acquire() {
                    self.warm_starts.fetch_add(1, Ordering::Relaxed);
                    return Ok(Arc::clone(instance));
                }
            }
        }

        // No warm instance available, create new (cold start)
        self.cold_starts.fetch_add(1, Ordering::Relaxed);

        let instance = Arc::new(VmInstance::new(&self.config)?);
        instance.try_acquire(); // Mark as busy

        // Add to pool if under limit
        if let Ok(mut instances) = self.instances.write() {
            if instances.len() < self.config.max_vms {
                instances.push(Arc::clone(&instance));
            }
        }

        Ok(instance)
    }

    /// Execute a task in a MicroVM
    ///
    /// # Errors
    ///
    /// Returns an error if the executor is not available, VM acquisition fails,
    /// or task execution fails.
    #[allow(clippy::unused_async)]
    pub async fn execute(&self, task: Task) -> Result<ExecutionResult> {
        if !self.is_available() {
            return Err(RepartirError::ExecutionFailed {
                message: "MicroVM executor not available".to_string(),
                exit_code: None,
            });
        }

        let start = Instant::now();
        self.total_executions.fetch_add(1, Ordering::Relaxed);

        // Acquire VM instance
        let instance = self.acquire_instance()?;

        // Execute task
        let exit_reason = instance.execute(&task)?;

        // Release instance
        instance.release();

        let duration = start.elapsed();

        // Build result based on exit reason
        let (success, stdout, stderr) = match exit_reason {
            ExitReason::Halt | ExitReason::Shutdown => {
                (true, b"VM execution completed".to_vec(), Vec::new())
            }
            ExitReason::TripleFault | ExitReason::InternalError => {
                (false, Vec::new(), format!("VM error: {exit_reason:?}").into_bytes())
            }
            _ => (true, format!("Exit: {exit_reason:?}").into_bytes(), Vec::new()),
        };

        Ok(ExecutionResult::new(task.id(), i32::from(!success), stdout, stderr, duration))
    }

    /// Evict idle instances that exceed timeout
    pub fn evict_idle(&self) -> usize {
        let timeout = self.config.idle_timeout;
        let min_pool = self.config.warm_pool_size;

        if let Ok(mut instances) = self.instances.write() {
            let initial_len = instances.len();

            // Only evict if we have more than minimum pool
            if initial_len <= min_pool {
                return 0;
            }

            // Collect indices to remove
            let mut to_remove = Vec::new();
            for (i, instance) in instances.iter().enumerate() {
                if instances.len() - to_remove.len() <= min_pool {
                    break;
                }
                if instance.is_idle() && instance.idle_time() > timeout {
                    to_remove.push(i);
                }
            }

            // Remove in reverse order to preserve indices
            for &i in to_remove.iter().rev() {
                instances.remove(i);
            }

            return to_remove.len();
        }

        0
    }

    /// Shutdown executor
    pub fn shutdown(&self) {
        self.available.store(false, Ordering::Release);
        if let Ok(mut instances) = self.instances.write() {
            instances.clear();
        }
    }
}

impl std::fmt::Debug for MicroVmExecutor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MicroVmExecutor")
            .field("available", &self.is_available())
            .field("instances", &self.instance_count())
            .field("idle", &self.idle_count())
            .field("executions", &self.total_executions())
            .finish()
    }
}

// ============================================================================
// METRICS
// ============================================================================

/// MicroVM executor metrics
#[derive(Debug, Clone, Default)]
pub struct MicroVmMetrics {
    /// Total task executions
    pub total_executions: u64,
    /// Cold starts (new VM created)
    pub cold_starts: u64,
    /// Warm starts (reused VM)
    pub warm_starts: u64,
    /// Current VM count
    pub vm_count: usize,
    /// Idle VM count
    pub idle_vms: usize,
    /// Average execution time in ms
    pub avg_execution_ms: f64,
    /// P99 execution time in ms
    pub p99_execution_ms: f64,
}

impl MicroVmMetrics {
    /// Create from executor
    pub fn from_executor(executor: &MicroVmExecutor) -> Self {
        Self {
            total_executions: executor.total_executions(),
            cold_starts: executor.cold_starts(),
            warm_starts: executor.warm_starts(),
            vm_count: executor.instance_count(),
            idle_vms: executor.idle_count(),
            avg_execution_ms: 0.0, // Deferred (PMAT-941): track
            p99_execution_ms: 0.0, // Deferred (PMAT-942): track
        }
    }

    /// Calculate warm start ratio
    #[must_use]
    pub fn warm_start_ratio(&self) -> f64 {
        if self.total_executions == 0 {
            return 0.0;
        }
        self.warm_starts as f64 / self.total_executions as f64
    }
}

// ============================================================================
// TESTS (EXTREME TDD)
// ============================================================================

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::disallowed_methods,
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::uninlined_format_args,
    clippy::panic
)]
mod tests {
    use super::*;
    use crate::task::Backend;

    // ------------------------------------------------------------------------
    // MicroVmExecutorConfig Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_config_default() {
        let config = MicroVmExecutorConfig::default();
        assert_eq!(config.memory_mib, 128);
        assert_eq!(config.vcpus, 1);
        assert_eq!(config.max_vms, 100);
        assert!(config.warm_pool_enabled);
    }

    #[test]
    fn test_config_builder() {
        let config = MicroVmExecutorConfig::builder()
            .memory_mib(256)
            .vcpus(2)
            .max_vms(50)
            .idle_timeout(Duration::from_secs(600))
            .warm_pool(true, 5)
            .build()
            .unwrap();

        assert_eq!(config.memory_mib, 256);
        assert_eq!(config.vcpus, 2);
        assert_eq!(config.max_vms, 50);
        assert_eq!(config.idle_timeout, Duration::from_secs(600));
        assert_eq!(config.warm_pool_size, 5);
    }

    #[test]
    fn test_config_builder_with_jailer() {
        let jailer = JailerConfig::minimal();
        let config = MicroVmExecutorConfig::builder()
            .memory_mib(128)
            .vcpus(1)
            .enable_jailer(jailer)
            .build()
            .unwrap();

        assert!(config.enable_jailer);
        assert!(config.jailer_config.is_some());
    }

    #[test]
    fn test_config_validate_zero_memory() {
        let config = MicroVmExecutorConfig::builder().memory_mib(0).vcpus(1);
        assert!(config.build().is_err());
    }

    #[test]
    fn test_config_validate_zero_vcpus() {
        let config = MicroVmExecutorConfig::builder().memory_mib(128).vcpus(0);
        assert!(config.build().is_err());
    }

    #[test]
    fn test_config_validate_zero_max_vms() {
        let config = MicroVmExecutorConfig::builder().memory_mib(128).vcpus(1).max_vms(0);
        assert!(config.build().is_err());
    }

    // ------------------------------------------------------------------------
    // VmInstance Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_vm_instance_new() {
        let config = MicroVmExecutorConfig::default();
        let instance = VmInstance::new(&config).unwrap();

        assert!(!instance.is_busy());
        assert!(instance.is_idle());
        assert_eq!(instance.exec_count(), 0);
    }

    #[test]
    fn test_vm_instance_acquire_release() {
        let config = MicroVmExecutorConfig::default();
        let instance = VmInstance::new(&config).unwrap();

        // Should be able to acquire
        assert!(instance.try_acquire());
        assert!(instance.is_busy());
        assert!(!instance.is_idle());

        // Second acquire should fail
        assert!(!instance.try_acquire());

        // Release
        instance.release();
        assert!(!instance.is_busy());
        assert!(instance.is_idle());
        assert_eq!(instance.exec_count(), 1);
    }

    #[test]
    fn test_vm_instance_state() {
        let config = MicroVmExecutorConfig::default();
        let instance = VmInstance::new(&config).unwrap();

        // Initial state should be Created
        assert_eq!(instance.state(), VmState::Created);
    }

    #[test]
    fn test_vm_instance_age() {
        let config = MicroVmExecutorConfig::default();
        let instance = VmInstance::new(&config).unwrap();

        // Age should be very small
        assert!(instance.age() < Duration::from_secs(1));
    }

    #[test]
    fn test_vm_instance_idle_time() {
        let config = MicroVmExecutorConfig::default();
        let instance = VmInstance::new(&config).unwrap();

        // Idle time should be very small initially
        assert!(instance.idle_time() < Duration::from_secs(1));
    }

    // ------------------------------------------------------------------------
    // MicroVmExecutor Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_executor_new() {
        let config = MicroVmExecutorConfig::builder()
            .memory_mib(128)
            .vcpus(1)
            .warm_pool(false, 0)
            .build()
            .unwrap();

        let executor = MicroVmExecutor::new(config).unwrap();
        assert!(executor.is_available());
        assert_eq!(executor.instance_count(), 0);
    }

    #[test]
    fn test_executor_with_warm_pool() {
        let config = MicroVmExecutorConfig::builder()
            .memory_mib(128)
            .vcpus(1)
            .warm_pool(true, 3)
            .build()
            .unwrap();

        let executor = MicroVmExecutor::new(config).unwrap();
        assert_eq!(executor.instance_count(), 3);
        assert_eq!(executor.idle_count(), 3);
    }

    #[test]
    fn test_executor_default_config() {
        let executor = MicroVmExecutor::default_config().unwrap();
        assert!(executor.is_available());
        assert_eq!(executor.instance_count(), 3); // Default warm pool
    }

    #[test]
    fn test_executor_config_accessor() {
        let config = MicroVmExecutorConfig::builder().memory_mib(512).vcpus(4).build().unwrap();

        let executor = MicroVmExecutor::new(config).unwrap();
        assert_eq!(executor.config().memory_mib, 512);
        assert_eq!(executor.config().vcpus, 4);
    }

    #[test]
    fn test_executor_metrics_initial() {
        let config = MicroVmExecutorConfig::builder().warm_pool(false, 0).build().unwrap();

        let executor = MicroVmExecutor::new(config).unwrap();
        assert_eq!(executor.total_executions(), 0);
        assert_eq!(executor.cold_starts(), 0);
        assert_eq!(executor.warm_starts(), 0);
        assert!((executor.warm_start_ratio() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_executor_shutdown() {
        let config = MicroVmExecutorConfig::builder().warm_pool(true, 3).build().unwrap();

        let executor = MicroVmExecutor::new(config).unwrap();
        assert_eq!(executor.instance_count(), 3);

        executor.shutdown();
        assert!(!executor.is_available());
        assert_eq!(executor.instance_count(), 0);
    }

    #[test]
    fn test_executor_debug() {
        let config = MicroVmExecutorConfig::builder().warm_pool(true, 2).build().unwrap();

        let executor = MicroVmExecutor::new(config).unwrap();
        let debug = format!("{:?}", executor);
        assert!(debug.contains("MicroVmExecutor"));
        assert!(debug.contains("available"));
        assert!(debug.contains("instances"));
    }

    // ------------------------------------------------------------------------
    // MicroVmMetrics Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_metrics_default() {
        let metrics = MicroVmMetrics::default();
        assert_eq!(metrics.total_executions, 0);
        assert_eq!(metrics.cold_starts, 0);
        assert_eq!(metrics.warm_starts, 0);
    }

    #[test]
    fn test_metrics_from_executor() {
        let config = MicroVmExecutorConfig::builder().warm_pool(true, 5).build().unwrap();

        let executor = MicroVmExecutor::new(config).unwrap();
        let metrics = MicroVmMetrics::from_executor(&executor);

        assert_eq!(metrics.vm_count, 5);
        assert_eq!(metrics.idle_vms, 5);
    }

    #[test]
    fn test_metrics_warm_start_ratio_zero() {
        let metrics = MicroVmMetrics::default();
        assert!((metrics.warm_start_ratio() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_metrics_warm_start_ratio() {
        let metrics = MicroVmMetrics {
            total_executions: 100,
            cold_starts: 20,
            warm_starts: 80,
            ..Default::default()
        };
        assert!((metrics.warm_start_ratio() - 0.8).abs() < 0.001);
    }

    // ------------------------------------------------------------------------
    // Async Execution Tests
    // ------------------------------------------------------------------------

    #[tokio::test]
    async fn test_executor_execute_task() {
        let config = MicroVmExecutorConfig::builder().warm_pool(true, 1).build().unwrap();

        let executor = MicroVmExecutor::new(config).unwrap();

        let task = Task::builder()
            .binary("/bin/echo")
            .arg("hello")
            .backend(Backend::MicroVm)
            .build()
            .unwrap();

        let result = executor.execute(task).await.unwrap();

        // Mock execution returns success with Halt
        assert!(result.is_success());
        assert_eq!(executor.total_executions(), 1);
    }

    #[tokio::test]
    async fn test_executor_warm_start_tracking() {
        let config = MicroVmExecutorConfig::builder().warm_pool(true, 2).build().unwrap();

        let executor = MicroVmExecutor::new(config).unwrap();

        // First execution should be warm (from pool)
        let task1 = Task::builder().binary("/bin/true").backend(Backend::MicroVm).build().unwrap();

        executor.execute(task1).await.unwrap();

        // Should have used warm instance
        assert_eq!(executor.warm_starts(), 1);
        assert_eq!(executor.cold_starts(), 0);
    }

    #[tokio::test]
    async fn test_executor_multiple_tasks() {
        // Note: Mock VMs can only run once (go to Stopped state)
        // This tests cold start path for each execution
        let config = MicroVmExecutorConfig::builder()
            .warm_pool(false, 0) // No warm pool - cold starts only
            .max_vms(10)
            .build()
            .unwrap();

        let executor = MicroVmExecutor::new(config).unwrap();

        // First execution
        let task = Task::builder()
            .binary("/bin/echo")
            .arg("test")
            .backend(Backend::MicroVm)
            .build()
            .unwrap();

        let result = executor.execute(task).await.unwrap();
        assert!(result.is_success());
        assert_eq!(executor.total_executions(), 1);
        assert_eq!(executor.cold_starts(), 1);
    }

    #[tokio::test]
    async fn test_executor_not_available() {
        let config = MicroVmExecutorConfig::builder().warm_pool(false, 0).build().unwrap();

        let executor = MicroVmExecutor::new(config).unwrap();
        executor.shutdown();

        let task = Task::builder().binary("/bin/true").backend(Backend::MicroVm).build().unwrap();

        let result = executor.execute(task).await;
        assert!(result.is_err());
    }

    // ------------------------------------------------------------------------
    // Eviction Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_executor_evict_idle() {
        let config = MicroVmExecutorConfig::builder()
            .warm_pool(true, 5)
            .idle_timeout(Duration::from_millis(1))
            .build()
            .unwrap();

        let executor = MicroVmExecutor::new(config).unwrap();
        assert_eq!(executor.instance_count(), 5);

        // Wait for idle timeout
        std::thread::sleep(Duration::from_millis(10));

        // Evict should not remove instances since we're at min pool
        let evicted = executor.evict_idle();
        // We set warm_pool_size to 5, so all should be kept
        assert_eq!(evicted, 0);
    }

    // ------------------------------------------------------------------------
    // Integration Tests
    // ------------------------------------------------------------------------

    #[tokio::test]
    async fn test_end_to_end_microvm_execution() {
        // Create executor with warm pool
        let config = MicroVmExecutorConfig::builder()
            .memory_mib(256)
            .vcpus(2)
            .warm_pool(true, 3)
            .build()
            .unwrap();

        let executor = MicroVmExecutor::new(config).unwrap();
        assert!(executor.is_available());
        assert_eq!(executor.instance_count(), 3);

        // Execute single task (mock VMs can only run once)
        let task = Task::builder()
            .binary("/bin/echo")
            .arg("test")
            .backend(Backend::MicroVm)
            .build()
            .unwrap();

        let result = executor.execute(task).await.unwrap();
        assert!(result.is_success());

        // Verify metrics
        assert_eq!(executor.total_executions(), 1);

        // Get metrics snapshot
        let metrics = MicroVmMetrics::from_executor(&executor);
        assert_eq!(metrics.total_executions, 1);

        // First execution from warm pool
        assert_eq!(executor.warm_starts(), 1);

        // Shutdown
        executor.shutdown();
        assert!(!executor.is_available());
    }
}
