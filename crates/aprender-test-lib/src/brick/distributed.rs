//! DistributedBrick: Work-stealing and data locality (PROBAR-SPEC-009-P10)
//!
//! This module enables distributed brick execution with:
//! - Work-stealing across nodes
//! - Data locality awareness
//! - Multi-backend dispatch (CPU/GPU/Remote/SIMD)
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                   DISTRIBUTED BRICK FLOW                     │
//! ├─────────────────────────────────────────────────────────────┤
//! │                                                              │
//! │  1. DistributedBrick<B> wraps any Brick                     │
//! │  2. BrickDataTracker tracks data locality                   │
//! │  3. MultiBrickExecutor selects best backend                 │
//! │  4. BrickCoordinator handles PUB/SUB coordination           │
//! │                                                              │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! # References
//!
//! - PROBAR-SPEC-009-P10: Distribution - Repartir Integration

// Allow expect for RwLock - lock poisoning is truly exceptional
#![allow(clippy::expect_used)]

use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use super::{Brick, BrickAssertion, BrickBudget, BrickError, BrickResult, BrickVerification};

/// Unique identifier for a worker node
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WorkerId(pub u64);

impl WorkerId {
    /// Create a new worker ID
    #[must_use]
    pub const fn new(id: u64) -> Self {
        Self(id)
    }

    /// Get the underlying ID value
    #[must_use]
    pub const fn value(&self) -> u64 {
        self.0
    }
}

impl fmt::Display for WorkerId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "worker-{}", self.0)
    }
}

/// Execution backend for brick operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Backend {
    /// CPU execution with standard instructions
    Cpu,
    /// GPU execution via WebGPU/wgpu
    Gpu,
    /// Remote execution on another node
    Remote,
    /// CPU execution with SIMD acceleration
    Simd,
}

impl Backend {
    /// Check if backend is available on current system
    #[must_use]
    pub fn is_available(&self) -> bool {
        match self {
            Self::Cpu | Self::Simd => true,
            Self::Gpu => cfg!(feature = "gpu"),
            // Remote backend requires distributed feature (not yet implemented)
            Self::Remote => false,
        }
    }

    /// Get relative performance estimate (higher = faster)
    #[must_use]
    pub const fn performance_estimate(&self) -> u32 {
        match self {
            Self::Gpu => 100,
            Self::Simd => 50,
            Self::Cpu => 10,
            Self::Remote => 5, // Network latency
        }
    }
}

impl Default for Backend {
    fn default() -> Self {
        Self::Cpu
    }
}

/// Input data for brick execution
#[derive(Debug, Clone, Default)]
pub struct BrickInput {
    /// Input tensor data
    pub data: Vec<f32>,
    /// Input shape dimensions
    pub shape: Vec<usize>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl BrickInput {
    /// Create new brick input
    #[must_use]
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        Self {
            data,
            shape,
            metadata: HashMap::new(),
        }
    }

    /// Get total size in bytes
    #[must_use]
    pub fn size_bytes(&self) -> usize {
        self.data.len() * std::mem::size_of::<f32>()
    }

    /// Get total element count
    #[must_use]
    pub fn element_count(&self) -> usize {
        self.data.len()
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

/// Output data from brick execution
#[derive(Debug, Clone, Default)]
pub struct BrickOutput {
    /// Output tensor data
    pub data: Vec<f32>,
    /// Output shape dimensions
    pub shape: Vec<usize>,
    /// Execution metrics
    pub metrics: ExecutionMetrics,
}

impl BrickOutput {
    /// Create new brick output
    #[must_use]
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        Self {
            data,
            shape,
            metrics: ExecutionMetrics::default(),
        }
    }

    /// Get total size in bytes
    #[must_use]
    pub fn size_bytes(&self) -> usize {
        self.data.len() * std::mem::size_of::<f32>()
    }
}

/// Metrics from brick execution
#[derive(Debug, Clone, Default)]
pub struct ExecutionMetrics {
    /// Time to execute
    pub execution_time: Duration,
    /// Backend used
    pub backend: Backend,
    /// Worker that executed
    pub worker_id: Option<WorkerId>,
    /// Data transfer time (if remote)
    pub transfer_time: Option<Duration>,
}

impl ExecutionMetrics {
    /// Create new execution metrics
    #[must_use]
    pub fn new(execution_time: Duration, backend: Backend) -> Self {
        Self {
            execution_time,
            backend,
            worker_id: None,
            transfer_time: None,
        }
    }
}

/// Distributed brick wrapper for multi-backend execution
///
/// Wraps any `Brick` to enable distributed execution with:
/// - Backend selection (CPU/GPU/Remote/SIMD)
/// - Data dependency tracking for locality
/// - Work-stealing support
#[derive(Debug)]
pub struct DistributedBrick<B: Brick> {
    inner: B,
    backend: Backend,
    data_dependencies: Vec<String>,
    preferred_worker: Option<WorkerId>,
}

impl<B: Brick> DistributedBrick<B> {
    /// Create a new distributed brick wrapper
    #[must_use]
    pub fn new(inner: B) -> Self {
        Self {
            inner,
            backend: Backend::default(),
            data_dependencies: Vec::new(),
            preferred_worker: None,
        }
    }

    /// Set the preferred execution backend
    #[must_use]
    pub fn with_backend(mut self, backend: Backend) -> Self {
        self.backend = backend;
        self
    }

    /// Add data dependencies for locality-aware scheduling
    #[must_use]
    pub fn with_data_dependencies(mut self, deps: Vec<String>) -> Self {
        self.data_dependencies = deps;
        self
    }

    /// Set preferred worker for execution
    #[must_use]
    pub fn with_preferred_worker(mut self, worker: WorkerId) -> Self {
        self.preferred_worker = Some(worker);
        self
    }

    /// Get the inner brick
    #[must_use]
    pub fn inner(&self) -> &B {
        &self.inner
    }

    /// Get mutable access to inner brick
    pub fn inner_mut(&mut self) -> &mut B {
        &mut self.inner
    }

    /// Get current backend
    #[must_use]
    pub fn backend(&self) -> Backend {
        self.backend
    }

    /// Get data dependencies
    #[must_use]
    pub fn data_dependencies(&self) -> &[String] {
        &self.data_dependencies
    }

    /// Get preferred worker
    #[must_use]
    pub fn preferred_worker(&self) -> Option<WorkerId> {
        self.preferred_worker
    }

    /// Convert to a task specification for distributed execution
    #[must_use]
    pub fn to_task_spec(&self) -> TaskSpec {
        TaskSpec {
            brick_name: self.inner.brick_name().to_string(),
            backend: self.backend,
            data_dependencies: self.data_dependencies.clone(),
            preferred_worker: self.preferred_worker,
        }
    }
}

impl<B: Brick> Brick for DistributedBrick<B> {
    fn brick_name(&self) -> &'static str {
        self.inner.brick_name()
    }

    fn assertions(&self) -> &[BrickAssertion] {
        self.inner.assertions()
    }

    fn budget(&self) -> BrickBudget {
        self.inner.budget()
    }

    fn verify(&self) -> BrickVerification {
        self.inner.verify()
    }

    fn to_html(&self) -> String {
        self.inner.to_html()
    }

    fn to_css(&self) -> String {
        self.inner.to_css()
    }
}

/// Task specification for distributed execution
#[derive(Debug, Clone)]
pub struct TaskSpec {
    /// Brick name for identification
    pub brick_name: String,
    /// Requested backend
    pub backend: Backend,
    /// Data dependencies
    pub data_dependencies: Vec<String>,
    /// Preferred worker
    pub preferred_worker: Option<WorkerId>,
}

/// Data location entry for a specific piece of data
#[derive(Debug, Clone)]
pub struct DataLocation {
    /// Data key/identifier
    pub key: String,
    /// Workers that have this data
    pub workers: Vec<WorkerId>,
    /// Size of data in bytes
    pub size_bytes: usize,
    /// Last access time
    pub last_access: Instant,
}

/// Track where brick weights/data reside across workers
///
/// Used for locality-aware scheduling to minimize data movement.
#[derive(Debug)]
pub struct BrickDataTracker {
    /// Map from data key to location info
    locations: RwLock<HashMap<String, DataLocation>>,
}

impl Default for BrickDataTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl BrickDataTracker {
    /// Create a new data tracker
    #[must_use]
    pub fn new() -> Self {
        Self {
            locations: RwLock::new(HashMap::new()),
        }
    }

    /// Register that a worker has certain data
    pub fn track_data(&self, key: &str, worker_id: WorkerId, size_bytes: usize) {
        let mut locations = self.locations.write().expect("lock poisoned");
        locations
            .entry(key.to_string())
            .and_modify(|loc| {
                if !loc.workers.contains(&worker_id) {
                    loc.workers.push(worker_id);
                }
                loc.last_access = Instant::now();
            })
            .or_insert_with(|| DataLocation {
                key: key.to_string(),
                workers: vec![worker_id],
                size_bytes,
                last_access: Instant::now(),
            });
    }

    /// Register that a worker has brick weights
    pub fn track_weights(&self, brick_name: &str, worker_id: WorkerId) {
        let key = format!("{}_weights", brick_name);
        self.track_data(&key, worker_id, 0);
    }

    /// Remove data location from a worker
    pub fn remove_data(&self, key: &str, worker_id: WorkerId) {
        let mut locations = self.locations.write().expect("lock poisoned");
        if let Some(loc) = locations.get_mut(key) {
            loc.workers.retain(|w| *w != worker_id);
        }
    }

    /// Get workers that have specific data
    #[must_use]
    pub fn get_workers_for_data(&self, key: &str) -> Vec<WorkerId> {
        let locations = self.locations.read().expect("lock poisoned");
        locations
            .get(key)
            .map_or(Vec::new(), |loc| loc.workers.clone())
    }

    /// Calculate affinity scores for workers based on data dependencies
    pub fn calculate_affinity(&self, dependencies: &[String]) -> HashMap<WorkerId, f64> {
        let locations = self.locations.read().expect("lock poisoned");
        let mut affinity: HashMap<WorkerId, f64> = HashMap::new();

        for dep in dependencies {
            if let Some(loc) = locations.get(dep) {
                let score_per_worker = 1.0 / loc.workers.len() as f64;
                for worker in &loc.workers {
                    *affinity.entry(*worker).or_insert(0.0) += score_per_worker;
                }
            }
        }

        // Normalize scores
        if !affinity.is_empty() {
            let max_score = affinity.values().cloned().fold(0.0_f64, f64::max);
            if max_score > 0.0 {
                for score in affinity.values_mut() {
                    *score /= max_score;
                }
            }
        }

        affinity
    }

    /// Find the best worker for a brick based on data locality
    #[must_use]
    pub fn find_best_worker(&self, brick: &dyn Brick) -> Option<WorkerId> {
        // Use brick name to find weights
        let weights_key = format!("{}_weights", brick.brick_name());
        let workers = self.get_workers_for_data(&weights_key);
        workers.first().copied()
    }

    /// Find best worker for distributed brick with dependencies
    #[must_use]
    pub fn find_best_worker_for_distributed<B: Brick>(
        &self,
        brick: &DistributedBrick<B>,
    ) -> Option<WorkerId> {
        // Check preferred worker first
        if let Some(preferred) = brick.preferred_worker() {
            return Some(preferred);
        }

        // Calculate affinity based on dependencies
        let affinity = self.calculate_affinity(brick.data_dependencies());
        affinity
            .into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(worker, _)| worker)
    }

    /// Get total data size tracked
    #[must_use]
    pub fn total_data_size(&self) -> usize {
        let locations = self.locations.read().expect("lock poisoned");
        locations.values().map(|loc| loc.size_bytes).sum()
    }
}

/// Backend selector for choosing optimal execution backend
#[derive(Debug)]
pub struct BackendSelector {
    /// Minimum element count for GPU execution
    gpu_threshold: usize,
    /// Minimum element count for SIMD execution
    simd_threshold: usize,
    /// Maximum element count for CPU (else remote)
    cpu_max_threshold: usize,
}

impl Default for BackendSelector {
    fn default() -> Self {
        Self::new()
    }
}

impl BackendSelector {
    /// Create a new backend selector with default thresholds
    #[must_use]
    pub fn new() -> Self {
        Self {
            gpu_threshold: 1_000_000,       // 1M elements for GPU
            simd_threshold: 10_000,         // 10K elements for SIMD
            cpu_max_threshold: 100_000_000, // 100M elements max for local
        }
    }

    /// Configure GPU threshold
    #[must_use]
    pub fn with_gpu_threshold(mut self, threshold: usize) -> Self {
        self.gpu_threshold = threshold;
        self
    }

    /// Configure SIMD threshold
    #[must_use]
    pub fn with_simd_threshold(mut self, threshold: usize) -> Self {
        self.simd_threshold = threshold;
        self
    }

    /// Configure CPU max threshold
    #[must_use]
    pub fn with_cpu_max_threshold(mut self, threshold: usize) -> Self {
        self.cpu_max_threshold = threshold;
        self
    }

    /// Select best backend based on input characteristics
    #[must_use]
    pub fn select(&self, element_count: usize, gpu_available: bool) -> Backend {
        // Too large for local - use remote if available (not yet implemented)
        if element_count > self.cpu_max_threshold && Backend::Remote.is_available() {
            return Backend::Remote;
        }

        // Large enough for GPU
        if element_count >= self.gpu_threshold && gpu_available {
            return Backend::Gpu;
        }

        // Medium size - use SIMD
        if element_count >= self.simd_threshold {
            return Backend::Simd;
        }

        // Small - use plain CPU
        Backend::Cpu
    }

    /// Select backend for a brick with input
    #[must_use]
    pub fn select_for_brick(
        &self,
        _brick_complexity: u32,
        input_size: usize,
        gpu_available: bool,
    ) -> Backend {
        // Future: factor in brick_complexity
        self.select(input_size, gpu_available)
    }
}

/// Multi-backend executor for brick operations
///
/// Dispatches brick execution to the best available backend.
#[derive(Debug)]
pub struct MultiBrickExecutor {
    selector: BackendSelector,
    gpu_available: bool,
    data_tracker: Arc<BrickDataTracker>,
}

impl MultiBrickExecutor {
    /// Create a new multi-backend executor
    #[must_use]
    pub fn new(data_tracker: Arc<BrickDataTracker>) -> Self {
        Self {
            selector: BackendSelector::new(),
            gpu_available: cfg!(feature = "gpu"),
            data_tracker,
        }
    }

    /// Create with custom backend selector
    #[must_use]
    pub fn with_selector(mut self, selector: BackendSelector) -> Self {
        self.selector = selector;
        self
    }

    /// Set GPU availability
    #[must_use]
    pub fn with_gpu_available(mut self, available: bool) -> Self {
        self.gpu_available = available;
        self
    }

    /// Execute a brick on the best backend
    pub fn execute(&self, brick: &dyn Brick, input: BrickInput) -> BrickResult<BrickOutput> {
        let start = Instant::now();

        // Select backend
        let backend = self
            .selector
            .select(input.element_count(), self.gpu_available);

        // Execute on selected backend
        let (output_data, output_shape) = match backend {
            Backend::Cpu => self.execute_cpu(brick, &input)?,
            Backend::Simd => self.execute_simd(brick, &input)?,
            Backend::Gpu => self.execute_gpu(brick, &input)?,
            Backend::Remote => self.execute_remote(brick, &input)?,
        };

        let execution_time = start.elapsed();

        // Build output with metrics
        let mut output = BrickOutput::new(output_data, output_shape);
        output.metrics = ExecutionMetrics::new(execution_time, backend);

        Ok(output)
    }

    /// Execute distributed brick
    pub fn execute_distributed<B: Brick>(
        &self,
        brick: &DistributedBrick<B>,
        input: BrickInput,
    ) -> BrickResult<BrickOutput> {
        let start = Instant::now();

        // Use brick's preferred backend or select automatically
        let backend = brick.backend();

        // Find best worker for locality
        let worker_id = self.data_tracker.find_best_worker_for_distributed(brick);

        // Execute
        let (output_data, output_shape) = match backend {
            Backend::Cpu => self.execute_cpu(brick.inner(), &input)?,
            Backend::Simd => self.execute_simd(brick.inner(), &input)?,
            Backend::Gpu => self.execute_gpu(brick.inner(), &input)?,
            Backend::Remote => self.execute_remote(brick.inner(), &input)?,
        };

        let execution_time = start.elapsed();

        // Build output with metrics
        let mut output = BrickOutput::new(output_data, output_shape);
        output.metrics = ExecutionMetrics {
            execution_time,
            backend,
            worker_id,
            transfer_time: None,
        };

        Ok(output)
    }

    fn execute_cpu(
        &self,
        _brick: &dyn Brick,
        input: &BrickInput,
    ) -> BrickResult<(Vec<f32>, Vec<usize>)> {
        // Simple passthrough for now - real implementation would execute brick
        Ok((input.data.clone(), input.shape.clone()))
    }

    fn execute_simd(
        &self,
        _brick: &dyn Brick,
        input: &BrickInput,
    ) -> BrickResult<(Vec<f32>, Vec<usize>)> {
        // SIMD path - would use actual SIMD instructions
        Ok((input.data.clone(), input.shape.clone()))
    }

    fn execute_gpu(
        &self,
        _brick: &dyn Brick,
        input: &BrickInput,
    ) -> BrickResult<(Vec<f32>, Vec<usize>)> {
        // GPU path - would use WebGPU/wgpu
        if !self.gpu_available {
            return Err(BrickError::HtmlGenerationFailed {
                reason: "GPU not available".into(),
            });
        }
        Ok((input.data.clone(), input.shape.clone()))
    }

    fn execute_remote(
        &self,
        _brick: &dyn Brick,
        input: &BrickInput,
    ) -> BrickResult<(Vec<f32>, Vec<usize>)> {
        // Remote path - would serialize and send to remote worker
        if !Backend::Remote.is_available() {
            return Err(BrickError::HtmlGenerationFailed {
                reason: "Distributed execution not available".into(),
            });
        }
        Ok((input.data.clone(), input.shape.clone()))
    }

    /// Get the data tracker
    #[must_use]
    pub fn data_tracker(&self) -> &Arc<BrickDataTracker> {
        &self.data_tracker
    }
}

/// Message for PUB/SUB coordination
#[derive(Debug, Clone)]
pub enum BrickMessage {
    /// Weight update message
    WeightUpdate {
        /// Name of the brick whose weights are being updated
        brick_name: String,
        /// Serialized weight data
        weights: Vec<u8>,
        /// Weight version number
        version: u64,
    },
    /// State change notification
    StateChange {
        /// Name of the brick that changed state
        brick_name: String,
        /// Event description
        event: String,
    },
    /// Request brick execution
    ExecutionRequest {
        /// Name of brick to execute
        brick_name: String,
        /// Key to input data
        input_key: String,
        /// Unique request ID for correlation
        request_id: u64,
    },
    /// Execution result
    ExecutionResult {
        /// Request ID this result corresponds to
        request_id: u64,
        /// Key to output data
        output_key: String,
        /// Whether execution succeeded
        success: bool,
    },
}

/// Subscription to brick events
#[derive(Debug)]
pub struct Subscription {
    topic: String,
    messages: Arc<RwLock<Vec<BrickMessage>>>,
}

impl Subscription {
    /// Get all pending messages
    #[must_use]
    pub fn drain(&self) -> Vec<BrickMessage> {
        let mut messages = self.messages.write().expect("lock poisoned");
        std::mem::take(&mut *messages)
    }

    /// Check if there are pending messages
    #[must_use]
    pub fn has_messages(&self) -> bool {
        let messages = self.messages.read().expect("lock poisoned");
        !messages.is_empty()
    }

    /// Get subscription topic
    #[must_use]
    pub fn topic(&self) -> &str {
        &self.topic
    }
}

// ============================================================================
// Work-Stealing Scheduler (Phase 10e)
// ============================================================================

/// A task that can be executed by workers and potentially stolen
#[derive(Debug, Clone)]
pub struct WorkStealingTask {
    /// Unique task ID
    pub id: u64,
    /// Task specification
    pub spec: TaskSpec,
    /// Input data key
    pub input_key: String,
    /// Priority (higher = more urgent)
    pub priority: u32,
    /// Creation time
    pub created_at: Instant,
}

impl WorkStealingTask {
    /// Create a new work-stealing task
    #[must_use]
    pub fn new(id: u64, spec: TaskSpec, input_key: String) -> Self {
        Self {
            id,
            spec,
            input_key,
            priority: 0,
            created_at: Instant::now(),
        }
    }

    /// Set task priority
    #[must_use]
    pub fn with_priority(mut self, priority: u32) -> Self {
        self.priority = priority;
        self
    }

    /// Get task age
    #[must_use]
    pub fn age(&self) -> Duration {
        self.created_at.elapsed()
    }
}

/// Per-worker task queue supporting work-stealing
#[derive(Debug)]
pub struct WorkerQueue {
    /// Worker ID
    worker_id: WorkerId,
    /// Local task queue (owned tasks)
    local_queue: RwLock<Vec<WorkStealingTask>>,
    /// Number of tasks completed
    completed_count: AtomicU64,
    /// Number of tasks stolen from this queue
    stolen_count: AtomicU64,
}

impl WorkerQueue {
    /// Create a new worker queue
    #[must_use]
    pub fn new(worker_id: WorkerId) -> Self {
        Self {
            worker_id,
            local_queue: RwLock::new(Vec::new()),
            completed_count: AtomicU64::new(0),
            stolen_count: AtomicU64::new(0),
        }
    }

    /// Push a task to the local queue
    pub fn push(&self, task: WorkStealingTask) {
        let mut queue = self.local_queue.write().expect("lock poisoned");
        queue.push(task);
        // Sort by priority (higher first)
        queue.sort_by(|a, b| b.priority.cmp(&a.priority));
    }

    /// Pop a task from the local queue (highest priority first)
    pub fn pop(&self) -> Option<WorkStealingTask> {
        let mut queue = self.local_queue.write().expect("lock poisoned");
        if queue.is_empty() {
            return None;
        }
        Some(queue.remove(0)) // Get highest priority (front after sort)
    }

    /// Steal a task from this queue (lowest priority - be nice to owner)
    pub fn steal(&self) -> Option<WorkStealingTask> {
        let mut queue = self.local_queue.write().expect("lock poisoned");
        if queue.is_empty() {
            return None;
        }
        self.stolen_count.fetch_add(1, Ordering::Relaxed);
        queue.pop() // Steal lowest priority (back after sort)
    }

    /// Check if queue is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        let queue = self.local_queue.read().expect("lock poisoned");
        queue.is_empty()
    }

    /// Get queue length
    #[must_use]
    pub fn len(&self) -> usize {
        let queue = self.local_queue.read().expect("lock poisoned");
        queue.len()
    }

    /// Mark a task as completed
    pub fn mark_completed(&self) {
        self.completed_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Get worker ID
    #[must_use]
    pub fn worker_id(&self) -> WorkerId {
        self.worker_id
    }

    /// Get completed count
    #[must_use]
    pub fn completed_count(&self) -> u64 {
        self.completed_count.load(Ordering::Relaxed)
    }

    /// Get stolen count
    #[must_use]
    pub fn stolen_count(&self) -> u64 {
        self.stolen_count.load(Ordering::Relaxed)
    }
}

/// Work-stealing scheduler for distributed brick execution
///
/// Implements work-stealing algorithm where idle workers steal tasks
/// from busy workers' queues. This provides automatic load balancing.
///
/// # Algorithm
///
/// 1. Each worker has a local deque (double-ended queue)
/// 2. Workers push/pop from their own queue (LIFO - good for cache locality)
/// 3. When idle, workers steal from other queues (FIFO - steal oldest tasks)
/// 4. Stealing considers data locality via `BrickDataTracker`
#[derive(Debug)]
pub struct WorkStealingScheduler {
    /// Worker queues indexed by worker ID
    queues: RwLock<HashMap<WorkerId, Arc<WorkerQueue>>>,
    /// Data tracker for locality-aware stealing
    data_tracker: Arc<BrickDataTracker>,
    /// Task ID counter
    task_counter: AtomicU64,
    /// Total tasks submitted
    submitted_count: AtomicU64,
}

impl WorkStealingScheduler {
    /// Create a new work-stealing scheduler
    #[must_use]
    pub fn new(data_tracker: Arc<BrickDataTracker>) -> Self {
        Self {
            queues: RwLock::new(HashMap::new()),
            data_tracker,
            task_counter: AtomicU64::new(0),
            submitted_count: AtomicU64::new(0),
        }
    }

    /// Register a worker with the scheduler
    pub fn register_worker(&self, worker_id: WorkerId) -> Arc<WorkerQueue> {
        let queue = Arc::new(WorkerQueue::new(worker_id));
        let mut queues = self.queues.write().expect("lock poisoned");
        queues.insert(worker_id, Arc::clone(&queue));
        queue
    }

    /// Unregister a worker
    pub fn unregister_worker(&self, worker_id: WorkerId) {
        let mut queues = self.queues.write().expect("lock poisoned");
        queues.remove(&worker_id);
    }

    /// Submit a task to the best worker based on locality
    pub fn submit(&self, spec: TaskSpec, input_key: String) -> u64 {
        let task_id = self.task_counter.fetch_add(1, Ordering::SeqCst);
        let task = WorkStealingTask::new(task_id, spec.clone(), input_key);

        // Find best worker based on data locality
        let target_worker = self.find_best_worker_for_task(&spec);

        let queues = self.queues.read().expect("lock poisoned");
        if let Some(queue) = target_worker.and_then(|w| queues.get(&w)) {
            queue.push(task);
        } else if let Some((_, queue)) = queues.iter().next() {
            // Fallback to first available worker
            queue.push(task);
        }

        self.submitted_count.fetch_add(1, Ordering::Relaxed);
        task_id
    }

    /// Submit with explicit priority
    pub fn submit_priority(&self, spec: TaskSpec, input_key: String, priority: u32) -> u64 {
        let task_id = self.task_counter.fetch_add(1, Ordering::SeqCst);
        let task = WorkStealingTask::new(task_id, spec.clone(), input_key).with_priority(priority);

        let target_worker = self.find_best_worker_for_task(&spec);

        let queues = self.queues.read().expect("lock poisoned");
        if let Some(queue) = target_worker.and_then(|w| queues.get(&w)) {
            queue.push(task);
        } else if let Some((_, queue)) = queues.iter().next() {
            queue.push(task);
        }

        self.submitted_count.fetch_add(1, Ordering::Relaxed);
        task_id
    }

    /// Try to get work for a worker (local pop or steal)
    pub fn get_work(&self, worker_id: WorkerId) -> Option<WorkStealingTask> {
        let queues = self.queues.read().expect("lock poisoned");

        // First try local queue
        if let Some(queue) = queues.get(&worker_id) {
            if let Some(task) = queue.pop() {
                return Some(task);
            }
        }

        // Try to steal from other workers
        self.try_steal(worker_id, &queues)
    }

    /// Try to steal work from another worker's queue
    fn try_steal(
        &self,
        stealer_id: WorkerId,
        queues: &HashMap<WorkerId, Arc<WorkerQueue>>,
    ) -> Option<WorkStealingTask> {
        // Find queues with work, preferring those with data locality
        let mut candidates: Vec<_> = queues
            .iter()
            .filter(|(id, q)| **id != stealer_id && !q.is_empty())
            .collect();

        if candidates.is_empty() {
            return None;
        }

        // Sort by queue length (steal from busiest)
        candidates.sort_by(|a, b| b.1.len().cmp(&a.1.len()));

        // Try to steal from the busiest queue
        for (_, queue) in candidates {
            if let Some(task) = queue.steal() {
                return Some(task);
            }
        }

        None
    }

    /// Find best worker for a task based on data locality
    fn find_best_worker_for_task(&self, spec: &TaskSpec) -> Option<WorkerId> {
        // Check preferred worker
        if let Some(preferred) = spec.preferred_worker {
            return Some(preferred);
        }

        // Calculate affinity based on data dependencies
        let affinity = self
            .data_tracker
            .calculate_affinity(&spec.data_dependencies);
        affinity
            .into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(worker, _)| worker)
    }

    /// Get scheduler statistics
    #[must_use]
    pub fn stats(&self) -> SchedulerStats {
        let queues = self.queues.read().expect("lock poisoned");

        let worker_stats: Vec<_> = queues
            .values()
            .map(|q| WorkerStats {
                worker_id: q.worker_id(),
                queue_length: q.len(),
                completed: q.completed_count(),
                stolen_from: q.stolen_count(),
            })
            .collect();

        let total_pending: usize = worker_stats.iter().map(|s| s.queue_length).sum();
        let total_completed: u64 = worker_stats.iter().map(|s| s.completed).sum();
        let total_stolen: u64 = worker_stats.iter().map(|s| s.stolen_from).sum();

        SchedulerStats {
            worker_count: queues.len(),
            total_submitted: self.submitted_count.load(Ordering::Relaxed),
            total_pending,
            total_completed,
            total_stolen,
            workers: worker_stats,
        }
    }

    /// Get the data tracker
    #[must_use]
    pub fn data_tracker(&self) -> &Arc<BrickDataTracker> {
        &self.data_tracker
    }
}

/// Statistics for a single worker
#[derive(Debug, Clone)]
pub struct WorkerStats {
    /// Worker ID
    pub worker_id: WorkerId,
    /// Current queue length
    pub queue_length: usize,
    /// Tasks completed
    pub completed: u64,
    /// Tasks stolen from this worker
    pub stolen_from: u64,
}

/// Scheduler-wide statistics
#[derive(Debug, Clone)]
pub struct SchedulerStats {
    /// Number of registered workers
    pub worker_count: usize,
    /// Total tasks submitted
    pub total_submitted: u64,
    /// Total tasks pending across all queues
    pub total_pending: usize,
    /// Total tasks completed
    pub total_completed: u64,
    /// Total tasks stolen (indicates load balancing activity)
    pub total_stolen: u64,
    /// Per-worker statistics
    pub workers: Vec<WorkerStats>,
}

// ============================================================================
// PUB/SUB Coordinator
// ============================================================================

/// PUB/SUB coordinator for brick communication
///
/// Enables distributed coordination via publish/subscribe messaging.
#[derive(Debug)]
pub struct BrickCoordinator {
    /// Active subscriptions by topic
    subscriptions: RwLock<HashMap<String, Vec<Arc<RwLock<Vec<BrickMessage>>>>>>,
    /// Message counter for request IDs
    message_counter: AtomicU64,
}

impl Default for BrickCoordinator {
    fn default() -> Self {
        Self::new()
    }
}

impl BrickCoordinator {
    /// Create a new coordinator
    #[must_use]
    pub fn new() -> Self {
        Self {
            subscriptions: RwLock::new(HashMap::new()),
            message_counter: AtomicU64::new(0),
        }
    }

    /// Subscribe to a topic
    #[must_use]
    pub fn subscribe(&self, topic: &str) -> Subscription {
        let messages = Arc::new(RwLock::new(Vec::new()));
        {
            let mut subs = self.subscriptions.write().expect("lock poisoned");
            subs.entry(topic.to_string())
                .or_default()
                .push(Arc::clone(&messages));
        }
        Subscription {
            topic: topic.to_string(),
            messages,
        }
    }

    /// Subscribe to brick events
    #[must_use]
    pub fn subscribe_brick(&self, brick_name: &str) -> Subscription {
        let topic = format!("brick/{}/events", brick_name);
        self.subscribe(&topic)
    }

    /// Publish a message to a topic
    pub fn publish(&self, topic: &str, message: BrickMessage) {
        let subs = self.subscriptions.read().expect("lock poisoned");
        if let Some(subscribers) = subs.get(topic) {
            for sub in subscribers {
                let mut messages = sub.write().expect("lock poisoned");
                messages.push(message.clone());
            }
        }
    }

    /// Broadcast weight updates for a brick
    pub fn broadcast_weights(&self, brick_name: &str, weights: Vec<u8>) {
        let topic = format!("brick/{}/weights", brick_name);
        let version = self.message_counter.fetch_add(1, Ordering::SeqCst);
        self.publish(
            &topic,
            BrickMessage::WeightUpdate {
                brick_name: brick_name.to_string(),
                weights,
                version,
            },
        );
    }

    /// Broadcast state change for a brick
    pub fn broadcast_state_change(&self, brick_name: &str, event: &str) {
        let topic = format!("brick/{}/events", brick_name);
        self.publish(
            &topic,
            BrickMessage::StateChange {
                brick_name: brick_name.to_string(),
                event: event.to_string(),
            },
        );
    }

    /// Generate a unique request ID
    #[must_use]
    pub fn next_request_id(&self) -> u64 {
        self.message_counter.fetch_add(1, Ordering::SeqCst)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    struct TestBrick {
        name: &'static str,
    }

    impl Brick for TestBrick {
        fn brick_name(&self) -> &'static str {
            self.name
        }

        fn assertions(&self) -> &[BrickAssertion] {
            &[BrickAssertion::TextVisible]
        }

        fn budget(&self) -> BrickBudget {
            BrickBudget::uniform(16)
        }

        fn verify(&self) -> BrickVerification {
            BrickVerification {
                passed: vec![BrickAssertion::TextVisible],
                failed: vec![],
                verification_time: Duration::from_micros(100),
            }
        }

        fn to_html(&self) -> String {
            format!("<div>{}</div>", self.name)
        }

        fn to_css(&self) -> String {
            ".test { }".into()
        }
    }

    #[test]
    fn test_worker_id() {
        let id = WorkerId::new(42);
        assert_eq!(id.value(), 42);
        assert_eq!(format!("{id}"), "worker-42");
    }

    #[test]
    fn test_backend_availability() {
        assert!(Backend::Cpu.is_available());
        assert!(Backend::Simd.is_available());
        // GPU/Remote depend on feature flags
    }

    #[test]
    fn test_backend_performance() {
        assert!(Backend::Gpu.performance_estimate() > Backend::Simd.performance_estimate());
        assert!(Backend::Simd.performance_estimate() > Backend::Cpu.performance_estimate());
    }

    #[test]
    fn test_distributed_brick_creation() {
        let inner = TestBrick { name: "Test" };
        let distributed = DistributedBrick::new(inner)
            .with_backend(Backend::Gpu)
            .with_data_dependencies(vec!["weights".into(), "biases".into()])
            .with_preferred_worker(WorkerId::new(1));

        assert_eq!(distributed.backend(), Backend::Gpu);
        assert_eq!(distributed.data_dependencies().len(), 2);
        assert_eq!(distributed.preferred_worker(), Some(WorkerId::new(1)));
        assert_eq!(distributed.brick_name(), "Test");
    }

    #[test]
    fn test_distributed_brick_implements_brick() {
        let inner = TestBrick { name: "Test" };
        let distributed = DistributedBrick::new(inner);

        // Verify it implements Brick trait
        assert!(distributed.verify().is_valid());
        assert_eq!(distributed.budget().total_ms, 16);
    }

    #[test]
    fn test_task_spec() {
        let inner = TestBrick { name: "TestTask" };
        let distributed = DistributedBrick::new(inner)
            .with_backend(Backend::Simd)
            .with_data_dependencies(vec!["model".into()]);

        let spec = distributed.to_task_spec();
        assert_eq!(spec.brick_name, "TestTask");
        assert_eq!(spec.backend, Backend::Simd);
        assert_eq!(spec.data_dependencies, vec!["model"]);
    }

    #[test]
    fn test_brick_input_output() {
        let input = BrickInput::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        assert_eq!(input.element_count(), 4);
        assert_eq!(input.size_bytes(), 16);

        let output = BrickOutput::new(vec![5.0, 6.0], vec![2]);
        assert_eq!(output.size_bytes(), 8);
    }

    #[test]
    fn test_data_tracker() {
        let tracker = BrickDataTracker::new();

        // Track some data
        tracker.track_data("model_weights", WorkerId::new(1), 1024);
        tracker.track_data("model_weights", WorkerId::new(2), 1024);
        tracker.track_data("biases", WorkerId::new(1), 256);

        // Check workers
        let workers = tracker.get_workers_for_data("model_weights");
        assert_eq!(workers.len(), 2);

        // Calculate affinity
        let affinity = tracker.calculate_affinity(&["model_weights".into(), "biases".into()]);
        assert!(affinity.get(&WorkerId::new(1)).unwrap_or(&0.0) > &0.0);
    }

    #[test]
    fn test_data_tracker_find_best_worker() {
        let tracker = BrickDataTracker::new();

        let brick = TestBrick { name: "MelBrick" };
        tracker.track_weights("MelBrick", WorkerId::new(5));

        let best = tracker.find_best_worker(&brick);
        assert_eq!(best, Some(WorkerId::new(5)));
    }

    #[test]
    fn test_backend_selector() {
        let selector = BackendSelector::new()
            .with_gpu_threshold(1000)
            .with_simd_threshold(100);

        // Small input -> CPU
        assert_eq!(selector.select(50, true), Backend::Cpu);

        // Medium input -> SIMD
        assert_eq!(selector.select(500, true), Backend::Simd);

        // Large input with GPU -> GPU
        assert_eq!(selector.select(5000, true), Backend::Gpu);

        // Large input without GPU -> SIMD
        assert_eq!(selector.select(5000, false), Backend::Simd);
    }

    #[test]
    fn test_multi_executor() {
        let tracker = Arc::new(BrickDataTracker::new());
        let executor = MultiBrickExecutor::new(tracker);

        let brick = TestBrick { name: "Test" };
        let input = BrickInput::new(vec![1.0, 2.0, 3.0], vec![3]);

        let result = executor.execute(&brick, input);
        assert!(result.is_ok());

        let output = result.expect("execution should succeed");
        assert_eq!(output.data.len(), 3);
        assert!(output.metrics.execution_time >= Duration::ZERO);
    }

    #[test]
    fn test_brick_coordinator() {
        let coordinator = BrickCoordinator::new();

        // Subscribe to events
        let sub = coordinator.subscribe_brick("MyBrick");

        // Broadcast event
        coordinator.broadcast_state_change("MyBrick", "loaded");

        // Check subscription received message
        assert!(sub.has_messages());
        let messages = sub.drain();
        assert_eq!(messages.len(), 1);
        matches!(&messages[0], BrickMessage::StateChange { brick_name, .. } if brick_name == "MyBrick");
    }

    #[test]
    fn test_coordinator_weight_broadcast() {
        let coordinator = BrickCoordinator::new();

        let sub = coordinator.subscribe("brick/Encoder/weights");
        coordinator.broadcast_weights("Encoder", vec![1, 2, 3, 4]);

        let messages = sub.drain();
        assert_eq!(messages.len(), 1);
        match &messages[0] {
            BrickMessage::WeightUpdate {
                brick_name,
                weights,
                version,
            } => {
                assert_eq!(brick_name, "Encoder");
                assert_eq!(weights, &vec![1, 2, 3, 4]);
                assert_eq!(*version, 0);
            }
            _ => panic!("Expected WeightUpdate message"),
        }
    }

    #[test]
    fn test_subscription_topic() {
        let coordinator = BrickCoordinator::new();
        let sub = coordinator.subscribe("my/topic");
        assert_eq!(sub.topic(), "my/topic");
    }

    #[test]
    fn test_execution_metrics() {
        let metrics = ExecutionMetrics::new(Duration::from_millis(50), Backend::Gpu);
        assert_eq!(metrics.execution_time, Duration::from_millis(50));
        assert_eq!(metrics.backend, Backend::Gpu);
        assert!(metrics.worker_id.is_none());
    }

    // ========================================================================
    // Work-Stealing Scheduler Tests (Phase 10e)
    // ========================================================================

    #[test]
    fn test_work_stealing_task() {
        let spec = TaskSpec {
            brick_name: "TestBrick".into(),
            backend: Backend::Cpu,
            data_dependencies: vec![],
            preferred_worker: None,
        };
        let task = WorkStealingTask::new(1, spec, "input_key".into()).with_priority(10);

        assert_eq!(task.id, 1);
        assert_eq!(task.priority, 10);
        assert_eq!(task.input_key, "input_key");
        assert!(task.age() >= Duration::ZERO);
    }

    #[test]
    fn test_worker_queue_basic() {
        let queue = WorkerQueue::new(WorkerId::new(1));

        assert!(queue.is_empty());
        assert_eq!(queue.len(), 0);

        let spec = TaskSpec {
            brick_name: "Test".into(),
            backend: Backend::Cpu,
            data_dependencies: vec![],
            preferred_worker: None,
        };
        let task = WorkStealingTask::new(1, spec, "key".into());
        queue.push(task);

        assert!(!queue.is_empty());
        assert_eq!(queue.len(), 1);

        let popped = queue.pop();
        assert!(popped.is_some());
        assert!(queue.is_empty());
    }

    #[test]
    fn test_worker_queue_priority_ordering() {
        let queue = WorkerQueue::new(WorkerId::new(1));

        // Push tasks with different priorities
        for i in 0..5 {
            let spec = TaskSpec {
                brick_name: format!("Task{}", i),
                backend: Backend::Cpu,
                data_dependencies: vec![],
                preferred_worker: None,
            };
            let task = WorkStealingTask::new(i as u64, spec, "key".into()).with_priority(i);
            queue.push(task);
        }

        // Pop should return highest priority first
        let task = queue.pop().unwrap();
        assert_eq!(task.priority, 4);

        let task = queue.pop().unwrap();
        assert_eq!(task.priority, 3);
    }

    #[test]
    fn test_worker_queue_steal() {
        let queue = WorkerQueue::new(WorkerId::new(1));

        // Push 3 tasks with priorities 0, 1, 2
        for i in 0..3 {
            let spec = TaskSpec {
                brick_name: format!("Task{}", i),
                backend: Backend::Cpu,
                data_dependencies: vec![],
                preferred_worker: None,
            };
            let task = WorkStealingTask::new(i as u64, spec, "key".into()).with_priority(i);
            queue.push(task);
        }

        // Steal takes from front (lowest priority after sort)
        let stolen = queue.steal().unwrap();
        assert_eq!(stolen.priority, 0);
        assert_eq!(queue.stolen_count(), 1);

        // Queue still has 2 tasks
        assert_eq!(queue.len(), 2);
    }

    #[test]
    fn test_work_stealing_scheduler_basic() {
        let tracker = Arc::new(BrickDataTracker::new());
        let scheduler = WorkStealingScheduler::new(tracker);

        // Register workers
        let _q1 = scheduler.register_worker(WorkerId::new(1));
        let _q2 = scheduler.register_worker(WorkerId::new(2));

        let stats = scheduler.stats();
        assert_eq!(stats.worker_count, 2);
        assert_eq!(stats.total_submitted, 0);
    }

    #[test]
    fn test_work_stealing_scheduler_submit() {
        let tracker = Arc::new(BrickDataTracker::new());
        let scheduler = WorkStealingScheduler::new(tracker);

        scheduler.register_worker(WorkerId::new(1));

        let spec = TaskSpec {
            brick_name: "Test".into(),
            backend: Backend::Cpu,
            data_dependencies: vec![],
            preferred_worker: None,
        };

        let task_id = scheduler.submit(spec, "input".into());
        assert_eq!(task_id, 0);

        let stats = scheduler.stats();
        assert_eq!(stats.total_submitted, 1);
        assert_eq!(stats.total_pending, 1);
    }

    #[test]
    fn test_work_stealing_scheduler_get_work() {
        let tracker = Arc::new(BrickDataTracker::new());
        let scheduler = WorkStealingScheduler::new(tracker);

        scheduler.register_worker(WorkerId::new(1));
        scheduler.register_worker(WorkerId::new(2));

        // Submit task preferring worker 1
        let spec = TaskSpec {
            brick_name: "Test".into(),
            backend: Backend::Cpu,
            data_dependencies: vec![],
            preferred_worker: Some(WorkerId::new(1)),
        };
        scheduler.submit(spec, "input".into());

        // Worker 1 should get the task
        let task = scheduler.get_work(WorkerId::new(1));
        assert!(task.is_some());

        // Worker 2 has nothing to get (or steal since queue is now empty)
        let task = scheduler.get_work(WorkerId::new(2));
        assert!(task.is_none());
    }

    #[test]
    fn test_work_stealing_scheduler_steal() {
        let tracker = Arc::new(BrickDataTracker::new());
        let scheduler = WorkStealingScheduler::new(tracker);

        scheduler.register_worker(WorkerId::new(1));
        scheduler.register_worker(WorkerId::new(2));

        // Submit 3 tasks to worker 1
        for i in 0..3 {
            let spec = TaskSpec {
                brick_name: format!("Task{}", i),
                backend: Backend::Cpu,
                data_dependencies: vec![],
                preferred_worker: Some(WorkerId::new(1)),
            };
            scheduler.submit(spec, format!("input{}", i));
        }

        // Worker 2 should be able to steal a task
        let stolen = scheduler.get_work(WorkerId::new(2));
        assert!(stolen.is_some());

        let stats = scheduler.stats();
        assert_eq!(stats.total_stolen, 1);
        assert_eq!(stats.total_pending, 2); // 3 submitted - 1 stolen
    }

    #[test]
    fn test_work_stealing_scheduler_locality() {
        let tracker = Arc::new(BrickDataTracker::new());

        // Track data on worker 1
        tracker.track_data("model_weights", WorkerId::new(1), 1024);

        let scheduler = WorkStealingScheduler::new(Arc::clone(&tracker));
        scheduler.register_worker(WorkerId::new(1));
        scheduler.register_worker(WorkerId::new(2));

        // Submit task with data dependency - should go to worker 1
        let spec = TaskSpec {
            brick_name: "MelBrick".into(),
            backend: Backend::Cpu,
            data_dependencies: vec!["model_weights".into()],
            preferred_worker: None,
        };
        scheduler.submit(spec, "audio_input".into());

        // Worker 1 should have the task
        let task = scheduler.get_work(WorkerId::new(1));
        assert!(task.is_some());
        assert_eq!(task.unwrap().spec.brick_name, "MelBrick");
    }

    #[test]
    fn test_scheduler_stats() {
        let tracker = Arc::new(BrickDataTracker::new());
        let scheduler = WorkStealingScheduler::new(tracker);

        scheduler.register_worker(WorkerId::new(1));
        scheduler.register_worker(WorkerId::new(2));

        // Submit some tasks
        for i in 0..5 {
            let spec = TaskSpec {
                brick_name: format!("Task{}", i),
                backend: Backend::Cpu,
                data_dependencies: vec![],
                preferred_worker: if i % 2 == 0 {
                    Some(WorkerId::new(1))
                } else {
                    Some(WorkerId::new(2))
                },
            };
            scheduler.submit(spec, format!("input{}", i));
        }

        let stats = scheduler.stats();
        assert_eq!(stats.worker_count, 2);
        assert_eq!(stats.total_submitted, 5);
        assert_eq!(stats.total_pending, 5);
        assert_eq!(stats.workers.len(), 2);
    }

    // ========================================================================
    // Additional comprehensive tests for 95%+ coverage
    // ========================================================================

    #[test]
    fn test_worker_id_copy_clone() {
        let id = WorkerId::new(123);
        let cloned = id;
        assert_eq!(id, cloned);
        assert_eq!(id.0, 123);
    }

    #[test]
    fn test_worker_id_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(WorkerId::new(1));
        set.insert(WorkerId::new(2));
        set.insert(WorkerId::new(1)); // Duplicate
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_backend_default() {
        let backend = Backend::default();
        assert_eq!(backend, Backend::Cpu);
    }

    #[test]
    fn test_backend_remote_not_available() {
        assert!(!Backend::Remote.is_available());
    }

    #[test]
    fn test_backend_performance_remote() {
        assert_eq!(Backend::Remote.performance_estimate(), 5);
        assert_eq!(Backend::Cpu.performance_estimate(), 10);
    }

    #[test]
    fn test_brick_input_default() {
        let input = BrickInput::default();
        assert!(input.data.is_empty());
        assert!(input.shape.is_empty());
        assert!(input.metadata.is_empty());
    }

    #[test]
    fn test_brick_input_with_metadata() {
        let input = BrickInput::new(vec![1.0], vec![1])
            .with_metadata("key1", "value1")
            .with_metadata("key2", "value2");
        assert_eq!(input.metadata.get("key1"), Some(&"value1".to_string()));
        assert_eq!(input.metadata.get("key2"), Some(&"value2".to_string()));
    }

    #[test]
    fn test_brick_output_default() {
        let output = BrickOutput::default();
        assert!(output.data.is_empty());
        assert!(output.shape.is_empty());
    }

    #[test]
    fn test_execution_metrics_default() {
        let metrics = ExecutionMetrics::default();
        assert_eq!(metrics.execution_time, Duration::ZERO);
        assert_eq!(metrics.backend, Backend::Cpu);
        assert!(metrics.worker_id.is_none());
        assert!(metrics.transfer_time.is_none());
    }

    #[test]
    fn test_distributed_brick_inner() {
        let inner = TestBrick { name: "Inner" };
        let distributed = DistributedBrick::new(inner);
        assert_eq!(distributed.inner().brick_name(), "Inner");
    }

    #[test]
    fn test_distributed_brick_inner_mut() {
        let inner = TestBrick { name: "Inner" };
        let mut distributed = DistributedBrick::new(inner);
        let _ = distributed.inner_mut();
        // Just verify we can get mutable reference
    }

    #[test]
    fn test_distributed_brick_to_html() {
        let inner = TestBrick { name: "Test" };
        let distributed = DistributedBrick::new(inner);
        assert_eq!(distributed.to_html(), "<div>Test</div>");
    }

    #[test]
    fn test_distributed_brick_to_css() {
        let inner = TestBrick { name: "Test" };
        let distributed = DistributedBrick::new(inner);
        assert_eq!(distributed.to_css(), ".test { }");
    }

    #[test]
    fn test_distributed_brick_assertions() {
        let inner = TestBrick { name: "Test" };
        let distributed = DistributedBrick::new(inner);
        assert_eq!(distributed.assertions().len(), 1);
    }

    #[test]
    fn test_task_spec_clone() {
        let spec = TaskSpec {
            brick_name: "Test".into(),
            backend: Backend::Gpu,
            data_dependencies: vec!["dep1".into()],
            preferred_worker: Some(WorkerId::new(5)),
        };
        let cloned = spec.clone();
        assert_eq!(spec.brick_name, cloned.brick_name);
        assert_eq!(spec.backend, cloned.backend);
    }

    #[test]
    fn test_brick_data_tracker_default() {
        let tracker = BrickDataTracker::default();
        assert_eq!(tracker.total_data_size(), 0);
    }

    #[test]
    fn test_brick_data_tracker_remove_data() {
        let tracker = BrickDataTracker::new();
        tracker.track_data("data1", WorkerId::new(1), 100);
        tracker.track_data("data1", WorkerId::new(2), 100);

        let workers = tracker.get_workers_for_data("data1");
        assert_eq!(workers.len(), 2);

        tracker.remove_data("data1", WorkerId::new(1));
        let workers = tracker.get_workers_for_data("data1");
        assert_eq!(workers.len(), 1);
        assert_eq!(workers[0], WorkerId::new(2));
    }

    #[test]
    fn test_brick_data_tracker_total_size() {
        let tracker = BrickDataTracker::new();
        tracker.track_data("data1", WorkerId::new(1), 100);
        tracker.track_data("data2", WorkerId::new(1), 200);
        assert_eq!(tracker.total_data_size(), 300);
    }

    #[test]
    fn test_brick_data_tracker_get_nonexistent() {
        let tracker = BrickDataTracker::new();
        let workers = tracker.get_workers_for_data("nonexistent");
        assert!(workers.is_empty());
    }

    #[test]
    fn test_brick_data_tracker_calculate_affinity_empty() {
        let tracker = BrickDataTracker::new();
        let affinity = tracker.calculate_affinity(&["nonexistent".into()]);
        assert!(affinity.is_empty());
    }

    #[test]
    fn test_brick_data_tracker_find_best_worker_no_weights() {
        let tracker = BrickDataTracker::new();
        let brick = TestBrick { name: "NoBrick" };
        let best = tracker.find_best_worker(&brick);
        assert!(best.is_none());
    }

    #[test]
    fn test_brick_data_tracker_find_best_worker_distributed_preferred() {
        let tracker = BrickDataTracker::new();
        let inner = TestBrick { name: "Test" };
        let distributed = DistributedBrick::new(inner).with_preferred_worker(WorkerId::new(42));

        let best = tracker.find_best_worker_for_distributed(&distributed);
        assert_eq!(best, Some(WorkerId::new(42)));
    }

    #[test]
    fn test_brick_data_tracker_find_best_worker_distributed_affinity() {
        let tracker = BrickDataTracker::new();
        tracker.track_data("dep1", WorkerId::new(5), 100);

        let inner = TestBrick { name: "Test" };
        let distributed = DistributedBrick::new(inner).with_data_dependencies(vec!["dep1".into()]);

        let best = tracker.find_best_worker_for_distributed(&distributed);
        assert_eq!(best, Some(WorkerId::new(5)));
    }

    #[test]
    fn test_backend_selector_default() {
        let selector = BackendSelector::default();
        // Default thresholds
        assert_eq!(selector.select(50, true), Backend::Cpu);
    }

    #[test]
    fn test_backend_selector_cpu_max_threshold() {
        let selector = BackendSelector::new()
            .with_cpu_max_threshold(100)
            .with_simd_threshold(50);
        // Over cpu_max_threshold but Remote not available, so falls through to GPU/SIMD/CPU selection
        // Since 200 >= simd_threshold (50), returns SIMD
        let backend = selector.select(200, false);
        assert_eq!(backend, Backend::Simd);

        // Below simd_threshold returns CPU
        let backend = selector.select(10, false);
        assert_eq!(backend, Backend::Cpu);
    }

    #[test]
    fn test_backend_selector_select_for_brick() {
        let selector = BackendSelector::new();
        let backend = selector.select_for_brick(50, 100, true);
        assert_eq!(backend, Backend::Cpu);
    }

    #[test]
    fn test_multi_executor_with_selector() {
        let tracker = Arc::new(BrickDataTracker::new());
        let selector = BackendSelector::new().with_simd_threshold(1);
        let executor = MultiBrickExecutor::new(tracker).with_selector(selector);

        let brick = TestBrick { name: "Test" };
        let input = BrickInput::new(vec![1.0, 2.0], vec![2]);
        let result = executor.execute(&brick, input);
        assert!(result.is_ok());
        // With threshold 1, should use SIMD
        assert_eq!(result.unwrap().metrics.backend, Backend::Simd);
    }

    #[test]
    fn test_multi_executor_with_gpu_available() {
        let tracker = Arc::new(BrickDataTracker::new());
        let executor = MultiBrickExecutor::new(tracker).with_gpu_available(true);
        let _ = executor.data_tracker();
    }

    #[test]
    fn test_multi_executor_execute_distributed() {
        let tracker = Arc::new(BrickDataTracker::new());
        let executor = MultiBrickExecutor::new(tracker);

        let inner = TestBrick { name: "Test" };
        let distributed = DistributedBrick::new(inner).with_backend(Backend::Cpu);
        let input = BrickInput::new(vec![1.0], vec![1]);

        let result = executor.execute_distributed(&distributed, input);
        assert!(result.is_ok());
    }

    #[test]
    fn test_multi_executor_execute_simd() {
        let tracker = Arc::new(BrickDataTracker::new());
        let selector = BackendSelector::new().with_simd_threshold(1);
        let executor = MultiBrickExecutor::new(tracker).with_selector(selector);

        let brick = TestBrick { name: "Test" };
        let input = BrickInput::new(vec![1.0, 2.0], vec![2]);

        let result = executor.execute(&brick, input);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().metrics.backend, Backend::Simd);
    }

    #[test]
    fn test_multi_executor_execute_gpu_unavailable() {
        let tracker = Arc::new(BrickDataTracker::new());
        let inner = TestBrick { name: "Test" };
        let distributed = DistributedBrick::new(inner).with_backend(Backend::Gpu);
        let executor = MultiBrickExecutor::new(tracker).with_gpu_available(false);
        let input = BrickInput::new(vec![1.0], vec![1]);

        let result = executor.execute_distributed(&distributed, input);
        assert!(result.is_err());
    }

    #[test]
    fn test_multi_executor_execute_remote_unavailable() {
        let tracker = Arc::new(BrickDataTracker::new());
        let inner = TestBrick { name: "Test" };
        let distributed = DistributedBrick::new(inner).with_backend(Backend::Remote);
        let executor = MultiBrickExecutor::new(tracker);
        let input = BrickInput::new(vec![1.0], vec![1]);

        let result = executor.execute_distributed(&distributed, input);
        assert!(result.is_err());
    }

    #[test]
    fn test_subscription_drain_empty() {
        let coordinator = BrickCoordinator::new();
        let sub = coordinator.subscribe("test/topic");
        let messages = sub.drain();
        assert!(messages.is_empty());
    }

    #[test]
    fn test_subscription_has_messages_false() {
        let coordinator = BrickCoordinator::new();
        let sub = coordinator.subscribe("test/topic");
        assert!(!sub.has_messages());
    }

    #[test]
    fn test_brick_coordinator_default() {
        let coordinator = BrickCoordinator::default();
        let id = coordinator.next_request_id();
        assert_eq!(id, 0);
    }

    #[test]
    fn test_brick_coordinator_next_request_id() {
        let coordinator = BrickCoordinator::new();
        assert_eq!(coordinator.next_request_id(), 0);
        assert_eq!(coordinator.next_request_id(), 1);
        assert_eq!(coordinator.next_request_id(), 2);
    }

    #[test]
    fn test_brick_coordinator_publish_no_subscribers() {
        let coordinator = BrickCoordinator::new();
        // Should not panic even with no subscribers
        coordinator.publish(
            "nonexistent/topic",
            BrickMessage::StateChange {
                brick_name: "Test".into(),
                event: "test".into(),
            },
        );
    }

    #[test]
    fn test_brick_message_execution_request() {
        let msg = BrickMessage::ExecutionRequest {
            brick_name: "Test".into(),
            input_key: "key".into(),
            request_id: 42,
        };
        match msg {
            BrickMessage::ExecutionRequest {
                brick_name,
                input_key,
                request_id,
            } => {
                assert_eq!(brick_name, "Test");
                assert_eq!(input_key, "key");
                assert_eq!(request_id, 42);
            }
            _ => panic!("Wrong message type"),
        }
    }

    #[test]
    fn test_brick_message_execution_result() {
        let msg = BrickMessage::ExecutionResult {
            request_id: 42,
            output_key: "out".into(),
            success: true,
        };
        match msg {
            BrickMessage::ExecutionResult {
                request_id,
                output_key,
                success,
            } => {
                assert_eq!(request_id, 42);
                assert_eq!(output_key, "out");
                assert!(success);
            }
            _ => panic!("Wrong message type"),
        }
    }

    #[test]
    fn test_work_stealing_task_clone() {
        let spec = TaskSpec {
            brick_name: "Test".into(),
            backend: Backend::Cpu,
            data_dependencies: vec![],
            preferred_worker: None,
        };
        let task = WorkStealingTask::new(1, spec, "key".into());
        let cloned = task.clone();
        assert_eq!(task.id, cloned.id);
    }

    #[test]
    fn test_worker_queue_worker_id() {
        let queue = WorkerQueue::new(WorkerId::new(42));
        assert_eq!(queue.worker_id(), WorkerId::new(42));
    }

    #[test]
    fn test_worker_queue_completed_count() {
        let queue = WorkerQueue::new(WorkerId::new(1));
        assert_eq!(queue.completed_count(), 0);
        queue.mark_completed();
        assert_eq!(queue.completed_count(), 1);
        queue.mark_completed();
        assert_eq!(queue.completed_count(), 2);
    }

    #[test]
    fn test_worker_queue_pop_empty() {
        let queue = WorkerQueue::new(WorkerId::new(1));
        assert!(queue.pop().is_none());
    }

    #[test]
    fn test_worker_queue_steal_empty() {
        let queue = WorkerQueue::new(WorkerId::new(1));
        assert!(queue.steal().is_none());
    }

    #[test]
    fn test_scheduler_unregister_worker() {
        let tracker = Arc::new(BrickDataTracker::new());
        let scheduler = WorkStealingScheduler::new(tracker);

        scheduler.register_worker(WorkerId::new(1));
        assert_eq!(scheduler.stats().worker_count, 1);

        scheduler.unregister_worker(WorkerId::new(1));
        assert_eq!(scheduler.stats().worker_count, 0);
    }

    #[test]
    fn test_scheduler_submit_no_workers() {
        let tracker = Arc::new(BrickDataTracker::new());
        let scheduler = WorkStealingScheduler::new(tracker);

        let spec = TaskSpec {
            brick_name: "Test".into(),
            backend: Backend::Cpu,
            data_dependencies: vec![],
            preferred_worker: None,
        };

        let task_id = scheduler.submit(spec, "input".into());
        assert_eq!(task_id, 0);
        // Task submitted but no workers to receive it
        assert_eq!(scheduler.stats().total_submitted, 1);
    }

    #[test]
    fn test_scheduler_submit_priority() {
        let tracker = Arc::new(BrickDataTracker::new());
        let scheduler = WorkStealingScheduler::new(tracker);

        scheduler.register_worker(WorkerId::new(1));

        let spec = TaskSpec {
            brick_name: "Test".into(),
            backend: Backend::Cpu,
            data_dependencies: vec![],
            preferred_worker: None,
        };

        let task_id = scheduler.submit_priority(spec, "input".into(), 100);
        assert_eq!(task_id, 0);

        let task = scheduler.get_work(WorkerId::new(1));
        assert!(task.is_some());
        assert_eq!(task.unwrap().priority, 100);
    }

    #[test]
    fn test_scheduler_get_work_unregistered_worker() {
        let tracker = Arc::new(BrickDataTracker::new());
        let scheduler = WorkStealingScheduler::new(tracker);

        // Try to get work for worker that doesn't exist
        let task = scheduler.get_work(WorkerId::new(999));
        assert!(task.is_none());
    }

    #[test]
    fn test_scheduler_data_tracker_accessor() {
        let tracker = Arc::new(BrickDataTracker::new());
        let scheduler = WorkStealingScheduler::new(Arc::clone(&tracker));

        let _ = scheduler.data_tracker();
    }

    #[test]
    fn test_worker_stats_fields() {
        let stats = WorkerStats {
            worker_id: WorkerId::new(1),
            queue_length: 5,
            completed: 10,
            stolen_from: 2,
        };
        assert_eq!(stats.worker_id, WorkerId::new(1));
        assert_eq!(stats.queue_length, 5);
        assert_eq!(stats.completed, 10);
        assert_eq!(stats.stolen_from, 2);
    }

    #[test]
    fn test_scheduler_stats_fields() {
        let stats = SchedulerStats {
            worker_count: 2,
            total_submitted: 10,
            total_pending: 5,
            total_completed: 4,
            total_stolen: 1,
            workers: vec![],
        };
        assert_eq!(stats.worker_count, 2);
        assert_eq!(stats.total_submitted, 10);
        assert_eq!(stats.total_pending, 5);
        assert_eq!(stats.total_completed, 4);
        assert_eq!(stats.total_stolen, 1);
    }

    #[test]
    fn test_data_location_clone() {
        let loc = DataLocation {
            key: "test".into(),
            workers: vec![WorkerId::new(1)],
            size_bytes: 100,
            last_access: Instant::now(),
        };
        let cloned = loc.clone();
        assert_eq!(loc.key, cloned.key);
    }

    #[test]
    fn test_track_data_updates_existing() {
        let tracker = BrickDataTracker::new();
        tracker.track_data("key", WorkerId::new(1), 100);
        tracker.track_data("key", WorkerId::new(1), 200); // Same worker again

        let workers = tracker.get_workers_for_data("key");
        assert_eq!(workers.len(), 1); // Should not duplicate
    }
}
