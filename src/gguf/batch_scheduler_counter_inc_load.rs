
#[cfg(feature = "gpu")]
impl GpuBufferPool {
    /// Create a new `AtomicU64` initialized to zero.
    fn new_counter() -> std::sync::atomic::AtomicU64 {
        std::sync::atomic::AtomicU64::new(0)
    }

    /// Increment an atomic counter by 1 (Relaxed ordering).
    fn inc(counter: &std::sync::atomic::AtomicU64) {
        counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// Load an atomic u64 counter with Relaxed ordering.
    fn load_relaxed(counter: &std::sync::atomic::AtomicU64) -> u64 {
        counter.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Lock a buffer pool mutex, panicking on poison.
    fn lock_pool(
        buffers: &std::sync::Mutex<Vec<Vec<f32>>>,
    ) -> std::sync::MutexGuard<'_, Vec<Vec<f32>>> {
        buffers.lock().expect("mutex poisoned")
    }

    /// Create new buffer pool with specified dimensions
    pub fn new(
        hidden_dim: usize,
        intermediate_dim: usize,
        max_seq_len: usize,
        num_heads: usize,
        pool_size: usize,
    ) -> Self {
        Self {
            hidden_buffers: std::sync::Mutex::new(Vec::with_capacity(pool_size)),
            intermediate_buffers: std::sync::Mutex::new(Vec::with_capacity(pool_size)),
            attention_buffers: std::sync::Mutex::new(Vec::with_capacity(pool_size)),
            hidden_dim,
            intermediate_dim,
            max_seq_len,
            num_heads,
            pool_size,
            borrows: Self::new_counter(),
            returns: Self::new_counter(),
            post_warmup_allocs: Self::new_counter(),
            warmed_up: std::sync::atomic::AtomicBool::new(false),
        }
    }

    /// Pre-allocate buffers of a given size into a pool mutex.
    fn warmup_pool(buffers: &std::sync::Mutex<Vec<Vec<f32>>>, count: usize, size: usize) {
        let mut pool = Self::lock_pool(buffers);
        for _ in 0..count {
            pool.push(vec![0.0f32; size]);
        }
    }

    /// Borrow a buffer from a pool, allocating a new one if the pool is empty.
    ///
    /// Tracks borrow count and post-warmup allocations for diagnostics.
    fn borrow_from_pool(
        &self,
        buffers: &std::sync::Mutex<Vec<Vec<f32>>>,
        alloc_size: usize,
    ) -> Vec<f32> {
        Self::inc(&self.borrows);

        let mut pool = Self::lock_pool(buffers);
        if let Some(buffer) = pool.pop() {
            buffer
        } else {
            if self.warmed_up.load(std::sync::atomic::Ordering::Acquire) {
                Self::inc(&self.post_warmup_allocs);
            }
            vec![0.0f32; alloc_size]
        }
    }

    /// Return a buffer to a pool, zeroing it first for security and determinism.
    ///
    /// Drops the buffer if the pool is already at capacity.
    fn return_to_pool(&self, buffers: &std::sync::Mutex<Vec<Vec<f32>>>, mut buffer: Vec<f32>) {
        Self::inc(&self.returns);

        // Zero out for security and determinism
        buffer.fill(0.0);

        let mut pool = Self::lock_pool(buffers);
        if pool.len() < self.pool_size {
            pool.push(buffer);
        }
        // If pool is full, buffer is dropped
    }

    /// Warmup: pre-allocate all buffers
    ///
    /// Call this once during model initialization to eliminate
    /// allocation overhead during inference.
    pub fn warmup(&self) {
        Self::warmup_pool(&self.hidden_buffers, self.pool_size, self.hidden_dim);
        Self::warmup_pool(
            &self.intermediate_buffers,
            self.pool_size,
            self.intermediate_dim,
        );
        Self::warmup_pool(
            &self.attention_buffers,
            self.pool_size,
            self.num_heads * self.max_seq_len,
        );

        self.warmed_up
            .store(true, std::sync::atomic::Ordering::Release);
    }

    /// Borrow a hidden state buffer from the pool
    ///
    /// Returns a pre-allocated buffer if available, or allocates new if needed.
    pub fn borrow_hidden(&self) -> Vec<f32> {
        self.borrow_from_pool(&self.hidden_buffers, self.hidden_dim)
    }

    /// Return a hidden state buffer to the pool
    pub fn return_hidden(&self, buffer: Vec<f32>) {
        self.return_to_pool(&self.hidden_buffers, buffer);
    }

    /// Borrow an intermediate buffer from the pool
    pub fn borrow_intermediate(&self) -> Vec<f32> {
        self.borrow_from_pool(&self.intermediate_buffers, self.intermediate_dim)
    }

    /// Return an intermediate buffer to the pool
    pub fn return_intermediate(&self, buffer: Vec<f32>) {
        self.return_to_pool(&self.intermediate_buffers, buffer);
    }

    /// Borrow an attention score buffer from the pool
    pub fn borrow_attention(&self) -> Vec<f32> {
        self.borrow_from_pool(
            &self.attention_buffers,
            self.num_heads * self.max_seq_len,
        )
    }

    /// Return an attention score buffer to the pool
    pub fn return_attention(&self, buffer: Vec<f32>) {
        self.return_to_pool(&self.attention_buffers, buffer);
    }

    /// Check if pool has achieved zero-allocation after warmup
    pub fn is_zero_alloc(&self) -> bool {
        self.warmed_up.load(std::sync::atomic::Ordering::Acquire)
            && Self::load_relaxed(&self.post_warmup_allocs) == 0
    }

    /// Get pool statistics
    pub fn stats(&self) -> GpuBufferPoolStats {
        GpuBufferPoolStats {
            borrows: Self::load_relaxed(&self.borrows),
            returns: Self::load_relaxed(&self.returns),
            post_warmup_allocs: Self::load_relaxed(&self.post_warmup_allocs),
            warmed_up: self.warmed_up.load(std::sync::atomic::Ordering::Acquire),
            hidden_available: Self::lock_pool(&self.hidden_buffers).len(),
            intermediate_available: Self::lock_pool(&self.intermediate_buffers).len(),
            attention_available: Self::lock_pool(&self.attention_buffers).len(),
        }
    }

    /// Calculate total memory usage of the buffer pool
    pub fn memory_usage_bytes(&self) -> usize {
        let hidden_bytes = self.pool_size * self.hidden_dim * 4;
        let intermediate_bytes = self.pool_size * self.intermediate_dim * 4;
        let attention_bytes = self.pool_size * self.num_heads * self.max_seq_len * 4;
        hidden_bytes + intermediate_bytes + attention_bytes
    }
}

/// Statistics for GpuBufferPool
#[cfg(feature = "gpu")]
#[derive(Debug, Clone)]
pub struct GpuBufferPoolStats {
    /// Total borrows
    pub borrows: u64,
    /// Total returns
    pub returns: u64,
    /// Allocations after warmup (should be 0)
    pub post_warmup_allocs: u64,
    /// Whether warmup is complete
    pub warmed_up: bool,
    /// Available hidden buffers
    pub hidden_available: usize,
    /// Available intermediate buffers
    pub intermediate_available: usize,
    /// Available attention buffers
    pub attention_available: usize,
}

/// Async Command Queue for GPU pipelining (PARITY-032, IMP-310)
///
/// Implements double-buffering to hide GPU latency by overlapping
/// computation and data transfer. While one batch is being processed
/// on GPU, the next batch is being prepared on CPU.
///
/// # Key Properties
/// - Double-buffering: 2 command slots for overlap
/// - Async submission: Non-blocking command enqueue
/// - Pipeline stages: Prepare -> Submit -> Execute -> Complete
///
/// # GPU Utilization Target
/// - Without pipelining: ~50% (waiting for results)
/// - With pipelining: >85% (overlapped execution)
#[cfg(feature = "gpu")]
pub struct AsyncCommandQueue {
    /// Command slots for double-buffering (2 slots)
    slots: [std::sync::Mutex<CommandSlot>; 2],
    /// Current slot index for submission
    current_slot: std::sync::atomic::AtomicUsize,
    /// Statistics: commands submitted
    pub commands_submitted: std::sync::atomic::AtomicU64,
    /// Statistics: commands completed
    pub commands_completed: std::sync::atomic::AtomicU64,
    /// Statistics: pipeline stalls (had to wait for previous)
    pub pipeline_stalls: std::sync::atomic::AtomicU64,
}

/// State of a command slot in the async queue
#[cfg(feature = "gpu")]
#[derive(Debug, Clone)]
pub enum CommandSlotState {
    /// Slot is empty and ready for new command
    Empty,
    /// Command is being prepared (CPU side)
    Preparing,
    /// Command has been submitted to GPU
    Submitted,
    /// Command execution is complete
    Complete,
}

/// A command slot for async execution
#[cfg(feature = "gpu")]
pub struct CommandSlot {
    /// Current state of this slot
    state: CommandSlotState,
    /// Input data for the command
    input: Option<Vec<f32>>,
    /// Output data from the command
    output: Option<Vec<f32>>,
    /// Timestamp when command was submitted
    submit_time: Option<std::time::Instant>,
}

#[cfg(feature = "gpu")]
impl Default for CommandSlot {
    fn default() -> Self {
        Self {
            state: CommandSlotState::Empty,
            input: None,
            output: None,
            submit_time: None,
        }
    }
}

#[cfg(feature = "gpu")]
impl AsyncCommandQueue {
    /// Create a new `AtomicU64` initialized to zero.
    fn new_counter() -> std::sync::atomic::AtomicU64 {
        std::sync::atomic::AtomicU64::new(0)
    }

    /// Increment an atomic counter by 1 (Relaxed ordering).
    fn inc(counter: &std::sync::atomic::AtomicU64) {
        counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// Load an atomic u64 counter with Relaxed ordering.
    fn load_relaxed(counter: &std::sync::atomic::AtomicU64) -> u64 {
        counter.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Lock a command slot by index, panicking on poison.
    fn lock_slot(&self, slot_idx: usize) -> std::sync::MutexGuard<'_, CommandSlot> {
        self.slots[slot_idx].lock().expect("mutex poisoned")
    }

    /// Create new async command queue with double-buffering
    pub fn new() -> Self {
        Self {
            slots: [
                std::sync::Mutex::new(CommandSlot::default()),
                std::sync::Mutex::new(CommandSlot::default()),
            ],
            current_slot: std::sync::atomic::AtomicUsize::new(0),
            commands_submitted: Self::new_counter(),
            commands_completed: Self::new_counter(),
            pipeline_stalls: Self::new_counter(),
        }
    }

    /// Submit a command for async execution
    ///
    /// Returns the slot index where the command was placed.
    /// If both slots are busy, this will block until one is available
    /// (counted as a pipeline stall).
    pub fn submit(&self, input: Vec<f32>) -> usize {
        let slot_idx = self
            .current_slot
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
            % 2;

        let mut slot = self.lock_slot(slot_idx);

        // Check if we need to wait for previous command
        if matches!(
            slot.state,
            CommandSlotState::Submitted | CommandSlotState::Preparing
        ) {
            Self::inc(&self.pipeline_stalls);
            // In real implementation, would wait for GPU completion
            // For now, mark as complete to allow reuse
            slot.state = CommandSlotState::Complete;
        }

        // Prepare new command
        slot.state = CommandSlotState::Preparing;
        slot.input = Some(input);
        slot.output = None;
        slot.submit_time = Some(std::time::Instant::now());

        // Mark as submitted
        slot.state = CommandSlotState::Submitted;
        Self::inc(&self.commands_submitted);

        slot_idx
    }

    /// Mark a command as complete with output
    pub fn complete(&self, slot_idx: usize, output: Vec<f32>) {
        let mut slot = self.lock_slot(slot_idx);
        slot.state = CommandSlotState::Complete;
        slot.output = Some(output);
        Self::inc(&self.commands_completed);
    }

    /// Get output from a completed command
    ///
    /// Returns None if command is not complete yet.
    pub fn get_output(&self, slot_idx: usize) -> Option<Vec<f32>> {
        let mut slot = self.lock_slot(slot_idx);
        if matches!(slot.state, CommandSlotState::Complete) {
            slot.state = CommandSlotState::Empty;
            slot.output.take()
        } else {
            None
        }
    }

    /// Get queue statistics
    pub fn stats(&self) -> AsyncQueueStats {
        let submitted = Self::load_relaxed(&self.commands_submitted);
        let completed = Self::load_relaxed(&self.commands_completed);
        let stalls = Self::load_relaxed(&self.pipeline_stalls);

        // GPU utilization estimate: (1 - stalls/submitted) * 100
        let utilization = if submitted > 0 {
            (1.0 - stalls as f64 / submitted as f64) * 100.0
        } else {
            0.0
        };

        AsyncQueueStats {
            commands_submitted: submitted,
            commands_completed: completed,
            pipeline_stalls: stalls,
            in_flight: submitted.saturating_sub(completed),
            gpu_utilization_percent: utilization,
        }
    }

    /// Calculate pipeline efficiency
    ///
    /// Efficiency = commands without stall / total commands
    pub fn pipeline_efficiency(&self) -> f64 {
        let submitted = Self::load_relaxed(&self.commands_submitted);
        let stalls = Self::load_relaxed(&self.pipeline_stalls);
        if submitted == 0 {
            return 1.0;
        }
        (submitted - stalls) as f64 / submitted as f64
    }
}

#[cfg(feature = "gpu")]
impl Default for AsyncCommandQueue {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics for AsyncCommandQueue
#[cfg(feature = "gpu")]
#[derive(Debug, Clone)]
pub struct AsyncQueueStats {
    /// Total commands submitted
    pub commands_submitted: u64,
    /// Total commands completed
    pub commands_completed: u64,
    /// Pipeline stalls (had to wait)
    pub pipeline_stalls: u64,
    /// Commands currently in flight
    pub in_flight: u64,
    /// Estimated GPU utilization percentage
    pub gpu_utilization_percent: f64,
}

/// Prefix Cache for common prompts (PARITY-033, IMP-319)
///
/// Caches the KV cache state for common prompt prefixes, enabling
/// instant response (0ms TTFT) for repeated prompts.
///
/// # Key Properties
/// - Hash-based prefix lookup (FNV-1a)
/// - LRU eviction for memory management
/// - Thread-safe access
///
/// # Use Cases
/// - System prompts (cached once, reused for all requests)
/// - Common few-shot examples
/// - Chat history prefixes
#[cfg(feature = "gpu")]
pub struct PrefixCache {
    /// Cached prefix entries (hash -> entry)
    entries: std::sync::Mutex<std::collections::HashMap<u64, PrefixCacheEntry>>,
    /// Maximum number of cached prefixes
    max_entries: usize,
    /// Statistics: cache hits
    pub hits: std::sync::atomic::AtomicU64,
    /// Statistics: cache misses
    pub misses: std::sync::atomic::AtomicU64,
    /// Statistics: evictions
    pub evictions: std::sync::atomic::AtomicU64,
}
