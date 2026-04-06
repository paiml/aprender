
/// A cached prefix entry
#[cfg(feature = "gpu")]
pub struct PrefixCacheEntry {
    /// The original prompt tokens
    pub tokens: Vec<u32>,
    /// Cached K state for each layer [num_layers, seq_len, hidden_dim]
    pub k_cache: Vec<Vec<f32>>,
    /// Cached V state for each layer [num_layers, seq_len, hidden_dim]
    pub v_cache: Vec<Vec<f32>>,
    /// Timestamp for LRU eviction
    pub last_access: std::time::Instant,
    /// Number of times this prefix was hit
    pub hit_count: u64,
}

#[cfg(feature = "gpu")]
impl PrefixCache {
    /// Create new prefix cache with specified capacity
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: std::sync::Mutex::new(std::collections::HashMap::with_capacity(max_entries)),
            max_entries,
            hits: std::sync::atomic::AtomicU64::new(0),
            misses: std::sync::atomic::AtomicU64::new(0),
            evictions: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Lock the entries mutex, panicking on poison.
    fn lock_entries(
        &self,
    ) -> std::sync::MutexGuard<'_, std::collections::HashMap<u64, PrefixCacheEntry>> {
        self.entries.lock().expect("mutex poisoned")
    }

    /// Increment an atomic counter by 1 (Relaxed ordering).
    fn inc(counter: &std::sync::atomic::AtomicU64) {
        counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// Load an atomic counter (Relaxed ordering).
    fn load(counter: &std::sync::atomic::AtomicU64) -> u64 {
        counter.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Hash tokens to create cache key (FNV-1a)
    fn hash_tokens(tokens: &[u32]) -> u64 {
        const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
        const FNV_PRIME: u64 = 0x0100_0000_01b3;

        let mut hash = FNV_OFFSET;
        for &token in tokens {
            hash ^= token as u64;
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        hash
    }

    /// Look up a prefix in the cache
    ///
    /// Returns the cached KV state if found, None otherwise.
    #[allow(clippy::type_complexity)]
    pub fn lookup(&self, tokens: &[u32]) -> Option<(Vec<Vec<f32>>, Vec<Vec<f32>>)> {
        let hash = Self::hash_tokens(tokens);

        let mut entries = self.lock_entries();
        if let Some(entry) = entries.get_mut(&hash) {
            // Verify tokens match (hash collision check)
            if entry.tokens == tokens {
                Self::inc(&self.hits);
                entry.last_access = std::time::Instant::now();
                entry.hit_count += 1;
                return Some((entry.k_cache.clone(), entry.v_cache.clone()));
            }
        }

        Self::inc(&self.misses);
        None
    }

    /// Insert a new prefix into the cache
    ///
    /// Evicts LRU entry if cache is full.
    pub fn insert(&self, tokens: Vec<u32>, k_cache: Vec<Vec<f32>>, v_cache: Vec<Vec<f32>>) {
        let hash = Self::hash_tokens(&tokens);

        let mut entries = self.lock_entries();

        // Evict LRU if at capacity
        if entries.len() >= self.max_entries {
            // Find oldest entry
            if let Some((&oldest_hash, _)) = entries.iter().min_by_key(|(_, e)| e.last_access) {
                entries.remove(&oldest_hash);
                Self::inc(&self.evictions);
            }
        }

        entries.insert(
            hash,
            PrefixCacheEntry {
                tokens,
                k_cache,
                v_cache,
                last_access: std::time::Instant::now(),
                hit_count: 0,
            },
        );
    }

    /// Check if a prefix is cached
    pub fn contains(&self, tokens: &[u32]) -> bool {
        let hash = Self::hash_tokens(tokens);
        self.lock_entries().contains_key(&hash)
    }

    /// Get cache statistics
    pub fn stats(&self) -> PrefixCacheStats {
        let hits = Self::load(&self.hits);
        let misses = Self::load(&self.misses);
        let total = hits + misses;

        PrefixCacheStats {
            hits,
            misses,
            evictions: Self::load(&self.evictions),
            entries: self.lock_entries().len(),
            hit_rate: if total > 0 {
                hits as f64 / total as f64
            } else {
                0.0
            },
        }
    }

    /// Clear all cached entries
    pub fn clear(&self) {
        self.lock_entries().clear();
    }

    /// Estimate memory usage of cached prefixes
    pub fn memory_usage_bytes(&self) -> usize {
        self.lock_entries()
            .values()
            .map(|e| {
                let k_bytes: usize = e.k_cache.iter().map(|v| v.len() * 4).sum();
                let v_bytes: usize = e.v_cache.iter().map(|v| v.len() * 4).sum();
                let token_bytes = e.tokens.len() * 4;
                k_bytes + v_bytes + token_bytes
            })
            .sum()
    }
}

#[cfg(feature = "gpu")]
impl Default for PrefixCache {
    fn default() -> Self {
        Self::new(16) // Default: cache 16 prefixes
    }
}

/// Statistics for PrefixCache
#[cfg(feature = "gpu")]
#[derive(Debug, Clone)]
pub struct PrefixCacheStats {
    /// Cache hits
    pub hits: u64,
    /// Cache misses
    pub misses: u64,
    /// Evictions due to capacity
    pub evictions: u64,
    /// Current number of cached entries
    pub entries: usize,
    /// Hit rate (0.0 - 1.0)
    pub hit_rate: f64,
}

// =============================================================================
// PARITY-034: Multi-Request Scheduler with Scheduling Policies (IMP-317)
// =============================================================================
//
// Extends PARITY-028's ContinuousBatchScheduler with:
// - Multiple scheduling policies (FCFS, SJF, Round-Robin)
// - Request queuing with priorities
// - TTFT (Time to First Token) tracking
// - Throughput scaling verification
//
// Architecture:
// - Incoming requests are queued with their KV cache states
// - Scheduler batches decode steps from multiple requests
// - GPU GEMM efficiency: batch_size > 1 enables GPU acceleration
// - Preemption: Long-running requests can be paused for new arrivals
// =============================================================================

/// Request state in the multi-request scheduler
#[cfg(feature = "gpu")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MultiRequestState {
    /// Waiting for prefill
    Pending,
    /// Prefill in progress
    Prefilling,
    /// Decode in progress
    Decoding,
    /// Request completed
    Completed,
    /// Request preempted (paused)
    Preempted,
}

/// A single inference request in the multi-request scheduler
#[cfg(feature = "gpu")]
#[derive(Clone)]
pub struct MultiSchedulerRequest {
    /// Unique request ID
    pub id: u64,
    /// Input tokens
    pub tokens: Vec<u32>,
    /// Generated tokens so far
    pub generated: Vec<u32>,
    /// Maximum tokens to generate
    pub max_tokens: usize,
    /// Current state
    pub state: MultiRequestState,
    /// KV cache position (how many tokens processed)
    pub kv_position: usize,
    /// Arrival time for FCFS scheduling
    pub arrival_time: std::time::Instant,
    /// Time first token generated (for TTFT metric)
    pub first_token_time: Option<std::time::Instant>,
}

#[cfg(feature = "gpu")]
impl MultiSchedulerRequest {
    /// Create new request
    pub fn new(id: u64, tokens: Vec<u32>, max_tokens: usize) -> Self {
        Self {
            id,
            tokens,
            generated: Vec::with_capacity(max_tokens),
            max_tokens,
            state: MultiRequestState::Pending,
            kv_position: 0,
            arrival_time: std::time::Instant::now(),
            first_token_time: None,
        }
    }

    /// Check if request is complete
    pub fn is_complete(&self) -> bool {
        self.state == MultiRequestState::Completed || self.generated.len() >= self.max_tokens
    }

    /// Time to first token (None if not yet generated)
    pub fn ttft_ms(&self) -> Option<f64> {
        self.first_token_time
            .map(|t| t.duration_since(self.arrival_time).as_secs_f64() * 1000.0)
    }
}

/// Scheduling policy for the batch scheduler
#[cfg(feature = "gpu")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedulingPolicy {
    /// First-come, first-served
    Fcfs,
    /// Shortest job first (by remaining tokens)
    Sjf,
    /// Round-robin with time slices
    RoundRobin,
}

/// Multi-request scheduler with scheduling policies (PARITY-034)
#[cfg(feature = "gpu")]
pub struct MultiRequestScheduler {
    /// Pending requests queue
    pending: std::sync::Mutex<std::collections::VecDeque<MultiSchedulerRequest>>,
    /// Active requests being processed
    active: std::sync::Mutex<Vec<MultiSchedulerRequest>>,
    /// Completed requests
    completed: std::sync::Mutex<Vec<MultiSchedulerRequest>>,
    /// Maximum batch size
    max_batch_size: usize,
    /// Maximum concurrent requests
    max_concurrent: usize,
    /// Scheduling policy
    policy: SchedulingPolicy,
    /// Request ID counter
    next_id: std::sync::atomic::AtomicU64,
    /// Requests submitted
    pub requests_submitted: std::sync::atomic::AtomicU64,
    /// Requests completed
    pub requests_completed: std::sync::atomic::AtomicU64,
    /// Total tokens generated
    pub tokens_generated: std::sync::atomic::AtomicU64,
    /// Batch iterations performed
    pub batch_iterations: std::sync::atomic::AtomicU64,
}

#[cfg(feature = "gpu")]
impl MultiRequestScheduler {
    /// Create a new `AtomicU64` initialized to zero.
    fn new_counter() -> std::sync::atomic::AtomicU64 {
        std::sync::atomic::AtomicU64::new(0)
    }

    /// Increment an atomic counter by 1 (Relaxed ordering).
    fn inc(counter: &std::sync::atomic::AtomicU64) {
        counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// Increment an atomic counter by `n` (Relaxed ordering).
    fn inc_by(counter: &std::sync::atomic::AtomicU64, n: u64) {
        counter.fetch_add(n, std::sync::atomic::Ordering::Relaxed);
    }

    /// Load an atomic counter (Relaxed ordering).
    fn load(counter: &std::sync::atomic::AtomicU64) -> u64 {
        counter.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Lock the pending queue mutex.
    fn lock_pending(
        &self,
    ) -> std::sync::MutexGuard<'_, std::collections::VecDeque<MultiSchedulerRequest>> {
        self.pending.lock().expect("mutex poisoned")
    }

    /// Lock the active requests mutex.
    fn lock_active(&self) -> std::sync::MutexGuard<'_, Vec<MultiSchedulerRequest>> {
        self.active.lock().expect("mutex poisoned")
    }

    /// Lock the completed requests mutex.
    fn lock_completed(&self) -> std::sync::MutexGuard<'_, Vec<MultiSchedulerRequest>> {
        self.completed.lock().expect("mutex poisoned")
    }

    /// Create new scheduler with given parameters
    pub fn new(max_batch_size: usize, max_concurrent: usize, policy: SchedulingPolicy) -> Self {
        Self {
            pending: std::sync::Mutex::new(std::collections::VecDeque::new()),
            active: std::sync::Mutex::new(Vec::with_capacity(max_concurrent)),
            completed: std::sync::Mutex::new(Vec::new()),
            max_batch_size,
            max_concurrent,
            policy,
            next_id: Self::new_counter(),
            requests_submitted: Self::new_counter(),
            requests_completed: Self::new_counter(),
            tokens_generated: Self::new_counter(),
            batch_iterations: Self::new_counter(),
        }
    }

    /// Submit a new request
    pub fn submit(&self, tokens: Vec<u32>, max_tokens: usize) -> u64 {
        let id = self
            .next_id
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let request = MultiSchedulerRequest::new(id, tokens, max_tokens);

        self.lock_pending().push_back(request);
        Self::inc(&self.requests_submitted);

        id
    }

    /// Get batch of requests ready for decode step
    ///
    /// Returns request IDs and their current positions
    pub fn get_decode_batch(&self) -> Vec<(u64, usize)> {
        let mut active = self.lock_active();
        let mut pending = self.lock_pending();

        // Promote pending requests to active (up to max_concurrent)
        while active.len() < self.max_concurrent && !pending.is_empty() {
            if let Some(mut req) = pending.pop_front() {
                req.state = MultiRequestState::Decoding;
                active.push(req);
            }
        }

        // Sort by policy
        match self.policy {
            SchedulingPolicy::Fcfs => {
                // Already in arrival order
            },
            SchedulingPolicy::Sjf => {
                active.sort_by_key(|r| r.max_tokens - r.generated.len());
            },
            SchedulingPolicy::RoundRobin => {
                // Rotate - move first to end
                if active.len() > 1 {
                    let first = active.remove(0);
                    active.push(first);
                }
            },
        }

        // Return batch of decoding requests
        active
            .iter()
            .filter(|r| r.state == MultiRequestState::Decoding)
            .take(self.max_batch_size)
            .map(|r| (r.id, r.kv_position))
            .collect()
    }

    /// Record generated token for a request
    pub fn record_token(&self, request_id: u64, token: u32) {
        let mut active = self.lock_active();

        if let Some(req) = active.iter_mut().find(|r| r.id == request_id) {
            // Record TTFT for first token
            if req.first_token_time.is_none() {
                req.first_token_time = Some(std::time::Instant::now());
            }

            req.generated.push(token);
            req.kv_position += 1;
            Self::inc(&self.tokens_generated);

            // Check if complete
            if req.is_complete() {
                req.state = MultiRequestState::Completed;
            }
        }
    }

    /// Move completed requests from active to completed
    pub fn collect_completed(&self) -> Vec<MultiSchedulerRequest> {
        let mut active = self.lock_active();
        let mut completed = self.lock_completed();

        let (done, still_active): (Vec<_>, Vec<_>) = active
            .drain(..)
            .partition(|r| r.state == MultiRequestState::Completed);

        *active = still_active;

        Self::inc_by(&self.requests_completed, done.len() as u64);

        completed.extend(done.iter().cloned());
        done
    }

    /// Run one batch iteration (for simulation)
    pub fn step(&self) {
        Self::inc(&self.batch_iterations);
    }

    /// Get scheduler statistics
    pub fn stats(&self) -> MultiRequestStats {
        let submitted = Self::load(&self.requests_submitted);
        let completed = Self::load(&self.requests_completed);
        let tokens = Self::load(&self.tokens_generated);
        let iterations = Self::load(&self.batch_iterations);

        let pending = self.lock_pending().len();
        let active = self.lock_active().len();

        MultiRequestStats {
            requests_submitted: submitted,
            requests_completed: completed,
            tokens_generated: tokens,
            batch_iterations: iterations,
            pending_requests: pending,
            active_requests: active,
            avg_batch_size: if iterations > 0 {
                tokens as f64 / iterations as f64
            } else {
                0.0
            },
        }
    }
}
