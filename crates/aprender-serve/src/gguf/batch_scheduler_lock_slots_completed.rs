
#[cfg(feature = "gpu")]
impl ContinuousBatchScheduler {
    /// Lock the slots mutex, panicking on poison.
    fn lock_slots(&self) -> std::sync::MutexGuard<'_, Vec<SlotState>> {
        self.slots.lock().expect("Mutex poisoned")
    }

    /// Lock the completed queue mutex, panicking on poison.
    fn lock_completed(&self) -> std::sync::MutexGuard<'_, Vec<(u64, Vec<u32>)>> {
        self.completed.lock().expect("Mutex poisoned")
    }

    /// Lock the KV caches mutex, panicking on poison.
    fn lock_caches(&self) -> std::sync::MutexGuard<'_, Vec<OwnedQuantizedKVCache>> {
        self.caches.lock().expect("Mutex poisoned")
    }

    /// Create scheduler with specified number of slots
    ///
    /// # Arguments
    /// * `num_slots` - Maximum concurrent requests (typically 32-64)
    /// * `num_layers` - Number of transformer layers (for KV cache)
    /// * `hidden_dim` - Hidden dimension (for KV cache)
    /// * `max_seq_len` - Maximum sequence length (for KV cache)
    pub fn new(num_slots: usize, num_layers: usize, hidden_dim: usize, max_seq_len: usize) -> Self {
        let slots = vec![SlotState::Empty; num_slots];
        let caches = (0..num_slots)
            .map(|_| OwnedQuantizedKVCache::new(num_layers, hidden_dim, max_seq_len))
            .collect();

        Self {
            slots: std::sync::Mutex::new(slots),
            caches: std::sync::Mutex::new(caches),
            num_slots,
            completed: std::sync::Mutex::new(Vec::new()),
            next_id: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Count slots matching a predicate.
    fn count_slots_where(&self, predicate: fn(&SlotState) -> bool) -> usize {
        self.lock_slots().iter().filter(|s| predicate(s)).count()
    }

    /// Submit a new request to the scheduler
    ///
    /// Returns request ID if slot available, None if all slots full
    pub fn submit(
        &self,
        prompt_tokens: Vec<u32>,
        max_tokens: usize,
        temperature: f32,
        top_k: usize,
    ) -> Option<u64> {
        let mut slots = self.lock_slots();

        // Find first empty slot
        let empty_idx = slots.iter().position(SlotState::is_empty)?;

        let request_id = self
            .next_id
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        slots[empty_idx] = SlotState::Active {
            request_id,
            prompt_tokens,
            generated_tokens: Vec::new(),
            max_tokens,
            temperature,
            top_k,
        };

        Some(request_id)
    }

    /// Get number of active slots
    pub fn active_count(&self) -> usize {
        self.count_slots_where(SlotState::is_active)
    }

    /// Get number of empty slots
    pub fn empty_count(&self) -> usize {
        self.count_slots_where(SlotState::is_empty)
    }

    /// Check if any slot has completed request
    pub fn has_completed(&self) -> bool {
        !self.lock_completed().is_empty()
    }

    /// Retrieve completed request results
    pub fn poll_completed(&self) -> Vec<(u64, Vec<u32>)> {
        std::mem::take(&mut *self.lock_completed())
    }

    /// Mark a request as completed and move to completed queue
    pub fn complete_request(&self, slot_idx: usize, tokens: Vec<u32>) {
        let mut slots = self.lock_slots();
        let mut completed = self.lock_completed();

        if slot_idx < slots.len() {
            if let SlotState::Active { request_id, .. } = &slots[slot_idx] {
                let id = *request_id;
                // Move to completed
                completed.push((id, tokens));
                // Free the slot
                slots[slot_idx] = SlotState::Empty;

                // Reset KV cache for this slot
                self.lock_caches()[slot_idx].reset();
            }
        }
    }

    /// Get active slot indices and their current positions
    pub fn get_active_slots(&self) -> Vec<(usize, usize)> {
        self.lock_slots()
            .iter()
            .enumerate()
            .filter_map(|(idx, slot)| match slot {
                SlotState::Active {
                    prompt_tokens,
                    generated_tokens,
                    ..
                } => {
                    let pos = prompt_tokens.len() + generated_tokens.len();
                    Some((idx, pos))
                },
                _ => None,
            })
            .collect()
    }

    /// Get utilization (active_slots / total_slots)
    pub fn utilization(&self) -> f64 {
        let active = self.active_count();
        active as f64 / self.num_slots as f64
    }
}

/// Speculative decoding configuration (PARITY-029)
#[cfg(feature = "gpu")]
#[derive(Debug, Clone)]
pub struct SpeculativeConfig {
    /// Number of tokens to speculatively generate per step
    pub speculation_length: usize,
    /// Temperature for draft model (lower = more deterministic)
    pub draft_temperature: f32,
    /// Whether to use same model for draft (self-speculative)
    pub self_speculative: bool,
}

#[cfg(feature = "gpu")]
impl Default for SpeculativeConfig {
    fn default() -> Self {
        Self {
            speculation_length: 4,
            draft_temperature: 0.0,
            self_speculative: true,
        }
    }
}

/// Result of speculative decoding verification step
#[cfg(feature = "gpu")]
#[derive(Debug, Clone)]
pub struct VerificationResult {
    /// Number of draft tokens accepted
    pub accepted_count: usize,
    /// Total draft tokens generated
    pub draft_count: usize,
    /// Accepted tokens (verified by target model)
    pub accepted_tokens: Vec<u32>,
    /// Whether all draft tokens were accepted
    pub all_accepted: bool,
}

/// Speculative decoder for accelerated token generation (PARITY-029)
///
/// Implements speculative decoding (Leviathan et al., 2023):
/// 1. Draft model generates K candidate tokens quickly
/// 2. Target model verifies all K tokens in parallel
/// 3. Accept tokens until first rejection, then resample
///
/// This enables O(K) speedup when draft acceptance rate is high.
#[cfg(feature = "gpu")]
pub struct SpeculativeDecoder {
    /// Speculative decoding configuration
    pub config: SpeculativeConfig,
    /// Statistics: total draft tokens generated
    pub total_draft_tokens: std::sync::atomic::AtomicU64,
    /// Statistics: total draft tokens accepted
    pub total_accepted_tokens: std::sync::atomic::AtomicU64,
}

#[cfg(feature = "gpu")]
impl SpeculativeDecoder {
    /// Create a new `AtomicU64` initialized to zero.
    fn new_counter() -> std::sync::atomic::AtomicU64 {
        std::sync::atomic::AtomicU64::new(0)
    }

    /// Load an atomic counter with Relaxed ordering.
    fn load_relaxed(counter: &std::sync::atomic::AtomicU64) -> u64 {
        counter.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Add a value to an atomic counter (Relaxed ordering).
    fn add(counter: &std::sync::atomic::AtomicU64, val: u64) {
        counter.fetch_add(val, std::sync::atomic::Ordering::Relaxed);
    }

    /// Reset an atomic counter to zero (Relaxed ordering).
    fn reset_counter(counter: &std::sync::atomic::AtomicU64) {
        counter.store(0, std::sync::atomic::Ordering::Relaxed);
    }

    /// Create new speculative decoder with default config
    pub fn new() -> Self {
        Self {
            config: SpeculativeConfig::default(),
            total_draft_tokens: Self::new_counter(),
            total_accepted_tokens: Self::new_counter(),
        }
    }

    /// Create speculative decoder with custom config
    pub fn with_config(config: SpeculativeConfig) -> Self {
        Self {
            config,
            total_draft_tokens: Self::new_counter(),
            total_accepted_tokens: Self::new_counter(),
        }
    }

    /// Get acceptance rate (accepted / total draft tokens)
    pub fn acceptance_rate(&self) -> f64 {
        let total = Self::load_relaxed(&self.total_draft_tokens);
        let accepted = Self::load_relaxed(&self.total_accepted_tokens);
        if total == 0 {
            return 0.0;
        }
        accepted as f64 / total as f64
    }

    /// Verify draft tokens against target model logits
    ///
    /// # Arguments
    /// * `draft_tokens` - Candidate tokens from draft model
    /// * `target_logits` - Logits from target model for each position
    /// * `temperature` - Sampling temperature for rejection sampling
    ///
    /// # Returns
    /// VerificationResult with accepted tokens and statistics
    pub fn verify_draft(
        &self,
        draft_tokens: &[u32],
        target_logits: &[Vec<f32>],
        temperature: f32,
    ) -> VerificationResult {
        let mut accepted_tokens = Vec::with_capacity(draft_tokens.len());
        let mut accepted_count = 0;

        // Verify each draft token against target model distribution
        for (i, &draft_token) in draft_tokens.iter().enumerate() {
            if i >= target_logits.len() {
                break;
            }

            let logits = &target_logits[i];

            // Find target model's top token
            let (target_token, _) = logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or((0, &0.0));

            // Accept if draft matches target (greedy case)
            if temperature == 0.0 {
                if draft_token == target_token as u32 {
                    accepted_tokens.push(draft_token);
                    accepted_count += 1;
                } else {
                    // Reject and use target's token instead
                    accepted_tokens.push(target_token as u32);
                    accepted_count += 1;
                    break; // Stop at first mismatch
                }
            } else {
                // Rejection sampling for non-greedy decoding
                // P(accept) = min(1, p_target(x) / p_draft(x))
                // For simplicity, accept if draft is in top-k of target
                let mut sorted_indices: Vec<usize> = (0..logits.len()).collect();
                sorted_indices.sort_by(|&a, &b| {
                    logits[b]
                        .partial_cmp(&logits[a])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });

                let top_k = 10; // Accept if in top-10
                let in_top_k = sorted_indices
                    .iter()
                    .take(top_k)
                    .any(|&idx| idx == draft_token as usize);

                if in_top_k {
                    accepted_tokens.push(draft_token);
                    accepted_count += 1;
                } else {
                    // Reject, use target's sampled token
                    accepted_tokens.push(sorted_indices[0] as u32);
                    accepted_count += 1;
                    break;
                }
            }
        }

        // Update statistics
        Self::add(&self.total_draft_tokens, draft_tokens.len() as u64);
        Self::add(&self.total_accepted_tokens, accepted_count as u64);

        VerificationResult {
            accepted_count,
            draft_count: draft_tokens.len(),
            accepted_tokens,
            all_accepted: accepted_count == draft_tokens.len(),
        }
    }

    /// Calculate expected speedup based on acceptance rate
    ///
    /// Speedup = K * acceptance_rate + 1 (always get at least 1 token)
    pub fn expected_speedup(&self) -> f64 {
        let k = self.config.speculation_length as f64;
        let acceptance_rate = self.acceptance_rate();
        k * acceptance_rate + 1.0
    }

    /// Reset statistics
    pub fn reset_stats(&self) {
        Self::reset_counter(&self.total_draft_tokens);
        Self::reset_counter(&self.total_accepted_tokens);
    }
}

#[cfg(feature = "gpu")]
impl Default for SpeculativeDecoder {
    fn default() -> Self {
        Self::new()
    }
}

/// GPU Buffer Pool for zero-allocation inference (PARITY-031, IMP-309)
///
/// Pre-allocates GPU buffers during warmup to eliminate allocation overhead
/// during generation. Uses a pool of reusable buffers for each tensor type.
///
/// # Key Properties
/// - Zero GPU malloc after warmup phase
/// - Pre-allocated buffers for common tensor sizes
/// - Thread-safe buffer borrowing and return
///
/// # Buffer Types
/// - Hidden state buffers: [batch_size, hidden_dim]
/// - Intermediate buffers: [batch_size, intermediate_dim]
/// - Attention score buffers: [batch_size, num_heads, seq_len]
/// - KV cache buffers: [num_layers, seq_len, hidden_dim]
#[cfg(feature = "gpu")]
pub struct GpuBufferPool {
    /// Pre-allocated hidden state buffers
    hidden_buffers: std::sync::Mutex<Vec<Vec<f32>>>,
    /// Pre-allocated intermediate buffers (FFN)
    intermediate_buffers: std::sync::Mutex<Vec<Vec<f32>>>,
    /// Pre-allocated attention score buffers
    attention_buffers: std::sync::Mutex<Vec<Vec<f32>>>,
    /// Buffer dimensions for validation
    hidden_dim: usize,
    intermediate_dim: usize,
    max_seq_len: usize,
    num_heads: usize,
    /// Pool size per buffer type
    pool_size: usize,
    /// Statistics: buffers borrowed
    pub borrows: std::sync::atomic::AtomicU64,
    /// Statistics: buffers returned
    pub returns: std::sync::atomic::AtomicU64,
    /// Statistics: allocations after warmup (should be 0)
    pub post_warmup_allocs: std::sync::atomic::AtomicU64,
    /// Whether warmup is complete
    warmed_up: std::sync::atomic::AtomicBool,
}
