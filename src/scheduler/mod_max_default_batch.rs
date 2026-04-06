
impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_ubatch_tokens: 512,
            max_sbatch_sequences: 8,
            prefer_pure_decode: true,
            max_context_length: 2048,
            dynamic_batching: true,
        }
    }
}

impl BatchConfig {
    /// Create batch config with custom max tokens
    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_ubatch_tokens = max_tokens;
        self
    }

    /// Create batch config with custom max sequences
    pub fn with_max_sequences(mut self, max_seqs: usize) -> Self {
        self.max_sbatch_sequences = max_seqs;
        self
    }
}

/// Batch scheduler for continuous batching
///
/// Coordinates micro-batch and sequence batch creation,
/// implementing llama.cpp-style continuous batching.
pub struct BatchScheduler {
    /// Configuration
    config: BatchConfig,
    /// Current sequence batch
    sbatch: SequenceBatch,
    /// Next sequence index
    next_seq_idx: usize,
    /// Statistics
    stats: BatchStats,
}

/// Statistics for batch scheduler
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BatchStats {
    /// Total micro-batches created
    pub ubatches_created: u64,
    /// Total sequence batches created
    pub sbatches_created: u64,
    /// Total tokens processed
    pub tokens_processed: u64,
    /// Total prefill tokens
    pub prefill_tokens: u64,
    /// Total decode tokens
    pub decode_tokens: u64,
    /// Average tokens per ubatch
    pub avg_ubatch_size: f64,
    /// Average sequences per sbatch
    pub avg_sbatch_size: f64,
}

impl BatchScheduler {
    /// Create a new batch scheduler with default config
    pub fn new() -> Self {
        Self::with_config(BatchConfig::default())
    }

    /// Create a new batch scheduler with custom config
    pub fn with_config(config: BatchConfig) -> Self {
        let max_seqs = config.max_sbatch_sequences;
        Self {
            config,
            sbatch: SequenceBatch::new(max_seqs),
            next_seq_idx: 0,
            stats: BatchStats::default(),
        }
    }

    /// Add a new sequence to the batch scheduler
    pub fn add_sequence(
        &mut self,
        slot_id: usize,
        request_id: u64,
        input_tokens: Vec<u32>,
    ) -> Option<usize> {
        if self.sbatch.is_full() {
            return None;
        }

        let seq_idx = self.next_seq_idx;
        self.next_seq_idx += 1;

        let entry = SequenceBatchEntry::new(seq_idx, slot_id, request_id).with_tokens(input_tokens);

        if self.sbatch.add_sequence(entry) {
            Some(seq_idx)
        } else {
            None
        }
    }

    /// Mark sequence as completed and remove from batch
    pub fn complete_sequence(&mut self, seq_idx: usize) -> Option<SequenceBatchEntry> {
        self.sbatch.remove_sequence(seq_idx)
    }

    /// Transition sequence from prefill to decode
    pub fn start_decode(&mut self, seq_idx: usize, position: usize) -> bool {
        if let Some(entry) = self.sbatch.get_mut(seq_idx) {
            entry.is_prefill = false;
            entry.position = position;
            entry.tokens.clear(); // No longer need prefill tokens
            true
        } else {
            false
        }
    }

    /// Create a micro-batch from current sequences
    ///
    /// Returns a micro-batch optimized for the current state:
    /// - Prefill: Process all prompt tokens for prefill sequences
    /// - Decode: Process one token per decode sequence
    /// - Mixed: Combines both (if config allows)
    pub fn create_ubatch(&mut self) -> MicroBatch {
        let mut ubatch = MicroBatch::with_capacity(self.config.max_ubatch_tokens);

        // Process prefill sequences first (if any)
        for entry in &self.sbatch.sequences {
            if entry.is_prefill {
                // Add all prefill tokens
                for (i, &token_id) in entry.tokens.iter().enumerate() {
                    if ubatch.len() >= self.config.max_ubatch_tokens {
                        break;
                    }
                    ubatch.add_token(BatchToken::new(token_id, entry.seq_idx, i, true));
                }
            }
        }

        // If prefer_pure_decode and we have prefill tokens, return early
        if self.config.prefer_pure_decode && !ubatch.is_empty() && ubatch.is_prefill() {
            self.record_ubatch(&ubatch);
            return ubatch;
        }

        // Add decode tokens
        for entry in &self.sbatch.sequences {
            if !entry.is_prefill {
                if ubatch.len() >= self.config.max_ubatch_tokens {
                    break;
                }
                // Decode sequences process one token at their current position
                // (the actual token ID will be filled in during inference)
                ubatch.add_token(BatchToken::new(
                    0, // Placeholder - will be filled by inference
                    entry.seq_idx,
                    entry.position,
                    false,
                ));
            }
        }

        self.record_ubatch(&ubatch);
        ubatch
    }

    /// Record ubatch statistics
    fn record_ubatch(&mut self, ubatch: &MicroBatch) {
        if ubatch.is_empty() {
            return;
        }

        self.stats.ubatches_created += 1;
        self.stats.tokens_processed += ubatch.len() as u64;
        self.stats.prefill_tokens += ubatch.n_prompt_tokens as u64;
        self.stats.decode_tokens += ubatch.n_decode_tokens as u64;

        // Update rolling average
        let n = self.stats.ubatches_created as f64;
        self.stats.avg_ubatch_size =
            self.stats.avg_ubatch_size * (n - 1.0) / n + ubatch.len() as f64 / n;
    }

    /// Get the current sequence batch
    pub fn sbatch(&self) -> &SequenceBatch {
        &self.sbatch
    }

    /// Get scheduler statistics
    pub fn stats(&self) -> &BatchStats {
        &self.stats
    }

    /// Get configuration
    pub fn config(&self) -> &BatchConfig {
        &self.config
    }

    /// Number of active sequences
    pub fn num_sequences(&self) -> usize {
        self.sbatch.len()
    }

    /// Check if scheduler has capacity
    pub fn has_capacity(&self) -> bool {
        !self.sbatch.is_full()
    }

    /// Current batch utilization
    pub fn utilization(&self) -> f64 {
        self.sbatch.utilization
    }
}

impl Default for BatchScheduler {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// DYNAMIC BATCH PRIORITY SCHEDULING
// ============================================================================
//
// Advanced priority scheduling with:
// - Age-based priority promotion (prevent starvation)
// - Deadline-aware scheduling (SLA support)
// - Priority-weighted token budgets
// - Multi-level feedback queue (MLFQ) style scheduling
// - Fair share allocation across priority levels
//
// Reference: Orca (Yu et al., 2022), vLLM priority scheduling
// ============================================================================

/// Deadline specification for a request
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Deadline {
    /// Target completion time (milliseconds from arrival)
    pub target_latency_ms: u64,
    /// Hard deadline (must complete by this time, else drop)
    pub hard_deadline_ms: Option<u64>,
    /// Soft SLA target (percentage of requests meeting target)
    pub sla_target: f64,
}

impl Default for Deadline {
    fn default() -> Self {
        Self {
            target_latency_ms: 1000, // 1 second default
            hard_deadline_ms: None,
            sla_target: 0.99, // 99% SLA
        }
    }
}

impl Deadline {
    /// Create a deadline with target latency
    pub fn with_target(target_ms: u64) -> Self {
        Self {
            target_latency_ms: target_ms,
            ..Default::default()
        }
    }

    /// Create a strict deadline with hard cutoff
    pub fn strict(target_ms: u64, hard_ms: u64) -> Self {
        Self {
            target_latency_ms: target_ms,
            hard_deadline_ms: Some(hard_ms),
            sla_target: 1.0,
        }
    }
}

/// Dynamic priority configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicPriorityConfig {
    /// Enable age-based priority promotion
    pub enable_age_promotion: bool,
    /// Time (ms) before promoting to next priority level
    pub promotion_interval_ms: u64,
    /// Maximum priority level after promotion (prevent runaway)
    pub max_promoted_priority: Priority,
    /// Token budget per priority level (proportion of batch)
    pub priority_budgets: [f64; 4], // Low, Normal, High, Critical
    /// Enable deadline-aware scheduling
    pub enable_deadline_scheduling: bool,
    /// Urgency boost factor for approaching deadlines
    pub urgency_factor: f64,
    /// Minimum tokens to allocate per request
    pub min_tokens_per_request: usize,
    /// Enable fair share scheduling
    pub enable_fair_share: bool,
}

impl Default for DynamicPriorityConfig {
    fn default() -> Self {
        Self {
            enable_age_promotion: true,
            promotion_interval_ms: 5000, // Promote after 5 seconds
            max_promoted_priority: Priority::High,
            // Budget allocation: Low=5%, Normal=30%, High=40%, Critical=25%
            priority_budgets: [0.05, 0.30, 0.40, 0.25],
            enable_deadline_scheduling: true,
            urgency_factor: 2.0,
            min_tokens_per_request: 1,
            enable_fair_share: true,
        }
    }
}

impl DynamicPriorityConfig {
    /// Create config with custom budgets
    pub fn with_budgets(budgets: [f64; 4]) -> Self {
        Self {
            priority_budgets: budgets,
            ..Default::default()
        }
    }

    /// Disable age promotion
    pub fn no_promotion(mut self) -> Self {
        self.enable_age_promotion = false;
        self
    }

    /// Set promotion interval
    pub fn with_promotion_interval(mut self, ms: u64) -> Self {
        self.promotion_interval_ms = ms;
        self
    }
}

/// Request entry with dynamic priority tracking
#[derive(Debug, Clone)]
pub struct DynamicRequest {
    /// Base request data
    pub request_id: u64,
    /// Input tokens
    pub input_ids: Vec<u32>,
    /// Maximum tokens to generate
    pub max_tokens: usize,
    /// Original priority (as submitted)
    pub original_priority: Priority,
    /// Effective priority (may be promoted)
    pub effective_priority: Priority,
    /// Arrival time
    pub arrival_time: Instant,
    /// Deadline specification
    pub deadline: Option<Deadline>,
    /// Number of times priority was promoted
    pub promotions: u32,
    /// Current state
    pub state: SequenceState,
    /// Generated tokens
    pub generated_tokens: Vec<u32>,
    /// Sequence ID (when scheduled)
    pub seq_id: Option<SeqId>,
    /// Time-to-first-token (if started)
    pub ttft_ms: Option<f64>,
}

impl DynamicRequest {
    /// Create a new dynamic request
    pub fn new(request_id: u64, input_ids: Vec<u32>, max_tokens: usize) -> Self {
        Self {
            request_id,
            input_ids,
            max_tokens,
            original_priority: Priority::Normal,
            effective_priority: Priority::Normal,
            arrival_time: Instant::now(),
            deadline: None,
            promotions: 0,
            state: SequenceState::Waiting,
            generated_tokens: Vec::new(),
            seq_id: None,
            ttft_ms: None,
        }
    }

    /// Set priority
    pub fn with_priority(mut self, priority: Priority) -> Self {
        self.original_priority = priority;
        self.effective_priority = priority;
        self
    }

    /// Set deadline
    pub fn with_deadline(mut self, deadline: Deadline) -> Self {
        self.deadline = Some(deadline);
        self
    }

    /// Wait time since arrival
    pub fn wait_time_ms(&self) -> u64 {
        self.arrival_time.elapsed().as_millis() as u64
    }

    /// Check if deadline is approaching (within 2x target latency)
    pub fn is_urgent(&self) -> bool {
        if let Some(deadline) = &self.deadline {
            let elapsed = self.wait_time_ms();
            elapsed >= deadline.target_latency_ms / 2
        } else {
            false
        }
    }

    /// Check if hard deadline has passed
    pub fn is_expired(&self) -> bool {
        if let Some(deadline) = &self.deadline {
            if let Some(hard) = deadline.hard_deadline_ms {
                return self.wait_time_ms() > hard;
            }
        }
        false
    }

    /// Calculate urgency score (0.0 to 1.0+)
    /// Higher score = more urgent
    pub fn urgency_score(&self) -> f64 {
        if let Some(deadline) = &self.deadline {
            let elapsed = self.wait_time_ms() as f64;
            let target = deadline.target_latency_ms as f64;
            if target > 0.0 {
                elapsed / target
            } else {
                0.0
            }
        } else {
            0.0
        }
    }

    /// Remaining tokens to generate
    pub fn remaining_tokens(&self) -> usize {
        self.max_tokens.saturating_sub(self.generated_tokens.len())
    }

    /// Total tokens (input + generated)
    pub fn total_tokens(&self) -> usize {
        self.input_ids.len() + self.generated_tokens.len()
    }
}
