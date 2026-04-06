
impl Scheduler {
    /// Create a new scheduler
    pub fn new(max_batch_size: usize, max_queue_size: usize) -> Self {
        Self {
            requests: HashMap::new(),
            waiting_queue: BinaryHeap::new(),
            running: Vec::new(),
            preempted: VecDeque::new(),
            max_batch_size,
            max_queue_size,
            max_tokens_per_batch: max_batch_size * 2048, // Default: assume 2k context
            next_request_id: 0,
            stats: SchedulerStats::default(),
            total_wait_time_ms: 0.0,
        }
    }

    /// Set maximum tokens per batch
    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens_per_batch = max_tokens;
        self
    }

    /// Add a new request to the queue
    pub fn add_request(
        &mut self,
        input_ids: Vec<u32>,
        max_tokens: usize,
    ) -> Result<u64, SchedulerError> {
        self.add_request_with_priority(input_ids, max_tokens, Priority::Normal)
    }

    /// Add a request with priority
    pub fn add_request_with_priority(
        &mut self,
        input_ids: Vec<u32>,
        max_tokens: usize,
        priority: Priority,
    ) -> Result<u64, SchedulerError> {
        if self.waiting_queue.len() >= self.max_queue_size {
            return Err(SchedulerError::QueueFull {
                capacity: self.max_queue_size,
            });
        }

        let request_id = self.next_request_id;
        self.next_request_id += 1;

        let request =
            SchedulerRequest::new(request_id, input_ids, max_tokens).with_priority(priority);
        let entry = PriorityEntry {
            priority,
            arrival_time: request.arrival_time,
            request_id,
        };

        self.requests.insert(request_id, request);
        self.waiting_queue.push(entry);
        self.stats.total_requests += 1;
        self.stats.queue_depth = self.waiting_queue.len();

        Ok(request_id)
    }

    /// Schedule one iteration of generation
    pub fn schedule(
        &mut self,
        kv_cache: &mut PagedKvCache,
        eos_token: u32,
    ) -> Result<SchedulerOutput, SchedulerError> {
        let mut output = SchedulerOutput::default();

        // 1. Check for completed sequences
        self.check_completions(&mut output, eos_token);

        // 2. Preempt if memory pressure (simplified: check if we can fit new sequences)
        self.handle_preemption(&mut output, kv_cache);

        // 3. Resume preempted sequences if possible
        self.resume_preempted(&mut output, kv_cache)?;

        // 4. Schedule new sequences from waiting queue
        self.schedule_waiting(&mut output, kv_cache)?;

        // 5. Build final output
        for &request_id in &self.running {
            if let Some(request) = self.requests.get(&request_id) {
                if let Some(seq_id) = request.seq_id {
                    output.scheduled_seq_ids.push(seq_id);
                    output.scheduled_request_ids.push(request_id);

                    if request.iterations == 0 {
                        // Prefill phase
                        output.num_prefill_tokens += request.input_ids.len();
                    } else {
                        // Decode phase (1 token per sequence)
                        output.num_decode_tokens += 1;
                    }
                }
            }
        }

        self.stats.running_count = self.running.len();
        self.stats.queue_depth = self.waiting_queue.len();

        Ok(output)
    }

    /// Schedule one step (alias for schedule).
    ///
    /// Contract: continuous-batching-v1 / token_budget
    pub fn schedule_step(
        &mut self,
        kv_cache: &mut PagedKvCache,
        eos_token: u32,
    ) -> Result<SchedulerOutput, SchedulerError> {
        contract_pre_token_budget!();
        self.schedule(kv_cache, eos_token)
    }

    /// Update scheduler after generation iteration
    pub fn update_after_iteration(&mut self, generated_tokens: &HashMap<u64, u32>) {
        for (&request_id, &token) in generated_tokens {
            if let Some(request) = self.requests.get_mut(&request_id) {
                request.generated_tokens.push(token);
                request.iterations += 1;
            }
        }
    }

    /// Mark request as complete
    pub fn complete_request(&mut self, request_id: u64, kv_cache: &mut PagedKvCache) {
        if let Some(request) = self.requests.get_mut(&request_id) {
            request.state = SequenceState::Completed;

            // Free KV cache
            if let Some(seq_id) = request.seq_id {
                kv_cache.free_sequence(seq_id);
            }

            // Update stats
            self.stats.completed_requests += 1;
            let wait_time = request.wait_time().as_secs_f64() * 1000.0;
            self.total_wait_time_ms += wait_time;
            self.stats.avg_wait_time_ms =
                self.total_wait_time_ms / self.stats.completed_requests as f64;
        }

        // Remove from running
        self.running.retain(|&id| id != request_id);
    }

    /// Get request by ID
    pub fn get_request(&self, request_id: u64) -> Option<&SchedulerRequest> {
        self.requests.get(&request_id)
    }

    /// Get scheduler statistics
    pub fn stats(&self) -> &SchedulerStats {
        &self.stats
    }

    /// Check for completed sequences
    fn check_completions(&mut self, output: &mut SchedulerOutput, eos_token: u32) {
        let completed: Vec<u64> = self
            .running
            .iter()
            .filter(|&&id| {
                self.requests
                    .get(&id)
                    .is_some_and(|r| r.is_complete(eos_token))
            })
            .copied()
            .collect();

        for request_id in completed {
            if let Some(request) = self.requests.get_mut(&request_id) {
                request.state = SequenceState::Completed;
            }
            output.completed_request_ids.push(request_id);
        }
    }

    /// Handle preemption under memory pressure
    fn handle_preemption(&mut self, output: &mut SchedulerOutput, kv_cache: &mut PagedKvCache) {
        contract_pre_scheduling_fairness!();
        // Simple preemption: if running at max and waiting queue has higher priority
        if self.running.len() >= self.max_batch_size && !self.waiting_queue.is_empty() {
            // Check if waiting has higher priority than lowest running
            if let Some(waiting_entry) = self.waiting_queue.peek() {
                let min_running_priority = self
                    .running
                    .iter()
                    .filter_map(|&id| self.requests.get(&id))
                    .map(|r| r.priority)
                    .min()
                    .unwrap_or(Priority::Critical);

                if waiting_entry.priority > min_running_priority {
                    // Find lowest priority running request to preempt
                    if let Some(&preempt_id) = self.running.iter().find(|&&id| {
                        self.requests
                            .get(&id)
                            .is_some_and(|r| r.priority == min_running_priority)
                    }) {
                        // Preempt the request
                        if let Some(request) = self.requests.get_mut(&preempt_id) {
                            request.state = SequenceState::Preempted;
                            if let Some(seq_id) = request.seq_id {
                                output.preempted_seq_ids.push(seq_id);
                                kv_cache.free_sequence(seq_id);
                            }
                            request.seq_id = None;
                        }
                        self.running.retain(|&id| id != preempt_id);
                        self.preempted.push_back(preempt_id);
                        self.stats.preemptions += 1;
                    }
                }
            }
        }
    }

    /// Resume preempted sequences
    fn resume_preempted(
        &mut self,
        _output: &mut SchedulerOutput,
        kv_cache: &mut PagedKvCache,
    ) -> Result<(), SchedulerError> {
        while self.running.len() < self.max_batch_size {
            if let Some(request_id) = self.preempted.pop_front() {
                if let Some(request) = self.requests.get_mut(&request_id) {
                    // Try to allocate KV cache
                    let total_tokens = request.total_tokens();
                    match kv_cache.allocate_sequence(total_tokens) {
                        Ok(seq_id) => {
                            request.seq_id = Some(seq_id);
                            request.state = SequenceState::Running;
                            self.running.push(request_id);
                        },
                        Err(_) => {
                            // Can't allocate, put back in preempted queue
                            self.preempted.push_front(request_id);
                            break;
                        },
                    }
                }
            } else {
                break;
            }
        }
        Ok(())
    }

    /// Schedule waiting requests
    fn schedule_waiting(
        &mut self,
        _output: &mut SchedulerOutput,
        kv_cache: &mut PagedKvCache,
    ) -> Result<(), SchedulerError> {
        while self.running.len() < self.max_batch_size {
            if let Some(entry) = self.waiting_queue.pop() {
                if let Some(request) = self.requests.get_mut(&entry.request_id) {
                    let total_tokens = request.input_ids.len();
                    match kv_cache.allocate_sequence(total_tokens) {
                        Ok(seq_id) => {
                            request.seq_id = Some(seq_id);
                            request.state = SequenceState::Running;
                            self.running.push(entry.request_id);
                        },
                        Err(_) => {
                            // Can't allocate, put back in queue (at front since already popped)
                            self.waiting_queue.push(entry);
                            break;
                        },
                    }
                }
            } else {
                break;
            }
        }
        Ok(())
    }

    /// Number of waiting requests
    pub fn waiting_count(&self) -> usize {
        self.waiting_queue.len()
    }

    /// Number of running requests
    pub fn running_count(&self) -> usize {
        self.running.len()
    }

    /// Number of preempted requests
    pub fn preempted_count(&self) -> usize {
        self.preempted.len()
    }
}

// ============================================================================
// SLOT-BASED SERVER CONCURRENCY (per llama.cpp)
// ============================================================================
//
// llama.cpp uses a slot-based architecture for concurrent inference:
// - Fixed number of slots, each with its own KV cache allocation
// - State machine: IDLE → PROCESSING → GENERATING → (complete) → IDLE
// - Slots can be dynamically assigned to incoming requests
// - Enables efficient handling of multiple concurrent clients
// ============================================================================

/// Slot state machine (per llama.cpp server architecture)
///
/// Each slot transitions through these states:
/// - IDLE: Ready to accept new requests
/// - PROCESSING: Initial prompt processing (prefill phase)
/// - GENERATING: Token generation (decode phase)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SlotState {
    /// Slot is available for new requests
    Idle,
    /// Processing initial prompt (prefill)
    Processing,
    /// Generating tokens (decode)
    Generating,
}

impl Default for SlotState {
    fn default() -> Self {
        Self::Idle
    }
}

/// Server slot for concurrent inference
///
/// Per llama.cpp: Each slot manages one inference request with its own
/// KV cache allocation and state machine.
#[derive(Debug, Clone)]
pub struct Slot {
    /// Unique slot ID
    pub id: usize,
    /// Current state
    pub state: SlotState,
    /// Assigned request ID (None if idle)
    pub request_id: Option<u64>,
    /// Sequence ID for KV cache
    pub seq_id: Option<SeqId>,
    /// Input tokens (prompt)
    pub input_tokens: Vec<u32>,
    /// Generated tokens so far
    pub generated_tokens: Vec<u32>,
    /// Maximum tokens to generate
    pub max_tokens: usize,
    /// Number of prompt tokens processed
    pub n_prompt_tokens_processed: usize,
    /// Generation start time
    pub generation_start: Option<Instant>,
    /// Total prompt processing time (ms)
    pub prompt_time_ms: f64,
    /// Total generation time (ms)
    pub generation_time_ms: f64,
}

impl Slot {
    /// Create a new idle slot
    pub fn new(id: usize) -> Self {
        Self {
            id,
            state: SlotState::Idle,
            request_id: None,
            seq_id: None,
            input_tokens: Vec::new(),
            generated_tokens: Vec::new(),
            max_tokens: 0,
            n_prompt_tokens_processed: 0,
            generation_start: None,
            prompt_time_ms: 0.0,
            generation_time_ms: 0.0,
        }
    }

    /// Check if slot is available
    pub fn is_idle(&self) -> bool {
        self.state == SlotState::Idle
    }

    /// Check if slot is actively generating
    pub fn is_generating(&self) -> bool {
        self.state == SlotState::Generating
    }

    /// Assign a request to this slot
    pub fn assign(&mut self, request_id: u64, input_tokens: Vec<u32>, max_tokens: usize) {
        self.state = SlotState::Processing;
        self.request_id = Some(request_id);
        self.input_tokens = input_tokens;
        self.max_tokens = max_tokens;
        self.generated_tokens.clear();
        self.n_prompt_tokens_processed = 0;
        self.prompt_time_ms = 0.0;
        self.generation_time_ms = 0.0;
        self.generation_start = None;
    }

    /// Transition from processing to generating
    pub fn start_generation(&mut self, prompt_time_ms: f64) {
        self.state = SlotState::Generating;
        self.prompt_time_ms = prompt_time_ms;
        self.generation_start = Some(Instant::now());
    }

    /// Add a generated token
    pub fn add_token(&mut self, token: u32) {
        self.generated_tokens.push(token);
    }

    /// Check if generation is complete
    pub fn is_complete(&self, eos_token: u32) -> bool {
        if self.generated_tokens.len() >= self.max_tokens {
            return true;
        }
        if let Some(&last) = self.generated_tokens.last() {
            if last == eos_token {
                return true;
            }
        }
        false
    }

    /// Finish generation and reset to idle
    pub fn finish(&mut self) {
        if let Some(start) = self.generation_start {
            self.generation_time_ms = start.elapsed().as_secs_f64() * 1000.0;
        }
        self.state = SlotState::Idle;
        self.request_id = None;
        self.seq_id = None;
    }

    /// Get tokens per second for this slot's generation
    pub fn tokens_per_second(&self) -> f64 {
        if self.generation_time_ms > 0.0 {
            self.generated_tokens.len() as f64 / (self.generation_time_ms / 1000.0)
        } else {
            0.0
        }
    }
}

/// Slot manager for concurrent inference
///
/// Per llama.cpp: Manages a fixed pool of slots for handling concurrent requests.
/// Each slot has its own KV cache allocation and can process one request at a time.
#[derive(Debug)]
pub struct SlotManager {
    /// Available slots
    slots: Vec<Slot>,
    /// Maximum context length per slot
    pub max_context_length: usize,
    /// Next request ID
    next_request_id: u64,
}
