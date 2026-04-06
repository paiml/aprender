
/// Statistics for dynamic priority scheduling
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DynamicSchedulerStats {
    /// Total requests processed
    pub total_requests: u64,
    /// Requests completed
    pub completed_requests: u64,
    /// Requests that met SLA
    pub sla_met: u64,
    /// Requests that missed SLA
    pub sla_missed: u64,
    /// Requests dropped (hard deadline exceeded)
    pub dropped_requests: u64,
    /// Total priority promotions
    pub promotions: u64,
    /// Average time-to-first-token (ms)
    pub avg_ttft_ms: f64,
    /// p99 time-to-first-token (ms)
    pub p99_ttft_ms: f64,
    /// Tokens allocated per priority level
    pub tokens_by_priority: [u64; 4],
    /// Current queue depth per priority
    pub queue_depth_by_priority: [usize; 4],
}

/// Dynamic batch priority scheduler
///
/// Implements advanced priority scheduling with:
/// - Age-based priority promotion (MLFQ-style)
/// - Deadline-aware scheduling for SLA compliance
/// - Fair share token budget allocation
/// - Urgency-based boosting for time-sensitive requests
pub struct DynamicPriorityScheduler {
    /// Configuration
    config: DynamicPriorityConfig,
    /// All requests by ID
    requests: HashMap<u64, DynamicRequest>,
    /// Priority queues (one per level)
    priority_queues: [VecDeque<u64>; 4],
    /// Running requests
    running: Vec<u64>,
    /// Next request ID
    next_request_id: u64,
    /// Statistics
    stats: DynamicSchedulerStats,
    /// TTFT samples for percentile calculation
    ttft_samples: Vec<f64>,
    /// Total batch token budget
    batch_token_budget: usize,
}

impl DynamicPriorityScheduler {
    /// Create a new dynamic priority scheduler
    pub fn new(batch_token_budget: usize) -> Self {
        Self::with_config(batch_token_budget, DynamicPriorityConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(batch_token_budget: usize, config: DynamicPriorityConfig) -> Self {
        Self {
            config,
            requests: HashMap::new(),
            priority_queues: [
                VecDeque::new(),
                VecDeque::new(),
                VecDeque::new(),
                VecDeque::new(),
            ],
            running: Vec::new(),
            next_request_id: 0,
            stats: DynamicSchedulerStats::default(),
            ttft_samples: Vec::new(),
            batch_token_budget,
        }
    }

    /// Add a request with priority and optional deadline
    pub fn add_request(
        &mut self,
        input_ids: Vec<u32>,
        max_tokens: usize,
        priority: Priority,
        deadline: Option<Deadline>,
    ) -> u64 {
        let request_id = self.next_request_id;
        self.next_request_id += 1;

        let mut request =
            DynamicRequest::new(request_id, input_ids, max_tokens).with_priority(priority);
        if let Some(d) = deadline {
            request = request.with_deadline(d);
        }

        // Add to appropriate priority queue
        let queue_idx = priority as usize;
        self.priority_queues[queue_idx].push_back(request_id);
        self.requests.insert(request_id, request);

        self.stats.total_requests += 1;
        self.update_queue_depths();

        request_id
    }

    /// Add a simple request (Normal priority, no deadline)
    pub fn add_simple_request(&mut self, input_ids: Vec<u32>, max_tokens: usize) -> u64 {
        self.add_request(input_ids, max_tokens, Priority::Normal, None)
    }

    /// Perform age-based priority promotion
    pub fn promote_aged_requests(&mut self) {
        if !self.config.enable_age_promotion {
            return;
        }

        let promotion_threshold = self.config.promotion_interval_ms;
        let max_priority = self.config.max_promoted_priority;

        // Check each queue except Critical (can't promote beyond Critical)
        for queue_idx in 0..3 {
            let current_priority = match queue_idx {
                0 => Priority::Low,
                1 => Priority::Normal,
                2 => Priority::High,
                _ => continue,
            };

            // Skip if current priority is already at max promoted level
            if current_priority >= max_priority {
                continue;
            }

            // Find requests to promote
            let mut to_promote = Vec::new();
            for &request_id in &self.priority_queues[queue_idx] {
                if let Some(request) = self.requests.get(&request_id) {
                    let promotions_time = promotion_threshold * (request.promotions as u64 + 1);
                    if request.wait_time_ms() >= promotions_time {
                        to_promote.push(request_id);
                    }
                }
            }

            // Promote requests
            for request_id in to_promote {
                self.promote_request(request_id);
            }
        }
    }

    /// Promote a single request to next priority level
    fn promote_request(&mut self, request_id: u64) {
        if let Some(request) = self.requests.get_mut(&request_id) {
            let current_idx = request.effective_priority as usize;
            let max_idx = self.config.max_promoted_priority as usize;

            if current_idx < max_idx {
                // Remove from current queue
                self.priority_queues[current_idx].retain(|&id| id != request_id);

                // Promote
                let new_priority = match current_idx + 1 {
                    1 => Priority::Normal,
                    2 => Priority::High,
                    3 => Priority::Critical,
                    _ => return,
                };
                request.effective_priority = new_priority;
                request.promotions += 1;

                // Add to new queue (at front since it's been waiting)
                self.priority_queues[current_idx + 1].push_front(request_id);
                self.stats.promotions += 1;
            }
        }
    }

    /// Drop expired requests (hard deadline exceeded)
    pub fn drop_expired(&mut self) -> Vec<u64> {
        let mut dropped = Vec::new();

        for queue in &mut self.priority_queues {
            let mut to_remove = Vec::new();
            for &request_id in queue.iter() {
                if let Some(request) = self.requests.get(&request_id) {
                    if request.is_expired() {
                        to_remove.push(request_id);
                    }
                }
            }

            for request_id in to_remove {
                queue.retain(|&id| id != request_id);
                if let Some(mut request) = self.requests.remove(&request_id) {
                    request.state = SequenceState::Failed;
                    dropped.push(request_id);
                    self.stats.dropped_requests += 1;
                }
            }
        }

        self.update_queue_depths();
        dropped
    }

    /// Schedule requests using dynamic priority and token budgets
    ///
    /// Returns (scheduled_request_ids, tokens_allocated_per_request)
    pub fn schedule(&mut self, available_slots: usize) -> Vec<(u64, usize)> {
        // First, handle promotions and expirations
        self.promote_aged_requests();
        self.drop_expired();

        let mut scheduled = Vec::new();
        let mut remaining_budget = self.batch_token_budget;
        let mut remaining_slots = available_slots;

        // Calculate token budgets per priority level
        let budgets: [usize; 4] = if self.config.enable_fair_share {
            self.config
                .priority_budgets
                .map(|b| (b * self.batch_token_budget as f64) as usize)
        } else {
            [
                remaining_budget,
                remaining_budget,
                remaining_budget,
                remaining_budget,
            ]
        };

        // Schedule from highest priority to lowest
        for queue_idx in (0..4).rev() {
            if remaining_slots == 0 || remaining_budget == 0 {
                break;
            }

            let queue = &mut self.priority_queues[queue_idx];
            let mut priority_budget = budgets[queue_idx].min(remaining_budget);

            // Sort queue by urgency for deadline-aware scheduling
            if self.config.enable_deadline_scheduling {
                let mut sorted: Vec<_> = queue.iter().copied().collect();
                sorted.sort_by(|&a, &b| {
                    let req_a = self.requests.get(&a);
                    let req_b = self.requests.get(&b);
                    match (req_a, req_b) {
                        (Some(ra), Some(rb)) => rb
                            .urgency_score()
                            .partial_cmp(&ra.urgency_score())
                            .unwrap_or(std::cmp::Ordering::Equal),
                        _ => std::cmp::Ordering::Equal,
                    }
                });
                *queue = sorted.into_iter().collect();
            }

            // Schedule requests from this priority level
            let mut scheduled_from_queue = Vec::new();
            for &request_id in queue.iter() {
                if remaining_slots == 0 || priority_budget < self.config.min_tokens_per_request {
                    break;
                }

                if let Some(request) = self.requests.get(&request_id) {
                    // Calculate tokens to allocate
                    let tokens_needed = request.remaining_tokens().max(1);
                    let tokens_to_allocate = tokens_needed
                        .min(priority_budget)
                        .max(self.config.min_tokens_per_request);

                    if tokens_to_allocate > 0 {
                        scheduled.push((request_id, tokens_to_allocate));
                        scheduled_from_queue.push(request_id);
                        priority_budget = priority_budget.saturating_sub(tokens_to_allocate);
                        remaining_budget = remaining_budget.saturating_sub(tokens_to_allocate);
                        remaining_slots -= 1;

                        // Track tokens by priority
                        self.stats.tokens_by_priority[queue_idx] += tokens_to_allocate as u64;
                    }
                }
            }

            // Remove scheduled requests from queue and update state
            for request_id in scheduled_from_queue {
                queue.retain(|&id| id != request_id);
                if let Some(request) = self.requests.get_mut(&request_id) {
                    request.state = SequenceState::Running;
                    self.running.push(request_id);

                    // Record TTFT if first time running
                    if request.ttft_ms.is_none() {
                        let ttft = request.wait_time_ms() as f64;
                        request.ttft_ms = Some(ttft);
                        self.ttft_samples.push(ttft);
                    }
                }
            }
        }

        self.update_queue_depths();
        scheduled
    }

    /// Complete a request and update statistics
    pub fn complete_request(&mut self, request_id: u64) -> Option<DynamicRequest> {
        // Remove from running
        self.running.retain(|&id| id != request_id);

        if let Some(mut request) = self.requests.remove(&request_id) {
            request.state = SequenceState::Completed;
            self.stats.completed_requests += 1;

            // Check SLA compliance
            if let Some(deadline) = &request.deadline {
                let elapsed = request.wait_time_ms();
                if elapsed <= deadline.target_latency_ms {
                    self.stats.sla_met += 1;
                } else {
                    self.stats.sla_missed += 1;
                }
            }

            // Update average TTFT
            self.update_ttft_stats();

            Some(request)
        } else {
            None
        }
    }

    /// Update TTFT statistics
    fn update_ttft_stats(&mut self) {
        if self.ttft_samples.is_empty() {
            return;
        }

        // Average
        let sum: f64 = self.ttft_samples.iter().sum();
        self.stats.avg_ttft_ms = sum / self.ttft_samples.len() as f64;

        // P99
        let mut sorted = self.ttft_samples.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let p99_idx = ((sorted.len() as f64) * 0.99) as usize;
        self.stats.p99_ttft_ms = sorted
            .get(p99_idx.min(sorted.len() - 1))
            .copied()
            .unwrap_or(0.0);
    }

    /// Update queue depth statistics
    fn update_queue_depths(&mut self) {
        for (i, queue) in self.priority_queues.iter().enumerate() {
            self.stats.queue_depth_by_priority[i] = queue.len();
        }
    }

    /// Get a request by ID
    pub fn get_request(&self, request_id: u64) -> Option<&DynamicRequest> {
        self.requests.get(&request_id)
    }

    /// Get statistics
    pub fn stats(&self) -> &DynamicSchedulerStats {
        &self.stats
    }

    /// Get configuration
    pub fn config(&self) -> &DynamicPriorityConfig {
        &self.config
    }

    /// Total waiting requests
    pub fn waiting_count(&self) -> usize {
        self.priority_queues.iter().map(VecDeque::len).sum()
    }

    /// Running requests
    pub fn running_count(&self) -> usize {
        self.running.len()
    }

    /// SLA compliance rate (0.0 to 1.0)
    pub fn sla_compliance_rate(&self) -> f64 {
        let total = self.stats.sla_met + self.stats.sla_missed;
        if total == 0 {
            1.0
        } else {
            self.stats.sla_met as f64 / total as f64
        }
    }

    /// Get queue depth for a priority level
    pub fn queue_depth(&self, priority: Priority) -> usize {
        self.priority_queues[priority as usize].len()
    }
}

// ============================================================================
// Tests
// ============================================================================

// Tests extracted to tests.rs (PMAT-802)
#[cfg(test)]
#[path = "tests.rs"]
mod scheduler_tests;

// Additional tests for coverage (Part 02)
#[cfg(test)]
#[path = "tests_dynamic_request.rs"]
mod scheduler_tests_part_02;

// Deep coverage tests (Part 03)
#[cfg(test)]
#[path = "tests_sequence_batch.rs"]
mod scheduler_tests_part_03;
