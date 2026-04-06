//! Elastic training — dynamic worker add/remove during training.
//!
//! Extends the distributed training infrastructure with the ability to:
//! - Add new workers mid-training (scale up)
//! - Gracefully remove workers (scale down)
//! - Continue training after worker failure (fault tolerance)
//!
//! # Protocol
//!
//! ## Worker Join (mid-training):
//! 1. New worker sends JoinRequest with `epoch_reached` field
//! 2. Coordinator pauses at next step boundary
//! 3. Coordinator sends current weights to new worker
//! 4. Coordinator adjusts world_size and shard assignments
//! 5. Training resumes with new worker participating
//!
//! ## Worker Leave (graceful):
//! 1. Worker sends LeaveRequest
//! 2. Coordinator removes worker from pool
//! 3. Coordinator redistributes shards
//! 4. Training continues with remaining workers
//!
//! ## Worker Failure (ungraceful):
//! 1. Heartbeat timeout detected by coordinator
//! 2. Coordinator marks worker as failed
//! 3. Coordinator redistributes shards
//! 4. If below min_workers, pause training
//! 5. Training continues when sufficient workers available
//!
//! # Contract (C-ELASTIC-001)
//!
//! - Adding/removing workers does not change model weights
//! - Data sharding is rebalanced to maintain disjointness (C-SHARD-001)
//! - All active workers hold identical weights after rebalance

use std::time::Instant;

/// State of a worker in the elastic pool.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WorkerState {
    /// Worker is actively participating in training
    Active,
    /// Worker has been accepted but is syncing weights
    Syncing,
    /// Worker has been marked for removal (will leave at next step boundary)
    Draining,
    /// Worker has failed (heartbeat timeout)
    Failed,
    /// Worker has gracefully left
    Left,
}

/// Information about a worker in the elastic pool.
#[derive(Debug, Clone)]
pub struct ElasticWorker {
    /// Worker ID (assigned at join time, stable across reconfigurations)
    pub worker_id: u32,
    /// Node identifier
    pub node_id: String,
    /// Current state
    pub state: WorkerState,
    /// Number of GPUs on this worker
    pub gpu_count: u32,
    /// Backend type
    pub backend: String,
    /// When the worker joined
    pub joined_at: Instant,
    /// Step at which the worker joined (for data shard calculation)
    pub joined_at_step: usize,
    /// Last heartbeat time
    pub last_heartbeat: Instant,
}

/// Elastic training coordinator.
///
/// Manages a pool of workers that can dynamically grow or shrink.
/// Tracks worker state and handles reconfiguration events.
#[derive(Debug)]
pub struct ElasticCoordinator {
    /// All known workers (active, syncing, draining, failed, left)
    workers: Vec<ElasticWorker>,
    /// Next worker ID to assign
    next_worker_id: u32,
    /// Minimum workers required for training (pause if below)
    min_workers: usize,
    /// Maximum workers allowed
    max_workers: usize,
    /// Current training step
    current_step: usize,
    /// Whether a reconfiguration is pending
    reconfig_pending: bool,
    /// Heartbeat timeout (milliseconds)
    heartbeat_timeout_ms: u64,
}

impl ElasticCoordinator {
    /// Create a new elastic coordinator.
    pub fn new(min_workers: usize, max_workers: usize, heartbeat_timeout_ms: u64) -> Self {
        Self {
            workers: Vec::new(),
            next_worker_id: 0,
            min_workers,
            max_workers,
            current_step: 0,
            reconfig_pending: false,
            heartbeat_timeout_ms,
        }
    }

    /// Add a new worker to the pool.
    ///
    /// Returns the assigned worker ID, or None if pool is full.
    pub fn add_worker(&mut self, node_id: String, gpu_count: u32, backend: String) -> Option<u32> {
        if self.active_count() >= self.max_workers {
            return None;
        }

        let worker_id = self.next_worker_id;
        self.next_worker_id += 1;
        let now = Instant::now();

        self.workers.push(ElasticWorker {
            worker_id,
            node_id,
            state: WorkerState::Syncing,
            gpu_count,
            backend,
            joined_at: now,
            joined_at_step: self.current_step,
            last_heartbeat: now,
        });

        self.reconfig_pending = true;
        Some(worker_id)
    }

    /// Mark a worker as active (weight sync complete).
    pub fn activate_worker(&mut self, worker_id: u32) -> bool {
        if let Some(w) = self.workers.iter_mut().find(|w| w.worker_id == worker_id) {
            if w.state == WorkerState::Syncing {
                w.state = WorkerState::Active;
                return true;
            }
        }
        false
    }

    /// Request graceful removal of a worker.
    pub fn remove_worker(&mut self, worker_id: u32) -> bool {
        if let Some(w) = self.workers.iter_mut().find(|w| w.worker_id == worker_id) {
            if w.state == WorkerState::Active {
                w.state = WorkerState::Draining;
                self.reconfig_pending = true;
                return true;
            }
        }
        false
    }

    /// Complete removal of a draining worker.
    pub fn finalize_removal(&mut self, worker_id: u32) -> bool {
        if let Some(w) = self.workers.iter_mut().find(|w| w.worker_id == worker_id) {
            if w.state == WorkerState::Draining {
                w.state = WorkerState::Left;
                return true;
            }
        }
        false
    }

    /// Check for failed workers based on heartbeat timeout.
    ///
    /// Returns list of worker IDs that have failed.
    pub fn check_heartbeats(&mut self) -> Vec<u32> {
        let now = Instant::now();
        let timeout = std::time::Duration::from_millis(self.heartbeat_timeout_ms);
        let mut failed = Vec::new();

        for w in &mut self.workers {
            if w.state == WorkerState::Active && now.duration_since(w.last_heartbeat) > timeout {
                w.state = WorkerState::Failed;
                failed.push(w.worker_id);
                self.reconfig_pending = true;
            }
        }

        failed
    }

    /// Update heartbeat for a worker.
    pub fn update_heartbeat(&mut self, worker_id: u32) {
        if let Some(w) = self.workers.iter_mut().find(|w| w.worker_id == worker_id) {
            w.last_heartbeat = Instant::now();
        }
    }

    /// Number of active workers.
    pub fn active_count(&self) -> usize {
        self.workers.iter().filter(|w| w.state == WorkerState::Active).count()
    }

    /// Whether training should be paused (below minimum workers).
    pub fn should_pause(&self) -> bool {
        self.active_count() < self.min_workers
    }

    /// Whether a reconfiguration is needed.
    pub fn needs_reconfig(&self) -> bool {
        self.reconfig_pending
    }

    /// Clear the reconfiguration flag.
    pub fn clear_reconfig(&mut self) {
        self.reconfig_pending = false;
    }

    /// Get list of active worker IDs.
    pub fn active_worker_ids(&self) -> Vec<u32> {
        self.workers
            .iter()
            .filter(|w| w.state == WorkerState::Active)
            .map(|w| w.worker_id)
            .collect()
    }

    /// Get all workers (for status display).
    pub fn all_workers(&self) -> &[ElasticWorker] {
        &self.workers
    }

    /// Update step counter.
    pub fn set_step(&mut self, step: usize) {
        self.current_step = step;
    }

    /// Get current effective world size (active workers only).
    pub fn effective_world_size(&self) -> usize {
        self.active_count()
    }

    /// Compute shard assignments for active workers.
    ///
    /// Returns (worker_id, shard_start, shard_end) for each active worker.
    pub fn compute_shards(&self, total_samples: usize) -> Vec<(u32, usize, usize)> {
        let active: Vec<u32> = self.active_worker_ids();
        let n = active.len();
        if n == 0 {
            return Vec::new();
        }

        let shard_size = total_samples / n;
        let remainder = total_samples % n;

        active
            .iter()
            .enumerate()
            .map(|(i, &wid)| {
                let start = if i < remainder {
                    i * (shard_size + 1)
                } else {
                    remainder * (shard_size + 1) + (i - remainder) * shard_size
                };
                let end = if i < remainder { start + shard_size + 1 } else { start + shard_size };
                (wid, start, end)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_elastic_coordinator_basic() {
        let mut coord = ElasticCoordinator::new(1, 8, 30000);
        assert_eq!(coord.active_count(), 0);
        assert!(coord.should_pause());

        let id = coord.add_worker("node-1".into(), 1, "cuda".into());
        assert_eq!(id, Some(0));
        assert_eq!(coord.active_count(), 0); // still syncing

        coord.activate_worker(0);
        assert_eq!(coord.active_count(), 1);
        assert!(!coord.should_pause());
    }

    #[test]
    fn test_elastic_add_remove() {
        let mut coord = ElasticCoordinator::new(1, 4, 30000);

        // Add 3 workers
        coord.add_worker("n1".into(), 1, "cuda".into());
        coord.add_worker("n2".into(), 1, "cuda".into());
        coord.add_worker("n3".into(), 2, "wgpu".into());
        coord.activate_worker(0);
        coord.activate_worker(1);
        coord.activate_worker(2);
        assert_eq!(coord.active_count(), 3);

        // Remove one
        coord.remove_worker(1);
        assert_eq!(coord.active_count(), 2); // draining doesn't count as active
        coord.finalize_removal(1);
        assert_eq!(coord.active_count(), 2);
    }

    #[test]
    fn test_elastic_max_workers() {
        let mut coord = ElasticCoordinator::new(1, 2, 30000);
        coord.add_worker("n1".into(), 1, "cuda".into());
        coord.activate_worker(0);
        coord.add_worker("n2".into(), 1, "cuda".into());
        coord.activate_worker(1);

        // Pool full
        let id = coord.add_worker("n3".into(), 1, "cuda".into());
        assert_eq!(id, None);
    }

    #[test]
    fn test_elastic_shard_computation() {
        let mut coord = ElasticCoordinator::new(1, 4, 30000);
        for i in 0..3 {
            coord.add_worker(format!("n{i}"), 1, "cuda".into());
            coord.activate_worker(i as u32);
        }

        let shards = coord.compute_shards(100);
        assert_eq!(shards.len(), 3);

        // 100 / 3 = 33 rem 1 → first gets 34, others 33
        let (_, s0, e0) = shards[0];
        let (_, s1, e1) = shards[1];
        let (_, s2, e2) = shards[2];

        assert_eq!(s0, 0);
        assert_eq!(e0, 34);
        assert_eq!(s1, 34);
        assert_eq!(e1, 67);
        assert_eq!(s2, 67);
        assert_eq!(e2, 100);

        // Complete coverage
        assert_eq!(e0 - s0 + e1 - s1 + e2 - s2, 100);
    }

    #[test]
    fn test_elastic_shard_disjointness() {
        // C-ELASTIC-001: shards are disjoint and complete
        let mut coord = ElasticCoordinator::new(1, 8, 30000);
        for i in 0..5 {
            coord.add_worker(format!("n{i}"), 1, "cuda".into());
            coord.activate_worker(i as u32);
        }

        let total = 10007; // prime, to test remainder handling
        let shards = coord.compute_shards(total);

        let mut covered = vec![false; total];
        for (_, start, end) in &shards {
            for i in *start..*end {
                assert!(!covered[i], "sample {i} covered by multiple shards");
                covered[i] = true;
            }
        }
        assert!(covered.iter().all(|&c| c), "not all samples covered");
    }

    #[test]
    fn test_elastic_reconfig_flag() {
        let mut coord = ElasticCoordinator::new(1, 4, 30000);
        assert!(!coord.needs_reconfig());

        coord.add_worker("n1".into(), 1, "cuda".into());
        assert!(coord.needs_reconfig());

        coord.clear_reconfig();
        assert!(!coord.needs_reconfig());
    }

    #[test]
    fn test_elastic_should_pause() {
        let mut coord = ElasticCoordinator::new(2, 4, 30000);
        assert!(coord.should_pause()); // 0 < 2

        coord.add_worker("n1".into(), 1, "cuda".into());
        coord.activate_worker(0);
        assert!(coord.should_pause()); // 1 < 2

        coord.add_worker("n2".into(), 1, "cuda".into());
        coord.activate_worker(1);
        assert!(!coord.should_pause()); // 2 >= 2
    }

    #[test]
    fn test_elastic_effective_world_size() {
        let mut coord = ElasticCoordinator::new(1, 4, 30000);
        coord.add_worker("n1".into(), 1, "cuda".into());
        coord.add_worker("n2".into(), 1, "cuda".into());
        coord.activate_worker(0);
        coord.activate_worker(1);

        assert_eq!(coord.effective_world_size(), 2);

        coord.remove_worker(0);
        assert_eq!(coord.effective_world_size(), 1);
    }

    // ── Additional coverage tests ─────────────────────────────────

    #[test]
    fn test_elastic_activate_non_syncing_worker() {
        let mut coord = ElasticCoordinator::new(1, 4, 30000);
        coord.add_worker("n1".into(), 1, "cuda".into());
        coord.activate_worker(0);
        // Activating an already-active worker should return false
        assert!(!coord.activate_worker(0));
    }

    #[test]
    fn test_elastic_activate_nonexistent_worker() {
        let mut coord = ElasticCoordinator::new(1, 4, 30000);
        assert!(!coord.activate_worker(999));
    }

    #[test]
    fn test_elastic_remove_non_active_worker() {
        let mut coord = ElasticCoordinator::new(1, 4, 30000);
        coord.add_worker("n1".into(), 1, "cuda".into());
        // Worker is still Syncing, not Active; remove should fail
        assert!(!coord.remove_worker(0));
    }

    #[test]
    fn test_elastic_remove_nonexistent_worker() {
        let mut coord = ElasticCoordinator::new(1, 4, 30000);
        assert!(!coord.remove_worker(999));
    }

    #[test]
    fn test_elastic_finalize_removal_not_draining() {
        let mut coord = ElasticCoordinator::new(1, 4, 30000);
        coord.add_worker("n1".into(), 1, "cuda".into());
        coord.activate_worker(0);
        // Worker is Active, not Draining; finalize should fail
        assert!(!coord.finalize_removal(0));
    }

    #[test]
    fn test_elastic_finalize_nonexistent_worker() {
        let mut coord = ElasticCoordinator::new(1, 4, 30000);
        assert!(!coord.finalize_removal(999));
    }

    #[test]
    fn test_elastic_update_heartbeat() {
        let mut coord = ElasticCoordinator::new(1, 4, 30000);
        coord.add_worker("n1".into(), 1, "cuda".into());
        coord.activate_worker(0);
        // Should not panic; just update the timestamp
        coord.update_heartbeat(0);
        // Nonexistent worker: no-op
        coord.update_heartbeat(999);
    }

    #[test]
    fn test_elastic_check_heartbeats_no_timeout() {
        let mut coord = ElasticCoordinator::new(1, 4, 30000);
        coord.add_worker("n1".into(), 1, "cuda".into());
        coord.activate_worker(0);
        // Immediately checking should not time out (30s timeout)
        let failed = coord.check_heartbeats();
        assert!(failed.is_empty());
    }

    #[test]
    fn test_elastic_check_heartbeats_instant_timeout() {
        // Timeout of 0ms means every active worker will immediately fail
        let mut coord = ElasticCoordinator::new(1, 4, 0);
        coord.add_worker("n1".into(), 1, "cuda".into());
        coord.activate_worker(0);
        coord.add_worker("n2".into(), 1, "cuda".into());
        coord.activate_worker(1);

        // Small delay to ensure the 0ms timeout is exceeded
        std::thread::sleep(std::time::Duration::from_millis(1));

        let failed = coord.check_heartbeats();
        assert_eq!(failed.len(), 2);
        assert!(coord.needs_reconfig());
        assert_eq!(coord.active_count(), 0);
    }

    #[test]
    fn test_elastic_set_step() {
        let mut coord = ElasticCoordinator::new(1, 4, 30000);
        coord.set_step(42);
        // Step is used for shard computation joined_at_step
        coord.add_worker("n1".into(), 1, "cuda".into());
        assert_eq!(coord.all_workers()[0].joined_at_step, 42);
    }

    #[test]
    fn test_elastic_compute_shards_empty() {
        let coord = ElasticCoordinator::new(1, 4, 30000);
        let shards = coord.compute_shards(100);
        assert!(shards.is_empty());
    }

    #[test]
    fn test_elastic_compute_shards_single_worker() {
        let mut coord = ElasticCoordinator::new(1, 4, 30000);
        coord.add_worker("n1".into(), 1, "cuda".into());
        coord.activate_worker(0);
        let shards = coord.compute_shards(100);
        assert_eq!(shards.len(), 1);
        assert_eq!(shards[0], (0, 0, 100));
    }

    #[test]
    fn test_elastic_compute_shards_even_division() {
        let mut coord = ElasticCoordinator::new(1, 4, 30000);
        for i in 0..4 {
            coord.add_worker(format!("n{i}"), 1, "cuda".into());
            coord.activate_worker(i as u32);
        }
        let shards = coord.compute_shards(100);
        assert_eq!(shards.len(), 4);
        // 100 / 4 = 25 each
        for (_, start, end) in &shards {
            assert_eq!(end - start, 25);
        }
    }

    #[test]
    fn test_elastic_compute_shards_zero_samples() {
        let mut coord = ElasticCoordinator::new(1, 4, 30000);
        coord.add_worker("n1".into(), 1, "cuda".into());
        coord.activate_worker(0);
        let shards = coord.compute_shards(0);
        assert_eq!(shards.len(), 1);
        assert_eq!(shards[0], (0, 0, 0));
    }

    #[test]
    fn test_elastic_all_workers() {
        let mut coord = ElasticCoordinator::new(1, 8, 30000);
        coord.add_worker("n1".into(), 2, "cuda".into());
        coord.add_worker("n2".into(), 4, "wgpu".into());

        let all = coord.all_workers();
        assert_eq!(all.len(), 2);
        assert_eq!(all[0].node_id, "n1");
        assert_eq!(all[0].gpu_count, 2);
        assert_eq!(all[0].backend, "cuda");
        assert_eq!(all[0].state, WorkerState::Syncing);
        assert_eq!(all[1].node_id, "n2");
        assert_eq!(all[1].gpu_count, 4);
    }

    #[test]
    fn test_elastic_active_worker_ids() {
        let mut coord = ElasticCoordinator::new(1, 4, 30000);
        coord.add_worker("n1".into(), 1, "cuda".into());
        coord.add_worker("n2".into(), 1, "cuda".into());
        coord.add_worker("n3".into(), 1, "cuda".into());
        coord.activate_worker(0);
        coord.activate_worker(2);
        // Worker 1 still syncing

        let active = coord.active_worker_ids();
        assert_eq!(active, vec![0, 2]);
    }

    #[test]
    fn test_elastic_worker_state_transitions() {
        let mut coord = ElasticCoordinator::new(1, 4, 30000);
        coord.add_worker("n1".into(), 1, "cuda".into());

        // Syncing -> Active
        assert_eq!(coord.all_workers()[0].state, WorkerState::Syncing);
        coord.activate_worker(0);
        assert_eq!(coord.all_workers()[0].state, WorkerState::Active);

        // Active -> Draining
        coord.remove_worker(0);
        assert_eq!(coord.all_workers()[0].state, WorkerState::Draining);

        // Draining -> Left
        coord.finalize_removal(0);
        assert_eq!(coord.all_workers()[0].state, WorkerState::Left);
    }

    #[test]
    fn test_elastic_worker_id_increments() {
        let mut coord = ElasticCoordinator::new(1, 8, 30000);
        let id0 = coord.add_worker("n1".into(), 1, "cuda".into());
        let id1 = coord.add_worker("n2".into(), 1, "cuda".into());
        let id2 = coord.add_worker("n3".into(), 1, "cuda".into());
        assert_eq!(id0, Some(0));
        assert_eq!(id1, Some(1));
        assert_eq!(id2, Some(2));
    }

    #[test]
    fn test_elastic_clear_reconfig_then_add() {
        let mut coord = ElasticCoordinator::new(1, 4, 30000);
        coord.add_worker("n1".into(), 1, "cuda".into());
        assert!(coord.needs_reconfig());
        coord.clear_reconfig();
        assert!(!coord.needs_reconfig());

        // Adding another worker sets reconfig again
        coord.add_worker("n2".into(), 1, "cuda".into());
        assert!(coord.needs_reconfig());
    }

    #[test]
    fn test_elastic_remove_sets_reconfig() {
        let mut coord = ElasticCoordinator::new(1, 4, 30000);
        coord.add_worker("n1".into(), 1, "cuda".into());
        coord.activate_worker(0);
        coord.clear_reconfig();
        assert!(!coord.needs_reconfig());

        coord.remove_worker(0);
        assert!(coord.needs_reconfig());
    }

    #[test]
    fn test_worker_state_eq() {
        assert_eq!(WorkerState::Active, WorkerState::Active);
        assert_eq!(WorkerState::Syncing, WorkerState::Syncing);
        assert_eq!(WorkerState::Draining, WorkerState::Draining);
        assert_eq!(WorkerState::Failed, WorkerState::Failed);
        assert_eq!(WorkerState::Left, WorkerState::Left);
        assert_ne!(WorkerState::Active, WorkerState::Syncing);
        assert_ne!(WorkerState::Draining, WorkerState::Failed);
    }

    #[test]
    fn test_elastic_worker_clone() {
        let mut coord = ElasticCoordinator::new(1, 4, 30000);
        coord.add_worker("n1".into(), 2, "wgpu".into());
        let worker = coord.all_workers()[0].clone();
        assert_eq!(worker.node_id, "n1");
        assert_eq!(worker.gpu_count, 2);
        assert_eq!(worker.backend, "wgpu");
    }
}
