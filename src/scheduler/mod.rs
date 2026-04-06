//! Task scheduling and work distribution.
//!
//! Implements work-stealing scheduler based on Blumofe & Leiserson (1999).
#![allow(clippy::cast_precision_loss, clippy::significant_drop_tightening)]

use crate::error::{RepartirError, Result};
use crate::task::{ExecutionResult, Task, TaskId};
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info};
use uuid::Uuid;

/// Unique identifier for a worker.
///
/// Per Iron Lotus Framework case study (Section 12.3),
/// we use UUIDs instead of indices to prevent invalidation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WorkerId(Uuid);

impl WorkerId {
    /// Creates a new random worker ID.
    #[must_use]
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for WorkerId {
    fn default() -> Self {
        Self::new()
    }
}

/// Locality metrics for tracking data-aware scheduling efficiency (v2.0).
#[derive(Debug, Clone, Default)]
pub struct LocalityMetrics {
    /// Total number of tasks submitted
    pub total_tasks: usize,
    /// Number of tasks with data dependencies
    pub tasks_with_dependencies: usize,
    /// Number of tasks that had locality matches
    pub tasks_with_locality: usize,
}

impl LocalityMetrics {
    /// Calculates the locality hit rate (0.0 to 1.0).
    ///
    /// Returns the ratio of tasks with locality matches to total tasks with dependencies.
    #[must_use]
    pub fn hit_rate(&self) -> f64 {
        if self.tasks_with_dependencies == 0 {
            0.0
        } else {
            self.tasks_with_locality as f64 / self.tasks_with_dependencies as f64
        }
    }
}

/// Task wrapper for priority queue.
///
/// Implements `Ord` to enable priority-based scheduling.
#[derive(Debug)]
struct PriorityTask {
    task: Task,
}

impl PartialEq for PriorityTask {
    fn eq(&self, other: &Self) -> bool {
        self.task.priority() == other.task.priority()
    }
}

impl Eq for PriorityTask {}

impl PartialOrd for PriorityTask {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PriorityTask {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // BinaryHeap is a max-heap, so reverse order for min-heap behavior
        self.task.priority().cmp(&other.task.priority())
    }
}

/// Simple scheduler for task distribution.
///
/// This is a v1.0 implementation. Future versions will implement
/// true work-stealing with per-worker deques.
///
/// v2.0 adds data-locality tracking for affinity-based scheduling.
pub struct Scheduler {
    /// Priority queue of pending tasks.
    queue: Arc<RwLock<BinaryHeap<PriorityTask>>>,
    /// Maximum queue capacity.
    max_queue_size: usize,
    /// Task results indexed by task ID.
    results: Arc<RwLock<HashMap<TaskId, ExecutionResult>>>,
    /// Data location tracker for locality-aware scheduling (v2.0).
    data_tracker: DataLocationTracker,
    /// Locality metrics (v2.0): `tasks_with_locality` / `total_tasks`.
    locality_metrics: Arc<RwLock<LocalityMetrics>>,
}

impl Scheduler {
    /// Creates a new scheduler with default capacity.
    #[must_use]
    pub fn new() -> Self {
        Self::with_capacity(10_000)
    }

    /// Creates a scheduler with specified queue capacity.
    #[must_use]
    pub fn with_capacity(max_queue_size: usize) -> Self {
        info!("Scheduler initialized with capacity {max_queue_size}");
        Self {
            queue: Arc::new(RwLock::new(BinaryHeap::new())),
            max_queue_size,
            results: Arc::new(RwLock::new(HashMap::new())),
            data_tracker: DataLocationTracker::new(),
            locality_metrics: Arc::new(RwLock::new(LocalityMetrics::default())),
        }
    }

    /// Returns a reference to the data location tracker (v2.0).
    #[must_use]
    pub const fn data_tracker(&self) -> &DataLocationTracker {
        &self.data_tracker
    }

    /// Returns current locality metrics (v2.0).
    pub async fn locality_metrics(&self) -> LocalityMetrics {
        self.locality_metrics.read().await.clone()
    }

    /// Submits a task to the scheduler.
    ///
    /// For tasks with data dependencies, considers data locality for scheduling.
    ///
    /// # Errors
    ///
    /// Returns an error if the queue is full.
    pub async fn submit(&self, task: Task) -> Result<TaskId> {
        let task_id = task.id();
        let task_priority = task.priority();
        let has_dependencies = !task.data_dependencies().is_empty();

        // Update locality metrics
        {
            let mut metrics = self.locality_metrics.write().await;
            metrics.total_tasks += 1;
            if has_dependencies {
                metrics.tasks_with_dependencies += 1;

                // Check if any worker has the required data
                let affinity = self.calculate_affinity(task.data_dependencies()).await;
                if !affinity.is_empty() {
                    metrics.tasks_with_locality += 1;
                    debug!("Task {task_id} has locality: {} workers with data", affinity.len());
                }
            }
        }

        {
            let mut queue = self.queue.write().await;

            if queue.len() >= self.max_queue_size {
                return Err(RepartirError::QueueFull { capacity: self.max_queue_size });
            }

            debug!("Scheduling task {task_id} with priority {task_priority:?}");
            queue.push(PriorityTask { task });
        } // Drop queue lock early

        Ok(task_id)
    }

    /// Calculates affinity scores for workers based on data dependencies.
    ///
    /// Returns a map of `worker_id` -> affinity score (0.0 to 1.0).
    /// Score = (data items present on worker) / (total data items requested)
    ///
    /// # Arguments
    ///
    /// * `data_keys` - Data items required by the task
    async fn calculate_affinity(&self, data_keys: &[String]) -> HashMap<WorkerId, f64> {
        if data_keys.is_empty() {
            return HashMap::new();
        }

        let counts = self.data_tracker.locate_data_batch(data_keys).await;
        let total = data_keys.len() as f64;

        counts.into_iter().map(|(worker, count)| (worker, count as f64 / total)).collect()
    }

    /// Submits a task with explicit worker affinity preferences (v2.0).
    ///
    /// This is a lower-level API that allows callers to specify custom affinity scores.
    /// For most use cases, use `submit()` which automatically calculates affinity
    /// based on task data dependencies.
    ///
    /// # Arguments
    ///
    /// * `task` - Task to schedule
    /// * `affinity` - Map of `worker_id` -> preference score (higher = better)
    ///
    /// # Returns
    ///
    /// Returns the task ID and the preferred worker (if any)
    ///
    /// # Errors
    ///
    /// Returns an error if the queue is full.
    pub async fn submit_with_affinity(
        &self,
        task: Task,
        affinity: HashMap<WorkerId, f64>,
    ) -> Result<(TaskId, Option<WorkerId>)> {
        let task_id = self.submit(task).await?;

        // Find worker with highest affinity
        let preferred_worker = affinity
            .into_iter()
            .max_by(|(_, score_a), (_, score_b)| {
                score_a.partial_cmp(score_b).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(worker, _)| worker);

        if let Some(worker) = preferred_worker {
            debug!("Task {task_id} prefers worker {:?}", worker);
        }

        Ok((task_id, preferred_worker))
    }

    /// Submit a task with data locality hints (v2.0).
    ///
    /// This is a convenience method that calculates affinity from data keys.
    /// For more control, use `submit_with_affinity` with pre-calculated scores.
    ///
    /// # Arguments
    ///
    /// * `task` - The task to submit
    /// * `data_keys` - Data dependencies that the task requires
    ///
    /// # Returns
    ///
    /// Returns the task ID and the preferred worker (if any) based on data locality.
    ///
    /// # Errors
    ///
    /// Returns an error if the queue is full.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use repartir::scheduler::{Scheduler, WorkerId};
    /// # use repartir::task::{Task, Backend};
    /// # #[tokio::main]
    /// # async fn main() -> repartir::error::Result<()> {
    /// let scheduler = Scheduler::with_capacity(100);
    ///
    /// // Track data location
    /// let worker_id = WorkerId::new();
    /// scheduler.data_tracker()
    ///     .track_data("dataset_A", worker_id).await;
    ///
    /// // Submit task with data keys (affinity calculated automatically)
    /// let task = Task::builder()
    ///     .binary("/usr/bin/process")
    ///     .backend(Backend::Cpu)
    ///     .build()?;
    ///
    /// let (task_id, preferred) = scheduler
    ///     .submit_with_data_locality(task, &["dataset_A".to_string()])
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn submit_with_data_locality(
        &self,
        task: Task,
        data_keys: &[String],
    ) -> Result<(TaskId, Option<WorkerId>)> {
        let affinity = self.calculate_affinity(data_keys).await;
        self.submit_with_affinity(task, affinity).await
    }

    /// Retrieves the next task from the queue.
    ///
    /// Returns `None` if the queue is empty.
    pub async fn next_task(&self) -> Option<Task> {
        let mut queue = self.queue.write().await;
        queue.pop().map(|pt| pt.task)
    }

    /// Returns the number of pending tasks.
    pub async fn pending_count(&self) -> usize {
        self.queue.read().await.len()
    }

    /// Stores a task result.
    pub async fn store_result(&self, result: ExecutionResult) {
        let task_id = result.task_id();
        debug!("Storing result for task {task_id}");
        self.results.write().await.insert(task_id, result);
    }

    /// Retrieves a task result.
    pub async fn get_result(&self, task_id: TaskId) -> Option<ExecutionResult> {
        self.results.read().await.get(&task_id).cloned()
    }

    /// Removes a task result from storage.
    pub async fn remove_result(&self, task_id: TaskId) -> Option<ExecutionResult> {
        self.results.write().await.remove(&task_id)
    }

    /// Clears all pending tasks, results, data locations, and metrics.
    pub async fn clear(&self) {
        self.queue.write().await.clear();
        self.results.write().await.clear();
        self.data_tracker.clear().await;
        *self.locality_metrics.write().await = LocalityMetrics::default();
        info!("Scheduler cleared");
    }
}

impl Default for Scheduler {
    fn default() -> Self {
        Self::new()
    }
}

/// Data-locality tracker for distributed scheduling (v2.0).
///
/// Tracks which data items (identified by string keys) are present on which workers,
/// enabling locality-aware task scheduling.
///
/// # Example
///
/// ```
/// use repartir::scheduler::{DataLocationTracker, WorkerId};
///
/// # tokio_test::block_on(async {
/// let tracker = DataLocationTracker::new();
/// let worker = WorkerId::new();
///
/// // Track that worker has tensor_batch_42
/// tracker.track_data("tensor_batch_42", worker).await;
///
/// // Query which workers have this data
/// let locations = tracker.locate_data("tensor_batch_42").await;
/// assert_eq!(locations.len(), 1);
/// # });
/// ```
pub struct DataLocationTracker {
    /// Maps data keys to the set of workers that have them.
    /// `data_key` -> `HashSet`<WorkerId>
    locations: Arc<RwLock<HashMap<String, HashSet<WorkerId>>>>,
}

impl DataLocationTracker {
    /// Creates a new data location tracker.
    #[must_use]
    pub fn new() -> Self {
        Self { locations: Arc::new(RwLock::new(HashMap::new())) }
    }

    /// Records that a worker has a specific data item.
    ///
    /// # Arguments
    ///
    /// * `data_key` - Unique identifier for the data item
    /// * `worker_id` - Worker that has the data
    pub async fn track_data(&self, data_key: impl Into<String>, worker_id: WorkerId) {
        let key = data_key.into();
        let mut locations = self.locations.write().await;

        locations.entry(key.clone()).or_insert_with(HashSet::new).insert(worker_id);

        debug!("Tracked data '{}' on worker {:?}", key, worker_id);
    }

    /// Finds all workers that have a specific data item.
    ///
    /// # Arguments
    ///
    /// * `data_key` - Unique identifier for the data item
    ///
    /// # Returns
    ///
    /// Vector of worker IDs that have the data (empty if no workers have it)
    pub async fn locate_data(&self, data_key: &str) -> Vec<WorkerId> {
        let locations = self.locations.read().await;

        locations.get(data_key).map(|workers| workers.iter().copied().collect()).unwrap_or_default()
    }

    /// Finds all workers that have any of the specified data items.
    ///
    /// Returns a map of `worker_id` -> count of data items present on that worker.
    /// Useful for calculating affinity scores.
    ///
    /// # Arguments
    ///
    /// * `data_keys` - Slice of data item identifiers
    ///
    /// # Returns
    ///
    /// `HashMap` mapping each worker to the number of requested data items it has
    pub async fn locate_data_batch(&self, data_keys: &[String]) -> HashMap<WorkerId, usize> {
        let locations = self.locations.read().await;
        let mut worker_counts: HashMap<WorkerId, usize> = HashMap::new();

        for key in data_keys {
            if let Some(workers) = locations.get(key) {
                for worker_id in workers {
                    *worker_counts.entry(*worker_id).or_insert(0) += 1;
                }
            }
        }

        worker_counts
    }

    /// Removes a data item from tracking (e.g., when data is deleted).
    ///
    /// # Arguments
    ///
    /// * `data_key` - Unique identifier for the data item
    ///
    /// # Returns
    ///
    /// True if the data was tracked and removed, false otherwise
    pub async fn remove_data(&self, data_key: &str) -> bool {
        let mut locations = self.locations.write().await;
        locations.remove(data_key).is_some()
    }

    /// Removes a worker from all data location records (e.g., when worker disconnects).
    ///
    /// # Arguments
    ///
    /// * `worker_id` - Worker to remove from all locations
    ///
    /// # Returns
    ///
    /// Number of data items that were updated
    pub async fn remove_worker(&self, worker_id: WorkerId) -> usize {
        let mut locations = self.locations.write().await;
        let mut removed_count = 0;

        locations.retain(|_, workers| {
            if workers.remove(&worker_id) {
                removed_count += 1;
            }
            !workers.is_empty() // Remove entries with no workers
        });

        if removed_count > 0 {
            info!("Removed worker {:?} from {} data items", worker_id, removed_count);
        }

        removed_count
    }

    /// Returns the total number of tracked data items.
    pub async fn data_count(&self) -> usize {
        self.locations.read().await.len()
    }

    /// Clears all data location records.
    pub async fn clear(&self) {
        self.locations.write().await.clear();
        info!("Data location tracker cleared");
    }
}

impl Default for DataLocationTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::disallowed_methods, clippy::float_cmp)]
mod tests {
    use super::*;
    use crate::task::{Backend, Priority};
    use std::time::Duration;

    #[tokio::test]
    async fn test_scheduler_submit_and_next() {
        let scheduler = Scheduler::new();

        let task =
            Task::builder().binary("/bin/echo").arg("test").backend(Backend::Cpu).build().unwrap();

        let task_id = scheduler.submit(task).await.unwrap();
        assert_eq!(scheduler.pending_count().await, 1);

        let next = scheduler.next_task().await;
        assert!(next.is_some());
        assert_eq!(next.unwrap().id(), task_id);
        assert_eq!(scheduler.pending_count().await, 0);
    }

    #[tokio::test]
    async fn test_scheduler_priority_ordering() {
        let scheduler = Scheduler::new();

        // Submit tasks in reverse priority order
        let low = Task::builder()
            .binary("/bin/echo")
            .arg("low")
            .backend(Backend::Cpu)
            .priority(Priority::Low)
            .build()
            .unwrap();

        let high = Task::builder()
            .binary("/bin/echo")
            .arg("high")
            .backend(Backend::Cpu)
            .priority(Priority::High)
            .build()
            .unwrap();

        let normal = Task::builder()
            .binary("/bin/echo")
            .arg("normal")
            .backend(Backend::Cpu)
            .priority(Priority::Normal)
            .build()
            .unwrap();

        scheduler.submit(low).await.unwrap();
        scheduler.submit(high).await.unwrap();
        scheduler.submit(normal).await.unwrap();

        // Should return in priority order: High, Normal, Low
        let first = scheduler.next_task().await.unwrap();
        assert_eq!(first.priority(), Priority::High);

        let second = scheduler.next_task().await.unwrap();
        assert_eq!(second.priority(), Priority::Normal);

        let third = scheduler.next_task().await.unwrap();
        assert_eq!(third.priority(), Priority::Low);
    }

    #[tokio::test]
    async fn test_scheduler_queue_full() {
        let scheduler = Scheduler::with_capacity(2);

        let task1 =
            Task::builder().binary("/bin/echo").arg("1").backend(Backend::Cpu).build().unwrap();

        let task2 =
            Task::builder().binary("/bin/echo").arg("2").backend(Backend::Cpu).build().unwrap();

        let task3 =
            Task::builder().binary("/bin/echo").arg("3").backend(Backend::Cpu).build().unwrap();

        assert!(scheduler.submit(task1).await.is_ok());
        assert!(scheduler.submit(task2).await.is_ok());

        // Third task should fail (queue full)
        let result = scheduler.submit(task3).await;
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), RepartirError::QueueFull { .. }));
    }

    #[tokio::test]
    async fn test_scheduler_result_storage() {
        let scheduler = Scheduler::new();

        let task_id = TaskId::new();
        let result = ExecutionResult::new(
            task_id,
            0,
            b"output".to_vec(),
            b"".to_vec(),
            Duration::from_secs(1),
        );

        scheduler.store_result(result.clone()).await;

        let retrieved = scheduler.get_result(task_id).await;
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().task_id(), task_id);

        let removed = scheduler.remove_result(task_id).await;
        assert!(removed.is_some());

        assert!(scheduler.get_result(task_id).await.is_none());
    }

    #[tokio::test]
    async fn test_scheduler_clear() {
        let scheduler = Scheduler::new();

        let task =
            Task::builder().binary("/bin/echo").arg("test").backend(Backend::Cpu).build().unwrap();

        scheduler.submit(task).await.unwrap();
        assert_eq!(scheduler.pending_count().await, 1);

        scheduler.clear().await;
        assert_eq!(scheduler.pending_count().await, 0);
    }

    #[test]
    fn test_worker_id_creation() {
        let id1 = WorkerId::new();
        let id2 = WorkerId::new();
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_worker_id_default() {
        let id = WorkerId::default();
        let id2 = WorkerId::new();
        assert_ne!(id, id2);
    }

    #[test]
    fn test_scheduler_default() {
        let scheduler = Scheduler::default();
        let scheduler2 = Scheduler::new();
        // Both should have same behavior
        assert_eq!(scheduler.max_queue_size, scheduler2.max_queue_size);
    }

    #[tokio::test]
    async fn test_scheduler_empty_queue() {
        let scheduler = Scheduler::new();
        let task = scheduler.next_task().await;
        assert!(task.is_none());
    }

    #[tokio::test]
    async fn test_scheduler_nonexistent_result() {
        let scheduler = Scheduler::new();
        let fake_id = TaskId::new();

        let result = scheduler.get_result(fake_id).await;
        assert!(result.is_none());

        let removed = scheduler.remove_result(fake_id).await;
        assert!(removed.is_none());
    }

    #[tokio::test]
    async fn test_scheduler_clear_with_results() {
        let scheduler = Scheduler::new();

        let task_id = TaskId::new();
        let result = ExecutionResult::new(
            task_id,
            0,
            b"output".to_vec(),
            b"".to_vec(),
            Duration::from_secs(1),
        );

        scheduler.store_result(result).await;
        scheduler.clear().await;

        assert!(scheduler.get_result(task_id).await.is_none());
    }

    // DataLocationTracker tests (v2.0)

    #[tokio::test]
    async fn test_data_location_tracker_basic() {
        let tracker = DataLocationTracker::new();
        let worker1 = WorkerId::new();
        let worker2 = WorkerId::new();

        // Track data on workers
        tracker.track_data("tensor_batch_42", worker1).await;
        tracker.track_data("tensor_batch_42", worker2).await;
        tracker.track_data("checkpoint_123", worker1).await;

        // Locate data
        let locations = tracker.locate_data("tensor_batch_42").await;
        assert_eq!(locations.len(), 2);
        assert!(locations.contains(&worker1));
        assert!(locations.contains(&worker2));

        let checkpoint_locations = tracker.locate_data("checkpoint_123").await;
        assert_eq!(checkpoint_locations.len(), 1);
        assert!(checkpoint_locations.contains(&worker1));

        // Non-existent data
        let empty = tracker.locate_data("nonexistent").await;
        assert_eq!(empty.len(), 0);
    }

    #[tokio::test]
    async fn test_data_location_tracker_batch() {
        let tracker = DataLocationTracker::new();
        let worker1 = WorkerId::new();
        let worker2 = WorkerId::new();
        let worker3 = WorkerId::new();

        // Worker 1 has: batch_1, batch_2, batch_3
        tracker.track_data("batch_1", worker1).await;
        tracker.track_data("batch_2", worker1).await;
        tracker.track_data("batch_3", worker1).await;

        // Worker 2 has: batch_1, batch_2
        tracker.track_data("batch_1", worker2).await;
        tracker.track_data("batch_2", worker2).await;

        // Worker 3 has: batch_1
        tracker.track_data("batch_1", worker3).await;

        // Batch query
        let data_keys = vec!["batch_1".to_string(), "batch_2".to_string(), "batch_3".to_string()];

        let counts = tracker.locate_data_batch(&data_keys).await;

        assert_eq!(counts.get(&worker1), Some(&3)); // Has all 3
        assert_eq!(counts.get(&worker2), Some(&2)); // Has 2
        assert_eq!(counts.get(&worker3), Some(&1)); // Has 1
    }

    #[tokio::test]
    async fn test_data_location_tracker_remove_data() {
        let tracker = DataLocationTracker::new();
        let worker = WorkerId::new();

        tracker.track_data("temp_data", worker).await;
        assert_eq!(tracker.data_count().await, 1);

        let removed = tracker.remove_data("temp_data").await;
        assert!(removed);
        assert_eq!(tracker.data_count().await, 0);

        // Removing again should return false
        let removed_again = tracker.remove_data("temp_data").await;
        assert!(!removed_again);
    }

    #[tokio::test]
    async fn test_data_location_tracker_remove_worker() {
        let tracker = DataLocationTracker::new();
        let worker1 = WorkerId::new();
        let worker2 = WorkerId::new();

        tracker.track_data("data_1", worker1).await;
        tracker.track_data("data_2", worker1).await;
        tracker.track_data("data_3", worker1).await;
        tracker.track_data("data_1", worker2).await;

        assert_eq!(tracker.data_count().await, 3);

        // Remove worker1
        let removed_count = tracker.remove_worker(worker1).await;
        assert_eq!(removed_count, 3);

        // data_1 should still exist (worker2 has it)
        // data_2 and data_3 should be gone (only worker1 had them)
        assert_eq!(tracker.data_count().await, 1);

        let data_1_locations = tracker.locate_data("data_1").await;
        assert_eq!(data_1_locations.len(), 1);
        assert!(data_1_locations.contains(&worker2));

        let data_2_locations = tracker.locate_data("data_2").await;
        assert_eq!(data_2_locations.len(), 0);
    }

    #[tokio::test]
    async fn test_data_location_tracker_clear() {
        let tracker = DataLocationTracker::new();
        let worker = WorkerId::new();

        tracker.track_data("data_1", worker).await;
        tracker.track_data("data_2", worker).await;
        assert_eq!(tracker.data_count().await, 2);

        tracker.clear().await;
        assert_eq!(tracker.data_count().await, 0);
    }

    #[tokio::test]
    async fn test_data_location_tracker_default() {
        let tracker = DataLocationTracker::default();
        assert_eq!(tracker.data_count().await, 0);
    }

    #[tokio::test]
    async fn test_data_location_tracker_duplicate_tracking() {
        let tracker = DataLocationTracker::new();
        let worker = WorkerId::new();

        // Track same data multiple times on same worker
        tracker.track_data("data", worker).await;
        tracker.track_data("data", worker).await;
        tracker.track_data("data", worker).await;

        // Should only appear once
        let locations = tracker.locate_data("data").await;
        assert_eq!(locations.len(), 1);
        assert!(locations.contains(&worker));
    }

    #[tokio::test]
    async fn test_data_location_tracker_batch_empty() {
        let tracker = DataLocationTracker::new();
        let worker = WorkerId::new();

        tracker.track_data("exists", worker).await;

        // Query for data that doesn't exist
        let data_keys = vec!["nonexistent1".to_string(), "nonexistent2".to_string()];
        let counts = tracker.locate_data_batch(&data_keys).await;

        assert_eq!(counts.len(), 0);
    }

    #[tokio::test]
    async fn test_data_location_tracker_batch_partial() {
        let tracker = DataLocationTracker::new();
        let worker = WorkerId::new();

        tracker.track_data("exists", worker).await;

        // Query mix of existing and non-existing
        let data_keys = vec!["exists".to_string(), "nonexistent".to_string()];
        let counts = tracker.locate_data_batch(&data_keys).await;

        assert_eq!(counts.get(&worker), Some(&1)); // Only counts "exists"
    }

    // Locality-aware scheduling tests (v2.0 Phase 2)

    #[tokio::test]
    async fn test_scheduler_locality_metrics() {
        let scheduler = Scheduler::new();

        // Task without dependencies
        let task1 =
            Task::builder().binary("/bin/echo").arg("test1").backend(Backend::Cpu).build().unwrap();

        scheduler.submit(task1).await.unwrap();

        let metrics = scheduler.locality_metrics().await;
        assert_eq!(metrics.total_tasks, 1);
        assert_eq!(metrics.tasks_with_dependencies, 0);
        assert_eq!(metrics.tasks_with_locality, 0);
        assert_eq!(metrics.hit_rate(), 0.0);

        // Task with dependencies (no workers have data yet)
        let task2 = Task::builder()
            .binary("/bin/echo")
            .arg("test2")
            .backend(Backend::Cpu)
            .data_dependency("data1")
            .build()
            .unwrap();

        scheduler.submit(task2).await.unwrap();

        let metrics = scheduler.locality_metrics().await;
        assert_eq!(metrics.total_tasks, 2);
        assert_eq!(metrics.tasks_with_dependencies, 1);
        assert_eq!(metrics.tasks_with_locality, 0);
        assert_eq!(metrics.hit_rate(), 0.0);

        // Track data on a worker
        let worker = WorkerId::new();
        scheduler.data_tracker().track_data("data1", worker).await;

        // Task with dependencies (worker has data)
        let task3 = Task::builder()
            .binary("/bin/echo")
            .arg("test3")
            .backend(Backend::Cpu)
            .data_dependency("data1")
            .build()
            .unwrap();

        scheduler.submit(task3).await.unwrap();

        let metrics = scheduler.locality_metrics().await;
        assert_eq!(metrics.total_tasks, 3);
        assert_eq!(metrics.tasks_with_dependencies, 2);
        assert_eq!(metrics.tasks_with_locality, 1);
        assert_eq!(metrics.hit_rate(), 0.5); // 1 out of 2
    }

    #[tokio::test]
    async fn test_scheduler_submit_with_affinity() {
        let scheduler = Scheduler::new();
        let worker1 = WorkerId::new();
        let worker2 = WorkerId::new();

        let mut affinity = HashMap::new();
        affinity.insert(worker1, 0.3);
        affinity.insert(worker2, 0.8);

        let task =
            Task::builder().binary("/bin/echo").arg("test").backend(Backend::Cpu).build().unwrap();

        let (task_id, preferred_worker) =
            scheduler.submit_with_affinity(task, affinity).await.unwrap();

        assert!(task_id != TaskId::default());
        assert_eq!(preferred_worker, Some(worker2)); // Higher affinity
    }

    #[tokio::test]
    async fn test_scheduler_affinity_calculation() {
        let scheduler = Scheduler::new();
        let worker1 = WorkerId::new();
        let worker2 = WorkerId::new();

        // Worker 1 has data1 and data2
        scheduler.data_tracker().track_data("data1", worker1).await;
        scheduler.data_tracker().track_data("data2", worker1).await;

        // Worker 2 has only data1
        scheduler.data_tracker().track_data("data1", worker2).await;

        // Task requires data1 and data2
        let task = Task::builder()
            .binary("/bin/echo")
            .arg("test")
            .backend(Backend::Cpu)
            .data_dependency("data1")
            .data_dependency("data2")
            .build()
            .unwrap();

        // Submit and check affinity via metrics
        scheduler.submit(task).await.unwrap();

        // Worker 1 should have affinity 1.0 (2/2)
        // Worker 2 should have affinity 0.5 (1/2)
        // This should be reflected in locality metrics
        let metrics = scheduler.locality_metrics().await;
        assert_eq!(metrics.tasks_with_locality, 1);
        assert_eq!(metrics.hit_rate(), 1.0);
    }

    #[tokio::test]
    async fn test_scheduler_clear_resets_locality() {
        let scheduler = Scheduler::new();
        let worker = WorkerId::new();

        scheduler.data_tracker().track_data("data", worker).await;

        let task = Task::builder()
            .binary("/bin/echo")
            .arg("test")
            .backend(Backend::Cpu)
            .data_dependency("data")
            .build()
            .unwrap();

        scheduler.submit(task).await.unwrap();

        let metrics_before = scheduler.locality_metrics().await;
        assert_eq!(metrics_before.total_tasks, 1);

        scheduler.clear().await;

        let metrics_after = scheduler.locality_metrics().await;
        assert_eq!(metrics_after.total_tasks, 0);
        assert_eq!(metrics_after.tasks_with_dependencies, 0);
        assert_eq!(metrics_after.tasks_with_locality, 0);

        // Data tracker should also be cleared
        assert_eq!(scheduler.data_tracker().data_count().await, 0);
    }

    #[tokio::test]
    async fn test_locality_metrics_hit_rate() {
        let mut metrics = LocalityMetrics::default();
        assert_eq!(metrics.hit_rate(), 0.0); // No dependencies

        metrics.tasks_with_dependencies = 10;
        metrics.tasks_with_locality = 7;
        assert_eq!(metrics.hit_rate(), 0.7); // 70% hit rate

        metrics.tasks_with_dependencies = 0;
        assert_eq!(metrics.hit_rate(), 0.0); // Division by zero guard
    }

    #[tokio::test]
    async fn test_scheduler_affinity_empty_dependencies() {
        let scheduler = Scheduler::new();

        let task =
            Task::builder().binary("/bin/echo").arg("test").backend(Backend::Cpu).build().unwrap();

        let affinity = HashMap::new();
        let (_, preferred_worker) = scheduler.submit_with_affinity(task, affinity).await.unwrap();

        assert_eq!(preferred_worker, None); // No affinity, no preference
    }
}
