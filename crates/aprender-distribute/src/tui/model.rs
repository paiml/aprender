//! TUI data model - Application state and data structures.
//!
//! All data structures for the job flow TUI are defined here.
//! Tests use property-based testing and probador for 100% coverage.

use crate::task::TaskId;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::net::SocketAddr;
use std::time::{Duration, Instant};

/// Maximum number of completions to keep in history.
const MAX_COMPLETIONS: usize = 100;

/// Maximum history length for sparklines.
const HISTORY_SIZE: usize = 60;

/// Backend type for compute resources.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BackendType {
    /// CPU execution.
    Cpu,
    /// NVIDIA CUDA GPU.
    Cuda,
    /// Apple Metal GPU.
    Metal,
    /// Vulkan GPU.
    Vulkan,
    /// AMD `ROCm` GPU.
    Rocm,
}

impl std::fmt::Display for BackendType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cpu => write!(f, "CPU"),
            Self::Cuda => write!(f, "CUDA"),
            Self::Metal => write!(f, "Metal"),
            Self::Vulkan => write!(f, "Vulkan"),
            Self::Rocm => write!(f, "ROCm"),
        }
    }
}

/// Node state in the cluster.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NodeState {
    /// Node is online and healthy.
    #[default]
    Online,
    /// Node missed heartbeats, suspected offline.
    Suspected,
    /// Node confirmed offline.
    Offline,
}

/// Status of a single backend on a node.
#[derive(Debug, Clone)]
pub struct BackendStatus {
    /// Backend type.
    pub backend_type: BackendType,
    /// Device name (e.g., "RTX 4090").
    pub device_name: String,
    /// Utilization percentage (0.0 - 100.0).
    pub utilization: f64,
    /// Memory usage percentage (0.0 - 100.0).
    pub memory_pct: f64,
    /// Temperature in Celsius (optional).
    pub temperature: Option<f64>,
}

impl BackendStatus {
    /// Creates a new backend status.
    #[must_use]
    pub fn new(backend_type: BackendType, device_name: &str) -> Self {
        Self {
            backend_type,
            device_name: device_name.to_string(),
            utilization: 0.0,
            memory_pct: 0.0,
            temperature: None,
        }
    }
}

/// Status of a node in the cluster.
#[derive(Debug, Clone)]
pub struct NodeStatus {
    /// Unique node identifier.
    pub node_id: uuid::Uuid,
    /// Human-readable node name.
    pub name: String,
    /// Network endpoint.
    pub endpoint: SocketAddr,
    /// Available backends on this node.
    pub backends: Vec<BackendStatus>,
    /// CPU utilization percentage.
    pub cpu_pct: f64,
    /// Memory utilization percentage.
    pub mem_pct: f64,
    /// Number of currently running tasks.
    pub running_tasks: u32,
    /// Network latency in milliseconds.
    pub latency_ms: u32,
    /// Current node state.
    pub state: NodeState,
    /// Historical load values for sparkline.
    pub load_history: VecDeque<u64>,
}

impl NodeStatus {
    /// Creates a new node status.
    #[must_use]
    pub fn new(name: &str, endpoint: SocketAddr) -> Self {
        Self {
            node_id: uuid::Uuid::new_v4(),
            name: name.to_string(),
            endpoint,
            backends: Vec::new(),
            cpu_pct: 0.0,
            mem_pct: 0.0,
            running_tasks: 0,
            latency_ms: 0,
            state: NodeState::Online,
            load_history: VecDeque::with_capacity(HISTORY_SIZE),
        }
    }

    /// Adds a backend to this node.
    pub fn add_backend(&mut self, backend: BackendStatus) {
        self.backends.push(backend);
    }

    /// Updates the load history.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn update_history(&mut self) {
        let load = (self.cpu_pct * 100.0) as u64;
        self.load_history.push_back(load);
        if self.load_history.len() > HISTORY_SIZE {
            self.load_history.pop_front();
        }
    }
}

/// Task queue statistics.
#[derive(Debug, Clone, Default)]
pub struct TaskQueue {
    /// Number of pending tasks.
    pub pending: usize,
    /// Number of in-flight tasks.
    pub in_flight: usize,
    /// High priority task count.
    pub high_priority: usize,
    /// Normal priority task count.
    pub normal_priority: usize,
    /// Low priority task count.
    pub low_priority: usize,
}

impl TaskQueue {
    /// Returns the total number of tasks.
    #[must_use]
    pub const fn total(&self) -> usize {
        self.pending + self.in_flight
    }

    /// Returns the queue utilization percentage.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn utilization_pct(&self) -> f64 {
        let total = self.total();
        if total == 0 {
            return 0.0;
        }
        // Assume max queue of 1000 for percentage calculation
        let max_queue = 1000.0;
        (total as f64 / max_queue * 100.0).min(100.0)
    }
}

/// Record of a completed task.
#[derive(Debug, Clone)]
pub struct CompletionRecord {
    /// Task ID.
    pub task_id: TaskId,
    /// Backend used for execution.
    pub backend: BackendType,
    /// Node that executed the task.
    pub node_name: String,
    /// Execution duration.
    pub duration: Duration,
    /// Whether the task succeeded.
    pub success: bool,
    /// Error message if failed.
    pub error: Option<String>,
    /// Completion timestamp.
    pub timestamp: Instant,
}

impl CompletionRecord {
    /// Creates a new successful completion record.
    #[must_use]
    pub fn success(
        task_id: TaskId,
        backend: BackendType,
        node_name: &str,
        duration: Duration,
    ) -> Self {
        Self {
            task_id,
            backend,
            node_name: node_name.to_string(),
            duration,
            success: true,
            error: None,
            timestamp: Instant::now(),
        }
    }

    /// Creates a new failed completion record.
    #[must_use]
    pub fn failure(task_id: TaskId, backend: BackendType, node_name: &str, error: &str) -> Self {
        Self {
            task_id,
            backend,
            node_name: node_name.to_string(),
            duration: Duration::ZERO,
            success: false,
            error: Some(error.to_string()),
            timestamp: Instant::now(),
        }
    }
}

/// Alert types for the TUI.
#[derive(Debug, Clone)]
pub enum Alert {
    /// Memory pressure on a node.
    MemoryPressure {
        /// Node name.
        node: String,
        /// Memory percentage.
        pct: f64,
    },
    /// Work imbalance between nodes.
    WorkImbalance {
        /// Overloaded node.
        overloaded: String,
        /// Underloaded node.
        underloaded: String,
    },
    /// Node suspected offline.
    NodeSuspected {
        /// Node name.
        node: String,
        /// Time since last seen.
        last_seen: Duration,
    },
    /// Task exceeded timeout.
    TaskTimeout {
        /// Task ID.
        task_id: TaskId,
        /// Elapsed time.
        elapsed: Duration,
    },
    /// Backend error.
    BackendError {
        /// Node name.
        node: String,
        /// Backend type.
        backend: BackendType,
        /// Error message.
        error: String,
    },
}

impl Alert {
    /// Returns the alert message for display.
    #[must_use]
    pub fn message(&self) -> String {
        match self {
            Self::MemoryPressure { node, pct } => {
                format!("{node}: memory pressure ({pct:.0}%)")
            }
            Self::WorkImbalance { overloaded, underloaded } => {
                format!("Work imbalance: {overloaded} overloaded, {underloaded} idle")
            }
            Self::NodeSuspected { node, last_seen } => {
                format!("{node}: suspected offline (last seen {last_seen:?} ago)")
            }
            Self::TaskTimeout { task_id, elapsed } => {
                format!("Task {task_id}: timeout after {elapsed:?}")
            }
            Self::BackendError { node, backend, error } => {
                format!("{node} {backend}: {error}")
            }
        }
    }
}

/// Current selection in the TUI.
#[derive(Debug, Clone, Default)]
pub enum Selection {
    /// Nothing selected.
    #[default]
    None,
    /// Node selected by index.
    Node(usize),
    /// Task selected by index.
    Task(usize),
    /// Completion selected by index.
    Completion(usize),
}

/// Focus panel in the TUI.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Focus {
    /// Cluster panel.
    #[default]
    Cluster,
    /// Queue panel.
    Queue,
    /// Completions panel.
    Completions,
}

/// Historical metrics for sparklines.
#[derive(Debug, Clone, Default)]
pub struct MetricsHistory {
    /// CPU history per node.
    pub cpu_history: Vec<VecDeque<u64>>,
    /// GPU history per node.
    pub gpu_history: Vec<VecDeque<u64>>,
}

impl MetricsHistory {
    /// Creates a new empty history.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Updates history with current node data.
    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn push(&mut self, nodes: &[NodeStatus]) {
        // Ensure we have enough history vectors
        while self.cpu_history.len() < nodes.len() {
            self.cpu_history.push(VecDeque::with_capacity(HISTORY_SIZE));
        }
        while self.gpu_history.len() < nodes.len() {
            self.gpu_history.push(VecDeque::with_capacity(HISTORY_SIZE));
        }

        for (i, node) in nodes.iter().enumerate() {
            // CPU history
            let cpu_val = (node.cpu_pct * 100.0) as u64;
            self.cpu_history[i].push_back(cpu_val);
            if self.cpu_history[i].len() > HISTORY_SIZE {
                self.cpu_history[i].pop_front();
            }

            // GPU history (average of all GPUs)
            let gpu_avg = if node.backends.is_empty() {
                0.0
            } else {
                node.backends.iter().map(|b| b.utilization).sum::<f64>()
                    / node.backends.len() as f64
            };
            let gpu_val = (gpu_avg * 100.0) as u64;
            self.gpu_history[i].push_back(gpu_val);
            if self.gpu_history[i].len() > HISTORY_SIZE {
                self.gpu_history[i].pop_front();
            }
        }
    }
}

/// Main application state.
#[derive(Debug)]
pub struct App {
    /// Connected nodes.
    pub nodes: Vec<NodeStatus>,
    /// Task queue statistics.
    pub queue: TaskQueue,
    /// Recent completions (ring buffer).
    pub completions: VecDeque<CompletionRecord>,
    /// Current selection.
    pub selected: Selection,
    /// Current focus panel.
    pub focus: Focus,
    /// Active alerts.
    pub alerts: Vec<Alert>,
    /// Tick rate for updates.
    pub tick_rate: Duration,
    /// Historical metrics.
    pub history: MetricsHistory,
    /// Force refresh flag.
    pub force_refresh: bool,
    /// Show help overlay.
    pub show_help: bool,
    /// Show alerts panel.
    pub show_alerts: bool,
    /// Tick counter.
    pub tick_count: u64,
}

impl Default for App {
    fn default() -> Self {
        Self::new()
    }
}

impl App {
    /// Creates a new application with empty state.
    #[must_use]
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            queue: TaskQueue::default(),
            completions: VecDeque::with_capacity(MAX_COMPLETIONS),
            selected: Selection::None,
            focus: Focus::Cluster,
            alerts: Vec::new(),
            tick_rate: Duration::from_millis(100),
            history: MetricsHistory::new(),
            force_refresh: false,
            show_help: false,
            show_alerts: true,
            tick_count: 0,
        }
    }

    /// Adds a node to the cluster.
    pub fn add_node(&mut self, node: NodeStatus) {
        self.nodes.push(node);
    }

    /// Adds a completion record.
    pub fn add_completion(&mut self, record: CompletionRecord) {
        self.completions.push_front(record);
        if self.completions.len() > MAX_COMPLETIONS {
            self.completions.pop_back();
        }
    }

    /// Selects the previous item.
    pub fn select_prev(&mut self) {
        match (&self.focus, &self.selected) {
            (Focus::Cluster, Selection::Node(i)) if *i > 0 => {
                self.selected = Selection::Node(i - 1);
            }
            (Focus::Cluster, Selection::None) if !self.nodes.is_empty() => {
                self.selected = Selection::Node(self.nodes.len() - 1);
            }
            (Focus::Completions, Selection::Completion(i)) if *i > 0 => {
                self.selected = Selection::Completion(i - 1);
            }
            (Focus::Completions, Selection::None) if !self.completions.is_empty() => {
                self.selected = Selection::Completion(self.completions.len() - 1);
            }
            _ => {}
        }
    }

    /// Selects the next item.
    pub fn select_next(&mut self) {
        match (&self.focus, &self.selected) {
            (Focus::Cluster, Selection::Node(i)) if *i + 1 < self.nodes.len() => {
                self.selected = Selection::Node(i + 1);
            }
            (Focus::Cluster, Selection::None) if !self.nodes.is_empty() => {
                self.selected = Selection::Node(0);
            }
            (Focus::Completions, Selection::Completion(i)) if *i + 1 < self.completions.len() => {
                self.selected = Selection::Completion(i + 1);
            }
            (Focus::Completions, Selection::None) if !self.completions.is_empty() => {
                self.selected = Selection::Completion(0);
            }
            _ => {}
        }
    }

    /// Cycles focus between panels.
    pub const fn cycle_focus(&mut self) {
        self.focus = match self.focus {
            Focus::Cluster => Focus::Queue,
            Focus::Queue => Focus::Completions,
            Focus::Completions => Focus::Cluster,
        };
        self.selected = Selection::None;
    }

    /// Toggles help overlay.
    pub const fn toggle_help(&mut self) {
        self.show_help = !self.show_help;
    }

    /// Toggles alerts panel.
    pub const fn toggle_alerts(&mut self) {
        self.show_alerts = !self.show_alerts;
    }

    /// Updates on each tick.
    pub fn tick(&mut self) {
        self.tick_count += 1;

        // Update node histories
        for node in &mut self.nodes {
            node.update_history();
        }

        // Update metrics history
        self.history.push(&self.nodes);

        // Generate alerts
        self.alerts = self.generate_alerts();
    }

    /// Generates alerts based on current state.
    fn generate_alerts(&self) -> Vec<Alert> {
        let mut alerts = Vec::new();

        for node in &self.nodes {
            // Memory pressure alert
            if node.mem_pct > 90.0 {
                alerts.push(Alert::MemoryPressure { node: node.name.clone(), pct: node.mem_pct });
            }

            // Check backends for memory pressure
            for backend in &node.backends {
                if backend.memory_pct > 90.0 {
                    alerts.push(Alert::MemoryPressure {
                        node: format!("{} {}", node.name, backend.backend_type),
                        pct: backend.memory_pct,
                    });
                }
            }

            // Node suspected alert
            if node.state == NodeState::Suspected {
                alerts.push(Alert::NodeSuspected {
                    node: node.name.clone(),
                    last_seen: Duration::from_secs(30), // Placeholder
                });
            }
        }

        // Work imbalance detection
        if self.nodes.len() >= 2 {
            let max_tasks = self.nodes.iter().map(|n| n.running_tasks).max().unwrap_or(0);
            let min_tasks = self.nodes.iter().map(|n| n.running_tasks).min().unwrap_or(0);

            if max_tasks > 0 && f64::from(max_tasks - min_tasks) / f64::from(max_tasks) > 0.5 {
                let overloaded = self.nodes.iter().find(|n| n.running_tasks == max_tasks);
                let underloaded = self.nodes.iter().find(|n| n.running_tasks == min_tasks);

                if let (Some(over), Some(under)) = (overloaded, underloaded) {
                    alerts.push(Alert::WorkImbalance {
                        overloaded: over.name.clone(),
                        underloaded: under.name.clone(),
                    });
                }
            }
        }

        alerts
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::disallowed_methods,
    clippy::float_cmp,
    clippy::cast_precision_loss,
    clippy::uninlined_format_args,
    unused_must_use,
    clippy::panic
)]
mod tests {
    use super::*;

    // =========================================================================
    // BackendType Tests
    // =========================================================================

    #[test]
    fn test_backend_type_display() {
        assert_eq!(format!("{}", BackendType::Cpu), "CPU");
        assert_eq!(format!("{}", BackendType::Cuda), "CUDA");
        assert_eq!(format!("{}", BackendType::Metal), "Metal");
        assert_eq!(format!("{}", BackendType::Vulkan), "Vulkan");
        assert_eq!(format!("{}", BackendType::Rocm), "ROCm");
    }

    // =========================================================================
    // NodeStatus Tests
    // =========================================================================

    #[test]
    fn test_node_status_new() {
        let node = NodeStatus::new("test-node", "127.0.0.1:9000".parse().unwrap());
        assert_eq!(node.name, "test-node");
        assert_eq!(node.cpu_pct, 0.0);
        assert_eq!(node.state, NodeState::Online);
        assert!(node.backends.is_empty());
    }

    #[test]
    fn test_node_status_add_backend() {
        let mut node = NodeStatus::new("test-node", "127.0.0.1:9000".parse().unwrap());
        let backend = BackendStatus::new(BackendType::Cuda, "RTX 4090");
        node.add_backend(backend);
        assert_eq!(node.backends.len(), 1);
        assert_eq!(node.backends[0].backend_type, BackendType::Cuda);
    }

    #[test]
    fn test_node_status_update_history() {
        let mut node = NodeStatus::new("test-node", "127.0.0.1:9000".parse().unwrap());
        node.cpu_pct = 50.0;

        for _ in 0..70 {
            node.update_history();
        }

        // Should be capped at HISTORY_SIZE
        assert_eq!(node.load_history.len(), HISTORY_SIZE);
    }

    // =========================================================================
    // TaskQueue Tests
    // =========================================================================

    #[test]
    fn test_task_queue_total() {
        let queue = TaskQueue {
            pending: 10,
            in_flight: 5,
            high_priority: 3,
            normal_priority: 10,
            low_priority: 2,
        };
        assert_eq!(queue.total(), 15);
    }

    #[test]
    fn test_task_queue_utilization_empty() {
        let queue = TaskQueue::default();
        assert_eq!(queue.utilization_pct(), 0.0);
    }

    #[test]
    fn test_task_queue_utilization() {
        let queue = TaskQueue { pending: 500, in_flight: 0, ..Default::default() };
        assert!((queue.utilization_pct() - 50.0).abs() < 0.1);
    }

    // =========================================================================
    // CompletionRecord Tests
    // =========================================================================

    #[test]
    fn test_completion_record_success() {
        let task_id = TaskId::new();
        let record = CompletionRecord::success(
            task_id,
            BackendType::Cuda,
            "node-1",
            Duration::from_millis(45),
        );
        assert!(record.success);
        assert!(record.error.is_none());
        assert_eq!(record.duration, Duration::from_millis(45));
    }

    #[test]
    fn test_completion_record_failure() {
        let task_id = TaskId::new();
        let record = CompletionRecord::failure(task_id, BackendType::Cuda, "node-1", "TIMEOUT");
        assert!(!record.success);
        assert_eq!(record.error, Some("TIMEOUT".to_string()));
    }

    // =========================================================================
    // Alert Tests
    // =========================================================================

    #[test]
    fn test_alert_memory_pressure_message() {
        let alert = Alert::MemoryPressure { node: "node-1".to_string(), pct: 95.0 };
        assert!(alert.message().contains("memory pressure"));
        assert!(alert.message().contains("95%"));
    }

    #[test]
    fn test_alert_work_imbalance_message() {
        let alert = Alert::WorkImbalance {
            overloaded: "node-1".to_string(),
            underloaded: "node-2".to_string(),
        };
        assert!(alert.message().contains("imbalance"));
    }

    #[test]
    fn test_alert_node_suspected_message() {
        let alert =
            Alert::NodeSuspected { node: "node-1".to_string(), last_seen: Duration::from_secs(30) };
        let msg = alert.message();
        assert!(msg.contains("suspected offline"));
        assert!(msg.contains("node-1"));
    }

    #[test]
    fn test_alert_task_timeout_message() {
        let alert = Alert::TaskTimeout { task_id: TaskId::new(), elapsed: Duration::from_secs(60) };
        let msg = alert.message();
        assert!(msg.contains("timeout"));
    }

    #[test]
    fn test_alert_backend_error_message() {
        let alert = Alert::BackendError {
            node: "node-1".to_string(),
            backend: BackendType::Cuda,
            error: "GPU memory full".to_string(),
        };
        let msg = alert.message();
        assert!(msg.contains("node-1"));
        assert!(msg.contains("CUDA"));
        assert!(msg.contains("GPU memory full"));
    }

    // =========================================================================
    // MetricsHistory Tests
    // =========================================================================

    #[test]
    fn test_metrics_history_push() {
        let mut history = MetricsHistory::new();
        let nodes = vec![NodeStatus::new("node-1", "127.0.0.1:9000".parse().unwrap())];

        history.push(&nodes);
        assert_eq!(history.cpu_history.len(), 1);
        assert_eq!(history.gpu_history.len(), 1);
    }

    #[test]
    fn test_metrics_history_capacity() {
        let mut history = MetricsHistory::new();
        let mut node = NodeStatus::new("node-1", "127.0.0.1:9000".parse().unwrap());
        node.cpu_pct = 50.0;

        for _ in 0..100 {
            history.push(&[node.clone()]);
        }

        assert_eq!(history.cpu_history[0].len(), HISTORY_SIZE);
    }

    // =========================================================================
    // App Tests
    // =========================================================================

    #[test]
    fn test_app_new() {
        let app = App::new();
        assert!(app.nodes.is_empty());
        assert!(app.completions.is_empty());
        assert!(app.alerts.is_empty());
        assert_eq!(app.focus, Focus::Cluster);
    }

    #[test]
    fn test_app_add_node() {
        let mut app = App::new();
        let node = NodeStatus::new("node-1", "127.0.0.1:9000".parse().unwrap());
        app.add_node(node);
        assert_eq!(app.nodes.len(), 1);
    }

    #[test]
    fn test_app_add_completion() {
        let mut app = App::new();
        let task_id = TaskId::new();
        let record = CompletionRecord::success(
            task_id,
            BackendType::Cpu,
            "node-1",
            Duration::from_millis(10),
        );
        app.add_completion(record);
        assert_eq!(app.completions.len(), 1);
    }

    #[test]
    fn test_app_completion_capacity() {
        let mut app = App::new();

        for _ in 0..150 {
            let record = CompletionRecord::success(
                TaskId::new(),
                BackendType::Cpu,
                "node-1",
                Duration::from_millis(10),
            );
            app.add_completion(record);
        }

        assert_eq!(app.completions.len(), MAX_COMPLETIONS);
    }

    #[test]
    fn test_app_select_next_empty() {
        let mut app = App::new();
        app.select_next();
        assert!(matches!(app.selected, Selection::None));
    }

    #[test]
    fn test_app_select_next_with_nodes() {
        let mut app = App::new();
        app.add_node(NodeStatus::new("node-1", "127.0.0.1:9000".parse().unwrap()));
        app.add_node(NodeStatus::new("node-2", "127.0.0.1:9001".parse().unwrap()));

        app.select_next();
        assert!(matches!(app.selected, Selection::Node(0)));

        app.select_next();
        assert!(matches!(app.selected, Selection::Node(1)));

        // Should not go beyond last
        app.select_next();
        assert!(matches!(app.selected, Selection::Node(1)));
    }

    #[test]
    fn test_app_select_prev_with_nodes() {
        let mut app = App::new();
        app.add_node(NodeStatus::new("node-1", "127.0.0.1:9000".parse().unwrap()));
        app.add_node(NodeStatus::new("node-2", "127.0.0.1:9001".parse().unwrap()));

        app.selected = Selection::Node(1);
        app.select_prev();
        assert!(matches!(app.selected, Selection::Node(0)));

        // Should not go below 0
        app.select_prev();
        assert!(matches!(app.selected, Selection::Node(0)));
    }

    #[test]
    fn test_app_cycle_focus() {
        let mut app = App::new();
        assert_eq!(app.focus, Focus::Cluster);

        app.cycle_focus();
        assert_eq!(app.focus, Focus::Queue);

        app.cycle_focus();
        assert_eq!(app.focus, Focus::Completions);

        app.cycle_focus();
        assert_eq!(app.focus, Focus::Cluster);
    }

    #[test]
    fn test_app_toggle_help() {
        let mut app = App::new();
        assert!(!app.show_help);
        app.toggle_help();
        assert!(app.show_help);
        app.toggle_help();
        assert!(!app.show_help);
    }

    #[test]
    fn test_app_toggle_alerts() {
        let mut app = App::new();
        assert!(app.show_alerts);
        app.toggle_alerts();
        assert!(!app.show_alerts);
    }

    #[test]
    fn test_app_tick() {
        let mut app = App::new();
        app.add_node(NodeStatus::new("node-1", "127.0.0.1:9000".parse().unwrap()));

        let initial_count = app.tick_count;
        app.tick();
        assert_eq!(app.tick_count, initial_count + 1);
    }

    #[test]
    fn test_app_generate_alerts_memory_pressure() {
        let mut app = App::new();
        let mut node = NodeStatus::new("node-1", "127.0.0.1:9000".parse().unwrap());
        node.mem_pct = 95.0;
        app.add_node(node);

        let alerts = app.generate_alerts();
        assert!(alerts.iter().any(|a| matches!(a, Alert::MemoryPressure { .. })));
    }

    #[test]
    fn test_app_generate_alerts_work_imbalance() {
        let mut app = App::new();

        let mut node1 = NodeStatus::new("node-1", "127.0.0.1:9000".parse().unwrap());
        node1.running_tasks = 10;
        app.add_node(node1);

        let mut node2 = NodeStatus::new("node-2", "127.0.0.1:9001".parse().unwrap());
        node2.running_tasks = 1;
        app.add_node(node2);

        let alerts = app.generate_alerts();
        assert!(alerts.iter().any(|a| matches!(a, Alert::WorkImbalance { .. })));
    }

    #[test]
    fn test_app_generate_alerts_node_suspected() {
        let mut app = App::new();
        let mut node = NodeStatus::new("node-1", "127.0.0.1:9000".parse().unwrap());
        node.state = NodeState::Suspected;
        app.add_node(node);

        let alerts = app.generate_alerts();
        assert!(alerts.iter().any(|a| matches!(a, Alert::NodeSuspected { .. })));
    }

    #[test]
    fn test_app_generate_alerts_backend_memory_pressure() {
        let mut app = App::new();
        let mut node = NodeStatus::new("node-1", "127.0.0.1:9000".parse().unwrap());
        node.add_backend(BackendStatus {
            backend_type: BackendType::Cuda,
            device_name: "RTX 4090".to_string(),
            utilization: 50.0,
            memory_pct: 95.0, // High memory pressure
            temperature: Some(65.0),
        });
        app.add_node(node);

        let alerts = app.generate_alerts();
        assert!(alerts.iter().any(|a| matches!(a, Alert::MemoryPressure { .. })));
    }

    #[test]
    fn test_app_select_completions() {
        let mut app = App::new();
        app.focus = Focus::Completions;

        app.add_completion(CompletionRecord::success(
            TaskId::new(),
            BackendType::Cpu,
            "node-1",
            Duration::from_millis(10),
        ));
        app.add_completion(CompletionRecord::success(
            TaskId::new(),
            BackendType::Cpu,
            "node-2",
            Duration::from_millis(20),
        ));

        // Navigate in completions
        app.select_next();
        assert!(matches!(app.selected, Selection::Completion(0)));

        app.select_next();
        assert!(matches!(app.selected, Selection::Completion(1)));

        app.select_prev();
        assert!(matches!(app.selected, Selection::Completion(0)));
    }

    #[test]
    fn test_app_select_prev_no_wrap() {
        let mut app = App::new();
        app.focus = Focus::Completions;

        app.add_completion(CompletionRecord::success(
            TaskId::new(),
            BackendType::Cpu,
            "node-1",
            Duration::from_millis(10),
        ));

        // Select prev from none should select last
        app.select_prev();
        assert!(matches!(app.selected, Selection::Completion(0)));
    }

    #[test]
    fn test_metrics_history_with_gpu() {
        let mut history = MetricsHistory::new();
        let mut node = NodeStatus::new("node-1", "127.0.0.1:9000".parse().unwrap());
        node.cpu_pct = 50.0;
        node.add_backend(BackendStatus {
            backend_type: BackendType::Cuda,
            device_name: "RTX 4090".to_string(),
            utilization: 80.0,
            memory_pct: 50.0,
            temperature: Some(65.0),
        });

        history.push(&[node]);

        assert_eq!(history.gpu_history.len(), 1);
        // GPU utilization should be recorded
        assert!(!history.gpu_history[0].is_empty());
    }

    #[test]
    fn test_app_default_impl() {
        let app = App::default();
        assert!(app.nodes.is_empty());
        assert_eq!(app.focus, Focus::Cluster);
    }

    #[test]
    fn test_backend_status_new() {
        let backend = BackendStatus::new(BackendType::Metal, "M1 Max");
        assert_eq!(backend.backend_type, BackendType::Metal);
        assert_eq!(backend.device_name, "M1 Max");
        assert_eq!(backend.utilization, 0.0);
        assert!(backend.temperature.is_none());
    }

    #[test]
    fn test_node_state_default() {
        let state = NodeState::default();
        assert_eq!(state, NodeState::Online);
    }

    #[test]
    fn test_selection_default() {
        let selection = Selection::default();
        assert!(matches!(selection, Selection::None));
    }

    #[test]
    fn test_focus_default() {
        let focus = Focus::default();
        assert_eq!(focus, Focus::Cluster);
    }

    // =========================================================================
    // Property-Based Tests
    // =========================================================================

    use proptest::prelude::*;

    proptest! {
        #[test]
        fn prop_task_queue_total_is_sum(pending in 0usize..1000, in_flight in 0usize..1000) {
            let queue = TaskQueue {
                pending,
                in_flight,
                ..Default::default()
            };
            prop_assert_eq!(queue.total(), pending + in_flight);
        }

        #[test]
        fn prop_utilization_is_bounded(pending in 0usize..2000, in_flight in 0usize..2000) {
            let queue = TaskQueue {
                pending,
                in_flight,
                ..Default::default()
            };
            let util = queue.utilization_pct();
            prop_assert!((0.0..=100.0).contains(&util));
        }

        #[test]
        fn prop_node_history_bounded(updates in 0usize..200) {
            let mut node = NodeStatus::new("test", "127.0.0.1:9000".parse().unwrap());
            node.cpu_pct = 50.0;

            for _ in 0..updates {
                node.update_history();
            }

            prop_assert!(node.load_history.len() <= HISTORY_SIZE);
        }

        #[test]
        fn prop_completions_bounded(count in 0usize..200) {
            let mut app = App::new();

            for _ in 0..count {
                let record = CompletionRecord::success(
                    TaskId::new(),
                    BackendType::Cpu,
                    "node",
                    Duration::from_millis(10),
                );
                app.add_completion(record);
            }

            prop_assert!(app.completions.len() <= MAX_COMPLETIONS);
        }
    }
}
