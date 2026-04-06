//! Load control panel brick (Layer 3)
//!
//! Interactive controls for load testing - start/stop, backend selection,
//! workload type, intensity slider, and real-time status display.

mod render;
mod types;

pub use types::{ComputeBackend, LoadStats, WorkloadType};

use crate::brick::{Brick, BrickAssertion, BrickBudget, BrickScore, BrickVerification};
use presentar_terminal::Theme;
use std::any::Any;

/// Load control panel for interactive load testing
pub struct LoadControlPanelBrick {
    /// Selected compute backend
    pub backend: ComputeBackend,
    /// Selected workload type
    pub workload: WorkloadType,
    /// Intensity level (0.0 to 100.0)
    pub intensity: f64,
    /// Problem size (affects memory usage)
    pub problem_size: usize,
    /// Whether load test is running
    pub is_running: bool,
    /// Current load statistics
    pub stats: LoadStats,
    /// Error message if any
    pub error: Option<String>,
    /// Selected menu item (0=backend, 1=workload, 2=intensity, 3=size, 4=start/stop)
    pub selected_item: usize,
    /// Theme for rendering
    pub theme: Theme,
    /// Current ComputeBrick quality score
    pub brick_score: Option<BrickScore>,
    /// Current GFLOP/s throughput
    pub gflops: f64,
}

impl LoadControlPanelBrick {
    /// Create a new load control panel
    pub fn new() -> Self {
        Self {
            backend: ComputeBackend::default(),
            workload: WorkloadType::default(),
            intensity: 50.0,
            problem_size: 1024,
            is_running: false,
            stats: LoadStats::default(),
            error: None,
            selected_item: 0,
            theme: Theme::tokyo_night(),
            brick_score: None,
            gflops: 0.0,
        }
    }

    /// Cycle to next backend
    pub fn next_backend(&mut self) {
        let idx = ComputeBackend::ALL.iter().position(|&b| b == self.backend).unwrap_or(0);
        self.backend = ComputeBackend::ALL[(idx + 1) % ComputeBackend::ALL.len()];
    }

    /// Cycle to previous backend
    pub fn prev_backend(&mut self) {
        let idx = ComputeBackend::ALL.iter().position(|&b| b == self.backend).unwrap_or(0);
        self.backend =
            ComputeBackend::ALL[(idx + ComputeBackend::ALL.len() - 1) % ComputeBackend::ALL.len()];
    }

    /// Cycle to next workload
    pub fn next_workload(&mut self) {
        let idx = WorkloadType::ALL.iter().position(|&w| w == self.workload).unwrap_or(0);
        self.workload = WorkloadType::ALL[(idx + 1) % WorkloadType::ALL.len()];
    }

    /// Cycle to previous workload
    pub fn prev_workload(&mut self) {
        let idx = WorkloadType::ALL.iter().position(|&w| w == self.workload).unwrap_or(0);
        self.workload =
            WorkloadType::ALL[(idx + WorkloadType::ALL.len() - 1) % WorkloadType::ALL.len()];
    }

    /// Increase intensity
    pub fn increase_intensity(&mut self) {
        self.intensity = (self.intensity + 5.0).min(100.0);
    }

    /// Decrease intensity
    pub fn decrease_intensity(&mut self) {
        self.intensity = (self.intensity - 5.0).max(0.0);
    }

    /// Increase problem size
    pub fn increase_size(&mut self) {
        self.problem_size = (self.problem_size * 2).min(65536);
    }

    /// Decrease problem size
    pub fn decrease_size(&mut self) {
        self.problem_size = (self.problem_size / 2).max(64);
    }

    /// Toggle running state
    pub fn toggle_running(&mut self) {
        self.is_running = !self.is_running;
        if self.is_running {
            self.stats = LoadStats::default();
            self.error = None;
        }
    }

    /// Update statistics from load generator
    pub fn update_stats(&mut self, stats: LoadStats) {
        self.stats = stats;
    }

    /// Update ComputeBrick score
    pub fn update_score(&mut self, score: BrickScore, gflops: f64) {
        self.brick_score = Some(score);
        self.gflops = gflops;
    }

    /// Set error message
    pub fn set_error(&mut self, error: String) {
        self.error = Some(error);
        self.is_running = false;
    }

    /// Navigate to next menu item
    pub fn next_item(&mut self) {
        self.selected_item = (self.selected_item + 1) % 5;
    }

    /// Navigate to previous menu item
    pub fn prev_item(&mut self) {
        self.selected_item = (self.selected_item + 4) % 5;
    }

    /// Handle left key based on selected item
    pub fn handle_left(&mut self) {
        match self.selected_item {
            0 => self.prev_backend(),
            1 => self.prev_workload(),
            2 => self.decrease_intensity(),
            3 => self.decrease_size(),
            4 => {} // No left action for button
            _ => {}
        }
    }

    /// Handle right key based on selected item
    pub fn handle_right(&mut self) {
        match self.selected_item {
            0 => self.next_backend(),
            1 => self.next_workload(),
            2 => self.increase_intensity(),
            3 => self.increase_size(),
            4 => {} // No right action for button
            _ => {}
        }
    }

    /// Handle enter key
    pub fn handle_enter(&mut self) {
        if self.selected_item == 4 {
            self.toggle_running();
        }
    }
}

impl Default for LoadControlPanelBrick {
    fn default() -> Self {
        Self::new()
    }
}

impl Brick for LoadControlPanelBrick {
    fn brick_name(&self) -> &'static str {
        "load_control_panel"
    }

    fn assertions(&self) -> Vec<BrickAssertion> {
        vec![
            BrickAssertion::MinWidth(50),
            BrickAssertion::MinHeight(20),
            BrickAssertion::max_latency_ms(8),
        ]
    }

    fn budget(&self) -> BrickBudget {
        BrickBudget::FRAME_60FPS
    }

    fn verify(&self) -> BrickVerification {
        let mut v = BrickVerification::new();
        for assertion in self.assertions() {
            v.check(&assertion);
        }
        v
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[cfg(test)]
mod tests;
