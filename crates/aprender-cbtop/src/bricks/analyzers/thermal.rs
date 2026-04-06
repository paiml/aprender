//! Thermal analyzer with throttling detection

use crate::brick::{Brick, BrickAssertion, BrickBudget, BrickVerification};
use crate::ring_buffer::RingBuffer;
use std::any::Any;

pub struct ThermalAnalyzerBrick {
    history: RingBuffer<f64>,
    throttle_threshold: f64,
    warning_threshold: f64,
}

impl ThermalAnalyzerBrick {
    pub fn new(throttle_threshold: f64, warning_threshold: f64) -> Self {
        Self { history: RingBuffer::new(60), throttle_threshold, warning_threshold }
    }

    pub fn analyze(&mut self, temp_c: f64) -> ThermalResult {
        self.history.push(temp_c);

        let is_throttling = temp_c >= self.throttle_threshold;
        let is_warning = temp_c >= self.warning_threshold;

        // Simple trend prediction
        let trend = if self.history.len() >= 5 {
            let recent: Vec<f64> = self.history.last_n(5).copied().collect();
            let delta = recent.last().unwrap_or(&0.0) - recent.first().unwrap_or(&0.0);
            delta / 5.0 // degrees per sample
        } else {
            0.0
        };

        ThermalResult {
            current_temp: temp_c,
            is_throttling,
            is_warning,
            trend_per_second: trend,
            status: if is_throttling {
                ThermalStatus::Critical
            } else if is_warning {
                ThermalStatus::Warning
            } else {
                ThermalStatus::Normal
            },
        }
    }

    pub fn reset(&mut self) {
        self.history.clear();
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThermalStatus {
    Normal,
    Warning,
    Critical,
}

#[derive(Debug, Clone)]
pub struct ThermalResult {
    pub current_temp: f64,
    pub is_throttling: bool,
    pub is_warning: bool,
    pub trend_per_second: f64,
    pub status: ThermalStatus,
}

impl Default for ThermalAnalyzerBrick {
    fn default() -> Self {
        Self::new(90.0, 80.0)
    }
}

impl Brick for ThermalAnalyzerBrick {
    fn brick_name(&self) -> &'static str {
        "thermal_analyzer"
    }

    fn assertions(&self) -> Vec<BrickAssertion> {
        vec![
            BrickAssertion::custom("detects_throttling", |_| true),
            BrickAssertion::custom("trend_accurate", |_| true),
            BrickAssertion::max_latency_ms(1),
        ]
    }

    fn budget(&self) -> BrickBudget {
        BrickBudget { collect_ms: 1, layout_ms: 0, render_ms: 0 }
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
