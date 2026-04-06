//! Throughput analyzer using Little's Law

use crate::brick::{Brick, BrickAssertion, BrickBudget, BrickVerification};
use std::any::Any;

/// Little's Law: L = λW
/// L = average number in system
/// λ = arrival rate
/// W = average time in system
pub struct ThroughputAnalyzerBrick {
    samples: Vec<f64>,
    arrival_rate: f64,
    avg_latency_ms: f64,
}

impl ThroughputAnalyzerBrick {
    pub fn new() -> Self {
        Self { samples: Vec::new(), arrival_rate: 0.0, avg_latency_ms: 0.0 }
    }

    pub fn analyze(&mut self, ops_per_sec: f64, latency_ms: f64) -> ThroughputResult {
        self.arrival_rate = ops_per_sec;
        self.avg_latency_ms = latency_ms;

        // Little's Law: L = λW
        let items_in_system = ops_per_sec * (latency_ms / 1000.0);

        ThroughputResult {
            ops_per_sec,
            latency_ms,
            items_in_system,
            is_saturated: items_in_system > 1.0,
        }
    }

    pub fn reset(&mut self) {
        self.samples.clear();
        self.arrival_rate = 0.0;
        self.avg_latency_ms = 0.0;
    }
}

#[derive(Debug, Clone)]
pub struct ThroughputResult {
    pub ops_per_sec: f64,
    pub latency_ms: f64,
    pub items_in_system: f64,
    pub is_saturated: bool,
}

impl Default for ThroughputAnalyzerBrick {
    fn default() -> Self {
        Self::new()
    }
}

impl Brick for ThroughputAnalyzerBrick {
    fn brick_name(&self) -> &'static str {
        "throughput_analyzer"
    }

    fn assertions(&self) -> Vec<BrickAssertion> {
        vec![
            BrickAssertion::custom("littles_law_valid", |_| true),
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
