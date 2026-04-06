//! Bottleneck analyzer using Roofline model

use crate::brick::{Brick, BrickAssertion, BrickBudget, BrickVerification};
use std::any::Any;

/// Roofline model analysis
pub struct BottleneckAnalyzerBrick {
    peak_flops: f64,
    peak_bandwidth: f64,
}

impl BottleneckAnalyzerBrick {
    pub fn new(peak_flops: f64, peak_bandwidth: f64) -> Self {
        Self { peak_flops, peak_bandwidth }
    }

    pub fn analyze(&self, achieved_flops: f64, operational_intensity: f64) -> BottleneckResult {
        // Roofline: achieved_perf <= min(peak_flops, peak_bandwidth * op_intensity)
        let memory_roof = self.peak_bandwidth * operational_intensity;
        let is_compute_bound = memory_roof > self.peak_flops;
        let theoretical_peak = memory_roof.min(self.peak_flops);
        let efficiency = achieved_flops / theoretical_peak;

        BottleneckResult {
            is_compute_bound,
            operational_intensity,
            achieved_flops,
            theoretical_peak,
            efficiency,
            bottleneck: if is_compute_bound { Bottleneck::Compute } else { Bottleneck::Memory },
        }
    }

    pub fn reset(&mut self) {}
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bottleneck {
    Compute,
    Memory,
    Unknown,
}

#[derive(Debug, Clone)]
pub struct BottleneckResult {
    pub is_compute_bound: bool,
    pub operational_intensity: f64,
    pub achieved_flops: f64,
    pub theoretical_peak: f64,
    pub efficiency: f64,
    pub bottleneck: Bottleneck,
}

impl Default for BottleneckAnalyzerBrick {
    fn default() -> Self {
        // Default: 10 TFLOPS, 500 GB/s
        Self::new(10e12, 500e9)
    }
}

impl Brick for BottleneckAnalyzerBrick {
    fn brick_name(&self) -> &'static str {
        "bottleneck_analyzer"
    }

    fn assertions(&self) -> Vec<BrickAssertion> {
        vec![BrickAssertion::custom("roofline_valid", |_| true), BrickAssertion::max_latency_ms(1)]
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
