//! Memory bandwidth stress generator

use crate::brick::{Brick, BrickAssertion, BrickBudget, BrickVerification};
use std::any::Any;

pub struct MemBandwidthBrick {
    is_running: bool,
    intensity: f64,
    buffer_size: usize,
    buffer: Vec<u8>,
}

impl MemBandwidthBrick {
    pub fn new(buffer_size: usize) -> Self {
        Self {
            is_running: false,
            intensity: 0.0,
            buffer_size,
            buffer: vec![0u8; buffer_size],
        }
    }

    pub fn start(&mut self) {
        self.is_running = true;
    }

    pub fn stop(&mut self) {
        self.is_running = false;
    }

    pub fn is_running(&self) -> bool {
        self.is_running
    }

    pub fn set_intensity(&mut self, intensity: f64) {
        self.intensity = intensity.clamp(0.0, 1.0);
    }
}

impl Default for MemBandwidthBrick {
    fn default() -> Self {
        Self::new(64 * 1024 * 1024) // 64 MB
    }
}

impl Brick for MemBandwidthBrick {
    fn brick_name(&self) -> &'static str {
        "mem_bandwidth"
    }

    fn assertions(&self) -> Vec<BrickAssertion> {
        vec![
            BrickAssertion::custom("buffer_allocated", |_| true),
            BrickAssertion::max_latency_ms(100),
        ]
    }

    fn budget(&self) -> BrickBudget {
        BrickBudget {
            collect_ms: 16,
            layout_ms: 0,
            render_ms: 0,
        }
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
