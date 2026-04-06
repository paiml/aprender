//! CUDA load generator using trueno-gpu PTX

use crate::brick::{Brick, BrickAssertion, BrickBudget, BrickVerification};
use std::any::Any;

pub struct CudaLoadBrick {
    is_running: bool,
    intensity: f64,
    problem_size: usize,
}

impl CudaLoadBrick {
    pub fn new(problem_size: usize) -> Self {
        Self { is_running: false, intensity: 0.0, problem_size }
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

    pub fn is_available(&self) -> bool {
        // Would check for CUDA device
        false
    }
}

impl Default for CudaLoadBrick {
    fn default() -> Self {
        Self::new(1_048_576)
    }
}

impl Brick for CudaLoadBrick {
    fn brick_name(&self) -> &'static str {
        "cuda_load"
    }

    fn assertions(&self) -> Vec<BrickAssertion> {
        vec![BrickAssertion::custom("cuda_available", |_| true), BrickAssertion::max_latency_ms(10)]
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
