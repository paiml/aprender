//! GPU metrics collector
//!
//! Collects GPU metrics via NVML/wgpu (Genchi Genbutsu)

use crate::brick::{Brick, BrickAssertion, BrickBudget, BrickVerification};
use crate::ring_buffer::RingBuffer;
use std::any::Any;
use std::time::Instant;

#[cfg(feature = "cuda")]
use trueno_gpu::monitor::{CudaDeviceInfo, CudaMemoryInfo};

/// GPU metrics
#[derive(Debug, Clone)]
pub struct GpuMetrics {
    pub timestamp: Instant,
    pub device_index: u32,
    pub device_name: String,
    pub utilization_gpu: u32,
    pub memory_used_mb: u64,
    pub memory_total_mb: u64,
}

impl Default for GpuMetrics {
    fn default() -> Self {
        Self {
            timestamp: Instant::now(),
            device_index: 0,
            device_name: "None".to_string(),
            utilization_gpu: 0,
            memory_used_mb: 0,
            memory_total_mb: 0,
        }
    }
}

/// GPU collector brick
pub struct GpuCollectorBrick {
    history: RingBuffer<GpuMetrics>,
    device_index: u32,
}

impl GpuCollectorBrick {
    pub fn new(device_index: u32) -> Self {
        Self {
            history: RingBuffer::new(120),
            device_index,
        }
    }

    pub fn collect(&mut self) -> GpuMetrics {
        let metrics = self.read_gpu();
        self.history.push(metrics.clone());
        metrics
    }

    #[cfg(feature = "cuda")]
    fn read_gpu(&self) -> GpuMetrics {
        let mut metrics = GpuMetrics::default();
        metrics.timestamp = Instant::now();
        metrics.device_index = self.device_index;

        if let Ok(info) = CudaDeviceInfo::query(self.device_index) {
            metrics.memory_total_mb = info.total_memory_mb();
            metrics.device_name = info.name;

            if let Ok(ctx) = trueno_gpu::driver::CudaContext::new(self.device_index as i32) {
                if let Ok(mem) = CudaMemoryInfo::query(&ctx) {
                    metrics.memory_used_mb = mem.used_mb();
                    metrics.utilization_gpu = (mem.usage_percent() as u32).min(100);
                }
            }
        }
        metrics
    }

    #[cfg(not(feature = "cuda"))]
    fn read_gpu(&self) -> GpuMetrics {
        GpuMetrics::default()
    }

    pub fn history(&self) -> &RingBuffer<GpuMetrics> {
        &self.history
    }
}

impl Brick for GpuCollectorBrick {
    fn brick_name(&self) -> &'static str {
        "gpu_collector"
    }

    fn assertions(&self) -> Vec<BrickAssertion> {
        vec![
            BrickAssertion::custom("memory_valid", |b| {
                let s = b
                    .downcast_ref::<GpuCollectorBrick>()
                    .expect("brick MUST be GpuCollectorBrick");
                s.history
                    .back()
                    .map_or(true, |m| m.memory_used_mb <= m.memory_total_mb)
            }),
            BrickAssertion::max_latency_ms(20),
        ]
    }

    fn budget(&self) -> BrickBudget {
        BrickBudget {
            collect_ms: 20,
            layout_ms: 0,
            render_ms: 0,
        }
    }

    fn verify(&self) -> BrickVerification {
        let mut v = BrickVerification::new();
        for a in self.assertions() {
            v.check(&a);
        }
        v
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_collector() {
        let mut collector = GpuCollectorBrick::new(0);
        let _ = collector.collect();
        let v = collector.verify();
        assert!(v.is_valid());
    }
}
