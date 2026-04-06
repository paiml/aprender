//! Memory metrics collector
//!
//! Collects memory usage from /proc/meminfo (Genchi Genbutsu)

use crate::brick::{Brick, BrickAssertion, BrickBudget, BrickVerification};
use crate::ring_buffer::RingBuffer;
use std::any::Any;
use std::fs::read_to_string;
use std::time::Instant;

#[derive(Debug, Clone)]
pub struct MemoryMetrics {
    pub timestamp: Instant,
    pub total_kb: u64,
    pub available_kb: u64,
    pub free_kb: u64,
    pub swap_total_kb: u64,
    pub swap_free_kb: u64,
}

impl Default for MemoryMetrics {
    fn default() -> Self {
        Self {
            timestamp: Instant::now(),
            total_kb: 0,
            available_kb: 0,
            free_kb: 0,
            swap_total_kb: 0,
            swap_free_kb: 0,
        }
    }
}

pub struct MemoryCollectorBrick {
    history: RingBuffer<MemoryMetrics>,
}

impl MemoryCollectorBrick {
    pub fn new() -> Self {
        Self {
            history: RingBuffer::new(120),
        }
    }

    pub fn collect(&mut self) -> MemoryMetrics {
        let metrics = self.read_meminfo().unwrap_or_default();
        self.history.push(metrics.clone());
        metrics
    }

    fn read_meminfo(&self) -> Result<MemoryMetrics, std::io::Error> {
        let content = read_to_string("/proc/meminfo")?;
        let mut metrics = MemoryMetrics {
            timestamp: Instant::now(),
            ..Default::default()
        };

        for line in content.lines() {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() < 2 {
                continue;
            }

            let value = parts[1].parse::<u64>().unwrap_or(0);
            match parts[0] {
                "MemTotal:" => metrics.total_kb = value,
                "MemAvailable:" => metrics.available_kb = value,
                "MemFree:" => metrics.free_kb = value,
                "SwapTotal:" => metrics.swap_total_kb = value,
                "SwapFree:" => metrics.swap_free_kb = value,
                _ => {}
            }
        }
        Ok(metrics)
    }

    pub fn history(&self) -> &RingBuffer<MemoryMetrics> {
        &self.history
    }
}

impl Brick for MemoryCollectorBrick {
    fn brick_name(&self) -> &'static str {
        "memory_collector"
    }

    fn assertions(&self) -> Vec<BrickAssertion> {
        vec![
            BrickAssertion::custom("mem_total_positive", |b| {
                let s = b
                    .downcast_ref::<MemoryCollectorBrick>()
                    .expect("brick MUST be MemoryCollectorBrick");
                s.history.back().map_or(true, |m| m.total_kb > 0)
            }),
            BrickAssertion::max_latency_ms(2),
        ]
    }

    fn budget(&self) -> BrickBudget {
        BrickBudget {
            collect_ms: 2,
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
    fn test_memory_collector() {
        let mut collector = MemoryCollectorBrick::new();
        let metrics = collector.collect();
        // On linux this should be > 0. In CI/other OS it might fail to read file.
        // We assert logic, not environment.
        if std::path::Path::new("/proc/meminfo").exists() {
            assert!(metrics.total_kb > 0);
        } else {
            // Fallback for non-linux
            assert!(metrics.total_kb == 0);
        }
    }
}
