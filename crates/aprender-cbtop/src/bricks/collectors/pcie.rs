//! PCIe metrics collector
//!
//! Collects PCIe link speed from sysfs (Genchi Genbutsu)

use crate::brick::{Brick, BrickAssertion, BrickBudget, BrickVerification};
use crate::ring_buffer::RingBuffer;
use std::any::Any;
use std::fs;
use std::time::Instant;

#[derive(Debug, Clone)]
pub struct PcieMetrics {
    pub timestamp: Instant,
    /// Number of Gen4 devices
    pub gen4_count: usize,
    /// Number of Gen5 devices
    pub gen5_count: usize,
    /// Max link width found
    pub max_width: u8,
}

impl Default for PcieMetrics {
    fn default() -> Self {
        Self {
            timestamp: Instant::now(),
            gen4_count: 0,
            gen5_count: 0,
            max_width: 0,
        }
    }
}

pub struct PcieCollectorBrick {
    history: RingBuffer<PcieMetrics>,
}

impl PcieCollectorBrick {
    pub fn new() -> Self {
        Self {
            history: RingBuffer::new(120),
        }
    }

    pub fn collect(&mut self) -> PcieMetrics {
        let metrics = self.read_pcie().unwrap_or_default();
        self.history.push(metrics.clone());
        metrics
    }

    fn read_pcie(&self) -> Result<PcieMetrics, std::io::Error> {
        let mut gen4 = 0;
        let mut gen5 = 0;
        let mut max_width = 0;

        if let Ok(entries) = fs::read_dir("/sys/bus/pci/devices") {
            for entry in entries {
                if let Ok(entry) = entry {
                    let path = entry.path();

                    // Check generation
                    if let Ok(speed) = fs::read_to_string(path.join("current_link_speed")) {
                        if speed.contains("16.0 GT/s") {
                            gen4 += 1;
                        } else if speed.contains("32.0 GT/s") {
                            gen5 += 1;
                        }
                    }

                    // Check width
                    if let Ok(width_str) = fs::read_to_string(path.join("current_link_width")) {
                        if let Ok(width) = width_str.trim().parse::<u8>() {
                            if width > max_width {
                                max_width = width;
                            }
                        }
                    }
                }
            }
        }

        Ok(PcieMetrics {
            timestamp: Instant::now(),
            gen4_count: gen4,
            gen5_count: gen5,
            max_width,
        })
    }

    pub fn history(&self) -> &RingBuffer<PcieMetrics> {
        &self.history
    }
}

impl Brick for PcieCollectorBrick {
    fn brick_name(&self) -> &'static str {
        "pcie_collector"
    }

    fn assertions(&self) -> Vec<BrickAssertion> {
        vec![
            BrickAssertion::custom("width_valid", |b| {
                let s = b
                    .downcast_ref::<PcieCollectorBrick>()
                    .expect("brick MUST be PcieCollectorBrick");
                s.history.back().map_or(true, |m| m.max_width <= 16)
            }),
            BrickAssertion::max_latency_ms(10), // Filesystem scan can be slow
        ]
    }

    fn budget(&self) -> BrickBudget {
        BrickBudget {
            collect_ms: 10,
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
    fn test_pcie_collector() {
        let mut collector = PcieCollectorBrick::new();
        let metrics = collector.collect();
        assert!(metrics.max_width <= 16);
    }
}
