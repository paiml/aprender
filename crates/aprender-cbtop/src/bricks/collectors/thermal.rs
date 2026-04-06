//! Thermal metrics collector
//!
//! Collects temperatures from /sys/class/thermal (Genchi Genbutsu)

use crate::brick::{Brick, BrickAssertion, BrickBudget, BrickVerification};
use crate::ring_buffer::RingBuffer;
use std::any::Any;
use std::fs;
use std::time::Instant;

#[derive(Debug, Clone)]
pub struct ThermalMetrics {
    pub timestamp: Instant,
    /// Max temperature found across all zones
    pub max_temp_c: f64,
    /// Average temperature
    pub avg_temp_c: f64,
}

impl Default for ThermalMetrics {
    fn default() -> Self {
        Self {
            timestamp: Instant::now(),
            max_temp_c: 0.0,
            avg_temp_c: 0.0,
        }
    }
}

pub struct ThermalCollectorBrick {
    history: RingBuffer<ThermalMetrics>,
}

impl ThermalCollectorBrick {
    pub fn new() -> Self {
        Self {
            history: RingBuffer::new(120),
        }
    }

    pub fn collect(&mut self) -> ThermalMetrics {
        let metrics = self.read_thermal().unwrap_or_default();
        self.history.push(metrics.clone());
        metrics
    }

    fn read_thermal(&self) -> Result<ThermalMetrics, std::io::Error> {
        let mut max_temp = 0.0;
        let mut total_temp = 0.0;
        let mut count = 0;

        // Try reading thermal zones
        if let Ok(entries) = fs::read_dir("/sys/class/thermal") {
            for entry in entries {
                if let Ok(entry) = entry {
                    let path = entry.path();
                    if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                        if name.starts_with("thermal_zone") {
                            if let Ok(content) = fs::read_to_string(path.join("temp")) {
                                if let Ok(temp_milli) = content.trim().parse::<f64>() {
                                    let temp = temp_milli / 1000.0;
                                    if temp > max_temp {
                                        max_temp = temp;
                                    }
                                    total_temp += temp;
                                    count += 1;
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(ThermalMetrics {
            timestamp: Instant::now(),
            max_temp_c: max_temp,
            avg_temp_c: if count > 0 {
                total_temp / count as f64
            } else {
                0.0
            },
        })
    }

    pub fn history(&self) -> &RingBuffer<ThermalMetrics> {
        &self.history
    }
}

impl Brick for ThermalCollectorBrick {
    fn brick_name(&self) -> &'static str {
        "thermal_collector"
    }

    fn assertions(&self) -> Vec<BrickAssertion> {
        vec![
            BrickAssertion::ValueInRange {
                min: 0.0,
                max: 200.0,
            }, // Temp range
            BrickAssertion::max_latency_ms(5),
        ]
    }

    fn budget(&self) -> BrickBudget {
        BrickBudget {
            collect_ms: 5,
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
    fn test_thermal_collector() {
        let mut collector = ThermalCollectorBrick::new();
        let metrics = collector.collect();
        assert!(metrics.max_temp_c >= 0.0);
    }
}
