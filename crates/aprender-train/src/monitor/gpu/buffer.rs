//! GPU metrics history buffer (ring buffer).

use std::collections::VecDeque;

use super::GpuMetrics;

/// GPU metrics history buffer (ring buffer)
#[derive(Debug)]
pub struct GpuMetricsBuffer {
    /// Capacity
    capacity: usize,
    /// Metrics per device
    buffers: Vec<VecDeque<GpuMetrics>>,
}

impl GpuMetricsBuffer {
    /// Create a new buffer with given capacity
    pub fn new(capacity: usize, num_devices: usize) -> Self {
        let buffers = (0..num_devices).map(|_| VecDeque::with_capacity(capacity)).collect();
        Self { capacity, buffers }
    }

    /// Push metrics for all devices
    pub fn push(&mut self, metrics: &[GpuMetrics]) {
        for m in metrics {
            let device_idx = m.device_id as usize;
            if device_idx >= self.buffers.len() {
                self.buffers.resize_with(device_idx + 1, || VecDeque::with_capacity(self.capacity));
            }

            let buffer = &mut self.buffers[device_idx];
            if buffer.len() >= self.capacity {
                buffer.pop_front();
            }
            buffer.push_back(m.clone());
        }
    }

    /// Get last N metrics for a device
    pub fn last_n(&self, device_id: u32, n: usize) -> Vec<&GpuMetrics> {
        let device_idx = device_id as usize;
        if device_idx >= self.buffers.len() {
            return Vec::new();
        }

        self.buffers[device_idx].iter().rev().take(n).rev().collect()
    }

    /// Get utilization history for a device (for sparkline)
    pub fn utilization_history(&self, device_id: u32) -> Vec<u32> {
        let device_idx = device_id as usize;
        if device_idx >= self.buffers.len() {
            return Vec::new();
        }

        self.buffers[device_idx].iter().map(|m| m.utilization_percent).collect()
    }

    /// Get temperature history for a device
    pub fn temperature_history(&self, device_id: u32) -> Vec<u32> {
        let device_idx = device_id as usize;
        if device_idx >= self.buffers.len() {
            return Vec::new();
        }

        self.buffers[device_idx].iter().map(|m| m.temperature_celsius).collect()
    }

    /// Get memory utilization history for a device
    pub fn memory_history(&self, device_id: u32) -> Vec<f64> {
        let device_idx = device_id as usize;
        if device_idx >= self.buffers.len() {
            return Vec::new();
        }

        self.buffers[device_idx].iter().map(GpuMetrics::memory_percent).collect()
    }

    /// Get number of samples for a device
    pub fn len(&self, device_id: u32) -> usize {
        let device_idx = device_id as usize;
        if device_idx >= self.buffers.len() {
            return 0;
        }
        self.buffers[device_idx].len()
    }

    /// Check if buffer for device is empty
    pub fn is_empty(&self, device_id: u32) -> bool {
        self.len(device_id) == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_push_and_last_n() {
        let mut buffer = GpuMetricsBuffer::new(10, 1);

        for i in 0..5 {
            buffer.push(&[GpuMetrics {
                device_id: 0,
                utilization_percent: i * 10,
                ..Default::default()
            }]);
        }

        let last3 = buffer.last_n(0, 3);
        assert_eq!(last3.len(), 3);
        assert_eq!(last3[0].utilization_percent, 20);
        assert_eq!(last3[1].utilization_percent, 30);
        assert_eq!(last3[2].utilization_percent, 40);
    }

    #[test]
    fn test_buffer_capacity_limit() {
        let mut buffer = GpuMetricsBuffer::new(5, 1);

        for i in 0..10 {
            buffer.push(&[GpuMetrics {
                device_id: 0,
                utilization_percent: i,
                ..Default::default()
            }]);
        }

        assert_eq!(buffer.len(0), 5);

        let history = buffer.utilization_history(0);
        assert_eq!(history, vec![5, 6, 7, 8, 9]);
    }

    #[test]
    fn test_buffer_utilization_history() {
        let mut buffer = GpuMetricsBuffer::new(10, 1);

        for i in 0..5 {
            buffer.push(&[GpuMetrics {
                device_id: 0,
                utilization_percent: i * 20,
                ..Default::default()
            }]);
        }

        let history = buffer.utilization_history(0);
        assert_eq!(history, vec![0, 20, 40, 60, 80]);
    }

    #[test]
    fn test_buffer_temperature_history() {
        let mut buffer = GpuMetricsBuffer::new(10, 1);

        for i in 0..3 {
            buffer.push(&[GpuMetrics {
                device_id: 0,
                temperature_celsius: 60 + i * 5,
                ..Default::default()
            }]);
        }

        let history = buffer.temperature_history(0);
        assert_eq!(history, vec![60, 65, 70]);
    }

    #[test]
    fn test_buffer_multiple_devices() {
        let mut buffer = GpuMetricsBuffer::new(10, 2);

        buffer.push(&[
            GpuMetrics { device_id: 0, utilization_percent: 50, ..Default::default() },
            GpuMetrics { device_id: 1, utilization_percent: 75, ..Default::default() },
        ]);

        assert_eq!(buffer.utilization_history(0), vec![50]);
        assert_eq!(buffer.utilization_history(1), vec![75]);
    }

    #[test]
    fn test_buffer_empty_device() {
        let buffer = GpuMetricsBuffer::new(10, 1);
        assert!(buffer.is_empty(0));
        assert!(buffer.utilization_history(5).is_empty()); // Non-existent device
    }

    #[test]
    fn test_buffer_memory_history() {
        let mut buffer = GpuMetricsBuffer::new(10, 1);

        for i in 0..3 {
            buffer.push(&[GpuMetrics {
                device_id: 0,
                memory_used_mb: i * 1000,
                memory_total_mb: 8000,
                memory_utilization_percent: (i as u32) * 10,
                ..Default::default()
            }]);
        }

        let history = buffer.memory_history(0);
        // memory_history returns memory_percent() values: 0/8000=0%, 1000/8000=12.5%, 2000/8000=25%
        assert_eq!(history.len(), 3);
        assert!((history[0] - 0.0).abs() < 0.1);
        assert!((history[1] - 12.5).abs() < 0.1);
        assert!((history[2] - 25.0).abs() < 0.1);
    }

    #[test]
    fn test_buffer_last_n_more_than_available() {
        let mut buffer = GpuMetricsBuffer::new(10, 1);

        buffer.push(&[GpuMetrics { device_id: 0, utilization_percent: 50, ..Default::default() }]);

        // Request more than available
        let last5 = buffer.last_n(0, 5);
        assert_eq!(last5.len(), 1);
    }

    #[test]
    fn test_buffer_last_nonexistent_device() {
        let buffer = GpuMetricsBuffer::new(10, 1);
        let last = buffer.last_n(99, 5);
        assert!(last.is_empty());
    }
}

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(200))]

        #[test]
        fn prop_buffer_respects_capacity(capacity in 1usize..20, pushes in 1usize..50) {
            let mut buffer = GpuMetricsBuffer::new(capacity, 1);
            for i in 0..pushes {
                buffer.push(&[GpuMetrics {
                    device_id: 0,
                    utilization_percent: i as u32 % 100,
                    ..Default::default()
                }]);
            }
            prop_assert!(buffer.len(0) <= capacity);
        }
    }
}
