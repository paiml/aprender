//! MetricsBuffer - Ring Buffer for streaming metrics (ENT-055)
//!
//! Fixed-size O(1) ring buffer for training metric visualization.

/// Fixed-size ring buffer for streaming metrics.
///
/// Provides O(1) push and O(n) iteration for visualization.
#[derive(Debug, Clone)]
pub struct MetricsBuffer {
    data: Vec<f32>,
    capacity: usize,
    write_idx: usize,
    len: usize,
}

impl MetricsBuffer {
    /// Create a new metrics buffer with given capacity.
    pub fn new(capacity: usize) -> Self {
        Self { data: vec![0.0; capacity], capacity, write_idx: 0, len: 0 }
    }

    /// Push a new value, overwriting oldest if full.
    pub fn push(&mut self, value: f32) {
        self.data[self.write_idx] = value;
        self.write_idx = (self.write_idx + 1) % self.capacity;
        if self.len < self.capacity {
            self.len += 1;
        }
    }

    /// Get the number of values in the buffer.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get the capacity of the buffer.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get the last N values in chronological order.
    pub fn last_n(&self, n: usize) -> Vec<f32> {
        let n = n.min(self.len);
        if n == 0 {
            return Vec::new();
        }

        let mut result = Vec::with_capacity(n);
        let start_idx = if self.len == self.capacity {
            (self.write_idx + self.capacity - n) % self.capacity
        } else {
            self.len.saturating_sub(n)
        };

        for i in 0..n {
            let idx = (start_idx + i) % self.capacity;
            result.push(self.data[idx]);
        }
        result
    }

    /// Get all values in chronological order.
    pub fn values(&self) -> Vec<f32> {
        self.last_n(self.len)
    }

    /// Get the most recent value.
    pub fn last(&self) -> Option<f32> {
        if self.len == 0 {
            None
        } else {
            let idx = (self.write_idx + self.capacity - 1) % self.capacity;
            Some(self.data[idx])
        }
    }

    /// Get min value.
    pub fn min(&self) -> Option<f32> {
        if self.len == 0 {
            return None;
        }
        self.values().into_iter().reduce(f32::min)
    }

    /// Get max value.
    pub fn max(&self) -> Option<f32> {
        if self.len == 0 {
            return None;
        }
        self.values().into_iter().reduce(f32::max)
    }

    /// Get mean value.
    pub fn mean(&self) -> Option<f32> {
        if self.len == 0 {
            return None;
        }
        let sum: f32 = self.values().iter().sum();
        Some(sum / self.len as f32)
    }

    /// Clear all values.
    pub fn clear(&mut self) {
        self.write_idx = 0;
        self.len = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_buffer_new() {
        let buf = MetricsBuffer::new(10);
        assert_eq!(buf.capacity(), 10);
        assert_eq!(buf.len(), 0);
        assert!(buf.is_empty());
    }

    #[test]
    fn test_metrics_buffer_push() {
        let mut buf = MetricsBuffer::new(5);
        buf.push(1.0);
        buf.push(2.0);
        buf.push(3.0);

        assert_eq!(buf.len(), 3);
        assert_eq!(buf.values(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_metrics_buffer_wraparound() {
        let mut buf = MetricsBuffer::new(3);
        buf.push(1.0);
        buf.push(2.0);
        buf.push(3.0);
        buf.push(4.0); // Overwrites 1.0
        buf.push(5.0); // Overwrites 2.0

        assert_eq!(buf.len(), 3);
        assert_eq!(buf.values(), vec![3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_metrics_buffer_last_n() {
        let mut buf = MetricsBuffer::new(10);
        for i in 0..10 {
            buf.push(i as f32);
        }

        assert_eq!(buf.last_n(3), vec![7.0, 8.0, 9.0]);
        assert_eq!(buf.last_n(1), vec![9.0]);
        assert_eq!(buf.last_n(0), Vec::<f32>::new());
    }

    #[test]
    fn test_metrics_buffer_last() {
        let mut buf = MetricsBuffer::new(5);
        assert_eq!(buf.last(), None);

        buf.push(1.0);
        assert_eq!(buf.last(), Some(1.0));

        buf.push(2.0);
        assert_eq!(buf.last(), Some(2.0));
    }

    #[test]
    fn test_metrics_buffer_min_max_mean() {
        let mut buf = MetricsBuffer::new(10);
        assert_eq!(buf.min(), None);
        assert_eq!(buf.max(), None);
        assert_eq!(buf.mean(), None);

        buf.push(1.0);
        buf.push(5.0);
        buf.push(3.0);

        assert_eq!(buf.min(), Some(1.0));
        assert_eq!(buf.max(), Some(5.0));
        assert_eq!(buf.mean(), Some(3.0));
    }

    #[test]
    fn test_metrics_buffer_clear() {
        let mut buf = MetricsBuffer::new(5);
        buf.push(1.0);
        buf.push(2.0);
        buf.clear();

        assert!(buf.is_empty());
        assert_eq!(buf.len(), 0);
    }

    #[test]
    fn test_metrics_buffer_last_n_wraparound() {
        let mut buf = MetricsBuffer::new(4);
        for i in 0..6 {
            buf.push(i as f32);
        }
        // Buffer contains [4, 5, 2, 3] with write_idx at 2
        // Chronological order: 2, 3, 4, 5
        assert_eq!(buf.last_n(2), vec![4.0, 5.0]);
        assert_eq!(buf.last_n(4), vec![2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_metrics_buffer_last_n_more_than_len() {
        let mut buf = MetricsBuffer::new(10);
        buf.push(1.0);
        buf.push(2.0);

        assert_eq!(buf.last_n(5), vec![1.0, 2.0]);
    }
}
