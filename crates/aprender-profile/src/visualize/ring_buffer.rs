//! SIMD-accelerated ring buffer for history tracking (trueno-viz pattern)
//!
//! Fixed-capacity circular buffer for storing metric history.
//! Uses trueno-viz SIMD kernels for accelerated statistics.
//!
//! # Performance Characteristics
//!
//! - Push: O(1) - constant time insertion
//! - Statistics: O(n) with AVX2/NEON SIMD acceleration (>4x vs scalar)
//! - Iteration: O(n) - oldest to newest order
//! - Memory: Fixed at initialization (no reallocation)
//!
//! # SIMD Acceleration
//!
//! Uses trueno-viz `monitor::simd::kernels` for vectorized:
//! - Sum/Mean: AVX2 horizontal reduction
//! - Min/Max: AVX2 parallel comparison
//! - Statistics: Combined pass for all metrics
//!
//! # Toyota Way Principle: Muda (Waste Elimination)
//!
//! Pre-allocated fixed buffer eliminates allocation during hot path.
//! SIMD acceleration eliminates unnecessary scalar loop iterations.

/// Fixed-capacity ring buffer for metric history
///
/// # Example
///
/// ```
/// use renacer::visualize::ring_buffer::HistoryBuffer;
///
/// let mut buf = HistoryBuffer::new(5);
/// buf.push(1.0);
/// buf.push(2.0);
/// buf.push(3.0);
///
/// // Iterate oldest to newest
/// let values: Vec<f64> = buf.iter().cloned().collect();
/// assert_eq!(values, vec![1.0, 2.0, 3.0]);
///
/// // Push more than capacity
/// buf.push(4.0);
/// buf.push(5.0);
/// buf.push(6.0); // Evicts 1.0
///
/// let values: Vec<f64> = buf.iter().cloned().collect();
/// assert_eq!(values, vec![2.0, 3.0, 4.0, 5.0, 6.0]);
/// ```
#[derive(Debug, Clone)]
pub struct HistoryBuffer<T> {
    data: Vec<T>,
    capacity: usize,
    head: usize,
    len: usize,
}

impl<T: Default + Clone> HistoryBuffer<T> {
    /// Create a new history buffer with the given capacity
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "Capacity must be positive");
        Self { data: vec![T::default(); capacity], capacity, head: 0, len: 0 }
    }

    /// Push a value, evicting the oldest if at capacity
    ///
    /// This is O(1) - no allocation occurs.
    #[inline]
    pub fn push(&mut self, value: T) {
        self.data[self.head] = value;
        self.head = (self.head + 1) % self.capacity;
        self.len = (self.len + 1).min(self.capacity);
    }

    /// Get the number of elements currently in the buffer
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if the buffer is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get the capacity of the buffer
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Check if the buffer is full
    #[inline]
    pub fn is_full(&self) -> bool {
        self.len == self.capacity
    }

    /// Clear all elements
    pub fn clear(&mut self) {
        self.head = 0;
        self.len = 0;
    }

    /// Get the most recent value (last pushed)
    pub fn last(&self) -> Option<&T> {
        if self.len == 0 {
            None
        } else {
            let idx = if self.head == 0 { self.capacity - 1 } else { self.head - 1 };
            Some(&self.data[idx])
        }
    }

    /// Get the oldest value (first to be evicted)
    pub fn first(&self) -> Option<&T> {
        if self.len == 0 {
            None
        } else {
            let start = if self.len < self.capacity { 0 } else { self.head };
            Some(&self.data[start])
        }
    }

    /// Iterate from oldest to newest
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        let start = if self.len < self.capacity { 0 } else { self.head };
        (0..self.len).map(move |i| &self.data[(start + i) % self.capacity])
    }

    /// Get values as a slice (may not be contiguous if wrapped)
    /// Returns oldest to newest order
    pub fn to_vec(&self) -> Vec<T>
    where
        T: Copy,
    {
        self.iter().copied().collect()
    }
}

impl<T: Default + Clone + Copy> HistoryBuffer<T> {
    /// Fill the buffer with a value
    pub fn fill(&mut self, value: T) {
        for item in &mut self.data {
            *item = value;
        }
        self.len = self.capacity;
        self.head = 0;
    }
}

impl HistoryBuffer<f64> {
    /// Get a slice of all current values directly from internal storage.
    ///
    /// For order-independent SIMD operations (sum, mean, min, max), the data
    /// doesn't need to be in temporal order - all current values are stored
    /// contiguously in `data[0..len]` regardless of wrap state.
    #[inline]
    fn as_slice(&self) -> &[f64] {
        &self.data[0..self.len]
    }

    /// Calculate the average of all values using SIMD acceleration
    ///
    /// Uses trueno-viz `simd_mean` for AVX2/NEON vectorized computation.
    /// Zero-allocation: operates directly on internal storage.
    #[inline]
    pub fn avg(&self) -> f64 {
        if self.len == 0 {
            return 0.0;
        }
        trueno_viz::monitor::simd::kernels::simd_mean(self.as_slice())
    }

    /// Calculate the min value using SIMD acceleration
    ///
    /// Uses trueno-viz `simd_min` for AVX2/NEON parallel comparison.
    /// Zero-allocation: operates directly on internal storage.
    #[inline]
    pub fn min(&self) -> f64 {
        if self.len == 0 {
            return f64::INFINITY;
        }
        trueno_viz::monitor::simd::kernels::simd_min(self.as_slice())
    }

    /// Calculate the max value using SIMD acceleration
    ///
    /// Uses trueno-viz `simd_max` for AVX2/NEON parallel comparison.
    /// Zero-allocation: operates directly on internal storage.
    #[inline]
    pub fn max(&self) -> f64 {
        if self.len == 0 {
            return f64::NEG_INFINITY;
        }
        trueno_viz::monitor::simd::kernels::simd_max(self.as_slice())
    }

    /// Calculate sum using SIMD acceleration
    ///
    /// Uses trueno-viz `simd_sum` for AVX2/NEON horizontal reduction.
    /// Zero-allocation: operates directly on internal storage.
    #[inline]
    pub fn sum(&self) -> f64 {
        if self.len == 0 {
            return 0.0;
        }
        trueno_viz::monitor::simd::kernels::simd_sum(self.as_slice())
    }

    /// Calculate standard deviation using SIMD-accelerated mean
    #[inline]
    pub fn stddev(&self) -> f64 {
        if self.len < 2 {
            return 0.0;
        }
        let mean = self.avg();
        let variance =
            self.as_slice().iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (self.len - 1) as f64;
        variance.sqrt()
    }

    /// Get all statistics in a single SIMD pass
    ///
    /// Uses trueno-viz `simd_statistics` for combined min/max/mean computation.
    /// Zero-allocation: operates directly on internal storage.
    /// Returns (min, max, mean) tuple.
    #[inline]
    pub fn stats(&self) -> (f64, f64, f64) {
        if self.len == 0 {
            return (f64::INFINITY, f64::NEG_INFINITY, 0.0);
        }
        let stats = trueno_viz::monitor::simd::kernels::simd_statistics(self.as_slice());
        (stats.min, stats.max, stats.mean())
    }

    /// Get latest N values as a vec
    pub fn latest(&self, n: usize) -> Vec<f64> {
        let n = n.min(self.len);
        if n == 0 {
            return Vec::new();
        }
        let start = if self.len < self.capacity {
            self.len.saturating_sub(n)
        } else {
            (self.head + self.capacity - n) % self.capacity
        };
        (0..n).map(|i| self.data[(start + i) % self.capacity]).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let buf: HistoryBuffer<f64> = HistoryBuffer::new(10);
        assert_eq!(buf.capacity(), 10);
        assert_eq!(buf.len(), 0);
        assert!(buf.is_empty());
    }

    #[test]
    #[should_panic(expected = "Capacity must be positive")]
    fn test_new_zero_capacity() {
        let _: HistoryBuffer<f64> = HistoryBuffer::new(0);
    }

    #[test]
    fn test_push_and_len() {
        let mut buf = HistoryBuffer::new(5);
        assert_eq!(buf.len(), 0);

        buf.push(1.0);
        assert_eq!(buf.len(), 1);

        buf.push(2.0);
        buf.push(3.0);
        assert_eq!(buf.len(), 3);
    }

    #[test]
    fn test_push_wrap() {
        let mut buf = HistoryBuffer::new(3);
        buf.push(1.0);
        buf.push(2.0);
        buf.push(3.0);
        assert_eq!(buf.len(), 3);
        assert!(buf.is_full());

        // Push fourth element, should evict first
        buf.push(4.0);
        assert_eq!(buf.len(), 3);

        let values: Vec<f64> = buf.iter().copied().collect();
        assert_eq!(values, vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_iter_order() {
        let mut buf = HistoryBuffer::new(5);
        for i in 1..=5 {
            buf.push(i as f64);
        }

        let values: Vec<f64> = buf.iter().copied().collect();
        assert_eq!(values, vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        // Wrap around
        buf.push(6.0);
        buf.push(7.0);

        let values: Vec<f64> = buf.iter().copied().collect();
        assert_eq!(values, vec![3.0, 4.0, 5.0, 6.0, 7.0]);
    }

    #[test]
    fn test_last_and_first() {
        let mut buf = HistoryBuffer::new(3);
        assert!(buf.last().is_none());
        assert!(buf.first().is_none());

        buf.push(1.0);
        assert_eq!(*buf.last().unwrap(), 1.0);
        assert_eq!(*buf.first().unwrap(), 1.0);

        buf.push(2.0);
        buf.push(3.0);
        assert_eq!(*buf.last().unwrap(), 3.0);
        assert_eq!(*buf.first().unwrap(), 1.0);

        buf.push(4.0); // Evicts 1.0
        assert_eq!(*buf.last().unwrap(), 4.0);
        assert_eq!(*buf.first().unwrap(), 2.0);
    }

    #[test]
    fn test_clear() {
        let mut buf = HistoryBuffer::new(5);
        buf.push(1.0);
        buf.push(2.0);
        buf.clear();
        assert!(buf.is_empty());
        assert_eq!(buf.len(), 0);
    }

    #[test]
    fn test_avg() {
        let mut buf = HistoryBuffer::new(5);
        buf.push(1.0);
        buf.push(2.0);
        buf.push(3.0);
        buf.push(4.0);
        buf.push(5.0);
        assert!((buf.avg() - 3.0).abs() < 0.001);
    }

    #[test]
    fn test_min_max() {
        let mut buf = HistoryBuffer::new(5);
        buf.push(3.0);
        buf.push(1.0);
        buf.push(5.0);
        buf.push(2.0);
        buf.push(4.0);
        assert!((buf.min() - 1.0).abs() < 0.001);
        assert!((buf.max() - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_stddev() {
        let mut buf = HistoryBuffer::new(5);
        // Standard deviation of [1, 2, 3, 4, 5] = sqrt(2.5) ≈ 1.58
        buf.push(1.0);
        buf.push(2.0);
        buf.push(3.0);
        buf.push(4.0);
        buf.push(5.0);
        assert!((buf.stddev() - 1.5811).abs() < 0.001);
    }

    #[test]
    fn test_sum() {
        let mut buf = HistoryBuffer::new(5);
        buf.push(1.0);
        buf.push(2.0);
        buf.push(3.0);
        buf.push(4.0);
        buf.push(5.0);
        assert!((buf.sum() - 15.0).abs() < 0.001);
    }

    #[test]
    fn test_stats_simd() {
        let mut buf = HistoryBuffer::new(5);
        buf.push(3.0);
        buf.push(1.0);
        buf.push(5.0);
        buf.push(2.0);
        buf.push(4.0);

        let (min, max, mean) = buf.stats();
        assert!((min - 1.0).abs() < 0.001);
        assert!((max - 5.0).abs() < 0.001);
        assert!((mean - 3.0).abs() < 0.001);
    }

    #[test]
    fn test_latest() {
        let mut buf = HistoryBuffer::new(5);
        for i in 1..=5 {
            buf.push(i as f64);
        }

        assert_eq!(buf.latest(3), vec![3.0, 4.0, 5.0]);
        assert_eq!(buf.latest(1), vec![5.0]);
        assert_eq!(buf.latest(10), vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_fill() {
        let mut buf = HistoryBuffer::new(5);
        buf.fill(1.0);
        assert_eq!(buf.len(), 5);
        assert!(buf.is_full());
        assert_eq!(buf.to_vec(), vec![1.0, 1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_clone() {
        let mut buf = HistoryBuffer::new(3);
        buf.push(1.0);
        buf.push(2.0);

        let cloned = buf.clone();
        assert_eq!(cloned.to_vec(), vec![1.0, 2.0]);
    }
}
