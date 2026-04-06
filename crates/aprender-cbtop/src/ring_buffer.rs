//! SIMD-optimized ring buffer for time-series data
//!
//! # Design
//!
//! - Bounded capacity (Muda: no unbounded growth)
//! - Zero-copy iteration
//! - Cache-friendly layout
//!
//! # Reference
//!
//! Ohno, T. (1988). "Toyota Production System" - Waste elimination

use std::collections::VecDeque;

/// Ring buffer with fixed capacity
#[derive(Debug, Clone)]
pub struct RingBuffer<T> {
    data: VecDeque<T>,
    capacity: usize,
}

impl<T> RingBuffer<T> {
    /// Create new ring buffer with specified capacity
    pub fn new(capacity: usize) -> Self {
        Self { data: VecDeque::with_capacity(capacity), capacity }
    }

    /// Push value, evicting oldest if at capacity
    pub fn push(&mut self, value: T) {
        if self.data.len() >= self.capacity {
            self.data.pop_front();
        }
        self.data.push_back(value);
    }

    /// Get number of elements
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get capacity
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get latest value
    pub fn back(&self) -> Option<&T> {
        self.data.back()
    }

    /// Get oldest value
    pub fn front(&self) -> Option<&T> {
        self.data.front()
    }

    /// Clear all data
    pub fn clear(&mut self) {
        self.data.clear();
    }

    /// Iterate from oldest to newest
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.data.iter()
    }

    /// Get value at index (0 = oldest)
    pub fn get(&self, index: usize) -> Option<&T> {
        self.data.get(index)
    }

    /// Get last N values as slice (newest last)
    pub fn last_n(&self, n: usize) -> impl Iterator<Item = &T> {
        let skip = self.data.len().saturating_sub(n);
        self.data.iter().skip(skip)
    }
}

impl<T: Clone> RingBuffer<T> {
    /// Get all values as a Vec (oldest first)
    pub fn to_vec(&self) -> Vec<T> {
        self.data.iter().cloned().collect()
    }
}

impl<T: Copy + Default> RingBuffer<T> {
    /// Get values as contiguous slice for SIMD operations
    /// Returns (slice1, slice2) where slice2 may be empty
    pub fn as_slices(&self) -> (&[T], &[T]) {
        self.data.as_slices()
    }
}

impl<T> Default for RingBuffer<T> {
    fn default() -> Self {
        Self::new(64)
    }
}

/// Statistics over ring buffer (SIMD-friendly when possible)
impl RingBuffer<f64> {
    /// Calculate mean of all values
    pub fn mean(&self) -> f64 {
        if self.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.data.iter().sum();
        sum / self.data.len() as f64
    }

    /// Calculate min value
    pub fn min(&self) -> f64 {
        self.data.iter().copied().fold(f64::INFINITY, f64::min)
    }

    /// Calculate max value
    pub fn max(&self) -> f64 {
        self.data.iter().copied().fold(f64::NEG_INFINITY, f64::max)
    }

    /// Calculate percentile (0.0 - 1.0)
    pub fn percentile(&self, p: f64) -> f64 {
        if self.is_empty() {
            return 0.0;
        }
        let mut sorted: Vec<f64> = self.data.iter().copied().collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let idx = ((sorted.len() - 1) as f64 * p.clamp(0.0, 1.0)) as usize;
        sorted[idx]
    }

    /// Calculate standard deviation (delegates to batuta-common).
    pub fn std_dev(&self) -> f64 {
        let data: Vec<f64> = self.data.iter().copied().collect();
        batuta_common::math::std_dev(&data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ring_buffer_push() {
        let mut rb = RingBuffer::new(3);
        rb.push(1);
        rb.push(2);
        rb.push(3);
        assert_eq!(rb.len(), 3);

        rb.push(4);
        assert_eq!(rb.len(), 3);
        assert_eq!(rb.front(), Some(&2));
        assert_eq!(rb.back(), Some(&4));
    }

    #[test]
    fn test_ring_buffer_iter() {
        let mut rb = RingBuffer::new(3);
        rb.push(1);
        rb.push(2);
        rb.push(3);

        let values: Vec<_> = rb.iter().copied().collect();
        assert_eq!(values, vec![1, 2, 3]);
    }

    #[test]
    fn test_ring_buffer_last_n() {
        let mut rb = RingBuffer::new(5);
        for i in 1..=5 {
            rb.push(i);
        }

        let last3: Vec<_> = rb.last_n(3).copied().collect();
        assert_eq!(last3, vec![3, 4, 5]);
    }

    #[test]
    fn test_ring_buffer_statistics() {
        let mut rb: RingBuffer<f64> = RingBuffer::new(5);
        rb.push(1.0);
        rb.push(2.0);
        rb.push(3.0);
        rb.push(4.0);
        rb.push(5.0);

        assert_eq!(rb.mean(), 3.0);
        assert_eq!(rb.min(), 1.0);
        assert_eq!(rb.max(), 5.0);
        assert_eq!(rb.percentile(0.5), 3.0);
    }

    #[test]
    fn test_ring_buffer_empty() {
        let rb: RingBuffer<f64> = RingBuffer::new(5);
        assert!(rb.is_empty());
        assert_eq!(rb.mean(), 0.0);
        assert_eq!(rb.percentile(0.5), 0.0);
    }
}
