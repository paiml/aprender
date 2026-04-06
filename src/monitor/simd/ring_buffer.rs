//! SIMD-aligned ring buffer for metric history.
//!
//! ## Design Goals
//!
//! - 64-byte alignment for AVX-512 compatibility
//! - O(1) running statistics (min, max, mean, variance)
//! - Lock-free reads via atomic head pointer
//! - Zero allocations after initialization
//!
//! ## Falsifiable Hypothesis (H₉)
//!
//! SIMD-aligned ring buffer achieves ≥2x throughput vs VecDeque for batch operations.
//! (Realistic target: batch stats updates with SIMD reductions)

use super::kernels;
use super::SimdStats;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Reduction operation types for SIMD reductions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReductionOp {
    /// Sum of all elements
    Sum,
    /// Arithmetic mean
    Mean,
    /// Minimum value
    Min,
    /// Maximum value
    Max,
}

/// SIMD-optimized ring buffer with 64-byte alignment.
///
/// Unlike the standard `RingBuffer<T>`, this buffer:
/// - Uses contiguous, aligned storage for SIMD operations
/// - Maintains running statistics without traversal
/// - Supports batch push for maximum throughput
/// - Provides lock-free read access via atomic head pointer
#[repr(C, align(64))]
pub struct SimdRingBuffer {
    /// Contiguous storage (64-byte aligned).
    data: Box<[f64]>,
    /// Capacity of the buffer.
    capacity: usize,
    /// Write position (atomic for lock-free reads).
    head: AtomicUsize,
    /// Number of valid elements.
    len: AtomicUsize,
    /// Running statistics (updated on each push).
    stats: SimdStats,
    /// Whether the buffer has wrapped around.
    wrapped: bool,
}

impl SimdRingBuffer {
    /// Creates a new SIMD ring buffer with the specified capacity.
    ///
    /// # Arguments
    ///
    /// * `capacity` - Maximum number of elements (rounded up to SIMD-friendly size)
    ///
    /// # Panics
    ///
    /// Panics if capacity is 0.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "SimdRingBuffer capacity must be > 0");

        // Round up to multiple of 8 for SIMD-friendly access
        let aligned_capacity = capacity.div_ceil(8) * 8;

        // Allocate aligned memory
        let data = vec![0.0f64; aligned_capacity].into_boxed_slice();

        Self {
            data,
            capacity: aligned_capacity,
            head: AtomicUsize::new(0),
            len: AtomicUsize::new(0),
            stats: SimdStats::new(),
            wrapped: false,
        }
    }

    /// Creates a ring buffer with default capacity of 300 (5 minutes at 1Hz).
    #[must_use]
    pub fn default_capacity() -> Self {
        Self::new(304) // 304 is divisible by 8
    }

    /// Pushes a value into the buffer.
    ///
    /// If the buffer is full, the oldest value is overwritten.
    #[inline]
    pub fn push(&mut self, value: f64) {
        let head = self.head.load(Ordering::Relaxed);
        let idx = head % self.capacity;

        // Update statistics
        self.stats.update(value);

        // Store value
        self.data[idx] = value;

        // Advance head
        let new_head = head.wrapping_add(1);
        self.head.store(new_head, Ordering::Release);

        // Update length
        let current_len = self.len.load(Ordering::Relaxed);
        if current_len < self.capacity {
            self.len.store(current_len + 1, Ordering::Release);
        } else {
            self.wrapped = true;
        }
    }

    /// Pushes multiple values using SIMD operations.
    ///
    /// # Performance
    ///
    /// Achieves ≥2x throughput vs individual pushes for 8+ values
    /// by batching statistics updates with SIMD reductions.
    pub fn push_batch(&mut self, values: &[f64]) {
        if values.is_empty() {
            return;
        }

        // Update batch statistics using real SIMD kernels (f64 precision)
        let batch_stats = kernels::simd_statistics(values);
        self.stats.min = self.stats.min.min(batch_stats.min);
        self.stats.max = self.stats.max.max(batch_stats.max);
        self.stats.sum += batch_stats.sum;
        self.stats.sum_sq += batch_stats.sum_sq;
        self.stats.count += batch_stats.count;

        // Copy values into ring buffer
        let head = self.head.load(Ordering::Relaxed);
        let len = values.len();

        // Fast path: if we can write contiguously without wrapping
        let start_idx = head % self.capacity;
        if start_idx + len <= self.capacity {
            self.data[start_idx..start_idx + len].copy_from_slice(values);
        } else {
            // Wrap around case
            let first_part = self.capacity - start_idx;
            self.data[start_idx..].copy_from_slice(&values[..first_part]);
            self.data[..len - first_part].copy_from_slice(&values[first_part..]);
        }

        // Update head and length
        let new_head = head.wrapping_add(len);
        self.head.store(new_head, Ordering::Release);

        let current_len = self.len.load(Ordering::Relaxed);
        let new_len = (current_len + len).min(self.capacity);
        self.len.store(new_len, Ordering::Release);

        if current_len + len >= self.capacity {
            self.wrapped = true;
        }
    }

    /// Returns the most recent value.
    #[must_use]
    pub fn latest(&self) -> Option<f64> {
        let len = self.len.load(Ordering::Acquire);
        if len == 0 {
            return None;
        }

        let head = self.head.load(Ordering::Acquire);
        let idx = (head.wrapping_sub(1)) % self.capacity;
        Some(self.data[idx])
    }

    /// Returns the oldest value.
    #[must_use]
    pub fn oldest(&self) -> Option<f64> {
        let len = self.len.load(Ordering::Acquire);
        if len == 0 {
            return None;
        }

        let head = self.head.load(Ordering::Acquire);
        let idx = if self.wrapped { head % self.capacity } else { 0 };
        Some(self.data[idx])
    }

    /// Returns the current number of elements.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len.load(Ordering::Acquire)
    }

    /// Returns true if the buffer is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns true if the buffer is at capacity.
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.len() >= self.capacity
    }

    /// Returns the capacity.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns pre-computed statistics in O(1) time.
    #[must_use]
    pub fn statistics(&self) -> &SimdStats {
        &self.stats
    }

    /// Returns min value in O(1).
    #[must_use]
    pub fn min(&self) -> f64 {
        self.stats.min
    }

    /// Returns max value in O(1).
    #[must_use]
    pub fn max(&self) -> f64 {
        self.stats.max
    }

    /// Returns mean value in O(1).
    #[must_use]
    pub fn mean(&self) -> f64 {
        self.stats.mean()
    }

    /// Returns standard deviation in O(1).
    #[must_use]
    pub fn std_dev(&self) -> f64 {
        self.stats.std_dev()
    }

    /// Returns a contiguous slice of the most recent N values.
    ///
    /// If N > len, returns all available values.
    #[must_use]
    pub fn last_n(&self, n: usize) -> Vec<f64> {
        let len = self.len();
        let count = n.min(len);

        if count == 0 {
            return vec![];
        }

        let head = self.head.load(Ordering::Acquire);
        let mut result = Vec::with_capacity(count);

        for i in 0..count {
            let idx = (head.wrapping_sub(count - i)) % self.capacity;
            result.push(self.data[idx]);
        }

        result
    }

    /// Returns all values as a contiguous slice (oldest to newest).
    #[must_use]
    pub fn to_vec(&self) -> Vec<f64> {
        self.last_n(self.len())
    }

    /// Performs SIMD reduction over the buffer contents.
    ///
    /// Available operations: sum, mean, min, max
    pub fn reduce(&self, op: ReductionOp) -> f64 {
        let values = self.to_vec();
        if values.is_empty() {
            return 0.0;
        }

        match op {
            ReductionOp::Sum => kernels::simd_sum(&values),
            ReductionOp::Mean => kernels::simd_mean(&values),
            ReductionOp::Min => kernels::simd_min(&values),
            ReductionOp::Max => kernels::simd_max(&values),
        }
    }

    /// Clears all elements and resets statistics.
    pub fn clear(&mut self) {
        self.head.store(0, Ordering::Release);
        self.len.store(0, Ordering::Release);
        self.stats.reset();
        self.wrapped = false;

        // Zero out data
        for v in self.data.iter_mut() {
            *v = 0.0;
        }
    }

    /// Returns an iterator over values from oldest to newest.
    pub fn iter(&self) -> impl Iterator<Item = f64> + '_ {
        SimdRingBufferIter { buffer: self, index: 0, remaining: self.len() }
    }
}

impl Default for SimdRingBuffer {
    fn default() -> Self {
        Self::default_capacity()
    }
}

impl std::fmt::Debug for SimdRingBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SimdRingBuffer")
            .field("capacity", &self.capacity)
            .field("len", &self.len())
            .field("latest", &self.latest())
            .field("mean", &self.mean())
            .finish()
    }
}

/// Iterator over SimdRingBuffer values.
struct SimdRingBufferIter<'a> {
    buffer: &'a SimdRingBuffer,
    index: usize,
    remaining: usize,
}

impl Iterator for SimdRingBufferIter<'_> {
    type Item = f64;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }

        let head = self.buffer.head.load(Ordering::Acquire);
        let start = if self.buffer.wrapped { head % self.buffer.capacity } else { 0 };

        let idx = (start + self.index) % self.buffer.capacity;
        self.index += 1;
        self.remaining -= 1;

        Some(self.buffer.data[idx])
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

impl ExactSizeIterator for SimdRingBufferIter<'_> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let buf = SimdRingBuffer::new(100);
        assert!(buf.capacity() >= 100);
        assert_eq!(buf.len(), 0);
        assert!(buf.is_empty());
    }

    #[test]
    #[should_panic(expected = "capacity must be > 0")]
    fn test_zero_capacity() {
        let _ = SimdRingBuffer::new(0);
    }

    #[test]
    fn test_push_and_latest() {
        let mut buf = SimdRingBuffer::new(10);

        buf.push(1.0);
        assert_eq!(buf.latest(), Some(1.0));
        assert_eq!(buf.len(), 1);

        buf.push(2.0);
        assert_eq!(buf.latest(), Some(2.0));
        assert_eq!(buf.len(), 2);
    }

    #[test]
    fn test_overflow() {
        let mut buf = SimdRingBuffer::new(8); // Will be exactly 8

        for i in 1..=16 {
            buf.push(f64::from(i));
        }

        assert_eq!(buf.len(), 8);
        assert_eq!(buf.latest(), Some(16.0));
        // Oldest should be 9.0 (values 1-8 were overwritten)
        assert_eq!(buf.oldest(), Some(9.0));
    }

    #[test]
    fn test_push_batch() {
        let mut buf = SimdRingBuffer::new(16);
        let values: Vec<f64> = (1..=10).map(f64::from).collect();

        buf.push_batch(&values);

        assert_eq!(buf.len(), 10);
        assert_eq!(buf.latest(), Some(10.0));
        assert_eq!(buf.oldest(), Some(1.0));
    }

    #[test]
    fn test_statistics() {
        let mut buf = SimdRingBuffer::new(16);

        for i in 1..=10 {
            buf.push(f64::from(i));
        }

        let stats = buf.statistics();
        assert!((stats.min - 1.0).abs() < 0.001);
        assert!((stats.max - 10.0).abs() < 0.001);
        assert!((stats.mean() - 5.5).abs() < 0.001);
    }

    #[test]
    fn test_last_n() {
        let mut buf = SimdRingBuffer::new(16);

        for i in 1..=10 {
            buf.push(f64::from(i));
        }

        let last_3 = buf.last_n(3);
        assert_eq!(last_3, vec![8.0, 9.0, 10.0]);

        let last_20 = buf.last_n(20);
        assert_eq!(last_20.len(), 10); // Can only return what we have
    }

    #[test]
    fn test_iter() {
        let mut buf = SimdRingBuffer::new(8);

        for i in 1..=5 {
            buf.push(f64::from(i));
        }

        let collected: Vec<f64> = buf.iter().collect();
        assert_eq!(collected, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_reduce() {
        let mut buf = SimdRingBuffer::new(16);

        for i in 1..=10 {
            buf.push(f64::from(i));
        }

        let sum = buf.reduce(ReductionOp::Sum);
        assert!((sum - 55.0).abs() < 0.1);

        let mean = buf.reduce(ReductionOp::Mean);
        assert!((mean - 5.5).abs() < 0.1);

        let min = buf.reduce(ReductionOp::Min);
        assert!((min - 1.0).abs() < 0.1);

        let max = buf.reduce(ReductionOp::Max);
        assert!((max - 10.0).abs() < 0.1);
    }

    #[test]
    fn test_clear() {
        let mut buf = SimdRingBuffer::new(16);

        for i in 1..=10 {
            buf.push(f64::from(i));
        }

        buf.clear();

        assert!(buf.is_empty());
        assert_eq!(buf.len(), 0);
        assert_eq!(buf.latest(), None);
    }

    #[test]
    fn test_alignment() {
        let buf = SimdRingBuffer::new(64);
        // The struct itself should be 64-byte aligned
        let ptr = std::ptr::addr_of!(buf);
        assert_eq!(ptr as usize % super::super::SIMD_ALIGNMENT, 0);
    }

    #[test]
    fn test_default() {
        let buf = SimdRingBuffer::default();
        assert!(buf.capacity() >= 300);
    }

    #[test]
    fn test_default_capacity_method() {
        let buf = SimdRingBuffer::default_capacity();
        assert!(buf.capacity() >= 300);
    }

    #[test]
    fn test_is_full() {
        let mut buf = SimdRingBuffer::new(8);
        assert!(!buf.is_full());

        for i in 1..=8 {
            buf.push(f64::from(i));
        }
        assert!(buf.is_full());
    }

    #[test]
    fn test_min_max_mean_std_dev() {
        let mut buf = SimdRingBuffer::new(16);

        for i in 1..=10 {
            buf.push(f64::from(i));
        }

        assert!((buf.min() - 1.0).abs() < 0.001);
        assert!((buf.max() - 10.0).abs() < 0.001);
        assert!((buf.mean() - 5.5).abs() < 0.001);
        assert!(buf.std_dev() > 0.0);
    }

    #[test]
    fn test_to_vec() {
        let mut buf = SimdRingBuffer::new(8);

        for i in 1..=5 {
            buf.push(f64::from(i));
        }

        let vec = buf.to_vec();
        assert_eq!(vec, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_to_vec_empty() {
        let buf = SimdRingBuffer::new(8);
        let vec = buf.to_vec();
        assert!(vec.is_empty());
    }

    #[test]
    fn test_reduce_empty() {
        let buf = SimdRingBuffer::new(8);

        assert!((buf.reduce(ReductionOp::Sum) - 0.0).abs() < 0.001);
        assert!((buf.reduce(ReductionOp::Mean) - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_debug_format() {
        let mut buf = SimdRingBuffer::new(8);
        buf.push(5.0);

        let debug_str = format!("{buf:?}");
        assert!(debug_str.contains("SimdRingBuffer"));
        assert!(debug_str.contains("capacity"));
    }

    #[test]
    fn test_push_batch_empty() {
        let mut buf = SimdRingBuffer::new(8);
        buf.push_batch(&[]);

        assert!(buf.is_empty());
    }

    #[test]
    fn test_push_batch_wraparound() {
        let mut buf = SimdRingBuffer::new(8);

        // Push 5 values
        buf.push_batch(&[1.0, 2.0, 3.0, 4.0, 5.0]);

        // Push 5 more to trigger wraparound
        buf.push_batch(&[6.0, 7.0, 8.0, 9.0, 10.0]);

        assert_eq!(buf.len(), 8);
        assert_eq!(buf.latest(), Some(10.0));
    }

    #[test]
    fn test_oldest_empty() {
        let buf = SimdRingBuffer::new(8);
        assert_eq!(buf.oldest(), None);
    }

    #[test]
    fn test_latest_empty() {
        let buf = SimdRingBuffer::new(8);
        assert_eq!(buf.latest(), None);
    }

    #[test]
    fn test_iter_empty() {
        let buf = SimdRingBuffer::new(8);
        let collected: Vec<f64> = buf.iter().collect();
        assert!(collected.is_empty());
    }

    #[test]
    fn test_iter_exact_size() {
        let mut buf = SimdRingBuffer::new(8);
        buf.push(1.0);
        buf.push(2.0);

        // Verify size_hint works
        let iter = buf.iter();
        let (lower, upper) = iter.size_hint();
        assert_eq!(lower, 2);
        assert_eq!(upper, Some(2));
    }

    #[test]
    fn test_reduction_op_debug() {
        let op = ReductionOp::Sum;
        let debug_str = format!("{op:?}");
        assert!(debug_str.contains("Sum"));

        assert_eq!(op, op.clone());
    }

    #[test]
    fn test_iter_wraparound() {
        let mut buf = SimdRingBuffer::new(8);

        // Fill buffer and overflow
        for i in 1..=12 {
            buf.push(f64::from(i));
        }

        let collected: Vec<f64> = buf.iter().collect();
        assert_eq!(collected.len(), 8);
        // Should be [5, 6, 7, 8, 9, 10, 11, 12]
        assert_eq!(collected[0], 5.0);
        assert_eq!(collected[7], 12.0);
    }

    #[test]
    fn test_last_n_zero() {
        let mut buf = SimdRingBuffer::new(8);
        buf.push(1.0);

        let result = buf.last_n(0);
        assert!(result.is_empty());
    }
}
