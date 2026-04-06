//! AWP-13: Buffer Reserve Strategy

/// Strategy for buffer reservation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReserveStrategy {
    /// Reserve exact amount needed
    Exact,
    /// Reserve with 50% growth headroom
    Grow50,
    /// Reserve with 100% growth headroom (double)
    Double,
    /// Reserve to next power of two
    PowerOfTwo,
}

/// Reserve buffer capacity according to strategy.
///
/// # Example
/// ```rust
/// use trueno::brick::{reserve_capacity, ReserveStrategy};
///
/// assert_eq!(reserve_capacity(100, ReserveStrategy::Exact), 100);
/// assert_eq!(reserve_capacity(100, ReserveStrategy::Grow50), 150);
/// assert_eq!(reserve_capacity(100, ReserveStrategy::Double), 200);
/// assert_eq!(reserve_capacity(100, ReserveStrategy::PowerOfTwo), 128);
/// ```
#[must_use]
pub fn reserve_capacity(needed: usize, strategy: ReserveStrategy) -> usize {
    match strategy {
        ReserveStrategy::Exact => needed,
        ReserveStrategy::Grow50 => needed + needed / 2,
        ReserveStrategy::Double => needed * 2,
        ReserveStrategy::PowerOfTwo => needed.next_power_of_two(),
    }
}

/// Buffer with configurable reserve strategy.
#[derive(Debug)]
pub struct StrategicBuffer {
    data: Vec<u8>,
    strategy: ReserveStrategy,
}

impl StrategicBuffer {
    /// Create a new buffer with the given strategy.
    pub fn new(strategy: ReserveStrategy) -> Self {
        Self { data: Vec::new(), strategy }
    }

    /// Create with initial capacity.
    pub fn with_capacity(capacity: usize, strategy: ReserveStrategy) -> Self {
        Self { data: Vec::with_capacity(reserve_capacity(capacity, strategy)), strategy }
    }

    /// Ensure capacity for additional bytes.
    pub fn reserve(&mut self, additional: usize) {
        let needed = self.data.len() + additional;
        if needed > self.data.capacity() {
            let new_cap = reserve_capacity(needed, self.strategy);
            self.data.reserve(new_cap - self.data.capacity());
        }
    }

    /// Write bytes to the buffer.
    pub fn write(&mut self, bytes: &[u8]) {
        contract_pre_write!();
        self.reserve(bytes.len());
        self.data.extend_from_slice(bytes);
    }

    /// Get the data.
    #[must_use]
    pub fn as_slice(&self) -> &[u8] {
        &self.data
    }

    /// Get current length.
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get capacity.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.data.capacity()
    }

    /// Clear the buffer.
    pub fn clear(&mut self) {
        self.data.clear();
    }
}

impl Default for StrategicBuffer {
    fn default() -> Self {
        Self::new(ReserveStrategy::Double)
    }
}
