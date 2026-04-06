// =============================================================================
// FIXED-SIZE RING BUFFER (trueno-viz O(1) history pattern)
// =============================================================================

/// Fixed-size ring buffer for O(1) history access (trueno-viz pattern)
///
/// Provides constant-time insert, latest N values, and rolling statistics.
/// No heap allocation after initial creation.
#[derive(Debug, Clone)]
pub struct RingBuffer<T, const N: usize> {
    data: [T; N],
    head: usize,
    len: usize,
}

impl<T: Default + Copy, const N: usize> Default for RingBuffer<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Default + Copy, const N: usize> RingBuffer<T, N> {
    /// Create a new empty ring buffer
    #[must_use]
    pub fn new() -> Self {
        Self {
            data: [T::default(); N],
            head: 0,
            len: 0,
        }
    }

    /// Push a value (O(1))
    pub fn push(&mut self, value: T) {
        self.data[self.head] = value;
        self.head = (self.head + 1) % N;
        self.len = self.len.saturating_add(1).min(N);
    }

    /// Get current length
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Check if full
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.len == N
    }

    /// Get capacity
    #[must_use]
    pub fn capacity(&self) -> usize {
        N
    }

    /// Get the most recent value (O(1))
    #[must_use]
    pub fn latest(&self) -> Option<&T> {
        if self.len == 0 {
            None
        } else {
            let idx = if self.head == 0 { N - 1 } else { self.head - 1 };
            Some(&self.data[idx])
        }
    }

    /// Get value at index from oldest (O(1))
    /// Index 0 = oldest, index len-1 = newest
    #[must_use]
    pub fn get(&self, index: usize) -> Option<&T> {
        if index >= self.len {
            return None;
        }
        let start = if self.len < N { 0 } else { self.head };
        let actual_idx = (start + index) % N;
        Some(&self.data[actual_idx])
    }

    /// Iterate from oldest to newest
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        let start = if self.len < N { 0 } else { self.head };
        (0..self.len).map(move |i| {
            let idx = (start + i) % N;
            &self.data[idx]
        })
    }

    /// Clear the buffer
    pub fn clear(&mut self) {
        self.head = 0;
        self.len = 0;
    }
}

/// Rolling statistics over a ring buffer (O(1) access)
impl<const N: usize> RingBuffer<f64, N> {
    /// Calculate running sum (O(n) but typically small N)
    #[must_use]
    pub fn sum(&self) -> f64 {
        self.iter().sum()
    }

    /// Calculate mean (O(n))
    #[must_use]
    pub fn mean(&self) -> f64 {
        if self.len == 0 {
            0.0
        } else {
            self.sum() / self.len as f64
        }
    }

    /// Get min value (O(n))
    #[must_use]
    pub fn min(&self) -> Option<f64> {
        self.iter()
            .copied()
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Get max value (O(n))
    #[must_use]
    pub fn max(&self) -> Option<f64> {
        self.iter()
            .copied()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
    }
}

// =============================================================================
// HISTOGRAM BIN (trueno-viz O(1) distribution pattern)
// =============================================================================

/// Fixed-bin histogram for O(1) distribution tracking (trueno-viz pattern)
///
/// Pre-defined bins for common latency ranges. Insert and query are O(1).
#[derive(Debug, Clone)]
pub struct LatencyHistogram {
    /// Counts for bins: [0-1ms, 1-5ms, 5-10ms, 10-50ms, 50-100ms, 100-500ms, 500ms+]
    bins: [u64; 7],
    /// Total count
    count: u64,
}

impl Default for LatencyHistogram {
    fn default() -> Self {
        Self::new()
    }
}

impl LatencyHistogram {
    /// Create a new histogram
    #[must_use]
    pub fn new() -> Self {
        Self {
            bins: [0; 7],
            count: 0,
        }
    }

    /// Record a latency value in microseconds (O(1))
    pub fn record(&mut self, latency_us: u64) {
        let bin = match latency_us {
            0..=999 => 0,         // 0-1ms
            1000..=4999 => 1,     // 1-5ms
            5000..=9999 => 2,     // 5-10ms
            10000..=49999 => 3,   // 10-50ms
            50000..=99999 => 4,   // 50-100ms
            100000..=499999 => 5, // 100-500ms
            _ => 6,               // 500ms+
        };
        self.bins[bin] += 1;
        self.count += 1;
    }

    /// Get total count
    #[must_use]
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Get count for a specific bin
    #[must_use]
    pub fn bin_count(&self, bin: usize) -> u64 {
        self.bins.get(bin).copied().unwrap_or(0)
    }

    /// Get percentage in each bin
    #[must_use]
    pub fn percentages(&self) -> [f64; 7] {
        if self.count == 0 {
            return [0.0; 7];
        }
        let mut pcts = [0.0; 7];
        for (i, &count) in self.bins.iter().enumerate() {
            pcts[i] = (count as f64 / self.count as f64) * 100.0;
        }
        pcts
    }

    /// Get bin label
    #[must_use]
    pub fn bin_label(bin: usize) -> &'static str {
        match bin {
            0 => "0-1ms",
            1 => "1-5ms",
            2 => "5-10ms",
            3 => "10-50ms",
            4 => "50-100ms",
            5 => "100-500ms",
            6 => "500ms+",
            _ => "?",
        }
    }

    /// Format as ASCII histogram
    #[must_use]
    pub fn ascii_histogram(&self, width: usize) -> String {
        let pcts = self.percentages();
        let mut lines = Vec::new();

        for (i, pct) in pcts.iter().enumerate() {
            let bar_len = ((*pct / 100.0) * width as f64) as usize;
            let bar: String = "█".repeat(bar_len);
            lines.push(format!("{:>10} {:5.1}% {}", Self::bin_label(i), pct, bar));
        }

        lines.join("\n")
    }

    /// Reset histogram
    pub fn reset(&mut self) {
        self.bins = [0; 7];
        self.count = 0;
    }
}

// =============================================================================
// TRACKER MACRO (PMAT-019: reduce ResourceManagement entropy)
// =============================================================================

/// Generates a tracker struct with all-zero `new()`, `Default`, `Clone`, and `reset()`.
/// Domain-specific methods are added via separate `impl` blocks.
#[allow(unused_macros)]
macro_rules! define_tracker {
    (
        $(#[$meta:meta])*
        $vis:vis struct $name:ident {
            $( $(#[$fmeta:meta])* $fvis:vis $fname:ident : $fty:ty ),+ $(,)?
        }
    ) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
        $vis struct $name {
            $( $(#[$fmeta])* $fvis $fname : $fty, )+
        }

        impl $name {
            /// Create new zeroed tracker.
            #[inline]
            #[must_use]
            pub const fn new() -> Self {
                Self { $( $fname: 0, )+ }
            }

            /// Reset all counters to zero.
            #[inline]
            pub fn reset(&mut self) {
                *self = Self::new();
            }
        }
    };
}
