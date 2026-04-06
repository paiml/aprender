//! Batch Splitting and Work Distribution
//!
//! LCP-05: Balance211 Work Distribution (Intel MKL pattern)
//! LCP-09: Batch Splitting Strategies

// ============================================================================
// LCP-05: Balance211 Work Distribution (Intel MKL pattern)
// ============================================================================

/// Balance211 work distribution (Intel MKL pattern).
///
/// Distributes N items across T threads such that no thread
/// has more than 1 extra item compared to any other.
///
/// # Example
/// ```rust
/// use trueno::brick::balance211;
///
/// let ranges = balance211(10, 3);
/// // Thread 0: (0, 4) - 4 items
/// // Thread 1: (4, 3) - 3 items
/// // Thread 2: (7, 3) - 3 items
/// assert_eq!(ranges.len(), 3);
/// ```
#[must_use]
pub fn balance211(n: usize, nthreads: usize) -> Vec<(usize, usize)> {
    if nthreads == 0 {
        return vec![];
    }
    let div = n / nthreads;
    let rem = n % nthreads;

    (0..nthreads)
        .map(|i| {
            let offset = if i < rem { (div + 1) * i } else { div * i + rem };
            let count = if i < rem { div + 1 } else { div };
            (offset, count)
        })
        .collect()
}

/// Iterator adapter for balanced work distribution.
pub struct Balance211Iter {
    ranges: Vec<(usize, usize)>,
    current: usize,
}

impl Balance211Iter {
    /// Create a new balanced work iterator.
    pub fn new(n: usize, nthreads: usize) -> Self {
        Self { ranges: balance211(n, nthreads), current: 0 }
    }
}

impl Iterator for Balance211Iter {
    type Item = std::ops::Range<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current >= self.ranges.len() {
            return None;
        }
        let (offset, count) = self.ranges[self.current];
        self.current += 1;
        Some(offset..offset + count)
    }
}

impl ExactSizeIterator for Balance211Iter {
    fn len(&self) -> usize {
        self.ranges.len() - self.current
    }
}

// ============================================================================
// LCP-09: Batch Splitting Strategies
// ============================================================================

/// Strategy for splitting batches across workers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BatchSplitStrategy {
    /// Simple equal division (may leave remainder)
    #[default]
    Simple,
    /// Equal distribution using Balance211
    Equal,
    /// Sequence-aware (keeps sequences together)
    SequenceAware,
}

/// Split a batch into chunks according to strategy.
///
/// # Example
/// ```rust
/// use trueno::brick::{split_batch, BatchSplitStrategy};
///
/// let chunks = split_batch(100, 4, BatchSplitStrategy::Equal);
/// assert_eq!(chunks.len(), 4);
/// assert_eq!(chunks.iter().sum::<usize>(), 100);
/// ```
#[must_use]
pub fn split_batch(total: usize, num_workers: usize, strategy: BatchSplitStrategy) -> Vec<usize> {
    if num_workers == 0 || total == 0 {
        return vec![];
    }

    match strategy {
        BatchSplitStrategy::Simple => {
            let chunk_size = total / num_workers;
            let mut chunks = vec![chunk_size; num_workers];
            // Last worker gets remainder
            if let Some(last) = chunks.last_mut() {
                *last += total % num_workers;
            }
            chunks
        }
        BatchSplitStrategy::Equal => {
            // Use Balance211 for even distribution
            balance211(total, num_workers).iter().map(|(_, count)| *count).collect()
        }
        BatchSplitStrategy::SequenceAware => {
            // For now, same as Equal (sequence boundaries would need external info)
            balance211(total, num_workers).iter().map(|(_, count)| *count).collect()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_balance211_basic() {
        let ranges = balance211(10, 3);
        assert_eq!(ranges.len(), 3);
        // First thread gets 4 items, others get 3
        assert_eq!(ranges[0], (0, 4));
        assert_eq!(ranges[1], (4, 3));
        assert_eq!(ranges[2], (7, 3));
    }

    #[test]
    fn test_balance211_even_division() {
        let ranges = balance211(12, 4);
        // 12 / 4 = 3 items each, no remainder
        for (i, &(offset, count)) in ranges.iter().enumerate() {
            assert_eq!(count, 3);
            assert_eq!(offset, i * 3);
        }
    }

    #[test]
    fn test_balance211_empty() {
        assert!(balance211(0, 4).iter().all(|&(_, c)| c == 0));
        assert!(balance211(10, 0).is_empty());
    }

    #[test]
    fn test_balance211_single_thread() {
        let ranges = balance211(100, 1);
        assert_eq!(ranges.len(), 1);
        assert_eq!(ranges[0], (0, 100));
    }

    #[test]
    fn test_balance211_more_threads_than_items() {
        let ranges = balance211(3, 5);
        assert_eq!(ranges.len(), 5);
        // First 3 threads get 1 item each, last 2 get 0
        let items: Vec<_> = ranges.iter().map(|(_, c)| *c).collect();
        assert_eq!(items, vec![1, 1, 1, 0, 0]);
    }

    #[test]
    fn test_balance211_iter_basic() {
        let mut iter = Balance211Iter::new(10, 3);
        assert_eq!(iter.len(), 3);

        assert_eq!(iter.next(), Some(0..4));
        assert_eq!(iter.next(), Some(4..7));
        assert_eq!(iter.next(), Some(7..10));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_balance211_iter_exact_size() {
        let iter = Balance211Iter::new(10, 3);
        assert_eq!(iter.len(), 3);

        let mut iter2 = Balance211Iter::new(10, 3);
        iter2.next();
        assert_eq!(iter2.len(), 2);
    }

    #[test]
    fn test_batch_split_strategy_default() {
        assert_eq!(BatchSplitStrategy::default(), BatchSplitStrategy::Simple);
    }

    #[test]
    fn test_split_batch_simple() {
        let chunks = split_batch(100, 4, BatchSplitStrategy::Simple);
        assert_eq!(chunks.len(), 4);
        // First 3 get 25, last gets 25 + 0 = 25
        assert_eq!(chunks, vec![25, 25, 25, 25]);
    }

    #[test]
    fn test_split_batch_simple_with_remainder() {
        let chunks = split_batch(10, 3, BatchSplitStrategy::Simple);
        assert_eq!(chunks.len(), 3);
        // 10 / 3 = 3, remainder = 1 goes to last
        assert_eq!(chunks, vec![3, 3, 4]);
        assert_eq!(chunks.iter().sum::<usize>(), 10);
    }

    #[test]
    fn test_split_batch_equal() {
        let chunks = split_batch(10, 3, BatchSplitStrategy::Equal);
        assert_eq!(chunks.len(), 3);
        // Balance211 gives first worker more: 4, 3, 3
        assert_eq!(chunks, vec![4, 3, 3]);
        assert_eq!(chunks.iter().sum::<usize>(), 10);
    }

    #[test]
    fn test_split_batch_sequence_aware() {
        let chunks = split_batch(10, 3, BatchSplitStrategy::SequenceAware);
        // Currently same as Equal
        assert_eq!(chunks, vec![4, 3, 3]);
    }

    #[test]
    fn test_split_batch_empty() {
        assert!(split_batch(0, 4, BatchSplitStrategy::Simple).is_empty());
        assert!(split_batch(100, 0, BatchSplitStrategy::Simple).is_empty());
    }

    #[test]
    fn test_split_batch_single_worker() {
        let chunks = split_batch(100, 1, BatchSplitStrategy::Simple);
        assert_eq!(chunks, vec![100]);
    }

    /// FALSIFICATION TEST: Verify total items preserved after split
    ///
    /// The sum of all chunks must equal the original total.
    #[test]
    fn test_falsify_split_batch_preserves_total() {
        for total in [1, 10, 100, 997, 1000, 10000] {
            for workers in [1, 2, 3, 4, 7, 16, 100] {
                for strategy in [
                    BatchSplitStrategy::Simple,
                    BatchSplitStrategy::Equal,
                    BatchSplitStrategy::SequenceAware,
                ] {
                    let chunks = split_batch(total, workers, strategy);
                    let sum: usize = chunks.iter().sum();
                    assert_eq!(
                        sum, total,
                        "FALSIFICATION FAILED: split_batch({}, {}, {:?}) sum {} != {}",
                        total, workers, strategy, sum, total
                    );
                }
            }
        }
    }

    /// FALSIFICATION TEST: Balance211 never gives more than 1 extra item
    ///
    /// The maximum difference between any two thread counts must be <= 1.
    #[test]
    fn test_falsify_balance211_max_diff_one() {
        for n in [1, 10, 100, 997, 1000] {
            for nthreads in [1, 2, 3, 4, 7, 16, 100] {
                let ranges = balance211(n, nthreads);
                if ranges.is_empty() {
                    continue;
                }
                let counts: Vec<_> = ranges.iter().map(|(_, c)| *c).collect();
                let max_count = *counts.iter().max().unwrap_or(&0);
                let min_count = *counts.iter().min().unwrap_or(&0);
                assert!(
                    max_count - min_count <= 1,
                    "FALSIFICATION FAILED: balance211({}, {}) has diff {} (max={}, min={})",
                    n,
                    nthreads,
                    max_count - min_count,
                    max_count,
                    min_count
                );
            }
        }
    }

    /// FALSIFICATION TEST: Balance211 ranges are contiguous and non-overlapping
    #[test]
    fn test_falsify_balance211_contiguous() {
        for n in [10, 100, 1000] {
            for nthreads in [2, 3, 4, 7] {
                let ranges = balance211(n, nthreads);
                let mut expected_offset = 0;
                for (i, &(offset, count)) in ranges.iter().enumerate() {
                    assert_eq!(
                        offset, expected_offset,
                        "FALSIFICATION FAILED: balance211({}, {}) range {} offset {} != expected {}",
                        n, nthreads, i, offset, expected_offset
                    );
                    expected_offset += count;
                }
                assert_eq!(
                    expected_offset, n,
                    "FALSIFICATION FAILED: balance211({}, {}) total {} != {}",
                    n, nthreads, expected_offset, n
                );
            }
        }
    }
}
