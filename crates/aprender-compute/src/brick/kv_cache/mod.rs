//! KV Cache Management and Batch Ordering
//!
//! LCP-10: KV Cache Slot tracking for transformer inference.
//! LCP-14: Sequential Batch Ordering for cache-friendly processing.

// ----------------------------------------------------------------------------
// LCP-10: KV Cache Slot Info
// ----------------------------------------------------------------------------

/// Metadata for a KV cache slot in transformer inference.
///
/// Tracks position, token info, and usage for cache management.
#[derive(Debug, Clone, Default)]
pub struct KvCacheSlotInfo {
    /// Sequence position this slot represents
    pub position: u32,
    /// Token ID stored in this slot
    pub token_id: u32,
    /// Layer index
    pub layer: u16,
    /// Head index
    pub head: u16,
    /// Whether this slot is valid/filled
    pub valid: bool,
    /// Last access time (in steps)
    pub last_access: u64,
}

impl KvCacheSlotInfo {
    /// Create a new slot info.
    pub fn new(position: u32, token_id: u32, layer: u16, head: u16) -> Self {
        Self { position, token_id, layer, head, valid: true, last_access: 0 }
    }

    /// Mark slot as accessed.
    pub fn touch(&mut self, step: u64) {
        self.last_access = step;
    }

    /// Invalidate the slot.
    pub fn invalidate(&mut self) {
        self.valid = false;
    }

    /// Check if slot can be evicted (LRU policy).
    #[must_use]
    pub fn eviction_priority(&self, current_step: u64) -> u64 {
        if !self.valid {
            return u64::MAX; // Invalid slots have highest eviction priority
        }
        current_step.saturating_sub(self.last_access)
    }
}

/// KV cache manager with slot tracking.
#[derive(Debug)]
pub struct KvCacheManager {
    /// Slot metadata
    slots: Vec<KvCacheSlotInfo>,
    /// Current step counter
    current_step: u64,
    /// Number of valid slots
    valid_count: usize,
}

impl KvCacheManager {
    /// Create manager with given capacity.
    pub fn new(capacity: usize) -> Self {
        Self { slots: vec![KvCacheSlotInfo::default(); capacity], current_step: 0, valid_count: 0 }
    }

    /// Allocate a slot.
    pub fn allocate(
        &mut self,
        position: u32,
        token_id: u32,
        layer: u16,
        head: u16,
    ) -> Option<usize> {
        // Find first invalid slot
        for (i, slot) in self.slots.iter_mut().enumerate() {
            if !slot.valid {
                *slot = KvCacheSlotInfo::new(position, token_id, layer, head);
                slot.touch(self.current_step);
                self.valid_count += 1;
                return Some(i);
            }
        }
        None // No free slots
    }

    /// Access a slot.
    pub fn access(&mut self, index: usize) -> Option<&KvCacheSlotInfo> {
        if index < self.slots.len() {
            self.slots[index].touch(self.current_step);
            Some(&self.slots[index])
        } else {
            None
        }
    }

    /// Evict LRU slot.
    pub fn evict_lru(&mut self) -> Option<usize> {
        let mut best_idx = None;
        let mut best_priority = 0u64;

        for (i, slot) in self.slots.iter().enumerate() {
            if slot.valid {
                let priority = slot.eviction_priority(self.current_step);
                // Use >= for first found slot (best_idx.is_none()), then > for ties
                if best_idx.is_none() || priority > best_priority {
                    best_priority = priority;
                    best_idx = Some(i);
                }
            }
        }

        if let Some(idx) = best_idx {
            self.slots[idx].invalidate();
            self.valid_count -= 1;
        }
        best_idx
    }

    /// Advance step counter.
    pub fn step(&mut self) {
        self.current_step += 1;
    }

    /// Get number of valid slots.
    #[must_use]
    pub fn valid_count(&self) -> usize {
        self.valid_count
    }

    /// Get capacity.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.slots.len()
    }
}

// ----------------------------------------------------------------------------
// LCP-14: Sequential Batch Ordering
// ----------------------------------------------------------------------------

/// Sequential batch orderer for cache-friendly processing.
///
/// Ensures batches are processed in optimal order for memory access patterns.
#[derive(Debug, Clone)]
pub struct SequentialBatchOrderer {
    /// Batch indices in processing order
    order: Vec<usize>,
    /// Current position in order
    position: usize,
}

impl SequentialBatchOrderer {
    /// Create orderer for n batches.
    pub fn new(n_batches: usize) -> Self {
        Self { order: (0..n_batches).collect(), position: 0 }
    }

    /// Create orderer with reverse order (sometimes better for certain patterns).
    pub fn reversed(n_batches: usize) -> Self {
        Self { order: (0..n_batches).rev().collect(), position: 0 }
    }

    /// Create orderer with interleaved order (for better cache utilization).
    pub fn interleaved(n_batches: usize) -> Self {
        let mut order = Vec::with_capacity(n_batches);
        let mid = n_batches / 2;

        // Interleave: 0, mid, 1, mid+1, 2, mid+2, ...
        for i in 0..mid {
            order.push(i);
            if mid + i < n_batches {
                order.push(mid + i);
            }
        }
        // Handle odd number of batches
        if !n_batches.is_multiple_of(2) {
            order.push(n_batches - 1);
        }

        Self { order, position: 0 }
    }

    /// Get next batch index.
    pub fn next_batch(&mut self) -> Option<usize> {
        if self.position < self.order.len() {
            let idx = self.order[self.position];
            self.position += 1;
            Some(idx)
        } else {
            None
        }
    }

    /// Reset to beginning.
    pub fn reset(&mut self) {
        self.position = 0;
    }

    /// Check if all batches have been processed.
    #[must_use]
    pub fn is_done(&self) -> bool {
        self.position >= self.order.len()
    }

    /// Get remaining count.
    #[must_use]
    pub fn remaining(&self) -> usize {
        self.order.len().saturating_sub(self.position)
    }
}

impl Iterator for SequentialBatchOrderer {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        self.next_batch()
    }
}

#[cfg(test)]
mod tests;
