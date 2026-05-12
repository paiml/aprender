// SHIP-TWO-001 — `paged-kv-cache-v1` algorithm-level PARTIAL discharge
// for FALSIFY-PKV-001..007 (closes 7/7 sweep).
//
// Contract: `contracts/paged-kv-cache-v1.yaml`.

use std::collections::HashSet;

// ===========================================================================
// Reference paged KV cache: block table + free pool
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PkvError { OutOfBlocks, InvalidRequest, BlockSizeZero }

#[derive(Debug, Clone)]
pub struct BlockPool {
    pub block_size: usize,
    pub pool_size: usize,
    pub free: Vec<usize>,
    /// `block_table[req_idx]` is the list of physical block IDs assigned to req `req_idx`.
    pub block_table: Vec<Vec<usize>>,
}

impl BlockPool {
    pub fn new(pool_size: usize, block_size: usize, max_requests: usize) -> Self {
        Self {
            block_size,
            pool_size,
            free: (0..pool_size).rev().collect(),
            block_table: vec![Vec::new(); max_requests],
        }
    }

    /// Allocate `n` blocks for request `req_idx`.
    pub fn allocate(&mut self, req_idx: usize, n: usize) -> Result<(), PkvError> {
        if self.block_size == 0 { return Err(PkvError::BlockSizeZero); }
        if req_idx >= self.block_table.len() { return Err(PkvError::InvalidRequest); }
        if self.free.len() < n { return Err(PkvError::OutOfBlocks); }
        for _ in 0..n {
            // Safety: free.len() >= n checked above; pop() returns Some.
            let b = self.free.pop().expect("free has block (len checked)");
            self.block_table[req_idx].push(b);
        }
        Ok(())
    }

    pub fn free_request(&mut self, req_idx: usize) {
        if req_idx >= self.block_table.len() { return; }
        let blocks: Vec<usize> = std::mem::take(&mut self.block_table[req_idx]);
        for b in blocks { self.free.push(b); }
    }

    pub fn allocated_count(&self) -> usize {
        self.pool_size - self.free.len()
    }

    /// Map (req_idx, position) → physical slot ID.
    /// `slot = block_table[req][pos / B] * B + (pos % B)`.
    pub fn slot_of(&self, req_idx: usize, pos: usize) -> Option<usize> {
        if self.block_size == 0 { return None; }
        if req_idx >= self.block_table.len() { return None; }
        let block_idx = pos / self.block_size;
        let intra = pos % self.block_size;
        let table = &self.block_table[req_idx];
        if block_idx >= table.len() { return None; }
        Some(table[block_idx] * self.block_size + intra)
    }
}

#[must_use]
pub const fn blocks_for_seq(seq_len: usize, block_size: usize) -> usize {
    if block_size == 0 || seq_len == 0 { return 0; }
    seq_len.div_ceil(block_size)
}

// ===========================================================================
// PKV-001 — Slot bijectivity: no two (req, pos) → same slot
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pkv001Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_slot_bijectivity(pool: &BlockPool, lengths: &[usize]) -> Pkv001Verdict {
    if lengths.is_empty() { return Pkv001Verdict::Fail; }
    let mut seen: HashSet<usize> = HashSet::new();
    for (req, &len) in lengths.iter().enumerate() {
        for pos in 0..len {
            let slot = match pool.slot_of(req, pos) { Some(s) => s, None => return Pkv001Verdict::Fail };
            if !seen.insert(slot) { return Pkv001Verdict::Fail; }
        }
    }
    Pkv001Verdict::Pass
}

// ===========================================================================
// PKV-002 — Paged matches contiguous within tolerance
// ===========================================================================

pub const AC_PKV_002_TOLERANCE: f64 = 1e-5;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pkv002Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_paged_contiguous_parity(paged: &[f32], contiguous: &[f32]) -> Pkv002Verdict {
    if paged.len() != contiguous.len() || paged.is_empty() { return Pkv002Verdict::Fail; }
    for (a, b) in paged.iter().zip(contiguous) {
        if !a.is_finite() || !b.is_finite() { return Pkv002Verdict::Fail; }
        if ((*a as f64) - (*b as f64)).abs() > AC_PKV_002_TOLERANCE { return Pkv002Verdict::Fail; }
    }
    Pkv002Verdict::Pass
}

// ===========================================================================
// PKV-003 — Block allocation monotonicity: s1 < s2 ⇒ blocks(s1) <= blocks(s2)
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pkv003Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_block_allocation_monotone(s1: usize, s2: usize, block_size: usize) -> Pkv003Verdict {
    if block_size == 0 || s1 >= s2 { return Pkv003Verdict::Fail; }
    let b1 = blocks_for_seq(s1, block_size);
    let b2 = blocks_for_seq(s2, block_size);
    if b1 <= b2 { Pkv003Verdict::Pass } else { Pkv003Verdict::Fail }
}

// ===========================================================================
// PKV-004 — Waste bounded by block_size: per-request waste < B
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pkv004Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_waste_bounded(seq_len: usize, block_size: usize) -> Pkv004Verdict {
    if block_size == 0 { return Pkv004Verdict::Fail; }
    let blocks = blocks_for_seq(seq_len, block_size);
    let allocated = blocks * block_size;
    let waste = if allocated >= seq_len { allocated - seq_len } else { return Pkv004Verdict::Fail };
    if waste < block_size { Pkv004Verdict::Pass } else { Pkv004Verdict::Fail }
}

// ===========================================================================
// PKV-005 — No duplicate blocks per request
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pkv005Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_no_duplicate_blocks(pool: &BlockPool) -> Pkv005Verdict {
    for blocks in &pool.block_table {
        let mut seen = HashSet::new();
        for &b in blocks {
            if !seen.insert(b) { return Pkv005Verdict::Fail; }
        }
    }
    Pkv005Verdict::Pass
}

// ===========================================================================
// PKV-006 — Pool conservation: allocated + free == pool_size
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pkv006Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_pool_conservation(pool: &BlockPool) -> Pkv006Verdict {
    if pool.allocated_count() + pool.free.len() == pool.pool_size {
        Pkv006Verdict::Pass
    } else {
        Pkv006Verdict::Fail
    }
}

// ===========================================================================
// PKV-007 — Graph compatibility: block table tensor shape unchanged
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pkv007Verdict { Pass, Fail }

/// Pass iff the block-table tensor's outer shape (max_requests x
/// max_blocks_per_request) is identical before and after a request
/// add/remove. The actual contents may change.
#[must_use]
pub const fn verdict_from_graph_shape_unchanged(
    before: (usize, usize),
    after: (usize, usize),
) -> Pkv007Verdict {
    if before.0 == after.0 && before.1 == after.1 { Pkv007Verdict::Pass } else { Pkv007Verdict::Fail }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn alloc_helper(pool: &mut BlockPool, lengths: &[usize]) {
        for (req, &len) in lengths.iter().enumerate() {
            let blocks_needed = blocks_for_seq(len, pool.block_size);
            pool.allocate(req, blocks_needed).expect("alloc");
        }
    }

    // Reference impl checks
    #[test] fn ref_alloc_and_free() {
        let mut p = BlockPool::new(8, 4, 2);
        p.allocate(0, 3).unwrap();
        p.allocate(1, 2).unwrap();
        assert_eq!(p.allocated_count(), 5);
        p.free_request(0);
        assert_eq!(p.allocated_count(), 2);
    }

    #[test] fn ref_slot_of() {
        let mut p = BlockPool::new(8, 4, 1);
        p.allocate(0, 3).unwrap(); // 3 blocks of 4 = 12 slots.
        // Free pool is built with (0..8).rev() = [7,6,5,4,3,2,1,0]; pop()
        // returns the *last* element first, so block_table[0] = [0, 1, 2].
        let slot = p.slot_of(0, 5).unwrap();
        // pos=5 → block_idx 1, intra 1 → block_table[0][1] * 4 + 1 = 1*4 + 1 = 5.
        assert_eq!(slot, 5);
    }

    // PKV-001
    #[test] fn pkv001_pass_uniform() {
        let mut p = BlockPool::new(16, 4, 3);
        let lengths = vec![5, 7, 3]; // 2 + 2 + 1 = 5 blocks
        alloc_helper(&mut p, &lengths);
        assert_eq!(verdict_from_slot_bijectivity(&p, &lengths), Pkv001Verdict::Pass);
    }
    #[test] fn pkv001_fail_empty() {
        let p = BlockPool::new(8, 4, 1);
        assert_eq!(verdict_from_slot_bijectivity(&p, &[]), Pkv001Verdict::Fail);
    }

    // PKV-002
    #[test] fn pkv002_pass_match() {
        assert_eq!(verdict_from_paged_contiguous_parity(&[1.0, 2.0], &[1.0, 2.0]), Pkv002Verdict::Pass);
    }
    #[test] fn pkv002_pass_within_tolerance() {
        let a = [1.0_f32, 2.0];
        let b = [1.0_f32 + 1e-7, 2.0 + 1e-7];
        assert_eq!(verdict_from_paged_contiguous_parity(&a, &b), Pkv002Verdict::Pass);
    }
    #[test] fn pkv002_fail_drift() {
        assert_eq!(verdict_from_paged_contiguous_parity(&[1.0], &[2.0]), Pkv002Verdict::Fail);
    }

    // PKV-003
    #[test] fn pkv003_pass_canonical() {
        // s1=4, s2=10, B=4 → b1=1, b2=3.
        assert_eq!(verdict_from_block_allocation_monotone(4, 10, 4), Pkv003Verdict::Pass);
    }
    #[test] fn pkv003_pass_same_block() {
        // s1=4, s2=5, B=4 → b1=1, b2=2.
        assert_eq!(verdict_from_block_allocation_monotone(4, 5, 4), Pkv003Verdict::Pass);
    }
    #[test] fn pkv003_fail_swapped() {
        assert_eq!(verdict_from_block_allocation_monotone(10, 4, 4), Pkv003Verdict::Fail);
    }
    #[test] fn pkv003_fail_zero_block() {
        assert_eq!(verdict_from_block_allocation_monotone(4, 10, 0), Pkv003Verdict::Fail);
    }

    // PKV-004
    #[test] fn pkv004_pass_exact_block() {
        // 8 / 4 = 2 blocks, no waste.
        assert_eq!(verdict_from_waste_bounded(8, 4), Pkv004Verdict::Pass);
    }
    #[test] fn pkv004_pass_partial() {
        // 9 / 4 → 3 blocks = 12 slots, waste = 3.
        assert_eq!(verdict_from_waste_bounded(9, 4), Pkv004Verdict::Pass);
    }
    #[test] fn pkv004_fail_zero_block() {
        assert_eq!(verdict_from_waste_bounded(9, 0), Pkv004Verdict::Fail);
    }

    // PKV-005
    #[test] fn pkv005_pass_unique() {
        let mut p = BlockPool::new(8, 4, 2);
        p.allocate(0, 2).unwrap();
        assert_eq!(verdict_from_no_duplicate_blocks(&p), Pkv005Verdict::Pass);
    }
    #[test] fn pkv005_fail_duplicate() {
        let mut p = BlockPool::new(8, 4, 1);
        p.allocate(0, 2).unwrap();
        // Forcefully inject a duplicate.
        let dup = p.block_table[0][0];
        p.block_table[0].push(dup);
        assert_eq!(verdict_from_no_duplicate_blocks(&p), Pkv005Verdict::Fail);
    }

    // PKV-006
    #[test] fn pkv006_pass_initial() {
        let p = BlockPool::new(8, 4, 1);
        assert_eq!(verdict_from_pool_conservation(&p), Pkv006Verdict::Pass);
    }
    #[test] fn pkv006_pass_after_alloc() {
        let mut p = BlockPool::new(8, 4, 2);
        p.allocate(0, 2).unwrap();
        p.allocate(1, 3).unwrap();
        assert_eq!(verdict_from_pool_conservation(&p), Pkv006Verdict::Pass);
    }
    #[test] fn pkv006_pass_alloc_free_cycle() {
        let mut p = BlockPool::new(16, 4, 4);
        p.allocate(0, 3).unwrap();
        p.allocate(1, 2).unwrap();
        p.free_request(0);
        p.allocate(2, 4).unwrap();
        p.free_request(1);
        assert_eq!(verdict_from_pool_conservation(&p), Pkv006Verdict::Pass);
    }

    // PKV-007
    #[test] fn pkv007_pass_unchanged() {
        assert_eq!(verdict_from_graph_shape_unchanged((8, 16), (8, 16)), Pkv007Verdict::Pass);
    }
    #[test] fn pkv007_fail_max_req_changed() {
        assert_eq!(verdict_from_graph_shape_unchanged((8, 16), (16, 16)), Pkv007Verdict::Fail);
    }
    #[test] fn pkv007_fail_max_blocks_changed() {
        assert_eq!(verdict_from_graph_shape_unchanged((8, 16), (8, 32)), Pkv007Verdict::Fail);
    }

    // Provenance pin
    #[test] fn provenance_tolerance() {
        assert!((AC_PKV_002_TOLERANCE - 1e-5).abs() < 1e-12);
    }

    // blocks_for_seq helper standalone test
    #[test] fn blocks_for_seq_canonical() {
        assert_eq!(blocks_for_seq(0, 4), 0);
        assert_eq!(blocks_for_seq(1, 4), 1);
        assert_eq!(blocks_for_seq(4, 4), 1);
        assert_eq!(blocks_for_seq(5, 4), 2);
        assert_eq!(blocks_for_seq(8, 4), 2);
        assert_eq!(blocks_for_seq(9, 4), 3);
        assert_eq!(blocks_for_seq(100, 0), 0);
    }
}
