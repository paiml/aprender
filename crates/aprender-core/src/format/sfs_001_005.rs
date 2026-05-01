// SHIP-TWO-001 — `safetensors-format-safety-v1` algorithm-level PARTIAL
// discharge for FALSIFY-ST-001..005 (closes 5/5 sweep).
//
// Contract: `contracts/safetensors-format-safety-v1.yaml`.
// Spec: HuggingFace safetensors binary format safety
// (CVE-2023-37470 mitigations).
//
// NOTE: gate IDs in this contract clash with `special_tokens_contract_falsify`'s
// FALSIFY-ST-* IDs but the safetensors gates are SAFETENSORS-class while the
// special-tokens module covers a different contract; the verdict module names
// disambiguate via the `Sfs0NN` / `Sfs0NNVerdict` prefix.

// ===========================================================================
// SFS-001 — Header size bounded: header_size < MAX AND header_size + 8 ≤ file_size
// ===========================================================================

pub const AC_SFS_001_MAX_HEADER_SIZE: u64 = 100 * 1024 * 1024; // 100 MB

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sfs001Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_header_size_bounded(header_size: u64, file_size: u64) -> Sfs001Verdict {
    if file_size < 8 { return Sfs001Verdict::Fail; } // need 8 bytes for header_size field
    if header_size == 0 { return Sfs001Verdict::Fail; } // header must contain at least empty JSON
    if header_size > AC_SFS_001_MAX_HEADER_SIZE { return Sfs001Verdict::Fail; }
    let total = match header_size.checked_add(8) {
        Some(t) => t,
        None => return Sfs001Verdict::Fail,
    };
    if total > file_size { return Sfs001Verdict::Fail; }
    Sfs001Verdict::Pass
}

// ===========================================================================
// SFS-002 — Tensor offset bounds: 0 ≤ begin < end AND data_start + end ≤ file_size
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sfs002Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_tensor_offset_bounds(
    data_start: u64,
    begin: u64,
    end: u64,
    file_size: u64,
) -> Sfs002Verdict {
    if file_size == 0 { return Sfs002Verdict::Fail; }
    if begin >= end { return Sfs002Verdict::Fail; } // empty or reversed range
    let abs_end = match data_start.checked_add(end) {
        Some(e) => e,
        None => return Sfs002Verdict::Fail,
    };
    if abs_end > file_size { return Sfs002Verdict::Fail; }
    Sfs002Verdict::Pass
}

// ===========================================================================
// SFS-003 — No overlap: ∀ pair, [t1.begin, t1.end) ∩ [t2.begin, t2.end) = ∅
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sfs003Verdict { Pass, Fail }

/// Pass iff no two ranges overlap. O(n log n) sorted-pair check.
#[must_use]
pub fn verdict_from_no_overlap(ranges: &[(u64, u64)]) -> Sfs003Verdict {
    if ranges.is_empty() { return Sfs003Verdict::Fail; }
    let mut sorted = ranges.to_vec();
    for &(b, e) in &sorted {
        if b >= e { return Sfs003Verdict::Fail; }
    }
    sorted.sort_by_key(|&(b, _)| b);
    for i in 1..sorted.len() {
        let (prev_b, prev_e) = sorted[i - 1];
        let (cur_b, _) = sorted[i];
        if prev_e > cur_b {
            return Sfs003Verdict::Fail; // overlap
        }
        // gap (prev_e < cur_b) and contiguous (prev_e == cur_b) are both OK
        let _ = prev_b;
    }
    Sfs003Verdict::Pass
}

// ===========================================================================
// SFS-004 — DType byte count: byte_count == product(shape) * dtype_size
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SfsDType { F16, F32, F64, Bf16, I8, I16, I32, I64, U8, Bool }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sfs004Verdict { Pass, Fail }

#[must_use]
pub const fn dtype_size(dtype: SfsDType) -> u64 {
    match dtype {
        SfsDType::F16 | SfsDType::Bf16 | SfsDType::I16 => 2,
        SfsDType::F32 | SfsDType::I32 => 4,
        SfsDType::F64 | SfsDType::I64 => 8,
        SfsDType::I8 | SfsDType::U8 | SfsDType::Bool => 1,
    }
}

#[must_use]
pub fn verdict_from_dtype_size_match(
    shape: &[u64],
    dtype: SfsDType,
    byte_count: u64,
) -> Sfs004Verdict {
    if shape.is_empty() { return Sfs004Verdict::Fail; }
    let mut product: u64 = 1;
    for &d in shape {
        if d == 0 { return Sfs004Verdict::Fail; }
        product = match product.checked_mul(d) {
            Some(p) => p,
            None => return Sfs004Verdict::Fail,
        };
    }
    let expected = match product.checked_mul(dtype_size(dtype)) {
        Some(s) => s,
        None => return Sfs004Verdict::Fail,
    };
    if byte_count == expected { Sfs004Verdict::Pass } else { Sfs004Verdict::Fail }
}

// ===========================================================================
// SFS-005 — Zero-copy mmap: heap allocation == 0 for tensor mmap
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sfs005Verdict { Pass, Fail }

/// Pass iff `heap_alloc_bytes == 0` AND `mmap_size == claimed_tensor_size`
/// (mmap returns a borrowed slice of exactly the tensor size, no copy).
#[must_use]
pub const fn verdict_from_zero_copy_mmap(
    heap_alloc_bytes: u64,
    mmap_size: u64,
    claimed_tensor_size: u64,
) -> Sfs005Verdict {
    if claimed_tensor_size == 0 { return Sfs005Verdict::Fail; }
    if heap_alloc_bytes != 0 { return Sfs005Verdict::Fail; } // ANY heap copy fails
    if mmap_size != claimed_tensor_size { return Sfs005Verdict::Fail; }
    Sfs005Verdict::Pass
}

#[cfg(test)]
mod tests {
    use super::*;

    // SFS-001 (header size)
    #[test] fn sfs001_pass_canonical() {
        // 1KB header in 16KB file.
        assert_eq!(verdict_from_header_size_bounded(1024, 16384), Sfs001Verdict::Pass);
    }
    #[test] fn sfs001_pass_at_max() {
        assert_eq!(
            verdict_from_header_size_bounded(AC_SFS_001_MAX_HEADER_SIZE, AC_SFS_001_MAX_HEADER_SIZE + 1024),
            Sfs001Verdict::Pass
        );
    }
    #[test] fn sfs001_fail_attack_max_u64() {
        // Contract's stated falsifier: header_size = 0xFFFFFFFFFFFFFFFF.
        assert_eq!(
            verdict_from_header_size_bounded(u64::MAX, 1024),
            Sfs001Verdict::Fail
        );
    }
    #[test] fn sfs001_fail_above_max() {
        // 200MB header — above 100MB cap.
        assert_eq!(
            verdict_from_header_size_bounded(200 * 1024 * 1024, u64::MAX),
            Sfs001Verdict::Fail
        );
    }
    #[test] fn sfs001_fail_exceeds_file() {
        // header_size + 8 > file_size.
        assert_eq!(verdict_from_header_size_bounded(2000, 1024), Sfs001Verdict::Fail);
    }
    #[test] fn sfs001_fail_zero_header() {
        assert_eq!(verdict_from_header_size_bounded(0, 1024), Sfs001Verdict::Fail);
    }
    #[test] fn sfs001_fail_file_too_small() {
        // file_size < 8 — can't even read header_size field.
        assert_eq!(verdict_from_header_size_bounded(100, 4), Sfs001Verdict::Fail);
    }

    // SFS-002 (tensor offset bounds)
    #[test] fn sfs002_pass_canonical() {
        // data_start=1024, begin=0, end=4096, file_size=8192.
        assert_eq!(verdict_from_tensor_offset_bounds(1024, 0, 4096, 8192), Sfs002Verdict::Pass);
    }
    #[test] fn sfs002_pass_at_end() {
        // data_start + end == file_size exactly.
        assert_eq!(verdict_from_tensor_offset_bounds(1024, 0, 7168, 8192), Sfs002Verdict::Pass);
    }
    #[test] fn sfs002_fail_oob() {
        // The contract's stated falsifier: end > file_size.
        assert_eq!(verdict_from_tensor_offset_bounds(1024, 0, 10000, 8192), Sfs002Verdict::Fail);
    }
    #[test] fn sfs002_fail_begin_eq_end() {
        // Empty range rejected.
        assert_eq!(verdict_from_tensor_offset_bounds(1024, 100, 100, 8192), Sfs002Verdict::Fail);
    }
    #[test] fn sfs002_fail_reversed() {
        // begin > end.
        assert_eq!(verdict_from_tensor_offset_bounds(1024, 200, 100, 8192), Sfs002Verdict::Fail);
    }
    #[test] fn sfs002_fail_overflow() {
        // data_start + end overflows u64.
        assert_eq!(
            verdict_from_tensor_offset_bounds(u64::MAX - 50, 0, 100, u64::MAX),
            Sfs002Verdict::Fail
        );
    }

    // SFS-003 (no overlap)
    #[test] fn sfs003_pass_disjoint() {
        // [0, 100), [100, 200), [200, 300) — contiguous, no overlap.
        let ranges = vec![(0_u64, 100), (100, 200), (200, 300)];
        assert_eq!(verdict_from_no_overlap(&ranges), Sfs003Verdict::Pass);
    }
    #[test] fn sfs003_pass_with_gaps() {
        let ranges = vec![(0_u64, 50), (100, 150), (200, 250)];
        assert_eq!(verdict_from_no_overlap(&ranges), Sfs003Verdict::Pass);
    }
    #[test] fn sfs003_pass_unsorted_input() {
        // Verdict sorts internally.
        let ranges = vec![(200_u64, 250), (0, 50), (100, 150)];
        assert_eq!(verdict_from_no_overlap(&ranges), Sfs003Verdict::Pass);
    }
    #[test] fn sfs003_fail_overlap() {
        // Contract's stated falsifier: A=[0,100), B=[50,150).
        let ranges = vec![(0_u64, 100), (50, 150)];
        assert_eq!(verdict_from_no_overlap(&ranges), Sfs003Verdict::Fail);
    }
    #[test] fn sfs003_fail_full_overlap() {
        let ranges = vec![(0_u64, 200), (50, 100)]; // B contained in A
        assert_eq!(verdict_from_no_overlap(&ranges), Sfs003Verdict::Fail);
    }
    #[test] fn sfs003_fail_zero_range() {
        let ranges = vec![(100_u64, 100)];
        assert_eq!(verdict_from_no_overlap(&ranges), Sfs003Verdict::Fail);
    }
    #[test] fn sfs003_fail_empty() {
        assert_eq!(verdict_from_no_overlap(&[]), Sfs003Verdict::Fail);
    }

    // SFS-004 (dtype size match)
    #[test] fn sfs004_pass_f32_canonical() {
        // F32 tensor shape [3] → 3 * 4 = 12 bytes.
        let shape = [3_u64];
        assert_eq!(verdict_from_dtype_size_match(&shape, SfsDType::F32, 12), Sfs004Verdict::Pass);
    }
    #[test] fn sfs004_pass_f16_canonical() {
        let shape = [4_u64, 8];
        // 4 * 8 * 2 = 64.
        assert_eq!(verdict_from_dtype_size_match(&shape, SfsDType::F16, 64), Sfs004Verdict::Pass);
    }
    #[test] fn sfs004_pass_i8_canonical() {
        let shape = [128_u64];
        assert_eq!(verdict_from_dtype_size_match(&shape, SfsDType::I8, 128), Sfs004Verdict::Pass);
    }
    #[test] fn sfs004_fail_wrong_byte_count() {
        // Contract's stated falsifier: F32 shape=[3] but 11 bytes.
        let shape = [3_u64];
        assert_eq!(verdict_from_dtype_size_match(&shape, SfsDType::F32, 11), Sfs004Verdict::Fail);
    }
    #[test] fn sfs004_fail_zero_dim() {
        let shape = [3_u64, 0];
        assert_eq!(verdict_from_dtype_size_match(&shape, SfsDType::F32, 0), Sfs004Verdict::Fail);
    }
    #[test] fn sfs004_fail_overflow_shape() {
        let shape = [u64::MAX, 2];
        assert_eq!(verdict_from_dtype_size_match(&shape, SfsDType::F32, 0), Sfs004Verdict::Fail);
    }
    #[test] fn sfs004_fail_empty_shape() {
        assert_eq!(verdict_from_dtype_size_match(&[], SfsDType::F32, 0), Sfs004Verdict::Fail);
    }
    #[test] fn dtype_size_table() {
        assert_eq!(dtype_size(SfsDType::F16), 2);
        assert_eq!(dtype_size(SfsDType::F32), 4);
        assert_eq!(dtype_size(SfsDType::F64), 8);
        assert_eq!(dtype_size(SfsDType::Bf16), 2);
        assert_eq!(dtype_size(SfsDType::I8), 1);
        assert_eq!(dtype_size(SfsDType::U8), 1);
        assert_eq!(dtype_size(SfsDType::Bool), 1);
    }

    // SFS-005 (zero-copy mmap)
    #[test] fn sfs005_pass_canonical() {
        // 4GB tensor mmapped, zero heap, mmap size matches claimed.
        let four_gb = 4u64 * 1024 * 1024 * 1024;
        assert_eq!(verdict_from_zero_copy_mmap(0, four_gb, four_gb), Sfs005Verdict::Pass);
    }
    #[test] fn sfs005_fail_heap_copy() {
        // Heap allocation > 0 indicates the data was copied.
        assert_eq!(verdict_from_zero_copy_mmap(1, 1024, 1024), Sfs005Verdict::Fail);
    }
    #[test] fn sfs005_fail_size_mismatch() {
        // mmap returned wrong size.
        assert_eq!(verdict_from_zero_copy_mmap(0, 512, 1024), Sfs005Verdict::Fail);
    }
    #[test] fn sfs005_fail_zero_size() {
        assert_eq!(verdict_from_zero_copy_mmap(0, 0, 0), Sfs005Verdict::Fail);
    }

    // Provenance
    #[test] fn provenance_constants() {
        assert_eq!(AC_SFS_001_MAX_HEADER_SIZE, 100 * 1024 * 1024);
    }
}
