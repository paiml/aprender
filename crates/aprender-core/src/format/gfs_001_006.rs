// SHIP-TWO-001 — `gguf-format-safety-v1` algorithm-level PARTIAL
// discharge for FALSIFY-GGUF-001..006 (closes 6/6 sweep).
//
// Contract: `contracts/gguf-format-safety-v1.yaml`.
// Spec: GGUF binary format safety (CVE-2024-25664/25631 mitigations).

// ===========================================================================
// GGUF-001 — Magic check before allocation: bytes[0..4] == "GGUF"
// ===========================================================================

pub const AC_GGUF_001_MAGIC: [u8; 4] = [0x47, 0x47, 0x55, 0x46]; // "GGUF"

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gguf001Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_magic_validation(bytes: &[u8]) -> Gguf001Verdict {
    if bytes.len() < 4 { return Gguf001Verdict::Fail; }
    if bytes[..4] == AC_GGUF_001_MAGIC { Gguf001Verdict::Pass } else { Gguf001Verdict::Fail }
}

// ===========================================================================
// GGUF-002 — Tensor shape product bounded: product * dtype_size ≤ file_size
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gguf002Verdict { Pass, Fail }

/// Pass iff shape product * dtype_size ≤ file_size, with overflow guards.
#[must_use]
pub fn verdict_from_shape_product_bounded(
    shape: &[u64],
    dtype_size: u64,
    file_size: u64,
) -> Gguf002Verdict {
    if shape.is_empty() || shape.len() > 4 { return Gguf002Verdict::Fail; }
    if dtype_size == 0 || file_size == 0 { return Gguf002Verdict::Fail; }
    let mut product: u64 = 1;
    for &dim in shape {
        if dim == 0 { return Gguf002Verdict::Fail; }
        product = match product.checked_mul(dim) {
            Some(p) => p,
            None => return Gguf002Verdict::Fail, // overflow → Fail (the regression class)
        };
    }
    let total_size = match product.checked_mul(dtype_size) {
        Some(s) => s,
        None => return Gguf002Verdict::Fail,
    };
    if total_size > file_size { return Gguf002Verdict::Fail; }
    Gguf002Verdict::Pass
}

// ===========================================================================
// GGUF-003 — No out-of-bounds tensor read: offset + size ≤ file_size
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gguf003Verdict { Pass, Fail }

#[must_use]
pub const fn verdict_from_tensor_bounds(
    offset: u64,
    tensor_size: u64,
    file_size: u64,
) -> Gguf003Verdict {
    if file_size == 0 || tensor_size == 0 { return Gguf003Verdict::Fail; }
    let end = match offset.checked_add(tensor_size) {
        Some(e) => e,
        None => return Gguf003Verdict::Fail,
    };
    if end <= file_size { Gguf003Verdict::Pass } else { Gguf003Verdict::Fail }
}

// ===========================================================================
// GGUF-004 — String length checked before allocation: len < MAX_STRING_LEN
// ===========================================================================

pub const AC_GGUF_004_MAX_STRING_LEN: u64 = 65536; // 64KB cap (CVE-2024-25664)

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gguf004Verdict { Pass, Fail }

#[must_use]
pub const fn verdict_from_string_length_bounded(claimed_length: u64) -> Gguf004Verdict {
    if claimed_length == 0 { return Gguf004Verdict::Fail; } // zero-length keys/strings rejected
    if claimed_length > AC_GGUF_004_MAX_STRING_LEN { return Gguf004Verdict::Fail; }
    Gguf004Verdict::Pass
}

// ===========================================================================
// GGUF-005 — Alignment is power of two: alignment.is_power_of_two()
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gguf005Verdict { Pass, Fail }

#[must_use]
pub const fn verdict_from_alignment_power_of_two(alignment: u64) -> Gguf005Verdict {
    if alignment == 0 { return Gguf005Verdict::Fail; }
    if alignment.is_power_of_two() { Gguf005Verdict::Pass } else { Gguf005Verdict::Fail }
}

// ===========================================================================
// GGUF-006 — Version compatibility: version ∈ {2, 3}; reject 1 (deprecated) and >3 (future)
// ===========================================================================

pub const AC_GGUF_006_SUPPORTED_VERSIONS: [u32; 2] = [2, 3];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gguf006Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_version_compatibility(version: u32) -> Gguf006Verdict {
    if AC_GGUF_006_SUPPORTED_VERSIONS.contains(&version) {
        Gguf006Verdict::Pass
    } else {
        Gguf006Verdict::Fail
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // GGUF-001 (magic validation)
    #[test] fn gguf001_pass_canonical() {
        let bytes = [0x47, 0x47, 0x55, 0x46, 0x03, 0x00, 0x00, 0x00];
        assert_eq!(verdict_from_magic_validation(&bytes), Gguf001Verdict::Pass);
    }
    #[test] fn gguf001_fail_ggml_legacy() {
        // Legacy GGML format ("GGML") — must be rejected immediately.
        let bytes = [0x47, 0x47, 0x4D, 0x4C];
        assert_eq!(verdict_from_magic_validation(&bytes), Gguf001Verdict::Fail);
    }
    #[test] fn gguf001_fail_safetensors() {
        // SafeTensors files start with JSON length — different magic.
        let bytes = [0x00, 0x00, 0x00, 0x00];
        assert_eq!(verdict_from_magic_validation(&bytes), Gguf001Verdict::Fail);
    }
    #[test] fn gguf001_fail_too_short() {
        let bytes = [0x47, 0x47, 0x55]; // only 3 bytes
        assert_eq!(verdict_from_magic_validation(&bytes), Gguf001Verdict::Fail);
    }
    #[test] fn gguf001_fail_empty() {
        assert_eq!(verdict_from_magic_validation(&[]), Gguf001Verdict::Fail);
    }

    // GGUF-002 (shape product bounded)
    #[test] fn gguf002_pass_canonical() {
        // [1024, 1024] f32 = 4MB; file size 8MB.
        let shape = [1024_u64, 1024];
        assert_eq!(verdict_from_shape_product_bounded(&shape, 4, 8_388_608), Gguf002Verdict::Pass);
    }
    #[test] fn gguf002_fail_overflow_attack() {
        // The CVE attack: [2^32, 2^32, 1, 1] overflows u64 product.
        let shape = [1_u64 << 32, 1_u64 << 32, 1, 1];
        assert_eq!(
            verdict_from_shape_product_bounded(&shape, 4, u64::MAX),
            Gguf002Verdict::Fail
        );
    }
    #[test] fn gguf002_fail_exceeds_file_size() {
        // [2048, 2048] f32 = 16MB; file claims 1KB.
        let shape = [2048_u64, 2048];
        assert_eq!(
            verdict_from_shape_product_bounded(&shape, 4, 1024),
            Gguf002Verdict::Fail
        );
    }
    #[test] fn gguf002_fail_zero_dim() {
        let shape = [1024_u64, 0];
        assert_eq!(verdict_from_shape_product_bounded(&shape, 4, 1024), Gguf002Verdict::Fail);
    }
    #[test] fn gguf002_fail_too_many_dims() {
        // GGUF supports n_dims ∈ [1, 4]; 5+ rejected.
        let shape = [2_u64, 2, 2, 2, 2];
        assert_eq!(verdict_from_shape_product_bounded(&shape, 4, 1024), Gguf002Verdict::Fail);
    }
    #[test] fn gguf002_fail_empty() {
        assert_eq!(verdict_from_shape_product_bounded(&[], 4, 1024), Gguf002Verdict::Fail);
    }

    // GGUF-003 (tensor bounds)
    #[test] fn gguf003_pass_canonical() {
        // offset=1024, size=4096, file=8192 → end=5120 ≤ 8192.
        assert_eq!(verdict_from_tensor_bounds(1024, 4096, 8192), Gguf003Verdict::Pass);
    }
    #[test] fn gguf003_fail_offset_beyond_file() {
        // CVE-2024-25631: offset > file_size.
        assert_eq!(verdict_from_tensor_bounds(10000, 100, 8192), Gguf003Verdict::Fail);
    }
    #[test] fn gguf003_fail_offset_overflow() {
        // offset + size overflows u64.
        assert_eq!(
            verdict_from_tensor_bounds(u64::MAX - 50, 100, u64::MAX),
            Gguf003Verdict::Fail
        );
    }
    #[test] fn gguf003_fail_zero_file_size() {
        assert_eq!(verdict_from_tensor_bounds(0, 100, 0), Gguf003Verdict::Fail);
    }
    #[test] fn gguf003_pass_at_end_boundary() {
        // offset + size == file_size (exactly).
        assert_eq!(verdict_from_tensor_bounds(8000, 192, 8192), Gguf003Verdict::Pass);
    }

    // GGUF-004 (string length)
    #[test] fn gguf004_pass_canonical() {
        assert_eq!(verdict_from_string_length_bounded(256), Gguf004Verdict::Pass);
    }
    #[test] fn gguf004_pass_at_max() {
        assert_eq!(verdict_from_string_length_bounded(65536), Gguf004Verdict::Pass);
    }
    #[test] fn gguf004_fail_above_max() {
        // CVE-2024-25664: claim 4GB length, attempt to alloc.
        assert_eq!(verdict_from_string_length_bounded(4_294_967_295), Gguf004Verdict::Fail);
    }
    #[test] fn gguf004_fail_zero() {
        // Zero-length strings rejected (semantically meaningless metadata key).
        assert_eq!(verdict_from_string_length_bounded(0), Gguf004Verdict::Fail);
    }

    // GGUF-005 (alignment power of two)
    #[test] fn gguf005_pass_canonical_32() {
        // GGUF v3 default alignment.
        assert_eq!(verdict_from_alignment_power_of_two(32), Gguf005Verdict::Pass);
    }
    #[test] fn gguf005_pass_all_powers_of_2() {
        for &a in &[1_u64, 2, 4, 8, 16, 32, 64, 128, 256, 1024, 4096] {
            assert_eq!(verdict_from_alignment_power_of_two(a), Gguf005Verdict::Pass);
        }
    }
    #[test] fn gguf005_fail_non_power_of_2() {
        // Contract's stated falsifier: alignment=7.
        assert_eq!(verdict_from_alignment_power_of_two(7), Gguf005Verdict::Fail);
    }
    #[test] fn gguf005_fail_3() {
        assert_eq!(verdict_from_alignment_power_of_two(3), Gguf005Verdict::Fail);
    }
    #[test] fn gguf005_fail_zero() {
        assert_eq!(verdict_from_alignment_power_of_two(0), Gguf005Verdict::Fail);
    }

    // GGUF-006 (version compatibility)
    #[test] fn gguf006_pass_v2() {
        assert_eq!(verdict_from_version_compatibility(2), Gguf006Verdict::Pass);
    }
    #[test] fn gguf006_pass_v3() {
        assert_eq!(verdict_from_version_compatibility(3), Gguf006Verdict::Pass);
    }
    #[test] fn gguf006_fail_v1_deprecated() {
        // Version 1 deprecated.
        assert_eq!(verdict_from_version_compatibility(1), Gguf006Verdict::Fail);
    }
    #[test] fn gguf006_fail_v4_future() {
        // Future version — reject to prevent silent misparse.
        assert_eq!(verdict_from_version_compatibility(4), Gguf006Verdict::Fail);
    }
    #[test] fn gguf006_fail_v0() {
        assert_eq!(verdict_from_version_compatibility(0), Gguf006Verdict::Fail);
    }
    #[test] fn gguf006_fail_max_u32() {
        assert_eq!(verdict_from_version_compatibility(u32::MAX), Gguf006Verdict::Fail);
    }

    // Provenance
    #[test] fn provenance_constants() {
        assert_eq!(AC_GGUF_001_MAGIC, [0x47, 0x47, 0x55, 0x46]);
        assert_eq!(AC_GGUF_004_MAX_STRING_LEN, 65536);
        assert_eq!(AC_GGUF_006_SUPPORTED_VERSIONS, [2, 3]);
    }
}
