// `compression-roundtrip-v1` algorithm-level PARTIAL discharge for the
// 2 LZ4/Zstd lossless-roundtrip falsifiers.
//
// Contract: `contracts/compression-roundtrip-v1.yaml`.
// Refs: LZ4 Frame format spec, Zstd compression format spec.
//
// ## What this file proves NOW (`PARTIAL_ALGORITHM_LEVEL`)
//
// Two pure decision predicates over (original_bytes, decompressed_bytes)
// pairs. Live discharge wraps these around the actual lz4/zstd
// implementations. Algorithm-level pinning prevents drift on the
// "byte-level equality, not floating-point tolerance" invariant.

/// LZ4 frame overhead estimate (~16 bytes per spec).
pub const AC_COMP_LZ4_OVERHEAD_BYTES: u32 = 16;

/// Zstd frame overhead estimate (~18 bytes per spec).
pub const AC_COMP_ZSTD_OVERHEAD_BYTES: u32 = 18;

// =============================================================================
// FALSIFY-COMP-001 — LZ4 lossless roundtrip
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lz4LosslessVerdict {
    /// decompress(compress(data)) == data byte-for-byte AND
    /// compressed_size <= original_size + LZ4_OVERHEAD.
    Pass,
    /// Either roundtrip diverges OR compressed size exceeds budget.
    Fail,
}

#[must_use]
pub fn verdict_from_lz4_lossless(
    original: &[u8],
    decompressed: &[u8],
    compressed_size: u32,
) -> Lz4LosslessVerdict {
    if original != decompressed {
        return Lz4LosslessVerdict::Fail;
    }
    let budget = original.len() as u64 + AC_COMP_LZ4_OVERHEAD_BYTES as u64;
    if compressed_size as u64 > budget {
        return Lz4LosslessVerdict::Fail;
    }
    Lz4LosslessVerdict::Pass
}

// =============================================================================
// FALSIFY-COMP-002 — Zstd lossless roundtrip
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZstdLosslessVerdict {
    /// decompress(compress(data)) == data byte-for-byte AND
    /// compressed_size <= original_size + ZSTD_OVERHEAD.
    Pass,
    /// Roundtrip diverges or size exceeds budget.
    Fail,
}

#[must_use]
pub fn verdict_from_zstd_lossless(
    original: &[u8],
    decompressed: &[u8],
    compressed_size: u32,
) -> ZstdLosslessVerdict {
    if original != decompressed {
        return ZstdLosslessVerdict::Fail;
    }
    let budget = original.len() as u64 + AC_COMP_ZSTD_OVERHEAD_BYTES as u64;
    if compressed_size as u64 > budget {
        return ZstdLosslessVerdict::Fail;
    }
    ZstdLosslessVerdict::Pass
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Section 1: Provenance pins.
    // -------------------------------------------------------------------------
    #[test]
    fn provenance_lz4_overhead_16() {
        assert_eq!(AC_COMP_LZ4_OVERHEAD_BYTES, 16);
    }

    #[test]
    fn provenance_zstd_overhead_18() {
        assert_eq!(AC_COMP_ZSTD_OVERHEAD_BYTES, 18);
    }

    // -------------------------------------------------------------------------
    // Section 2: COMP-001 LZ4 lossless.
    // -------------------------------------------------------------------------
    #[test]
    fn fc001_pass_identity_compressible() {
        // 1024-byte zeros → highly compressible.
        let data = vec![0u8; 1024];
        assert_eq!(
            verdict_from_lz4_lossless(&data, &data, 50),
            Lz4LosslessVerdict::Pass
        );
    }

    #[test]
    fn fc001_pass_incompressible() {
        // Random data: compressed_size ≈ original + 16 overhead.
        let data: Vec<u8> = (0..1024).map(|i| (i % 256) as u8).collect();
        assert_eq!(
            verdict_from_lz4_lossless(&data, &data, 1024 + 16),
            Lz4LosslessVerdict::Pass
        );
    }

    #[test]
    fn fc001_pass_empty() {
        let data: Vec<u8> = vec![];
        // Empty: compressed = overhead (16).
        assert_eq!(
            verdict_from_lz4_lossless(&data, &data, 16),
            Lz4LosslessVerdict::Pass
        );
    }

    #[test]
    fn fc001_fail_byte_drift() {
        let original = vec![0u8, 1, 2, 3];
        let decompressed = vec![0u8, 1, 2, 99]; // last byte wrong
        assert_eq!(
            verdict_from_lz4_lossless(&original, &decompressed, 100),
            Lz4LosslessVerdict::Fail
        );
    }

    #[test]
    fn fc001_fail_length_drift() {
        let original = vec![0u8; 10];
        let decompressed = vec![0u8; 9]; // truncated by 1
        assert_eq!(
            verdict_from_lz4_lossless(&original, &decompressed, 50),
            Lz4LosslessVerdict::Fail
        );
    }

    #[test]
    fn fc001_fail_compressed_exceeds_budget() {
        // 1024 + 16 = 1040 budget; 2000 exceeds.
        let data = vec![0u8; 1024];
        assert_eq!(
            verdict_from_lz4_lossless(&data, &data, 2000),
            Lz4LosslessVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 3: COMP-002 Zstd lossless.
    // -------------------------------------------------------------------------
    #[test]
    fn fz002_pass_identity_compressible() {
        let data = vec![0u8; 1024];
        assert_eq!(
            verdict_from_zstd_lossless(&data, &data, 30),
            ZstdLosslessVerdict::Pass
        );
    }

    #[test]
    fn fz002_pass_incompressible() {
        let data: Vec<u8> = (0..1024).map(|i| (i % 256) as u8).collect();
        assert_eq!(
            verdict_from_zstd_lossless(&data, &data, 1024 + 18),
            ZstdLosslessVerdict::Pass
        );
    }

    #[test]
    fn fz002_pass_empty() {
        let data: Vec<u8> = vec![];
        assert_eq!(
            verdict_from_zstd_lossless(&data, &data, 18),
            ZstdLosslessVerdict::Pass
        );
    }

    #[test]
    fn fz002_fail_byte_drift() {
        let original = vec![1u8, 2, 3];
        let decompressed = vec![1u8, 2, 4];
        assert_eq!(
            verdict_from_zstd_lossless(&original, &decompressed, 100),
            ZstdLosslessVerdict::Fail
        );
    }

    #[test]
    fn fz002_fail_compressed_exceeds_budget() {
        let data = vec![0u8; 100];
        // Budget = 100 + 18 = 118; 200 exceeds.
        assert_eq!(
            verdict_from_zstd_lossless(&data, &data, 200),
            ZstdLosslessVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 4: Realistic — full healthy compression passes both.
    // -------------------------------------------------------------------------
    #[test]
    fn realistic_healthy_roundtrip_passes_both() {
        // 100 MB tensor data, compressible 50:1 with LZ4, 100:1 with Zstd.
        let data = vec![0u8; 100_000_000];
        assert_eq!(
            verdict_from_lz4_lossless(&data, &data, 2_000_000),
            Lz4LosslessVerdict::Pass
        );
        assert_eq!(
            verdict_from_zstd_lossless(&data, &data, 1_000_000),
            ZstdLosslessVerdict::Pass
        );
    }

    #[test]
    fn realistic_pre_fix_both_failures() {
        // LZ4 dropped a byte at end of stream.
        let original = vec![1u8, 2, 3, 4, 5];
        let lz4_decompressed = vec![1u8, 2, 3, 4]; // truncated
        assert_eq!(
            verdict_from_lz4_lossless(&original, &lz4_decompressed, 50),
            Lz4LosslessVerdict::Fail
        );
        // Zstd corrupted middle byte.
        let zstd_decompressed = vec![1u8, 2, 99, 4, 5];
        assert_eq!(
            verdict_from_zstd_lossless(&original, &zstd_decompressed, 50),
            ZstdLosslessVerdict::Fail
        );
    }
}
