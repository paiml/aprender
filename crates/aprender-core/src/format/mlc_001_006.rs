// SHIP-TWO-001 — `apr-model-lifecycle-v1` algorithm-level PARTIAL
// discharge for FALSIFY-MLC-001..006.
//
// Contract: `contracts/apr-model-lifecycle-v1.yaml`.
// Spec: `docs/specifications/aprender-train/ship-two-models-spec.md`.
//
// ## What this file proves NOW (PARTIAL_ALGORITHM_LEVEL)
//
// Six model-lifecycle gates:
//
// - MLC-001 (cache content-addressed): two pulls of same model →
//   same cache path.
// - MLC-002 (import/export round-trip): export(import(F), F) is
//   bit-identical to F.
// - MLC-003 (quantization compresses): Q4_0 size < F32 size.
// - MLC-004 (merge preserves tensor count): merge(M, M) has same
//   tensor count.
// - MLC-005 (import never modifies source): SHA-256 unchanged after
//   import.
// - MLC-006 (partial download cache integrity): interrupted pull
//   leaves no file at final cache path.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MlcVerdict {
    Pass,
    Fail,
}

// -----------------------------------------------------------------------------
// Verdict 1: MLC-001 — cache is content-addressed.
// -----------------------------------------------------------------------------

/// Pass iff `path1 == path2` for two pulls of the same model.
#[must_use]
pub fn verdict_from_cache_content_addressed(path1: &str, path2: &str) -> MlcVerdict {
    if path1.is_empty() || path2.is_empty() {
        return MlcVerdict::Fail;
    }
    if path1 == path2 {
        MlcVerdict::Pass
    } else {
        MlcVerdict::Fail
    }
}

// -----------------------------------------------------------------------------
// Verdict 2: MLC-002 — import/export round-trip.
// -----------------------------------------------------------------------------

/// Pass iff `original_bytes == round_tripped_bytes`.
#[must_use]
pub fn verdict_from_roundtrip_bit_identical(
    original_bytes: &[u8],
    round_tripped_bytes: &[u8],
) -> MlcVerdict {
    if original_bytes.is_empty() {
        return MlcVerdict::Fail;
    }
    if original_bytes == round_tripped_bytes {
        MlcVerdict::Pass
    } else {
        MlcVerdict::Fail
    }
}

// -----------------------------------------------------------------------------
// Verdict 3: MLC-003 — quantization compresses.
// -----------------------------------------------------------------------------

/// Pass iff `quant_size_bytes < f32_size_bytes`.
#[must_use]
pub fn verdict_from_quantization_compresses(
    f32_size_bytes: u64,
    quant_size_bytes: u64,
) -> MlcVerdict {
    if f32_size_bytes == 0 {
        return MlcVerdict::Fail;
    }
    if quant_size_bytes < f32_size_bytes {
        MlcVerdict::Pass
    } else {
        MlcVerdict::Fail
    }
}

// -----------------------------------------------------------------------------
// Verdict 4: MLC-004 — merge preserves tensor count.
// -----------------------------------------------------------------------------

/// Pass iff `merged_count == max(input_count_a, input_count_b)`.
/// (For identical inputs, merged_count == input_count_a == input_count_b.)
#[must_use]
pub fn verdict_from_merge_tensor_count(
    input_count_a: usize,
    input_count_b: usize,
    merged_count: usize,
) -> MlcVerdict {
    if input_count_a == 0 || input_count_b == 0 {
        return MlcVerdict::Fail;
    }
    let expected = input_count_a.max(input_count_b);
    if merged_count == expected {
        MlcVerdict::Pass
    } else {
        MlcVerdict::Fail
    }
}

// -----------------------------------------------------------------------------
// Verdict 5: MLC-005 — import never modifies source.
// -----------------------------------------------------------------------------

#[must_use]
pub fn verdict_from_import_readonly(
    sha_before: &[u8; 32],
    sha_after: &[u8; 32],
) -> MlcVerdict {
    if sha_before == sha_after {
        MlcVerdict::Pass
    } else {
        MlcVerdict::Fail
    }
}

// -----------------------------------------------------------------------------
// Verdict 6: MLC-006 — partial download cache integrity.
// -----------------------------------------------------------------------------

/// `interrupted_at_pct` is the percentage at which pull was killed.
/// `final_path_exists` is whether the final cache path exists after kill.
///
/// Pass iff:
///   - interrupted_at_pct < 100 (genuine interrupt) AND final_path_exists == false,
///     OR
///   - interrupted_at_pct >= 100 (completed) AND final_path_exists == true.
#[must_use]
pub fn verdict_from_partial_download_integrity(
    interrupted_at_pct: u32,
    final_path_exists: bool,
) -> MlcVerdict {
    if interrupted_at_pct < 100 {
        // Was interrupted: final path must NOT exist.
        if final_path_exists {
            MlcVerdict::Fail
        } else {
            MlcVerdict::Pass
        }
    } else {
        // Completed: final path SHOULD exist.
        if final_path_exists {
            MlcVerdict::Pass
        } else {
            MlcVerdict::Fail
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Section 1: MLC-001 — cache content-addressed.
    // -------------------------------------------------------------------------
    #[test]
    fn mlc001_pass_same_path() {
        let p = "~/.cache/pacha/models/sha256-abcd1234.gguf";
        assert_eq!(
            verdict_from_cache_content_addressed(p, p),
            MlcVerdict::Pass
        );
    }

    #[test]
    fn mlc001_fail_paths_differ() {
        // Bug: cache path depends on timestamp.
        assert_eq!(
            verdict_from_cache_content_addressed(
                "~/.cache/pacha/models/m_2026-04-01.gguf",
                "~/.cache/pacha/models/m_2026-04-02.gguf"
            ),
            MlcVerdict::Fail
        );
    }

    #[test]
    fn mlc001_fail_empty_path() {
        assert_eq!(
            verdict_from_cache_content_addressed("", "/path"),
            MlcVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 2: MLC-002 — round-trip.
    // -------------------------------------------------------------------------
    #[test]
    fn mlc002_pass_byte_identical() {
        let bytes = vec![0xAB_u8; 1024];
        assert_eq!(
            verdict_from_roundtrip_bit_identical(&bytes, &bytes),
            MlcVerdict::Pass
        );
    }

    #[test]
    fn mlc002_fail_one_byte_changed() {
        let original = vec![1_u8, 2, 3, 4];
        let mut roundtrip = original.clone();
        roundtrip[2] = 99;
        assert_eq!(
            verdict_from_roundtrip_bit_identical(&original, &roundtrip),
            MlcVerdict::Fail
        );
    }

    #[test]
    fn mlc002_fail_length_mismatch() {
        let a = vec![1_u8, 2, 3];
        let b = vec![1_u8, 2];
        assert_eq!(
            verdict_from_roundtrip_bit_identical(&a, &b),
            MlcVerdict::Fail
        );
    }

    #[test]
    fn mlc002_fail_empty_original() {
        let empty: Vec<u8> = vec![];
        assert_eq!(
            verdict_from_roundtrip_bit_identical(&empty, &empty),
            MlcVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 3: MLC-003 — quantization compresses.
    // -------------------------------------------------------------------------
    #[test]
    fn mlc003_pass_q4_compresses_8x() {
        // F32 model 16GB → Q4_0 ~2GB.
        assert_eq!(
            verdict_from_quantization_compresses(16_000_000_000, 2_000_000_000),
            MlcVerdict::Pass
        );
    }

    #[test]
    fn mlc003_pass_marginal_compression() {
        // Even 1 byte smaller suffices.
        assert_eq!(
            verdict_from_quantization_compresses(1000, 999),
            MlcVerdict::Pass
        );
    }

    #[test]
    fn mlc003_fail_no_compression() {
        // Quantization made the file LARGER (overhead).
        assert_eq!(
            verdict_from_quantization_compresses(1000, 1100),
            MlcVerdict::Fail
        );
    }

    #[test]
    fn mlc003_fail_same_size() {
        assert_eq!(
            verdict_from_quantization_compresses(1000, 1000),
            MlcVerdict::Fail
        );
    }

    #[test]
    fn mlc003_fail_zero_f32() {
        assert_eq!(
            verdict_from_quantization_compresses(0, 0),
            MlcVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 4: MLC-004 — merge tensor count.
    // -------------------------------------------------------------------------
    #[test]
    fn mlc004_pass_identical_inputs() {
        // Merging 339 tensors with 339 tensors gives 339 tensors.
        assert_eq!(
            verdict_from_merge_tensor_count(339, 339, 339),
            MlcVerdict::Pass
        );
    }

    #[test]
    fn mlc004_pass_different_sizes_use_max() {
        assert_eq!(
            verdict_from_merge_tensor_count(339, 400, 400),
            MlcVerdict::Pass
        );
    }

    #[test]
    fn mlc004_fail_dropped_tensor() {
        // Bug: merge dropped some tensors.
        assert_eq!(
            verdict_from_merge_tensor_count(339, 339, 338),
            MlcVerdict::Fail
        );
    }

    #[test]
    fn mlc004_fail_duplicated_tensors() {
        assert_eq!(
            verdict_from_merge_tensor_count(339, 339, 678),
            MlcVerdict::Fail
        );
    }

    #[test]
    fn mlc004_fail_zero_input() {
        assert_eq!(
            verdict_from_merge_tensor_count(0, 100, 100),
            MlcVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 5: MLC-005 — import readonly.
    // -------------------------------------------------------------------------
    #[test]
    fn mlc005_pass_sha_unchanged() {
        let sha = [0xAB_u8; 32];
        assert_eq!(
            verdict_from_import_readonly(&sha, &sha),
            MlcVerdict::Pass
        );
    }

    #[test]
    fn mlc005_fail_sha_modified() {
        let before = [0x00_u8; 32];
        let mut after = before;
        after[5] = 0xFF;
        assert_eq!(
            verdict_from_import_readonly(&before, &after),
            MlcVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 6: MLC-006 — partial download cache integrity.
    // -------------------------------------------------------------------------
    #[test]
    fn mlc006_pass_interrupted_no_file() {
        // Killed at 50%, no file at final path.
        assert_eq!(
            verdict_from_partial_download_integrity(50, false),
            MlcVerdict::Pass
        );
    }

    #[test]
    fn mlc006_pass_completed_file_exists() {
        assert_eq!(
            verdict_from_partial_download_integrity(100, true),
            MlcVerdict::Pass
        );
    }

    #[test]
    fn mlc006_fail_interrupted_with_partial_file() {
        // The exact regression: partial file at final path.
        assert_eq!(
            verdict_from_partial_download_integrity(50, true),
            MlcVerdict::Fail
        );
    }

    #[test]
    fn mlc006_fail_completed_no_file() {
        assert_eq!(
            verdict_from_partial_download_integrity(100, false),
            MlcVerdict::Fail
        );
    }

    #[test]
    fn mlc006_pass_zero_pct_no_file() {
        // Killed before any bytes written.
        assert_eq!(
            verdict_from_partial_download_integrity(0, false),
            MlcVerdict::Pass
        );
    }

    // -------------------------------------------------------------------------
    // Section 7: Realistic — full lifecycle pipeline.
    // -------------------------------------------------------------------------
    #[test]
    fn realistic_full_lifecycle_pipeline() {
        // Synthesize a Qwen2.5-Coder lifecycle:
        let cache_path = "~/.cache/pacha/models/sha256-c0de1234.gguf";

        // MLC-001:
        assert_eq!(
            verdict_from_cache_content_addressed(cache_path, cache_path),
            MlcVerdict::Pass
        );
        // MLC-002:
        let bytes = vec![1_u8; 1024];
        assert_eq!(
            verdict_from_roundtrip_bit_identical(&bytes, &bytes),
            MlcVerdict::Pass
        );
        // MLC-003:
        assert_eq!(
            verdict_from_quantization_compresses(15_000_000_000, 4_200_000_000),
            MlcVerdict::Pass
        );
        // MLC-004:
        assert_eq!(
            verdict_from_merge_tensor_count(339, 339, 339),
            MlcVerdict::Pass
        );
        // MLC-005:
        let sha = [0xAB_u8; 32];
        assert_eq!(
            verdict_from_import_readonly(&sha, &sha),
            MlcVerdict::Pass
        );
        // MLC-006:
        assert_eq!(
            verdict_from_partial_download_integrity(100, true),
            MlcVerdict::Pass
        );
    }

    #[test]
    fn realistic_pre_fix_all_6_failures() {
        // Pre-fix: cache timestamp; reordered tensors; quant overhead;
        // merge dropped; import wrote; partial file at final.
        assert_eq!(
            verdict_from_cache_content_addressed("/a", "/b"),
            MlcVerdict::Fail
        );
        let original = vec![1_u8, 2, 3];
        let modified = vec![3_u8, 2, 1]; // reordered
        assert_eq!(
            verdict_from_roundtrip_bit_identical(&original, &modified),
            MlcVerdict::Fail
        );
        assert_eq!(
            verdict_from_quantization_compresses(100, 110),
            MlcVerdict::Fail
        );
        assert_eq!(
            verdict_from_merge_tensor_count(339, 339, 200),
            MlcVerdict::Fail
        );
        let sha_a = [0_u8; 32];
        let sha_b = [1_u8; 32];
        assert_eq!(
            verdict_from_import_readonly(&sha_a, &sha_b),
            MlcVerdict::Fail
        );
        assert_eq!(
            verdict_from_partial_download_integrity(50, true),
            MlcVerdict::Fail
        );
    }
}
