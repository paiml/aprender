// `apr-corpus-*` family algorithm-level PARTIAL discharge for the 5
// falsification conditions shared by every `apr-corpus-*-v1.yaml` contract.
//
// Contracts: `contracts/apr-corpus-*.yaml` (13 contracts × 5 conditions =
// 65 falsifier instantiations).
//
// All apr-corpus-* contracts (vllm, jax, hugging-face, lean, ludwig, tgi,
// databricks, etc.) share an identical schema. Each declares 5
// falsification conditions:
//
//   1. "Repository contains zero test files" (P0 — add_tests)
//   2. "No Rust equivalent mapping documented" (P1 — add_mapping)
//   3. "Corpus not indexed in oracle RAG" (P1 — reindex)
//   4. "Last commit older than 180 days (staleness)" (P1 — refresh_or_archive)
//   5. "Ground-truth pattern contradicts current aprender API" (P0 — update_pattern)
//
// One verdict module satisfies the algorithm-level pin for all 13
// apr-corpus contracts. Live discharge is `apr oracle --rag-index` +
// `gh repo` queries; this module pins the predicates so future bulk
// corpus operations cannot drift on the shape of any of the 5
// invariants.
//
// Local IDs FALSIFY-APRCORPUS-001..005 are synthesized to give each
// unkeyed YAML entry a stable handle.

/// Staleness threshold per FALSIFY-APRCORPUS-004.
pub const AC_APRCORPUS_STALENESS_DAYS: u32 = 180;

/// Minimum test file count per FALSIFY-APRCORPUS-001 (the contract test
/// is "Repository contains zero test files" — exactly 0 fails).
pub const AC_APRCORPUS_MIN_TEST_FILES: u32 = 1;

// =============================================================================
// FALSIFY-APRCORPUS-001 — repository has at least 1 test file
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CorpusTestFilesVerdict {
    /// At least 1 test file present.
    Pass,
    /// Zero test files (P0 regression).
    Fail,
}

#[must_use]
pub fn verdict_from_corpus_test_files(test_file_count: u32) -> CorpusTestFilesVerdict {
    if test_file_count >= AC_APRCORPUS_MIN_TEST_FILES {
        CorpusTestFilesVerdict::Pass
    } else {
        CorpusTestFilesVerdict::Fail
    }
}

// =============================================================================
// FALSIFY-APRCORPUS-002 — Rust equivalent mapping documented
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RustMappingVerdict {
    /// Mapping documented (e.g., README has a "Rust equivalent" / "aprender
    /// mapping" section, or every pattern file declares its Rust analogue).
    Pass,
    /// No mapping documented (P1).
    Fail,
}

#[must_use]
pub fn verdict_from_rust_mapping(mapping_documented: bool) -> RustMappingVerdict {
    if mapping_documented {
        RustMappingVerdict::Pass
    } else {
        RustMappingVerdict::Fail
    }
}

// =============================================================================
// FALSIFY-APRCORPUS-003 — corpus indexed in oracle RAG
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RagIndexedVerdict {
    /// `apr oracle --rag` finds at least 1 pattern in the corpus's domain.
    Pass,
    /// Domain query returns zero matches (P1).
    Fail,
}

#[must_use]
pub fn verdict_from_rag_indexed(rag_match_count: u32) -> RagIndexedVerdict {
    if rag_match_count >= 1 {
        RagIndexedVerdict::Pass
    } else {
        RagIndexedVerdict::Fail
    }
}

// =============================================================================
// FALSIFY-APRCORPUS-004 — last commit within 180 days
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CorpusFreshnessVerdict {
    /// Last commit ≤ 180 days ago.
    Pass,
    /// Last commit > 180 days ago (P1 — refresh_or_archive).
    Fail,
}

#[must_use]
pub fn verdict_from_corpus_freshness(days_since_last_commit: u32) -> CorpusFreshnessVerdict {
    if days_since_last_commit <= AC_APRCORPUS_STALENESS_DAYS {
        CorpusFreshnessVerdict::Pass
    } else {
        CorpusFreshnessVerdict::Fail
    }
}

// =============================================================================
// FALSIFY-APRCORPUS-005 — ground-truth pattern doesn't contradict aprender API
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApiConsistencyVerdict {
    /// All ground-truth patterns reference current aprender API surface.
    Pass,
    /// At least one pattern references removed/renamed API (P0 — update_pattern).
    Fail,
}

#[must_use]
pub fn verdict_from_api_consistency(contradicting_pattern_count: u32) -> ApiConsistencyVerdict {
    if contradicting_pattern_count == 0 {
        ApiConsistencyVerdict::Pass
    } else {
        ApiConsistencyVerdict::Fail
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Section 1: Provenance pins.
    // -------------------------------------------------------------------------
    #[test]
    fn provenance_staleness_180_days() {
        assert_eq!(AC_APRCORPUS_STALENESS_DAYS, 180);
    }

    #[test]
    fn provenance_min_test_files_is_1() {
        // Contract gate is "Repository contains ZERO test files" → fail.
        // Pass requires ≥ 1.
        assert_eq!(AC_APRCORPUS_MIN_TEST_FILES, 1);
    }

    // -------------------------------------------------------------------------
    // Section 2: APRCORPUS-001 test files.
    // -------------------------------------------------------------------------
    #[test]
    fn ac001_pass_one_test_file() {
        assert_eq!(verdict_from_corpus_test_files(1), CorpusTestFilesVerdict::Pass);
    }

    #[test]
    fn ac001_pass_many_test_files() {
        assert_eq!(verdict_from_corpus_test_files(100), CorpusTestFilesVerdict::Pass);
    }

    #[test]
    fn ac001_fail_zero_test_files() {
        assert_eq!(verdict_from_corpus_test_files(0), CorpusTestFilesVerdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 3: APRCORPUS-002 Rust mapping.
    // -------------------------------------------------------------------------
    #[test]
    fn ac002_pass_mapping_documented() {
        assert_eq!(verdict_from_rust_mapping(true), RustMappingVerdict::Pass);
    }

    #[test]
    fn ac002_fail_no_mapping() {
        assert_eq!(verdict_from_rust_mapping(false), RustMappingVerdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 4: APRCORPUS-003 RAG indexed.
    // -------------------------------------------------------------------------
    #[test]
    fn ac003_pass_one_match() {
        assert_eq!(verdict_from_rag_indexed(1), RagIndexedVerdict::Pass);
    }

    #[test]
    fn ac003_pass_many_matches() {
        assert_eq!(verdict_from_rag_indexed(50), RagIndexedVerdict::Pass);
    }

    #[test]
    fn ac003_fail_zero_matches() {
        assert_eq!(verdict_from_rag_indexed(0), RagIndexedVerdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 5: APRCORPUS-004 freshness.
    // -------------------------------------------------------------------------
    #[test]
    fn ac004_pass_recent_commit() {
        assert_eq!(verdict_from_corpus_freshness(7), CorpusFreshnessVerdict::Pass);
    }

    #[test]
    fn ac004_pass_at_threshold() {
        // The contract says "older than 180 days" — exactly 180 still passes.
        assert_eq!(verdict_from_corpus_freshness(180), CorpusFreshnessVerdict::Pass);
    }

    #[test]
    fn ac004_fail_one_day_past() {
        assert_eq!(verdict_from_corpus_freshness(181), CorpusFreshnessVerdict::Fail);
    }

    #[test]
    fn ac004_fail_long_stale() {
        assert_eq!(verdict_from_corpus_freshness(365), CorpusFreshnessVerdict::Fail);
    }

    #[test]
    fn ac004_pass_zero_days() {
        // Just-pushed corpus.
        assert_eq!(verdict_from_corpus_freshness(0), CorpusFreshnessVerdict::Pass);
    }

    // -------------------------------------------------------------------------
    // Section 6: APRCORPUS-005 API consistency.
    // -------------------------------------------------------------------------
    #[test]
    fn ac005_pass_no_contradictions() {
        assert_eq!(verdict_from_api_consistency(0), ApiConsistencyVerdict::Pass);
    }

    #[test]
    fn ac005_fail_one_contradiction() {
        assert_eq!(verdict_from_api_consistency(1), ApiConsistencyVerdict::Fail);
    }

    #[test]
    fn ac005_fail_many_contradictions() {
        assert_eq!(verdict_from_api_consistency(20), ApiConsistencyVerdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 7: Realistic — full healthy corpus passes all 5.
    // -------------------------------------------------------------------------
    #[test]
    fn realistic_full_healthy_corpus_passes_all_5() {
        // Maps to a typical paiml/<lang>-ground-truth-corpus repo state:
        // 35 test files, mapping documented, RAG returns 12 matches,
        // last commit 14 days ago, no API contradictions.
        assert_eq!(verdict_from_corpus_test_files(35), CorpusTestFilesVerdict::Pass);
        assert_eq!(verdict_from_rust_mapping(true), RustMappingVerdict::Pass);
        assert_eq!(verdict_from_rag_indexed(12), RagIndexedVerdict::Pass);
        assert_eq!(verdict_from_corpus_freshness(14), CorpusFreshnessVerdict::Pass);
        assert_eq!(verdict_from_api_consistency(0), ApiConsistencyVerdict::Pass);
    }

    #[test]
    fn realistic_pre_fix_all_5_failures() {
        // The exact regression class for each gate.
        assert_eq!(verdict_from_corpus_test_files(0), CorpusTestFilesVerdict::Fail);
        assert_eq!(verdict_from_rust_mapping(false), RustMappingVerdict::Fail);
        assert_eq!(verdict_from_rag_indexed(0), RagIndexedVerdict::Fail);
        assert_eq!(verdict_from_corpus_freshness(365), CorpusFreshnessVerdict::Fail);
        assert_eq!(verdict_from_api_consistency(3), ApiConsistencyVerdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 8: Family coverage — verdicts are corpus-identity-agnostic.
    // -------------------------------------------------------------------------
    #[test]
    fn family_coverage_uniform_schema() {
        // The 13 apr-corpus-* contracts share this falsification schema.
        // Verdict logic doesn't depend on which corpus is being checked.
        let corpora = [
            "vllm-ground-truth",
            "jax-ground-truth",
            "hugging-face-ground-truth",
            "lean-ground-truth",
            "ludwig-ground-truth",
            "tgi-ground-truth",
            "databricks-ground-truth",
            "databricks-scala-ground-truth",
            "mixed-python-rust-ground-truth",
            "mixed-rust-lean-ground-truth",
            "safe-lua-groundtruth",
            "tiny-model-ground-truth",
            "algorithm-competition-corpus",
        ];
        assert_eq!(corpora.len(), 13);
        for c in corpora {
            assert_eq!(
                verdict_from_corpus_test_files(10),
                CorpusTestFilesVerdict::Pass,
                "corpus {c} healthy"
            );
        }
    }
}
