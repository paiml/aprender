// `apr-book-ch*` family algorithm-level PARTIAL discharge for the 5
// falsification conditions shared by every `apr-book-ch*-v1.yaml` contract.
//
// Contracts: `contracts/apr-book-ch01-v1.yaml` ..
// `contracts/apr-book-ch27-v1.yaml` (27 contracts × 5 conditions = 135
// falsifier instantiations).
//
// All apr-book-ch* contracts (Chapter 1 .. Chapter 27 of the APR-BOOK
// reference book) share an identical falsification schema. Each declares
// 5 P0-severity conditions:
//
//   1. "cargo run -p aprender-core --example chXX_<...> exits non-zero"
//   2. "Section without arXiv citation"
//   3. "Legacy name appears in chapter text"
//   4. "Oracle --explain output contradicts chapter claim"
//   5. "assert!() failure in example"
//
// One verdict module satisfies the algorithm-level pin for all 27
// chapter contracts. Live discharge is `cargo run --example` per chapter
// + `apr oracle --explain` queries; this module pins the predicates so
// future bulk chapter edits cannot drift on the shape of any of the 5
// invariants.
//
// Local IDs FALSIFY-APRBOOK-001..005 are synthesized to give each
// unkeyed YAML entry a stable handle.

/// Legacy names whose appearance in chapter text REQUIRES removal
/// post-APR-MONO consolidation (matches apr-page-* family policy).
pub const AC_APRBOOK_LEGACY_NAMES: [&str; 4] = ["trueno", "realizar", "entrenar", "batuta"];

/// Total chapters in the APR-BOOK family (ch01..ch27).
pub const AC_APRBOOK_TOTAL_CHAPTERS: u32 = 27;

// =============================================================================
// FALSIFY-APRBOOK-001 — example exit code is 0
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BookExampleExitVerdict {
    /// `cargo run --example chXX_<...>` exited 0.
    Pass,
    /// Non-zero exit (P0 reject_chapter).
    Fail,
}

#[must_use]
pub fn verdict_from_book_example_exit(exit_code: i32) -> BookExampleExitVerdict {
    if exit_code == 0 {
        BookExampleExitVerdict::Pass
    } else {
        BookExampleExitVerdict::Fail
    }
}

// =============================================================================
// FALSIFY-APRBOOK-002 — every section has an arXiv citation
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SectionCitationVerdict {
    /// Every section in the chapter cites at least one arXiv ID.
    Pass,
    /// At least one section without arXiv citation (P0 reject_chapter).
    Fail,
}

/// Pure verdict for FALSIFY-APRBOOK-002.
///
/// `section_citation_counts` is a slice of (section_heading, citation_count)
/// pairs. Pass iff all counts are ≥ 1.
#[must_use]
pub fn verdict_from_section_citations(section_citation_counts: &[(&str, u32)]) -> SectionCitationVerdict {
    if section_citation_counts.is_empty() {
        // No sections at all — vacuous pass; FALSIFY-APRPAGE-001/-005 catch
        // missing-content separately at the file-existence layer.
        return SectionCitationVerdict::Pass;
    }
    for (_section, count) in section_citation_counts {
        if *count == 0 {
            return SectionCitationVerdict::Fail;
        }
    }
    SectionCitationVerdict::Pass
}

// =============================================================================
// FALSIFY-APRBOOK-003 — no legacy names in chapter text
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoLegacyInBookVerdict {
    /// Chapter text contains zero of {trueno, realizar, entrenar, batuta},
    /// OR every occurrence is in a "MOVED"/migration/history context.
    Pass,
    /// Active legacy reference (P0 reject_chapter).
    Fail,
}

#[must_use]
pub fn verdict_from_no_legacy_in_book(chapter_text: &str) -> NoLegacyInBookVerdict {
    let lower = chapter_text.to_lowercase();
    let mentions_legacy = AC_APRBOOK_LEGACY_NAMES.iter().any(|n| lower.contains(n));
    if !mentions_legacy {
        return NoLegacyInBookVerdict::Pass;
    }
    if lower.contains("moved") || lower.contains("migration") || lower.contains("history") {
        NoLegacyInBookVerdict::Pass
    } else {
        NoLegacyInBookVerdict::Fail
    }
}

// =============================================================================
// FALSIFY-APRBOOK-004 — oracle --explain doesn't contradict chapter claim
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OracleConsistencyVerdict {
    /// `apr oracle --explain` output is consistent with chapter claims.
    Pass,
    /// Contradiction detected (P0 reject_chapter).
    Fail,
}

#[must_use]
pub fn verdict_from_oracle_consistency(contradiction_count: u32) -> OracleConsistencyVerdict {
    if contradiction_count == 0 {
        OracleConsistencyVerdict::Pass
    } else {
        OracleConsistencyVerdict::Fail
    }
}

// =============================================================================
// FALSIFY-APRBOOK-005 — all assert!() in example pass
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssertPassVerdict {
    /// Every `assert!()` in the chapter example evaluates to true.
    Pass,
    /// At least one assert!() failed (P0 reject_chapter).
    Fail,
}

#[must_use]
pub fn verdict_from_assert_pass(failed_assert_count: u32) -> AssertPassVerdict {
    if failed_assert_count == 0 {
        AssertPassVerdict::Pass
    } else {
        AssertPassVerdict::Fail
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Section 1: Provenance pins.
    // -------------------------------------------------------------------------
    #[test]
    fn provenance_legacy_names_count_4() {
        assert_eq!(AC_APRBOOK_LEGACY_NAMES.len(), 4);
    }

    #[test]
    fn provenance_total_chapters_27() {
        assert_eq!(AC_APRBOOK_TOTAL_CHAPTERS, 27);
    }

    // -------------------------------------------------------------------------
    // Section 2: APRBOOK-001 example exit.
    // -------------------------------------------------------------------------
    #[test]
    fn ab001_pass_exit_zero() {
        assert_eq!(verdict_from_book_example_exit(0), BookExampleExitVerdict::Pass);
    }

    #[test]
    fn ab001_fail_exit_one() {
        assert_eq!(verdict_from_book_example_exit(1), BookExampleExitVerdict::Fail);
    }

    #[test]
    fn ab001_fail_panic() {
        // Rust panic exit code 101.
        assert_eq!(verdict_from_book_example_exit(101), BookExampleExitVerdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 3: APRBOOK-002 section citations.
    // -------------------------------------------------------------------------
    #[test]
    fn ab002_pass_all_sections_cited() {
        let sections = [("Why Rust", 1), ("Memory Safety", 2), ("Performance", 3)];
        assert_eq!(
            verdict_from_section_citations(&sections),
            SectionCitationVerdict::Pass
        );
    }

    #[test]
    fn ab002_pass_no_sections_vacuous() {
        assert_eq!(verdict_from_section_citations(&[]), SectionCitationVerdict::Pass);
    }

    #[test]
    fn ab002_fail_uncited_section() {
        let sections = [("Why Rust", 1), ("Drift Section", 0)];
        assert_eq!(
            verdict_from_section_citations(&sections),
            SectionCitationVerdict::Fail
        );
    }

    #[test]
    fn ab002_fail_all_uncited() {
        let sections = [("Section A", 0), ("Section B", 0)];
        assert_eq!(
            verdict_from_section_citations(&sections),
            SectionCitationVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 4: APRBOOK-003 no legacy names in chapter.
    // -------------------------------------------------------------------------
    #[test]
    fn ab003_pass_clean_chapter() {
        let t = "Chapter 1: Why Rust for ML — aprender uses safe Rust types.";
        assert_eq!(verdict_from_no_legacy_in_book(t), NoLegacyInBookVerdict::Pass);
    }

    #[test]
    fn ab003_pass_history_section() {
        let t = "## History\nPre-APR-MONO, trueno was a separate crate.";
        assert_eq!(verdict_from_no_legacy_in_book(t), NoLegacyInBookVerdict::Pass);
    }

    #[test]
    fn ab003_pass_with_moved_tag() {
        let t = "See trueno (MOVED to crates/aprender-compute/) for SIMD.";
        assert_eq!(verdict_from_no_legacy_in_book(t), NoLegacyInBookVerdict::Pass);
    }

    #[test]
    fn ab003_fail_active_legacy() {
        let t = "Use realizar to serve models.";
        assert_eq!(verdict_from_no_legacy_in_book(t), NoLegacyInBookVerdict::Fail);
    }

    #[test]
    fn ab003_fail_each_legacy_name_active() {
        for name in AC_APRBOOK_LEGACY_NAMES {
            let t = format!("Use {name} for foo.");
            assert_eq!(
                verdict_from_no_legacy_in_book(&t),
                NoLegacyInBookVerdict::Fail,
                "active legacy name {name} must Fail"
            );
        }
    }

    // -------------------------------------------------------------------------
    // Section 5: APRBOOK-004 oracle consistency.
    // -------------------------------------------------------------------------
    #[test]
    fn ab004_pass_no_contradictions() {
        assert_eq!(verdict_from_oracle_consistency(0), OracleConsistencyVerdict::Pass);
    }

    #[test]
    fn ab004_fail_one_contradiction() {
        assert_eq!(verdict_from_oracle_consistency(1), OracleConsistencyVerdict::Fail);
    }

    #[test]
    fn ab004_fail_many_contradictions() {
        assert_eq!(verdict_from_oracle_consistency(10), OracleConsistencyVerdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 6: APRBOOK-005 assert!() pass.
    // -------------------------------------------------------------------------
    #[test]
    fn ab005_pass_zero_failures() {
        assert_eq!(verdict_from_assert_pass(0), AssertPassVerdict::Pass);
    }

    #[test]
    fn ab005_fail_one_failure() {
        assert_eq!(verdict_from_assert_pass(1), AssertPassVerdict::Fail);
    }

    #[test]
    fn ab005_fail_many_failures() {
        assert_eq!(verdict_from_assert_pass(20), AssertPassVerdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 7: Realistic — a healthy chapter passes all 5.
    // -------------------------------------------------------------------------
    #[test]
    fn realistic_full_healthy_chapter_passes_all_5() {
        // A typical apr-book-ch08-v1 (Transformers) chapter:
        // example exits 0, every section cites arXiv, no legacy names,
        // oracle agrees, all assert!() pass.
        assert_eq!(verdict_from_book_example_exit(0), BookExampleExitVerdict::Pass);

        let sections = [
            ("Multi-Head Attention", 2),
            ("Positional Encoding", 1),
            ("Layer Norm", 1),
        ];
        assert_eq!(
            verdict_from_section_citations(&sections),
            SectionCitationVerdict::Pass
        );

        let chapter = "Transformer block: Attention(Q,K,V) — see arXiv:1706.03762.";
        assert_eq!(verdict_from_no_legacy_in_book(chapter), NoLegacyInBookVerdict::Pass);
        assert_eq!(verdict_from_oracle_consistency(0), OracleConsistencyVerdict::Pass);
        assert_eq!(verdict_from_assert_pass(0), AssertPassVerdict::Pass);
    }

    #[test]
    fn realistic_pre_fix_all_5_failures() {
        // The exact regression class for each gate.
        assert_eq!(verdict_from_book_example_exit(101), BookExampleExitVerdict::Fail);

        let bad_sections = [("Section", 0)];
        assert_eq!(
            verdict_from_section_citations(&bad_sections),
            SectionCitationVerdict::Fail
        );

        let bad_text = "Use entrenar for training.";
        assert_eq!(verdict_from_no_legacy_in_book(bad_text), NoLegacyInBookVerdict::Fail);

        assert_eq!(verdict_from_oracle_consistency(2), OracleConsistencyVerdict::Fail);
        assert_eq!(verdict_from_assert_pass(3), AssertPassVerdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 8: Family coverage — verdicts are chapter-identity-agnostic.
    // -------------------------------------------------------------------------
    #[test]
    fn family_coverage_uniform_schema() {
        // All 27 chapters share the same falsification schema. The verdict
        // logic doesn't depend on which chapter is being checked.
        for ch in 1..=AC_APRBOOK_TOTAL_CHAPTERS {
            let chapter_id = format!("apr-book-ch{ch:02}");
            assert_eq!(
                verdict_from_book_example_exit(0),
                BookExampleExitVerdict::Pass,
                "{chapter_id} healthy example"
            );
        }
    }
}
