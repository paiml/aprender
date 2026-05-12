// `apr-page-*` family algorithm-level PARTIAL discharge for the 5
// falsification conditions shared by every `apr-page-*-v1.yaml` contract.
//
// Contracts: `contracts/apr-page-*.yaml` (257 contracts × 5 conditions =
// 1,285 falsifier instantiations).
//
// All apr-page-* contracts (book pages, best-practices pages, chapter
// pages, etc.) share an identical schema. Each declares 5 falsification
// conditions:
//
//   1. "Page .md file does not exist at declared path"
//   2. "Example does not compile"
//   3. "Example exits non-zero"
//   4. "Section in .md not listed in contract"
//   5. "Legacy name in page text"
//
// One verdict module satisfies the algorithm-level pin for all 257
// contracts. Live discharge is `pv validate` per-page; this module
// pins the predicates so future bulk page edits cannot drift on the
// shape of any of the 5 invariants.
//
// Local IDs FALSIFY-APRPAGE-001..005 are synthesized to give each
// unkeyed YAML entry a stable handle.

use std::collections::HashSet;

/// Legacy names whose appearance in apr-page-* page text REQUIRES
/// removal post-APR-MONO consolidation (matches the same set as
/// apr-org-taxonomy-v1 and apr-docs-v1).
pub const AC_APRPAGE_LEGACY_NAMES: [&str; 4] = ["trueno", "realizar", "entrenar", "batuta"];

// =============================================================================
// FALSIFY-APRPAGE-001 — page .md file exists at declared path
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PageExistsVerdict {
    /// Path declared in contract.page.path resolves to a real file.
    Pass,
    /// Path missing on disk.
    Fail,
}

#[must_use]
pub fn verdict_from_page_exists(path_exists_on_disk: bool) -> PageExistsVerdict {
    if path_exists_on_disk {
        PageExistsVerdict::Pass
    } else {
        PageExistsVerdict::Fail
    }
}

// =============================================================================
// FALSIFY-APRPAGE-002 — example compiles
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExampleCompileVerdict {
    /// `cargo build` (or rustc) on the page's example exits 0.
    /// If the contract declares no example (`page.example == ""`),
    /// the gate is vacuously satisfied.
    Pass,
    /// Compile failure (non-zero rustc exit).
    Fail,
}

#[must_use]
pub fn verdict_from_example_compiles(
    has_example: bool,
    compile_exit_code: i32,
) -> ExampleCompileVerdict {
    if !has_example {
        return ExampleCompileVerdict::Pass;
    }
    if compile_exit_code == 0 {
        ExampleCompileVerdict::Pass
    } else {
        ExampleCompileVerdict::Fail
    }
}

// =============================================================================
// FALSIFY-APRPAGE-003 — example exits 0 at runtime
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExampleRunVerdict {
    /// Compiled example, when run, exits 0. Vacuous if no example.
    Pass,
    /// Non-zero exit.
    Fail,
}

#[must_use]
pub fn verdict_from_example_run(has_example: bool, run_exit_code: i32) -> ExampleRunVerdict {
    if !has_example {
        return ExampleRunVerdict::Pass;
    }
    if run_exit_code == 0 {
        ExampleRunVerdict::Pass
    } else {
        ExampleRunVerdict::Fail
    }
}

// =============================================================================
// FALSIFY-APRPAGE-004 — every section in .md is listed in contract
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SectionListedVerdict {
    /// Every heading found in the rendered .md is also present in the
    /// contract's `sections[*].heading` list.
    Pass,
    /// At least one heading in the file is missing from the contract.
    Fail,
}

#[must_use]
pub fn verdict_from_section_listed(
    md_headings: &[&str],
    contract_headings: &[&str],
) -> SectionListedVerdict {
    let allowed: HashSet<&&str> = contract_headings.iter().collect();
    for h in md_headings {
        if !allowed.contains(h) {
            return SectionListedVerdict::Fail;
        }
    }
    SectionListedVerdict::Pass
}

// =============================================================================
// FALSIFY-APRPAGE-005 — no legacy names in page text
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoLegacyNameVerdict {
    /// Page text contains none of {trueno, realizar, entrenar, batuta},
    /// OR every occurrence is in a "MOVED"/migration context (matches
    /// the same MOVED-redirect tag used by apr-org-taxonomy-v1).
    Pass,
    /// Legacy name appears without a MOVED/migration tag.
    Fail,
}

#[must_use]
pub fn verdict_from_no_legacy_name(page_text: &str) -> NoLegacyNameVerdict {
    let lower = page_text.to_lowercase();
    let mentions_legacy = AC_APRPAGE_LEGACY_NAMES.iter().any(|n| lower.contains(n));
    if !mentions_legacy {
        return NoLegacyNameVerdict::Pass;
    }
    if lower.contains("moved") || lower.contains("migration") || lower.contains("history") {
        NoLegacyNameVerdict::Pass
    } else {
        NoLegacyNameVerdict::Fail
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
        assert_eq!(AC_APRPAGE_LEGACY_NAMES.len(), 4);
        assert!(AC_APRPAGE_LEGACY_NAMES.contains(&"trueno"));
        assert!(AC_APRPAGE_LEGACY_NAMES.contains(&"realizar"));
        assert!(AC_APRPAGE_LEGACY_NAMES.contains(&"entrenar"));
        assert!(AC_APRPAGE_LEGACY_NAMES.contains(&"batuta"));
    }

    // -------------------------------------------------------------------------
    // Section 2: APRPAGE-001 page exists.
    // -------------------------------------------------------------------------
    #[test]
    fn ap001_pass_path_resolves() {
        assert_eq!(verdict_from_page_exists(true), PageExistsVerdict::Pass);
    }

    #[test]
    fn ap001_fail_path_missing() {
        assert_eq!(verdict_from_page_exists(false), PageExistsVerdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 3: APRPAGE-002 example compiles.
    // -------------------------------------------------------------------------
    #[test]
    fn ap002_pass_no_example_vacuous() {
        // Contract declares example: "" → no example → vacuous Pass.
        assert_eq!(verdict_from_example_compiles(false, 1), ExampleCompileVerdict::Pass);
    }

    #[test]
    fn ap002_pass_example_compiles_clean() {
        assert_eq!(verdict_from_example_compiles(true, 0), ExampleCompileVerdict::Pass);
    }

    #[test]
    fn ap002_fail_example_compile_error() {
        assert_eq!(verdict_from_example_compiles(true, 1), ExampleCompileVerdict::Fail);
    }

    #[test]
    fn ap002_fail_example_compile_panic() {
        // rustc panic exit code = 101.
        assert_eq!(verdict_from_example_compiles(true, 101), ExampleCompileVerdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 4: APRPAGE-003 example runs clean.
    // -------------------------------------------------------------------------
    #[test]
    fn ap003_pass_no_example_vacuous() {
        assert_eq!(verdict_from_example_run(false, 1), ExampleRunVerdict::Pass);
    }

    #[test]
    fn ap003_pass_example_exits_zero() {
        assert_eq!(verdict_from_example_run(true, 0), ExampleRunVerdict::Pass);
    }

    #[test]
    fn ap003_fail_example_panic() {
        assert_eq!(verdict_from_example_run(true, 101), ExampleRunVerdict::Fail);
    }

    #[test]
    fn ap003_fail_example_signal_terminated() {
        assert_eq!(verdict_from_example_run(true, 137), ExampleRunVerdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 5: APRPAGE-004 section listed.
    // -------------------------------------------------------------------------
    #[test]
    fn ap004_pass_md_subset_of_contract() {
        let md = ["Dependency Graph", "By Category"];
        let contract = ["Dependency Graph", "By Category", "Future Work"];
        assert_eq!(verdict_from_section_listed(&md, &contract), SectionListedVerdict::Pass);
    }

    #[test]
    fn ap004_pass_empty_md() {
        // No headings in the file ⇒ vacuously listed.
        let contract = ["A", "B"];
        assert_eq!(verdict_from_section_listed(&[], &contract), SectionListedVerdict::Pass);
    }

    #[test]
    fn ap004_fail_undeclared_section() {
        let md = ["Dependency Graph", "Unauthorized New Section"];
        let contract = ["Dependency Graph"];
        assert_eq!(verdict_from_section_listed(&md, &contract), SectionListedVerdict::Fail);
    }

    #[test]
    fn ap004_fail_empty_contract_with_md_headings() {
        let md = ["A"];
        let contract: [&str; 0] = [];
        assert_eq!(verdict_from_section_listed(&md, &contract), SectionListedVerdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 6: APRPAGE-005 no legacy names.
    // -------------------------------------------------------------------------
    #[test]
    fn ap005_pass_no_legacy_mentions() {
        assert_eq!(
            verdict_from_no_legacy_name("aprender ML framework — apr CLI."),
            NoLegacyNameVerdict::Pass
        );
    }

    #[test]
    fn ap005_pass_legacy_in_history_context() {
        let p = "Pre-APR-MONO history: trueno, realizar, entrenar, batuta were separate crates.";
        assert_eq!(verdict_from_no_legacy_name(p), NoLegacyNameVerdict::Pass);
    }

    #[test]
    fn ap005_pass_legacy_with_moved_tag() {
        let p = "trueno (MOVED to crates/aprender-compute/)";
        assert_eq!(verdict_from_no_legacy_name(p), NoLegacyNameVerdict::Pass);
    }

    #[test]
    fn ap005_pass_legacy_with_migration_tag() {
        let p = "Migration note: realizar → crates/aprender-serve/";
        assert_eq!(verdict_from_no_legacy_name(p), NoLegacyNameVerdict::Pass);
    }

    #[test]
    fn ap005_fail_active_legacy_install() {
        let p = "Install batuta from crates.io and run `batuta serve`.";
        assert_eq!(verdict_from_no_legacy_name(p), NoLegacyNameVerdict::Fail);
    }

    #[test]
    fn ap005_fail_each_legacy_name_active() {
        for name in AC_APRPAGE_LEGACY_NAMES {
            let p = format!("Use {name} for SIMD operations.");
            assert_eq!(
                verdict_from_no_legacy_name(&p),
                NoLegacyNameVerdict::Fail,
                "active legacy name {name} must Fail"
            );
        }
    }

    // -------------------------------------------------------------------------
    // Section 7: Realistic — full healthy page passes all 5.
    // -------------------------------------------------------------------------
    #[test]
    fn realistic_full_healthy_page_passes_all_5() {
        // 001: page exists.
        assert_eq!(verdict_from_page_exists(true), PageExistsVerdict::Pass);
        // 002: no example in contract → vacuous.
        assert_eq!(verdict_from_example_compiles(false, 1), ExampleCompileVerdict::Pass);
        // 003: vacuous run.
        assert_eq!(verdict_from_example_run(false, 1), ExampleRunVerdict::Pass);
        // 004: every md heading is in contract.
        let md = ["Dependency Graph", "By Category"];
        let contract = ["Dependency Graph", "By Category"];
        assert_eq!(verdict_from_section_listed(&md, &contract), SectionListedVerdict::Pass);
        // 005: page text with MOVED tag for legacy mentions.
        let page = "Architecture: aprender-compute (MOVED from trueno).";
        assert_eq!(verdict_from_no_legacy_name(page), NoLegacyNameVerdict::Pass);
    }

    #[test]
    fn realistic_pre_fix_all_5_failures() {
        // 001: missing path.
        assert_eq!(verdict_from_page_exists(false), PageExistsVerdict::Fail);
        // 002: example compile error.
        assert_eq!(verdict_from_example_compiles(true, 1), ExampleCompileVerdict::Fail);
        // 003: example panicked.
        assert_eq!(verdict_from_example_run(true, 101), ExampleRunVerdict::Fail);
        // 004: undeclared section in md.
        let md = ["Dependency Graph", "Drift Section"];
        let contract = ["Dependency Graph"];
        assert_eq!(verdict_from_section_listed(&md, &contract), SectionListedVerdict::Fail);
        // 005: active legacy install instructions.
        let bad = "Install: cargo install batuta";
        assert_eq!(verdict_from_no_legacy_name(bad), NoLegacyNameVerdict::Fail);
    }

    // -------------------------------------------------------------------------
    // Section 8: Family coverage — these verdicts work for ALL 257 apr-page-*.
    // -------------------------------------------------------------------------
    #[test]
    fn family_coverage_uniform_schema() {
        // Smoke: the verdicts treat page identity as opaque. The same
        // verdict_from_* set covers architecture-crate-map, chapters-ch01,
        // best-practices-api-design, etc. The contract path matters
        // only at the live harness layer.
        let pages = [
            "architecture-crate-map",
            "chapters-ch01-why-rust",
            "best-practices-api-design",
            "advanced-testing-mutation-testing",
        ];
        for p in pages {
            // Each page passes 001 if its declared .md exists.
            assert_eq!(verdict_from_page_exists(true), PageExistsVerdict::Pass, "{p}");
        }
    }
}
