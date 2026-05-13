// SHIP-TWO-001 — `apr-list-disk-reconciliation-v1` algorithm-level
// PARTIAL discharge for FALSIFY-LIST-DISK-001..003.
//
// Contract: `contracts/apr-list-disk-reconciliation-v1.yaml`.
// Spec: `docs/specifications/aprender-train/ship-two-models-spec.md`.
//
// ## What this file proves NOW (PARTIAL_ALGORITHM_LEVEL)
//
// Three apr-list disk-reconciliation gates:
//
// - LIST-DISK-001 (≥ 1 entry when disk has cached files).
// - LIST-DISK-002 (apr list --json total ≥ disk file count).
// - LIST-DISK-003 (apr list text doesn't claim "No cached models found"
//   when disk has files).

/// Sentinel string emitted by `apr list` when the catalog is empty.
pub const AC_LISTD_003_EMPTY_SENTINEL: &str = "No cached models found";

/// Recognized model file extensions (case-insensitive).
pub const AC_LISTD_MODEL_EXTENSIONS: [&str; 3] = ["gguf", "apr", "safetensors"];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ListdVerdict {
    Pass,
    Fail,
}

// -----------------------------------------------------------------------------
// In-module reference helpers.
// -----------------------------------------------------------------------------

/// True iff `path` ends in one of the recognized model extensions
/// (case-insensitive).
#[must_use]
pub fn is_model_file(path: &str) -> bool {
    let lower = path.to_ascii_lowercase();
    AC_LISTD_MODEL_EXTENSIONS
        .iter()
        .any(|ext| lower.ends_with(&format!(".{ext}")))
}

/// Count model files (by path string list).
#[must_use]
pub fn count_model_files(paths: &[&str]) -> usize {
    paths.iter().filter(|p| is_model_file(p)).count()
}

// -----------------------------------------------------------------------------
// Verdict 1: LIST-DISK-001 — ≥ 1 entry when disk has files.
// -----------------------------------------------------------------------------

/// Pass iff:
///   - disk_count == 0 (vacuously true), OR
///   - disk_count > 0 AND list_count ≥ 1.
#[must_use]
pub fn verdict_from_at_least_one_when_disk_nonempty(
    disk_count: usize,
    list_count: usize,
) -> ListdVerdict {
    if disk_count == 0 {
        return ListdVerdict::Pass;
    }
    if list_count >= 1 {
        ListdVerdict::Pass
    } else {
        ListdVerdict::Fail
    }
}

// -----------------------------------------------------------------------------
// Verdict 2: LIST-DISK-002 — apr list total ≥ disk count.
// -----------------------------------------------------------------------------

/// Pass iff `list_total >= disk_count` (list may include catalog
/// entries with no on-disk file, but must not undercount disk).
#[must_use]
pub fn verdict_from_total_covers_disk(list_total: usize, disk_count: usize) -> ListdVerdict {
    if list_total >= disk_count {
        ListdVerdict::Pass
    } else {
        ListdVerdict::Fail
    }
}

// -----------------------------------------------------------------------------
// Verdict 3: LIST-DISK-003 — text doesn't say "No cached models found".
// -----------------------------------------------------------------------------

/// Pass iff `text_output` does NOT contain the empty-catalog sentinel,
/// when `disk_count > 0`. When `disk_count == 0`, the sentinel is OK.
#[must_use]
pub fn verdict_from_text_not_empty_sentinel(
    text_output: &str,
    disk_count: usize,
) -> ListdVerdict {
    let has_sentinel = text_output.contains(AC_LISTD_003_EMPTY_SENTINEL);
    if disk_count == 0 {
        // Empty disk → either sentinel or empty body is fine.
        return ListdVerdict::Pass;
    }
    if has_sentinel {
        ListdVerdict::Fail
    } else {
        ListdVerdict::Pass
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // Section 1: Provenance pins.
    // -------------------------------------------------------------------------
    #[test]
    fn provenance_empty_sentinel() {
        assert_eq!(AC_LISTD_003_EMPTY_SENTINEL, "No cached models found");
    }

    #[test]
    fn provenance_model_extensions() {
        assert!(AC_LISTD_MODEL_EXTENSIONS.contains(&"gguf"));
        assert!(AC_LISTD_MODEL_EXTENSIONS.contains(&"apr"));
        assert!(AC_LISTD_MODEL_EXTENSIONS.contains(&"safetensors"));
    }

    // -------------------------------------------------------------------------
    // Section 2: is_model_file / count_model_files.
    // -------------------------------------------------------------------------
    #[test]
    fn domain_is_model_file_basic() {
        assert!(is_model_file("model.gguf"));
        assert!(is_model_file("model.apr"));
        assert!(is_model_file("model.safetensors"));
    }

    #[test]
    fn domain_is_model_file_case_insensitive() {
        assert!(is_model_file("Model.GGUF"));
        assert!(is_model_file("X.SAFETENSORS"));
    }

    #[test]
    fn domain_is_model_file_rejects_others() {
        assert!(!is_model_file("README.md"));
        assert!(!is_model_file("model.pt"));
        assert!(!is_model_file(""));
    }

    #[test]
    fn domain_count_model_files() {
        let paths = vec![
            "model1.gguf",
            "README.md",
            "weights.safetensors",
            "config.json",
            "M2.apr",
        ];
        assert_eq!(count_model_files(&paths), 3);
    }

    // -------------------------------------------------------------------------
    // Section 3: LIST-DISK-001 — ≥ 1 entry when disk has files.
    // -------------------------------------------------------------------------
    #[test]
    fn list001_pass_disk_empty_list_empty() {
        // Vacuous: no disk files, no list entries.
        assert_eq!(
            verdict_from_at_least_one_when_disk_nonempty(0, 0),
            ListdVerdict::Pass
        );
    }

    #[test]
    fn list001_pass_disk_empty_list_nonempty() {
        // Catalog has entries with no disk files (downloaded later).
        assert_eq!(
            verdict_from_at_least_one_when_disk_nonempty(0, 5),
            ListdVerdict::Pass
        );
    }

    #[test]
    fn list001_pass_one_disk_one_list() {
        assert_eq!(
            verdict_from_at_least_one_when_disk_nonempty(1, 1),
            ListdVerdict::Pass
        );
    }

    #[test]
    fn list001_pass_many_disk_one_list() {
        // Even single-entry list covers disk reconciliation requirement.
        assert_eq!(
            verdict_from_at_least_one_when_disk_nonempty(5, 1),
            ListdVerdict::Pass
        );
    }

    #[test]
    fn list001_fail_disk_nonempty_list_zero() {
        // The exact regression: disk has files, list shows nothing.
        assert_eq!(
            verdict_from_at_least_one_when_disk_nonempty(3, 0),
            ListdVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 4: LIST-DISK-002 — total ≥ disk count.
    // -------------------------------------------------------------------------
    #[test]
    fn list002_pass_total_equals_disk() {
        assert_eq!(
            verdict_from_total_covers_disk(5, 5),
            ListdVerdict::Pass
        );
    }

    #[test]
    fn list002_pass_total_exceeds_disk() {
        // Catalog entries for not-yet-downloaded models.
        assert_eq!(
            verdict_from_total_covers_disk(10, 5),
            ListdVerdict::Pass
        );
    }

    #[test]
    fn list002_pass_both_zero() {
        assert_eq!(
            verdict_from_total_covers_disk(0, 0),
            ListdVerdict::Pass
        );
    }

    #[test]
    fn list002_fail_undercount() {
        // Bug: list shows fewer than disk.
        assert_eq!(
            verdict_from_total_covers_disk(2, 5),
            ListdVerdict::Fail
        );
    }

    #[test]
    fn list002_fail_zero_total_with_disk() {
        assert_eq!(
            verdict_from_total_covers_disk(0, 3),
            ListdVerdict::Fail
        );
    }

    // -------------------------------------------------------------------------
    // Section 5: LIST-DISK-003 — text not empty sentinel when disk nonempty.
    // -------------------------------------------------------------------------
    #[test]
    fn list003_pass_disk_empty_sentinel_ok() {
        let text = "No cached models found";
        assert_eq!(
            verdict_from_text_not_empty_sentinel(text, 0),
            ListdVerdict::Pass
        );
    }

    #[test]
    fn list003_pass_disk_nonempty_real_listing() {
        let text = "qwen2.5-coder-7b-q4_k_m.gguf — 4.2GB\nqwen3-30b.apr — 18GB";
        assert_eq!(
            verdict_from_text_not_empty_sentinel(text, 2),
            ListdVerdict::Pass
        );
    }

    #[test]
    fn list003_pass_disk_empty_empty_text_ok() {
        assert_eq!(
            verdict_from_text_not_empty_sentinel("", 0),
            ListdVerdict::Pass
        );
    }

    #[test]
    fn list003_fail_disk_nonempty_with_sentinel() {
        // The exact regression: disk has files but text says no cached.
        let text = "No cached models found in ~/.cache/pacha/models";
        assert_eq!(
            verdict_from_text_not_empty_sentinel(text, 5),
            ListdVerdict::Fail
        );
    }

    #[test]
    fn list003_pass_text_partial_overlap_no_sentinel() {
        // Text mentions "No" or "cached" but not full sentinel.
        let text = "Cached entries: 3";
        assert_eq!(
            verdict_from_text_not_empty_sentinel(text, 3),
            ListdVerdict::Pass
        );
    }

    // -------------------------------------------------------------------------
    // Section 6: Realistic — full reconciliation scenarios.
    // -------------------------------------------------------------------------
    #[test]
    fn realistic_disk_orphan_caught() {
        // LIST-DISK-001 if_fails: "Disk files orphaned from list output".
        assert_eq!(
            verdict_from_at_least_one_when_disk_nonempty(3, 0),
            ListdVerdict::Fail
        );
    }

    #[test]
    fn realistic_undercount_caught() {
        // LIST-DISK-002 if_fails: "list undercounts relative to disk —
        // reconciliation incomplete".
        assert_eq!(
            verdict_from_total_covers_disk(1, 3),
            ListdVerdict::Fail
        );
    }

    #[test]
    fn realistic_phantom_empty_message_caught() {
        // LIST-DISK-003 if_fails: "list still shows 'No cached models
        // found' despite disk files".
        assert_eq!(
            verdict_from_text_not_empty_sentinel(
                "No cached models found",
                10
            ),
            ListdVerdict::Fail
        );
    }

    #[test]
    fn realistic_full_pipeline_passes_all_3_gates() {
        // Synthesize a realistic post-fix apr list invocation.
        let disk_paths = vec![
            "qwen2.5-coder-7b-instruct-q4_k_m.gguf",
            "qwen3-30b-a3b.apr",
            "tiny-llama-1.1b.safetensors",
            "README.md",  // not a model
            "config.json",
        ];
        let disk_count = count_model_files(&disk_paths);
        assert_eq!(disk_count, 3);

        let list_total = 5_usize; // catalog entries (3 disk + 2 metadata)
        let text = "qwen2.5-coder-7b-instruct-q4_k_m.gguf — 4.2GB\nqwen3-30b-a3b.apr — 18GB\n...";

        // Gate 1:
        assert_eq!(
            verdict_from_at_least_one_when_disk_nonempty(disk_count, list_total),
            ListdVerdict::Pass
        );
        // Gate 2:
        assert_eq!(
            verdict_from_total_covers_disk(list_total, disk_count),
            ListdVerdict::Pass
        );
        // Gate 3:
        assert_eq!(
            verdict_from_text_not_empty_sentinel(text, disk_count),
            ListdVerdict::Pass
        );
    }

    #[test]
    fn realistic_pre_fix_disk_scan_missing() {
        // Pre-fix: disk-scan fallback missing → list shows 0 despite
        // disk having 3 files.
        let disk_count = 3_usize;
        let list_total = 0_usize;
        let text = "No cached models found";

        // All 3 gates Fail.
        assert_eq!(
            verdict_from_at_least_one_when_disk_nonempty(disk_count, list_total),
            ListdVerdict::Fail
        );
        assert_eq!(
            verdict_from_total_covers_disk(list_total, disk_count),
            ListdVerdict::Fail
        );
        assert_eq!(
            verdict_from_text_not_empty_sentinel(text, disk_count),
            ListdVerdict::Fail
        );
    }
}
