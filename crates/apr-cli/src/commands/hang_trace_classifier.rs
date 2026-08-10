//! Deadlock/hang detector stack dump classifier (CRUX-F-14).
//!
//! Pure, deterministic classifiers that discharge FALSIFY-CRUX-F-14-{001,002,003}
//! at the PARTIAL_ALGORITHM_LEVEL — algorithm-level necessary conditions on
//! a captured `$APR_TRACE_DIR` after a hang/timeout (or a successful run):
//!
//!   * `classify_timeout_dump` — directory contains exactly `world_size`
//!     `rank{N}.py.txt` files (for N in 0..world_size), each non-empty;
//!     optional `.native.txt` and `.nccl.json` files are accepted.
//!   * `classify_empty_on_success` — directory is empty (no rank files)
//!     when the run completed normally.
//!   * `classify_exit_code` — exit code distinguishes timeout (124) vs
//!     other failures (any other non-zero) vs success (0).
//!
//! Full discharge of -001/-002/-003 requires a live `apr train` watchdog
//! actually emitting these files on hang — tracked as
//! BLOCKER-UPSTREAM-MISSING.

/// Canonical timeout exit code (POSIX `coreutils timeout` convention).
pub const F14_TIMEOUT_EXIT_CODE: i32 = 124;

/// Outcome of `classify_timeout_dump`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HangTimeoutOutcome {
    Ok {
        ranks_seen: usize,
    },
    /// `world_size == 0`: the gate has nothing to assert, so it cannot pass.
    WorldSizeZero,
    DirEmpty,
    MissingRank {
        rank: usize,
    },
    EmptyFile {
        rank: usize,
    },
    UnexpectedFilename {
        name: String,
    },
}

/// Outcome of `classify_empty_on_success`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HangEmptyOnSuccessOutcome {
    Ok,
    UnexpectedFile { name: String },
}

/// Outcome of `classify_exit_code`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HangExitOutcome {
    OkTimeout,
    OkOtherError { code: i32 },
    OkSuccess,
    ExitCodeMismatch { got: i32, expected: i32 },
}

/// Listing of a captured trace-dir as parallel arrays so the classifier
/// is testable without touching the filesystem.
#[derive(Debug, Clone)]
pub struct TraceDirListing<'a> {
    /// File names (basename only, e.g. `rank0.py.txt`).
    pub names: Vec<&'a str>,
    /// Byte sizes for each file (parallel to `names`).
    pub sizes: Vec<u64>,
}

impl<'a> TraceDirListing<'a> {
    pub fn from_pairs(pairs: &'a [(&'a str, u64)]) -> Self {
        Self {
            names: pairs.iter().map(|p| p.0).collect(),
            sizes: pairs.iter().map(|p| p.1).collect(),
        }
    }
}

/// FALSIFY-CRUX-F-14-001: verify timeout dumped exactly `world_size`
/// non-empty `rank{N}.py.txt` files (with N in `0..world_size`).
pub fn classify_timeout_dump(
    listing: &TraceDirListing<'_>,
    world_size: usize,
) -> HangTimeoutOutcome {
    if world_size == 0 {
        // A timeout dump with zero expected ranks asserts nothing. Treating
        // it as Ok made an EMPTY trace dir pass, so a CI job whose
        // `--world-size` came from an env var that resolved to empty got a
        // green gate on no evidence at all. Unknown is not a pass.
        return HangTimeoutOutcome::WorldSizeZero;
    }
    if listing.names.is_empty() {
        return HangTimeoutOutcome::DirEmpty;
    }
    // Validate every observed file is a recognized rank file.
    for name in &listing.names {
        if !is_recognized_rank_file(name) {
            return HangTimeoutOutcome::UnexpectedFilename {
                name: (*name).to_string(),
            };
        }
    }
    // Each rank must have at least a .py.txt file, non-empty.
    for r in 0..world_size {
        let needle = format!("rank{r}.py.txt");
        let Some(idx) = listing.names.iter().position(|n| *n == needle) else {
            return HangTimeoutOutcome::MissingRank { rank: r };
        };
        if listing.sizes.get(idx).copied().unwrap_or(0) == 0 {
            return HangTimeoutOutcome::EmptyFile { rank: r };
        }
    }
    HangTimeoutOutcome::Ok {
        ranks_seen: world_size,
    }
}

/// FALSIFY-CRUX-F-14-002: verify the trace-dir is empty after a successful
/// run (no false-trigger dumps).
pub fn classify_empty_on_success(listing: &TraceDirListing<'_>) -> HangEmptyOnSuccessOutcome {
    if let Some(name) = listing.names.first() {
        return HangEmptyOnSuccessOutcome::UnexpectedFile {
            name: (*name).to_string(),
        };
    }
    HangEmptyOnSuccessOutcome::Ok
}

/// FALSIFY-CRUX-F-14-003: verify exit code matches expectation
/// (124 = timeout, any other non-zero = other error, 0 = success).
pub fn classify_exit_code(got: i32, expected: i32) -> HangExitOutcome {
    if got == expected {
        return match got {
            F14_TIMEOUT_EXIT_CODE => HangExitOutcome::OkTimeout,
            0 => HangExitOutcome::OkSuccess,
            other => HangExitOutcome::OkOtherError { code: other },
        };
    }
    HangExitOutcome::ExitCodeMismatch { got, expected }
}

fn is_recognized_rank_file(name: &str) -> bool {
    // Accepted shapes: rank{N}.py.txt, rank{N}.native.txt, rank{N}.nccl.json
    let Some(rest) = name.strip_prefix("rank") else {
        return false;
    };
    let Some(dot) = rest.find('.') else {
        return false;
    };
    let (num, suffix) = rest.split_at(dot);
    if num.is_empty() || !num.bytes().all(|b| b.is_ascii_digit()) {
        return false;
    }
    matches!(suffix, ".py.txt" | ".native.txt" | ".nccl.json")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ok_listing_for(world_size: usize) -> Vec<(String, u64)> {
        (0..world_size)
            .map(|r| (format!("rank{r}.py.txt"), 256u64))
            .collect()
    }

    fn listing_view(pairs: &[(String, u64)]) -> TraceDirListing<'_> {
        let names: Vec<&str> = pairs.iter().map(|p| p.0.as_str()).collect();
        let sizes: Vec<u64> = pairs.iter().map(|p| p.1).collect();
        TraceDirListing { names, sizes }
    }

    #[test]
    fn timeout_dump_ok_when_all_ranks_present() {
        let pairs = ok_listing_for(2);
        let listing = listing_view(&pairs);
        assert_eq!(
            classify_timeout_dump(&listing, 2),
            HangTimeoutOutcome::Ok { ranks_seen: 2 }
        );
    }

    #[test]
    fn timeout_dump_rejects_empty_dir() {
        let listing = TraceDirListing {
            names: vec![],
            sizes: vec![],
        };
        assert_eq!(
            classify_timeout_dump(&listing, 2),
            HangTimeoutOutcome::DirEmpty
        );
    }

    #[test]
    fn timeout_dump_reports_missing_rank() {
        let pairs = vec![("rank0.py.txt".to_string(), 256u64)];
        let listing = listing_view(&pairs);
        assert_eq!(
            classify_timeout_dump(&listing, 2),
            HangTimeoutOutcome::MissingRank { rank: 1 }
        );
    }

    #[test]
    fn timeout_dump_reports_empty_file() {
        let pairs = vec![
            ("rank0.py.txt".to_string(), 256u64),
            ("rank1.py.txt".to_string(), 0u64),
        ];
        let listing = listing_view(&pairs);
        assert_eq!(
            classify_timeout_dump(&listing, 2),
            HangTimeoutOutcome::EmptyFile { rank: 1 }
        );
    }

    #[test]
    fn timeout_dump_reports_unexpected_filename() {
        let pairs = vec![("scratchpad.txt".to_string(), 10u64)];
        let listing = listing_view(&pairs);
        match classify_timeout_dump(&listing, 1) {
            HangTimeoutOutcome::UnexpectedFilename { name } => assert_eq!(name, "scratchpad.txt"),
            other => panic!("expected UnexpectedFilename, got {other:?}"),
        }
    }

    // Dogfood 0.63.0 #2377 finding 4: `--world-size 0` short-circuited to
    // `Ok { ranks_seen: 0 }`, so an EMPTY trace dir passed the timeout gate.
    // A CI job whose world_size came from an env var that resolved to empty
    // got a green gate on no evidence at all.
    #[test]
    fn timeout_dump_world_size_zero_does_not_pass_on_empty_dir() {
        let pairs: Vec<(String, u64)> = vec![];
        let listing = listing_view(&pairs);
        assert_eq!(
            classify_timeout_dump(&listing, 0),
            HangTimeoutOutcome::WorldSizeZero
        );
    }

    #[test]
    fn timeout_dump_world_size_zero_does_not_pass_on_populated_dir() {
        let pairs = vec![("rank0.py.txt".to_string(), 256u64)];
        let listing = listing_view(&pairs);
        assert_eq!(
            classify_timeout_dump(&listing, 0),
            HangTimeoutOutcome::WorldSizeZero
        );
    }

    #[test]
    fn timeout_dump_accepts_extra_native_and_nccl_files() {
        let pairs = vec![
            ("rank0.py.txt".to_string(), 256u64),
            ("rank0.native.txt".to_string(), 512u64),
            ("rank0.nccl.json".to_string(), 1024u64),
        ];
        let listing = listing_view(&pairs);
        assert_eq!(
            classify_timeout_dump(&listing, 1),
            HangTimeoutOutcome::Ok { ranks_seen: 1 }
        );
    }

    #[test]
    fn empty_on_success_ok_when_dir_is_empty() {
        let listing = TraceDirListing {
            names: vec![],
            sizes: vec![],
        };
        assert_eq!(
            classify_empty_on_success(&listing),
            HangEmptyOnSuccessOutcome::Ok
        );
    }

    #[test]
    fn empty_on_success_reports_unexpected_file() {
        let pairs = vec![("rank0.py.txt".to_string(), 256u64)];
        let listing = listing_view(&pairs);
        match classify_empty_on_success(&listing) {
            HangEmptyOnSuccessOutcome::UnexpectedFile { name } => {
                assert_eq!(name, "rank0.py.txt");
            }
            other => panic!("expected UnexpectedFile, got {other:?}"),
        }
    }

    #[test]
    fn exit_code_124_matches_timeout() {
        assert_eq!(
            classify_exit_code(F14_TIMEOUT_EXIT_CODE, F14_TIMEOUT_EXIT_CODE),
            HangExitOutcome::OkTimeout
        );
    }

    #[test]
    fn exit_code_1_is_other_error() {
        assert_eq!(
            classify_exit_code(1, 1),
            HangExitOutcome::OkOtherError { code: 1 }
        );
    }

    #[test]
    fn exit_code_0_is_success() {
        assert_eq!(classify_exit_code(0, 0), HangExitOutcome::OkSuccess);
    }

    #[test]
    fn exit_code_mismatch_reports_both_values() {
        assert_eq!(
            classify_exit_code(1, F14_TIMEOUT_EXIT_CODE),
            HangExitOutcome::ExitCodeMismatch {
                got: 1,
                expected: 124
            }
        );
    }
}
