//! Refcount-safe blob GC planner for `apr rm` / `apr gc` (CRUX-A-25).
//!
//! Contract: `contracts/crux-A-25-v1.yaml`.
//!
//! Three pure algorithm-level necessary conditions:
//!
//! 1. `compute_refcounts(manifests)` derives `sha256 → usize` from the
//!    current set of live manifests. A blob's refcount is the number of
//!    live manifests that reference it. This is the data source for
//!    every downstream GC decision.
//!
//! 2. `plan_gc(all_blobs, refcounts)` returns the set of blobs with
//!    refcount==0 (candidates for unlink). `apply_rm(manifests, tag)`
//!    removes a single manifest and returns the new manifest set so
//!    `plan_gc` can be re-run on the reduced state. Composed: rm then
//!    gc frees exactly the blobs the removed manifest uniquely owned.
//!
//! 3. `plan_gc` is idempotent — a second call on the same post-unlink
//!    state returns the empty candidate list. `--dry-run` is modelled
//!    as "compute the plan, but do not apply it" and the plan string
//!    is the same identifiable object whether the caller unlinks or not,
//!    so the plan/apply separation makes the dry-run invariant
//!    structurally impossible to violate at this layer.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// A single live manifest in the registry.
///
/// A manifest's identity is its `tag`; its payload is the set of blob
/// digests it references. Duplicate digests within one manifest count
/// as a single reference (this matches Ollama's behavior where a
/// manifest links each blob once regardless of how many tensors live
/// inside).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Manifest {
    pub tag: String,
    pub blobs: BTreeSet<String>,
}

impl Manifest {
    pub fn new(tag: impl Into<String>, blobs: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self {
            tag: tag.into(),
            blobs: blobs.into_iter().map(Into::into).collect(),
        }
    }
}

/// Compute the refcount table for the current set of live manifests.
///
/// Output is a `BTreeMap` so iteration order is stable — required for
/// deterministic plan output and for the dry-run/real-run equality
/// invariant (FALSIFY-CRUX-A-25-003).
pub fn compute_refcounts(manifests: &[Manifest]) -> BTreeMap<String, usize> {
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for m in manifests {
        for blob in &m.blobs {
            *counts.entry(blob.clone()).or_insert(0) += 1;
        }
    }
    counts
}

/// A blob marked for unlink by the GC planner.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GcCandidate {
    pub sha256: String,
    pub refcount: usize,
}

/// A deterministic GC plan, ready for either `--dry-run` print or
/// `apply_plan` execution.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GcPlan {
    pub candidates: Vec<GcCandidate>,
}

impl GcPlan {
    /// True iff the plan would free nothing — a second gc after a
    /// successful one, or a gc with no orphans.
    pub fn is_noop(&self) -> bool {
        self.candidates.is_empty()
    }
}

/// Compute the GC plan: every blob in `all_blobs` whose refcount is
/// zero (or absent from the refcount table) is a candidate for unlink.
///
/// `all_blobs` is the filesystem's current set of blob digests; the
/// classifier takes it as input so it remains pure (no I/O).
pub fn plan_gc(all_blobs: &BTreeSet<String>, refcounts: &BTreeMap<String, usize>) -> GcPlan {
    let candidates: Vec<GcCandidate> = all_blobs
        .iter()
        .filter_map(|sha| {
            let rc = refcounts.get(sha).copied().unwrap_or(0);
            if rc == 0 {
                Some(GcCandidate {
                    sha256: sha.clone(),
                    refcount: 0,
                })
            } else {
                None
            }
        })
        .collect();
    GcPlan { candidates }
}

/// Remove a manifest by tag, returning the reduced manifest set.
///
/// No-op if the tag is not present — callers decide whether that is
/// an error or a benign idempotent re-run.
pub fn apply_rm(manifests: &[Manifest], tag: &str) -> Vec<Manifest> {
    manifests
        .iter()
        .filter(|m| m.tag != tag)
        .cloned()
        .collect()
}

/// Apply a GC plan by returning the reduced blob set. Pure — callers
/// are responsible for the filesystem unlink separately, but the
/// post-state set produced here is the exact set the filesystem
/// should contain after gc completes.
pub fn apply_plan(all_blobs: &BTreeSet<String>, plan: &GcPlan) -> BTreeSet<String> {
    let to_drop: BTreeSet<&str> = plan.candidates.iter().map(|c| c.sha256.as_str()).collect();
    all_blobs
        .iter()
        .filter(|sha| !to_drop.contains(sha.as_str()))
        .cloned()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn blobs(xs: &[&str]) -> BTreeSet<String> {
        xs.iter().map(|s| s.to_string()).collect()
    }

    // ===== compute_refcounts =====

    #[test]
    fn refcount_counts_unique_referrers() {
        let ms = vec![
            Manifest::new("gpt2:latest", ["a", "b"]),
            Manifest::new("gpt2:dup", ["a", "b"]),
            Manifest::new("other", ["c"]),
        ];
        let rc = compute_refcounts(&ms);
        assert_eq!(rc.get("a"), Some(&2));
        assert_eq!(rc.get("b"), Some(&2));
        assert_eq!(rc.get("c"), Some(&1));
    }

    #[test]
    fn refcount_empty_manifest_set_has_no_refs() {
        let rc = compute_refcounts(&[]);
        assert!(rc.is_empty());
    }

    #[test]
    fn refcount_duplicate_blob_in_manifest_counts_once() {
        // BTreeSet dedups within a manifest, so the refcount should be 1.
        let m = Manifest::new("x", ["a", "a", "a"]);
        let rc = compute_refcounts(&[m]);
        assert_eq!(rc.get("a"), Some(&1));
    }

    #[test]
    fn refcount_is_deterministic() {
        let ms = vec![Manifest::new("a", ["x", "y"])];
        assert_eq!(compute_refcounts(&ms), compute_refcounts(&ms));
    }

    // ===== plan_gc =====

    #[test]
    fn plan_gc_flags_orphaned_blob() {
        // FALSIFY-CRUX-A-25-001 sub-claim: after rm of the last
        // referrer, the orphan shows up in the plan.
        let all = blobs(&["a", "b", "c"]);
        let rc = compute_refcounts(&[Manifest::new("live", ["a", "b"])]);
        let plan = plan_gc(&all, &rc);
        let shas: Vec<_> = plan.candidates.iter().map(|c| c.sha256.as_str()).collect();
        assert_eq!(shas, vec!["c"]);
    }

    #[test]
    fn plan_gc_never_flags_referenced_blob() {
        // FALSIFY-CRUX-A-25-002: the core safety property.
        let all = blobs(&["a", "b"]);
        let rc = compute_refcounts(&[Manifest::new("live", ["a", "b"])]);
        let plan = plan_gc(&all, &rc);
        assert!(plan.is_noop());
    }

    #[test]
    fn plan_gc_empty_when_registry_is_empty() {
        let all = blobs(&[]);
        let rc = BTreeMap::new();
        assert!(plan_gc(&all, &rc).is_noop());
    }

    #[test]
    fn plan_gc_is_deterministic() {
        let all = blobs(&["c", "a", "b"]);
        let rc = compute_refcounts(&[]);
        let a = plan_gc(&all, &rc);
        let b = plan_gc(&all, &rc);
        assert_eq!(a, b);
    }

    #[test]
    fn plan_gc_candidates_are_sorted() {
        // BTreeSet iteration is sorted — plan output must be stable so
        // dry-run and real-run emit identical strings.
        let all = blobs(&["z", "a", "m"]);
        let rc = compute_refcounts(&[]);
        let plan = plan_gc(&all, &rc);
        let shas: Vec<_> = plan.candidates.iter().map(|c| c.sha256.as_str()).collect();
        assert_eq!(shas, vec!["a", "m", "z"]);
    }

    // ===== apply_rm + end-to-end compose =====

    #[test]
    fn rm_then_gc_frees_unique_owned_blobs() {
        // Two manifests share some blobs; rm should free only blobs
        // uniquely owned by the removed manifest.
        let ms = vec![
            Manifest::new("a", ["x", "y", "z"]),
            Manifest::new("b", ["y", "z"]),
        ];
        let all = blobs(&["x", "y", "z"]);
        let ms = apply_rm(&ms, "a");
        let rc = compute_refcounts(&ms);
        let plan = plan_gc(&all, &rc);
        let shas: Vec<_> = plan.candidates.iter().map(|c| c.sha256.as_str()).collect();
        assert_eq!(shas, vec!["x"]);
    }

    #[test]
    fn rm_of_one_of_two_duplicate_tags_frees_nothing() {
        // FALSIFY-CRUX-A-25-002 integration: two tags alias the same
        // blob set. rm of one leaves the other, gc frees nothing.
        let ms = vec![
            Manifest::new("gpt2:latest", ["a", "b"]),
            Manifest::new("gpt2:dup", ["a", "b"]),
        ];
        let all = blobs(&["a", "b"]);
        let ms = apply_rm(&ms, "gpt2:latest");
        let plan = plan_gc(&all, &compute_refcounts(&ms));
        assert!(plan.is_noop(), "plan must be noop, got {plan:?}");
    }

    #[test]
    fn rm_missing_tag_is_idempotent() {
        let ms = vec![Manifest::new("a", ["x"])];
        let reduced = apply_rm(&ms, "does-not-exist");
        assert_eq!(reduced, ms);
    }

    // ===== idempotence and dry-run invariant =====

    #[test]
    fn gc_is_idempotent_second_run_is_noop() {
        // FALSIFY contract invariant: gc is idempotent — second run
        // frees 0 bytes.
        let all = blobs(&["a", "b", "c"]);
        let rc = compute_refcounts(&[Manifest::new("live", ["a"])]);
        let plan1 = plan_gc(&all, &rc);
        assert!(!plan1.is_noop());
        let after = apply_plan(&all, &plan1);
        // Second plan run on the reduced state.
        let plan2 = plan_gc(&after, &rc);
        assert!(plan2.is_noop());
    }

    #[test]
    fn dry_run_plan_equals_real_run_plan() {
        // FALSIFY-CRUX-A-25-003: the dry-run plan must equal the real
        // plan that would be executed. Since they are produced by the
        // same pure function, equality is by construction.
        let all = blobs(&["a", "b", "c"]);
        let rc = compute_refcounts(&[Manifest::new("live", ["a"])]);
        let dry = plan_gc(&all, &rc);
        let real = plan_gc(&all, &rc);
        assert_eq!(dry, real);
    }

    #[test]
    fn apply_plan_removes_exactly_the_candidates() {
        let all = blobs(&["a", "b", "c", "d"]);
        let plan = GcPlan {
            candidates: vec![
                GcCandidate {
                    sha256: "b".into(),
                    refcount: 0,
                },
                GcCandidate {
                    sha256: "d".into(),
                    refcount: 0,
                },
            ],
        };
        let after = apply_plan(&all, &plan);
        assert_eq!(after, blobs(&["a", "c"]));
    }

    #[test]
    fn apply_plan_preserves_blobs_not_in_plan() {
        // Safety regression: applying a plan must never remove a blob
        // that is absent from the plan, even if the plan mentions a
        // sha not currently on disk.
        let all = blobs(&["a", "b"]);
        let plan = GcPlan {
            candidates: vec![GcCandidate {
                sha256: "z-not-on-disk".into(),
                refcount: 0,
            }],
        };
        let after = apply_plan(&all, &plan);
        assert_eq!(after, all);
    }

    #[test]
    fn plan_serializes_to_stable_json() {
        // Dry-run output is human-parseable — the plan must round-trip
        // through serde_json deterministically.
        let plan = GcPlan {
            candidates: vec![GcCandidate {
                sha256: "abc".into(),
                refcount: 0,
            }],
        };
        let s = serde_json::to_string(&plan).unwrap();
        let parsed: GcPlan = serde_json::from_str(&s).unwrap();
        assert_eq!(plan, parsed);
    }
}
