//! `apr stop` pure state-transition classifier.
//!
//! This module models the observable state transition of `apr stop <model>`
//! without touching real daemon state. It proves the sub-claims for
//! FALSIFY-CRUX-A-13-001 (stop removes target from resident set) and
//! FALSIFY-CRUX-A-13-002 (idempotent) at PARTIAL_ALGORITHM_LEVEL.
//!
//! Integration-only gates (VRAM reclaim, ollama golden parity) are NOT
//! discharged here; they require a live `apr serve` daemon + nvidia-smi.

use std::collections::BTreeSet;

/// A sorted set of resident model IDs in `apr serve`.
///
/// Stored as `BTreeSet<String>` so equality is order-independent and the
/// classifier is deterministic.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResidentSet(BTreeSet<String>);

impl ResidentSet {
    pub fn new<I: IntoIterator<Item = S>, S: Into<String>>(iter: I) -> Self {
        Self(iter.into_iter().map(Into::into).collect())
    }

    pub fn contains(&self, model: &str) -> bool {
        self.0.contains(model)
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn as_sorted_vec(&self) -> Vec<String> {
        self.0.iter().cloned().collect()
    }
}

/// Classifier of an `apr stop` attempt.
///
/// - `Evicted`: target was resident, is now gone.
/// - `NotLoaded`: target was not resident; exit code still 0, stderr
///   MUST say "not loaded" / "already stopped", NOT "error".
/// - `InvalidName`: empty or shell-unsafe model name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StopOutcome {
    Evicted,
    NotLoaded,
    InvalidName,
}

impl StopOutcome {
    /// Exit code that `apr stop` MUST emit for this outcome.
    ///
    /// `Evicted` and `NotLoaded` BOTH return 0 (idempotency contract).
    /// `InvalidName` returns 2 (user error).
    pub const fn exit_code(&self) -> i32 {
        match self {
            Self::Evicted | Self::NotLoaded => 0,
            Self::InvalidName => 2,
        }
    }

    /// The stderr category; used by tests but not printed verbatim.
    pub const fn stderr_class(&self) -> &'static str {
        match self {
            Self::Evicted => "stopped",
            Self::NotLoaded => "not loaded",
            Self::InvalidName => "invalid name",
        }
    }
}

/// Full result of `plan_stop`: outcome plus the new resident set.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StopPlan {
    pub outcome: StopOutcome,
    pub new_set: ResidentSet,
}

/// Reject empty / shell-unsafe names. Mirrors `tag_is_valid` in `copy_tag`.
fn model_name_is_valid(name: &str) -> bool {
    !name.is_empty()
        && !name.contains('\0')
        && !name.contains('/')
        && !name.contains('\\')
        && !name.starts_with('-')
}

/// Pure classifier: apply `apr stop <target>` to a resident-set snapshot.
///
/// Post-conditions (checked by unit tests):
/// - If `target` was in `set` and name is valid, outcome is `Evicted` and
///   the new set is `set \ {target}`.
/// - If `target` was NOT in `set` and name is valid, outcome is `NotLoaded`
///   and the new set equals the old set.
/// - For invalid names, outcome is `InvalidName` and the set is unchanged.
/// - The operation is idempotent: `plan_stop(plan_stop(set, m).new_set, m)`
///   always produces `NotLoaded` and the same new_set.
pub fn plan_stop(set: &ResidentSet, target: &str) -> StopPlan {
    if !model_name_is_valid(target) {
        return StopPlan {
            outcome: StopOutcome::InvalidName,
            new_set: set.clone(),
        };
    }
    if set.contains(target) {
        let mut next = set.0.clone();
        next.remove(target);
        StopPlan {
            outcome: StopOutcome::Evicted,
            new_set: ResidentSet(next),
        }
    } else {
        StopPlan {
            outcome: StopOutcome::NotLoaded,
            new_set: set.clone(),
        }
    }
}

/// True iff `target` is absent from the plan's resulting resident set.
/// Used directly to discharge FALSIFY-001's sub-claim at algorithm level.
pub fn plan_removes_target(plan: &StopPlan, target: &str) -> bool {
    !plan.new_set.contains(target)
}

/// True iff applying `plan_stop` twice produces the same new set the first
/// call produced. Directly discharges FALSIFY-002's idempotency sub-claim.
pub fn plan_is_idempotent(set: &ResidentSet, target: &str) -> bool {
    let first = plan_stop(set, target);
    let second = plan_stop(&first.new_set, target);
    second.new_set == first.new_set
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r(items: &[&str]) -> ResidentSet {
        ResidentSet::new(items.iter().map(|s| s.to_string()))
    }

    // ---- FALSIFY-001 sub-claim: stop removes target from resident set ----

    #[test]
    fn falsify_001_sub_claim_evicts_resident_target() {
        let set = r(&["qwen-7b", "llama-1b"]);
        let plan = plan_stop(&set, "qwen-7b");
        assert_eq!(plan.outcome, StopOutcome::Evicted);
        assert!(plan_removes_target(&plan, "qwen-7b"));
        assert!(plan.new_set.contains("llama-1b"));
        assert_eq!(plan.new_set.len(), 1);
    }

    #[test]
    fn stop_not_resident_is_noop() {
        let set = r(&["qwen-7b"]);
        let plan = plan_stop(&set, "never-loaded");
        assert_eq!(plan.outcome, StopOutcome::NotLoaded);
        assert_eq!(plan.new_set, set);
    }

    #[test]
    fn stop_from_empty_set_is_noop() {
        let set = r(&[]);
        let plan = plan_stop(&set, "anything");
        assert_eq!(plan.outcome, StopOutcome::NotLoaded);
        assert!(plan.new_set.is_empty());
    }

    #[test]
    fn stop_preserves_other_models() {
        let set = r(&["a", "b", "c"]);
        let plan = plan_stop(&set, "b");
        assert!(plan.new_set.contains("a"));
        assert!(plan.new_set.contains("c"));
        assert!(!plan.new_set.contains("b"));
    }

    // ---- FALSIFY-002 sub-claim: idempotent ----

    #[test]
    fn falsify_002_sub_claim_idempotent_on_resident() {
        let set = r(&["qwen-7b"]);
        assert!(plan_is_idempotent(&set, "qwen-7b"));
        let first = plan_stop(&set, "qwen-7b");
        let second = plan_stop(&first.new_set, "qwen-7b");
        assert_eq!(first.new_set, second.new_set);
        assert_eq!(second.outcome, StopOutcome::NotLoaded);
        assert_eq!(second.outcome.exit_code(), 0);
    }

    #[test]
    fn idempotent_on_never_loaded() {
        let set = r(&["other"]);
        assert!(plan_is_idempotent(&set, "never-loaded"));
        let first = plan_stop(&set, "never-loaded");
        let second = plan_stop(&first.new_set, "never-loaded");
        assert_eq!(first, second);
    }

    #[test]
    fn idempotent_on_empty() {
        let set = r(&[]);
        assert!(plan_is_idempotent(&set, "anything"));
    }

    // ---- exit-code contract ----

    #[test]
    fn evicted_exit_code_is_zero() {
        assert_eq!(StopOutcome::Evicted.exit_code(), 0);
    }

    #[test]
    fn not_loaded_exit_code_is_zero() {
        assert_eq!(StopOutcome::NotLoaded.exit_code(), 0);
    }

    #[test]
    fn invalid_name_exit_code_is_two() {
        assert_eq!(StopOutcome::InvalidName.exit_code(), 2);
    }

    #[test]
    fn stderr_classes_are_distinct() {
        assert_ne!(
            StopOutcome::Evicted.stderr_class(),
            StopOutcome::NotLoaded.stderr_class()
        );
        assert_ne!(
            StopOutcome::NotLoaded.stderr_class(),
            StopOutcome::InvalidName.stderr_class()
        );
    }

    // ---- name validation ----

    #[test]
    fn empty_name_is_invalid() {
        let set = r(&[]);
        let plan = plan_stop(&set, "");
        assert_eq!(plan.outcome, StopOutcome::InvalidName);
    }

    #[test]
    fn null_byte_name_is_invalid() {
        let set = r(&[]);
        let plan = plan_stop(&set, "qwen\0evil");
        assert_eq!(plan.outcome, StopOutcome::InvalidName);
    }

    #[test]
    fn slash_in_name_is_invalid() {
        let set = r(&[]);
        let plan = plan_stop(&set, "a/b");
        assert_eq!(plan.outcome, StopOutcome::InvalidName);
    }

    #[test]
    fn leading_dash_is_invalid() {
        let set = r(&[]);
        let plan = plan_stop(&set, "-rf");
        assert_eq!(plan.outcome, StopOutcome::InvalidName);
    }

    #[test]
    fn invalid_name_does_not_mutate_set() {
        let before = r(&["qwen"]);
        let plan = plan_stop(&before, "");
        assert_eq!(plan.new_set, before);
    }

    // ---- structural invariants ----

    #[test]
    fn resident_set_is_deduplicated() {
        let set = r(&["a", "a", "b"]);
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn resident_set_order_independent_equality() {
        assert_eq!(r(&["a", "b", "c"]), r(&["c", "b", "a"]));
    }

    #[test]
    fn sorted_vec_is_lexicographically_sorted() {
        let v = r(&["c", "a", "b"]).as_sorted_vec();
        assert_eq!(v, vec!["a", "b", "c"]);
    }

    #[test]
    fn eviction_reduces_len_by_exactly_one() {
        let set = r(&["a", "b", "c"]);
        let plan = plan_stop(&set, "b");
        assert_eq!(plan.new_set.len(), set.len() - 1);
    }

    #[test]
    fn noop_preserves_len() {
        let set = r(&["a", "b"]);
        let plan = plan_stop(&set, "c");
        assert_eq!(plan.new_set.len(), set.len());
    }

    // ---- full cycle: stop then re-add then stop again ----

    #[test]
    fn stop_after_reload_evicts_again() {
        let set = r(&["m"]);
        let after_stop = plan_stop(&set, "m").new_set;
        assert!(!after_stop.contains("m"));
        let reloaded = r(&["m"]);
        let plan = plan_stop(&reloaded, "m");
        assert_eq!(plan.outcome, StopOutcome::Evicted);
    }
}
