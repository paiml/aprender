//! Registry byte-quota classifier for `apr pull` (CRUX-A-22).
//!
//! Contract: `contracts/crux-A-22-v1.yaml`.
//!
//! Two pure algorithm-level necessary conditions:
//!
//! 1. `classify_pull_against_quota(quota, used, incoming)` is a pure
//!    function of three scalars. It returns Allow iff
//!    `used + incoming ≤ quota`, else Reject with the used/free/needed
//!    triple. Because the decision is pre-download and depends only on
//!    these three inputs, the "no partial blobs on disk after reject"
//!    invariant (FALSIFY-CRUX-A-22-002) cannot be violated at this
//!    layer: rejection happens before any network I/O starts.
//!
//! 2. `render_quota_error_json(used, free, needed)` emits the machine-
//!    parseable error body with the three contract-defined fields.
//!    FALSIFY-CRUX-A-22-001 asserts a downstream JSON body with
//!    used / free / needed keys; this classifier proves the renderer
//!    always includes them and the values are self-consistent.

use serde::Serialize;

/// Decision for a single pull request against the current registry state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QuotaOutcome {
    /// Pull fits in the remaining budget.
    Allow { free_after: u64 },
    /// Pull would exceed the quota — rejected pre-download.
    Reject {
        used: u64,
        free: u64,
        needed: u64,
    },
}

/// Decide whether a pull of `incoming` bytes is allowed given the
/// current registry state.
///
/// Saturating arithmetic on `used + incoming` prevents integer overflow
/// from masking a legitimate reject (e.g. u64::MAX used + 1 incoming).
pub fn classify_pull_against_quota(quota: u64, used: u64, incoming: u64) -> QuotaOutcome {
    let free = quota.saturating_sub(used);
    let required = used.saturating_add(incoming);
    if required <= quota {
        QuotaOutcome::Allow {
            free_after: quota.saturating_sub(required),
        }
    } else {
        QuotaOutcome::Reject {
            used,
            free,
            needed: incoming,
        }
    }
}

/// Machine-parseable quota-exceeded error body.
///
/// FALSIFY-CRUX-A-22-001 requires the error to carry `used`, `free`,
/// and `needed` fields. Keeping the struct derived and serialized via
/// serde guarantees the keys cannot drift out of sync with callers.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct QuotaErrorBody {
    pub error: &'static str,
    pub used: u64,
    pub free: u64,
    pub needed: u64,
}

impl QuotaErrorBody {
    pub fn new(used: u64, free: u64, needed: u64) -> Self {
        Self {
            error: "registry_quota_exceeded",
            used,
            free,
            needed,
        }
    }
}

/// Render the quota-exceeded error body to a single JSON line.
///
/// Single line is required by the FALSIFY test which does
/// `open('/tmp/err').read().split('\n')[-2]` — the body must occupy
/// exactly one line so that shell pipelines can locate it.
pub fn render_quota_error_json(used: u64, free: u64, needed: u64) -> String {
    let body = QuotaErrorBody::new(used, free, needed);
    serde_json::to_string(&body).expect("QuotaErrorBody serializes as JSON")
}

#[cfg(test)]
mod tests {
    use super::*;

    // ===== classify_pull_against_quota =====

    #[test]
    fn allow_when_incoming_fits() {
        let r = classify_pull_against_quota(1000, 200, 300);
        assert_eq!(r, QuotaOutcome::Allow { free_after: 500 });
    }

    #[test]
    fn allow_exactly_at_quota_boundary() {
        // Used 600 + incoming 400 = 1000 == quota → exact fit is Allow.
        let r = classify_pull_against_quota(1000, 600, 400);
        assert_eq!(r, QuotaOutcome::Allow { free_after: 0 });
    }

    #[test]
    fn reject_one_byte_over_quota() {
        let r = classify_pull_against_quota(1000, 600, 401);
        assert_eq!(
            r,
            QuotaOutcome::Reject {
                used: 600,
                free: 400,
                needed: 401,
            }
        );
    }

    #[test]
    fn reject_when_already_over_quota() {
        // Used > quota can happen if quota was lowered after earlier pulls.
        // Any incoming > 0 must still reject.
        let r = classify_pull_against_quota(500, 800, 1);
        match r {
            QuotaOutcome::Reject { used, free, needed } => {
                assert_eq!(used, 800);
                assert_eq!(free, 0); // saturating_sub
                assert_eq!(needed, 1);
            }
            other => panic!("expected Reject, got {other:?}"),
        }
    }

    #[test]
    fn reject_when_incoming_is_zero_but_used_over_quota() {
        // Edge: used already exceeds quota AND incoming=0 → we choose to
        // Allow because no new bytes are requested. This matches the
        // "allow(pull) = free ≥ size(incoming)" formula where free may
        // be 0 and size(incoming) is 0.
        let r = classify_pull_against_quota(500, 800, 0);
        // 800 + 0 = 800, 800 > 500 → Reject (we DO reject — honest to formula)
        assert!(matches!(r, QuotaOutcome::Reject { .. }));
    }

    #[test]
    fn saturating_arithmetic_prevents_overflow() {
        // u64::MAX used + any positive incoming must reject, not wrap.
        let r = classify_pull_against_quota(1000, u64::MAX, 1);
        assert!(matches!(r, QuotaOutcome::Reject { .. }));
    }

    #[test]
    fn classifier_is_deterministic() {
        let a = classify_pull_against_quota(1000, 200, 300);
        let b = classify_pull_against_quota(1000, 200, 300);
        assert_eq!(a, b);
    }

    #[test]
    fn reject_preserves_all_three_fields() {
        // FALSIFY-CRUX-A-22-001 requires used/free/needed on every reject.
        let r = classify_pull_against_quota(1000, 300, 900);
        match r {
            QuotaOutcome::Reject { used, free, needed } => {
                assert_eq!(used, 300);
                assert_eq!(free, 700);
                assert_eq!(needed, 900);
                assert!(used + needed > used + free, "needed must exceed free");
            }
            other => panic!("expected Reject, got {other:?}"),
        }
    }

    // ===== render_quota_error_json =====

    #[test]
    fn error_json_contains_required_keys() {
        let s = render_quota_error_json(100, 50, 200);
        let v: serde_json::Value = serde_json::from_str(&s).unwrap();
        for k in ["error", "used", "free", "needed"] {
            assert!(v.get(k).is_some(), "missing key {k}: {s}");
        }
    }

    #[test]
    fn error_json_values_roundtrip() {
        let s = render_quota_error_json(123, 456, 789);
        let v: serde_json::Value = serde_json::from_str(&s).unwrap();
        assert_eq!(v["used"].as_u64().unwrap(), 123);
        assert_eq!(v["free"].as_u64().unwrap(), 456);
        assert_eq!(v["needed"].as_u64().unwrap(), 789);
    }

    #[test]
    fn error_json_discriminator_is_stable() {
        // Callers may programmatically match on the `error` field.
        let s = render_quota_error_json(0, 0, 0);
        let v: serde_json::Value = serde_json::from_str(&s).unwrap();
        assert_eq!(v["error"].as_str().unwrap(), "registry_quota_exceeded");
    }

    #[test]
    fn error_json_is_single_line() {
        // FALSIFY test picks up the last line of stderr — body must not
        // contain embedded newlines.
        let s = render_quota_error_json(100, 50, 200);
        assert!(!s.contains('\n'), "quota error body must be single-line: {s:?}");
    }

    #[test]
    fn error_json_is_deterministic() {
        let a = render_quota_error_json(100, 50, 200);
        let b = render_quota_error_json(100, 50, 200);
        assert_eq!(a, b);
    }

    #[test]
    fn error_json_parses_with_same_arithmetic_invariant() {
        // Regression guard: the FALSIFY test asserts
        //   e['used'] + e['needed'] > e['used'] + e['free']
        // which simplifies to needed > free. Our classifier only emits
        // Reject when used + incoming > quota, i.e. incoming > free, so
        // the invariant must hold on every Reject we serialize.
        let r = classify_pull_against_quota(1000, 600, 500);
        match r {
            QuotaOutcome::Reject { used, free, needed } => {
                let s = render_quota_error_json(used, free, needed);
                let v: serde_json::Value = serde_json::from_str(&s).unwrap();
                let needed = v["needed"].as_u64().unwrap();
                let free = v["free"].as_u64().unwrap();
                assert!(needed > free, "needed must exceed free on reject");
            }
            other => panic!("expected Reject, got {other:?}"),
        }
    }
}
