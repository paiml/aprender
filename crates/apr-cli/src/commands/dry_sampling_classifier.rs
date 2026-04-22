//! CRUX-C-23 — DRY (Don't Repeat Yourself) sampling classifiers
//!
//! Discharges `contracts/crux-C-23-v1.yaml` FALSIFY gates at PARTIAL_ALGORITHM_LEVEL:
//! - FALSIFY-CRUX-C-23-001: multiplier=0 is identity (penalty disabled)
//! - FALSIFY-CRUX-C-23-002: DRY reduces exact-phrase repetition
//!
//! Algorithm (llama.cpp DRY sampler):
//!   for candidate token t:
//!     match_len = longest suffix-match of (ctx + [t]) ending earlier in ctx
//!     if match_len >= allowed_length:
//!       penalty = multiplier * base^(match_len - allowed_length)
//!       logit[t] -= penalty

#![allow(dead_code)]

use std::collections::HashSet;

pub(crate) const DEFAULT_DRY_MULTIPLIER: f64 = 0.8;
pub(crate) const DEFAULT_DRY_BASE: f64 = 1.75;
pub(crate) const DEFAULT_DRY_ALLOWED_LENGTH: u32 = 2;

/// Parameter-range gate: multiplier ≥ 0, base ≥ 1, allowed_length ≥ 1.
#[derive(Debug, PartialEq)]
pub(crate) enum DryParamOutcome {
    Valid,
    NotFinite { field: &'static str },
    MultiplierNegative { multiplier: f64 },
    BaseBelowOne { base: f64 },
    AllowedLengthZero,
}

pub(crate) fn classify_dry_params(
    multiplier: f64,
    base: f64,
    allowed_length: u32,
) -> DryParamOutcome {
    if !multiplier.is_finite() {
        return DryParamOutcome::NotFinite {
            field: "multiplier",
        };
    }
    if !base.is_finite() {
        return DryParamOutcome::NotFinite { field: "base" };
    }
    if multiplier < 0.0 {
        return DryParamOutcome::MultiplierNegative { multiplier };
    }
    if base < 1.0 {
        return DryParamOutcome::BaseBelowOne { base };
    }
    if allowed_length == 0 {
        return DryParamOutcome::AllowedLengthZero;
    }
    DryParamOutcome::Valid
}

/// Identity gate (FALSIFY-CRUX-C-23-001): multiplier=0 → every token unchanged.
#[derive(Debug, PartialEq)]
pub(crate) enum IdentityOutcome {
    Ok,
    InvalidInput {
        reason: &'static str,
    },
    LogitsChanged {
        first_diff_index: usize,
        before: f64,
        after: f64,
    },
}

pub(crate) fn classify_dry_identity_zero_multiplier(
    logits_before: &[f64],
    logits_after: &[f64],
    multiplier: f64,
) -> IdentityOutcome {
    if logits_before.is_empty() {
        return IdentityOutcome::InvalidInput {
            reason: "logits_before is empty",
        };
    }
    if logits_before.len() != logits_after.len() {
        return IdentityOutcome::InvalidInput {
            reason: "logits length mismatch",
        };
    }
    if !multiplier.is_finite() || multiplier != 0.0 {
        return IdentityOutcome::InvalidInput {
            reason: "multiplier != 0.0",
        };
    }
    for (i, (&b, &a)) in logits_before.iter().zip(logits_after.iter()).enumerate() {
        if !b.is_finite() || !a.is_finite() {
            return IdentityOutcome::InvalidInput {
                reason: "non-finite logit",
            };
        }
        if (b - a).abs() > f64::EPSILON * b.abs().max(1.0) {
            return IdentityOutcome::LogitsChanged {
                first_diff_index: i,
                before: b,
                after: a,
            };
        }
    }
    IdentityOutcome::Ok
}

/// Computes the longest suffix of (ctx + [candidate]) that matches a substring
/// ending earlier in ctx. This is the classifier-level specification of
/// llama.cpp's DRY match_len.
///
/// Returns 0 if no match ≥ 1 token is found.
pub(crate) fn classify_dry_match_len(
    ctx: &[u32],
    candidate: u32,
    seq_breakers: &HashSet<u32>,
) -> u32 {
    // Build extended context with candidate appended.
    let mut ext: Vec<u32> = ctx.to_vec();
    ext.push(candidate);

    // Walk back looking for a suffix-of-ext match ending earlier in ctx.
    // max_len can be at most ctx.len() (match must end BEFORE position ctx.len()).
    let ctx_len = ctx.len();
    if ctx_len == 0 {
        return 0;
    }
    let mut best: u32 = 0;

    // For each possible earlier-ending position j < ctx_len:
    //   match_len = longest L s.t. ext[ext.len()-L ..] == ctx[j+1-L .. j+1]
    // Scan j from ctx_len-1 down; stop if we see a seq_breaker at ctx[j].
    for j in (0..ctx_len).rev() {
        if seq_breakers.contains(&ctx[j]) {
            // Seq breaker resets counter for THIS branch; skip to earlier j.
            continue;
        }
        // Compare ctx[..=j] tail with ext tail.
        let mut l: usize = 0;
        loop {
            let ext_idx = ext.len() - 1 - l;
            let ctx_idx_opt = j.checked_sub(l);
            match ctx_idx_opt {
                None => break,
                Some(ctx_idx) => {
                    if seq_breakers.contains(&ctx[ctx_idx]) {
                        break;
                    }
                    if ext[ext_idx] != ctx[ctx_idx] {
                        break;
                    }
                    l += 1;
                    if ext_idx == 0 {
                        break;
                    }
                }
            }
        }
        let l_u32 = u32::try_from(l).unwrap_or(u32::MAX);
        if l_u32 > best {
            best = l_u32;
        }
    }
    best
}

/// Penalty-formula gate: penalty = multiplier * base^(match_len - allowed_length)
/// when match_len >= allowed_length; 0 otherwise; penalty ≥ 0 always.
#[derive(Debug, PartialEq)]
pub(crate) enum PenaltyOutcome {
    Ok { penalty: f64 },
    InvalidInput { reason: &'static str },
    Negative { penalty: f64 },
}

pub(crate) fn classify_dry_penalty(
    match_len: u32,
    allowed_length: u32,
    multiplier: f64,
    base: f64,
) -> PenaltyOutcome {
    if !multiplier.is_finite() || !base.is_finite() {
        return PenaltyOutcome::InvalidInput {
            reason: "non-finite multiplier or base",
        };
    }
    if multiplier < 0.0 {
        return PenaltyOutcome::InvalidInput {
            reason: "multiplier negative",
        };
    }
    if base < 1.0 {
        return PenaltyOutcome::InvalidInput {
            reason: "base < 1.0",
        };
    }
    if allowed_length == 0 {
        return PenaltyOutcome::InvalidInput {
            reason: "allowed_length == 0",
        };
    }
    if match_len < allowed_length {
        // Below threshold — penalty is zero by spec.
        return PenaltyOutcome::Ok { penalty: 0.0 };
    }
    let exponent = f64::from(match_len - allowed_length);
    let penalty = multiplier * base.powf(exponent);
    if !penalty.is_finite() {
        return PenaltyOutcome::InvalidInput {
            reason: "penalty overflow",
        };
    }
    if penalty < 0.0 {
        return PenaltyOutcome::Negative { penalty };
    }
    PenaltyOutcome::Ok { penalty }
}

/// Monotonicity gate: penalty grows monotonically (non-strictly) with match_len.
/// This captures "DRY reduces exact-phrase repetition" algebraically.
#[derive(Debug, PartialEq)]
pub(crate) enum MonotonicityOutcome {
    Ok,
    InvalidInput {
        reason: &'static str,
    },
    Violation {
        match_len_a: u32,
        match_len_b: u32,
        penalty_a: f64,
        penalty_b: f64,
    },
}

pub(crate) fn classify_dry_penalty_monotone_in_match_len(
    match_len_a: u32,
    match_len_b: u32,
    allowed_length: u32,
    multiplier: f64,
    base: f64,
) -> MonotonicityOutcome {
    if match_len_a > match_len_b {
        return MonotonicityOutcome::InvalidInput {
            reason: "match_len_a must be <= match_len_b",
        };
    }
    let pa = match classify_dry_penalty(match_len_a, allowed_length, multiplier, base) {
        PenaltyOutcome::Ok { penalty } => penalty,
        PenaltyOutcome::InvalidInput { reason } => {
            return MonotonicityOutcome::InvalidInput { reason }
        }
        PenaltyOutcome::Negative { .. } => {
            return MonotonicityOutcome::InvalidInput {
                reason: "penalty_a negative",
            }
        }
    };
    let pb = match classify_dry_penalty(match_len_b, allowed_length, multiplier, base) {
        PenaltyOutcome::Ok { penalty } => penalty,
        PenaltyOutcome::InvalidInput { reason } => {
            return MonotonicityOutcome::InvalidInput { reason }
        }
        PenaltyOutcome::Negative { .. } => {
            return MonotonicityOutcome::InvalidInput {
                reason: "penalty_b negative",
            }
        }
    };
    if pb + f64::EPSILON < pa {
        return MonotonicityOutcome::Violation {
            match_len_a,
            match_len_b,
            penalty_a: pa,
            penalty_b: pb,
        };
    }
    MonotonicityOutcome::Ok
}

#[cfg(test)]
#[path = "dry_sampling_classifier_tests.rs"]
mod tests;
