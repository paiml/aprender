//! CRUX-C-15 — Tensor + pipeline parallelism classifiers.
//!
//! Pure, deterministic algorithm-level gates discharging FALSIFY-CRUX-C-15-001..004
//! at PARTIAL_ALGORITHM_LEVEL. Full discharge is blocked on `apr serve --tp/--pp`
//! surface wired to a real multi-GPU runtime.
//!
//! Design rule: no silent passes — every ill-formed input maps to a distinct
//! Outcome variant so defect classes cannot collapse into each other.

use serde_json::Value;

/// vLLM-style default: tensor-parallel must divide num_heads exactly.
pub const DIVISIBILITY_REQUIRED: bool = true;

/// Near-linear scaling floor per the contract (70% efficiency).
pub const MIN_TP_SCALING_ALPHA: f64 = 0.70;

// ─────────────────────────────────────────────────────────────────────────────
// Classifier 1: world_size arithmetic
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WorldSizeOutcome {
    Ok,
    ZeroTp,
    ZeroPp,
    Mismatch {
        tp: u32,
        pp: u32,
        expected: u32,
        got: u32,
    },
}

pub fn classify_world_size(tp: u32, pp: u32, world_size: u32) -> WorldSizeOutcome {
    if tp == 0 {
        return WorldSizeOutcome::ZeroTp;
    }
    if pp == 0 {
        return WorldSizeOutcome::ZeroPp;
    }
    let expected = tp.saturating_mul(pp);
    if world_size != expected {
        return WorldSizeOutcome::Mismatch {
            tp,
            pp,
            expected,
            got: world_size,
        };
    }
    WorldSizeOutcome::Ok
}

// ─────────────────────────────────────────────────────────────────────────────
// Classifier 2: divisibility preconditions (num_heads % tp, num_layers % pp)
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DivisibilityOutcome {
    Ok,
    ZeroTp,
    ZeroPp,
    ZeroNumHeads,
    ZeroNumLayers,
    HeadsNotDivisible { num_heads: u32, tp: u32 },
    LayersNotDivisible { num_layers: u32, pp: u32 },
}

pub fn classify_divisibility(
    num_heads: u32,
    num_layers: u32,
    tp: u32,
    pp: u32,
) -> DivisibilityOutcome {
    if tp == 0 {
        return DivisibilityOutcome::ZeroTp;
    }
    if pp == 0 {
        return DivisibilityOutcome::ZeroPp;
    }
    if num_heads == 0 {
        return DivisibilityOutcome::ZeroNumHeads;
    }
    if num_layers == 0 {
        return DivisibilityOutcome::ZeroNumLayers;
    }
    if num_heads % tp != 0 {
        return DivisibilityOutcome::HeadsNotDivisible { num_heads, tp };
    }
    if num_layers % pp != 0 {
        return DivisibilityOutcome::LayersNotDivisible { num_layers, pp };
    }
    DivisibilityOutcome::Ok
}

// ─────────────────────────────────────────────────────────────────────────────
// Classifier 3: greedy-sampling byte-identical token parity
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TpParityOutcome {
    Ok,
    EmptinessMismatch {
        ref_empty: bool,
        parallel_empty: bool,
    },
    LengthMismatch {
        ref_len: usize,
        parallel_len: usize,
    },
    TokenDivergence {
        at_index: usize,
        ref_token: u32,
        parallel_token: u32,
    },
}

pub fn classify_tp_parity(ref_tokens: &[u32], parallel_tokens: &[u32]) -> TpParityOutcome {
    let ref_empty = ref_tokens.is_empty();
    let parallel_empty = parallel_tokens.is_empty();
    if ref_empty != parallel_empty {
        return TpParityOutcome::EmptinessMismatch {
            ref_empty,
            parallel_empty,
        };
    }
    if ref_tokens.len() != parallel_tokens.len() {
        return TpParityOutcome::LengthMismatch {
            ref_len: ref_tokens.len(),
            parallel_len: parallel_tokens.len(),
        };
    }
    for (i, (r, p)) in ref_tokens.iter().zip(parallel_tokens.iter()).enumerate() {
        if r != p {
            return TpParityOutcome::TokenDivergence {
                at_index: i,
                ref_token: *r,
                parallel_token: *p,
            };
        }
    }
    TpParityOutcome::Ok
}

// ─────────────────────────────────────────────────────────────────────────────
// Classifier 4: TP scaling efficiency floor (α-adjusted)
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum ScalingEfficiencyOutcome {
    Ok {
        observed_alpha: f64,
    },
    InvalidInput {
        reason: &'static str,
    },
    Regression {
        base_tps: f64,
        parallel_tps: f64,
        observed_alpha: f64,
    },
    BelowThreshold {
        observed_alpha: f64,
        required_alpha: f64,
        tp: u32,
    },
}

pub fn classify_scaling_efficiency(
    base_tps: f64,
    parallel_tps: f64,
    tp: u32,
    min_alpha: f64,
) -> ScalingEfficiencyOutcome {
    if !base_tps.is_finite() || !parallel_tps.is_finite() || !min_alpha.is_finite() {
        return ScalingEfficiencyOutcome::InvalidInput {
            reason: "non-finite input",
        };
    }
    if base_tps <= 0.0 {
        return ScalingEfficiencyOutcome::InvalidInput {
            reason: "base_tps <= 0",
        };
    }
    if parallel_tps < 0.0 {
        return ScalingEfficiencyOutcome::InvalidInput {
            reason: "parallel_tps < 0",
        };
    }
    if tp < 2 {
        return ScalingEfficiencyOutcome::InvalidInput { reason: "tp < 2" };
    }
    if !(0.0..=1.0).contains(&min_alpha) {
        return ScalingEfficiencyOutcome::InvalidInput {
            reason: "min_alpha out of [0.0, 1.0]",
        };
    }
    // observed_alpha = parallel_tps / (base_tps * tp)  (1.0 == perfect linear)
    let observed_alpha = parallel_tps / (base_tps * f64::from(tp));
    if parallel_tps < base_tps {
        return ScalingEfficiencyOutcome::Regression {
            base_tps,
            parallel_tps,
            observed_alpha,
        };
    }
    if observed_alpha < min_alpha {
        return ScalingEfficiencyOutcome::BelowThreshold {
            observed_alpha,
            required_alpha: min_alpha,
            tp,
        };
    }
    ScalingEfficiencyOutcome::Ok { observed_alpha }
}

// ─────────────────────────────────────────────────────────────────────────────
// Classifier 5: `--json` .distributed.{tp, pp, world_size} metadata shape
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DistributedMetadataOutcome {
    Ok { tp: u32, pp: u32, world_size: u32 },
    NotObject,
    MissingDistributed,
    DistributedNotObject,
    MissingField { field: &'static str },
    TypeMismatch { field: &'static str },
    WorldSizeMismatch { tp: u32, pp: u32, got: u32 },
}

pub fn classify_distributed_metadata(value: &Value) -> DistributedMetadataOutcome {
    let root = match value.as_object() {
        Some(o) => o,
        None => return DistributedMetadataOutcome::NotObject,
    };
    let dist = match root.get("distributed") {
        Some(d) => d,
        None => return DistributedMetadataOutcome::MissingDistributed,
    };
    let dist_obj = match dist.as_object() {
        Some(o) => o,
        None => return DistributedMetadataOutcome::DistributedNotObject,
    };

    fn read_u32_field(
        obj: &serde_json::Map<String, Value>,
        field: &'static str,
    ) -> Result<u32, DistributedMetadataOutcome> {
        match obj.get(field) {
            None => Err(DistributedMetadataOutcome::MissingField { field }),
            Some(v) => match v.as_u64() {
                Some(n) if n <= u64::from(u32::MAX) => Ok(n as u32),
                _ => Err(DistributedMetadataOutcome::TypeMismatch { field }),
            },
        }
    }

    let tp = match read_u32_field(dist_obj, "tp") {
        Ok(v) => v,
        Err(e) => return e,
    };
    let pp = match read_u32_field(dist_obj, "pp") {
        Ok(v) => v,
        Err(e) => return e,
    };
    let world_size = match read_u32_field(dist_obj, "world_size") {
        Ok(v) => v,
        Err(e) => return e,
    };

    // Consistency: world_size must equal tp * pp.
    let expected = tp.saturating_mul(pp);
    if world_size != expected {
        return DistributedMetadataOutcome::WorldSizeMismatch {
            tp,
            pp,
            got: world_size,
        };
    }

    DistributedMetadataOutcome::Ok { tp, pp, world_size }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests — all pure, deterministic.
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "tp_pp_classifier_tests.rs"]
mod tests;
