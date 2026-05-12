// SHIP-TWO-001 — `qwen35-hybrid-forward-v1` algorithm-level PARTIAL
// discharge for FALSIFY-QHF-001..007 (closes 7/7 sweep).
//
// Contract: `contracts/qwen35-hybrid-forward-v1.yaml`.
// Spec: Qwen3.5 hybrid forward pass — attention/GDN layer interleaving
// with numerical stability (pre-norm, residual stream, no NaN through L).

// ===========================================================================
// QHF-001 — Attention sublayer shape preservation: shape(attn(x)) == shape(x)
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qhf001Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_attn_shape(input: &[u64], output: &[u64]) -> Qhf001Verdict {
    if input.is_empty() || output.is_empty() { return Qhf001Verdict::Fail; }
    if input == output { Qhf001Verdict::Pass } else { Qhf001Verdict::Fail }
}

// ===========================================================================
// QHF-002 — GDN sublayer shape preservation: shape(gdn(x)) == shape(x)
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qhf002Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_gdn_shape(input: &[u64], output: &[u64]) -> Qhf002Verdict {
    if input.is_empty() || output.is_empty() { return Qhf002Verdict::Fail; }
    if input == output { Qhf002Verdict::Pass } else { Qhf002Verdict::Fail }
}

// ===========================================================================
// QHF-003 — FFN sublayer shape preservation: shape(ffn(x)) == shape(x)
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qhf003Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_ffn_shape(input: &[u64], output: &[u64]) -> Qhf003Verdict {
    if input.is_empty() || output.is_empty() { return Qhf003Verdict::Fail; }
    if input == output { Qhf003Verdict::Pass } else { Qhf003Verdict::Fail }
}

// ===========================================================================
// QHF-004 — Each layer is attention XOR GDN (exclusive partition)
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerType { Attention, Gdn }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qhf004Verdict { Pass, Fail }

/// Pass iff every layer in the schedule has exactly one assigned type
/// (the LayerType enum makes "both" structurally impossible — what we
/// must guard against is empty schedules and confirm len > 0).
#[must_use]
pub fn verdict_from_layer_partition(schedule: &[LayerType]) -> Qhf004Verdict {
    if schedule.is_empty() { return Qhf004Verdict::Fail; }
    Qhf004Verdict::Pass
}

/// Stronger variant: layer schedule must contain at least one of each
/// type (true "hybrid" architecture; pure-attention or pure-GDN fails).
#[must_use]
pub fn verdict_from_hybrid_schedule(schedule: &[LayerType]) -> Qhf004Verdict {
    if schedule.is_empty() { return Qhf004Verdict::Fail; }
    let has_attn = schedule.iter().any(|t| *t == LayerType::Attention);
    let has_gdn = schedule.iter().any(|t| *t == LayerType::Gdn);
    if has_attn && has_gdn { Qhf004Verdict::Pass } else { Qhf004Verdict::Fail }
}

// ===========================================================================
// QHF-005 — Activation stability: no NaN/Inf after L layers
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qhf005Verdict { Pass, Fail }

#[must_use]
pub fn verdict_from_activation_stability(activations: &[f32]) -> Qhf005Verdict {
    if activations.is_empty() { return Qhf005Verdict::Fail; }
    if activations.iter().all(|v| v.is_finite()) { Qhf005Verdict::Pass } else { Qhf005Verdict::Fail }
}

/// Magnitude-bounded variant: all entries finite AND |a| <= bound.
#[must_use]
pub fn verdict_from_activation_magnitude(activations: &[f32], bound: f32) -> Qhf005Verdict {
    if activations.is_empty() || !bound.is_finite() || bound <= 0.0 { return Qhf005Verdict::Fail; }
    if activations.iter().all(|v| v.is_finite() && v.abs() <= bound) {
        Qhf005Verdict::Pass
    } else {
        Qhf005Verdict::Fail
    }
}

// ===========================================================================
// QHF-006 — Pre-norm architecture: RMSNorm precedes each sublayer
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SublayerOp { RmsNorm, Attention, Gdn, Ffn }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qhf006Verdict { Pass, Fail }

/// Pass iff every Attention/Gdn/Ffn op is immediately preceded by an
/// RmsNorm in the trace. Falsifies the "missing pre-norm" regression
/// (e.g., if someone wires h directly into attention without norm).
#[must_use]
pub fn verdict_from_pre_norm_trace(trace: &[SublayerOp]) -> Qhf006Verdict {
    if trace.is_empty() { return Qhf006Verdict::Fail; }
    for (i, op) in trace.iter().enumerate() {
        match op {
            SublayerOp::Attention | SublayerOp::Gdn | SublayerOp::Ffn => {
                if i == 0 { return Qhf006Verdict::Fail; } // sublayer with nothing before it
                if trace[i - 1] != SublayerOp::RmsNorm { return Qhf006Verdict::Fail; }
            }
            SublayerOp::RmsNorm => {}
        }
    }
    Qhf006Verdict::Pass
}

// ===========================================================================
// QHF-007 — Residual: h_{l+1} - h_l = sublayer(norm(h_l)) within tolerance
// ===========================================================================

pub const AC_QHF_007_TOLERANCE: f32 = 1.0e-6;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Qhf007Verdict { Pass, Fail }

/// Pass iff `||h_next - h_prev - sublayer_out||_inf <= tolerance` for all dims.
#[must_use]
pub fn verdict_from_residual_arith(
    h_prev: &[f32],
    h_next: &[f32],
    sublayer_out: &[f32],
) -> Qhf007Verdict {
    if h_prev.is_empty() || h_next.is_empty() || sublayer_out.is_empty() {
        return Qhf007Verdict::Fail;
    }
    if h_prev.len() != h_next.len() || h_prev.len() != sublayer_out.len() {
        return Qhf007Verdict::Fail;
    }
    for ((&a, &b), &c) in h_prev.iter().zip(h_next.iter()).zip(sublayer_out.iter()) {
        let residual = b - a;
        let drift = (residual - c).abs();
        if !drift.is_finite() || drift > AC_QHF_007_TOLERANCE {
            return Qhf007Verdict::Fail;
        }
    }
    Qhf007Verdict::Pass
}

#[cfg(test)]
mod tests {
    use super::*;

    // QHF-001 (attention shape)
    #[test] fn qhf001_pass_match() {
        let s = vec![16, 4096];
        assert_eq!(verdict_from_attn_shape(&s, &s), Qhf001Verdict::Pass);
    }
    #[test] fn qhf001_fail_drift() {
        assert_eq!(verdict_from_attn_shape(&[16, 4096], &[16, 3584]), Qhf001Verdict::Fail);
    }
    #[test] fn qhf001_fail_empty() {
        assert_eq!(verdict_from_attn_shape(&[], &[16, 4096]), Qhf001Verdict::Fail);
    }

    // QHF-002 (GDN shape)
    #[test] fn qhf002_pass_match() {
        let s = vec![16, 4096];
        assert_eq!(verdict_from_gdn_shape(&s, &s), Qhf002Verdict::Pass);
    }
    #[test] fn qhf002_fail_drift() {
        assert_eq!(verdict_from_gdn_shape(&[16, 4096], &[16, 4097]), Qhf002Verdict::Fail);
    }

    // QHF-003 (FFN shape)
    #[test] fn qhf003_pass_match() {
        let s = vec![16, 4096];
        assert_eq!(verdict_from_ffn_shape(&s, &s), Qhf003Verdict::Pass);
    }
    #[test] fn qhf003_fail_drift() {
        // SwiGLU intermediate 3*d but down-proj must restore d_model.
        assert_eq!(verdict_from_ffn_shape(&[16, 4096], &[16, 12288]), Qhf003Verdict::Fail);
    }

    // QHF-004 (layer partition)
    #[test] fn qhf004_pass_partition() {
        let sched = [LayerType::Attention, LayerType::Gdn, LayerType::Attention, LayerType::Gdn];
        assert_eq!(verdict_from_layer_partition(&sched), Qhf004Verdict::Pass);
    }
    #[test] fn qhf004_fail_empty() {
        let sched: [LayerType; 0] = [];
        assert_eq!(verdict_from_layer_partition(&sched), Qhf004Verdict::Fail);
    }
    #[test] fn qhf004_pass_hybrid() {
        let sched = [LayerType::Attention, LayerType::Gdn, LayerType::Attention];
        assert_eq!(verdict_from_hybrid_schedule(&sched), Qhf004Verdict::Pass);
    }
    #[test] fn qhf004_fail_pure_attention() {
        let sched = [LayerType::Attention, LayerType::Attention];
        assert_eq!(verdict_from_hybrid_schedule(&sched), Qhf004Verdict::Fail);
    }
    #[test] fn qhf004_fail_pure_gdn() {
        let sched = [LayerType::Gdn, LayerType::Gdn];
        assert_eq!(verdict_from_hybrid_schedule(&sched), Qhf004Verdict::Fail);
    }

    // QHF-005 (stability)
    #[test] fn qhf005_pass_finite() {
        let a = [0.1, 0.5, -0.3, 1.0];
        assert_eq!(verdict_from_activation_stability(&a), Qhf005Verdict::Pass);
    }
    #[test] fn qhf005_fail_nan() {
        let a = [0.1, f32::NAN, -0.3, 1.0];
        assert_eq!(verdict_from_activation_stability(&a), Qhf005Verdict::Fail);
    }
    #[test] fn qhf005_fail_inf() {
        let a = [0.1, f32::INFINITY, -0.3, 1.0];
        assert_eq!(verdict_from_activation_stability(&a), Qhf005Verdict::Fail);
    }
    #[test] fn qhf005_pass_magnitude_bounded() {
        let a = [0.1, 0.5, -0.3, 1.0];
        assert_eq!(verdict_from_activation_magnitude(&a, 10.0), Qhf005Verdict::Pass);
    }
    #[test] fn qhf005_fail_magnitude_exceeded() {
        let a = [0.1, 0.5, -0.3, 100.0];
        assert_eq!(verdict_from_activation_magnitude(&a, 10.0), Qhf005Verdict::Fail);
    }

    // QHF-006 (pre-norm)
    #[test] fn qhf006_pass_canonical() {
        let trace = [
            SublayerOp::RmsNorm, SublayerOp::Attention,
            SublayerOp::RmsNorm, SublayerOp::Ffn,
        ];
        assert_eq!(verdict_from_pre_norm_trace(&trace), Qhf006Verdict::Pass);
    }
    #[test] fn qhf006_pass_gdn_block() {
        let trace = [
            SublayerOp::RmsNorm, SublayerOp::Gdn,
            SublayerOp::RmsNorm, SublayerOp::Ffn,
        ];
        assert_eq!(verdict_from_pre_norm_trace(&trace), Qhf006Verdict::Pass);
    }
    #[test] fn qhf006_fail_missing_norm_before_attn() {
        // The exact regression: attention with no preceding norm.
        let trace = [SublayerOp::Attention, SublayerOp::RmsNorm, SublayerOp::Ffn];
        assert_eq!(verdict_from_pre_norm_trace(&trace), Qhf006Verdict::Fail);
    }
    #[test] fn qhf006_fail_norm_after_sublayer() {
        // Post-norm (norm placed after, not before) violates pre-norm.
        let trace = [SublayerOp::Attention, SublayerOp::RmsNorm];
        assert_eq!(verdict_from_pre_norm_trace(&trace), Qhf006Verdict::Fail);
    }

    // QHF-007 (residual arithmetic)
    #[test] fn qhf007_pass_clean_residual() {
        let h_prev = [1.0_f32, 2.0, 3.0];
        let sublayer = [0.1_f32, -0.2, 0.5];
        let h_next = [1.1_f32, 1.8, 3.5]; // = h_prev + sublayer exactly
        assert_eq!(verdict_from_residual_arith(&h_prev, &h_next, &sublayer), Qhf007Verdict::Pass);
    }
    #[test] fn qhf007_pass_within_tolerance() {
        let h_prev = [1.0_f32];
        let sublayer = [0.1_f32];
        let h_next = [1.1_f32 + 1.0e-7]; // within 1e-6
        assert_eq!(verdict_from_residual_arith(&h_prev, &h_next, &sublayer), Qhf007Verdict::Pass);
    }
    #[test] fn qhf007_fail_above_tolerance() {
        let h_prev = [1.0_f32];
        let sublayer = [0.1_f32];
        let h_next = [1.5_f32]; // residual=0.5, sublayer_out=0.1, drift=0.4 > 1e-6
        assert_eq!(verdict_from_residual_arith(&h_prev, &h_next, &sublayer), Qhf007Verdict::Fail);
    }
    #[test] fn qhf007_fail_length_mismatch() {
        let h_prev = [1.0_f32, 2.0];
        let sublayer = [0.1_f32];
        let h_next = [1.1_f32, 2.1];
        assert_eq!(verdict_from_residual_arith(&h_prev, &h_next, &sublayer), Qhf007Verdict::Fail);
    }
    #[test] fn qhf007_fail_empty() {
        assert_eq!(verdict_from_residual_arith(&[], &[], &[]), Qhf007Verdict::Fail);
    }

    // Provenance
    #[test] fn provenance_constants() {
        assert!((AC_QHF_007_TOLERANCE - 1.0e-6).abs() < 1e-12);
    }
}
