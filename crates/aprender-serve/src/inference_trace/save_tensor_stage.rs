//! Stage enum + comma-list parser for `apr trace --save-tensor`.
//!
//! Contract: [`contracts/apr-cli-trace-save-tensor-v1.yaml`] v1.0.0 (PROPOSED).
//! Sub-stage extension: [`contracts/trace-attn-sub-stages-v1.yaml`] v1.1.0
//! (PROPOSED) — adds `attn_scores` + `attn_softmax` for SHIP-007 layer-0
//! attention bisection.
//!
//! The combined enumeration is 21 capture-point names:
//!
//! ```text
//! embedding, attn_norm, qkv_matmul, qkv_bias, q_post_rope, k_post_rope,
//! attn_scores, attn_softmax, attention, attn_out, post_attn_residual,
//! ffn_norm, ffn_gate, ffn_up, ffn_silu, ffn_swigl, ffn_out,
//! post_ffn_residual, layer_output (alias for post_ffn_residual),
//! final_norm, lm_head
//! ```
//!
//! ## What this module provides
//!
//! - [`SaveTensorStage`] — typed enum over the 19 capture points.
//! - `FromStr` for case-insensitive single-name parsing.
//! - [`SaveTensorStage::is_per_layer`] — distinguishes per-layer stages
//!   (which need a layer index in the file header) from whole-model stages
//!   (`final_norm`, `lm_head`, which use the WHOLE_MODEL_LAYER sentinel).
//! - [`SaveTensorStage::canonical_name`] — exact name as referenced in the
//!   contract; used for both file-naming (`<DIR>/layer-<N>/<NAME>.bin`)
//!   and CLI-help.
//! - [`parse_stage_list`] — comma-delimited list parser, partial-discharges
//!   `FALSIFY-APR-TRACE-SAVE-005` (multi-stage in one run).
//!
//! ## Discharge status
//!
//! Partial-discharge of `FALSIFY-APR-TRACE-SAVE-005` (multi-stage parser)
//! at the parser level. Full discharge requires the `apr trace --save-tensor`
//! CLI implementation that calls the writer at each chosen stage.

use std::str::FromStr;

/// One of the 19 stages where `apr trace --save-tensor` may capture an F32
/// tensor for per-element APR-vs-GGUF comparison.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SaveTensorStage {
    /// Token embedding lookup output.
    Embedding,
    /// Post-RMSNorm pre-QKV.
    AttnNorm,
    /// Post matmul, pre-bias.
    QkvMatmul,
    /// Post-bias add, pre-RoPE.
    QkvBias,
    /// Q after RoPE.
    QPostRope,
    /// K after RoPE.
    KPostRope,
    /// Q·Kᵀ / sqrt(head_dim), pre-softmax + pre-causal-mask.
    /// Per `contracts/trace-attn-sub-stages-v1.yaml` v1.1.0 — closes the
    /// SHIP-007 layer-0 attention bisection gap between `KPostRope` and
    /// `AttnSoftmax`.
    AttnScores,
    /// softmax(scores + causal_mask), pre-V-multiply.
    /// Per `contracts/trace-attn-sub-stages-v1.yaml` v1.1.0 — closes the
    /// SHIP-007 layer-0 attention bisection gap between `AttnScores` and
    /// `Attention`.
    AttnSoftmax,
    /// Post softmax(Q@Kᵀ)@V, pre O-proj.
    Attention,
    /// Post-O-projection.
    AttnOut,
    /// Hidden state post layer-N attention residual.
    PostAttnResidual,
    /// Post-FFN-RMSNorm pre-gate.
    FfnNorm,
    /// Post gate matmul.
    FfnGate,
    /// Post up matmul.
    FfnUp,
    /// silu(gate).
    FfnSilu,
    /// silu(gate) × up.
    FfnSwigl,
    /// Post down-projection.
    FfnOut,
    /// Hidden state post layer-N FFN residual. Same as `LayerOutput`; both
    /// names accepted on parse, `PostFfnResidual` is the canonical form.
    PostFfnResidual,
    /// **MoE-GPU bisection** (per `contracts/trace-moe-gpu-sub-stages-v1.yaml`
    /// v1.0.0, M-MOE-SUB-1): top-k expert weights post-softmax + renormalize.
    /// `[k]` (k = num_experts_per_tok) for the active layer's MoE router output.
    /// Captured AFTER `FfnNorm` and BEFORE the per-expert SwiGLU dispatches.
    /// Used by M-GPU-MOE-1.4 to bisect CPU-vs-GPU divergence at the MoE router
    /// stage independently from the per-expert SwiGLU computation.
    MoeRouter,
    /// **MoE-GPU bisection** (same contract as `MoeRouter`): aggregated MoE
    /// FFN output `Σ_e top_k_w[e] * expert_out[e]` (+ optional shared expert).
    /// `[hidden_dim]` for the active layer. Captured AFTER all per-expert
    /// computations and BEFORE the post-FFN residual add.
    MoeFfnOut,
    /// **MoE-GPU bisection L47 sub-cascade** (per M-GPU-MOE-3 PR-3e2,
    /// issue #1583): top-k expert INDICES post-softmax + argsort. `[k]`
    /// (k = num_experts_per_tok) for the active layer's MoE router output.
    /// Captured AFTER `FfnNorm` and BEFORE the per-expert SwiGLU dispatches.
    ///
    /// Indices are `u32` semantically (expert id in `[0, num_experts)`) but
    /// stored as `f32` to fit the existing `maybe_save_stage` `&[f32]`
    /// write path. The cast is lossless for any `num_experts ≤ 2^24` —
    /// Qwen3 has 128 experts, well within range.
    ///
    /// Sibling of [`MoeRouter`] (weights). The pair (`MoeRouter`,
    /// `MoeRouterIndices`) together capture the full router state needed
    /// to falsify or confirm H(ii) routing-divergence at L47: equal
    /// indices + equal weights ⇒ identical routing decision; equal
    /// weights + DIFFERENT indices ⇒ disjoint expert sets with similar
    /// weight histogram shapes (the case the weight-only `MoeRouter`
    /// vector cannot distinguish, see PR-3e #1741 probe verdict).
    MoeRouterIndices,
    /// Post-output-norm (whole-model, NOT per-layer).
    FinalNorm,
    /// Logits (whole-model, NOT per-layer).
    LmHead,
}

impl SaveTensorStage {
    /// All 23 distinct stages (computation order), excluding the `LayerOutput`
    /// alias for `PostFfnResidual`. `AttnScores` and `AttnSoftmax` are the 2
    /// variants per `contracts/trace-attn-sub-stages-v1.yaml` v1.1.0 (closes
    /// the SHIP-007 layer-0 attention bisection gap inside the Q·Kᵀ → softmax
    /// → ·V chain). `MoeRouter` and `MoeFfnOut` are the 2 variants per
    /// `contracts/trace-moe-gpu-sub-stages-v1.yaml` v1.0.0 (M-MOE-SUB-1, for
    /// the M-GPU-MOE-1.4 NaN/Inf bisection on the GPU MoE FFN path).
    /// `MoeRouterIndices` is added in M-GPU-MOE-3 PR-3e2 (#1583) — top-k
    /// expert INDICES cast to f32, sibling of `MoeRouter` (weights). Pair
    /// is needed to confirm/falsify H(ii) expert-set divergence at L47.
    pub const ALL: [SaveTensorStage; 23] = [
        Self::Embedding,
        Self::AttnNorm,
        Self::QkvMatmul,
        Self::QkvBias,
        Self::QPostRope,
        Self::KPostRope,
        Self::AttnScores,
        Self::AttnSoftmax,
        Self::Attention,
        Self::AttnOut,
        Self::PostAttnResidual,
        Self::FfnNorm,
        Self::FfnGate,
        Self::FfnUp,
        Self::FfnSilu,
        Self::FfnSwigl,
        Self::FfnOut,
        Self::PostFfnResidual,
        Self::MoeRouter,
        Self::MoeFfnOut,
        Self::MoeRouterIndices,
        Self::FinalNorm,
        Self::LmHead,
    ];

    /// Canonical name as referenced in the contract `cli_signature` and as
    /// used for file paths (`<DIR>/layer-<N>/<NAME>.bin`).
    #[must_use]
    pub fn canonical_name(&self) -> &'static str {
        match self {
            Self::Embedding => "embedding",
            Self::AttnNorm => "attn_norm",
            Self::QkvMatmul => "qkv_matmul",
            Self::QkvBias => "qkv_bias",
            Self::QPostRope => "q_post_rope",
            Self::KPostRope => "k_post_rope",
            Self::AttnScores => "attn_scores",
            Self::AttnSoftmax => "attn_softmax",
            Self::Attention => "attention",
            Self::AttnOut => "attn_out",
            Self::PostAttnResidual => "post_attn_residual",
            Self::FfnNorm => "ffn_norm",
            Self::FfnGate => "ffn_gate",
            Self::FfnUp => "ffn_up",
            Self::FfnSilu => "ffn_silu",
            Self::FfnSwigl => "ffn_swigl",
            Self::FfnOut => "ffn_out",
            Self::PostFfnResidual => "post_ffn_residual",
            Self::MoeRouter => "moe_router",
            Self::MoeFfnOut => "moe_ffn_out",
            Self::MoeRouterIndices => "moe_router_indices",
            Self::FinalNorm => "final_norm",
            Self::LmHead => "lm_head",
        }
    }

    /// `true` if this stage emits one tensor per decoder layer; `false` if
    /// it emits exactly one whole-model tensor (used for `final_norm` and
    /// `lm_head` only).
    #[must_use]
    pub fn is_per_layer(&self) -> bool {
        !matches!(self, Self::FinalNorm | Self::LmHead)
    }

    /// `true` if this stage stores **integer indices cast to f32** rather
    /// than genuine f32 activations. The cast is lossless for `num_experts
    /// ≤ 2^24`. Currently only [`MoeRouterIndices`]. Downstream consumers
    /// (e.g. apr diff) MUST reinterpret these values as `u32` for equality
    /// comparison — cosine on indices would be meaningless.
    #[must_use]
    pub fn is_index_payload(&self) -> bool {
        matches!(self, Self::MoeRouterIndices)
    }
}

/// Errors that can arise parsing a stage name.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum StageParseError {
    /// Not one of the recognised stage names.
    #[error("unknown save-tensor stage {got:?}; valid stages: {valid}")]
    Unknown {
        /// The unrecognised input string.
        got: String,
        /// Comma-joined list of valid stage names for help text.
        valid: String,
    },
    /// Empty token (e.g. an empty `--save-tensor` value, or a stray comma
    /// like `embedding,,ffn_gate`).
    #[error("save-tensor stage cannot be an empty string")]
    Empty,
}

impl FromStr for SaveTensorStage {
    type Err = StageParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let trimmed = s.trim();
        if trimmed.is_empty() {
            return Err(StageParseError::Empty);
        }
        match trimmed.to_lowercase().as_str() {
            "embedding" => Ok(Self::Embedding),
            "attn_norm" => Ok(Self::AttnNorm),
            "qkv_matmul" => Ok(Self::QkvMatmul),
            "qkv_bias" => Ok(Self::QkvBias),
            "q_post_rope" => Ok(Self::QPostRope),
            "k_post_rope" => Ok(Self::KPostRope),
            "attn_scores" => Ok(Self::AttnScores),
            "attn_softmax" => Ok(Self::AttnSoftmax),
            "attention" => Ok(Self::Attention),
            "attn_out" => Ok(Self::AttnOut),
            "post_attn_residual" => Ok(Self::PostAttnResidual),
            "ffn_norm" => Ok(Self::FfnNorm),
            "ffn_gate" => Ok(Self::FfnGate),
            "ffn_up" => Ok(Self::FfnUp),
            "ffn_silu" => Ok(Self::FfnSilu),
            "ffn_swigl" => Ok(Self::FfnSwigl),
            "ffn_out" => Ok(Self::FfnOut),
            "post_ffn_residual" | "layer_output" => Ok(Self::PostFfnResidual),
            "moe_router" => Ok(Self::MoeRouter),
            "moe_ffn_out" => Ok(Self::MoeFfnOut),
            "moe_router_indices" => Ok(Self::MoeRouterIndices),
            "final_norm" => Ok(Self::FinalNorm),
            "lm_head" => Ok(Self::LmHead),
            _ => Err(StageParseError::Unknown {
                got: trimmed.to_string(),
                valid: SaveTensorStage::ALL
                    .iter()
                    .map(|s| s.canonical_name())
                    .collect::<Vec<_>>()
                    .join(","),
            }),
        }
    }
}

/// Parse a comma-delimited list of stage names.
///
/// Whitespace around commas is tolerated. Duplicates are preserved
/// (caller decides whether to dedupe). Empty list parses to `Ok(vec![])`
/// — a no-op `--save-tensor=` is treated as "no stages selected", same as
/// not passing the flag at all. A list containing an empty token like
/// `embedding,,ffn_gate` is a parse error.
///
/// # Errors
///
/// Returns [`StageParseError`] on the first bad token; remaining tokens
/// are NOT parsed.
///
/// # Example
///
/// ```
/// # use realizar::inference_trace::save_tensor_stage::{parse_stage_list, SaveTensorStage};
/// let stages = parse_stage_list("embedding, ffn_gate ,ffn_swigl").unwrap();
/// assert_eq!(stages, vec![
///     SaveTensorStage::Embedding,
///     SaveTensorStage::FfnGate,
///     SaveTensorStage::FfnSwigl,
/// ]);
/// ```
pub fn parse_stage_list(s: &str) -> Result<Vec<SaveTensorStage>, StageParseError> {
    let trimmed = s.trim();
    if trimmed.is_empty() {
        return Ok(vec![]);
    }
    trimmed.split(',').map(SaveTensorStage::from_str).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_stages_have_unique_canonical_names() {
        let mut names: Vec<&str> = SaveTensorStage::ALL
            .iter()
            .map(|s| s.canonical_name())
            .collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(
            names.len(),
            SaveTensorStage::ALL.len(),
            "stage canonical_names must be unique"
        );
    }

    #[test]
    fn canonical_names_match_contract_enumeration() {
        // Per apr-cli-trace-save-tensor-v1.yaml `cli_signature` equation +
        // trace-attn-sub-stages-v1.yaml v1.1.0 (attn_scores, attn_softmax) +
        // trace-moe-gpu-sub-stages-v1.yaml v1.0.0 (moe_router, moe_ffn_out).
        let expected = [
            "embedding",
            "attn_norm",
            "qkv_matmul",
            "qkv_bias",
            "q_post_rope",
            "k_post_rope",
            "attn_scores",
            "attn_softmax",
            "attention",
            "attn_out",
            "post_attn_residual",
            "ffn_norm",
            "ffn_gate",
            "ffn_up",
            "ffn_silu",
            "ffn_swigl",
            "ffn_out",
            "post_ffn_residual",
            "moe_router",
            "moe_ffn_out",
            "moe_router_indices",
            "final_norm",
            "lm_head",
        ];
        let actual: Vec<&str> = SaveTensorStage::ALL
            .iter()
            .map(|s| s.canonical_name())
            .collect();
        assert_eq!(actual, expected);
    }

    #[test]
    fn from_str_round_trip_for_every_canonical_name() {
        for stage in SaveTensorStage::ALL {
            let parsed: SaveTensorStage = stage
                .canonical_name()
                .parse()
                .unwrap_or_else(|_| panic!("canonical name must round-trip: {stage:?}"));
            assert_eq!(parsed, stage);
        }
    }

    #[test]
    fn from_str_is_case_insensitive() {
        let parsed: SaveTensorStage = "FFN_GATE".parse().expect("upper-case must parse");
        assert_eq!(parsed, SaveTensorStage::FfnGate);
        let parsed: SaveTensorStage = "Ffn_Gate".parse().expect("mixed-case must parse");
        assert_eq!(parsed, SaveTensorStage::FfnGate);
    }

    #[test]
    fn from_str_trims_whitespace() {
        let parsed: SaveTensorStage = "  embedding  ".parse().expect("trim must work");
        assert_eq!(parsed, SaveTensorStage::Embedding);
    }

    #[test]
    fn from_str_layer_output_is_alias_for_post_ffn_residual() {
        let alias: SaveTensorStage = "layer_output".parse().expect("alias must parse");
        let canonical: SaveTensorStage = "post_ffn_residual".parse().unwrap();
        assert_eq!(alias, canonical);
        assert_eq!(alias, SaveTensorStage::PostFfnResidual);
    }

    #[test]
    fn from_str_rejects_empty() {
        let err = SaveTensorStage::from_str("").expect_err("empty must fail");
        assert_eq!(err, StageParseError::Empty);
        let err = SaveTensorStage::from_str("   ").expect_err("whitespace-only must fail");
        assert_eq!(err, StageParseError::Empty);
    }

    #[test]
    fn from_str_rejects_unknown_stage() {
        let err = SaveTensorStage::from_str("not_a_stage").expect_err("unknown must fail");
        match err {
            StageParseError::Unknown { got, valid } => {
                assert_eq!(got, "not_a_stage");
                assert!(valid.contains("embedding"));
                assert!(valid.contains("ffn_gate"));
            },
            StageParseError::Empty => panic!("expected Unknown, got Empty"),
        }
    }

    #[test]
    fn is_per_layer_correct_for_each_stage() {
        for stage in SaveTensorStage::ALL {
            let expected = !matches!(stage, SaveTensorStage::FinalNorm | SaveTensorStage::LmHead);
            assert_eq!(
                stage.is_per_layer(),
                expected,
                "is_per_layer mismatch for {stage:?}"
            );
        }
    }

    #[test]
    fn is_per_layer_count_matches_contract() {
        // Per parent + sub-stages contracts: per-layer stages (one per decoder
        // layer N) + 2 whole-model stages (final_norm, lm_head). The
        // PostFfnResidual / LayerOutput alias collapses to one variant.
        //   - trace-attn-sub-stages-v1 v1.1.0 added attn_scores + attn_softmax
        //   - trace-moe-gpu-sub-stages-v1 v1.0.0 added moe_router + moe_ffn_out
        //   - M-GPU-MOE-3 PR-3e2 (#1583) added moe_router_indices
        // total distinct stages = 23; per-layer = 21; whole-model = 2.
        let per_layer = SaveTensorStage::ALL
            .iter()
            .filter(|s| s.is_per_layer())
            .count();
        let whole_model = SaveTensorStage::ALL
            .iter()
            .filter(|s| !s.is_per_layer())
            .count();
        assert_eq!(per_layer, 21);
        assert_eq!(whole_model, 2);
        assert_eq!(per_layer + whole_model, SaveTensorStage::ALL.len());
    }

    // =========================================================================
    // FALSIFY-ATTN-SUB-001 (trace-attn-sub-stages-v1.yaml v1.1.0): the 2 new
    // sub-stage variants exist on `SaveTensorStage` enum without breaking
    // existing callers. Round-trip + parse-list coverage for AttnScores and
    // AttnSoftmax.
    // =========================================================================

    #[test]
    fn falsify_attn_sub_001_attn_scores_round_trip() {
        let parsed: SaveTensorStage = "attn_scores".parse().expect("attn_scores must parse");
        assert_eq!(parsed, SaveTensorStage::AttnScores);
        assert_eq!(SaveTensorStage::AttnScores.canonical_name(), "attn_scores");
        assert!(SaveTensorStage::AttnScores.is_per_layer());
    }

    #[test]
    fn falsify_attn_sub_001_attn_softmax_round_trip() {
        let parsed: SaveTensorStage = "attn_softmax".parse().expect("attn_softmax must parse");
        assert_eq!(parsed, SaveTensorStage::AttnSoftmax);
        assert_eq!(
            SaveTensorStage::AttnSoftmax.canonical_name(),
            "attn_softmax"
        );
        assert!(SaveTensorStage::AttnSoftmax.is_per_layer());
    }

    #[test]
    fn falsify_attn_sub_001_2_new_stages_in_canonical_order() {
        // Per trace-attn-sub-stages-v1.yaml v1.1.0 ordering proof_obligation:
        // QkvBias → QPostRope → KPostRope → AttnScores → AttnSoftmax → Attention → AttnOut
        let attn_block: Vec<SaveTensorStage> = SaveTensorStage::ALL
            .iter()
            .copied()
            .skip_while(|s| !matches!(s, SaveTensorStage::QkvBias))
            .take_while(|s| {
                !matches!(
                    s,
                    SaveTensorStage::PostAttnResidual | SaveTensorStage::FfnNorm
                )
            })
            .collect();
        assert_eq!(
            attn_block,
            vec![
                SaveTensorStage::QkvBias,
                SaveTensorStage::QPostRope,
                SaveTensorStage::KPostRope,
                SaveTensorStage::AttnScores,
                SaveTensorStage::AttnSoftmax,
                SaveTensorStage::Attention,
                SaveTensorStage::AttnOut,
            ]
        );
    }

    #[test]
    fn falsify_attn_sub_001_parse_list_accepts_2_new_stages_together() {
        let stages =
            parse_stage_list("attn_scores,attn_softmax").expect("2-element comma list must parse");
        assert_eq!(
            stages,
            vec![SaveTensorStage::AttnScores, SaveTensorStage::AttnSoftmax]
        );
    }

    #[test]
    fn falsify_attn_sub_001_parse_list_accepts_full_attn_block_chain() {
        // Per trace-attn-sub-stages-v1.yaml `bisection_chain_layer_0` equation:
        // the 9-element cosine sequence requires all 9 stage names parsing
        // cleanly in one comma-delimited call.
        let stages = parse_stage_list(
            "attn_norm,qkv_matmul,qkv_bias,q_post_rope,k_post_rope,attn_scores,attn_softmax,attention,attn_out",
        )
        .expect("9-stage layer-0 attention chain must parse");
        assert_eq!(stages.len(), 9);
        assert_eq!(stages[5], SaveTensorStage::AttnScores);
        assert_eq!(stages[6], SaveTensorStage::AttnSoftmax);
    }

    // =========================================================================
    // FALSIFY-MOE-SUB-001 (trace-moe-gpu-sub-stages-v1.yaml v1.0.0): the 2 new
    // MoE-GPU sub-stage variants exist on `SaveTensorStage` enum without
    // breaking existing callers. Round-trip + parse-list coverage for
    // MoeRouter and MoeFfnOut. Discharges M-MOE-SUB-1 acceptance criterion.
    // =========================================================================

    #[test]
    fn falsify_moe_sub_001_moe_router_round_trip() {
        let parsed: SaveTensorStage = "moe_router".parse().expect("moe_router must parse");
        assert_eq!(parsed, SaveTensorStage::MoeRouter);
        assert_eq!(SaveTensorStage::MoeRouter.canonical_name(), "moe_router");
        assert!(SaveTensorStage::MoeRouter.is_per_layer());
    }

    #[test]
    fn falsify_moe_sub_001_moe_ffn_out_round_trip() {
        let parsed: SaveTensorStage = "moe_ffn_out".parse().expect("moe_ffn_out must parse");
        assert_eq!(parsed, SaveTensorStage::MoeFfnOut);
        assert_eq!(SaveTensorStage::MoeFfnOut.canonical_name(), "moe_ffn_out");
        assert!(SaveTensorStage::MoeFfnOut.is_per_layer());
    }

    #[test]
    fn falsify_moe_sub_001_2_new_stages_in_canonical_order() {
        // Per trace-moe-gpu-sub-stages-v1.yaml v1.0.0 ordering proof_obligation:
        // FfnNorm → MoeRouter → MoeFfnOut → PostFfnResidual (NOTE: MoeRouter
        // and MoeFfnOut are placed AFTER PostFfnResidual in `ALL` for back-
        // compat with the existing 18-stage ordering; the contract's
        // "moe_block_order" is logical ordering, not array position.)
        //
        // Position assertion: MoeRouter and MoeFfnOut appear in `ALL` after
        // PostFfnResidual and before FinalNorm.
        let post_ffn_idx = SaveTensorStage::ALL
            .iter()
            .position(|s| matches!(s, SaveTensorStage::PostFfnResidual))
            .expect("PostFfnResidual must be in ALL");
        let moe_router_idx = SaveTensorStage::ALL
            .iter()
            .position(|s| matches!(s, SaveTensorStage::MoeRouter))
            .expect("MoeRouter must be in ALL");
        let moe_ffn_out_idx = SaveTensorStage::ALL
            .iter()
            .position(|s| matches!(s, SaveTensorStage::MoeFfnOut))
            .expect("MoeFfnOut must be in ALL");
        let final_norm_idx = SaveTensorStage::ALL
            .iter()
            .position(|s| matches!(s, SaveTensorStage::FinalNorm))
            .expect("FinalNorm must be in ALL");
        assert!(post_ffn_idx < moe_router_idx);
        assert!(moe_router_idx < moe_ffn_out_idx);
        assert!(moe_ffn_out_idx < final_norm_idx);
    }

    #[test]
    fn falsify_moe_sub_001_parse_list_accepts_2_new_stages_together() {
        let stages =
            parse_stage_list("moe_router,moe_ffn_out").expect("2-element comma list must parse");
        assert_eq!(
            stages,
            vec![SaveTensorStage::MoeRouter, SaveTensorStage::MoeFfnOut]
        );
    }

    #[test]
    fn falsify_moe_sub_001_parse_list_accepts_full_moe_block_chain() {
        // Per trace-moe-gpu-sub-stages-v1.yaml `bisection_chain_moe_gpu`
        // equation: 3-element cosine sequence (ffn_norm + moe_router +
        // moe_ffn_out) for the M-GPU-MOE-1.4 bisection on lambda-vector.
        let stages = parse_stage_list("ffn_norm,moe_router,moe_ffn_out")
            .expect("3-stage MoE-GPU bisection chain must parse");
        assert_eq!(stages.len(), 3);
        assert_eq!(stages[0], SaveTensorStage::FfnNorm);
        assert_eq!(stages[1], SaveTensorStage::MoeRouter);
        assert_eq!(stages[2], SaveTensorStage::MoeFfnOut);
    }

    // =========================================================================
    // FALSIFY-APR-TRACE-SAVE-005 (multi-stage in one run): parser-level
    // partial-discharge. Full discharge requires the CLI implementation that
    // produces 3 files per layer when `--save-tensor embedding,ffn_gate,ffn_swigl`
    // is passed; this test pins the input parsing.
    // =========================================================================

    #[test]
    fn falsify_apr_trace_save_005_multi_stage_parsing() {
        let stages = parse_stage_list("embedding,ffn_gate,ffn_swigl")
            .expect("3-element comma list must parse");
        assert_eq!(
            stages,
            vec![
                SaveTensorStage::Embedding,
                SaveTensorStage::FfnGate,
                SaveTensorStage::FfnSwigl,
            ]
        );
    }

    #[test]
    fn parse_stage_list_tolerates_whitespace_around_commas() {
        let stages = parse_stage_list("embedding, ffn_gate , ffn_swigl ")
            .expect("whitespace must be trimmed");
        assert_eq!(
            stages,
            vec![
                SaveTensorStage::Embedding,
                SaveTensorStage::FfnGate,
                SaveTensorStage::FfnSwigl,
            ]
        );
    }

    #[test]
    fn parse_stage_list_empty_returns_empty_vec() {
        assert_eq!(parse_stage_list("").unwrap(), vec![]);
        assert_eq!(parse_stage_list("   ").unwrap(), vec![]);
    }

    #[test]
    fn parse_stage_list_rejects_double_comma() {
        let err = parse_stage_list("embedding,,ffn_gate").expect_err("double comma must fail");
        assert_eq!(err, StageParseError::Empty);
    }

    #[test]
    fn parse_stage_list_rejects_trailing_comma() {
        let err = parse_stage_list("embedding,ffn_gate,").expect_err("trailing comma must fail");
        assert_eq!(err, StageParseError::Empty);
    }

    #[test]
    fn parse_stage_list_rejects_unknown_token() {
        let err = parse_stage_list("embedding,not_a_stage,ffn_gate")
            .expect_err("unknown token must fail");
        assert!(matches!(err, StageParseError::Unknown { .. }));
    }

    #[test]
    fn parse_stage_list_preserves_duplicates() {
        // Caller decides whether to dedupe (e.g., to avoid double-write).
        let stages = parse_stage_list("ffn_gate,ffn_gate,ffn_gate").expect("dupes must parse");
        assert_eq!(stages, vec![SaveTensorStage::FfnGate; 3]);
    }

    #[test]
    fn parse_stage_list_single_stage() {
        let stages = parse_stage_list("ffn_swigl").expect("single must parse");
        assert_eq!(stages, vec![SaveTensorStage::FfnSwigl]);
    }

    #[test]
    fn parse_stage_list_all_stages_in_one_call() {
        let csv = SaveTensorStage::ALL
            .iter()
            .map(|s| s.canonical_name())
            .collect::<Vec<_>>()
            .join(",");
        let stages = parse_stage_list(&csv).expect("full list must parse");
        assert_eq!(stages, SaveTensorStage::ALL.to_vec());
    }
}
