//! Plan-builder for `apr trace --save-tensor` (SHIP-007 PR-B).
//!
//! Contract: [`contracts/apr-cli-trace-save-tensor-v1.yaml`] v1.0.0 (PROPOSED).
//!
//! ## Role
//!
//! [`SaveTensorPlan`] is the bridge between the parsed CLI arguments
//! (`--save-tensor <STAGES>`, `--save-tensor-dir <DIR>`,
//! `--save-tensor-layers <RANGE>`) and the future `forward_traced` integration
//! that will consult the plan at every stage transition.
//!
//! PR-B keeps this plan **pure** (no I/O, no model state, no transformer
//! coupling) so the CLI dispatch site can validate the user's args eagerly
//! and the forward pass (PR-C) can attach the plan as a side-channel later
//! without dragging clap-specific types into `aprender-serve`.
//!
//! ## Decisions captured here
//!
//! - The `all` keyword expands to [`SaveTensorStage::ALL`] (all 20 stages).
//! - Stage list is comma-delimited (existing [`parse_stage_list`] semantics).
//! - Layer range is parsed as Rust `START..END` syntax with `END` exclusive.
//!   `0..1` (the clap default) selects layer 0 only.
//! - The layer filter applies **only** to per-layer stages
//!   ([`SaveTensorStage::is_per_layer`] = true). Whole-model stages
//!   (`final_norm`, `lm_head`) are always saved if selected, regardless of
//!   the layer range.
//!
//! ## What this module does NOT do (deferred)
//!
//! - PR-C: thread the plan into [`AprTransformer::forward_traced`] and call
//!   [`super::save_tensor_compose::write_stage_file`] at each stage boundary.
//! - PR-D: live integration test on a real APR teacher.
//!
//! Keeping the plan-builder isolated from the forward pass means a typo
//! like `--save-tensor wrng_stage` errors out at CLI dispatch time, not
//! after a multi-second model load.

use std::ops::Range;
use std::path::{Path, PathBuf};

use super::save_tensor::WHOLE_MODEL_LAYER;
use super::save_tensor_paths::output_path;
use super::save_tensor_stage::{parse_stage_list, SaveTensorStage, StageParseError};

/// The validated, ready-to-execute plan for `apr trace --save-tensor`.
///
/// Built from CLI strings via [`SaveTensorPlan::from_cli`]. Once constructed,
/// the plan is consulted by the forward pass (PR-C) to decide whether each
/// stage produces a file, and where it goes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SaveTensorPlan {
    /// The stages selected by the user, in the order they were listed.
    /// `all` expands to [`SaveTensorStage::ALL`] in canonical order.
    /// Empty plans are not constructable — `--save-tensor` with no value
    /// errors out at parse time.
    pub stages: Vec<SaveTensorStage>,
    /// Layer range as parsed from `--save-tensor-layers` (`START..END`,
    /// END-exclusive). Whole-model stages ignore this filter.
    pub layer_range: Range<u32>,
    /// Root directory for saved tensors. Per-layer stages go under
    /// `<output_dir>/layer-<N>/<stage>.bin`; whole-model stages go under
    /// `<output_dir>/<stage>.bin`.
    pub output_dir: PathBuf,
}

/// Errors that can arise building a [`SaveTensorPlan`] from CLI strings.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum PlanParseError {
    /// A stage name was unrecognised (forwarded from
    /// [`super::save_tensor_stage::parse_stage_list`]).
    #[error("save-tensor stage list invalid: {0}")]
    Stage(StageParseError),
    /// The `--save-tensor-layers` string did not match `START..END` syntax,
    /// or the bounds were not parseable as `u32`, or `END <= START`.
    #[error("save-tensor layer range invalid ({got:?}): {reason}")]
    LayerRange {
        /// The original input string.
        got: String,
        /// One-line human-readable explanation.
        reason: String,
    },
}

impl SaveTensorPlan {
    /// Build a [`SaveTensorPlan`] from the raw clap argument strings.
    ///
    /// `stages_str` accepts either a comma-separated list of stage names
    /// (e.g. `"embedding,qkv_matmul,attention"`) or the literal token
    /// `"all"` (case-insensitive) which expands to every stage in
    /// [`SaveTensorStage::ALL`].
    ///
    /// `layer_range_str` is `START..END` (Rust Range syntax, END exclusive).
    /// `output_dir` is taken as-is (no auto-creation here; PR-C ensures the
    /// directory exists before writing the first file).
    ///
    /// # Errors
    ///
    /// - [`PlanParseError::Stage`] if any token in `stages_str` is not a
    ///   known canonical stage name (or alias `layer_output`).
    /// - [`PlanParseError::LayerRange`] if `layer_range_str` is malformed,
    ///   or if `END <= START`.
    pub fn from_cli(
        stages_str: &str,
        layer_range_str: &str,
        output_dir: PathBuf,
    ) -> Result<Self, PlanParseError> {
        let stages = parse_stages_arg(stages_str).map_err(PlanParseError::Stage)?;
        let layer_range = parse_layer_range(layer_range_str)?;
        Ok(Self {
            stages,
            layer_range,
            output_dir,
        })
    }

    /// Returns `true` if the plan calls for saving the given (stage, layer)
    /// pair. Whole-model stages ignore the layer parameter and the layer
    /// range — they are saved iff selected by the user.
    #[must_use]
    pub fn should_save(&self, stage: SaveTensorStage, layer: u32) -> bool {
        if !self.stages.contains(&stage) {
            return false;
        }
        if !stage.is_per_layer() {
            return true;
        }
        self.layer_range.contains(&layer)
    }

    /// Build the output path for a single saved tensor file. Mirrors
    /// [`super::save_tensor_paths::output_path`] but uses the plan's
    /// `output_dir`.
    ///
    /// For whole-model stages, pass [`WHOLE_MODEL_LAYER`] as `layer`.
    #[must_use]
    pub fn stage_path(&self, stage: SaveTensorStage, layer: u32) -> PathBuf {
        let effective_layer = if stage.is_per_layer() {
            layer
        } else {
            WHOLE_MODEL_LAYER
        };
        output_path(&self.output_dir, effective_layer, stage.canonical_name())
    }
}

/// Parse the `--save-tensor` argument string (either `all` or a
/// comma-list of stage names).
fn parse_stages_arg(s: &str) -> Result<Vec<SaveTensorStage>, StageParseError> {
    let trimmed = s.trim();
    if trimmed.eq_ignore_ascii_case("all") {
        return Ok(SaveTensorStage::ALL.to_vec());
    }
    parse_stage_list(trimmed)
}

/// Parse the `--save-tensor-layers` argument string (`START..END`).
fn parse_layer_range(s: &str) -> Result<Range<u32>, PlanParseError> {
    let trimmed = s.trim();
    let (start_str, end_str) =
        trimmed
            .split_once("..")
            .ok_or_else(|| PlanParseError::LayerRange {
                got: trimmed.to_string(),
                reason: "expected `START..END`".to_string(),
            })?;
    let start: u32 = start_str
        .trim()
        .parse()
        .map_err(|_| PlanParseError::LayerRange {
            got: trimmed.to_string(),
            reason: format!("START {start_str:?} is not a valid u32"),
        })?;
    let end: u32 = end_str
        .trim()
        .parse()
        .map_err(|_| PlanParseError::LayerRange {
            got: trimmed.to_string(),
            reason: format!("END {end_str:?} is not a valid u32"),
        })?;
    if end <= start {
        return Err(PlanParseError::LayerRange {
            got: trimmed.to_string(),
            reason: format!("END ({end}) must be > START ({start})"),
        });
    }
    Ok(start..end)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Provenance pin: PR-B is rev 1; if a future PR moves the plan
    /// constructor, this test should fail to force a memory update.
    #[test]
    fn provenance_pin_pr_b_rev1() {
        // Sentinel: signature shape that downstream PR-C / PR-D rely on.
        let p = SaveTensorPlan::from_cli("embedding", "0..1", PathBuf::from("/tmp/x")).unwrap();
        assert_eq!(p.stages.len(), 1);
        assert_eq!(p.layer_range, 0..1);
        assert_eq!(p.output_dir, PathBuf::from("/tmp/x"));
    }

    // ── Pass band ─────────────────────────────────────────────────────────

    #[test]
    fn realistic_healthy_three_stages_layer_zero() {
        let plan = SaveTensorPlan::from_cli(
            "embedding,qkv_matmul,attention",
            "0..1",
            PathBuf::from("trace_out"),
        )
        .expect("realistic args should parse");
        assert_eq!(plan.stages.len(), 3);
        assert_eq!(plan.stages[0], SaveTensorStage::Embedding);
        assert_eq!(plan.stages[1], SaveTensorStage::QkvMatmul);
        assert_eq!(plan.stages[2], SaveTensorStage::Attention);
        assert_eq!(plan.layer_range, 0..1);
    }

    #[test]
    fn all_keyword_expands_to_twenty_stages() {
        let plan = SaveTensorPlan::from_cli("all", "0..1", PathBuf::from("/tmp")).unwrap();
        assert_eq!(plan.stages.len(), 20);
        assert_eq!(plan.stages, SaveTensorStage::ALL.to_vec());
    }

    #[test]
    fn all_keyword_case_insensitive() {
        for variant in ["all", "ALL", "All", "aLL"] {
            let plan = SaveTensorPlan::from_cli(variant, "0..1", PathBuf::from("/tmp"))
                .expect("case variant should parse");
            assert_eq!(plan.stages.len(), 20);
        }
    }

    #[test]
    fn whitespace_in_stage_list_tolerated() {
        let plan =
            SaveTensorPlan::from_cli(" embedding , ffn_gate ", "0..1", PathBuf::from("/tmp"))
                .unwrap();
        assert_eq!(plan.stages.len(), 2);
    }

    #[test]
    fn whitespace_around_layer_range_tolerated() {
        let plan = SaveTensorPlan::from_cli("embedding", "  3..7 ", PathBuf::from("/tmp")).unwrap();
        assert_eq!(plan.layer_range, 3..7);
    }

    #[test]
    fn wide_layer_range_parses() {
        let plan = SaveTensorPlan::from_cli("ffn_gate", "0..32", PathBuf::from("/tmp")).unwrap();
        assert_eq!(plan.layer_range, 0..32);
    }

    // ── Fail band ─────────────────────────────────────────────────────────

    #[test]
    fn fail_unknown_stage_name() {
        let err = SaveTensorPlan::from_cli("not_a_stage", "0..1", PathBuf::from("/tmp"))
            .expect_err("typo should error");
        match err {
            PlanParseError::Stage(StageParseError::Unknown { got, .. }) => {
                assert_eq!(got, "not_a_stage");
            },
            _ => panic!("expected StageParseError::Unknown, got {err:?}"),
        }
    }

    #[test]
    fn fail_empty_token_in_stage_list() {
        let err = SaveTensorPlan::from_cli("embedding,,ffn_gate", "0..1", PathBuf::from("/tmp"))
            .expect_err("empty token should error");
        assert!(matches!(err, PlanParseError::Stage(StageParseError::Empty)));
    }

    #[test]
    fn fail_layer_range_missing_dotdot() {
        let err =
            SaveTensorPlan::from_cli("embedding", "0-3", PathBuf::from("/tmp")).expect_err("");
        match err {
            PlanParseError::LayerRange { got, reason } => {
                assert_eq!(got, "0-3");
                assert!(reason.contains("START..END"));
            },
            _ => panic!("expected LayerRange, got {err:?}"),
        }
    }

    #[test]
    fn fail_layer_range_negative_start() {
        // -1 is not a valid u32; we don't accept signed integers.
        let err =
            SaveTensorPlan::from_cli("embedding", "-1..3", PathBuf::from("/tmp")).expect_err("");
        assert!(matches!(err, PlanParseError::LayerRange { .. }));
    }

    #[test]
    fn fail_layer_range_end_le_start() {
        let err =
            SaveTensorPlan::from_cli("embedding", "5..5", PathBuf::from("/tmp")).expect_err("");
        match err {
            PlanParseError::LayerRange { reason, .. } => {
                assert!(reason.contains("END") && reason.contains("START"));
            },
            _ => panic!("expected LayerRange, got {err:?}"),
        }
    }

    #[test]
    fn fail_layer_range_end_lt_start() {
        let err =
            SaveTensorPlan::from_cli("embedding", "10..3", PathBuf::from("/tmp")).expect_err("");
        assert!(matches!(err, PlanParseError::LayerRange { .. }));
    }

    #[test]
    fn fail_layer_range_garbage_end() {
        let err =
            SaveTensorPlan::from_cli("embedding", "0..abc", PathBuf::from("/tmp")).expect_err("");
        match err {
            PlanParseError::LayerRange { reason, .. } => {
                assert!(reason.contains("END"));
            },
            _ => panic!("expected LayerRange, got {err:?}"),
        }
    }

    // ── should_save semantics ─────────────────────────────────────────────

    #[test]
    fn should_save_per_layer_in_range() {
        let plan = SaveTensorPlan::from_cli("ffn_gate", "0..3", PathBuf::from("/tmp")).unwrap();
        assert!(plan.should_save(SaveTensorStage::FfnGate, 0));
        assert!(plan.should_save(SaveTensorStage::FfnGate, 1));
        assert!(plan.should_save(SaveTensorStage::FfnGate, 2));
    }

    #[test]
    fn should_not_save_per_layer_outside_range() {
        let plan = SaveTensorPlan::from_cli("ffn_gate", "0..3", PathBuf::from("/tmp")).unwrap();
        assert!(!plan.should_save(SaveTensorStage::FfnGate, 3)); // END exclusive
        assert!(!plan.should_save(SaveTensorStage::FfnGate, 27));
    }

    #[test]
    fn should_not_save_unselected_stage() {
        let plan = SaveTensorPlan::from_cli("ffn_gate", "0..3", PathBuf::from("/tmp")).unwrap();
        assert!(!plan.should_save(SaveTensorStage::Attention, 0));
    }

    #[test]
    fn should_save_whole_model_stage_ignores_layer_range() {
        let plan = SaveTensorPlan::from_cli("lm_head", "5..7", PathBuf::from("/tmp")).unwrap();
        // lm_head is whole-model; layer range is irrelevant.
        assert!(plan.should_save(SaveTensorStage::LmHead, 0));
        assert!(plan.should_save(SaveTensorStage::LmHead, 99));
        assert!(plan.should_save(SaveTensorStage::LmHead, WHOLE_MODEL_LAYER));
    }

    #[test]
    fn should_save_default_range_zero_to_one_only_layer_zero() {
        // Default `0..1` (the clap default) saves layer 0 only.
        let plan = SaveTensorPlan::from_cli("embedding", "0..1", PathBuf::from("/tmp")).unwrap();
        assert!(plan.should_save(SaveTensorStage::Embedding, 0));
        assert!(!plan.should_save(SaveTensorStage::Embedding, 1));
    }

    // ── stage_path semantics ──────────────────────────────────────────────

    #[test]
    fn stage_path_per_layer_layer_zero() {
        let plan =
            SaveTensorPlan::from_cli("embedding", "0..1", PathBuf::from("trace_out")).unwrap();
        let p = plan.stage_path(SaveTensorStage::Embedding, 0);
        assert_eq!(p, PathBuf::from("trace_out/layer-0/embedding.bin"));
    }

    #[test]
    fn stage_path_per_layer_layer_three() {
        let plan =
            SaveTensorPlan::from_cli("ffn_gate", "0..32", PathBuf::from("trace_out")).unwrap();
        let p = plan.stage_path(SaveTensorStage::FfnGate, 3);
        assert_eq!(p, PathBuf::from("trace_out/layer-3/ffn_gate.bin"));
    }

    #[test]
    fn stage_path_whole_model_skips_layer_segment() {
        let plan = SaveTensorPlan::from_cli("lm_head", "0..1", PathBuf::from("trace_out")).unwrap();
        // Whole-model stage: even if caller passes layer=5, output ignores it.
        let p = plan.stage_path(SaveTensorStage::LmHead, 5);
        assert_eq!(p, PathBuf::from("trace_out/lm_head.bin"));
    }

    #[test]
    fn stage_path_whole_model_with_sentinel_layer() {
        let plan =
            SaveTensorPlan::from_cli("final_norm", "0..1", PathBuf::from("trace_out")).unwrap();
        let p = plan.stage_path(SaveTensorStage::FinalNorm, WHOLE_MODEL_LAYER);
        assert_eq!(p, PathBuf::from("trace_out/final_norm.bin"));
    }

    // ── Edge / mutation survey ────────────────────────────────────────────

    #[test]
    fn layer_output_alias_resolves_to_post_ffn_residual() {
        let plan = SaveTensorPlan::from_cli("layer_output", "0..1", PathBuf::from("/tmp")).unwrap();
        assert_eq!(plan.stages, vec![SaveTensorStage::PostFfnResidual]);
    }

    #[test]
    fn duplicate_stages_preserved_for_caller_dedup() {
        // The plan does NOT auto-dedupe — caller's responsibility if they
        // want unique writes. (Currently the comma parser preserves order;
        // this test pins that contract so dedup-on-construction would be a
        // breaking change.)
        let plan =
            SaveTensorPlan::from_cli("embedding,embedding", "0..1", PathBuf::from("/tmp")).unwrap();
        assert_eq!(plan.stages.len(), 2);
    }

    #[test]
    fn min_valid_range_one_layer() {
        let plan = SaveTensorPlan::from_cli("embedding", "0..1", PathBuf::from("/tmp")).unwrap();
        assert_eq!(plan.layer_range.end - plan.layer_range.start, 1);
    }

    #[test]
    fn high_layer_index_in_range() {
        // Some models have many layers; sanity-check large u32 values.
        let plan = SaveTensorPlan::from_cli("ffn_gate", "0..1000", PathBuf::from("/tmp")).unwrap();
        assert!(plan.should_save(SaveTensorStage::FfnGate, 999));
        assert!(!plan.should_save(SaveTensorStage::FfnGate, 1000));
    }

    #[test]
    fn plan_is_clone_and_eq() {
        // Plan must be Clone so it can be threaded through forward_traced
        // by value; PartialEq simplifies test assertions in PR-C/D.
        let a = SaveTensorPlan::from_cli("embedding", "0..1", PathBuf::from("/tmp")).unwrap();
        let b = a.clone();
        assert_eq!(a, b);
    }
}
