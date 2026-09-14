//! L0-1b per-op tap (#2971, PP-066): the CPU reference forward emits every
//! [`SaveTensorStage`] it computes, so `apr parity --per-op` can compare the
//! CPU and GPU trees stage by stage and name the FIRST op whose cosine falls.
//!
//! Design (why a thread-local and not a parameter): the reference path is
//! `OwnedQuantizedModel::forward_single_with_cache`, the exact function the
//! load-time gate and `apr parity` compare against. Threading a plan through
//! its signature would change a public API used across the crate and rename
//! a function the per-function complexity ratchet records; a struct field
//! would touch 61 constructors. The tap is armed on the calling thread by the
//! diagnostic command, read at each stage, and is `None` in every other
//! process — a single `RefCell` borrow per stage, no allocation.
//!
//! The gate bypass lives here too: `apr parity --per-op` must construct the
//! CUDA model even when the load-time gate would refuse it (the table IS the
//! verdict), and it must do so INTERNALLY — never through `SKIP_PARITY_GATE`,
//! which is an operator override and is printed as one.
use std::cell::{Cell, RefCell};

use super::GpuStageDumpConfig;
use crate::inference_trace::save_tensor_emit::write_stage_file;
use crate::inference_trace::save_tensor_plan::SaveTensorPlan;
use crate::inference_trace::save_tensor_stage::SaveTensorStage;

thread_local! {
    static PLAN: RefCell<Option<SaveTensorPlan>> = const { RefCell::new(None) };
    static GATE_BYPASS: Cell<bool> = const { Cell::new(false) };
    static GPU_DUMP: RefCell<Option<GpuStageDumpConfig>> = const { RefCell::new(None) };
}

/// Arm (or clear) the GPU-side stage dump on the calling thread: the executor's
/// per-phase dump points and [`GpuStageDumpConfig::from_env`] both read it, so
/// the whole-model SHIP-007 sites honour it without an env variable.
pub fn set_gpu_dump(cfg: Option<GpuStageDumpConfig>) {
    GPU_DUMP.with(|g| *g.borrow_mut() = cfg);
}

/// The armed GPU dump config, if any (a clone; the config is one path).
#[must_use]
pub fn gpu_dump() -> Option<GpuStageDumpConfig> {
    GPU_DUMP.with(|g| g.borrow().clone())
}

/// Arm (or clear, with `None`) the per-op tap on the calling thread.
pub fn set_plan(plan: Option<SaveTensorPlan>) {
    PLAN.with(|p| *p.borrow_mut() = plan);
}

/// Whether a plan is armed on the calling thread.
#[must_use]
pub fn is_armed() -> bool {
    PLAN.with(|p| p.borrow().is_some())
}

/// Emit `values` for `(stage, layer)` when the armed plan selects it. Non-fatal:
/// an I/O failure is printed and inference continues — the tap is diagnostic.
pub fn tap(stage: SaveTensorStage, layer: u32, values: &[f32]) {
    PLAN.with(|p| {
        let guard = p.borrow();
        let Some(plan) = guard.as_ref() else { return };
        if !plan.should_save(stage, layer) {
            return;
        }
        if let Err(e) = write_stage_file(&plan.output_dir, stage, layer, values) {
            eprintln!(
                "[per-op-tap] {}: layer {layer}: {e}",
                stage.canonical_name()
            );
        }
    });
}

/// Like [`tap`], but `values` is computed only when the plan selects the stage —
/// for stages the reference path never materialises (its fused kernels fold
/// the norm into the matmul), so the side computation costs nothing unarmed.
pub fn tap_with(stage: SaveTensorStage, layer: u32, values: impl FnOnce() -> Vec<f32>) {
    let selected = PLAN.with(|p| {
        p.borrow()
            .as_ref()
            .is_some_and(|plan| plan.should_save(stage, layer))
    });
    if selected {
        tap(stage, layer, &values());
    }
}

/// [`tap`] on the `Ok` side of a result, so a caller adds no branch of its own.
pub fn tap_ok<E>(stage: SaveTensorStage, layer: u32, r: &Result<Vec<f32>, E>) {
    if let Ok(v) = r {
        tap(stage, layer, v);
    }
}

/// [`tap`] of the norm a fused kernel folded away, computed only when the plan
/// selects the stage; slices in, no closure at the call site.
pub fn tap_norm(
    stage: SaveTensorStage,
    layer: u32,
    x: &[f32],
    weight: Option<&[f32]>,
    bias: Option<&[f32]>,
    eps: f32,
    rmsnorm: bool,
) {
    let selected = PLAN.with(|p| {
        p.borrow()
            .as_ref()
            .is_some_and(|plan| plan.should_save(stage, layer))
    });
    if selected {
        tap(stage, layer, &norm_for_tap(x, weight, bias, eps, rmsnorm));
    }
}

/// The norm a fused kernel would have produced: RMSNorm (`weight`) or LayerNorm
/// (`weight`, `bias`); `None` weight means the stage's input is passed through.
#[must_use]
pub fn norm_for_tap(
    x: &[f32],
    weight: Option<&[f32]>,
    bias: Option<&[f32]>,
    eps: f32,
    rmsnorm: bool,
) -> Vec<f32> {
    match weight {
        None => x.to_vec(),
        Some(w) if rmsnorm => crate::gguf::ops::rms_norm(x, w, eps),
        Some(w) => crate::gguf::ops::layer_norm(x, w, bias, eps),
    }
}

/// Arm the internal load-time-gate bypass for the calling thread (the diagnostic
/// command only). Recorded on the model as `skipped` with a basis naming this
/// path — printed, never silent.
pub fn arm_gate_bypass() {
    GATE_BYPASS.with(|g| g.set(true));
}

/// Whether the diagnostic gate bypass is armed on the calling thread.
#[must_use]
pub fn gate_bypass_armed() -> bool {
    GATE_BYPASS.with(Cell::get)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference_trace::save_tensor_compose::read_stage_file;

    #[test]
    fn unarmed_tap_writes_nothing_and_computes_nothing() {
        set_plan(None);
        let tmp = std::env::temp_dir().join(format!("per-op-tap-unarmed-{}", std::process::id()));
        tap(SaveTensorStage::FfnSwigl, 3, &[1.0, 2.0]);
        let mut computed = false;
        tap_with(SaveTensorStage::FfnNorm, 3, || {
            computed = true;
            vec![0.0]
        });
        assert!(!computed, "the side computation must not run unarmed");
        assert!(!tmp.exists());
        assert!(!is_armed());
    }

    #[test]
    fn armed_tap_writes_the_selected_stage_and_reads_back() {
        let tmp = std::env::temp_dir().join(format!("per-op-tap-armed-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&tmp);
        let plan = SaveTensorPlan::from_cli("ffn_swigl,lm_head", "0..4", tmp.clone())
            .expect("plan parses");
        set_plan(Some(plan));
        assert!(is_armed());
        tap(SaveTensorStage::FfnSwigl, 3, &[0.5, -1.5, 2.0]);
        tap(SaveTensorStage::FfnSwigl, 7, &[9.0]); // outside 0..4: not selected
        tap(SaveTensorStage::AttnNorm, 3, &[9.0]); // not in the stage list
        tap_with(SaveTensorStage::LmHead, 0, || vec![1.0, 2.0]);
        set_plan(None);
        let (hdr, vals) =
            read_stage_file(&tmp.join("layer-3").join("ffn_swigl.bin")).expect("written");
        assert_eq!(hdr.layer, 3);
        assert_eq!(vals, vec![0.5, -1.5, 2.0]);
        assert!(
            !tmp.join("layer-7").exists(),
            "layer 7 is outside the plan's range"
        );
        assert!(
            !tmp.join("layer-3").join("attn_norm.bin").exists(),
            "attn_norm was not selected"
        );
        assert!(
            tmp.join("lm_head.bin").exists(),
            "whole-model stages land at the root"
        );
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn norm_for_tap_passes_through_without_a_weight() {
        assert_eq!(
            norm_for_tap(&[1.0, 2.0], None, None, 1e-6, true),
            vec![1.0, 2.0]
        );
    }

    #[test]
    fn gate_bypass_is_off_by_default_and_thread_local() {
        assert!(!gate_bypass_armed());
        let other = std::thread::spawn(|| {
            arm_gate_bypass();
            gate_bypass_armed()
        })
        .join()
        .expect("thread");
        assert!(other);
        assert!(
            !gate_bypass_armed(),
            "arming on another thread must not leak here"
        );
    }
}
