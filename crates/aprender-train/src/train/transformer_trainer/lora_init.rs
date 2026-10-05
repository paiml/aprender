//! Initial LoRA adapters of the CUDA trainer's NF4 blocks, by target (R15a C5b,
//! contract `lora-target-selection-v1` equation `nf4_init_targets`,
//! FALSIFY-LORA_TARGET_SELECTION_V1_011).
//!
//! Each block gets one adapter per selected target in slot order, in the device
//! layout (A [d_in, r], B [r, d_out]). A is a small sinusoid whose phase depends on
//! the layer and the target, and B is zero, so every adapter starts as a no-op. q_proj
//! and v_proj keep the phases the NF4 init used before C5b.

use super::lora_checkpoint::DeviceAdapters;
use super::trainer::trainer_targets;
use crate::lora::LoraTarget;
use crate::transformer::TransformerConfig;

/// The targets the CUDA trainer's NF4 blocks hold: the CPU trainer's targets
/// (`trainer_targets`), or none if no name is one of the seven projections.
pub(crate) fn nf4_targets(modules: Option<&[String]>) -> Vec<LoraTarget> {
    trainer_targets(modules).map(|t| t.as_slice().to_vec()).unwrap_or_default()
}

/// `(d_out, d_in)` of each target of [`nf4_targets`]`(modules)`, in slot order: the
/// shapes the trainer pre-warms the LoRA backward GEMMs for (R15a C6,
/// FALSIFY-LORA_TARGET_SELECTION_V1_012).
pub(crate) fn lora_prewarm_dims(
    config: &TransformerConfig,
    modules: Option<&[String]>,
) -> Vec<(u32, u32)> {
    nf4_targets(modules)
        .iter()
        .map(|t| {
            let (d_out, d_in) = t.dims(config);
            (d_out as u32, d_in as u32)
        })
        .collect()
}

/// `(j + φ(layer, target))·0.1`, the argument of A's sinusoid at index `j`. q_proj and
/// v_proj add in the order the pre-C5b init did, so their A is the same bit for bit.
fn init_arg(j: usize, layer: usize, target: LoraTarget) -> f32 {
    let (j, l) = (j as f32, layer as f32);
    match target {
        LoraTarget::Q => (j + l * 1000.0) * 0.1,
        LoraTarget::V => (j + l * 2000.0 + 500.0) * 0.1,
        t => (j + l * 1000.0 + 100_000.0 * t as usize as f32 + 333.0) * 0.1,
    }
}

/// Block `layer`'s initial adapters for `targets`, in that order: A_t[j] =
/// sin(init_arg)·0.01 of length d_in·r and B_t = 0 of length r·d_out.
pub(crate) fn nf4_init_adapters(
    config: &TransformerConfig,
    layer: usize,
    targets: &[LoraTarget],
    rank: usize,
) -> DeviceAdapters {
    targets
        .iter()
        .map(|&t| {
            let (d_out, d_in) = t.dims(config);
            let a = (0..d_in * rank).map(|j| init_arg(j, layer, t).sin() * 0.01).collect();
            (t, a, vec![0.0f32; rank * d_out])
        })
        .collect()
}

/// The adapter of `target` among `adapters`, as the (A, B) pair
/// `CudaNf4TransformerBlock::new` takes.
pub(crate) fn adapter_pair(
    adapters: &DeviceAdapters,
    target: LoraTarget,
) -> Option<(&[f32], &[f32])> {
    adapters.iter().find(|(t, _, _)| *t == target).map(|(_, a, b)| (a.as_slice(), b.as_slice()))
}

/// The adapters `new` does not take (every target but q_proj and v_proj), which the
/// block gets through `add_lora_adapter`, in slot order.
pub(crate) fn added_adapters(
    adapters: &DeviceAdapters,
) -> impl Iterator<Item = &(LoraTarget, Vec<f32>, Vec<f32>)> {
    adapters.iter().filter(|(t, _, _)| !matches!(t, LoraTarget::Q | LoraTarget::V))
}
