//! Names and layouts of the CUDA trainer's LoRA checkpoint, by target (R15a C5a,
//! contract `lora-target-selection-v1` equation `checkpoint_names`,
//! FALSIFY-LORA_TARGET_SELECTION_V1_010).
//!
//! A block holds its adapters on the device as (target, A [d_in, r], σB [r, d_out]),
//! σ = alpha/rank. The APR checkpoint stores them in that layout under
//! `lora.{layer}.{module}.lora_{a,b}`; the PEFT adapter stores A [r, d_in] and unscaled
//! B [d_out, r] under `model.layers.{layer}.{self_attn|mlp}.{module}`.

use std::collections::BTreeSet;

use aprender::serialization::apr::{AprReader, AprWriter};

use crate::lora::{LoRALayer, LoraTarget};
use crate::transformer::TransformerConfig;

/// One block's adapters as the device holds them: (target, A, σB).
pub(crate) type DeviceAdapters = Vec<(LoraTarget, Vec<f32>, Vec<f32>)>;

/// The APR tensor names of `target`'s A and B in block `layer`.
pub(crate) fn apr_tensor_names(layer: usize, target: LoraTarget) -> (String, String) {
    target.apr_tensor_names(layer)
}

/// The PEFT module path of `target` in block `layer`, as `TransformerTrainer::save_lora_adapter`
/// names it.
pub(crate) fn peft_module_path(layer: usize, target: LoraTarget) -> String {
    let group = match target {
        LoraTarget::Q | LoraTarget::K | LoraTarget::V | LoraTarget::O => "self_attn",
        LoraTarget::Gate | LoraTarget::Up | LoraTarget::Down => "mlp",
    };
    format!("model.layers.{layer}.{group}.{}", target.module_name())
}

/// Write block `layer`'s adapters to an APR checkpoint as flat tensors. An adapter with an
/// empty A is skipped.
pub(crate) fn write_apr_adapters(writer: &mut AprWriter, layer: usize, adapters: &DeviceAdapters) {
    for (target, a, sigma_b) in adapters {
        if a.is_empty() {
            continue;
        }
        let (name_a, name_b) = apr_tensor_names(layer, *target);
        writer.add_tensor_f32(name_a, vec![a.len()], a);
        writer.add_tensor_f32(name_b, vec![sigma_b.len()], sigma_b);
    }
}

/// Read back block `layer`'s adapters for the targets it holds, in `targets`' order. A target
/// is returned only when the file holds both its A and its B.
pub(crate) fn read_apr_adapters(
    reader: &AprReader,
    layer: usize,
    targets: &[LoraTarget],
) -> DeviceAdapters {
    targets
        .iter()
        .filter_map(|&target| {
            let (name_a, name_b) = apr_tensor_names(layer, target);
            let a = reader.read_tensor_f32(&name_a).ok().filter(|a| !a.is_empty())?;
            let b = reader.read_tensor_f32(&name_b).ok().filter(|b| !b.is_empty())?;
            Some((target, a, b))
        })
        .collect()
}

/// Convert one adapter from the device layout to PEFT's: A [d_in, r] → [r, d_in], and
/// σB [r, d_out] → B [d_out, r] divided by σ (by 1 when σ is about 0).
pub(crate) fn device_to_peft(
    a: &[f32],
    sigma_b: &[f32],
    (d_out, d_in): (usize, usize),
    rank: usize,
    scale: f32,
) -> (Vec<f32>, Vec<f32>) {
    let mut a_t = vec![0.0f32; rank * d_in];
    for r in 0..d_in {
        for c in 0..rank {
            a_t[c * d_in + r] = a[r * rank + c];
        }
    }
    let inv_scale = if scale.abs() > 1e-10 { 1.0 / scale } else { 1.0 };
    let mut b_t = vec![0.0f32; d_out * rank];
    for r in 0..rank {
        for c in 0..d_out {
            b_t[c * rank + r] = sigma_b[r * d_out + c] * inv_scale;
        }
    }
    (a_t, b_t)
}

/// The PEFT adapter for one device adapter: its module path and a `LoRALayer` holding the
/// converted A and B with the target's (d_out, d_in).
pub(crate) fn peft_adapter(
    layer: usize,
    (target, a, sigma_b): (LoraTarget, &[f32], &[f32]),
    config: &TransformerConfig,
    rank: usize,
    alpha: f32,
) -> (String, LoRALayer) {
    let (d_out, d_in) = target.dims(config);
    let (a_t, b_t) = device_to_peft(a, sigma_b, (d_out, d_in), rank, alpha / rank as f32);
    let base_weight = crate::autograd::Tensor::zeros(d_out * d_in, false);
    let mut lora = LoRALayer::new(base_weight, d_out, d_in, rank, alpha);
    lora.lora_a_mut().data_mut().assign(&ndarray::Array1::from(a_t));
    lora.lora_b_mut().data_mut().assign(&ndarray::Array1::from(b_t));
    (peft_module_path(layer, target), lora)
}

/// The module names of every target any block holds, in slot order (`LoraTarget::ALL`).
pub(crate) fn held_module_names<'a>(
    blocks: impl IntoIterator<Item = &'a DeviceAdapters>,
) -> Vec<&'static str> {
    let held: BTreeSet<LoraTarget> =
        blocks.into_iter().flat_map(|adapters| adapters.iter().map(|(t, _, _)| *t)).collect();
    held.into_iter().map(LoraTarget::module_name).collect()
}
