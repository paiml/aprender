//! Moving the instruct pipeline's LoRA adapters between its LoRA layers and its CUDA NF4
//! blocks by target (FALSIFY-LORA_TARGET_SELECTION_V1_007, contract equation `device_sync`).
//!
//! Slot `|T|·layer + pos_T(target)` holds the adapter of `target` in `layer` (`target_slots`).
//! The upload lists every selected adapter of a layer, and the sync writes every adapter
//! downloaded from a block back to its own slot, whatever the targets.

use crate::lora::{LoRALayer, LoraTarget, LoraTargets};

/// The A and B weights of LoRA slot `idx` in the layout the NF4 block computes with, `(Aᵀ, Bᵀ)`,
/// or `None` past the last slot. The block runs `(x·A)·B`, so the PEFT layout copied raw would
/// be a different adapter (FALSIFY-CUDA-NF4-TRAIN-LOSS-PARITY-003).
pub(super) fn lora_slot(lora_layers: &[LoRALayer], idx: usize) -> Option<(Vec<f32>, Vec<f32>)> {
    lora_layers.get(idx).map(LoRALayer::device_layout)
}

/// Every selected adapter of `layer` in slot order, as `(target, Aᵀ, Bᵀ)` in the device layout.
/// A slot past the last LoRA layer is left out.
pub(super) fn layer_adapters(
    lora_layers: &[LoRALayer],
    targets: &LoraTargets,
    layer: usize,
) -> Vec<(LoraTarget, Vec<f32>, Vec<f32>)> {
    targets
        .as_slice()
        .iter()
        .filter_map(|&target| {
            let (a, b) = lora_slot(lora_layers, targets.slot(layer, target)?)?;
            Some((target, a, b))
        })
        .collect()
}

/// Write the adapters downloaded from `layer`'s block, `(target, Aᵀ, σ·Bᵀ)` in slot order, into
/// their own slots, with B multiplied by `inv_scale` = 1/σ.
///
/// Every adapter is checked before any is written, so a refusal changes no slot.
///
/// # Errors
/// A target outside `targets`, a slot past the last LoRA layer, or an A or B whose length is
/// not its slot's, named in the message.
pub(super) fn place_layer_adapters(
    lora_layers: &mut [LoRALayer],
    targets: &LoraTargets,
    layer: usize,
    adapters: &[(LoraTarget, Vec<f32>, Vec<f32>)],
    inv_scale: f32,
) -> Result<(), String> {
    let mut slots = Vec::with_capacity(adapters.len());
    for (target, a, b) in adapters {
        let slot = targets.slot(layer, *target).ok_or_else(|| {
            format!("layer {layer}: block adapter {target} is not a selected target ({targets})")
        })?;
        let lora = lora_layers.get(slot).ok_or_else(|| {
            format!("layer {layer}: {target} slot {slot} is past the last LoRA layer")
        })?;
        let (want_a, want_b) = (lora.rank() * lora.d_in(), lora.rank() * lora.d_out());
        if a.len() != want_a || b.len() != want_b {
            return Err(format!(
                "layer {layer}: {target} A has {} and B {} values, slot {slot} takes {want_a} and {want_b}",
                a.len(),
                b.len()
            ));
        }
        slots.push(slot);
    }
    for (slot, (_, a, b)) in slots.into_iter().zip(adapters) {
        let b_unscaled: Vec<f32> = b.iter().map(|&v| v * inv_scale).collect();
        lora_layers[slot].set_from_device_layout(a, &b_unscaled);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tensor;

    const RANK: usize = 2;

    /// `(d_out, d_in)` of each target, as `target_slots` gives them, for hidden 4, q 6, kv 2
    /// and intermediate 5: q, kv, h and i all differ, so only same-shape targets share lengths.
    fn dims(target: LoraTarget) -> (usize, usize) {
        match target {
            LoraTarget::Q => (6, 4),
            LoraTarget::K | LoraTarget::V => (2, 4),
            LoraTarget::O => (4, 6),
            LoraTarget::Gate | LoraTarget::Up => (5, 4),
            LoraTarget::Down => (4, 5),
        }
    }

    /// LoRA layers for `layers` layers under `targets`, in slot order. Slot `s` holds
    /// `fill(s, i)` at entry `i` of A and its negation in B, so every slot is distinct.
    fn lora_layers(
        targets: &LoraTargets,
        layers: usize,
        fill: fn(usize, usize) -> f32,
    ) -> Vec<LoRALayer> {
        let mut out = Vec::new();
        for _ in 0..layers {
            for &target in targets.as_slice() {
                let s = out.len();
                let (d_out, d_in) = dims(target);
                let mut lora =
                    LoRALayer::new(Tensor::zeros(d_out * d_in, false), d_out, d_in, RANK, 4.0);
                *lora.lora_a_mut() =
                    Tensor::from_vec((0..RANK * d_in).map(|i| fill(s, i)).collect(), true);
                *lora.lora_b_mut() =
                    Tensor::from_vec((0..d_out * RANK).map(|i| -fill(s, i)).collect(), true);
                out.push(lora);
            }
        }
        out
    }

    fn distinct(s: usize, i: usize) -> f32 {
        (1000 * s + i + 1) as f32
    }

    fn zero(_: usize, _: usize) -> f32 {
        0.0
    }

    /// The (A, B) of every slot, in the PEFT layout `LoRALayer` keeps.
    fn weights(lora_layers: &[LoRALayer]) -> Vec<(Vec<f32>, Vec<f32>)> {
        lora_layers
            .iter()
            .map(|l| (l.lora_a().data().to_vec(), l.lora_b().data().to_vec()))
            .collect()
    }

    fn all_linear() -> LoraTargets {
        LoraTargets::parse(&["all_linear"]).expect("all_linear parses")
    }

    #[test]
    fn falsify_lora_target_selection_v1_007_layer_adapters_are_every_target_in_slot_order() {
        let targets = all_linear();
        let layers = lora_layers(&targets, 2, distinct);
        let got = layer_adapters(&layers, &targets, 1);
        let got_targets: Vec<LoraTarget> = got.iter().map(|(t, _, _)| *t).collect();
        assert_eq!(got_targets, targets.as_slice(), "layer 1: the seven targets in slot order");
        for (p, (target, a, b)) in got.iter().enumerate() {
            let (want_a, want_b) = layers[7 + p].device_layout();
            assert_eq!((a, b), (&want_a, &want_b), "layer 1 {target} is not slot {}", 7 + p);
        }
    }

    #[test]
    fn falsify_lora_target_selection_v1_007_default_targets_are_the_q_v_slots() {
        let targets = LoraTargets::default();
        let layers = lora_layers(&targets, 3, distinct);
        for l in 0..3 {
            let (q, v) = (lora_slot(&layers, 2 * l), lora_slot(&layers, 2 * l + 1));
            let (q, v) = (q.expect("q slot"), v.expect("v slot"));
            let want = vec![(LoraTarget::Q, q.0, q.1), (LoraTarget::V, v.0, v.1)];
            assert_eq!(
                layer_adapters(&layers, &targets, l),
                want,
                "layer {l}: slots 2l and 2l + 1"
            );
        }
    }

    #[test]
    fn falsify_lora_target_selection_v1_007_place_gives_back_every_slot() {
        for targets in [LoraTargets::default(), all_linear()] {
            let src = lora_layers(&targets, 2, distinct);
            let mut dst = lora_layers(&targets, 2, zero);
            for l in 0..2 {
                // As the block downloads them: B scaled by σ = 2, a power of two, so exact.
                let downloaded: Vec<_> = layer_adapters(&src, &targets, l)
                    .into_iter()
                    .map(|(t, a, b)| (t, a, b.iter().map(|&x| 2.0 * x).collect()))
                    .collect();
                place_layer_adapters(&mut dst, &targets, l, &downloaded, 0.5)
                    .unwrap_or_else(|e| panic!("{targets}: layer {l} refused: {e}"));
            }
            assert_eq!(weights(&dst), weights(&src), "{targets}: every slot back bit for bit");
        }
    }

    #[test]
    fn falsify_lora_target_selection_v1_007_place_refuses_before_writing() {
        let targets = LoraTargets::default();
        let src = lora_layers(&targets, 2, distinct);
        let mut dst = lora_layers(&targets, 2, zero);
        let before = weights(&dst);
        let adapters = layer_adapters(&src, &targets, 1);
        let (_, a_v, b_v) = adapters[1].clone();

        // k_proj has v_proj's shape, so only the target check can refuse it.
        let k = [(LoraTarget::K, a_v.clone(), b_v.clone())];
        let err = place_layer_adapters(&mut dst, &targets, 1, &k, 1.0).expect_err("k_proj refused");
        assert!(err.contains("k_proj"), "the error names k_proj: {err}");
        assert_eq!(weights(&dst), before, "a refused k_proj changes no slot");

        // A valid q_proj pair first, then a v_proj A one element short.
        let short = [adapters[0].clone(), (LoraTarget::V, a_v[1..].to_vec(), b_v)];
        let err =
            place_layer_adapters(&mut dst, &targets, 1, &short, 1.0).expect_err("short A refused");
        assert!(err.contains("v_proj"), "the error names v_proj: {err}");
        assert_eq!(weights(&dst), before, "a refused layer leaves its q_proj slot as it was");
    }
}
