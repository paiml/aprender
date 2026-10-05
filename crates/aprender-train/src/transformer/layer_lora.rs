//! The LoRA adapters of one decoder layer, by target, for the CPU forward
//! (contract `lora-target-selection-v1`, cpu_forward,
//! FALSIFY-LORA_TARGET_SELECTION_V1_008).

use crate::lora::{LoRALayer, LoraTarget, LoraTargets};
use crate::Tensor;

/// One adapter's delta `scale·(x·Aᵀ)·Bᵀ`, with A `(rank, d_in)` and B `(d_out, rank)`.
#[derive(Clone, Copy)]
pub(crate) struct LoraDelta<'a> {
    pub(crate) a: &'a Tensor,
    pub(crate) b: &'a Tensor,
    pub(crate) rank: usize,
    pub(crate) scale: f32,
}

impl<'a> LoraDelta<'a> {
    /// The delta of `layer`, with its own rank and scale.
    pub(crate) fn of(layer: &'a LoRALayer) -> Self {
        Self { a: layer.lora_a(), b: layer.lora_b(), rank: layer.rank(), scale: layer.scale() }
    }
}

/// `base + scale·(x·Aᵀ)·Bᵀ` with `delta`, or `base` without one.
///
/// KAIZEN-011: `x·Aᵀ` and `mid·Bᵀ` are `matmul_nt` on the adapter's own A and
/// B, so the gradient reaches the trainable tensors, in the order the q_proj
/// and v_proj adapters have always used.
pub(crate) fn add_lora(
    delta: Option<&LoraDelta<'_>>,
    base: Tensor,
    x: &Tensor,
    seq_len: usize,
    d_in: usize,
    d_out: usize,
) -> Tensor {
    let Some(d) = delta else {
        return base;
    };
    let mid = crate::autograd::matmul_nt(x, d.a, seq_len, d_in, d.rank);
    let lora = crate::autograd::matmul_nt(&mid, d.b, seq_len, d.rank, d_out);
    crate::autograd::add_scaled(&base, &lora, d.scale)
}

/// The adapters of one layer, at most one per projection.
#[derive(Clone, Copy, Default)]
pub(crate) struct LayerLora<'a> {
    pub(crate) q: Option<LoraDelta<'a>>,
    pub(crate) k: Option<LoraDelta<'a>>,
    pub(crate) v: Option<LoraDelta<'a>>,
    pub(crate) o: Option<LoraDelta<'a>>,
    pub(crate) gate: Option<LoraDelta<'a>>,
    pub(crate) up: Option<LoraDelta<'a>>,
    pub(crate) down: Option<LoraDelta<'a>>,
}

impl<'a> LayerLora<'a> {
    /// Layer `layer`'s adapters from `lora_layers` laid out by `targets`: the
    /// adapter for target t is slot `|T|·layer + pos_T(t)`. None when the
    /// layer's slots run past `lora_layers`, so that layer runs without LoRA.
    pub(crate) fn from_slots(
        lora_layers: &'a [LoRALayer],
        targets: &LoraTargets,
        layer: usize,
    ) -> Option<Self> {
        if targets.per_layer() * (layer + 1) > lora_layers.len() {
            return None;
        }
        let mut out = Self::default();
        for &target in targets.as_slice() {
            let slot = targets.slot(layer, target)?;
            *out.get_mut(target) = Some(LoraDelta::of(lora_layers.get(slot)?));
        }
        Some(out)
    }

    fn get_mut(&mut self, target: LoraTarget) -> &mut Option<LoraDelta<'a>> {
        match target {
            LoraTarget::Q => &mut self.q,
            LoraTarget::K => &mut self.k,
            LoraTarget::V => &mut self.v,
            LoraTarget::O => &mut self.o,
            LoraTarget::Gate => &mut self.gate,
            LoraTarget::Up => &mut self.up,
            LoraTarget::Down => &mut self.down,
        }
    }
}
