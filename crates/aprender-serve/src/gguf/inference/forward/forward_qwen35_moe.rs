//! Qwen3.5-MoE (`qwen35moe`, #4665): the hybrid Gated `DeltaNet` / attention stack of
//! [`super::Qwen35Model`] with a mixture-of-experts FFN in every layer.
//!
//! The FFN is llama.cpp `qwen35moe.cpp` `build_layer_ffn`:
//!
//! ```text
//! out = Σ_{e ∈ top-k(softmax(R·x))} p̂_e · down_e(silu(gate_e·x) ⊙ up_e·x)      routed
//!     + sigmoid(s·x) · down_sh(silu(gate_sh·x) ⊙ up_sh·x)                       shared
//! ```
//!
//! with `p̂` the top-k probabilities renormalized to sum to 1 — the same rule as the qwen3moe
//! forward, so both call the ONE [`route_top_k`]. The shared expert is an ordinary dense
//! SwiGLU, so a layer's `ffn_gate`/`ffn_up`/`ffn_down` carry it and the dense code computes
//! it unchanged; [`Qwen35MoeFfn::combine_into`] then scales that output and adds the routed
//! sum. A dense Qwen3.5 file has no [`Qwen35MoeFfn`] and runs exactly as before.

use super::Qwen35Model;
use crate::error::{RealizarError, Result};
use crate::gguf::quantized::{OwnedQuantizedTensor, QuantizedTensorRef};
use crate::gguf::qwen35_load::Qwen35MoeRefs;
use crate::gguf::qwen3_moe_load::route_top_k;
use crate::gguf::GGUFModel;

/// The routed experts and shared-expert gate of one Qwen3.5-MoE layer, owned.
pub(crate) struct Qwen35MoeFfn {
    /// `[num_experts × hidden]` row-major F32 router.
    pub(crate) router: Vec<f32>,
    /// `[hidden]` F32 shared-expert gate.
    pub(crate) shared_gate: Vec<f32>,
    /// One `hidden → expert_dim` gate projection per expert.
    pub(crate) gate_exps: Vec<OwnedQuantizedTensor>,
    /// One `hidden → expert_dim` up projection per expert.
    pub(crate) up_exps: Vec<OwnedQuantizedTensor>,
    /// One `expert_dim → hidden` down projection per expert.
    pub(crate) down_exps: Vec<OwnedQuantizedTensor>,
    /// Experts summed per token (`{arch}.expert_used_count`).
    pub(crate) top_k: usize,
}

/// The MoE shape a Qwen3.5-MoE file records: expert count and top-k from metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Qwen35MoeShape {
    pub(crate) num_experts: usize,
    pub(crate) top_k: usize,
}

impl Qwen35MoeShape {
    /// Read `{arch}.expert_count` / `{arch}.expert_used_count`. Fails closed: a guessed
    /// expert count would slice every stacked tensor at the wrong stride.
    pub(crate) fn from_model(model: &GGUFModel) -> Result<Self> {
        let missing = |key: &str| RealizarError::FormatError {
            reason: format!(
                "qwen35moe: {}.{key} is missing from the GGUF metadata; refusing to guess the \
                 expert layout",
                model.architecture().unwrap_or("<no architecture>")
            ),
        };
        let num_experts = model
            .expert_count()
            .ok_or_else(|| missing("expert_count"))?;
        let top_k = model
            .expert_used_count()
            .ok_or_else(|| missing("expert_used_count"))?;
        if num_experts == 0 || top_k == 0 || top_k > num_experts {
            return Err(RealizarError::FormatError {
                reason: format!(
                    "qwen35moe: expert_used_count {top_k} must lie in 1..={num_experts} \
                     (expert_count)"
                ),
            });
        }
        Ok(Self { num_experts, top_k })
    }
}

/// Split a stacked `[E × out × in]` expert tensor into `E` owned `in → out` tensors.
/// Expert `e` is the contiguous `e`-th of the bytes — each expert is whole rows of whole
/// quant blocks, which the size check proves before anything is sliced.
fn split_experts(
    name: &str,
    stacked: &QuantizedTensorRef,
    data: &[u8],
    num_experts: usize,
    in_dim: usize,
) -> Result<(Vec<OwnedQuantizedTensor>, usize)> {
    let per_expert_elems = stacked.num_elements / num_experts;
    let out_dim = per_expert_elems / in_dim;
    let bytes = data
        .get(stacked.offset..stacked.offset + stacked.byte_size)
        .ok_or_else(|| RealizarError::FormatError {
            reason: format!("qwen35moe: '{name}' lies outside the file"),
        })?;
    if per_expert_elems * num_experts != stacked.num_elements
        || out_dim * in_dim != per_expert_elems
        || stacked.byte_size % num_experts != 0
    {
        return Err(RealizarError::FormatError {
            reason: format!(
                "qwen35moe: '{name}' has {} elements in {} bytes, which is not {num_experts} \
                 experts of whole {in_dim}-wide rows",
                stacked.num_elements, stacked.byte_size
            ),
        });
    }
    let stride = stacked.byte_size / num_experts;
    let experts = bytes
        .chunks_exact(stride)
        .map(|chunk| OwnedQuantizedTensor {
            data: chunk.to_vec(),
            in_dim,
            out_dim,
            qtype: stacked.qtype,
        })
        .collect();
    Ok((experts, out_dim))
}

impl Qwen35MoeFfn {
    /// Own one layer's router, shared gate and experts.
    ///
    /// # Errors
    /// A router or shared gate that is not F32 of the expected length, or an expert stack
    /// whose size is not `num_experts` whole experts of matching widths.
    pub(crate) fn own(
        refs: &Qwen35MoeRefs,
        data: &[u8],
        hidden_dim: usize,
        shape: Qwen35MoeShape,
    ) -> Result<Self> {
        let router = super::load_f32_vec(&refs.router, data)?;
        let shared_gate = super::load_f32_vec(&refs.shared_gate, data)?;
        if router.len() != shape.num_experts * hidden_dim || shared_gate.len() != hidden_dim {
            return Err(RealizarError::FormatError {
                reason: format!(
                    "qwen35moe: router has {} values (want {} experts × {hidden_dim}) and the \
                     shared-expert gate {} (want {hidden_dim})",
                    router.len(),
                    shape.num_experts,
                    shared_gate.len()
                ),
            });
        }
        let n = shape.num_experts;
        let (gate_exps, gate_dim) =
            split_experts("ffn_gate_exps", &refs.gate_exps, data, n, hidden_dim)?;
        let (up_exps, up_dim) = split_experts("ffn_up_exps", &refs.up_exps, data, n, hidden_dim)?;
        let expert_dim = refs.down_exps.num_elements / (n * hidden_dim).max(1);
        let (down_exps, down_out) =
            split_experts("ffn_down_exps", &refs.down_exps, data, n, expert_dim)?;
        if gate_dim != up_dim || up_dim != expert_dim || down_out != hidden_dim {
            return Err(RealizarError::FormatError {
                reason: format!(
                    "qwen35moe: expert widths disagree — gate {gate_dim}, up {up_dim}, down \
                     {expert_dim}→{down_out} (hidden {hidden_dim})"
                ),
            });
        }
        Ok(Self {
            router,
            shared_gate,
            gate_exps,
            up_exps,
            down_exps,
            top_k: shape.top_k,
        })
    }

    /// The router logits for one normed token `x`.
    pub(crate) fn router_logits(&self, x: &[f32]) -> Vec<f32> {
        self.router
            .chunks_exact(x.len())
            .map(|row| row.iter().zip(x).map(|(w, v)| w * v).sum())
            .collect()
    }

    /// `sigmoid(shared_gate · x)`.
    pub(crate) fn shared_scale(&self, x: &[f32]) -> f32 {
        let g: f32 = self.shared_gate.iter().zip(x).map(|(w, v)| w * v).sum();
        1.0 / (1.0 + (-g).exp())
    }
}

impl Qwen35Model<'_> {
    /// Turn the dense (shared-expert) FFN output `ffn_out` of one token into the full MoE
    /// output, in place: scale it by the shared gate, then add every routed expert's
    /// SwiGLU output weighted by its renormalized router probability. `x` is the
    /// post-attention-normed token both halves read.
    pub(crate) fn moe_combine_into(
        &self,
        moe: &Qwen35MoeFfn,
        x: &[f32],
        ffn_out: &mut [f32],
    ) -> Result<()> {
        let scale = moe.shared_scale(x);
        for v in ffn_out.iter_mut() {
            *v *= scale;
        }
        for (e, weight) in route_top_k(&moe.router_logits(x), moe.top_k) {
            let gate_w = &moe.gate_exps[e];
            let mut gate = vec![0.0; gate_w.out_dim];
            self.base.fused_matmul_into(x, gate_w, &mut gate)?;
            let mut up = vec![0.0; moe.up_exps[e].out_dim];
            self.base.fused_matmul_into(x, &moe.up_exps[e], &mut up)?;
            for (u, &g) in up.iter_mut().zip(&gate) {
                *u *= g / (1.0 + (-g).exp());
            }
            let mut down = vec![0.0; moe.down_exps[e].out_dim];
            self.base
                .fused_matmul_into(&up, &moe.down_exps[e], &mut down)?;
            for (o, d) in ffn_out.iter_mut().zip(&down) {
                *o += weight * d;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "forward_qwen35_moe_tests.rs"]
mod tests;
