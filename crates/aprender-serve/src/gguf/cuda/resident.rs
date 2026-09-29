//! #3715: what a CUDA model's load makes resident on the device, so a decision taken
//! before the upload (the QK-norm batched-prefill fit check) can see whether the
//! FP16 prefill weight cache and its workspace will fit beside the weights.

use crate::gguf::quantized::{OwnedQKVWeights, OwnedQuantizedLayer, OwnedQuantizedTensor};

/// The prefill weight cache holds the seven projection matrices of every layer
/// plus the LM head (the "197 matrices" a 28-layer model logs): the quantized
/// tensors load uploads. Returns `(quantized bytes, elements)` of those tensors;
/// the cache is `elements` × its bytes per element. Read from the tensors
/// themselves, so it cannot drift from the model the way a dims formula can.
pub(crate) fn projection_sizes<'a>(
    tensors: impl IntoIterator<Item = &'a OwnedQuantizedTensor>,
) -> (usize, usize) {
    tensors.into_iter().fold((0, 0), |(bytes, elems), t| {
        (bytes + t.data.len(), elems + t.in_dim * t.out_dim)
    })
}

/// The projection tensors of one layer, fused QKV or separate.
pub(crate) fn layer_projections(l: &OwnedQuantizedLayer) -> Vec<&OwnedQuantizedTensor> {
    let mut v = match &l.qkv_weight {
        OwnedQKVWeights::Fused(t) => vec![t],
        OwnedQKVWeights::Separate { q, k, v } => vec![q, k, v],
    };
    v.extend([&l.attn_output_weight, &l.ffn_up_weight, &l.ffn_down_weight]);
    v.extend(l.ffn_gate_weight.as_ref());
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(in_dim: usize, out_dim: usize, bytes: usize) -> OwnedQuantizedTensor {
        OwnedQuantizedTensor {
            data: vec![0; bytes],
            in_dim,
            out_dim,
            qtype: 12,
        }
    }

    /// A Qwen3 layer: separate Q/K/V, gated FFN. `bytes` goes on every tensor.
    fn layer(hidden: usize, kv: usize, inter: usize, bytes: usize) -> OwnedQuantizedLayer {
        OwnedQuantizedLayer {
            attn_norm_weight: Vec::new(),
            attn_norm_bias: None,
            qkv_weight: OwnedQKVWeights::Separate {
                q: t(hidden, hidden, bytes),
                k: t(hidden, kv, bytes),
                v: t(hidden, kv, bytes),
            },
            qkv_bias: None,
            attn_output_weight: t(hidden, hidden, bytes),
            attn_output_bias: None,
            ffn_up_weight: t(hidden, inter, bytes),
            ffn_up_bias: None,
            ffn_down_weight: t(inter, hidden, bytes),
            ffn_down_bias: None,
            ffn_gate_weight: Some(t(hidden, inter, bytes)),
            ffn_gate_bias: None,
            ffn_norm_weight: None,
            ffn_norm_bias: None,
            attn_q_norm_weight: None,
            attn_k_norm_weight: None,
            post_attn_norm_weight: None,
            post_ffw_norm_weight: None,
        }
    }

    /// FP16 cache bytes of a Qwen3 model with these dims, via the tensor walk.
    fn fp16_cache(layers: usize, hidden: usize, kv: usize, inter: usize) -> usize {
        let l = layer(hidden, kv, inter, 0);
        let head = t(hidden, 151_936, 0);
        let per_layer = projection_sizes(layer_projections(&l)).1;
        2 * (layers * per_layer + projection_sizes([&head]).1)
    }

    #[test]
    fn fp16_cache_estimate_matches_the_logged_qwen3_1_7b_cache() {
        // [PMAT-037] FP16 weight cache: 197 matrices cached (3281.5 MB), measured on lambda.
        let bytes = fp16_cache(28, 2048, 1024, 6144);
        let logged = 3281.5 * 1024.0 * 1024.0;
        assert!(
            (bytes as f64 - logged).abs() / logged < 0.02,
            "{bytes} vs {logged}"
        );
    }

    #[test]
    fn projection_walk_counts_every_tensor_once() {
        // Seven projections, distinct dims so a dropped or doubled tensor shows.
        let mut l = layer(2, 3, 5, 7);
        assert_eq!(layer_projections(&l).len(), 7);
        let (bytes, elems) = projection_sizes(layer_projections(&l));
        // q 2x2, k 2x3, v 2x3, o 2x2, up 2x5, down 5x2, gate 2x5
        assert_eq!((bytes, elems), (7 * 7, 4 + 6 + 6 + 4 + 10 + 10 + 10));
        // A fused QKV is one tensor; an ungated FFN has no gate.
        l.qkv_weight = OwnedQKVWeights::Fused(t(2, 8, 7));
        l.ffn_gate_weight = None;
        assert_eq!(layer_projections(&l).len(), 4);
        assert_eq!(
            projection_sizes(layer_projections(&l)),
            (4 * 7, 16 + 4 + 10 + 10)
        );
    }
}
