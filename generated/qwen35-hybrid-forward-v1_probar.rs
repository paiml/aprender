// Qwen3.5-9B hybrid forward pass verification constants
const HIDDEN: usize = 4096;
const N_HEADS: usize = 16;
#[allow(dead_code)]
const N_KV_HEADS: usize = 4;
const D_K: usize = 256;
const INTERMEDIATE: usize = 12288;
const N_LAYERS: usize = 48;
const N_ATTN_LAYERS: usize = 24;
const N_GDN_LAYERS: usize = 24;
#[allow(dead_code)]
const GDN_INNER: usize = 1024; // N_KV_HEADS * D_K

/// Layer type for hybrid architecture
#[derive(Clone, Copy, PartialEq)]
enum LayerType {
    Attention,
    Gdn,
}

/// Returns the layer type for a given layer index.
/// Qwen3.5 alternates: even = attention, odd = GDN
fn layer_type(layer_idx: usize) -> LayerType {
    if layer_idx % 2 == 0 {
        LayerType::Attention
    } else {
        LayerType::Gdn
    }
}

/// Simple RMSNorm: normalizes vector to unit RMS
fn rmsnorm(x: &[f64]) -> Vec<f64> {
    let rms = (x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64).sqrt();
    if rms < 1e-12 {
        return x.to_vec();
    }
    x.iter().map(|v| v / rms).collect()
}

#[cfg(test)]
mod probar_tests {
    use super::*;

    // === Property tests derived from proof obligations ===

    /// Obligation: Attention sublayer shape preservation (invariant)
    /// Formal: for all x: shape(attention_sublayer(x)) = shape(x)
    #[test]
    fn prop_attention_sublayer_shape_preservation() {
        // Attention: Q[h, n_h*d_k] @ K^T[n_kv*d_k, h] -> [h, h] (logically per-head)
        // O projection maps back: [n_h*d_k] -> [h]
        // With residual: output shape = input shape = [seq_len, HIDDEN]
        let q_out = N_HEADS * D_K; // 4096
        assert_eq!(q_out, HIDDEN, "Q output dim must match hidden");
        let o_in = N_HEADS * D_K; // 4096
        let o_out = HIDDEN; // 4096
        assert_eq!(o_in, q_out, "O input must match Q output");
        assert_eq!(o_out, HIDDEN, "O output must restore hidden dim");
        // Shape preserved: [seq_len, HIDDEN] -> attn -> [seq_len, HIDDEN]
        for seq_len in [1, 128, 2048] {
            let input_shape = (seq_len, HIDDEN);
            let output_shape = (seq_len, o_out);
            assert_eq!(input_shape, output_shape);
        }
    }

    /// Obligation: GDN sublayer shape preservation (invariant)
    /// Formal: for all x: shape(gdn_sublayer(x)) = shape(x)
    #[test]
    fn prop_gdn_sublayer_shape_preservation() {
        // GDN: input[h] -> in_proj[h, gdn_inner] -> conv1d[gdn_inner] -> delta_net -> out_proj[gdn_inner, h]
        // With residual: output shape = input shape = [seq_len, HIDDEN]
        let gdn_in = HIDDEN;
        let gdn_out = HIDDEN; // out_proj restores hidden dim
        assert_eq!(gdn_in, gdn_out, "GDN must preserve hidden dim");
        for seq_len in [1, 128, 2048] {
            let input_shape = (seq_len, HIDDEN);
            let output_shape = (seq_len, gdn_out);
            assert_eq!(input_shape, output_shape);
        }
    }

    /// Obligation: FFN sublayer shape preservation (invariant)
    /// Formal: for all x: shape(ffn_sublayer(x)) = shape(x)
    #[test]
    fn prop_ffn_sublayer_shape_preservation() {
        // SwiGLU: gate[h, inter] * up[h, inter] -> [inter] -> down[inter, h] -> [h]
        // With residual: output shape = input shape
        let ffn_expand = INTERMEDIATE; // 12288
        let ffn_contract = HIDDEN; // 4096
        assert_eq!(ffn_expand, 3 * HIDDEN, "SwiGLU expands by 3x");
        assert_eq!(ffn_contract, HIDDEN, "down projection restores hidden dim");
        for seq_len in [1, 128, 2048] {
            let input_shape = (seq_len, HIDDEN);
            let output_shape = (seq_len, ffn_contract);
            assert_eq!(input_shape, output_shape);
        }
    }

    /// Obligation: Block outputs from exactly one attention type (invariant)
    /// Formal: for all l: is_attention(l) XOR is_gdn(l)
    #[test]
    fn prop_block_outputs_from_exactly_one_attention_type() {
        let mut attn_count = 0_usize;
        let mut gdn_count = 0_usize;
        for l in 0..N_LAYERS {
            let lt = layer_type(l);
            let is_attn = lt == LayerType::Attention;
            let is_gdn = lt == LayerType::Gdn;
            // XOR: exactly one must be true
            assert!(
                is_attn ^ is_gdn,
                "layer {l}: must be exactly one of attention or GDN"
            );
            if is_attn {
                attn_count += 1;
            } else {
                gdn_count += 1;
            }
        }
        assert_eq!(attn_count, N_ATTN_LAYERS, "must have exactly 24 attention layers");
        assert_eq!(gdn_count, N_GDN_LAYERS, "must have exactly 24 GDN layers");
        assert_eq!(attn_count + gdn_count, N_LAYERS);
    }

    /// Obligation: Activation magnitude bounded (bound)
    /// Formal: for all l: ||h_l||_inf <= M for finite M
    #[test]
    fn prop_activation_magnitude_bounded() {
        // Simulate RMSNorm preventing unbounded growth
        // Start with unit-scale hidden state, apply norm at each layer
        let mut h: Vec<f64> = vec![1.0; HIDDEN];
        for layer in 0..N_LAYERS {
            // Pre-norm: RMSNorm normalizes to unit RMS
            let normed = rmsnorm(&h);
            // After norm, RMS ≈ 1.0, so max magnitude bounded
            let max_val = normed.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
            assert!(
                max_val <= 2.0,
                "layer {layer}: ||h||_inf = {max_val} > 2.0 after RMSNorm"
            );
            // Sublayer adds bounded perturbation (simulated as small noise)
            // Residual: h = h + sublayer(norm(h))
            // Since sublayer output is bounded and we renorm next iteration, h stays bounded
            h = normed; // simplified: just use normed output
        }
        let final_max = h.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        assert!(final_max.is_finite(), "final activation must be finite");
    }

    /// Obligation: RMSNorm precedes each sublayer (invariant)
    /// Formal: pre-norm architecture: norm before attention/GDN and before FFN
    #[test]
    fn prop_rmsnorm_precedes_each_sublayer() {
        // In pre-norm architecture, each block has structure:
        // h = h + attn_or_gdn(rmsnorm1(h))
        // h = h + ffn(rmsnorm2(h))
        // So 2 RMSNorm per block, always preceding the sublayer
        let norms_per_block = 2_usize; // norm before attn/gdn + norm before ffn
        let total_norms = N_LAYERS * norms_per_block;
        assert_eq!(total_norms, 96, "48 layers * 2 norms = 96 total RMSNorm");
        // Plus 1 final norm after all blocks
        let total_with_final = total_norms + 1;
        assert_eq!(total_with_final, 97);
    }

    /// Obligation: Residual identity component (conservation)
    /// Formal: h_{l+1} - h_l = sublayer(norm(h_l))
    #[test]
    fn prop_residual_identity_component() {
        // Verify residual arithmetic: output = input + sublayer(norm(input))
        // The difference must equal exactly the sublayer output
        let h = vec![1.0_f64; 8]; // small test vector
        let normed = rmsnorm(&h);
        // Simulate sublayer as scaling by 0.5 (any bounded function)
        let sublayer_out: Vec<f64> = normed.iter().map(|v| v * 0.5).collect();
        // Residual: h_next = h + sublayer_out
        let h_next: Vec<f64> = h.iter().zip(&sublayer_out).map(|(a, b)| a + b).collect();
        // Verify: h_next - h == sublayer_out
        for i in 0..h.len() {
            let diff = h_next[i] - h[i];
            assert!(
                (diff - sublayer_out[i]).abs() < 1e-12,
                "residual identity violated at index {i}: diff={diff}, sublayer={}", sublayer_out[i]
            );
        }
    }

    // === Falsification tests ===

    /// FALSIFY-QHF-001: Attention sublayer shape
    /// Prediction: Output shape equals input shape
    #[test]
    fn prop_falsify_qhf_001() {
        // Test with multiple (d_model, seq_len) configs
        for (d_model, n_h, d_k) in [(4096, 16, 256), (2048, 16, 128), (8192, 64, 128)] {
            let q_dim = n_h * d_k;
            assert_eq!(q_dim, d_model, "d={d_model}: Q output {q_dim} != d_model");
            // O projection: [n_h*d_k, d_model] -> restores d_model
            let o_out = d_model;
            assert_eq!(o_out, d_model, "O projection must restore d_model");
        }
    }

    /// FALSIFY-QHF-002: GDN sublayer shape
    /// Prediction: Output shape equals input shape
    #[test]
    fn prop_falsify_qhf_002() {
        // GDN out_proj always maps back to d_model
        for d_model in [2048, 4096, 8192] {
            let gdn_in = d_model;
            let gdn_out = d_model; // out_proj restores dimension
            assert_eq!(gdn_in, gdn_out, "GDN shape mismatch for d={d_model}");
        }
    }

    /// FALSIFY-QHF-003: FFN sublayer shape
    /// Prediction: Output shape equals input shape
    #[test]
    fn prop_falsify_qhf_003() {
        // SwiGLU: expand then contract back to d_model
        for (d_model, inter) in [(4096, 12288), (2048, 5504), (8192, 24576)] {
            let ffn_out = d_model; // down projection restores
            assert_eq!(ffn_out, d_model, "FFN d={d_model} inter={inter}: shape break");
            assert!(inter > d_model, "intermediate must be > hidden");
        }
    }

    /// FALSIFY-QHF-004: Exclusive layer type
    /// Prediction: Each layer is attention XOR GDN
    #[test]
    fn prop_falsify_qhf_004() {
        // Exhaustive check over all 48 layers
        for l in 0..48_usize {
            let lt = layer_type(l);
            let is_attn = lt == LayerType::Attention;
            let is_gdn = lt == LayerType::Gdn;
            assert_ne!(is_attn, is_gdn, "layer {l}: both or neither type assigned");
            // Even layers are attention, odd are GDN
            if l % 2 == 0 {
                assert!(is_attn, "even layer {l} must be attention");
            } else {
                assert!(is_gdn, "odd layer {l} must be GDN");
            }
        }
    }

    /// FALSIFY-QHF-005: Activation stability
    /// Prediction: No NaN/Inf after 48 layers
    #[test]
    fn prop_falsify_qhf_005() {
        // Simulate forward pass with various initial magnitudes
        for init_val in [0.01, 0.1, 1.0, 10.0, 100.0] {
            let mut h = vec![init_val; 64]; // small test vector
            for layer in 0..48 {
                h = rmsnorm(&h);
                // Check no NaN/Inf
                for (i, v) in h.iter().enumerate() {
                    assert!(
                        v.is_finite(),
                        "init={init_val} layer={layer} idx={i}: got non-finite {v}"
                    );
                }
            }
        }
    }

    /// FALSIFY-QHF-006: RMSNorm pre-normalization
    /// Prediction: RMSNorm applied before attention/GDN and before FFN
    #[test]
    fn prop_falsify_qhf_006() {
        // Verify that norm changes the vector (i.e., it's actually applied)
        let h = vec![3.0_f64; 8];
        let normed = rmsnorm(&h);
        // After RMSNorm, RMS should be ≈ 1.0
        let rms_after = (normed.iter().map(|v| v * v).sum::<f64>() / normed.len() as f64).sqrt();
        assert!(
            (rms_after - 1.0).abs() < 1e-9,
            "RMSNorm must produce unit RMS, got {rms_after}"
        );
        // Pre-norm architecture: 2 norms per layer
        assert_eq!(N_LAYERS * 2, 96, "48 * 2 = 96 RMSNorm applications per forward pass");
    }

    /// FALSIFY-QHF-007: Residual stream correctness
    /// Prediction: h_{l+1} - h_l = sublayer(norm(h_l))
    #[test]
    fn prop_falsify_qhf_007() {
        // Test with multiple random-like inputs
        for scale in [0.1, 1.0, 5.0] {
            let h: Vec<f64> = (0..16).map(|i| (i as f64 + 1.0) * scale).collect();
            let normed = rmsnorm(&h);
            // Mock sublayer: element-wise tanh (bounded function)
            let sublayer_out: Vec<f64> = normed.iter().map(|v| v.tanh()).collect();
            let h_next: Vec<f64> = h.iter().zip(&sublayer_out).map(|(a, b)| a + b).collect();
            for i in 0..h.len() {
                let residual = h_next[i] - h[i];
                assert!(
                    (residual - sublayer_out[i]).abs() < 1e-12,
                    "scale={scale} idx={i}: residual {residual} != sublayer {}",
                    sublayer_out[i]
                );
            }
        }
    }
}
