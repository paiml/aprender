
/// SiLU activation: x * sigmoid(x)
///
/// ONE PATH: Delegates to `trueno::silu_scalar` (UCBD §4).
#[inline]
pub(crate) fn silu(x: f32) -> f32 {
    trueno::silu_scalar(x)
}

/// GELU activation (tanh approximation)
///
/// ONE PATH: Delegates to `trueno::gelu_scalar` (UCBD §4).
#[inline]
pub(crate) fn gelu(x: f32) -> f32 {
    trueno::gelu_scalar(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_silu_basic() {
        assert!((silu(0.0) - 0.0).abs() < 1e-6);
        assert!(silu(1.0) > 0.5); // SiLU(1) ≈ 0.731
        assert!(silu(-1.0) < 0.0); // SiLU(-1) ≈ -0.269
    }

    #[test]
    fn test_gelu_basic() {
        assert!((gelu(0.0) - 0.0).abs() < 1e-6);
        assert!(gelu(1.0) > 0.8); // GELU(1) ≈ 0.841
        assert!(gelu(-1.0) < 0.0); // GELU(-1) ≈ -0.159
    }

    #[test]
    fn test_gpu_model_q4_config() {
        let model = GpuModelQ4 {
            config: AprTransformerConfig {
                architecture: "llama".to_string(),
                hidden_dim: 2048,
                num_layers: 22,
                num_heads: 32,
                num_kv_heads: 4,
                vocab_size: 32000,
                intermediate_dim: 5632,
                context_length: 2048,
                rope_theta: 10000.0,
                eps: 1e-5,
            eos_token_id: None,
            ..Default::default()
            },
            token_embedding: vec![0.0; 32000 * 2048],
            output_norm_weight: vec![1.0; 2048],
            layer_norms: vec![
                LayerNorms {
                    attn_norm: vec![1.0; 2048],
                    ffn_norm: vec![1.0; 2048],
                };
                22
            ],
            num_layers: 22,
            has_gate: true,
        };

        assert_eq!(model.num_layers, 22);
        assert!(model.has_gate);
        assert_eq!(model.config.hidden_dim, 2048);
    }

    #[test]
    fn test_rms_norm() {
        let model = GpuModelQ4 {
            config: AprTransformerConfig {
                architecture: "test".to_string(),
                hidden_dim: 4,
                num_layers: 1,
                num_heads: 1,
                num_kv_heads: 1,
                vocab_size: 100,
                intermediate_dim: 8,
                context_length: 128,
                rope_theta: 10000.0,
                eps: 1e-5,
            eos_token_id: None,
            ..Default::default()
            },
            token_embedding: vec![0.0; 400],
            output_norm_weight: vec![1.0; 4],
            layer_norms: vec![LayerNorms {
                attn_norm: vec![1.0; 4],
                ffn_norm: vec![1.0; 4],
            }],
            num_layers: 1,
            has_gate: false,
        };

        let mut x = vec![1.0, 2.0, 3.0, 4.0];
        let weight = vec![1.0, 1.0, 1.0, 1.0];

        model.rms_norm_inplace(&mut x, &weight);

        // RMS = sqrt((1+4+9+16)/4) = sqrt(7.5) ≈ 2.739
        // Normalized: [0.365, 0.730, 1.095, 1.461]
        let rms = (30.0_f32 / 4.0).sqrt();
        assert!((x[0] - 1.0 / rms).abs() < 1e-4);
        assert!((x[1] - 2.0 / rms).abs() < 1e-4);
    }

    #[test]
    fn test_attention_single_token() {
        let model = GpuModelQ4 {
            config: AprTransformerConfig {
                architecture: "test".to_string(),
                hidden_dim: 8,
                num_layers: 1,
                num_heads: 2,
                num_kv_heads: 2,
                vocab_size: 100,
                intermediate_dim: 16,
                context_length: 128,
                rope_theta: 10000.0,
                eps: 1e-5,
            eos_token_id: None,
            ..Default::default()
            },
            token_embedding: vec![0.0; 800],
            output_norm_weight: vec![1.0; 8],
            layer_norms: vec![LayerNorms {
                attn_norm: vec![1.0; 8],
                ffn_norm: vec![1.0; 8],
            }],
            num_layers: 1,
            has_gate: false,
        };

        // QKV: Q[8] + K[8] + V[8] = 24
        let qkv = vec![
            // Q
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, // K
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, // V
            0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
        ];

        let out = model.attention_cpu(&qkv, 1, 8, 2, 2);

        // For single token, output = V (all scores softmax to 1.0)
        assert_eq!(out.len(), 8);
        for &v in &out {
            assert!((v - 0.5).abs() < 1e-6);
        }
    }

    #[test]
    fn test_layer_norms_creation() {
        let norms = LayerNorms {
            attn_norm: vec![1.0; 128],
            ffn_norm: vec![1.0; 128],
        };

        assert_eq!(norms.attn_norm.len(), 128);
        assert_eq!(norms.ffn_norm.len(), 128);
    }

    #[test]
    fn test_adapter_create_model() {
        use crate::apr_transformer::{QuantizedAprLayerQ4, QuantizedAprTensorQ4};

        let apr = QuantizedAprTransformerQ4 {
            config: AprTransformerConfig {
                architecture: "test".to_string(),
                hidden_dim: 256,
                num_layers: 2,
                num_heads: 4,
                num_kv_heads: 4,
                vocab_size: 1000,
                intermediate_dim: 512,
                context_length: 128,
                rope_theta: 10000.0,
                eps: 1e-5,
            eos_token_id: None,
            ..Default::default()
            },
            token_embedding: vec![0.0; 1000 * 256],
            layers: vec![
                QuantizedAprLayerQ4 {
                    attn_norm_weight: vec![1.0; 256],
                    qkv_weight: QuantizedAprTensorQ4::zeros(256, 256 * 3),
                    attn_output_weight: QuantizedAprTensorQ4::zeros(256, 256),
                    ffn_up_weight: QuantizedAprTensorQ4::zeros(256, 512),
                    ffn_down_weight: QuantizedAprTensorQ4::zeros(512, 256),
                    ffn_gate_weight: Some(QuantizedAprTensorQ4::zeros(256, 512)),
                    ffn_norm_weight: Some(vec![1.0; 256]),
                },
                QuantizedAprLayerQ4 {
                    attn_norm_weight: vec![1.0; 256],
                    qkv_weight: QuantizedAprTensorQ4::zeros(256, 256 * 3),
                    attn_output_weight: QuantizedAprTensorQ4::zeros(256, 256),
                    ffn_up_weight: QuantizedAprTensorQ4::zeros(256, 512),
                    ffn_down_weight: QuantizedAprTensorQ4::zeros(512, 256),
                    ffn_gate_weight: Some(QuantizedAprTensorQ4::zeros(256, 512)),
                    ffn_norm_weight: Some(vec![1.0; 256]),
                },
            ],
            output_norm_weight: vec![1.0; 256],
            lm_head_weight: QuantizedAprTensorQ4::zeros(256, 1000),
        };

        let model = AprQ4ToGpuAdapter::create_model(&apr);

        assert_eq!(model.num_layers, 2);
        assert!(model.has_gate);
        assert_eq!(model.layer_norms.len(), 2);
        assert_eq!(model.token_embedding.len(), 256000);
    }
}
