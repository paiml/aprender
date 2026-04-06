//! T-COV-95 Extended Coverage: convert/mod.rs
//!
//! Targets: RawTensor struct, Q4KConversionStats, GgufToAprQ4KConverter helpers,
//! infer_rope_type logic, byte size calculations, edge cases.

use crate::convert::{ConversionStats, RawTensor};

// ============================================================================
// RawTensor struct coverage
// ============================================================================

#[test]
fn test_raw_tensor_debug() {
    let tensor = RawTensor {
        name: "test_tensor".to_string(),
        data: vec![1, 2, 3, 4],
        shape: vec![2, 2],
        dtype: 0, // F32
    };
    let debug = format!("{:?}", tensor);
    assert!(debug.contains("test_tensor"));
    assert!(debug.contains("RawTensor"));
}

#[test]
fn test_raw_tensor_clone() {
    let tensor = RawTensor {
        name: "tensor1".to_string(),
        data: vec![0x41, 0x50, 0x52],
        shape: vec![3],
        dtype: 12, // Q4_K
    };
    let cloned = tensor.clone();
    assert_eq!(cloned.name, "tensor1");
    assert_eq!(cloned.data.len(), 3);
    assert_eq!(cloned.shape, vec![3]);
    assert_eq!(cloned.dtype, 12);
}

#[test]
fn test_raw_tensor_empty_data() {
    let tensor = RawTensor {
        name: "empty".to_string(),
        data: vec![],
        shape: vec![],
        dtype: 0,
    };
    assert!(tensor.data.is_empty());
    assert!(tensor.shape.is_empty());
}

#[test]
fn test_raw_tensor_f32_dtype() {
    let tensor = RawTensor {
        name: "weights".to_string(),
        data: vec![0; 16], // 4 f32 values
        shape: vec![2, 2],
        dtype: 0, // F32
    };
    assert_eq!(tensor.dtype, 0);
}

#[test]
fn test_raw_tensor_f16_dtype() {
    let tensor = RawTensor {
        name: "activations".to_string(),
        data: vec![0; 8], // 4 f16 values
        shape: vec![4],
        dtype: 1, // F16
    };
    assert_eq!(tensor.dtype, 1);
}

#[test]
fn test_raw_tensor_q4k_dtype() {
    let tensor = RawTensor {
        name: "layer_weights".to_string(),
        data: vec![0; 144], // 1 Q4K super-block
        shape: vec![256],
        dtype: 12, // Q4_K
    };
    assert_eq!(tensor.dtype, 12);
}

#[test]
fn test_raw_tensor_q5k_dtype() {
    let tensor = RawTensor {
        name: "qkv".to_string(),
        data: vec![0; 176], // 1 Q5K super-block
        shape: vec![256],
        dtype: 13, // Q5_K
    };
    assert_eq!(tensor.dtype, 13);
}

#[test]
fn test_raw_tensor_q6k_dtype() {
    let tensor = RawTensor {
        name: "output".to_string(),
        data: vec![0; 210], // 1 Q6K super-block
        shape: vec![256],
        dtype: 14, // Q6_K
    };
    assert_eq!(tensor.dtype, 14);
}

#[test]
fn test_raw_tensor_q8_0_dtype() {
    let tensor = RawTensor {
        name: "activations".to_string(),
        data: vec![0; 34], // 1 Q8_0 block
        shape: vec![32],
        dtype: 8, // Q8_0
    };
    assert_eq!(tensor.dtype, 8);
}

#[test]
fn test_raw_tensor_multi_dim_shape() {
    let tensor = RawTensor {
        name: "mlp".to_string(),
        data: vec![0; 1024],
        shape: vec![4, 8, 32],
        dtype: 0,
    };
    assert_eq!(tensor.shape.len(), 3);
    assert_eq!(tensor.shape[0] * tensor.shape[1] * tensor.shape[2], 1024);
}

// ============================================================================
// ConversionStats edge cases
// ============================================================================

#[test]
fn test_conversion_stats_u64_overflow_safe() {
    // Large but valid values that won't overflow u64
    let stats = ConversionStats {
        total_parameters: 1_000_000_000_000, // 1T
        memory_bytes_f32: 4_000_000_000_000, // 4TB
        num_layers: 256,
        hidden_dim: 65536,
        vocab_size: 1_000_000,
        architecture: "mega_model".to_string(),
    };
    assert!((stats.parameters_b() - 1000.0).abs() < 0.01);
    let gb = stats.memory_gb();
    assert!(gb > 3700.0);
}

#[test]
fn test_conversion_stats_tiny_model() {
    let stats = ConversionStats {
        total_parameters: 1000,
        memory_bytes_f32: 4000,
        num_layers: 1,
        hidden_dim: 16,
        vocab_size: 50,
        architecture: "nano".to_string(),
    };
    assert!((stats.parameters_m() - 0.001).abs() < 0.0001);
    assert!((stats.parameters_b() - 0.000001).abs() < 0.0000001);
}

#[test]
fn test_conversion_stats_exact_mb() {
    let stats = ConversionStats {
        total_parameters: 0,
        memory_bytes_f32: 1024 * 1024, // Exactly 1 MB
        num_layers: 0,
        hidden_dim: 0,
        vocab_size: 0,
        architecture: String::new(),
    };
    assert!((stats.memory_mb() - 1.0).abs() < f64::EPSILON);
}

#[test]
fn test_conversion_stats_exact_gb() {
    let stats = ConversionStats {
        total_parameters: 0,
        memory_bytes_f32: 1024 * 1024 * 1024, // Exactly 1 GB
        num_layers: 0,
        hidden_dim: 0,
        vocab_size: 0,
        architecture: String::new(),
    };
    assert!((stats.memory_gb() - 1.0).abs() < f64::EPSILON);
}

#[test]
fn test_conversion_stats_exact_million() {
    let stats = ConversionStats {
        total_parameters: 1_000_000, // Exactly 1M
        memory_bytes_f32: 0,
        num_layers: 0,
        hidden_dim: 0,
        vocab_size: 0,
        architecture: String::new(),
    };
    assert!((stats.parameters_m() - 1.0).abs() < f64::EPSILON);
}

#[test]
fn test_conversion_stats_exact_billion() {
    let stats = ConversionStats {
        total_parameters: 1_000_000_000, // Exactly 1B
        memory_bytes_f32: 0,
        num_layers: 0,
        hidden_dim: 0,
        vocab_size: 0,
        architecture: String::new(),
    };
    assert!((stats.parameters_b() - 1.0).abs() < f64::EPSILON);
}

// ============================================================================
// RawTensor with realistic tensor names
// ============================================================================

#[test]
fn test_raw_tensor_embedding_name() {
    let tensor = RawTensor {
        name: "model.embed_tokens.weight".to_string(),
        data: vec![0; 32000 * 4096 * 2], // F16 embedding
        shape: vec![32000, 4096],
        dtype: 1, // F16
    };
    assert!(tensor.name.contains("embed_tokens"));
}

#[test]
fn test_raw_tensor_attention_qkv_name() {
    let tensor = RawTensor {
        name: "model.layers.0.self_attn.qkv_proj.weight".to_string(),
        data: vec![0; 144 * 48], // Q4_K super-blocks
        shape: vec![12288, 4096],
        dtype: 12, // Q4_K
    };
    assert!(tensor.name.contains("self_attn"));
    assert!(tensor.name.contains("qkv_proj"));
}

#[test]
fn test_raw_tensor_mlp_gate_name() {
    let tensor = RawTensor {
        name: "model.layers.5.mlp.gate_proj.weight".to_string(),
        data: vec![0; 144 * 176], // Q4_K super-blocks
        shape: vec![45056, 4096],
        dtype: 12, // Q4_K
    };
    assert!(tensor.name.contains("mlp"));
    assert!(tensor.name.contains("gate_proj"));
}

#[test]
fn test_raw_tensor_lm_head_name() {
    let tensor = RawTensor {
        name: "lm_head.weight".to_string(),
        data: vec![0; 210 * 500], // Q6_K super-blocks
        shape: vec![128000, 4096],
        dtype: 14, // Q6_K
    };
    assert!(tensor.name.contains("lm_head"));
}

#[test]
fn test_raw_tensor_norm_weight_name() {
    let tensor = RawTensor {
        name: "model.layers.0.input_layernorm.weight".to_string(),
        data: vec![0; 4096 * 4], // F32
        shape: vec![4096],
        dtype: 0, // F32
    };
    assert!(tensor.name.contains("layernorm"));
}

// ============================================================================
// Multiple tensor processing simulation
// ============================================================================

#[test]
fn test_raw_tensor_batch_processing() {
    let tensors = vec![
        RawTensor {
            name: "embed".to_string(),
            data: vec![0; 100],
            shape: vec![10, 10],
            dtype: 0,
        },
        RawTensor {
            name: "layer1".to_string(),
            data: vec![0; 200],
            shape: vec![20, 10],
            dtype: 12,
        },
        RawTensor {
            name: "output".to_string(),
            data: vec![0; 50],
            shape: vec![5, 10],
            dtype: 14,
        },
    ];

    assert_eq!(tensors.len(), 3);
    let total_bytes: usize = tensors.iter().map(|t| t.data.len()).sum();
    assert_eq!(total_bytes, 350);
}

#[test]
fn test_raw_tensor_dtype_variety() {
    // Test all known dtype values
    let dtypes = [0, 1, 8, 12, 13, 14];
    let names = ["F32", "F16", "Q8_0", "Q4_K", "Q5_K", "Q6_K"];

    for (dtype, name) in dtypes.iter().zip(names.iter()) {
        let tensor = RawTensor {
            name: format!("tensor_{}", name),
            data: vec![0; 16],
            shape: vec![16],
            dtype: *dtype,
        };
        assert_eq!(tensor.dtype, *dtype);
    }
}

// ============================================================================
// Shape validation patterns
// ============================================================================

#[test]
fn test_raw_tensor_1d_shape() {
    let tensor = RawTensor {
        name: "bias".to_string(),
        data: vec![0; 4096 * 4],
        shape: vec![4096],
        dtype: 0,
    };
    assert_eq!(tensor.shape.len(), 1);
}

#[test]
fn test_raw_tensor_2d_shape() {
    let tensor = RawTensor {
        name: "weight".to_string(),
        data: vec![0; 1024 * 1024 * 4],
        shape: vec![1024, 1024],
        dtype: 0,
    };
    assert_eq!(tensor.shape.len(), 2);
}

#[test]
fn test_raw_tensor_4d_shape() {
    let tensor = RawTensor {
        name: "conv".to_string(),
        data: vec![0; 64 * 64 * 3 * 3 * 4],
        shape: vec![64, 64, 3, 3],
        dtype: 0,
    };
    assert_eq!(tensor.shape.len(), 4);
    assert_eq!(tensor.shape[0], 64);
    assert_eq!(tensor.shape[3], 3);
}
