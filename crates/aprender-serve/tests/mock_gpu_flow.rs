//! Phase 43 - Mock GPU Flow Tests
//!
//! Tests GpuModel forward pass flow using MockExecutor.
//! This enables testing of model logic without CUDA hardware.

use realizar::gpu::executor::{CpuExecutor, ExecutorCall, GpuExecutorTrait, MockExecutor};
use realizar::gpu::scheduler::{
    AttentionBuffers, BlockWeights, GpuGenerateConfig, GpuModel, GpuModelConfig,
};

// ============================================================================
// GpuExecutorTrait Tests
// ============================================================================

#[test]
fn test_mock_executor_basic() {
    let mock = MockExecutor::new("test_mock");

    assert_eq!(mock.name(), "test_mock");
    assert!(mock.is_available());
    assert_eq!(mock.call_count(), 0);
}

#[test]
fn test_mock_executor_matmul_recording() {
    let mut mock = MockExecutor::new("recorder");

    // Perform matmul
    let a = vec![1.0f32; 64]; // 8x8
    let b = vec![1.0f32; 64]; // 8x8
    let result = mock.matmul(&a, &b, 8, 8, 8).unwrap();

    // Verify recording
    assert_eq!(mock.call_count(), 1);
    assert_eq!(mock.matmul_count(), 1);
    assert_eq!(result.len(), 64); // 8x8 output

    // Check the call details
    let call = mock.last_call().unwrap();
    match call {
        ExecutorCall::Matmul {
            a_len,
            b_len,
            m,
            k,
            n,
        } => {
            assert_eq!(a_len, 64);
            assert_eq!(b_len, 64);
            assert_eq!(m, 8);
            assert_eq!(k, 8);
            assert_eq!(n, 8);
        },
        _ => panic!("Expected Matmul call"),
    }
}

#[test]
fn test_mock_executor_multiple_calls() {
    let mut mock = MockExecutor::new("multi");

    // QKV projection: 1x512 @ 512x1536 = 1x1536
    let _ = mock.matmul(&vec![1.0; 512], &vec![1.0; 512 * 1536], 1, 512, 1536);

    // Output projection: 1x512 @ 512x512 = 1x512
    let _ = mock.matmul(&vec![1.0; 512], &vec![1.0; 512 * 512], 1, 512, 512);

    // FFN up: 1x512 @ 512x2048 = 1x2048
    let _ = mock.matmul(&vec![1.0; 512], &vec![1.0; 512 * 2048], 1, 512, 2048);

    // FFN down: 1x2048 @ 2048x512 = 1x512
    let _ = mock.matmul(&vec![1.0; 2048], &vec![1.0; 2048 * 512], 1, 2048, 512);

    assert_eq!(mock.matmul_count(), 4);
}

#[test]
fn test_mock_executor_unavailable() {
    let mock = MockExecutor::unavailable("disabled");
    assert!(!mock.is_available());
}

#[test]
fn test_mock_executor_custom_result() {
    let custom_result = vec![42.0f32; 4];
    let mut mock = MockExecutor::new("custom").with_matmul_result(custom_result.clone());

    let result = mock.matmul(&[1.0; 4], &[1.0; 4], 2, 2, 2).unwrap();
    assert_eq!(result, custom_result);
}

#[test]
fn test_mock_executor_failure() {
    let mut mock = MockExecutor::new("failing").with_matmul_failure();

    let result = mock.matmul(&[1.0; 4], &[1.0; 4], 2, 2, 2);
    assert!(result.is_err());

    // Even failed calls are recorded
    assert_eq!(mock.call_count(), 1);
}

#[test]
fn test_mock_executor_clear() {
    let mut mock = MockExecutor::new("clear_test");

    let _ = mock.matmul(&[1.0; 4], &[1.0; 4], 2, 2, 2);
    let _ = mock.matmul(&[1.0; 4], &[1.0; 4], 2, 2, 2);
    assert_eq!(mock.call_count(), 2);

    mock.clear_calls();
    assert_eq!(mock.call_count(), 0);
}

// ============================================================================
// CpuExecutor Tests
// ============================================================================

#[test]
fn test_cpu_executor_basic() {
    let cpu = CpuExecutor::new();
    assert_eq!(cpu.name(), "CpuExecutor");
    assert!(cpu.is_available());
}

#[test]
fn test_cpu_executor_matmul_identity() {
    let mut cpu = CpuExecutor::new();

    // Identity matrix multiplication
    // [1,0] @ [1,0]   [1,0]
    // [0,1]   [0,1] = [0,1]
    let a = vec![1.0, 0.0, 0.0, 1.0];
    let b = vec![1.0, 0.0, 0.0, 1.0];

    let c = cpu.matmul(&a, &b, 2, 2, 2).unwrap();

    assert!((c[0] - 1.0).abs() < 1e-6);
    assert!((c[1] - 0.0).abs() < 1e-6);
    assert!((c[2] - 0.0).abs() < 1e-6);
    assert!((c[3] - 1.0).abs() < 1e-6);
}

#[test]
fn test_cpu_executor_matmul_vector_matrix() {
    let mut cpu = CpuExecutor::new();

    // Row vector times matrix: [1,2] @ [[3,4],[5,6]] = [13, 16]
    // 1*3+2*5=13, 1*4+2*6=16
    let a = vec![1.0, 2.0]; // 1x2
    let b = vec![3.0, 4.0, 5.0, 6.0]; // 2x2

    let c = cpu.matmul(&a, &b, 1, 2, 2).unwrap();

    assert_eq!(c.len(), 2);
    assert!((c[0] - 13.0).abs() < 1e-6);
    assert!((c[1] - 16.0).abs() < 1e-6);
}

// ============================================================================
// Trait Object Polymorphism Tests
// ============================================================================

#[test]
fn test_executor_polymorphism() {
    fn compute_with_executor(exec: &mut dyn GpuExecutorTrait, size: usize) -> Vec<f32> {
        let a = vec![1.0f32; size * size];
        let b = vec![1.0f32; size * size];
        exec.matmul(&a, &b, size, size, size).unwrap()
    }

    let mut mock = MockExecutor::new("mock");
    let mut cpu = CpuExecutor::new();

    let mock_result = compute_with_executor(&mut mock, 4);
    let cpu_result = compute_with_executor(&mut cpu, 4);

    // Mock returns zeros
    assert!(mock_result.iter().all(|&x| x == 0.0));

    // CPU returns actual sums (each element is sum of row * col = size)
    assert!(cpu_result.iter().all(|&x| (x - 4.0).abs() < 1e-6));
}

#[test]
fn test_boxed_executor() {
    let executor: Box<dyn GpuExecutorTrait> = Box::new(MockExecutor::new("boxed"));

    assert_eq!(executor.name(), "boxed");
    assert!(executor.is_available());
    assert!(executor.synchronize().is_ok());
}

// ============================================================================
// Model Config Tests
// ============================================================================

#[test]
fn test_gpu_model_config() {
    let config = GpuModelConfig {
        vocab_size: 32000,
        hidden_dim: 2048,
        num_heads: 32,
        num_kv_heads: 8, // GQA
        num_layers: 22,
        intermediate_dim: 5632,
        eps: 1e-5,
        rope_theta: 10000.0,
        explicit_head_dim: None,
        layer_types: None,
        linear_key_head_dim: None,
        linear_value_head_dim: None,
        linear_num_key_heads: None,
        linear_num_value_heads: None,
        linear_conv_kernel_dim: None,
        constraints: None,
        num_experts: None,
        num_experts_per_tok: None,
        expert_intermediate_size: None,
    };

    assert_eq!(config.head_dim(), 64); // 2048/32
    assert_eq!(config.kv_dim(), 512); // 8*64
    assert_eq!(config.qkv_dim(), 3072); // 2048 + 2*512
    assert!(config.is_gqa());
}

#[test]
fn test_gpu_model_config_mha() {
    let config = GpuModelConfig {
        vocab_size: 50257,
        hidden_dim: 768,
        num_heads: 12,
        num_kv_heads: 12, // MHA (not GQA)
        num_layers: 12,
        intermediate_dim: 3072,
        eps: 1e-5,
        rope_theta: 10000.0,
        explicit_head_dim: None,
        layer_types: None,
        linear_key_head_dim: None,
        linear_value_head_dim: None,
        linear_num_key_heads: None,
        linear_num_value_heads: None,
        linear_conv_kernel_dim: None,
        constraints: None,
        num_experts: None,
        num_experts_per_tok: None,
        expert_intermediate_size: None,
    };

    assert!(!config.is_gqa());
    assert_eq!(config.qkv_dim(), 3 * 768); // 3 * hidden_dim for MHA
}

// ============================================================================
// Generate Config Tests
// ============================================================================

#[test]
fn test_generate_config_default() {
    let config = GpuGenerateConfig::default();

    assert_eq!(config.max_tokens, 64);
    assert_eq!(config.temperature, 0.0);
    assert_eq!(config.top_k, 1);
    assert!(config.stop_tokens.is_empty());
}

#[test]
fn test_generate_config_deterministic() {
    let config = GpuGenerateConfig::deterministic(128);

    assert_eq!(config.max_tokens, 128);
    assert_eq!(config.temperature, 0.0);
    assert_eq!(config.top_k, 1);
}

#[test]
fn test_generate_config_sampling() {
    let config = GpuGenerateConfig::with_sampling(256, 0.7, 40);

    assert_eq!(config.max_tokens, 256);
    assert_eq!(config.temperature, 0.7);
    assert_eq!(config.top_k, 40);
}

#[test]
fn test_generate_config_stop_tokens() {
    let config = GpuGenerateConfig::deterministic(64).with_stop_tokens(vec![1, 2, 3]);

    assert_eq!(config.stop_tokens, vec![1, 2, 3]);
}

// ============================================================================
// AttentionBuffers Tests
// ============================================================================

#[test]
fn test_attention_buffers_creation() {
    let config = GpuModelConfig {
        vocab_size: 1000,
        hidden_dim: 256,
        num_heads: 4,
        num_kv_heads: 4,
        num_layers: 2,
        intermediate_dim: 512,
        eps: 1e-5,
        rope_theta: 10000.0,
        explicit_head_dim: None,
        layer_types: None,
        linear_key_head_dim: None,
        linear_value_head_dim: None,
        linear_num_key_heads: None,
        linear_num_value_heads: None,
        linear_conv_kernel_dim: None,
        constraints: None,
        num_experts: None,
        num_experts_per_tok: None,
        expert_intermediate_size: None,
    };

    let buffers = AttentionBuffers::new(&config, 512);

    assert_eq!(buffers.q_buffer.len(), 256);
    assert_eq!(buffers.scores_buffer.len(), 4 * 512); // num_heads * max_seq_len
    assert_eq!(buffers.output_buffer.len(), 256);
    assert_eq!(buffers.kv_proj_buffer.len(), 256);
    assert_eq!(buffers.ffn_buffer.len(), 512);
    assert_eq!(buffers.max_seq_len, 512);
}

#[test]
fn test_attention_buffers_reset() {
    let config = GpuModelConfig {
        vocab_size: 100,
        hidden_dim: 64,
        num_heads: 2,
        num_kv_heads: 2,
        num_layers: 1,
        intermediate_dim: 128,
        eps: 1e-5,
        rope_theta: 10000.0,
        explicit_head_dim: None,
        layer_types: None,
        linear_key_head_dim: None,
        linear_value_head_dim: None,
        linear_num_key_heads: None,
        linear_num_value_heads: None,
        linear_conv_kernel_dim: None,
        constraints: None,
        num_experts: None,
        num_experts_per_tok: None,
        expert_intermediate_size: None,
    };

    let mut buffers = AttentionBuffers::new(&config, 128);

    // Fill with non-zero values
    buffers.q_buffer.fill(1.0);
    buffers.scores_buffer.fill(2.0);
    buffers.output_buffer.fill(3.0);

    // Reset
    buffers.reset();

    // Verify all zeros
    assert!(buffers.q_buffer.iter().all(|&x| x == 0.0));
    assert!(buffers.scores_buffer.iter().all(|&x| x == 0.0));
    assert!(buffers.output_buffer.iter().all(|&x| x == 0.0));
}

// ============================================================================
// BlockWeights Tests
// ============================================================================

#[test]
fn test_block_weights_structure() {
    let hidden_dim = 256;
    let intermediate_dim = 512;
    let qkv_dim = 768; // 3 * hidden_dim for MHA

    let block = BlockWeights {
        attn_norm_weight: vec![1.0; hidden_dim],
        attn_norm_bias: vec![0.0; hidden_dim],
        qkv_weight: vec![0.01; hidden_dim * qkv_dim],
        qkv_bias: vec![0.0; qkv_dim],
        out_weight: vec![0.01; hidden_dim * hidden_dim],
        out_bias: vec![0.0; hidden_dim],
        ffn_norm_weight: vec![1.0; hidden_dim],
        ffn_norm_bias: vec![0.0; hidden_dim],
        ffn_fc1_weight: vec![0.01; hidden_dim * intermediate_dim],
        ffn_fc1_bias: vec![0.0; intermediate_dim],
        ffn_fc2_weight: vec![0.01; intermediate_dim * hidden_dim],
        ffn_fc2_bias: vec![0.0; hidden_dim],
        ffn_gate_weight: None,
        linear_attn: None,
        moe_experts: None,
    };

    assert_eq!(block.attn_norm_weight.len(), hidden_dim);
    assert_eq!(block.qkv_weight.len(), hidden_dim * qkv_dim);
    assert_eq!(block.ffn_fc1_weight.len(), hidden_dim * intermediate_dim);
    assert!(block.ffn_gate_weight.is_none());
}

#[test]
fn test_block_weights_with_gate() {
    let hidden_dim = 256;
    let intermediate_dim = 512;

    let block = BlockWeights {
        attn_norm_weight: vec![1.0; hidden_dim],
        attn_norm_bias: vec![0.0; hidden_dim],
        qkv_weight: vec![0.01; hidden_dim * 3 * hidden_dim],
        qkv_bias: vec![],
        out_weight: vec![0.01; hidden_dim * hidden_dim],
        out_bias: vec![0.0; hidden_dim],
        ffn_norm_weight: vec![1.0; hidden_dim],
        ffn_norm_bias: vec![0.0; hidden_dim],
        ffn_fc1_weight: vec![0.01; hidden_dim * intermediate_dim],
        ffn_fc1_bias: vec![0.0; intermediate_dim],
        ffn_fc2_weight: vec![0.01; intermediate_dim * hidden_dim],
        ffn_fc2_bias: vec![0.0; hidden_dim],
        ffn_gate_weight: Some(vec![0.01; hidden_dim * intermediate_dim]), // SwiGLU
        linear_attn: None,
        moe_experts: None,
    };

    assert!(block.ffn_gate_weight.is_some());
    assert_eq!(
        block.ffn_gate_weight.as_ref().unwrap().len(),
        hidden_dim * intermediate_dim
    );
}

// ============================================================================
// GpuModel Creation Tests
// ============================================================================

#[test]
fn test_gpu_model_creation() {
    let config = GpuModelConfig {
        vocab_size: 100,
        hidden_dim: 64,
        num_heads: 2,
        num_kv_heads: 2,
        num_layers: 2,
        intermediate_dim: 128,
        eps: 1e-5,
        rope_theta: 10000.0,
        explicit_head_dim: None,
        layer_types: None,
        linear_key_head_dim: None,
        linear_value_head_dim: None,
        linear_num_key_heads: None,
        linear_num_value_heads: None,
        linear_conv_kernel_dim: None,
        constraints: None,
        num_experts: None,
        num_experts_per_tok: None,
        expert_intermediate_size: None,
    };

    let model = GpuModel::new(config);
    assert!(model.is_ok());

    let model = model.unwrap();
    assert_eq!(model.config.vocab_size, 100);
    assert_eq!(model.config.hidden_dim, 64);
    assert_eq!(model.config.num_layers, 2);
}

// ============================================================================
// Simulated Forward Flow Tests (using MockExecutor concepts)
// ============================================================================

/// Simulates the matmul calls that would occur during a transformer forward pass
#[test]
fn test_simulated_forward_flow() {
    let mut mock = MockExecutor::new("forward_flow");

    let hidden_dim = 256;
    let num_heads = 4;
    let num_kv_heads = 4;
    let head_dim = hidden_dim / num_heads;
    let kv_dim = num_kv_heads * head_dim;
    let qkv_dim = hidden_dim + 2 * kv_dim;
    let intermediate_dim = 512;

    // Simulate one block's forward pass:

    // 1. QKV projection: [1, hidden_dim] @ [hidden_dim, qkv_dim]
    let _ = mock.matmul(
        &vec![1.0; hidden_dim],
        &vec![1.0; hidden_dim * qkv_dim],
        1,
        hidden_dim,
        qkv_dim,
    );

    // 2. Output projection: [1, hidden_dim] @ [hidden_dim, hidden_dim]
    let _ = mock.matmul(
        &vec![1.0; hidden_dim],
        &vec![1.0; hidden_dim * hidden_dim],
        1,
        hidden_dim,
        hidden_dim,
    );

    // 3. FFN up: [1, hidden_dim] @ [hidden_dim, intermediate_dim]
    let _ = mock.matmul(
        &vec![1.0; hidden_dim],
        &vec![1.0; hidden_dim * intermediate_dim],
        1,
        hidden_dim,
        intermediate_dim,
    );

    // 4. FFN down: [1, intermediate_dim] @ [intermediate_dim, hidden_dim]
    let _ = mock.matmul(
        &vec![1.0; intermediate_dim],
        &vec![1.0; intermediate_dim * hidden_dim],
        1,
        intermediate_dim,
        hidden_dim,
    );

    // Verify: One block = 4 matmul calls (QKV, Out, FFN up, FFN down)
    assert_eq!(mock.matmul_count(), 4);

    // Verify call sequence
    let calls = mock.calls();

    // QKV projection
    match &calls[0] {
        ExecutorCall::Matmul { m, k, n, .. } => {
            assert_eq!(*m, 1);
            assert_eq!(*k, hidden_dim);
            assert_eq!(*n, qkv_dim);
        },
        _ => panic!("Expected Matmul"),
    }

    // Output projection
    match &calls[1] {
        ExecutorCall::Matmul { m, k, n, .. } => {
            assert_eq!(*m, 1);
            assert_eq!(*k, hidden_dim);
            assert_eq!(*n, hidden_dim);
        },
        _ => panic!("Expected Matmul"),
    }
}

/// Simulates full model forward pass across multiple layers
#[test]
fn test_simulated_multi_layer_forward() {
    let mut mock = MockExecutor::new("multi_layer");

    let num_layers = 4;
    let hidden_dim = 256;
    let qkv_dim = 768;
    let intermediate_dim = 512;
    let vocab_size = 1000;

    // Simulate forward through all layers
    for _layer in 0..num_layers {
        // QKV
        let _ = mock.matmul(
            &vec![1.0; hidden_dim],
            &vec![1.0; hidden_dim * qkv_dim],
            1,
            hidden_dim,
            qkv_dim,
        );
        // Out
        let _ = mock.matmul(
            &vec![1.0; hidden_dim],
            &vec![1.0; hidden_dim * hidden_dim],
            1,
            hidden_dim,
            hidden_dim,
        );
        // FFN up
        let _ = mock.matmul(
            &vec![1.0; hidden_dim],
            &vec![1.0; hidden_dim * intermediate_dim],
            1,
            hidden_dim,
            intermediate_dim,
        );
        // FFN down
        let _ = mock.matmul(
            &vec![1.0; intermediate_dim],
            &vec![1.0; intermediate_dim * hidden_dim],
            1,
            intermediate_dim,
            hidden_dim,
        );
    }

    // LM head projection
    let _ = mock.matmul(
        &vec![1.0; hidden_dim],
        &vec![1.0; hidden_dim * vocab_size],
        1,
        hidden_dim,
        vocab_size,
    );

    // 4 matmuls per layer * 4 layers + 1 LM head = 17 total
    assert_eq!(mock.matmul_count(), 17);
}

/// Simulates SwiGLU FFN (with gate projection)
#[test]
fn test_simulated_swiglu_forward() {
    let mut mock = MockExecutor::new("swiglu");

    let hidden_dim = 256;
    let intermediate_dim = 512;

    // SwiGLU has 3 projections: up, gate, down
    // up: hidden -> intermediate
    let _ = mock.matmul(
        &vec![1.0; hidden_dim],
        &vec![1.0; hidden_dim * intermediate_dim],
        1,
        hidden_dim,
        intermediate_dim,
    );

    // gate: hidden -> intermediate
    let _ = mock.matmul(
        &vec![1.0; hidden_dim],
        &vec![1.0; hidden_dim * intermediate_dim],
        1,
        hidden_dim,
        intermediate_dim,
    );

    // down: intermediate -> hidden
    let _ = mock.matmul(
        &vec![1.0; intermediate_dim],
        &vec![1.0; intermediate_dim * hidden_dim],
        1,
        intermediate_dim,
        hidden_dim,
    );

    // SwiGLU FFN = 3 matmuls (up, gate, down)
    assert_eq!(mock.matmul_count(), 3);
}

// ============================================================================
// Phase 43: GpuModel with Test Executor Tests
// ============================================================================

/// Test injecting MockExecutor into GpuModel
#[test]
fn test_gpu_model_with_test_executor() {
    let config = GpuModelConfig {
        vocab_size: 100,
        hidden_dim: 64,
        num_heads: 2,
        num_kv_heads: 2,
        num_layers: 2,
        intermediate_dim: 128,
        eps: 1e-5,
        rope_theta: 10000.0,
        explicit_head_dim: None,
        layer_types: None,
        linear_key_head_dim: None,
        linear_value_head_dim: None,
        linear_num_key_heads: None,
        linear_num_value_heads: None,
        linear_conv_kernel_dim: None,
        constraints: None,
        num_experts: None,
        num_experts_per_tok: None,
        expert_intermediate_size: None,
    };

    let mut model = GpuModel::new(config).unwrap();

    // Verify no test executor initially
    assert!(!model.has_test_executor());

    // Inject mock executor
    let mock = MockExecutor::new("injected");
    model.with_test_executor(Box::new(mock));

    // Verify test executor is set
    assert!(model.has_test_executor());
}

/// Test that do_matmul uses the injected test executor
#[test]
fn test_gpu_model_do_matmul_with_mock() {
    let config = GpuModelConfig {
        vocab_size: 100,
        hidden_dim: 64,
        num_heads: 2,
        num_kv_heads: 2,
        num_layers: 2,
        intermediate_dim: 128,
        eps: 1e-5,
        rope_theta: 10000.0,
        explicit_head_dim: None,
        layer_types: None,
        linear_key_head_dim: None,
        linear_value_head_dim: None,
        linear_num_key_heads: None,
        linear_num_value_heads: None,
        linear_conv_kernel_dim: None,
        constraints: None,
        num_experts: None,
        num_experts_per_tok: None,
        expert_intermediate_size: None,
    };

    let mut model = GpuModel::new(config).unwrap();

    // Inject mock with custom result (size = m * n = 2 * 2 = 4)
    let mock = MockExecutor::new("matmul_test").with_matmul_result(vec![42.0f32; 4]);
    model.with_test_executor(Box::new(mock));

    // Perform matmul through model: [2x4] @ [4x2] = [2x2]
    let a = vec![1.0f32; 8]; // 2x4
    let b = vec![1.0f32; 8]; // 4x2
    let result = model.do_matmul(&a, &b, 2, 4, 2).unwrap();

    // Verify mock returned our custom result (not actual computation)
    assert_eq!(result.len(), 4); // 2x2 output size
    assert_eq!(result, vec![42.0; 4]); // Mock returns our custom result
}

/// Test clearing test executor restores normal operation
#[test]
fn test_gpu_model_clear_test_executor() {
    let config = GpuModelConfig {
        vocab_size: 100,
        hidden_dim: 64,
        num_heads: 2,
        num_kv_heads: 2,
        num_layers: 2,
        intermediate_dim: 128,
        eps: 1e-5,
        rope_theta: 10000.0,
        explicit_head_dim: None,
        layer_types: None,
        linear_key_head_dim: None,
        linear_value_head_dim: None,
        linear_num_key_heads: None,
        linear_num_value_heads: None,
        linear_conv_kernel_dim: None,
        constraints: None,
        num_experts: None,
        num_experts_per_tok: None,
        expert_intermediate_size: None,
    };

    let mut model = GpuModel::new(config).unwrap();

    // Inject mock
    model.with_test_executor(Box::new(MockExecutor::new("temp")));
    assert!(model.has_test_executor());

    // Clear it
    model.clear_test_executor();
    assert!(!model.has_test_executor());
}

/// Test that mock failure propagates through do_matmul
#[test]
fn test_gpu_model_mock_failure() {
    let config = GpuModelConfig {
        vocab_size: 100,
        hidden_dim: 64,
        num_heads: 2,
        num_kv_heads: 2,
        num_layers: 2,
        intermediate_dim: 128,
        eps: 1e-5,
        rope_theta: 10000.0,
        explicit_head_dim: None,
        layer_types: None,
        linear_key_head_dim: None,
        linear_value_head_dim: None,
        linear_num_key_heads: None,
        linear_num_value_heads: None,
        linear_conv_kernel_dim: None,
        constraints: None,
        num_experts: None,
        num_experts_per_tok: None,
        expert_intermediate_size: None,
    };

    let mut model = GpuModel::new(config).unwrap();

    // Inject failing mock
    let mock = MockExecutor::new("failing").with_matmul_failure();
    model.with_test_executor(Box::new(mock));

    // do_matmul should fail
    let result = model.do_matmul(&[1.0; 4], &[1.0; 4], 2, 2, 2);
    assert!(result.is_err());
}
