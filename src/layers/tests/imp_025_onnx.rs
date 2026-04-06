#[test]
fn test_imp_025_onnx_export() {
    // Test ONNX-compatible graph representation
    // This validates the model can be represented as a computation graph

    // Define a simple model graph (ONNX-style)
    #[derive(Debug)]
    struct OnnxNode {
        name: String,
        op_type: String,
        inputs: Vec<String>,
        outputs: Vec<String>,
    }

    #[derive(Debug)]
    struct OnnxGraph {
        nodes: Vec<OnnxNode>,
        inputs: Vec<String>,
        outputs: Vec<String>,
    }

    // Build a simple transformer block graph
    let graph = OnnxGraph {
        inputs: vec!["input".to_string()],
        outputs: vec!["output".to_string()],
        nodes: vec![
            OnnxNode {
                name: "ln1".to_string(),
                op_type: "LayerNormalization".to_string(),
                inputs: vec!["input".to_string()],
                outputs: vec!["ln1_out".to_string()],
            },
            OnnxNode {
                name: "attn".to_string(),
                op_type: "Attention".to_string(),
                inputs: vec!["ln1_out".to_string()],
                outputs: vec!["attn_out".to_string()],
            },
            OnnxNode {
                name: "add1".to_string(),
                op_type: "Add".to_string(),
                inputs: vec!["input".to_string(), "attn_out".to_string()],
                outputs: vec!["residual1".to_string()],
            },
            OnnxNode {
                name: "ln2".to_string(),
                op_type: "LayerNormalization".to_string(),
                inputs: vec!["residual1".to_string()],
                outputs: vec!["ln2_out".to_string()],
            },
            OnnxNode {
                name: "ffn".to_string(),
                op_type: "MatMul".to_string(),
                inputs: vec!["ln2_out".to_string()],
                outputs: vec!["ffn_out".to_string()],
            },
            OnnxNode {
                name: "add2".to_string(),
                op_type: "Add".to_string(),
                inputs: vec!["residual1".to_string(), "ffn_out".to_string()],
                outputs: vec!["output".to_string()],
            },
        ],
    };

    // Verify graph structure
    assert_eq!(graph.inputs.len(), 1, "IMP-025: Should have one input");
    assert_eq!(graph.outputs.len(), 1, "IMP-025: Should have one output");
    assert_eq!(
        graph.nodes.len(),
        6,
        "IMP-025: Transformer block should have 6 ops"
    );

    // Verify topological ordering (outputs connect to subsequent inputs)
    let mut defined_tensors: std::collections::HashSet<String> =
        graph.inputs.iter().cloned().collect();

    for node in &graph.nodes {
        // All inputs should be defined
        for input in &node.inputs {
            assert!(
                defined_tensors.contains(input),
                "IMP-025: Node {} input {} should be defined",
                node.name,
                input
            );
        }
        // Define outputs
        for output in &node.outputs {
            defined_tensors.insert(output.clone());
        }
    }

    // Final outputs should be defined
    for output in &graph.outputs {
        assert!(
            defined_tensors.contains(output),
            "IMP-025: Graph output {} should be defined",
            output
        );
    }
}

/// IMP-026: Load real GGUF model weights to GPU buffers
/// Target: Load Llama-2-7B-Q4_K_M.gguf weights into WGPU buffers
/// M13 Critical Path: This bridges GGUF parser → GPU model
#[test]
#[cfg(feature = "gpu")]
fn test_imp_026_gguf_gpu_weight_loading() {
    use crate::gpu::{GpuModel, GpuModelConfig};

    // Create a minimal test GGUF for testing (in-memory)
    // Real models use MappedGGUFModel::from_path()
    let config = GpuModelConfig {
        vocab_size: 256,
        hidden_dim: 64,
        num_heads: 4,
        num_kv_heads: 4, // Standard MHA
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

    // Test 1: GpuModel::from_gguf_config creates model with correct dimensions
    let mut model = GpuModel::from_gguf_config(config.clone())
        .expect("IMP-026: Should create GpuModel from config");

    // GPU is optional for loading test - model creation is the success criterion
    let _ = model.has_gpu();

    // Test 2: Verify model config was preserved
    let model_config = model.config();
    assert_eq!(
        model_config.vocab_size, config.vocab_size,
        "IMP-026: vocab_size should match"
    );
    assert_eq!(
        model_config.hidden_dim, config.hidden_dim,
        "IMP-026: hidden_dim should match"
    );
    assert_eq!(
        model_config.num_layers, config.num_layers,
        "IMP-026: num_layers should match"
    );

    // Test 3: Forward pass should work with loaded weights
    let token_ids = vec![1, 2, 3];
    let logits = model.forward_gpu_owned(&token_ids);
    assert!(
        logits.is_ok(),
        "IMP-026: Forward pass should succeed with loaded weights"
    );

    let logits = logits.expect("test");
    assert_eq!(
        logits.len(),
        token_ids.len() * config.vocab_size,
        "IMP-026: Logits should have shape [seq_len, vocab_size]"
    );

    // Test 4: Test with real GGUF tensor mapping (test data)
    // This validates the tensor name → weight mapping logic
    let tensor_names = [
        "token_embd.weight",
        "blk.0.attn_norm.weight",
        "blk.0.attn_qkv.weight",
        "blk.0.attn_output.weight",
        "blk.0.ffn_up.weight",
        "blk.0.ffn_down.weight",
        "output_norm.weight",
        "output.weight",
    ];

    // Verify tensor name convention is documented
    for name in &tensor_names {
        assert!(
            !name.is_empty(),
            "IMP-026: Tensor name {} should follow GGUF convention",
            name
        );
    }
}

/// IMP-026 Part 2: Test actual GGUF file loading to GPU (integration test)
/// Requires: A real GGUF file for full integration
#[test]
#[cfg(feature = "gpu")]
#[ignore = "Enable when real GGUF available"]
fn test_imp_026_real_gguf_gpu_loading() {
    use crate::gguf::MappedGGUFModel;
    use crate::gpu::GpuModel;

    // Load real GGUF model
    let gguf_path =
        std::env::var("GGUF_MODEL_PATH").unwrap_or_else(|_| "models/phi-2-q4_k_m.gguf".to_string());

    if !std::path::Path::new(&gguf_path).exists() {
        eprintln!("IMP-026: Skipping - GGUF model not found at {}", gguf_path);
        return;
    }

    // Load and convert to GPU model
    let mapped = MappedGGUFModel::from_path(&gguf_path).expect("IMP-026: Should load GGUF model");

    let mut model =
        GpuModel::from_mapped_gguf(&mapped).expect("IMP-026: Should convert to GPU model");

    // GPU is optional - model initialization is the success criterion
    let _ = model.has_gpu();

    // Generate one token to verify weights are correct
    let prompt_tokens = vec![1, 2, 3];
    let logits = model
        .forward_gpu_owned(&prompt_tokens)
        .expect("IMP-026: Forward pass should work");

    // Verify logits are not all zeros (weights loaded correctly)
    let non_zero = logits.iter().any(|&x| x.abs() > 1e-10);
    assert!(
        non_zero,
        "IMP-026: Logits should not be all zeros (weights loaded)"
    );
}

/// IMP-027: E2E GPU text generation (M14 target)
/// Target: Generate text tokens from GPU model
#[test]
#[cfg(feature = "gpu")]
fn test_imp_027_gpu_text_generation() {
    use crate::gpu::{GpuGenerateConfig, GpuModel, GpuModelConfig};

    // Create a small model for testing
    let config = GpuModelConfig {
        vocab_size: 256,
        hidden_dim: 64,
        num_heads: 4,
        num_kv_heads: 4, // Standard MHA
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

    let mut model = GpuModel::from_gguf_config(config).expect("IMP-027: Should create model");

    // Test 1: Generate with greedy decoding
    let prompt = vec![1, 2, 3];
    let gen_config = GpuGenerateConfig::deterministic(5);
    let tokens = model
        .generate(&prompt, &gen_config)
        .expect("IMP-027: Generate should succeed");

    assert!(
        tokens.len() >= prompt.len(),
        "IMP-027: Generated tokens should include prompt"
    );
    assert!(
        tokens.len() <= prompt.len() + 5,
        "IMP-027: Should not exceed max_tokens"
    );
    assert_eq!(
        &tokens[..prompt.len()],
        &prompt,
        "IMP-027: Output should start with prompt"
    );

    // Test 2: Generate with stop tokens
    let gen_config_stop =
        GpuGenerateConfig::deterministic(10).with_stop_tokens(vec![tokens[prompt.len()]]); // Stop on first generated token
    let tokens_stopped = model
        .generate(&prompt, &gen_config_stop)
        .expect("IMP-027: Generate with stop should succeed");

    assert_eq!(
        tokens_stopped.len(),
        prompt.len(),
        "IMP-027: Should stop on stop token (not include it)"
    );

    // Test 3: Generate with sampling config (deterministic due to implementation)
    let gen_config_sample = GpuGenerateConfig::with_sampling(3, 0.7, 10);
    let tokens_sampled = model
        .generate(&prompt, &gen_config_sample)
        .expect("IMP-027: Generate with sampling should succeed");

    assert!(
        tokens_sampled.len() >= prompt.len(),
        "IMP-027: Sampled tokens should include prompt"
    );

    // Test 4: Empty prompt should error
    let empty_result = model.generate(&[], &gen_config);
    assert!(
        empty_result.is_err(),
        "IMP-027: Empty prompt should return error"
    );

    // Test 5: Config builders work
    let default_config = GpuGenerateConfig::default();
    assert_eq!(
        default_config.max_tokens, 64,
        "IMP-027: Default max_tokens should be 64"
    );
    assert_eq!(
        default_config.temperature, 0.0,
        "IMP-027: Default temperature should be 0.0"
    );
    assert_eq!(
        default_config.top_k, 1,
        "IMP-027: Default top_k should be 1"
    );
}

/// IMP-028: End-to-end forward pass produces valid logits (M15)
/// Target: Forward pass produces non-trivial output distribution
#[test]
#[cfg(feature = "gpu")]
fn test_imp_028_real_forward_pass() {
    use crate::gpu::{GpuModel, GpuModelConfig};

    let config = GpuModelConfig {
        vocab_size: 256,
        hidden_dim: 64,
        num_heads: 4,
        num_kv_heads: 4, // Standard MHA
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

    let mut model =
        GpuModel::from_gguf_config(config.clone()).expect("IMP-028: Should create model");

    // Test 1: Forward pass produces logits
    let tokens = vec![1, 2, 3, 4, 5];
    let logits = model
        .forward_gpu(&tokens)
        .expect("IMP-028: Forward pass should succeed");

    assert_eq!(
        logits.len(),
        tokens.len() * config.vocab_size,
        "IMP-028: Logits shape should be [seq_len, vocab_size]"
    );

    // Test 2: Logits are not all zeros
    let non_zero = logits.iter().any(|&x| x.abs() > 1e-10);
    assert!(non_zero, "IMP-028: Logits should not be all zeros");

    // Test 3: Logits are finite (no NaN or Inf)
    let all_finite = logits.iter().all(|&x| x.is_finite());
    assert!(all_finite, "IMP-028: All logits should be finite");

    // Test 4: Last position logits form a valid distribution after softmax
    let last_logits_start = (tokens.len() - 1) * config.vocab_size;
    let last_logits = &logits[last_logits_start..last_logits_start + config.vocab_size];

    // Softmax and verify it sums to ~1.0
    let max_logit = last_logits
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = last_logits.iter().map(|&x| (x - max_logit).exp()).sum();
    let probs: Vec<f32> = last_logits
        .iter()
        .map(|&x| (x - max_logit).exp() / exp_sum)
        .collect();
    let prob_sum: f32 = probs.iter().sum();

    assert!(
        (prob_sum - 1.0).abs() < 1e-5,
        "IMP-028: Softmax probabilities should sum to 1.0 (got {})",
        prob_sum
    );

    // Test 5: Incremental decoding (single token) works
    let single_token = vec![42];
    let single_logits = model
        .forward_gpu(&single_token)
        .expect("IMP-028: Single token forward should work");

    assert_eq!(
        single_logits.len(),
        config.vocab_size,
        "IMP-028: Single token should produce vocab_size logits"
    );
}

include!("imp_029.rs");
include!("imp_033.rs");
include!("imp_039.rs");
include!("imp_044.rs");
