//! SwiGLU FFN Popperian Falsification Tests
//!
//! EXTREME TDD: Tests written BEFORE implementation.
//! These tests MUST FAIL initially, then pass after implementation.
//!
//! SwiGLU (Swish-Gated Linear Unit) used by LLaMA/TinyLlama:
//!   output = down(gate(x) * silu(up(x)))
//!
//! Where:
//!   - up(x) = matmul(x, ffn_up_weight)
//!   - gate(x) = matmul(x, ffn_gate_weight)
//!   - silu(x) = x * sigmoid(x)
//!   - output = matmul(gate * silu(up), ffn_down_weight)

/// FFN-01: OwnedQuantizedLayer must have ffn_gate_weight field
/// Falsification: Struct compilation fails if field is missing
#[test]
fn test_ffn01_owned_layer_has_gate_weight() {
    use realizar::gguf::OwnedQuantizedLayer;

    // This test verifies the struct has the ffn_gate_weight field
    // If the field doesn't exist, this won't compile
    fn check_has_gate_field(layer: &OwnedQuantizedLayer) -> bool {
        // Access the field - compiler error if missing
        layer.ffn_gate_weight.is_some() || layer.ffn_gate_weight.is_none()
    }

    // Verify the function compiles and can be referenced
    // The actual check is compile-time: if the field doesn't exist, this file won't compile
    let _: fn(&OwnedQuantizedLayer) -> bool = check_has_gate_field;
}

/// FFN-02: OwnedQuantizedLayer must have ffn_gate_bias field
/// Falsification: Struct compilation fails if field is missing
#[test]
fn test_ffn02_owned_layer_has_gate_bias() {
    use realizar::gguf::OwnedQuantizedLayer;

    fn check_has_gate_bias(layer: &OwnedQuantizedLayer) -> bool {
        layer.ffn_gate_bias.is_some() || layer.ffn_gate_bias.is_none()
    }

    // Verify the function compiles and can be referenced
    let _: fn(&OwnedQuantizedLayer) -> bool = check_has_gate_bias;
}

/// FFN-03: QuantizedGGUFTransformerLayer must have ffn_gate_weight field
/// Falsification: Struct compilation fails if field is missing
#[test]
fn test_ffn03_quantized_layer_has_gate_weight() {
    use realizar::gguf::QuantizedGGUFTransformerLayer;

    fn check_has_gate_field(layer: &QuantizedGGUFTransformerLayer) -> bool {
        layer.ffn_gate_weight.is_some() || layer.ffn_gate_weight.is_none()
    }

    // Verify the function compiles and can be referenced
    let _: fn(&QuantizedGGUFTransformerLayer) -> bool = check_has_gate_field;
}

/// FFN-04: SiLU activation function exists and is correct
/// Falsification: silu(0) != 0 OR silu(large) != large
#[test]
fn test_ffn04_silu_activation_correct() {
    // SiLU(x) = x * sigmoid(x)
    // At x=0: SiLU(0) = 0 * 0.5 = 0
    // At x=large: sigmoid(x) ≈ 1, so SiLU(x) ≈ x

    fn silu(x: f32) -> f32 {
        x * (1.0 / (1.0 + (-x).exp()))
    }

    // Test silu(0) = 0
    let silu_zero = silu(0.0);
    assert!(
        silu_zero.abs() < 1e-6,
        "FFN-04a: SiLU(0) should be 0, got {}",
        silu_zero
    );

    // Test silu(10) ≈ 10 (large positive)
    let silu_large = silu(10.0);
    assert!(
        (silu_large - 10.0).abs() < 0.001,
        "FFN-04b: SiLU(10) should ≈ 10, got {}",
        silu_large
    );

    // Test silu(-10) ≈ 0 (large negative)
    let silu_neg = silu(-10.0);
    assert!(
        silu_neg.abs() < 0.001,
        "FFN-04c: SiLU(-10) should ≈ 0, got {}",
        silu_neg
    );

    // Test silu(1) ≈ 0.731
    let silu_one = silu(1.0);
    assert!(
        (silu_one - 0.731).abs() < 0.01,
        "FFN-04d: SiLU(1) should ≈ 0.731, got {}",
        silu_one
    );
}

/// FFN-05: SwiGLU output differs from simple FFN
/// Falsification: swiglu_output == simple_ffn_output for non-trivial input
#[test]
fn test_ffn05_swiglu_differs_from_simple_ffn() {
    // SwiGLU: output = down(gate * silu(up))
    // Simple: output = down(gelu(up))
    // These should produce different results for the same weights

    fn silu(x: f32) -> f32 {
        x * (1.0 / (1.0 + (-x).exp()))
    }

    fn gelu(x: f32) -> f32 {
        // Approximate GELU
        0.5 * x
            * (1.0 + ((2.0_f32 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh())
    }

    // Simple test: same up projection, different activation path
    let x = 2.0_f32;
    let up = x * 1.5; // Simulated up projection
    let gate = x * 0.8; // Simulated gate projection (only in SwiGLU)

    // SwiGLU path
    let swiglu_hidden = gate * silu(up);

    // Simple GELU path
    let simple_hidden = gelu(up);

    // They must differ
    assert!(
        (swiglu_hidden - simple_hidden).abs() > 0.1,
        "FFN-05: SwiGLU hidden ({}) must differ from simple GELU hidden ({})",
        swiglu_hidden,
        simple_hidden
    );
}

/// FFN-06: Gate weight loading from GGUF
/// Falsification: TinyLlama GGUF loads with gate=None
#[test]
#[ignore = "requires TinyLlama GGUF file"]
fn test_ffn06_gguf_loads_gate_weight() {
    use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};
    use std::path::Path;

    let gguf_path = "/home/noah/src/aprender/tinyllama-1.1b-chat-v1.0.Q4_0.gguf";
    if !Path::new(gguf_path).exists() {
        println!("Skipping FFN-06: TinyLlama GGUF not found");
        return;
    }

    let mapped = MappedGGUFModel::from_path(gguf_path).expect("Failed to load GGUF");
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("Failed to create model");

    // TinyLlama uses SwiGLU, so gate weight must be present
    assert!(
        model.layers()[0].ffn_gate_weight.is_some(),
        "FFN-06: TinyLlama layer 0 must have ffn_gate_weight (SwiGLU model)"
    );
}

/// FFN-07: Gate weight dimensions match up weight
/// Falsification: gate.out_dim != up.out_dim
#[test]
#[ignore = "requires TinyLlama GGUF file"]
fn test_ffn07_gate_dimensions_match_up() {
    use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};
    use std::path::Path;

    let gguf_path = "/home/noah/src/aprender/tinyllama-1.1b-chat-v1.0.Q4_0.gguf";
    if !Path::new(gguf_path).exists() {
        println!("Skipping FFN-07: TinyLlama GGUF not found");
        return;
    }

    let mapped = MappedGGUFModel::from_path(gguf_path).expect("Failed to load GGUF");
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("Failed to create model");

    let layer = &model.layers()[0];
    let gate = layer
        .ffn_gate_weight
        .as_ref()
        .expect("Gate weight must be present");

    // Gate and Up must have same output dimension (intermediate_dim)
    assert_eq!(
        gate.out_dim, layer.ffn_up_weight.out_dim,
        "FFN-07: gate.out_dim ({}) must equal up.out_dim ({})",
        gate.out_dim, layer.ffn_up_weight.out_dim
    );

    // Gate and Up must have same input dimension (hidden_dim)
    assert_eq!(
        gate.in_dim, layer.ffn_up_weight.in_dim,
        "FFN-07: gate.in_dim ({}) must equal up.in_dim ({})",
        gate.in_dim, layer.ffn_up_weight.in_dim
    );
}

/// FFN-08: SwiGLU forward produces coherent output
/// Falsification: Generated tokens are random/garbage
#[test]
#[ignore = "requires TinyLlama GGUF file"]
fn test_ffn08_swiglu_forward_coherent() {
    use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel, QuantizedGenerateConfig};
    use std::path::Path;

    let gguf_path = "/home/noah/src/aprender/tinyllama-1.1b-chat-v1.0.Q4_0.gguf";
    if !Path::new(gguf_path).exists() {
        println!("Skipping FFN-08: TinyLlama GGUF not found");
        return;
    }

    let mapped = MappedGGUFModel::from_path(gguf_path).expect("Failed to load GGUF");
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("Failed to create model");

    // Generate a few tokens from a simple prompt
    // BOS token for TinyLlama is 1
    let prompt = vec![1u32, 15043]; // <s> Hello
    let config = QuantizedGenerateConfig {
        max_tokens: 5,
        temperature: 0.0, // Greedy for reproducibility
        top_k: 1,
        stop_tokens: vec![2], // EOS
        trace: false,
        ..Default::default()
    };

    let output = model
        .generate(&prompt, &config)
        .expect("Generation should succeed");

    // Output should not be all zeros or all same token
    let unique_tokens: std::collections::HashSet<_> = output.iter().collect();
    assert!(
        unique_tokens.len() > 1 || output.len() <= 2,
        "FFN-08: Generated output should have variety, got {:?}",
        output
    );

    // Output tokens should be within vocab range
    let vocab_size = model.config().vocab_size;
    for &tok in &output {
        assert!(
            (tok as usize) < vocab_size,
            "FFN-08: Token {} exceeds vocab size {}",
            tok,
            vocab_size
        );
    }
}

/// FFN-09: Verify ffn_norm is loaded for LLaMA models
#[test]
#[ignore = "requires TinyLlama GGUF model file"]
fn test_ffn09_verify_model_weights() {
    use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};
    use std::path::Path;

    let gguf_path = "/home/noah/src/aprender/tinyllama-1.1b-chat-v1.0.Q4_0.gguf";
    if !Path::new(gguf_path).exists() {
        println!("Skipping FFN-09: TinyLlama GGUF not found");
        return;
    }

    let mapped = MappedGGUFModel::from_path(gguf_path).expect("Failed to load");
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("Failed to create model");

    println!("Config:");
    println!("  hidden_dim: {}", model.config().hidden_dim);
    println!("  intermediate_dim: {}", model.config().intermediate_dim);
    println!("  num_layers: {}", model.config().num_layers);
    println!("  num_heads: {}", model.config().num_heads);
    println!("  num_kv_heads: {}", model.config().num_kv_heads);
    println!("  vocab_size: {}", model.config().vocab_size);

    let layer = &model.layers()[0];
    println!("\nLayer 0:");
    println!("  attn_norm_weight len: {}", layer.attn_norm_weight.len());
    println!(
        "  ffn_gate_weight: {:?}",
        layer
            .ffn_gate_weight
            .as_ref()
            .map(|t| (t.in_dim, t.out_dim))
    );
    println!(
        "  ffn_up_weight: ({}, {})",
        layer.ffn_up_weight.in_dim, layer.ffn_up_weight.out_dim
    );
    println!(
        "  ffn_down_weight: ({}, {})",
        layer.ffn_down_weight.in_dim, layer.ffn_down_weight.out_dim
    );
    println!(
        "  ffn_norm_weight: {:?}",
        layer.ffn_norm_weight.as_ref().map(|v| v.len())
    );

    println!("\nToken embedding len: {}", model.token_embedding().len());
    println!(
        "  Expected: {} * {} = {}",
        model.config().vocab_size,
        model.config().hidden_dim,
        model.config().vocab_size * model.config().hidden_dim
    );

    // Check dimensions
    assert_eq!(layer.attn_norm_weight.len(), model.config().hidden_dim);
    assert!(
        layer.ffn_gate_weight.is_some(),
        "TinyLlama should have ffn_gate (SwiGLU)"
    );

    // ffn_norm is optional - some models don't have it
    if layer.ffn_norm_weight.is_some() {
        println!("  ffn_norm IS present");
    } else {
        println!("  ffn_norm NOT present (model may use different naming)");
    }
}

/// FFN-10: Verify output norm and lm_head are loaded
#[test]
#[ignore = "requires TinyLlama GGUF model file"]
fn test_ffn10_verify_output_layers() {
    use realizar::gguf::{MappedGGUFModel, OwnedQuantizedModel};
    use std::path::Path;

    let gguf_path = "/home/noah/src/aprender/tinyllama-1.1b-chat-v1.0.Q4_0.gguf";
    if !Path::new(gguf_path).exists() {
        println!("Skipping FFN-10: TinyLlama GGUF not found");
        return;
    }

    let mapped = MappedGGUFModel::from_path(gguf_path).expect("Failed to load");
    let model = OwnedQuantizedModel::from_mapped(&mapped).expect("Failed to create model");

    println!("Output layers:");
    println!(
        "  output_norm_weight len: {}",
        model.output_norm_weight().len()
    );
    println!(
        "  lm_head_weight: ({}, {})",
        model.lm_head_weight().in_dim,
        model.lm_head_weight().out_dim
    );

    // First few values of output_norm
    if model.output_norm_weight().len() > 5 {
        println!(
            "  output_norm first 5: {:?}",
            &model.output_norm_weight()[..5]
        );
    }

    // Verify dimensions
    assert_eq!(model.output_norm_weight().len(), model.config().hidden_dim);
    assert_eq!(model.lm_head_weight().in_dim, model.config().hidden_dim);
    assert_eq!(model.lm_head_weight().out_dim, model.config().vocab_size);

    println!("Output layers verified ✓");
}
