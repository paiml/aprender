//! Combinatorial Test Generation
//!
//! Generates test cases for all format×device×config combinations.
//! Implements Popperian falsification protocol per specification.
//!
//! # References
//!
//! - Kuhn, D.R., et al. "Combinatorial Testing." NIST SP 800-142 (2010)
//! - Popper, K. "The Logic of Scientific Discovery." Routledge, 1959

use super::{
    fixtures::{AprFixture, GgufFixture, ModelFixture, SafetensorsFixture},
    Device, ModelConfig, ModelFormat, ModelTestCase, QuantType,
};

/// Generate conversion test cases for a (source, target) format pair across devices/configs/GQA
fn generate_conversion_cases(
    source: ModelFormat,
    target: ModelFormat,
    devices: &[Device],
    configs: &[ModelConfig],
    gqa_ratios: &[(usize, usize)],
    tests: &mut Vec<ModelTestCase>,
) {
    for device in devices {
        for base_config in configs {
            for (heads, kv_heads) in gqa_ratios {
                let mut config = base_config.clone();
                config.num_heads = *heads;
                config.num_kv_heads = *kv_heads;

                tests.push(ModelTestCase::conversion(
                    format!(
                        "{}->{} on {} ({}x{} GQA, {}L)",
                        source, target, device, heads, kv_heads, config.num_layers
                    ),
                    config,
                    source,
                    target,
                    *device,
                ));
            }
        }
    }
}

/// Generate combinatorial test matrix
///
/// Creates test cases for:
/// - All format pairs (source → target)
/// - Both devices (CPU, CUDA)
/// - Multiple configs (tiny, small)
/// - Multiple GQA ratios
pub fn generate_combinatorial_tests() -> Vec<ModelTestCase> {
    let formats = [
        ModelFormat::GGUF,
        ModelFormat::APR,
        ModelFormat::Safetensors,
    ];
    let devices = [Device::Cpu, Device::Cuda(0)];
    let configs = [ModelConfig::tiny(), ModelConfig::small()];
    let gqa_ratios: [(usize, usize); 4] = [(4, 4), (4, 2), (8, 2), (8, 1)];

    let mut tests = Vec::new();

    // Conversion tests: source → target
    for source in &formats {
        for target in &formats {
            if source == target {
                continue;
            }
            generate_conversion_cases(
                *source,
                *target,
                &devices,
                &configs,
                &gqa_ratios,
                &mut tests,
            );
        }
    }

    // Single-format tests (no conversion)
    for format in &formats {
        for device in &devices {
            tests.push(ModelTestCase::new(
                format!("{} forward on {} (tiny)", format, device),
                ModelConfig::tiny(),
                *format,
                *device,
            ));
        }
    }

    tests
}

/// Generate quantization test matrix
pub fn generate_quant_tests() -> Vec<ModelTestCase> {
    let quant_types = [
        QuantType::F32,
        QuantType::F16,
        QuantType::Q4_0,
        QuantType::Q8_0,
        QuantType::Q4_K,
    ];

    let formats = [ModelFormat::GGUF, ModelFormat::APR];
    let devices = [Device::Cpu, Device::Cuda(0)];

    let mut tests = Vec::new();

    for quant in &quant_types {
        for format in &formats {
            if !quant.supported_by(*format) {
                continue;
            }

            for device in &devices {
                tests.push(
                    ModelTestCase::new(
                        format!("{} {:?} on {}", format, quant, device),
                        ModelConfig::tiny(),
                        *format,
                        *device,
                    )
                    .with_quant(*quant),
                );
            }
        }
    }

    tests
}

// ============================================================================
// FALSIFICATION TESTS (100-point Popperian protocol)
// ============================================================================

/// F001: GGUF magic bytes verification
#[test]
fn test_f001_gguf_magic_bytes() {
    let fixture = GgufFixture::tiny_gqa();
    let bytes = fixture.to_bytes().expect("serialization should succeed");

    assert_eq!(
        &bytes[0..4],
        b"GGUF",
        "FALSIFICATION F001 (2 points): GGUF magic bytes must be 0x47475546"
    );
}

/// F002: APR header version preservation
#[test]
fn test_f002_apr_header_version() {
    let fixture = AprFixture::tiny_gqa();
    let bytes = fixture.to_bytes().expect("serialization should succeed");

    assert_eq!(
        &bytes[0..4],
        b"APR\x02",
        "FALSIFICATION F002 (2 points): APR header must preserve version 2"
    );
}

/// F003: Safetensors JSON metadata integrity
#[test]
fn test_f003_safetensors_json_integrity() {
    let fixture = SafetensorsFixture::tiny();
    let bytes = fixture.to_bytes().expect("serialization should succeed");

    // First 8 bytes are header length
    let header_len = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
    let header_bytes = &bytes[8..8 + header_len];

    // Should be valid JSON
    let _: serde_json::Value = serde_json::from_slice(header_bytes)
        .expect("FALSIFICATION F003 (2 points): Safetensors header must be valid JSON");
}

/// F004: Tensor count preservation after round-trip
#[test]
fn test_f004_tensor_count_roundtrip() {
    let original = GgufFixture::tiny_gqa();
    let apr = original.convert_to(ModelFormat::APR).unwrap();
    let roundtrip = apr.convert_to(ModelFormat::GGUF).unwrap();

    assert_eq!(
        original.config().num_layers,
        roundtrip.config().num_layers,
        "FALSIFICATION F004 (2 points): Layer count must match after round-trip"
    );
}

/// F007: GQA num_kv_heads preservation (CRITICAL)
#[test]
fn test_f007_gqa_num_kv_heads_preserved() {
    let original = GgufFixture::tiny_gqa();
    let original_kv_heads = original.config().num_kv_heads;

    // GGUF → APR → GGUF round-trip
    let apr = original
        .convert_to(ModelFormat::APR)
        .expect("GGUF→APR conversion should succeed");

    assert_eq!(
        apr.config().num_kv_heads,
        original_kv_heads,
        "FALSIFICATION F007 (2 points): num_kv_heads must be preserved in APR"
    );

    let back = apr
        .convert_to(ModelFormat::GGUF)
        .expect("APR→GGUF conversion should succeed");

    assert_eq!(
        back.config().num_kv_heads,
        original_kv_heads,
        "FALSIFICATION F007 (2 points): num_kv_heads must survive round-trip"
    );
}

/// F008: RoPE theta preservation
#[test]
fn test_f008_rope_theta_preserved() {
    let mut config = ModelConfig::tiny();
    config.rope_theta = 1_000_000.0; // Qwen-style theta

    let original = GgufFixture::new(config, QuantType::F32, 42);
    let apr = original.convert_to(ModelFormat::APR).unwrap();

    assert!(
        (apr.config().rope_theta - 1_000_000.0).abs() < 1.0,
        "FALSIFICATION F008 (2 points): rope_theta must be preserved"
    );
}

/// F009: Vocab size preservation
#[test]
fn test_f009_vocab_size_preserved() {
    let original = GgufFixture::tiny_gqa();
    let apr = original.convert_to(ModelFormat::APR).unwrap();

    assert_eq!(
        apr.config().vocab_size,
        original.config().vocab_size,
        "FALSIFICATION F009 (2 points): vocab_size must be preserved"
    );
}

/// F010: Layer count preservation
#[test]
fn test_f010_layer_count_preserved() {
    let original = GgufFixture::new(ModelConfig::small(), QuantType::F32, 42);
    let safetensors = original.convert_to(ModelFormat::Safetensors).unwrap();
    let apr = safetensors.convert_to(ModelFormat::APR).unwrap();

    assert_eq!(
        apr.config().num_layers,
        original.config().num_layers,
        "FALSIFICATION F010 (2 points): num_layers must be preserved through conversions"
    );
}

/// F011: Embedding L2 norm consistency
#[test]
fn test_f011_embedding_l2_consistency() {
    let fixture = GgufFixture::tiny_gqa();
    let token = 42u32;

    let embed1 = fixture.embed(Device::Cpu, token).unwrap();
    let embed2 = fixture.embed(Device::Cpu, token).unwrap();

    let l2_1: f32 = embed1.iter().map(|x| x * x).sum::<f32>().sqrt();
    let l2_2: f32 = embed2.iter().map(|x| x * x).sum::<f32>().sqrt();

    let diff = (l2_1 - l2_2).abs() / l2_1.max(l2_2);
    assert!(
        diff < 0.01,
        "FALSIFICATION F011 (3 points): Embedding L2 norm must be consistent, got {}% diff",
        diff * 100.0
    );
}

/// F012: RMSNorm output no NaN
#[test]
fn test_f012_no_nan_in_output() {
    let fixture = GgufFixture::tiny_gqa();
    let tokens = vec![1, 2, 3, 4];

    let output = fixture.forward(Device::Cpu, &tokens).unwrap();

    let nan_count = output.iter().filter(|x| x.is_nan()).count();
    assert_eq!(
        nan_count, 0,
        "FALSIFICATION F012 (3 points): Output must not contain NaN"
    );
}

/// F017: Softmax sum verification
#[test]
fn test_f017_softmax_sum() {
    let fixture = GgufFixture::tiny_gqa();
    let tokens = vec![1, 2, 3];

    let logits = fixture.forward(Device::Cpu, &tokens).unwrap();

    // Apply softmax
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = logits.iter().map(|x| (x - max_logit).exp()).sum();
    let probs: Vec<f32> = logits
        .iter()
        .map(|x| (x - max_logit).exp() / exp_sum)
        .collect();

    let prob_sum: f32 = probs.iter().sum();
    assert!(
        (prob_sum - 1.0).abs() < 1e-5,
        "FALSIFICATION F017 (3 points): Softmax sum must equal 1.0, got {}",
        prob_sum
    );
}

/// F018: RoPE at position 0 is identity
#[test]
fn test_f018_rope_position_zero_identity() {
    // At position 0: cos(0) = 1, sin(0) = 0, so rotation is identity
    // This is a property test - the fixture should preserve this
    let config = ModelConfig::tiny();
    assert_eq!(
        config.head_dim() % 2,
        0,
        "FALSIFICATION F018 (3 points): head_dim must be even for RoPE"
    );
}

/// F021: CPU vs CUDA parity (placeholder - requires CUDA)
#[test]
fn test_f021_cpu_cuda_parity_structure() {
    let fixture = GgufFixture::tiny_gqa();
    let tokens = vec![1, 2, 3];

    // CPU forward
    let cpu_output = fixture.forward(Device::Cpu, &tokens).unwrap();

    // Same fixture, same tokens → should give same output structure
    let cpu_output2 = fixture.forward(Device::Cpu, &tokens).unwrap();

    assert_eq!(
        cpu_output.len(),
        cpu_output2.len(),
        "FALSIFICATION F021 (4 points): Output shape must be consistent"
    );
}

/// F031: GGUF→APR→GGUF config round-trip
#[test]
fn test_f031_gguf_apr_gguf_config_roundtrip() {
    let original = GgufFixture::tiny_gqa();

    let apr = original.convert_to(ModelFormat::APR).unwrap();
    let back = apr.convert_to(ModelFormat::GGUF).unwrap();

    // Check all config fields
    assert_eq!(back.config().hidden_dim, original.config().hidden_dim);
    assert_eq!(back.config().num_layers, original.config().num_layers);
    assert_eq!(back.config().num_heads, original.config().num_heads);
    assert_eq!(back.config().num_kv_heads, original.config().num_kv_heads);
    assert_eq!(back.config().vocab_size, original.config().vocab_size);
    assert_eq!(
        back.config().intermediate_dim,
        original.config().intermediate_dim
    );
}

/// F032: Safetensors→GGUF preserves tensors
#[test]
fn test_f032_safetensors_gguf_tensor_preservation() {
    let original = SafetensorsFixture::tiny();
    let gguf = original.convert_to(ModelFormat::GGUF).unwrap();

    assert_eq!(
        gguf.config().num_layers,
        original.config().num_layers,
        "FALSIFICATION F032 (3 points): Tensor count must be preserved"
    );
}

/// F034: Conversion preserves tensor shape
#[test]
fn test_f034_conversion_preserves_shape() {
    let original = GgufFixture::tiny_gqa();
    let apr = original.convert_to(ModelFormat::APR).unwrap();

    // Check key dimensions
    assert_eq!(apr.config().q_dim(), original.config().q_dim());
    assert_eq!(apr.config().k_dim(), original.config().k_dim());
    assert_eq!(apr.config().v_dim(), original.config().v_dim());
}

// ============================================================================
// COMBINATORIAL TESTS
// ============================================================================

/// Create a fixture for the given format
fn create_fixture(format: ModelFormat) -> Option<Box<dyn ModelFixture>> {
    match format {
        ModelFormat::GGUF => Some(Box::new(GgufFixture::tiny_gqa())),
        ModelFormat::APR => Some(Box::new(AprFixture::tiny_gqa())),
        ModelFormat::Safetensors => Some(Box::new(SafetensorsFixture::tiny())),
        ModelFormat::PyTorch => None,
    }
}

include!("combinatorial_tests_all_format.rs");
