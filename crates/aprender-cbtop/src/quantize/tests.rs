use super::*;

#[test]
fn test_quant_format_bits() {
    assert_eq!(QuantFormat::F32.bits_per_weight(), 32.0);
    assert_eq!(QuantFormat::F16.bits_per_weight(), 16.0);
    assert_eq!(QuantFormat::Q4_K.bits_per_weight(), 4.5);
    assert_eq!(QuantFormat::Q8_0.bits_per_weight(), 8.5);
}

#[test]
fn test_quant_format_memory_ratio() {
    // Q4_K should use ~28% of F16 memory
    let ratio = QuantFormat::Q4_K.memory_ratio();
    assert!(ratio < 0.3);
    assert!(ratio > 0.25);
}

#[test]
fn test_quant_format_display() {
    assert_eq!(format!("{}", QuantFormat::Q4_K), "Q4_K");
    assert_eq!(format!("{}", QuantFormat::Q8_0), "Q8_0");
    assert_eq!(
        format!(
            "{}",
            QuantFormat::Gptq {
                bits: 4,
                group_size: 128
            }
        ),
        "GPTQ-4bit-g128"
    );
}

#[test]
fn test_quantized_weights() {
    let weights = QuantizedWeights::new(
        QuantFormat::Q4_K,
        vec![0u8; 144 * 16], // 16 super-blocks
        (4096, 4096),
        "model.layers.0.self_attn.q_proj",
    );

    assert_eq!(weights.num_weights(), 4096 * 4096);
    assert_eq!(weights.memory_bytes(), 144 * 16);
    assert!(weights.compression_ratio() > 10.0); // Should be ~14x for Q4_K
}

#[test]
fn test_quant_stats() {
    let mut stats = QuantStats::new();

    let weights1 = QuantizedWeights::new(QuantFormat::Q4_K, vec![0u8; 1000], (100, 100), "layer1");
    let weights2 = QuantizedWeights::new(QuantFormat::Q8_0, vec![0u8; 2000], (100, 100), "layer2");

    stats.add_layer(&weights1);
    stats.add_layer(&weights2);

    assert_eq!(stats.total_weights, 20000);
    assert_eq!(stats.total_memory_bytes, 3000);
    assert_eq!(stats.layer_stats.len(), 2);
}

#[test]
fn test_gguf_header_parse() {
    let mut loader = GgufLoader::new("/tmp/test.gguf");

    // Valid GGUF header (version 3)
    let mut data = vec![0u8; 24];
    data[0..4].copy_from_slice(b"GGUF");
    data[4..8].copy_from_slice(&3u32.to_le_bytes());
    data[8..16].copy_from_slice(&100u64.to_le_bytes());
    data[16..24].copy_from_slice(&50u64.to_le_bytes());

    loader.parse_header(&data).unwrap();

    let header = loader.header().unwrap();
    assert_eq!(header.version, 3);
    assert_eq!(header.tensor_count, 100);
    assert_eq!(header.metadata_kv_count, 50);
}

#[test]
fn test_gguf_invalid_magic() {
    let mut loader = GgufLoader::new("/tmp/test.gguf");

    let mut data = vec![0u8; 24];
    data[0..4].copy_from_slice(b"XXXX");

    let result = loader.parse_header(&data);
    assert!(matches!(result, Err(GgufError::InvalidMagic(_))));
}

#[test]
fn test_quantized_brick() {
    let weights =
        QuantizedWeights::new(QuantFormat::Q4_K, vec![0u8; 144], (256, 256), "test_layer");

    let brick = QuantizedBrick::new("matmul")
        .with_weights(weights)
        .with_dequant_strategy(DequantStrategy::Fused)
        .with_budget(50_000);

    assert!(brick.has_weights());
    assert_eq!(brick.format(), Some(QuantFormat::Q4_K));
    assert_eq!(brick.memory_bytes(), 144);
}

#[test]
fn test_dequant_strategy_default() {
    let strategy = DequantStrategy::default();
    assert_eq!(strategy, DequantStrategy::Fused);
}

#[test]
fn test_ggml_type_mapping() {
    assert_eq!(ggml_type_to_format(0), Some(QuantFormat::F32));
    assert_eq!(ggml_type_to_format(1), Some(QuantFormat::F16));
    assert_eq!(ggml_type_to_format(12), Some(QuantFormat::Q4_K));
    assert_eq!(ggml_type_to_format(255), None);
}

#[test]
fn test_block_sizes() {
    assert_eq!(QuantFormat::Q4_0.block_size(), 32);
    assert_eq!(QuantFormat::Q4_K.block_size(), 256);
    assert_eq!(QuantFormat::Q8_0.block_size(), 32);
}

#[test]
fn test_bytes_per_block() {
    assert_eq!(QuantFormat::Q4_0.bytes_per_block(), 18);
    assert_eq!(QuantFormat::Q4_K.bytes_per_block(), 144);
    assert_eq!(QuantFormat::Q5_K.bytes_per_block(), 176);
    assert_eq!(QuantFormat::Q8_0.bytes_per_block(), 34);
}

#[test]
fn test_expected_ppl_delta() {
    // Q4_K should have lower perplexity delta than Q4_0
    assert!(QuantFormat::Q4_K.expected_ppl_delta() < QuantFormat::Q4_0.expected_ppl_delta());
    // Q8_0 should have very low perplexity delta
    assert!(QuantFormat::Q8_0.expected_ppl_delta() < 0.1);
}
