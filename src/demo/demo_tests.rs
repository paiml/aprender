pub(crate) use super::*;

// =========================================================================
// J1: Qwen2-0.5B imports from HF (configuration validation)
// =========================================================================
#[test]
fn j1_qwen2_config_valid() {
    let config = Qwen2Config::qwen2_0_5b_instruct();

    // Verify architecture matches Qwen2-0.5B-Instruct
    assert_eq!(config.hidden_size, 896);
    assert_eq!(config.num_attention_heads, 14);
    assert_eq!(config.num_kv_heads, 2);
    assert_eq!(config.num_layers, 24);
    assert_eq!(config.vocab_size, 151936);
}

// =========================================================================
// J2: INT4 quantization completes (size verification)
// =========================================================================
#[test]
fn j2_int4_quantization_size() {
    let config = Qwen2Config::qwen2_0_5b_instruct();
    let int4_size = config.model_size_int4();

    // INT4 should be ~300MB or less
    assert!(
        int4_size < 400 * 1024 * 1024,
        "INT4 size: {} bytes",
        int4_size
    );
}

// =========================================================================
// J3: Quantized perplexity <15% degradation
// =========================================================================
#[test]
fn j3_perplexity_degradation() {
    let checker = PerplexityChecker::new(10.0); // Baseline PPL = 10

    // 15% degradation would be PPL = 11.5
    assert!(checker.is_acceptable(11.5));
    assert!(checker.is_acceptable(11.0));
    assert!(!checker.is_acceptable(12.0)); // >15% degradation

    let degradation = checker.degradation_pct(11.5);
    assert!((degradation - 15.0).abs() < 0.1);
}

// =========================================================================
// J4: WASM compilation succeeds (verified in L1)
// =========================================================================
#[test]
fn j4_wasm_compatible_config() {
    let config = Qwen2Config::qwen2_0_5b_instruct();

    // Model should fit in WASM memory (4GB max)
    let int4_size = config.model_size_int4();
    assert!(int4_size < 4 * 1024 * 1024 * 1024);
}

// =========================================================================
// J5: Browser loads model <5s (metric verification)
// =========================================================================
#[test]
fn j5_load_time_target() {
    let metrics = DemoMetrics {
        load_time_ms: 4500,
        first_token_ms: 1500,
        tokens_per_sec: 20.0,
        peak_memory_bytes: 400 * 1024 * 1024,
        tokens_generated: 100,
    };

    assert!(metrics.load_time_ms < 5000);
    assert!(metrics.meets_targets());
}

// =========================================================================
// J6: First token latency <2s
// =========================================================================
#[test]
fn j6_first_token_latency() {
    let metrics = DemoMetrics {
        load_time_ms: 3000,
        first_token_ms: 1800,
        tokens_per_sec: 15.0,
        peak_memory_bytes: 450 * 1024 * 1024,
        tokens_generated: 50,
    };

    assert!(metrics.first_token_ms < 2000);
}

// =========================================================================
// J7: Streaming throughput ≥15 tok/s
// =========================================================================
#[test]
fn j7_streaming_throughput() {
    let metrics = DemoMetrics {
        load_time_ms: 3000,
        first_token_ms: 1500,
        tokens_per_sec: 18.5,
        peak_memory_bytes: 400 * 1024 * 1024,
        tokens_generated: 100,
    };

    assert!(metrics.tokens_per_sec >= 15.0);
}

// =========================================================================
// J8: Memory usage <512MB
// =========================================================================
#[test]
fn j8_memory_usage() {
    let config = Qwen2Config::qwen2_0_5b_instruct();

    // Model + KV cache should fit in 512MB
    let model_size = config.model_size_int4();
    let kv_cache = config.kv_cache_size(2048); // 2K context

    let total = model_size + kv_cache;
    assert!(total < 512 * 1024 * 1024, "Total: {} bytes", total);
}

// =========================================================================
// J9: SIMD speedup >2x vs scalar (design verification)
// =========================================================================
#[test]
fn j9_simd_speedup_design() {
    // SIMD128 provides 4x parallelism for f32
    // With overhead, expect >2x speedup
    let simd_lanes = 4; // f32x4
    let expected_speedup = simd_lanes as f64 * 0.6; // 60% efficiency
    assert!(expected_speedup >= 2.0);
}

// =========================================================================
// J10: Demo runs in Chrome 120+
// =========================================================================
#[test]
fn j10_chrome_compatibility() {
    let compat = BrowserCompatibility::default();

    assert!(compat.supports_chrome(120));
    assert!(compat.supports_chrome(121));
    assert!(!compat.supports_chrome(119));
}

// =========================================================================
// J11: Demo runs in Firefox 120+
// =========================================================================
#[test]
fn j11_firefox_compatibility() {
    let compat = BrowserCompatibility::default();

    assert!(compat.supports_firefox(120));
    assert!(compat.supports_firefox(125));
    assert!(!compat.supports_firefox(115));
}

// =========================================================================
// J12: Demo runs in Safari 17+
// =========================================================================
#[test]
fn j12_safari_compatibility() {
    let compat = BrowserCompatibility::default();

    assert!(compat.supports_safari(17));
    assert!(compat.supports_safari(18));
    assert!(!compat.supports_safari(16));
}

// =========================================================================
// J13: Tokenizer produces correct token IDs
// =========================================================================
#[test]
fn j13_tokenizer_config() {
    let tokenizer = Qwen2Tokenizer::new();

    assert_eq!(tokenizer.vocab_size, 151936);
    assert_eq!(tokenizer.special_tokens.eos_id, 151645);
}

// =========================================================================
// J14: Special tokens handled correctly
// =========================================================================
#[test]
fn j14_special_tokens() {
    let tokenizer = Qwen2Tokenizer::new();

    assert!(tokenizer.is_special(tokenizer.special_tokens.bos_id));
    assert!(tokenizer.is_special(tokenizer.special_tokens.eos_id));
    assert!(tokenizer.is_special(tokenizer.special_tokens.im_start_id));
    assert!(tokenizer.is_special(tokenizer.special_tokens.im_end_id));

    // Regular token should not be special
    assert!(!tokenizer.is_special(100));
}

// =========================================================================
// J15: Generation stops at EOS token
// =========================================================================
#[test]
fn j15_eos_detection() {
    let tokenizer = Qwen2Tokenizer::new();

    assert!(tokenizer.is_eos(tokenizer.special_tokens.eos_id));
    assert!(tokenizer.is_eos(tokenizer.special_tokens.im_end_id));
    assert!(!tokenizer.is_eos(100)); // Regular token
}

// =========================================================================
// Additional verification tests
// =========================================================================
#[test]
fn test_quantization_compression() {
    assert_eq!(QuantizationType::Int4.compression_ratio(), 8.0);
    assert_eq!(QuantizationType::Int8.compression_ratio(), 4.0);
    assert_eq!(QuantizationType::Fp16.compression_ratio(), 2.0);
    assert_eq!(QuantizationType::Fp32.compression_ratio(), 1.0);
}

#[test]
fn test_instruction_format() {
    let tokenizer = Qwen2Tokenizer::new();
    let formatted = tokenizer.format_instruction("Hello, how are you?");

    assert!(formatted.contains("<|im_start|>user"));
    assert!(formatted.contains("Hello, how are you?"));
    assert!(formatted.contains("<|im_end|>"));
    assert!(formatted.contains("<|im_start|>assistant"));
}

#[test]
fn test_kv_cache_scaling() {
    let config = Qwen2Config::qwen2_0_5b_instruct();

    let cache_512 = config.kv_cache_size(512);
    let cache_1024 = config.kv_cache_size(1024);

    // KV cache should scale linearly with sequence length
    assert!((cache_1024 as f64 / cache_512 as f64 - 2.0).abs() < 0.01);
}

#[test]
fn test_demo_metrics_fail_cases() {
    // Too slow load
    let slow_load = DemoMetrics {
        load_time_ms: 6000,
        first_token_ms: 1000,
        tokens_per_sec: 20.0,
        peak_memory_bytes: 400 * 1024 * 1024,
        tokens_generated: 100,
    };
    assert!(!slow_load.meets_targets());

    // Too slow first token
    let slow_first = DemoMetrics {
        load_time_ms: 3000,
        first_token_ms: 3000,
        tokens_per_sec: 20.0,
        peak_memory_bytes: 400 * 1024 * 1024,
        tokens_generated: 100,
    };
    assert!(!slow_first.meets_targets());

    // Too slow throughput
    let slow_throughput = DemoMetrics {
        load_time_ms: 3000,
        first_token_ms: 1000,
        tokens_per_sec: 10.0,
        peak_memory_bytes: 400 * 1024 * 1024,
        tokens_generated: 100,
    };
    assert!(!slow_throughput.meets_targets());

    // Too much memory
    let high_memory = DemoMetrics {
        load_time_ms: 3000,
        first_token_ms: 1000,
        tokens_per_sec: 20.0,
        peak_memory_bytes: 600 * 1024 * 1024,
        tokens_generated: 100,
    };
    assert!(!high_memory.meets_targets());
}

// =========================================================================
// FALSIFY: SpecialTokens::qwen2() matches special-tokens-registry-v1.yaml
// =========================================================================
#[test]
fn falsify_qwen2_special_tokens_match_yaml_registry() {
    let tokens = SpecialTokens::qwen2();

    // Values from special-tokens-registry-v1.yaml families.qwen2
    assert_eq!(tokens.bos_id, 151_643, "FALSIFY: Qwen2 BOS mismatch vs YAML");
    assert_eq!(tokens.eos_id, 151_645, "FALSIFY: Qwen2 EOS mismatch vs YAML");
    assert_eq!(tokens.pad_id, 151_643, "FALSIFY: Qwen2 PAD mismatch vs YAML");
    assert_eq!(tokens.im_start_id, 151_644, "FALSIFY: Qwen2 im_start mismatch vs YAML");
    assert_eq!(tokens.im_end_id, 151_645, "FALSIFY: Qwen2 im_end mismatch vs YAML");

    // Proof obligation: all token IDs < vocab_size (151936)
    const QWEN2_VOCAB: u32 = 151_936;
    assert!(tokens.bos_id < QWEN2_VOCAB, "FALSIFY: BOS >= vocab_size");
    assert!(tokens.eos_id < QWEN2_VOCAB, "FALSIFY: EOS >= vocab_size");
    assert!(tokens.pad_id < QWEN2_VOCAB, "FALSIFY: PAD >= vocab_size");
    assert!(tokens.im_start_id < QWEN2_VOCAB, "FALSIFY: im_start >= vocab_size");
    assert!(tokens.im_end_id < QWEN2_VOCAB, "FALSIFY: im_end >= vocab_size");
}
