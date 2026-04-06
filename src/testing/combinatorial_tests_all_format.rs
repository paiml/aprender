
#[test]
fn test_all_format_conversions() {
    let formats = [
        ModelFormat::GGUF,
        ModelFormat::APR,
        ModelFormat::Safetensors,
    ];

    for source in &formats {
        let Some(fixture) = create_fixture(*source) else {
            continue;
        };

        for target in &formats {
            if source == target {
                continue;
            }

            let result = fixture.convert_to(*target);
            assert!(
                result.is_ok(),
                "Conversion {:?} -> {:?} should succeed",
                source,
                target
            );

            let converted = result.expect("converted");
            assert_eq!(converted.format(), *target);
            assert_eq!(
                converted.config().num_heads,
                fixture.config().num_heads,
                "num_heads not preserved in {:?} -> {:?}",
                source,
                target
            );
        }
    }
}

#[test]
fn test_all_gqa_ratios() {
    let gqa_configs = [
        (4, 4, "MHA"),     // Multi-head attention (no GQA)
        (4, 2, "GQA-2:1"), // 2:1 GQA ratio
        (8, 2, "GQA-4:1"), // 4:1 GQA ratio
        (8, 1, "MQA"),     // Multi-query attention
    ];

    for (heads, kv_heads, desc) in &gqa_configs {
        let mut config = ModelConfig::tiny();
        config.num_heads = *heads;
        config.num_kv_heads = *kv_heads;

        let fixture = GgufFixture::new(config.clone(), QuantType::F32, 42);

        // Test conversion preserves GQA config
        let apr = fixture.convert_to(ModelFormat::APR).expect("apr");

        assert_eq!(
            apr.config().num_kv_heads,
            *kv_heads,
            "{}: num_kv_heads not preserved",
            desc
        );

        assert_eq!(
            apr.config().gqa_group_size(),
            heads / kv_heads,
            "{}: GQA group size wrong",
            desc
        );
    }
}

#[test]
fn test_forward_determinism() {
    let fixture = GgufFixture::tiny_gqa();
    let tokens = vec![1, 2, 3, 4, 5];

    // Multiple forward passes should give same result
    let out1 = fixture.forward(Device::Cpu, &tokens).expect("out1");
    let out2 = fixture.forward(Device::Cpu, &tokens).expect("out2");

    assert_eq!(
        out1.len(),
        out2.len(),
        "Forward pass output length must be deterministic"
    );

    for (i, (a, b)) in out1.iter().zip(out2.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-6,
            "Forward pass must be deterministic, element {} differs: {} vs {}",
            i,
            a,
            b
        );
    }
}

#[test]
fn test_embedding_determinism() {
    let fixture = GgufFixture::tiny_gqa();

    for token in [0u32, 1, 42, 255] {
        let embed1 = fixture.embed(Device::Cpu, token).expect("embed1");
        let embed2 = fixture.embed(Device::Cpu, token).expect("embed2");

        assert_eq!(
            embed1, embed2,
            "Embedding must be deterministic for token {}",
            token
        );
    }
}

#[test]
fn test_memory_footprint_ordering() {
    let tiny = GgufFixture::tiny_gqa();
    let small = GgufFixture::new(ModelConfig::small(), QuantType::Q4_0, 42);

    assert!(
        small.memory_bytes() > tiny.memory_bytes(),
        "Larger config should use more memory"
    );
}

/// Run all combinatorial tests and report coverage
#[test]
fn test_combinatorial_coverage_report() {
    let tests = generate_combinatorial_tests();
    let quant_tests = generate_quant_tests();

    println!("=== Combinatorial Test Coverage ===");
    println!("Format conversion tests: {}", tests.len());
    println!("Quantization tests: {}", quant_tests.len());
    println!("Total test cases: {}", tests.len() + quant_tests.len());

    // Count by category
    let cpu_tests = tests.iter().filter(|t| t.device == Device::Cpu).count();
    let cuda_tests = tests.iter().filter(|t| t.device == Device::Cuda(0)).count();

    println!("\nDevice distribution:");
    println!("  CPU:  {} tests", cpu_tests);
    println!("  CUDA: {} tests", cuda_tests);

    // Verify we have good coverage
    assert!(
        tests.len() >= 50,
        "Should have at least 50 conversion tests"
    );
    assert!(
        quant_tests.len() >= 20,
        "Should have at least 20 quant tests"
    );
}
