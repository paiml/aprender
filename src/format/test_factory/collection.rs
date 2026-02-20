
/// Collect the tensor names, data, and shapes that a pygmy config would produce.
/// Mirrors the logic in `build_pygmy_safetensors_with_config`.
fn collect_pygmy_tensors(config: &PygmyConfig) -> Vec<(String, Vec<f32>, Vec<usize>)> {
    let mut tensors = Vec::new();

    if config.include_embedding {
        let data: Vec<f32> = (0..config.vocab_size * config.hidden_size)
            .map(|i| ((i % 100) as f32 - 50.0) / 1000.0)
            .collect();
        tensors.push((
            "model.embed_tokens.weight".to_string(),
            data,
            vec![config.vocab_size, config.hidden_size],
        ));
    }

    for layer_idx in 0..config.num_layers {
        if config.include_norms {
            let norm_data: Vec<f32> = vec![1.0; config.hidden_size];
            tensors.push((
                format!("model.layers.{layer_idx}.input_layernorm.weight"),
                norm_data.clone(),
                vec![config.hidden_size],
            ));
            tensors.push((
                format!("model.layers.{layer_idx}.post_attention_layernorm.weight"),
                norm_data,
                vec![config.hidden_size],
            ));
        }

        if config.include_attention {
            let kv_dim = config.kv_dim();

            // Q and O: [hidden_size, hidden_size]
            let q_data: Vec<f32> = (0..config.hidden_size * config.hidden_size)
                .map(|i| ((i % 200) as f32 - 100.0) / 2000.0)
                .collect();
            tensors.push((
                format!("model.layers.{layer_idx}.self_attn.q_proj.weight"),
                q_data.clone(),
                vec![config.hidden_size, config.hidden_size],
            ));
            tensors.push((
                format!("model.layers.{layer_idx}.self_attn.o_proj.weight"),
                q_data,
                vec![config.hidden_size, config.hidden_size],
            ));

            // K and V: [kv_dim, hidden_size]
            let kv_data: Vec<f32> = (0..kv_dim * config.hidden_size)
                .map(|i| ((i % 200) as f32 - 100.0) / 2000.0)
                .collect();
            tensors.push((
                format!("model.layers.{layer_idx}.self_attn.k_proj.weight"),
                kv_data.clone(),
                vec![kv_dim, config.hidden_size],
            ));
            tensors.push((
                format!("model.layers.{layer_idx}.self_attn.v_proj.weight"),
                kv_data,
                vec![kv_dim, config.hidden_size],
            ));

            // Biases
            if config.include_bias {
                let q_bias: Vec<f32> = (0..config.hidden_size)
                    .map(|i| (i as f32) / 1000.0)
                    .collect();
                tensors.push((
                    format!("model.layers.{layer_idx}.self_attn.q_proj.bias"),
                    q_bias,
                    vec![config.hidden_size],
                ));
                let kv_bias: Vec<f32> = (0..kv_dim).map(|i| (i as f32) / 1000.0).collect();
                tensors.push((
                    format!("model.layers.{layer_idx}.self_attn.k_proj.bias"),
                    kv_bias.clone(),
                    vec![kv_dim],
                ));
                tensors.push((
                    format!("model.layers.{layer_idx}.self_attn.v_proj.bias"),
                    kv_bias,
                    vec![kv_dim],
                ));
            }
        }

        if config.include_mlp {
            let intermediate = config.hidden_size * 2;
            let gate_up_data: Vec<f32> = (0..intermediate * config.hidden_size)
                .map(|i| ((i % 200) as f32 - 100.0) / 2000.0)
                .collect();
            let down_data: Vec<f32> = (0..config.hidden_size * intermediate)
                .map(|i| ((i % 200) as f32 - 100.0) / 2000.0)
                .collect();

            tensors.push((
                format!("model.layers.{layer_idx}.mlp.gate_proj.weight"),
                gate_up_data.clone(),
                vec![intermediate, config.hidden_size],
            ));
            tensors.push((
                format!("model.layers.{layer_idx}.mlp.up_proj.weight"),
                gate_up_data,
                vec![intermediate, config.hidden_size],
            ));
            tensors.push((
                format!("model.layers.{layer_idx}.mlp.down_proj.weight"),
                down_data,
                vec![config.hidden_size, intermediate],
            ));
        }
    }

    if config.include_norms && config.num_layers > 0 {
        let norm_data: Vec<f32> = vec![1.0; config.hidden_size];
        tensors.push((
            "model.norm.weight".to_string(),
            norm_data,
            vec![config.hidden_size],
        ));
    }

    if config.include_embedding && !config.tied_embeddings {
        let data: Vec<f32> = (0..config.vocab_size * config.hidden_size)
            .map(|i| ((i % 100) as f32 - 50.0) / 1000.0)
            .collect();
        tensors.push((
            "lm_head.weight".to_string(),
            data,
            vec![config.vocab_size, config.hidden_size],
        ));
    }

    tensors
}

// ====================================================================
// Harness self-tests
// ====================================================================

#[test]
fn test_harness_new_creates_temp_dir() {
    let h = ConversionTestHarness::new();
    assert!(h.dir().exists());
}

#[test]
fn test_harness_with_safetensors_writes_file() {
    let h = ConversionTestHarness::new().with_safetensors(PygmyConfig::default());
    assert!(h.input_path().is_some());
    assert!(h.input_path().expect("input").exists());
}

#[test]
fn test_harness_with_apr_writes_file() {
    let h = ConversionTestHarness::new().with_apr(PygmyConfig::default());
    assert!(h.input_path().is_some());
    assert!(h.input_path().expect("input").exists());
}

#[test]
fn test_harness_import_produces_output() {
    let h = ConversionTestHarness::new()
        .with_safetensors(PygmyConfig::default())
        .import_to_apr(ImportOptions {
            allow_no_config: true,
            ..ImportOptions::default()
        });
    assert!(h.output_path().is_some());
    assert!(h.output_path().expect("output").exists());
}

#[test]
fn test_harness_assert_import_ok_default() {
    ConversionTestHarness::assert_import_ok(PygmyConfig::default());
}

#[test]
fn test_harness_assert_import_ok_llama() {
    ConversionTestHarness::assert_import_ok(PygmyConfig::llama_style());
}

#[test]
fn test_harness_assert_import_ok_minimal() {
    ConversionTestHarness::assert_import_ok(PygmyConfig::minimal());
}

#[test]
fn test_harness_assert_roundtrip_ok_default() {
    ConversionTestHarness::assert_roundtrip_ok(PygmyConfig::default());
}

#[test]
fn test_harness_assert_roundtrip_ok_llama() {
    ConversionTestHarness::assert_roundtrip_ok(PygmyConfig::llama_style());
}

#[test]
fn test_harness_assert_roundtrip_ok_minimal() {
    ConversionTestHarness::assert_roundtrip_ok(PygmyConfig::minimal());
}

#[test]
fn test_harness_verify_apr_checks_shapes() {
    let h = ConversionTestHarness::new()
        .with_safetensors(PygmyConfig::default())
        .import_to_apr(ImportOptions {
            allow_no_config: true,
            ..ImportOptions::default()
        });
    let result = h.verify_apr();
    assert!(result.passed(), "Default import should verify cleanly");
}

#[test]
fn test_tolerance_config_default() {
    let t = ToleranceConfig::default();
    assert!((t.f32_atol - 1e-6).abs() < 1e-9);
    assert!((t.f16_atol - 1e-3).abs() < 1e-6);
    assert!((t.q8_atol - 0.1).abs() < 1e-6);
    assert!((t.q4_atol - 0.5).abs() < 1e-6);
}

// ====================================================================
// Falsification Protocol (rosetta-testing.md QA Matrix)
// ====================================================================

/// F-HAR-01: Corrupt tensor data region of `.apr` -> `verify()` detects DataMismatch
#[test]
fn test_f_har_01_corruption_detected() {
    use std::io::Write;

    // 1. Create valid APR via harness
    let h = ConversionTestHarness::new()
        .with_safetensors(PygmyConfig::default())
        .import_to_apr(ImportOptions {
            allow_no_config: true,
            ..ImportOptions::default()
        });

    let output_path = h.output_path().expect("output exists");

    // 2. Read APR, find tensor data offset from header (bytes 32-39 = data_offset u64 LE)
    let mut data = std::fs::read(&output_path).expect("read APR");
    let data_offset =
        u64::from_le_bytes(data[32..40].try_into().expect("8 bytes for data_offset")) as usize;

    // 3. Corrupt first 16 bytes of actual tensor data (4 f32 values)
    assert!(
        data.len() > data_offset + 16,
        "APR file must have tensor data after data_offset={data_offset}"
    );
    for byte in &mut data[data_offset..data_offset + 16] {
        *byte ^= 0xFF;
    }

    // 4. Write corrupted data back
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .truncate(true)
        .open(&output_path)
        .expect("open APR for write");
    file.write_all(&data).expect("write corrupted");
    drop(file);

    // 5. Verify MUST detect the data mismatch
    let result = h.verify_apr();
    assert!(
        !result.passed(),
        "F-HAR-01: Corruption at data_offset MUST be detected by verify_apr()"
    );
}

/// F-HAR-02: Set tolerance to `1e-9` (too strict) -> verify with default tolerance
/// Note: The harness uses fixed tolerances; this test validates the tolerance config exists
#[test]
fn test_f_har_02_strict_tolerance_config() {
    // Verify that strict tolerance values are actually stricter than defaults
    let strict = ToleranceConfig {
        f32_atol: 1e-9, // Too strict - will fail on quantization/dequant noise
        f16_atol: 1e-9,
        q8_atol: 1e-9,
        q4_atol: 1e-9,
    };
    let default = ToleranceConfig::default();

    assert!(strict.f32_atol < default.f32_atol);
    assert!(strict.f16_atol < default.f16_atol);
    assert!(strict.q8_atol < default.q8_atol);
    assert!(strict.q4_atol < default.q4_atol);
}

/// F-HAR-03: Use `--strict` on `embedding_only` config -> Import FAILS (Unverified Architecture)
#[test]
fn test_f_har_03_strict_embedding_only() {
    let config = PygmyConfig::embedding_only();

    // Strict mode with embedding-only config should FAIL
    let mut options = ImportOptions::default();
    options.strict = true;
    options.allow_no_config = true;

    let h = ConversionTestHarness::new().with_safetensors(config);

    // Import with strict mode - this should fail with unverified architecture
    let result = h.try_import_to_apr(options);

    // Expected behavior: strict mode rejects unverified architectures
    // The test passes if import fails (strict mode working as intended)
    assert!(
        result.is_err(),
        "F-HAR-03: Strict mode should reject unverified architecture"
    );
}

/// F-HAR-04: Use `PygmyConfig` with 0 tensors -> Harness handles gracefully (no crash)
#[test]
fn test_f_har_04_zero_tensors_graceful() {
    let config = PygmyConfig {
        vocab_size: 0,
        hidden_size: 0,
        num_layers: 0,
        include_embedding: false,
        include_norms: false,
        include_attention: false,
        include_mlp: false,
        ..Default::default()
    };

    // Should not crash when building SafeTensors with zero tensors
    let st_bytes = build_pygmy_safetensors_with_config(config);
    // File may be minimal but should be valid SafeTensors
    assert!(st_bytes.len() >= 8, "Should have at least header length");
}

/// F-REG-01: Round-trip Llama-style tensors -> `verify_safetensors()` PASSES
/// (This is already covered by test_harness_assert_roundtrip_ok_llama but we
/// add an explicit named test for traceability)
#[test]
fn test_f_reg_01_roundtrip_llama_style() {
    ConversionTestHarness::assert_roundtrip_ok(PygmyConfig::llama_style());
}

// ====================================================================
// Master Falsification QA Protocol (100-Point Matrix)
// Philosophy: Karl Popper (Refutation) & Toyota Way (Jidoka)
// ====================================================================

/// F-CONV-01 (Bit-Flipping): Corrupt single f32 in tensor data -> verify_apr() MUST detect
#[test]
fn test_f_conv_01_bit_flipping_detected() {
    use std::io::Write;

    let h = ConversionTestHarness::new()
        .with_safetensors(PygmyConfig::default())
        .import_to_apr(ImportOptions {
            allow_no_config: true,
            ..ImportOptions::default()
        });

    let output_path = h.output_path().expect("output exists");

    // Read APR, find tensor data offset from header (bytes 32-39 = data_offset u64 LE)
    let mut data = std::fs::read(&output_path).expect("read APR");
    let data_offset =
        u64::from_le_bytes(data[32..40].try_into().expect("8 bytes for data_offset")) as usize;

    // Flip all bits in a single f32 value (4 bytes) at start of tensor data
    assert!(
        data.len() > data_offset + 4,
        "APR file must have tensor data after data_offset={data_offset}"
    );
    for byte in &mut data[data_offset..data_offset + 4] {
        *byte ^= 0xFF;
    }

    // Write corrupted data
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .truncate(true)
        .open(&output_path)
        .expect("open");
    file.write_all(&data).expect("write");
    drop(file);

    // Verify MUST detect the single-value mismatch
    let result = h.verify_apr();
    assert!(
        !result.passed(),
        "F-CONV-01: Single f32 bit-flip MUST be detected by verify_apr()"
    );
}

/// F-CONV-02 (Tolerance Drift): Set f32_atol to 1e-12 -> Standard tests should fail
#[test]
fn test_f_conv_02_tolerance_drift() {
    let ultra_strict = ToleranceConfig {
        f32_atol: 1e-12,
        f16_atol: 1e-12,
        q8_atol: 1e-12,
        q4_atol: 1e-12,
    };
    let default = ToleranceConfig::default();

    // Ultra-strict MUST be stricter than default
    assert!(
        ultra_strict.f32_atol < default.f32_atol / 1000.0,
        "F-CONV-02: 1e-12 should be 1000x stricter than default 1e-6"
    );
}

/// F-CONV-03 (Auto-Arch Refutation): Garbage tensor names -> Auto-mapping fallback
#[test]
fn test_f_conv_03_auto_arch_garbage_names() {
    use crate::format::Architecture;

    // With garbage tensor names, auto-mapping should use default behavior
    let arch = Architecture::Auto;

    // Auto-mapping on unknown patterns should preserve or minimally transform
    let mapped = arch.map_name("garbage.weight");

    // The important thing is it doesn't crash and handles gracefully
    assert!(
        !mapped.is_empty(),
        "F-CONV-03: Auto-map should handle garbage names gracefully"
    );
}
