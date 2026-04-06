//! Phase 54: Fixture-Based GGUF Loader Tests
//!
//! Uses the ModelFixture pattern for standardized testing with RAII-based cleanup.
//! Covers loader.rs dark zones with various model configurations.

use crate::fixtures::{ModelConfig, ModelFixture, ModelFormat};
use crate::gguf::GGUFModel;

// =============================================================================
// Architecture Loading Tests
// =============================================================================

/// Test loading a tiny llama-style model via fixture
#[test]
fn test_fixture_load_tiny_llama() {
    let fixture = ModelFixture::gguf("tiny_llama", ModelConfig::tiny());
    let bytes = fixture.read_bytes().expect("read fixture");

    let model = GGUFModel::from_bytes(&bytes).expect("parse GGUF");

    // Verify architecture
    assert_eq!(model.architecture(), Some("llama"));

    // Verify dimensions from config
    assert_eq!(model.embedding_dim(), Some(64));
    assert_eq!(model.num_layers(), Some(1));
    assert_eq!(model.num_heads(), Some(4));
    assert_eq!(model.num_kv_heads(), Some(4));

    // Verify tensors exist
    assert!(!model.tensors.is_empty());
}

/// Test loading a small model (larger than tiny)
#[test]
fn test_fixture_load_small_model() {
    let fixture = ModelFixture::gguf("small", ModelConfig::small());
    let bytes = fixture.read_bytes().expect("read fixture");

    let model = GGUFModel::from_bytes(&bytes).expect("parse GGUF");

    assert_eq!(model.embedding_dim(), Some(256));
    assert_eq!(model.num_layers(), Some(2));
    assert_eq!(model.num_heads(), Some(8));
    assert_eq!(model.context_length(), Some(1024));
}

/// Test loading a GQA model (different num_heads vs num_kv_heads)
#[test]
fn test_fixture_load_gqa_model() {
    let fixture = ModelFixture::gguf("gqa", ModelConfig::gqa());
    let bytes = fixture.read_bytes().expect("read fixture");

    let model = GGUFModel::from_bytes(&bytes).expect("parse GGUF");

    // GQA has 8 Q heads but only 2 KV heads
    assert_eq!(model.num_heads(), Some(8));
    assert_eq!(model.num_kv_heads(), Some(2));
}

/// Test loading a Phi-2 style model (LayerNorm + GELU)
#[test]
fn test_fixture_load_phi_model() {
    let fixture = ModelFixture::gguf("phi2", ModelConfig::phi());
    let bytes = fixture.read_bytes().expect("read fixture");

    let model = GGUFModel::from_bytes(&bytes).expect("parse GGUF");

    assert_eq!(model.architecture(), Some("phi2"));
}

/// Test loading a Qwen-style model
#[test]
fn test_fixture_load_qwen_model() {
    let fixture = ModelFixture::gguf("qwen", ModelConfig::qwen());
    let bytes = fixture.read_bytes().expect("read fixture");

    let model = GGUFModel::from_bytes(&bytes).expect("parse GGUF");

    assert_eq!(model.architecture(), Some("qwen2"));
    assert_eq!(model.num_kv_heads(), Some(4));
    // Qwen uses different RoPE theta
    assert!(
        model
            .rope_freq_base()
            .expect("test value should be present")
            > 100000.0
    );
}

// =============================================================================
// Error Handling Tests
// =============================================================================

/// Test invalid magic detection
#[test]
fn test_fixture_invalid_magic_detected() {
    let fixture = ModelFixture::gguf_invalid_magic("bad_magic");
    let bytes = fixture.read_bytes().expect("read fixture");

    let result = GGUFModel::from_bytes(&bytes);
    assert!(result.is_err());

    let err = result.unwrap_err();
    let err_str = err.to_string();
    // Should mention invalid magic or GGUF header
    assert!(err_str.contains("magic") || err_str.contains("GGUF") || err_str.contains("invalid"));
}

/// Test invalid version detection
#[test]
fn test_fixture_invalid_version_detected() {
    let fixture = ModelFixture::gguf_invalid_version("bad_version");
    let bytes = fixture.read_bytes().expect("read fixture");

    let result = GGUFModel::from_bytes(&bytes);
    assert!(result.is_err());

    let err = result.unwrap_err();
    let err_str = err.to_string();
    // Should mention version or unsupported
    assert!(
        err_str.contains("version") || err_str.contains("unsupported") || err_str.contains("999")
    );
}

// =============================================================================
// Metadata Tests
// =============================================================================

/// Test RoPE frequency base metadata
#[test]
fn test_fixture_rope_freq_base() {
    let config = ModelConfig::tiny().with_architecture("llama");
    let fixture = ModelFixture::gguf("rope_test", config);
    let bytes = fixture.read_bytes().expect("read fixture");

    let model = GGUFModel::from_bytes(&bytes).expect("parse GGUF");

    let rope_base = model.rope_freq_base();
    assert!(rope_base.is_some());
    assert!((rope_base.expect("test value should be present") - 10000.0).abs() < 1.0);
}

/// Test RMS epsilon metadata
#[test]
fn test_fixture_rms_epsilon() {
    let config = ModelConfig::tiny();
    let fixture = ModelFixture::gguf("eps_test", config);
    let bytes = fixture.read_bytes().expect("read fixture");

    let model = GGUFModel::from_bytes(&bytes).expect("parse GGUF");

    let eps = model.rms_epsilon();
    assert!(eps.is_some());
    assert!(eps.expect("test value should be present") < 0.001);
}

/// Test context length metadata
#[test]
fn test_fixture_context_length() {
    let config = ModelConfig::small();
    let fixture = ModelFixture::gguf("ctx_test", config);
    let bytes = fixture.read_bytes().expect("read fixture");

    let model = GGUFModel::from_bytes(&bytes).expect("parse GGUF");

    assert_eq!(model.context_length(), Some(1024));
}

// =============================================================================
// Custom Configuration Tests
// =============================================================================

/// Test builder pattern for custom configs
#[test]
fn test_fixture_custom_config() {
    let config = ModelConfig::tiny()
        .with_hidden_dim(128)
        .with_layers(3)
        .with_vocab_size(5000)
        .with_gqa(8, 2);

    let fixture = ModelFixture::gguf("custom", config);
    let bytes = fixture.read_bytes().expect("read fixture");

    let model = GGUFModel::from_bytes(&bytes).expect("parse GGUF");

    assert_eq!(model.embedding_dim(), Some(128));
    assert_eq!(model.num_layers(), Some(3));
    assert_eq!(model.num_heads(), Some(8));
    assert_eq!(model.num_kv_heads(), Some(2));
}

/// Test model with many layers
#[test]
fn test_fixture_many_layers() {
    let config = ModelConfig::tiny().with_layers(8);
    let fixture = ModelFixture::gguf("deep", config);
    let bytes = fixture.read_bytes().expect("read fixture");

    let model = GGUFModel::from_bytes(&bytes).expect("parse GGUF");

    assert_eq!(model.num_layers(), Some(8));
}

/// Test model with large vocabulary
#[test]
fn test_fixture_large_vocab() {
    let config = ModelConfig::tiny().with_vocab_size(32000);
    let fixture = ModelFixture::gguf("large_vocab", config);
    let bytes = fixture.read_bytes().expect("read fixture");

    let model = GGUFModel::from_bytes(&bytes).expect("parse GGUF");

    // Verify embedding tensor exists
    let embed = model.tensors.iter().find(|t| t.name.contains("embd"));
    assert!(embed.is_some());
}

// =============================================================================
// Format Tests
// =============================================================================

/// Test ModelFormat::Gguf extension
#[test]
fn test_fixture_format_gguf() {
    let fixture = ModelFixture::gguf("fmt_test", ModelConfig::tiny());
    assert_eq!(fixture.format(), ModelFormat::Gguf);
    assert!(fixture.path().to_string_lossy().ends_with(".gguf"));
}

/// Test ModelFormat::SafeTensors extension
#[test]
fn test_fixture_format_safetensors() {
    let fixture = ModelFixture::safetensors("st_test", ModelConfig::tiny());
    assert_eq!(fixture.format(), ModelFormat::SafeTensors);
    assert!(fixture.path().to_string_lossy().ends_with(".safetensors"));
}

/// Test ModelFormat::Apr extension
#[test]
fn test_fixture_format_apr() {
    let fixture = ModelFixture::apr("apr_test", ModelConfig::tiny());
    assert_eq!(fixture.format(), ModelFormat::Apr);
    assert!(fixture.path().to_string_lossy().ends_with(".apr"));
}

// =============================================================================
// RAII Cleanup Tests
// =============================================================================

/// Test that fixture cleanup works correctly
#[test]
fn test_fixture_cleanup() {
    let path = {
        let fixture = ModelFixture::gguf("cleanup_test", ModelConfig::tiny());
        let p = fixture.path().to_path_buf();
        assert!(p.exists(), "File should exist during fixture lifetime");
        p
    };
    // After fixture drops, file should be cleaned up
    assert!(
        !path.exists(),
        "File should be cleaned up after fixture drops"
    );
}

/// Test multiple fixtures in same test
#[test]
fn test_fixture_multiple_coexist() {
    let f1 = ModelFixture::gguf("multi_1", ModelConfig::tiny());
    let f2 = ModelFixture::gguf("multi_2", ModelConfig::small());
    let f3 = ModelFixture::gguf("multi_3", ModelConfig::gqa());

    // All should exist simultaneously
    assert!(f1.path().exists());
    assert!(f2.path().exists());
    assert!(f3.path().exists());

    // Paths should be different
    assert_ne!(f1.path(), f2.path());
    assert_ne!(f2.path(), f3.path());
}

// =============================================================================
// Tensor Loading Tests
// =============================================================================

/// Test loading token embedding tensor
#[test]
fn test_fixture_load_embedding_tensor() {
    let fixture = ModelFixture::gguf("embed_test", ModelConfig::tiny());
    let bytes = fixture.read_bytes().expect("read fixture");

    let model = GGUFModel::from_bytes(&bytes).expect("parse GGUF");

    // Find and load embedding
    let embed_tensor = model.tensors.iter().find(|t| t.name == "token_embd.weight");
    assert!(
        embed_tensor.is_some(),
        "Should have token_embd.weight tensor"
    );

    let tensor_info = embed_tensor.expect("test value should be present");
    let embed_data = model.get_tensor_f32(&tensor_info.name, &bytes);
    assert!(embed_data.is_ok(), "Should be able to load embedding data");
}

/// Test loading quantized Q4_K tensor
#[test]
fn test_fixture_load_q4k_tensor() {
    let fixture = ModelFixture::gguf("q4k_test", ModelConfig::tiny());
    let bytes = fixture.read_bytes().expect("read fixture");

    let model = GGUFModel::from_bytes(&bytes).expect("parse GGUF");

    // Find a Q4_K tensor (attention weights)
    let q4k_tensor = model.tensors.iter().find(|t| t.name.contains("attn_qkv"));
    assert!(q4k_tensor.is_some(), "Should have Q4_K attention tensor");

    let tensor_info = q4k_tensor.expect("test value should be present");
    // Q4_K type is 12
    assert_eq!(tensor_info.qtype, 12, "Should be Q4_K type");
}

/// Test loading output norm tensor (F32)
#[test]
fn test_fixture_load_f32_tensor() {
    let fixture = ModelFixture::gguf("f32_test", ModelConfig::tiny());
    let bytes = fixture.read_bytes().expect("read fixture");

    let model = GGUFModel::from_bytes(&bytes).expect("parse GGUF");

    // Find norm tensor (should be F32)
    let norm_tensor = model
        .tensors
        .iter()
        .find(|t| t.name.contains("output_norm"));
    assert!(norm_tensor.is_some(), "Should have output_norm tensor");

    let tensor_info = norm_tensor.expect("test value should be present");
    // F32 type is 0
    assert_eq!(tensor_info.qtype, 0, "Norm should be F32 type");
}

// =============================================================================
// Cross-Architecture Tests
// =============================================================================

/// Test that different architectures produce different metadata
#[test]
fn test_fixture_architecture_differences() {
    let llama = ModelFixture::gguf("arch_llama", ModelConfig::tiny().with_architecture("llama"));
    let phi = ModelFixture::gguf("arch_phi", ModelConfig::phi());
    let qwen = ModelFixture::gguf("arch_qwen", ModelConfig::qwen());

    let llama_bytes = llama.read_bytes().expect("read");
    let phi_bytes = phi.read_bytes().expect("read");
    let qwen_bytes = qwen.read_bytes().expect("read");

    let llama_model = GGUFModel::from_bytes(&llama_bytes).expect("parse");
    let phi_model = GGUFModel::from_bytes(&phi_bytes).expect("parse");
    let qwen_model = GGUFModel::from_bytes(&qwen_bytes).expect("parse");

    // Different architectures
    assert_ne!(llama_model.architecture(), phi_model.architecture());
    assert_ne!(phi_model.architecture(), qwen_model.architecture());

    // All should be valid
    assert!(llama_model.architecture().is_some());
    assert!(phi_model.architecture().is_some());
    assert!(qwen_model.architecture().is_some());
}

/// Test MHA vs GQA configurations
#[test]
fn test_fixture_mha_vs_gqa() {
    let mha_config = ModelConfig::tiny(); // num_heads == num_kv_heads
    let gqa_config = ModelConfig::gqa(); // num_heads != num_kv_heads

    let mha = ModelFixture::gguf("mha", mha_config);
    let gqa = ModelFixture::gguf("gqa", gqa_config);

    let mha_bytes = mha.read_bytes().expect("read");
    let gqa_bytes = gqa.read_bytes().expect("read");

    let mha_model = GGUFModel::from_bytes(&mha_bytes).expect("parse");
    let gqa_model = GGUFModel::from_bytes(&gqa_bytes).expect("parse");

    // MHA: num_heads == num_kv_heads
    assert_eq!(mha_model.num_heads(), mha_model.num_kv_heads());

    // GQA: num_heads != num_kv_heads
    assert_ne!(gqa_model.num_heads(), gqa_model.num_kv_heads());

    // GQA should have more Q heads than KV heads
    assert!(
        gqa_model.num_heads().expect("test value should be present")
            > gqa_model
                .num_kv_heads()
                .expect("test value should be present")
    );
}

// =============================================================================
// Header Validation Tests
// =============================================================================

/// Test GGUF magic bytes are correct
#[test]
fn test_fixture_gguf_magic() {
    let fixture = ModelFixture::gguf("magic_test", ModelConfig::tiny());
    let bytes = fixture.read_bytes().expect("read");

    // First 4 bytes should be GGUF magic: 0x46554747 ("GGUF" in little endian)
    assert_eq!(&bytes[0..4], &[0x47, 0x47, 0x55, 0x46]);
}

/// Test GGUF version is 3
#[test]
fn test_fixture_gguf_version() {
    let fixture = ModelFixture::gguf("version_test", ModelConfig::tiny());
    let bytes = fixture.read_bytes().expect("read");

    // Bytes 4-8 should be version (little endian u32)
    let version = u32::from_le_bytes(bytes[4..8].try_into().expect("try_into conversion failed"));
    assert_eq!(version, 3, "Should be GGUF v3");
}

// =============================================================================
// Config Accessor Tests
// =============================================================================

/// Test ModelConfig accessors
#[test]
fn test_fixture_config_accessors() {
    let config = ModelConfig::small();
    let fixture = ModelFixture::gguf("accessor_test", config.clone());

    let returned_config = fixture.config();
    assert_eq!(returned_config.hidden_dim, 256);
    assert_eq!(returned_config.intermediate_dim, 512);
    assert_eq!(returned_config.num_heads, 8);
    assert_eq!(returned_config.num_layers, 2);
}

/// Test ModelConfig::default is tiny
#[test]
fn test_model_config_default() {
    let default = ModelConfig::default();
    let tiny = ModelConfig::tiny();

    assert_eq!(default.hidden_dim, tiny.hidden_dim);
    assert_eq!(default.num_layers, tiny.num_layers);
}
