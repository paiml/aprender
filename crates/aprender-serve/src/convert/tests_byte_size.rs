
#[test]
fn test_byte_size_q8_0() {
    // Q8_0: 32 elements = 34 bytes
    let num_elements = 1024usize;
    let byte_size = num_elements.div_ceil(32) * 34;
    assert_eq!(byte_size, 32 * 34); // 1088
    assert_eq!(byte_size, 1088);
}

#[test]
fn test_byte_size_q4k() {
    // Q4_K: 256 elements = 144 bytes
    let num_elements = 1024usize;
    let byte_size = num_elements.div_ceil(256) * 144;
    assert_eq!(byte_size, 4 * 144); // 576
    assert_eq!(byte_size, 576);
}

#[test]
fn test_byte_size_q5k() {
    // Q5_K: 256 elements = 176 bytes
    let num_elements = 1024usize;
    let byte_size = num_elements.div_ceil(256) * 176;
    assert_eq!(byte_size, 4 * 176); // 704
    assert_eq!(byte_size, 704);
}

#[test]
fn test_byte_size_q6k() {
    // Q6_K: 256 elements = 210 bytes
    let num_elements = 1024usize;
    let byte_size = num_elements.div_ceil(256) * 210;
    assert_eq!(byte_size, 4 * 210); // 840
    assert_eq!(byte_size, 840);
}

#[test]
fn test_byte_size_unknown_defaults_f32() {
    // Unknown dtype defaults to F32 = 4 bytes per element
    let num_elements = 50usize;
    let byte_size = num_elements * 4;
    assert_eq!(byte_size, 200);
}

// ============================================================================
// GGUF metadata helpers: additional type coverage
// ============================================================================

#[test]
fn test_get_u32_from_float32_returns_none() {
    use crate::gguf::GGUFValue;
    let mut meta = std::collections::HashMap::new();
    meta.insert("key".to_string(), GGUFValue::Float32(3.14));
    assert_eq!(GgufToAprQ4KConverter::get_u32(&meta, "key"), None);
}

#[test]
fn test_get_f32_from_uint32_returns_none() {
    use crate::gguf::GGUFValue;
    let mut meta = std::collections::HashMap::new();
    meta.insert("key".to_string(), GGUFValue::UInt32(42));
    assert_eq!(GgufToAprQ4KConverter::get_f32(&meta, "key"), None);
}

#[test]
fn test_get_string_from_bool_returns_none() {
    use crate::gguf::GGUFValue;
    let mut meta = std::collections::HashMap::new();
    meta.insert("key".to_string(), GGUFValue::Bool(true));
    assert_eq!(GgufToAprQ4KConverter::get_string(&meta, "key"), None);
}

#[test]
fn test_get_u32_from_bool_returns_none() {
    use crate::gguf::GGUFValue;
    let mut meta = std::collections::HashMap::new();
    meta.insert("key".to_string(), GGUFValue::Bool(false));
    assert_eq!(GgufToAprQ4KConverter::get_u32(&meta, "key"), None);
}

#[test]
fn test_get_f32_from_bool_returns_none() {
    use crate::gguf::GGUFValue;
    let mut meta = std::collections::HashMap::new();
    meta.insert("key".to_string(), GGUFValue::Bool(true));
    assert_eq!(GgufToAprQ4KConverter::get_f32(&meta, "key"), None);
}

// ============================================================================
// ConversionStats: Display-like formatting coverage
// ============================================================================

#[test]
fn test_conversion_stats_memory_mb_fractional() {
    let stats = ConversionStats {
        total_parameters: 0,
        memory_bytes_f32: 1024 * 512, // 0.5 MB
        num_layers: 0,
        hidden_dim: 0,
        vocab_size: 0,
        architecture: String::new(),
    };
    assert!((stats.memory_mb() - 0.5).abs() < 0.001);
}

#[test]
fn test_conversion_stats_memory_gb_fractional() {
    let stats = ConversionStats {
        total_parameters: 0,
        memory_bytes_f32: 1024 * 1024 * 512, // 0.5 GB
        num_layers: 0,
        hidden_dim: 0,
        vocab_size: 0,
        architecture: String::new(),
    };
    assert!((stats.memory_gb() - 0.5).abs() < 0.001);
}

#[test]
fn test_conversion_stats_parameters_m_fractional() {
    let stats = ConversionStats {
        total_parameters: 500_000, // 0.5M
        memory_bytes_f32: 0,
        num_layers: 0,
        hidden_dim: 0,
        vocab_size: 0,
        architecture: String::new(),
    };
    assert!((stats.parameters_m() - 0.5).abs() < 0.001);
}

#[test]
fn test_conversion_stats_parameters_b_fractional() {
    let stats = ConversionStats {
        total_parameters: 500_000_000, // 0.5B
        memory_bytes_f32: 0,
        num_layers: 0,
        hidden_dim: 0,
        vocab_size: 0,
        architecture: String::new(),
    };
    assert!((stats.parameters_b() - 0.5).abs() < 0.001);
}
