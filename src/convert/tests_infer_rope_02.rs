
#[test]
fn test_infer_rope_type_norm_default() {
    let empty_meta = std::collections::HashMap::new();

    // LLaMA and unknown architectures default to NORM (0)
    assert_eq!(
        GgufToAprQ4KConverter::infer_rope_type("llama", &empty_meta),
        0
    );
    assert_eq!(
        GgufToAprQ4KConverter::infer_rope_type("unknown_arch", &empty_meta),
        0
    );
    assert_eq!(
        GgufToAprQ4KConverter::infer_rope_type("tinyllama", &empty_meta),
        0
    );
}

#[test]
fn test_infer_rope_type_explicit_scaling() {
    use crate::gguf::GGUFValue;

    let mut meta = std::collections::HashMap::new();
    meta.insert(
        "qwen2.rope.scaling.type".to_string(),
        GGUFValue::String("yarn".to_string()),
    );

    // Explicit "yarn" -> NEOX (2)
    assert_eq!(GgufToAprQ4KConverter::infer_rope_type("qwen2", &meta), 2);
}

#[test]
fn test_infer_rope_type_explicit_none_scaling() {
    use crate::gguf::GGUFValue;

    let mut meta = std::collections::HashMap::new();
    meta.insert(
        "llama.rope.scaling.type".to_string(),
        GGUFValue::String("none".to_string()),
    );

    // Explicit "none" -> NORM (0)
    assert_eq!(GgufToAprQ4KConverter::infer_rope_type("llama", &meta), 0);
}

#[test]
fn test_infer_rope_type_explicit_linear_scaling() {
    use crate::gguf::GGUFValue;

    let mut meta = std::collections::HashMap::new();
    meta.insert(
        "llama.rope.scaling.type".to_string(),
        GGUFValue::String("linear".to_string()),
    );

    assert_eq!(GgufToAprQ4KConverter::infer_rope_type("llama", &meta), 0);
}

#[test]
fn test_infer_rope_type_explicit_neox_scaling() {
    use crate::gguf::GGUFValue;

    let mut meta = std::collections::HashMap::new();
    meta.insert(
        "llama.rope.scaling.type".to_string(),
        GGUFValue::String("neox".to_string()),
    );

    // Explicit "neox" overrides architecture default
    assert_eq!(GgufToAprQ4KConverter::infer_rope_type("llama", &meta), 2);
}

#[test]
fn test_infer_rope_type_unknown_explicit_falls_through() {
    use crate::gguf::GGUFValue;

    let mut meta = std::collections::HashMap::new();
    meta.insert(
        "llama.rope.scaling.type".to_string(),
        GGUFValue::String("unknown_type".to_string()),
    );

    // Unknown explicit type falls through to architecture-based detection
    // "llama" defaults to NORM (0)
    assert_eq!(GgufToAprQ4KConverter::infer_rope_type("llama", &meta), 0);
}

// ============================================================================
// GGUF metadata helpers
// ============================================================================

#[test]
fn test_get_string_helper() {
    use crate::gguf::GGUFValue;

    let mut meta = std::collections::HashMap::new();
    meta.insert("key1".to_string(), GGUFValue::String("value1".to_string()));
    meta.insert("key2".to_string(), GGUFValue::UInt32(42));

    assert_eq!(
        GgufToAprQ4KConverter::get_string(&meta, "key1"),
        Some("value1".to_string())
    );
    assert_eq!(GgufToAprQ4KConverter::get_string(&meta, "key2"), None);
    assert_eq!(
        GgufToAprQ4KConverter::get_string(&meta, "nonexistent"),
        None
    );
}

#[test]
fn test_get_u32_helper() {
    use crate::gguf::GGUFValue;

    let mut meta = std::collections::HashMap::new();
    meta.insert("u32".to_string(), GGUFValue::UInt32(42));
    meta.insert("i32".to_string(), GGUFValue::Int32(100));
    meta.insert("u64".to_string(), GGUFValue::UInt64(200));
    meta.insert(
        "str".to_string(),
        GGUFValue::String("not a number".to_string()),
    );

    assert_eq!(GgufToAprQ4KConverter::get_u32(&meta, "u32"), Some(42));
    assert_eq!(GgufToAprQ4KConverter::get_u32(&meta, "i32"), Some(100));
    assert_eq!(GgufToAprQ4KConverter::get_u32(&meta, "u64"), Some(200));
    assert_eq!(GgufToAprQ4KConverter::get_u32(&meta, "str"), None);
    assert_eq!(GgufToAprQ4KConverter::get_u32(&meta, "missing"), None);
}

#[test]
fn test_get_f32_helper() {
    use crate::gguf::GGUFValue;

    let mut meta = std::collections::HashMap::new();
    meta.insert("f32".to_string(), GGUFValue::Float32(3.14));
    meta.insert("f64".to_string(), GGUFValue::Float64(2.718));
    meta.insert("u32".to_string(), GGUFValue::UInt32(42));

    let f32_val = GgufToAprQ4KConverter::get_f32(&meta, "f32");
    assert!(f32_val.is_some());
    assert!((f32_val.unwrap() - 3.14).abs() < 0.001);

    let f64_val = GgufToAprQ4KConverter::get_f32(&meta, "f64");
    assert!(f64_val.is_some());
    assert!((f64_val.unwrap() - 2.718).abs() < 0.01);

    assert_eq!(GgufToAprQ4KConverter::get_f32(&meta, "u32"), None);
    assert_eq!(GgufToAprQ4KConverter::get_f32(&meta, "missing"), None);
}
