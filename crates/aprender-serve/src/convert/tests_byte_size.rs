
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

// ---------------------------------------------------------------------------
// FALSIFY-QTYPE-001: the converter's byte-size table must agree with the
// reader's, and must refuse to guess.
//
// Every test ABOVE this line re-implements the arithmetic inline --
//
//     let byte_size = num_elements.div_ceil(32) * 34;
//     assert_eq!(byte_size, 32 * 34);
//
// -- which asserts an expression against itself and never calls
// `ggml_tensor_byte_size_h` at all. That is why none of them noticed that the
// production function had no arm for Q2_K, Q3_K or BF16 and silently fell back
// to `num_elements * 4`. These call the real function.
// ---------------------------------------------------------------------------

use crate::convert::GgufToAprQ4KConverter as Conv;
use crate::quantize::QK_K;

/// The size that `gguf/metadata.rs` uses to READ a Q2_K tensor. If the
/// converter disagrees with the reader, one of them walks off the end of the
/// tensor.
const Q2_K_SUPER_BLOCK_BYTES: usize = 84;
const Q3_K_SUPER_BLOCK_BYTES: usize = 110;

#[test]
fn q2_k_is_sized_by_its_super_block_not_as_f32() {
    let n = 1024usize;
    let got = Conv::ggml_tensor_byte_size_h(crate::gguf::GGUF_TYPE_Q2_K, n)
        .expect("Q2_K is a type this converter must know");

    assert_eq!(
        got,
        n.div_ceil(QK_K) * Q2_K_SUPER_BLOCK_BYTES,
        "Q2_K must agree with the super-block size gguf/metadata.rs reads with"
    );

    // The specific regression: the old `_ => num_elements * 4` returned 4096
    // here instead of 336 -- 12.2x too many bytes, which slices past this
    // tensor into the next one.
    assert_ne!(got, n * 4, "Q2_K is being sized as F32 again");
    assert_eq!(got, 336);
}

#[test]
fn q3_k_is_sized_by_its_super_block_not_as_f32() {
    let n = 1024usize;
    let got = Conv::ggml_tensor_byte_size_h(crate::gguf::GGUF_TYPE_Q3_K, n)
        .expect("Q3_K is a type this converter must know");
    assert_eq!(got, n.div_ceil(QK_K) * Q3_K_SUPER_BLOCK_BYTES);
    assert_ne!(got, n * 4, "Q3_K is being sized as F32 again");
}

#[test]
fn bf16_is_two_bytes_per_element_not_four() {
    let n = 1024usize;
    let got = Conv::ggml_tensor_byte_size_h(crate::gguf::GGUF_TYPE_BF16, n)
        .expect("BF16 is a type this converter must know");
    assert_eq!(got, n * 2, "BF16 is 2 bytes/element");
    assert_ne!(got, n * 4, "BF16 is being sized as F32 again");
}

/// Non-vacuity companion, and the load-bearing one. Every assertion above
/// would still pass if the function kept a permissive `_ => num_elements * 4`
/// arm and merely gained the three missing types -- the silent fallback would
/// survive for the NEXT unlisted type. This proves it is gone.
#[test]
fn an_unknown_ggml_type_is_an_error_not_a_guess() {
    // 16 is IQ2_XXS. The point is not that type specifically -- it is that a
    // type this converter cannot size must SAY SO rather than assume F32.
    let err = Conv::ggml_tensor_byte_size_h(16, 1024);
    assert!(
        err.is_err(),
        "an unsizable ggml type returned Ok, so the silent F32 fallback is back"
    );

    // And a known type must still succeed, or the arm above could be satisfied
    // by a function that simply fails for everything.
    assert!(
        Conv::ggml_tensor_byte_size_h(crate::gguf::GGUF_TYPE_Q4_K, 1024).is_ok(),
        "Q4_K must still be sizable"
    );
}
