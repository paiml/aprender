
#[test]
fn test_multiple_tensors() {
    let data1: Vec<f32> = vec![1.0; 4];
    let data2: Vec<f32> = vec![2.0; 8];
    let data = GGUFBuilder::new()
        .architecture("test")
        .add_f32_tensor("first", &[4], &data1)
        .add_f32_tensor("second", &[8], &data2)
        .build();

    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert_eq!(model.tensors.len(), 2);

    let first = model.get_tensor_f32("first", &data).expect("get first");
    let second = model.get_tensor_f32("second", &data).expect("get second");
    assert_eq!(first.len(), 4);
    assert_eq!(second.len(), 8);
}

// =============================================================================
// Empty/Zero Cases
// =============================================================================

#[test]
fn test_empty_metadata() {
    let mut data = Vec::new();
    data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    data.extend_from_slice(&GGUF_VERSION_V3.to_le_bytes());
    data.extend_from_slice(&0u64.to_le_bytes()); // tensor_count = 0
    data.extend_from_slice(&0u64.to_le_bytes()); // metadata_count = 0

    // Align to 32 bytes
    data.resize(32, 0);

    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert!(model.metadata.is_empty());
    assert!(model.tensors.is_empty());
}

#[test]
fn test_empty_string_metadata() {
    let bytes = 0u64.to_le_bytes().to_vec(); // empty string
    let data = build_gguf_with_metadata("test.empty", 8, bytes);
    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert!(matches!(
        model.metadata.get("test.empty"),
        Some(GGUFValue::String(v)) if v.is_empty()
    ));
}

#[test]
fn test_empty_array_metadata() {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&4u32.to_le_bytes()); // element_type = u32
    bytes.extend_from_slice(&0u64.to_le_bytes()); // array_len = 0

    let data = build_gguf_with_metadata("test.empty_arr", 9, bytes);
    let model = GGUFModel::from_bytes(&data).expect("parse");

    if let Some(GGUFValue::Array(arr)) = model.metadata.get("test.empty_arr") {
        assert!(arr.is_empty());
    } else {
        panic!("Expected Array");
    }
}

// =============================================================================
// Boundary Value Tests
// =============================================================================

#[test]
fn test_max_u8_value() {
    let data = build_gguf_with_metadata("test.max_u8", 0, vec![255u8]);
    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert!(matches!(
        model.metadata.get("test.max_u8"),
        Some(GGUFValue::UInt8(255))
    ));
}

#[test]
fn test_min_i8_value() {
    let data = build_gguf_with_metadata("test.min_i8", 1, vec![0x80u8]); // -128 as i8
    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert!(matches!(
        model.metadata.get("test.min_i8"),
        Some(GGUFValue::Int8(-128))
    ));
}

#[test]
fn test_max_u16_value() {
    let data = build_gguf_with_metadata("test.max_u16", 2, u16::MAX.to_le_bytes().to_vec());
    let model = GGUFModel::from_bytes(&data).expect("parse");
    assert!(matches!(
        model.metadata.get("test.max_u16"),
        Some(GGUFValue::UInt16(u16::MAX))
    ));
}

#[test]
fn test_special_float_nan() {
    let data = build_gguf_with_metadata("test.nan", 6, f32::NAN.to_le_bytes().to_vec());
    let model = GGUFModel::from_bytes(&data).expect("parse");
    if let Some(GGUFValue::Float32(v)) = model.metadata.get("test.nan") {
        assert!(v.is_nan());
    } else {
        panic!("Expected Float32");
    }
}

#[test]
fn test_special_float_infinity() {
    let data = build_gguf_with_metadata("test.inf", 6, f32::INFINITY.to_le_bytes().to_vec());
    let model = GGUFModel::from_bytes(&data).expect("parse");
    if let Some(GGUFValue::Float32(v)) = model.metadata.get("test.inf") {
        assert!(v.is_infinite() && v.is_sign_positive());
    } else {
        panic!("Expected Float32");
    }
}
