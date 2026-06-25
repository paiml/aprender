use super::*;

/// GH-200: AprV2ReaderRef also handles Q6K.
#[test]
fn test_q6k_via_ref_reader() {
    let raw_q6k = vec![0u8; 210];
    let mut writer = AprV2Writer::new(AprV2Metadata::new("test"));
    writer.add_q6k_raw_tensor("ref_q6k", vec![16, 16], raw_q6k);
    let bytes = writer.write().expect("write APR");

    let reader = AprV2ReaderRef::from_bytes(&bytes).expect("parse APR ref");
    let f32_data = reader.get_tensor_as_f32("ref_q6k");
    assert!(f32_data.is_some(), "Q6K via ref reader must dequantize");
    assert_eq!(f32_data.unwrap().len(), 256);
}

#[path = "tests_layout.rs"]
mod tests_layout;
