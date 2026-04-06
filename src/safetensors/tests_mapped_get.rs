
// ========== MappedSafeTensorsModel tests ==========

#[cfg(not(target_arch = "wasm32"))]
mod mapped_tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_mapped_get_tensor_auto_not_found() {
        let file = create_temp_safetensors(&[(
            "weight",
            SafetensorsDtype::F32,
            &[1],
            &1.0f32.to_le_bytes(),
        )]);

        let model = MappedSafeTensorsModel::load(file.path()).expect("load");
        let result = model.get_tensor_auto("nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_mapped_tensor_names() {
        let file = create_temp_safetensors(&[
            (
                "weight1",
                SafetensorsDtype::F32,
                &[1],
                &1.0f32.to_le_bytes(),
            ),
            (
                "weight2",
                SafetensorsDtype::F32,
                &[1],
                &2.0f32.to_le_bytes(),
            ),
        ]);

        let model = MappedSafeTensorsModel::load(file.path()).expect("load");
        let names = model.tensor_names();
        assert_eq!(names.len(), 2);
        assert!(names.contains(&"weight1"));
        assert!(names.contains(&"weight2"));
    }

    #[test]
    fn test_mapped_get_tensor_info() {
        let tensor_data: Vec<u8> = [1.0f32, 2.0f32, 3.0f32]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let file =
            create_temp_safetensors(&[("weight", SafetensorsDtype::F32, &[3], &tensor_data)]);

        let model = MappedSafeTensorsModel::load(file.path()).expect("load");
        let info = model.get_tensor_info("weight");
        assert!(info.is_some());
        let info = info.expect("tensor info");
        assert_eq!(info.dtype, SafetensorsDtype::F32);
        assert_eq!(info.shape, vec![3]);

        assert!(model.get_tensor_info("nonexistent").is_none());
    }

    #[test]
    fn test_mapped_path() {
        let file = create_temp_safetensors(&[(
            "weight",
            SafetensorsDtype::F32,
            &[1],
            &1.0f32.to_le_bytes(),
        )]);

        let model = MappedSafeTensorsModel::load(file.path()).expect("load");
        assert_eq!(model.path(), file.path());
    }

    #[test]
    fn test_mapped_file_size() {
        let tensor_data: Vec<u8> = [1.0f32, 2.0f32]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let file =
            create_temp_safetensors(&[("weight", SafetensorsDtype::F32, &[2], &tensor_data)]);

        let model = MappedSafeTensorsModel::load(file.path()).expect("load");
        // File size = 8 (header) + json length + 8 (tensor data)
        assert!(model.file_size() > 8);
    }

    #[test]
    fn test_mapped_tensor_count() {
        let file = create_temp_safetensors(&[
            (
                "weight1",
                SafetensorsDtype::F32,
                &[1],
                &1.0f32.to_le_bytes(),
            ),
            (
                "weight2",
                SafetensorsDtype::F32,
                &[1],
                &2.0f32.to_le_bytes(),
            ),
            (
                "weight3",
                SafetensorsDtype::F32,
                &[1],
                &3.0f32.to_le_bytes(),
            ),
        ]);

        let model = MappedSafeTensorsModel::load(file.path()).expect("load");
        assert_eq!(model.tensor_count(), 3);
    }

    #[test]
    fn test_mapped_metadata_key_skipped() {
        // Test that __metadata__ key is skipped in mapped model
        let mut file = tempfile::NamedTempFile::new().expect("temp file");
        let json = r#"{
                "__metadata__":{"format":"pt"},
                "weight":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}
            }"#;
        file.write_all(&(json.len() as u64).to_le_bytes())
            .expect("write header");
        file.write_all(json.as_bytes()).expect("write json");
        file.write_all(&[0u8; 8]).expect("write data");
        file.flush().expect("flush");

        let model = MappedSafeTensorsModel::load(file.path()).expect("load");
        assert_eq!(model.tensor_count(), 1);
        assert!(model.has_tensor("weight"));
        assert!(!model.has_tensor("__metadata__"));
    }

    #[test]
    fn test_mapped_multiple_tensors() {
        let w1: Vec<u8> = [1.0f32, 2.0f32]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let w2: Vec<u8> = [half::f16::from_f32(3.0), half::f16::from_f32(4.0)]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let w3: Vec<u8> = [half::bf16::from_f32(5.0)]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();

        let file = create_temp_safetensors(&[
            ("f32_tensor", SafetensorsDtype::F32, &[2], &w1),
            ("f16_tensor", SafetensorsDtype::F16, &[2], &w2),
            ("bf16_tensor", SafetensorsDtype::BF16, &[1], &w3),
        ]);

        let model = MappedSafeTensorsModel::load(file.path()).expect("load");
        assert_eq!(model.tensor_count(), 3);

        let f32_vals = model.get_tensor_f32("f32_tensor").expect("f32");
        assert_eq!(f32_vals, vec![1.0, 2.0]);

        let f16_vals = model.get_tensor_f16_as_f32("f16_tensor").expect("f16");
        assert!((f16_vals[0] - 3.0).abs() < 0.01);
        assert!((f16_vals[1] - 4.0).abs() < 0.01);

        let bf16_vals = model.get_tensor_bf16_as_f32("bf16_tensor").expect("bf16");
        assert!((bf16_vals[0] - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_mapped_empty_tensors() {
        // Test with empty tensor list (only {} metadata)
        let mut file = tempfile::NamedTempFile::new().expect("temp file");
        let json = b"{}";
        file.write_all(&(json.len() as u64).to_le_bytes())
            .expect("write header");
        file.write_all(json).expect("write json");
        file.flush().expect("flush");

        let model = MappedSafeTensorsModel::load(file.path()).expect("load");
        assert_eq!(model.tensor_count(), 0);
        assert!(model.tensor_names().is_empty());
    }
include!("tests_temp_file_helper.rs");
}
