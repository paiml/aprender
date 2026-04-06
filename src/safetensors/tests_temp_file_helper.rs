
    /// Helper to create a temporary safetensors file
    fn create_temp_safetensors(
        tensors: &[(&str, SafetensorsDtype, &[usize], &[u8])],
    ) -> tempfile::NamedTempFile {
        let mut json_map = serde_json::Map::new();
        let mut tensor_data = Vec::new();
        let mut offset = 0usize;

        for (name, dtype, shape, data) in tensors {
            let dtype_str = match dtype {
                SafetensorsDtype::F32 => "F32",
                SafetensorsDtype::F16 => "F16",
                SafetensorsDtype::BF16 => "BF16",
                SafetensorsDtype::I32 => "I32",
                SafetensorsDtype::I64 => "I64",
                SafetensorsDtype::U8 => "U8",
                SafetensorsDtype::Bool => "Bool",
            };

            let end = offset + data.len();
            json_map.insert(
                (*name).to_string(),
                serde_json::json!({
                    "dtype": dtype_str,
                    "shape": shape,
                    "data_offsets": [offset, end]
                }),
            );

            tensor_data.extend_from_slice(data);
            offset = end;
        }

        let json_str = serde_json::to_string(&json_map).expect("JSON serialization");
        let json_bytes = json_str.as_bytes();

        let mut file = tempfile::NamedTempFile::new().expect("temp file creation");
        file.write_all(&(json_bytes.len() as u64).to_le_bytes())
            .expect("write header");
        file.write_all(json_bytes).expect("write metadata");
        file.write_all(&tensor_data).expect("write tensor data");
        file.flush().expect("flush file");

        file
    }

    #[test]
    fn test_mapped_load_basic() {
        // Create temp file with one F32 tensor
        let tensor_data: Vec<u8> = [1.0f32, 2.0f32]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let file =
            create_temp_safetensors(&[("weight", SafetensorsDtype::F32, &[2], &tensor_data)]);

        let model = MappedSafeTensorsModel::load(file.path()).expect("load");
        assert_eq!(model.tensor_count(), 1);
        assert!(model.has_tensor("weight"));
        assert!(!model.has_tensor("nonexistent"));
    }

    #[test]
    fn test_mapped_file_not_found() {
        let result = MappedSafeTensorsModel::load("/nonexistent/path/model.safetensors");
        assert!(result.is_err());
    }

    #[test]
    fn test_mapped_file_too_small() {
        // Create a file with only 4 bytes (less than header size)
        let mut file = tempfile::NamedTempFile::new().expect("temp file");
        file.write_all(&[0u8; 4]).expect("write");
        file.flush().expect("flush");

        let result = MappedSafeTensorsModel::load(file.path());
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            format!("{err:?}").contains("File too small"),
            "Expected 'File too small' error, got: {err:?}"
        );
    }

    #[test]
    fn test_mapped_truncated_metadata() {
        // Create a file that claims metadata is 100 bytes but only has 10
        let mut file = tempfile::NamedTempFile::new().expect("temp file");
        file.write_all(&100u64.to_le_bytes()).expect("write header");
        file.write_all(b"{}").expect("write short json");
        file.flush().expect("flush");

        let result = MappedSafeTensorsModel::load(file.path());
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            format!("{err:?}").contains("truncated"),
            "Expected 'truncated' error, got: {err:?}"
        );
    }

    #[test]
    fn test_mapped_invalid_json() {
        let mut file = tempfile::NamedTempFile::new().expect("temp file");
        let invalid_json = b"not valid json!!";
        file.write_all(&(invalid_json.len() as u64).to_le_bytes())
            .expect("write header");
        file.write_all(invalid_json).expect("write json");
        file.flush().expect("flush");

        let result = MappedSafeTensorsModel::load(file.path());
        assert!(result.is_err());
    }

    #[test]
    fn test_mapped_json_not_object() {
        let mut file = tempfile::NamedTempFile::new().expect("temp file");
        let json = b"[]"; // Array instead of object
        file.write_all(&(json.len() as u64).to_le_bytes())
            .expect("write header");
        file.write_all(json).expect("write json");
        file.flush().expect("flush");

        let result = MappedSafeTensorsModel::load(file.path());
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            format!("{err:?}").contains("Expected JSON object"),
            "Expected 'Expected JSON object' error, got: {err:?}"
        );
    }

    #[test]
    fn test_mapped_tensor_metadata_parse_error() {
        let mut file = tempfile::NamedTempFile::new().expect("temp file");
        // Missing dtype field
        let json = r#"{"weight":{"shape":[2],"data_offsets":[0,8]}}"#;
        file.write_all(&(json.len() as u64).to_le_bytes())
            .expect("write header");
        file.write_all(json.as_bytes()).expect("write json");
        file.write_all(&[0u8; 8]).expect("write data");
        file.flush().expect("flush");

        let result = MappedSafeTensorsModel::load(file.path());
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            format!("{err:?}").contains("Failed to parse tensor"),
            "Expected tensor parse error, got: {err:?}"
        );
    }

    #[test]
    fn test_mapped_get_tensor_bytes() {
        let tensor_data: Vec<u8> = [1.0f32, 2.0f32, 3.0f32]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let file =
            create_temp_safetensors(&[("weight", SafetensorsDtype::F32, &[3], &tensor_data)]);

        let model = MappedSafeTensorsModel::load(file.path()).expect("load");
        let bytes = model.get_tensor_bytes("weight").expect("get bytes");
        assert_eq!(bytes.len(), 12); // 3 * 4 bytes
    }

    #[test]
    fn test_mapped_get_tensor_bytes_not_found() {
        let file = create_temp_safetensors(&[(
            "weight",
            SafetensorsDtype::F32,
            &[1],
            &0.0f32.to_le_bytes(),
        )]);

        let model = MappedSafeTensorsModel::load(file.path()).expect("load");
        let result = model.get_tensor_bytes("nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_mapped_get_tensor_bytes_offset_exceeds() {
        // Create a file with a tensor that claims more data than exists
        let mut file = tempfile::NamedTempFile::new().expect("temp file");
        let json = r#"{"weight":{"dtype":"F32","shape":[100],"data_offsets":[0,400]}}"#;
        file.write_all(&(json.len() as u64).to_le_bytes())
            .expect("write header");
        file.write_all(json.as_bytes()).expect("write json");
        file.write_all(&[0u8; 8])
            .expect("write only 8 bytes of data");
        file.flush().expect("flush");

        // GH-213: Truncated files are now caught at load time (Layer 3 safety net)
        let result = MappedSafeTensorsModel::load(file.path());
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            format!("{err:?}").contains("truncated"),
            "Expected 'truncated' error at load time, got: {err:?}"
        );
    }

    #[test]
    fn test_mapped_get_tensor_f32() {
        let tensor_data: Vec<u8> = [1.0f32, 2.0f32, 3.0f32, 4.0f32]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let file =
            create_temp_safetensors(&[("weight", SafetensorsDtype::F32, &[4], &tensor_data)]);

        let model = MappedSafeTensorsModel::load(file.path()).expect("load");
        let values = model.get_tensor_f32("weight").expect("get f32");
        assert_eq!(values, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_mapped_get_tensor_f32_not_found() {
        let file = create_temp_safetensors(&[(
            "weight",
            SafetensorsDtype::F32,
            &[1],
            &1.0f32.to_le_bytes(),
        )]);

        let model = MappedSafeTensorsModel::load(file.path()).expect("load");
        let result = model.get_tensor_f32("nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_mapped_get_tensor_f32_wrong_dtype() {
        let file = create_temp_safetensors(&[("weight", SafetensorsDtype::I32, &[2], &[0u8; 8])]);

        let model = MappedSafeTensorsModel::load(file.path()).expect("load");
        let result = model.get_tensor_f32("weight");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            format!("{err:?}").contains("expected F32"),
            "Expected wrong dtype error, got: {err:?}"
        );
    }

    #[test]
    fn test_mapped_get_tensor_f32_not_multiple_of_4() {
        // Create file with misaligned data
        let mut file = tempfile::NamedTempFile::new().expect("temp file");
        let json = r#"{"weight":{"dtype":"F32","shape":[1],"data_offsets":[0,7]}}"#;
        file.write_all(&(json.len() as u64).to_le_bytes())
            .expect("write header");
        file.write_all(json.as_bytes()).expect("write json");
        file.write_all(&[0u8; 7]).expect("write 7 bytes");
        file.flush().expect("flush");

        let model = MappedSafeTensorsModel::load(file.path()).expect("load");
        let result = model.get_tensor_f32("weight");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            format!("{err:?}").contains("not a multiple of 4"),
            "Expected alignment error, got: {err:?}"
        );
    }

    #[test]
    fn test_mapped_get_tensor_f16_bytes() {
        let tensor_data: Vec<u8> = [half::f16::from_f32(1.0), half::f16::from_f32(2.0)]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let file =
            create_temp_safetensors(&[("weight", SafetensorsDtype::F16, &[2], &tensor_data)]);

        let model = MappedSafeTensorsModel::load(file.path()).expect("load");
        let bytes = model.get_tensor_f16_bytes("weight").expect("get f16 bytes");
        assert_eq!(bytes.len(), 4); // 2 * 2 bytes
    }

    #[test]
    fn test_mapped_get_tensor_f16_bytes_not_found() {
        let file = create_temp_safetensors(&[("weight", SafetensorsDtype::F16, &[1], &[0u8; 2])]);

        let model = MappedSafeTensorsModel::load(file.path()).expect("load");
        let result = model.get_tensor_f16_bytes("nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_mapped_get_tensor_f16_bytes_wrong_dtype() {
        let file = create_temp_safetensors(&[(
            "weight",
            SafetensorsDtype::F32,
            &[1],
            &1.0f32.to_le_bytes(),
        )]);

        let model = MappedSafeTensorsModel::load(file.path()).expect("load");
        let result = model.get_tensor_f16_bytes("weight");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            format!("{err:?}").contains("expected F16"),
            "Expected wrong dtype error, got: {err:?}"
        );
    }

    #[test]
    fn test_mapped_get_tensor_f16_as_f32() {
        let tensor_data: Vec<u8> = [half::f16::from_f32(1.0), half::f16::from_f32(2.0)]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let file =
            create_temp_safetensors(&[("weight", SafetensorsDtype::F16, &[2], &tensor_data)]);

        let model = MappedSafeTensorsModel::load(file.path()).expect("load");
        let values = model
            .get_tensor_f16_as_f32("weight")
            .expect("get f16 as f32");
        assert_eq!(values.len(), 2);
        assert!((values[0] - 1.0).abs() < 0.01);
        assert!((values[1] - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_mapped_get_tensor_bf16_bytes() {
        let tensor_data: Vec<u8> = [half::bf16::from_f32(1.0), half::bf16::from_f32(2.0)]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let file =
            create_temp_safetensors(&[("weight", SafetensorsDtype::BF16, &[2], &tensor_data)]);

        let model = MappedSafeTensorsModel::load(file.path()).expect("load");
        let bytes = model
            .get_tensor_bf16_bytes("weight")
            .expect("get bf16 bytes");
        assert_eq!(bytes.len(), 4); // 2 * 2 bytes
    }

    #[test]
    fn test_mapped_get_tensor_bf16_bytes_not_found() {
        let file = create_temp_safetensors(&[("weight", SafetensorsDtype::BF16, &[1], &[0u8; 2])]);

        let model = MappedSafeTensorsModel::load(file.path()).expect("load");
        let result = model.get_tensor_bf16_bytes("nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_mapped_get_tensor_bf16_bytes_wrong_dtype() {
        let file = create_temp_safetensors(&[(
            "weight",
            SafetensorsDtype::F32,
            &[1],
            &1.0f32.to_le_bytes(),
        )]);

        let model = MappedSafeTensorsModel::load(file.path()).expect("load");
        let result = model.get_tensor_bf16_bytes("weight");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            format!("{err:?}").contains("expected BF16"),
            "Expected wrong dtype error, got: {err:?}"
        );
    }

    #[test]
    fn test_mapped_get_tensor_bf16_as_f32() {
        let tensor_data: Vec<u8> = [half::bf16::from_f32(1.0), half::bf16::from_f32(2.0)]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let file =
            create_temp_safetensors(&[("weight", SafetensorsDtype::BF16, &[2], &tensor_data)]);

        let model = MappedSafeTensorsModel::load(file.path()).expect("load");
        let values = model
            .get_tensor_bf16_as_f32("weight")
            .expect("get bf16 as f32");
        assert_eq!(values.len(), 2);
        assert!((values[0] - 1.0).abs() < 0.01);
        assert!((values[1] - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_mapped_get_tensor_auto_f32() {
        let tensor_data: Vec<u8> = [1.0f32, 2.0f32]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let file =
            create_temp_safetensors(&[("weight", SafetensorsDtype::F32, &[2], &tensor_data)]);

        let model = MappedSafeTensorsModel::load(file.path()).expect("load");
        let values = model.get_tensor_auto("weight").expect("get auto");
        assert_eq!(values, vec![1.0, 2.0]);
    }

    #[test]
    fn test_mapped_get_tensor_auto_f16() {
        let tensor_data: Vec<u8> = [half::f16::from_f32(1.0), half::f16::from_f32(2.0)]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let file =
            create_temp_safetensors(&[("weight", SafetensorsDtype::F16, &[2], &tensor_data)]);

        let model = MappedSafeTensorsModel::load(file.path()).expect("load");
        let values = model.get_tensor_auto("weight").expect("get auto");
        assert_eq!(values.len(), 2);
    }

    #[test]
    fn test_mapped_get_tensor_auto_bf16() {
        let tensor_data: Vec<u8> = [half::bf16::from_f32(1.0), half::bf16::from_f32(2.0)]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let file =
            create_temp_safetensors(&[("weight", SafetensorsDtype::BF16, &[2], &tensor_data)]);

        let model = MappedSafeTensorsModel::load(file.path()).expect("load");
        let values = model.get_tensor_auto("weight").expect("get auto");
        assert_eq!(values.len(), 2);
    }

    #[test]
    fn test_mapped_get_tensor_auto_unsupported() {
        let file = create_temp_safetensors(&[("weight", SafetensorsDtype::I32, &[2], &[0u8; 8])]);

        let model = MappedSafeTensorsModel::load(file.path()).expect("load");
        let result = model.get_tensor_auto("weight");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            format!("{err:?}").contains("Unsupported dtype"),
            "Expected unsupported dtype error, got: {err:?}"
        );
    }
