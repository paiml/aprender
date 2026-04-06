
// ============================================================================
// ShardedSafeTensorsModel: load_from_index with temp files
// ============================================================================

#[cfg(not(target_arch = "wasm32"))]
mod sharded_tests {
    use super::*;
    use std::io::Write;

    /// Helper: create a single shard file with given tensors
    fn create_shard(
        dir: &std::path::Path,
        filename: &str,
        tensors: &[(&str, SafetensorsDtype, &[usize], &[u8])],
    ) {
        let mut json_map = serde_json::Map::new();
        let mut tensor_data = Vec::new();
        let mut offset = 0usize;

        for (name, dtype, shape, data) in tensors {
            let dtype_str = match dtype {
                SafetensorsDtype::F32 => "F32",
                SafetensorsDtype::F16 => "F16",
                SafetensorsDtype::BF16 => "BF16",
                _ => "F32",
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

        let json_str = serde_json::to_string(&json_map).expect("serialize");
        let json_bytes = json_str.as_bytes();

        let path = dir.join(filename);
        let mut f = std::fs::File::create(&path).expect("create shard");
        f.write_all(&(json_bytes.len() as u64).to_le_bytes())
            .expect("header");
        f.write_all(json_bytes).expect("metadata");
        f.write_all(&tensor_data).expect("data");
        f.flush().expect("flush");
    }

    #[test]
    fn test_sharded_basic_load() {
        let dir = tempfile::tempdir().expect("tmpdir");

        // Create two shard files
        let w1: Vec<u8> = [1.0f32, 2.0f32]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let w2: Vec<u8> = [3.0f32, 4.0f32]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();

        create_shard(
            dir.path(),
            "model-00001-of-00002.safetensors",
            &[("layer.0.weight", SafetensorsDtype::F32, &[2], &w1)],
        );
        create_shard(
            dir.path(),
            "model-00002-of-00002.safetensors",
            &[("layer.1.weight", SafetensorsDtype::F32, &[2], &w2)],
        );

        // Create index.json
        let index = serde_json::json!({
            "metadata": {"total_size": 16},
            "weight_map": {
                "layer.0.weight": "model-00001-of-00002.safetensors",
                "layer.1.weight": "model-00002-of-00002.safetensors"
            }
        });
        let index_path = dir.path().join("model.safetensors.index.json");
        std::fs::write(&index_path, serde_json::to_string(&index).expect("json"))
            .expect("write index");

        let model = ShardedSafeTensorsModel::load_from_index(&index_path).expect("load");
        assert_eq!(model.tensor_count(), 2);
        assert_eq!(model.shard_count(), 2);
        assert!(model.has_tensor("layer.0.weight"));
        assert!(model.has_tensor("layer.1.weight"));
        assert!(!model.has_tensor("nonexistent"));

        let names = model.tensor_names();
        assert_eq!(names.len(), 2);
    }

    #[test]
    fn test_sharded_get_tensor_auto() {
        let dir = tempfile::tempdir().expect("tmpdir");

        let w1: Vec<u8> = [42.0f32, 43.0f32]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();

        create_shard(
            dir.path(),
            "shard-001.safetensors",
            &[("attn.weight", SafetensorsDtype::F32, &[2], &w1)],
        );

        let index = serde_json::json!({
            "weight_map": {
                "attn.weight": "shard-001.safetensors"
            }
        });
        let index_path = dir.path().join("model.safetensors.index.json");
        std::fs::write(&index_path, serde_json::to_string(&index).expect("json"))
            .expect("write index");

        let model = ShardedSafeTensorsModel::load_from_index(&index_path).expect("load");
        let values = model.get_tensor_auto("attn.weight").expect("get");
        assert_eq!(values, vec![42.0, 43.0]);
    }

    #[test]
    fn test_sharded_get_tensor_auto_not_found() {
        let dir = tempfile::tempdir().expect("tmpdir");

        let w1: Vec<u8> = [1.0f32].iter().flat_map(|v| v.to_le_bytes()).collect();
        create_shard(
            dir.path(),
            "shard.safetensors",
            &[("w", SafetensorsDtype::F32, &[1], &w1)],
        );

        let index = serde_json::json!({
            "weight_map": { "w": "shard.safetensors" }
        });
        let index_path = dir.path().join("model.safetensors.index.json");
        std::fs::write(&index_path, serde_json::to_string(&index).expect("json")).expect("write");

        let model = ShardedSafeTensorsModel::load_from_index(&index_path).expect("load");
        let result = model.get_tensor_auto("nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_sharded_get_tensor_info() {
        let dir = tempfile::tempdir().expect("tmpdir");

        let w1: Vec<u8> = [1.0f32, 2.0, 3.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        create_shard(
            dir.path(),
            "shard.safetensors",
            &[("matrix", SafetensorsDtype::F32, &[3], &w1)],
        );

        let index = serde_json::json!({
            "weight_map": { "matrix": "shard.safetensors" }
        });
        let index_path = dir.path().join("model.safetensors.index.json");
        std::fs::write(&index_path, serde_json::to_string(&index).expect("json")).expect("write");

        let model = ShardedSafeTensorsModel::load_from_index(&index_path).expect("load");
        let info = model.get_tensor_info("matrix").expect("info");
        assert_eq!(info.shape, vec![3]);
        assert_eq!(info.dtype, SafetensorsDtype::F32);

        assert!(model.get_tensor_info("nonexistent").is_none());
    }

    #[test]
    fn test_sharded_path() {
        let dir = tempfile::tempdir().expect("tmpdir");

        let w1: Vec<u8> = [1.0f32].iter().flat_map(|v| v.to_le_bytes()).collect();
        create_shard(
            dir.path(),
            "shard.safetensors",
            &[("w", SafetensorsDtype::F32, &[1], &w1)],
        );

        let index = serde_json::json!({
            "weight_map": { "w": "shard.safetensors" }
        });
        let index_path = dir.path().join("model.safetensors.index.json");
        std::fs::write(&index_path, serde_json::to_string(&index).expect("json")).expect("write");

        let model = ShardedSafeTensorsModel::load_from_index(&index_path).expect("load");
        assert_eq!(model.path(), dir.path());
    }

    #[test]
    fn test_sharded_multiple_tensors_same_shard() {
        let dir = tempfile::tempdir().expect("tmpdir");

        let w1: Vec<u8> = [1.0f32, 2.0].iter().flat_map(|v| v.to_le_bytes()).collect();
        let w2: Vec<u8> = [3.0f32].iter().flat_map(|v| v.to_le_bytes()).collect();

        create_shard(
            dir.path(),
            "shard.safetensors",
            &[
                ("a", SafetensorsDtype::F32, &[2], &w1),
                ("b", SafetensorsDtype::F32, &[1], &w2),
            ],
        );

        let index = serde_json::json!({
            "weight_map": {
                "a": "shard.safetensors",
                "b": "shard.safetensors"
            }
        });
        let index_path = dir.path().join("model.safetensors.index.json");
        std::fs::write(&index_path, serde_json::to_string(&index).expect("json")).expect("write");

        let model = ShardedSafeTensorsModel::load_from_index(&index_path).expect("load");
        assert_eq!(model.tensor_count(), 2);
        assert_eq!(model.shard_count(), 1); // same shard, deduplicated

        let va = model.get_tensor_auto("a").expect("a");
        assert_eq!(va, vec![1.0, 2.0]);
        let vb = model.get_tensor_auto("b").expect("b");
        assert_eq!(vb, vec![3.0]);
    }

    #[test]
    fn test_sharded_index_not_found() {
        let result = ShardedSafeTensorsModel::load_from_index(std::path::Path::new(
            "/nonexistent/index.json",
        ));
        assert!(result.is_err());
    }

    #[test]
    fn test_sharded_index_invalid_json() {
        let dir = tempfile::tempdir().expect("tmpdir");
        let index_path = dir.path().join("model.safetensors.index.json");
        std::fs::write(&index_path, "not valid json").expect("write");

        let result = ShardedSafeTensorsModel::load_from_index(&index_path);
        assert!(result.is_err());
    }

    #[test]
    fn test_sharded_shard_file_missing() {
        let dir = tempfile::tempdir().expect("tmpdir");

        let index = serde_json::json!({
            "weight_map": { "w": "nonexistent.safetensors" }
        });
        let index_path = dir.path().join("model.safetensors.index.json");
        std::fs::write(&index_path, serde_json::to_string(&index).expect("json")).expect("write");

        let result = ShardedSafeTensorsModel::load_from_index(&index_path);
        assert!(result.is_err());
    }

    #[test]
    fn test_sharded_debug_format() {
        let dir = tempfile::tempdir().expect("tmpdir");

        let w: Vec<u8> = [1.0f32].iter().flat_map(|v| v.to_le_bytes()).collect();
        create_shard(
            dir.path(),
            "shard.safetensors",
            &[("w", SafetensorsDtype::F32, &[1], &w)],
        );

        let index = serde_json::json!({
            "weight_map": { "w": "shard.safetensors" }
        });
        let index_path = dir.path().join("model.safetensors.index.json");
        std::fs::write(&index_path, serde_json::to_string(&index).expect("json")).expect("write");

        let model = ShardedSafeTensorsModel::load_from_index(&index_path).expect("load");
        let debug = format!("{:?}", model);
        assert!(debug.contains("ShardedSafeTensorsModel"));
    }

    #[test]
    fn test_sharded_bf16_tensor_cross_shard() {
        let dir = tempfile::tempdir().expect("tmpdir");

        let w_f32: Vec<u8> = [1.0f32, 2.0].iter().flat_map(|v| v.to_le_bytes()).collect();
        let w_bf16: Vec<u8> = [half::bf16::from_f32(3.0), half::bf16::from_f32(4.0)]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();

        create_shard(
            dir.path(),
            "shard-001.safetensors",
            &[("f32_tensor", SafetensorsDtype::F32, &[2], &w_f32)],
        );
        create_shard(
            dir.path(),
            "shard-002.safetensors",
            &[("bf16_tensor", SafetensorsDtype::BF16, &[2], &w_bf16)],
        );

        let index = serde_json::json!({
            "weight_map": {
                "f32_tensor": "shard-001.safetensors",
                "bf16_tensor": "shard-002.safetensors"
            }
        });
        let index_path = dir.path().join("model.safetensors.index.json");
        std::fs::write(&index_path, serde_json::to_string(&index).expect("json")).expect("write");

        let model = ShardedSafeTensorsModel::load_from_index(&index_path).expect("load");

        let f32_vals = model.get_tensor_auto("f32_tensor").expect("f32");
        assert_eq!(f32_vals, vec![1.0, 2.0]);

        let bf16_vals = model.get_tensor_auto("bf16_tensor").expect("bf16");
        assert_eq!(bf16_vals.len(), 2);
        assert!((bf16_vals[0] - 3.0).abs() < 0.1);
        assert!((bf16_vals[1] - 4.0).abs() < 0.1);
    }
}
