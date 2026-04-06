
// ========================================================================
// unfuse_qkv_tensors: Passthrough when no fused tensors present
// ========================================================================

#[test]
fn test_unfuse_qkv_tensors_no_fused_passthrough() {
    let mut tensors: BTreeMap<String, (Vec<f32>, Vec<usize>)> = BTreeMap::new();
    tensors.insert(
        "model.layers.0.self_attn.q_proj.weight".to_string(),
        (vec![1.0; 16], vec![4, 4]),
    );
    tensors.insert(
        "model.layers.0.self_attn.k_proj.weight".to_string(),
        (vec![2.0; 16], vec![4, 4]),
    );
    tensors.insert(
        "model.layers.0.self_attn.v_proj.weight".to_string(),
        (vec![3.0; 16], vec![4, 4]),
    );

    // Non-APR path means read_apr_metadata returns None, but since no
    // fused tensors exist, the early return fires first.
    let result = unfuse_qkv_tensors(tensors.clone(), Path::new("/tmp/fake.safetensors"));

    assert_eq!(result.len(), 3);
    assert!(result.contains_key("model.layers.0.self_attn.q_proj.weight"));
    assert!(result.contains_key("model.layers.0.self_attn.k_proj.weight"));
    assert!(result.contains_key("model.layers.0.self_attn.v_proj.weight"));
}

/// PMAT-260: Verify BF16 dtype preservation in SafeTensors round-trip.
#[test]
fn test_pmat_260_bf16_dtype_preserved_in_safetensors_export() {
    use crate::serialization::safetensors::{
        save_safetensors_typed, MappedSafeTensors,
    };

    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("test_bf16.safetensors");

    let mut tensors = BTreeMap::new();
    // Values that are exact in BF16 (upper 16 bits of F32)
    tensors.insert(
        "weight".to_string(),
        (vec![1.0_f32, -2.0, 0.5, 0.0], vec![2, 2]),
    );
    tensors.insert(
        "bias".to_string(),
        (vec![0.1, 0.2, 0.3], vec![3]),
    );

    let mut dtypes = BTreeMap::new();
    dtypes.insert("weight".to_string(), "BF16".to_string());
    // bias has no original dtype → should default to F32

    save_safetensors_typed(&path, &tensors, &dtypes).expect("save");

    let mapped = MappedSafeTensors::open(&path).expect("open");
    let dtype_map = mapped.dtype_map();

    assert_eq!(dtype_map.get("weight").map(String::as_str), Some("BF16"));
    assert_eq!(dtype_map.get("bias").map(String::as_str), Some("F32"));

    // Verify data round-trips correctly
    let weight_data = mapped.get_tensor("weight").expect("read weight");
    assert_eq!(weight_data, vec![1.0_f32, -2.0, 0.5, 0.0]);

    let bias_data = mapped.get_tensor("bias").expect("read bias");
    assert_eq!(bias_data, vec![0.1_f32, 0.2, 0.3]);
}

/// PMAT-260: Verify F16 dtype preservation in SafeTensors round-trip.
#[test]
fn test_pmat_260_f16_dtype_preserved_in_safetensors_export() {
    use crate::serialization::safetensors::{
        save_safetensors_typed, MappedSafeTensors,
    };

    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("test_f16.safetensors");

    let mut tensors = BTreeMap::new();
    tensors.insert(
        "model.weight".to_string(),
        (vec![1.0_f32, -1.0, 0.0, 0.5], vec![2, 2]),
    );

    let mut dtypes = BTreeMap::new();
    dtypes.insert("model.weight".to_string(), "F16".to_string());

    save_safetensors_typed(&path, &tensors, &dtypes).expect("save");

    let mapped = MappedSafeTensors::open(&path).expect("open");
    assert_eq!(
        mapped.dtype_map().get("model.weight").map(String::as_str),
        Some("F16")
    );

    // Verify data round-trips correctly
    let data = mapped.get_tensor("model.weight").expect("read");
    assert_eq!(data, vec![1.0_f32, -1.0, 0.0, 0.5]);
}

/// PMAT-260: Verify that empty dtype map falls back to F32 (backward compat).
#[test]
fn test_pmat_260_empty_dtypes_defaults_to_f32() {
    use crate::serialization::safetensors::{
        save_safetensors_typed, MappedSafeTensors,
    };

    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("test_default_f32.safetensors");

    let mut tensors = BTreeMap::new();
    tensors.insert(
        "w".to_string(),
        (vec![1.0_f32, 2.0], vec![2]),
    );

    let dtypes = BTreeMap::new(); // empty = all F32

    save_safetensors_typed(&path, &tensors, &dtypes).expect("save");

    let mapped = MappedSafeTensors::open(&path).expect("open");
    assert_eq!(mapped.dtype_map().get("w").map(String::as_str), Some("F32"));
}

/// PMAT-260: Verify metadata variant also preserves dtypes.
#[test]
fn test_pmat_260_metadata_variant_preserves_bf16() {
    use crate::serialization::safetensors::{
        save_safetensors_with_metadata_typed, MappedSafeTensors, UserMetadata,
    };

    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("test_meta_bf16.safetensors");

    let mut tensors = BTreeMap::new();
    tensors.insert(
        "attn.weight".to_string(),
        (vec![1.0_f32, -0.5, 0.25, 0.0], vec![2, 2]),
    );

    let mut dtypes = BTreeMap::new();
    dtypes.insert("attn.weight".to_string(), "BF16".to_string());

    let mut user_meta = UserMetadata::new();
    user_meta.insert("format".to_string(), "pt".to_string());

    save_safetensors_with_metadata_typed(&path, &tensors, &user_meta, &dtypes)
        .expect("save");

    let mapped = MappedSafeTensors::open(&path).expect("open");
    assert_eq!(
        mapped.dtype_map().get("attn.weight").map(String::as_str),
        Some("BF16")
    );
    assert_eq!(
        mapped.user_metadata().get("format").map(String::as_str),
        Some("pt")
    );
}

/// PMAT-260: Verify extract_source_dtypes returns empty for non-safetensors.
#[test]
fn test_pmat_260_extract_source_dtypes_non_safetensors() {
    let dtypes = super::extract_source_dtypes(Path::new("/tmp/model.apr"));
    assert!(dtypes.is_empty());

    let dtypes = super::extract_source_dtypes(Path::new("/tmp/model.gguf"));
    assert!(dtypes.is_empty());
}

#[path = "export_tests_unfuse_qkv.rs"]
mod export_tests_unfuse_qkv;
#[path = "export_tests_infer_gqa.rs"]
mod export_tests_infer_gqa;
#[path = "export_tests_infer_attn.rs"]
mod export_tests_infer_attn;
#[path = "export_tests_metadata_gguf.rs"]
mod export_tests_metadata_gguf;
#[path = "export_tests_tied_gguf.rs"]
mod export_tests_tied_gguf;
#[path = "export_tests_e2e_gguf.rs"]
mod export_tests_e2e_gguf;
#[path = "export_tests_arch_metadata.rs"]
mod export_tests_arch_metadata;
#[path = "export_tests_unfuse_with_metadata.rs"]
mod export_tests_unfuse_with_metadata;
