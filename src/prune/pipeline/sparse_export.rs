//! Sparse model export with sparsity metadata
//!
//! Exports pruned model weights along with a `sparsity_metadata.json` sidecar
//! containing per-tensor sparsity statistics.

use crate::prune::pipeline::metrics::PruningMetrics;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Result of sparse model export
#[derive(Debug, Clone)]
pub struct SparseExportResult {
    /// Path to the exported weight file
    pub weights_path: PathBuf,
    /// Path to the sparsity metadata sidecar
    pub metadata_path: PathBuf,
    /// Global sparsity ratio
    pub global_sparsity: f32,
    /// Number of tensors exported
    pub num_tensors: usize,
}

/// Sparsity metadata sidecar (serialized to sparsity_metadata.json)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparsityMetadata {
    /// Format version
    pub version: String,
    /// Global sparsity (fraction of zero parameters)
    pub global_sparsity: f32,
    /// Total parameters
    pub total_parameters: usize,
    /// Parameters pruned (zero)
    pub parameters_pruned: usize,
    /// Per-tensor sparsity information
    pub tensors: Vec<TensorSparsityInfo>,
}

/// Per-tensor sparsity statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorSparsityInfo {
    /// Tensor name
    pub name: String,
    /// Sparsity ratio for this tensor
    pub sparsity: f32,
    /// Number of zero elements
    pub zero_count: usize,
    /// Total elements
    pub total_count: usize,
}

/// Export a sparse (pruned) model with sparsity metadata sidecar
///
/// Produces:
/// - Weight file (SafeTensors format via bytemuck)
/// - `sparsity_metadata.json` with per-tensor sparsity stats
pub fn export_sparse_model(
    weights: &HashMap<String, Vec<f32>>,
    shapes: &HashMap<String, Vec<usize>>,
    metrics: &PruningMetrics,
    output_dir: impl AsRef<Path>,
    filename: &str,
) -> Result<SparseExportResult, std::io::Error> {
    let output_dir = output_dir.as_ref();
    std::fs::create_dir_all(output_dir)?;

    // Compute per-tensor sparsity
    let mut tensor_infos = Vec::new();
    let mut total_zeros = 0usize;
    let mut total_elements = 0usize;

    // Sort keys for deterministic output
    let mut names: Vec<&String> = weights.keys().collect();
    names.sort();

    for name in &names {
        let data = &weights[*name];
        let zero_count = data.iter().filter(|&&v| v == 0.0).count();
        let total = data.len();

        tensor_infos.push(TensorSparsityInfo {
            name: (*name).clone(),
            sparsity: if total > 0 { zero_count as f32 / total as f32 } else { 0.0 },
            zero_count,
            total_count: total,
        });

        total_zeros += zero_count;
        total_elements += total;
    }

    let global_sparsity =
        if total_elements > 0 { total_zeros as f32 / total_elements as f32 } else { 0.0 };

    // Build metadata
    let metadata = SparsityMetadata {
        version: "1.0".to_string(),
        global_sparsity,
        total_parameters: metrics.total_parameters,
        parameters_pruned: metrics.parameters_pruned,
        tensors: tensor_infos,
    };

    // Write weight file as SafeTensors
    let weights_path = output_dir.join(filename);
    {
        use safetensors::tensor::{Dtype, TensorView};

        let tensor_data: Vec<(String, Vec<u8>, Vec<usize>)> = names
            .iter()
            .map(|name| {
                let data = &weights[*name];
                let bytes: Vec<u8> = bytemuck::cast_slice(data).to_vec();
                let shape = shapes.get(*name).cloned().unwrap_or_else(|| vec![data.len()]);
                ((*name).clone(), bytes, shape)
            })
            .collect();

        let views: Vec<(&str, TensorView<'_>)> = tensor_data
            .iter()
            .map(|(name, bytes, shape)| {
                let view = TensorView::new(Dtype::F32, shape.clone(), bytes)
                    .expect("TensorView construction must not fail for valid F32 data");
                (name.as_str(), view)
            })
            .collect();

        let safetensor_bytes = safetensors::serialize(views, None)
            .map_err(|e| std::io::Error::other(e.to_string()))?;

        std::fs::write(&weights_path, safetensor_bytes)?;
    }

    // Write sparsity metadata
    let metadata_path = output_dir.join("sparsity_metadata.json");
    let metadata_json = serde_json::to_string_pretty(&metadata)
        .map_err(|e| std::io::Error::other(e.to_string()))?;
    std::fs::write(&metadata_path, metadata_json)?;

    Ok(SparseExportResult {
        weights_path,
        metadata_path,
        global_sparsity,
        num_tensors: names.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn make_test_data() -> (HashMap<String, Vec<f32>>, HashMap<String, Vec<usize>>) {
        let mut weights = HashMap::new();
        let mut shapes = HashMap::new();

        // 50% sparse tensor
        let data = vec![1.0, 0.0, 0.0, 2.0, 0.0, 3.0, 0.0, 0.0];
        weights.insert("layer.0.weight".to_string(), data);
        shapes.insert("layer.0.weight".to_string(), vec![2, 4]);

        // 0% sparse tensor
        weights.insert("layer.0.bias".to_string(), vec![0.1, 0.2]);
        shapes.insert("layer.0.bias".to_string(), vec![2]);

        (weights, shapes)
    }

    #[test]
    fn test_export_sparse_creates_files() {
        let (weights, shapes) = make_test_data();
        let metrics = PruningMetrics::new(0.5);
        let tmp = TempDir::new().expect("temp file creation should succeed");

        let result =
            export_sparse_model(&weights, &shapes, &metrics, tmp.path(), "sparse.safetensors")
                .expect("operation should succeed");

        assert!(result.weights_path.exists());
        assert!(result.metadata_path.exists());
        assert_eq!(result.num_tensors, 2);
    }

    #[test]
    fn test_export_sparse_metadata_content() {
        let (weights, shapes) = make_test_data();
        let mut metrics = PruningMetrics::new(0.5);
        metrics.update_sparsity(5, 10);
        let tmp = TempDir::new().expect("temp file creation should succeed");

        export_sparse_model(&weights, &shapes, &metrics, tmp.path(), "sparse.safetensors")
            .expect("parsing should succeed");

        let json = std::fs::read_to_string(tmp.path().join("sparsity_metadata.json"))
            .expect("file read should succeed");
        let meta: SparsityMetadata =
            serde_json::from_str(&json).expect("JSON deserialization should succeed");

        assert_eq!(meta.version, "1.0");
        assert_eq!(meta.total_parameters, 10);
        assert_eq!(meta.parameters_pruned, 5);
        assert_eq!(meta.tensors.len(), 2);
    }

    #[test]
    fn test_per_tensor_sparsity() {
        let (weights, shapes) = make_test_data();
        let metrics = PruningMetrics::new(0.5);
        let tmp = TempDir::new().expect("temp file creation should succeed");

        export_sparse_model(&weights, &shapes, &metrics, tmp.path(), "sparse.safetensors")
            .expect("parsing should succeed");

        let json = std::fs::read_to_string(tmp.path().join("sparsity_metadata.json"))
            .expect("file read should succeed");
        let meta: SparsityMetadata =
            serde_json::from_str(&json).expect("JSON deserialization should succeed");

        // layer.0.bias should have 0% sparsity
        let bias_info = meta
            .tensors
            .iter()
            .find(|t| t.name == "layer.0.bias")
            .expect("operation should succeed");
        assert_eq!(bias_info.sparsity, 0.0);
        assert_eq!(bias_info.zero_count, 0);

        // layer.0.weight should have 5/8 = 62.5% sparsity
        let weight_info = meta
            .tensors
            .iter()
            .find(|t| t.name == "layer.0.weight")
            .expect("operation should succeed");
        assert!(weight_info.sparsity > 0.5);
        assert_eq!(weight_info.zero_count, 5);
    }

    #[test]
    fn test_export_sparse_safetensors_valid() {
        let (weights, shapes) = make_test_data();
        let metrics = PruningMetrics::new(0.5);
        let tmp = TempDir::new().expect("temp file creation should succeed");

        let result =
            export_sparse_model(&weights, &shapes, &metrics, tmp.path(), "sparse.safetensors")
                .expect("operation should succeed");

        // Verify safetensors is valid
        let data = std::fs::read(&result.weights_path).expect("file read should succeed");
        let loaded = safetensors::SafeTensors::deserialize(&data).expect("load should succeed");
        assert_eq!(loaded.len(), 2);
    }

    #[test]
    fn test_export_empty_weights() {
        let weights = HashMap::new();
        let shapes = HashMap::new();
        let metrics = PruningMetrics::new(0.0);
        let tmp = TempDir::new().expect("temp file creation should succeed");

        let result =
            export_sparse_model(&weights, &shapes, &metrics, tmp.path(), "empty.safetensors")
                .expect("operation should succeed");

        assert_eq!(result.num_tensors, 0);
        assert_eq!(result.global_sparsity, 0.0);
    }
}
