//! Model inspection utilities.

use crate::architecture::{ArchitectureDetector, ArchitectureInfo};
use entrenar_common::{EntrenarError, Result};
use std::collections::HashMap;
use std::path::Path;

/// Information about a model file.
#[derive(Debug, Clone)]
pub struct ModelInfo {
    /// File path
    pub path: std::path::PathBuf,
    /// File size in bytes
    pub size_bytes: u64,
    /// Detected format
    pub format: ModelFormat,
    /// Architecture information
    pub architecture: ArchitectureInfo,
    /// Total parameters
    pub total_params: u64,
    /// List of tensors
    pub tensors: Vec<TensorInfo>,
}

impl ModelInfo {
    /// Format file size as human-readable string.
    pub fn size_human(&self) -> String {
        entrenar_common::output::format_bytes(self.size_bytes)
    }

    /// Get parameters in billions.
    pub fn params_b(&self) -> f64 {
        self.total_params as f64 / 1e9
    }
}

/// Information about a single tensor.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    /// Tensor name
    pub name: String,
    /// Tensor shape
    pub shape: Vec<usize>,
    /// Data type
    pub dtype: DataType,
    /// Number of elements
    pub num_elements: u64,
    /// Size in bytes
    pub size_bytes: u64,
}

impl TensorInfo {
    /// Get parameter count.
    pub fn params(&self) -> u64 {
        self.num_elements
    }
}

/// Tensor data type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataType {
    F32,
    F16,
    BF16,
    I32,
    I8,
    U8,
    Unknown,
}

impl DataType {
    /// Bytes per element.
    pub fn size(&self) -> usize {
        match self {
            Self::F32 | Self::I32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::I8 | Self::U8 => 1,
            Self::Unknown => 0,
        }
    }
}

/// Model file format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFormat {
    /// SafeTensors format
    SafeTensors,
    /// GGUF format
    Gguf,
    /// APR format
    Apr,
    /// PyTorch pickle (unsafe)
    PyTorch,
    /// Unknown format
    Unknown,
}

/// Inspect a model file.
pub fn inspect_model(path: impl AsRef<Path>) -> Result<ModelInfo> {
    let path = path.as_ref();

    if !path.exists() {
        return Err(EntrenarError::ModelNotFound { path: path.to_path_buf() });
    }

    let metadata = std::fs::metadata(path).map_err(|e| EntrenarError::Io {
        context: format!("reading model metadata: {}", path.display()),
        source: e,
    })?;

    let format = detect_format(path);

    // For real implementation, would parse the actual file
    // Here we return simulated data based on file size
    let estimated_params = estimate_params_from_size(metadata.len(), &format);

    let tensors = generate_mock_tensors(estimated_params);
    let tensor_names: Vec<String> = tensors.iter().map(|t| t.name.clone()).collect();

    let shapes: HashMap<String, Vec<usize>> =
        tensors.iter().map(|t| (t.name.clone(), t.shape.clone())).collect();

    let detector = ArchitectureDetector::new().with_tensors(tensor_names);
    let architecture = detector.detect_from_shapes(&shapes);

    Ok(ModelInfo {
        path: path.to_path_buf(),
        size_bytes: metadata.len(),
        format,
        architecture,
        total_params: estimated_params,
        tensors,
    })
}

fn detect_format(path: &Path) -> ModelFormat {
    let extension = path.extension().and_then(|e| e.to_str()).unwrap_or("");

    match extension.to_lowercase().as_str() {
        "safetensors" => ModelFormat::SafeTensors,
        "gguf" => ModelFormat::Gguf,
        "apr" => ModelFormat::Apr,
        "pt" | "pth" | "bin" => ModelFormat::PyTorch,
        _ => ModelFormat::Unknown,
    }
}

fn estimate_params_from_size(size_bytes: u64, format: &ModelFormat) -> u64 {
    let bytes_per_param = match format {
        ModelFormat::SafeTensors | ModelFormat::PyTorch => 2, // Assume FP16
        ModelFormat::Gguf => 1,                               // Assume 8-bit average
        ModelFormat::Apr => 2,
        ModelFormat::Unknown => 2,
    };

    size_bytes / bytes_per_param as u64
}

fn generate_mock_tensors(total_params: u64) -> Vec<TensorInfo> {
    // Generate representative tensor structure
    let hidden_dim = if total_params > 10_000_000_000 {
        4096
    } else if total_params > 1_000_000_000 {
        2048
    } else {
        768
    };

    let num_layers =
        (total_params / (hidden_dim as u64 * hidden_dim as u64 * 12)).clamp(1, 80) as usize;
    // N-05 (Meyer DbC): derive vocab_size from embedding params and hidden_dim.
    // vocab_size ≈ embed_params / hidden_dim. Clamp to plausible range.
    let embed_params = total_params / 10; // ~10% of params in embeddings
    let vocab_size = (embed_params as usize / hidden_dim).clamp(256, 200_000);

    let mut tensors = Vec::new();

    // Embedding
    tensors.push(TensorInfo {
        name: "model.embed_tokens.weight".to_string(),
        shape: vec![vocab_size, hidden_dim],
        dtype: DataType::F16,
        num_elements: (vocab_size * hidden_dim) as u64,
        size_bytes: (vocab_size * hidden_dim * 2) as u64,
    });

    // Layers
    for i in 0..num_layers {
        // Q, K, V, O projections
        for proj in &["q_proj", "k_proj", "v_proj", "o_proj"] {
            tensors.push(TensorInfo {
                name: format!("model.layers.{i}.self_attn.{proj}.weight"),
                shape: vec![hidden_dim, hidden_dim],
                dtype: DataType::F16,
                num_elements: (hidden_dim * hidden_dim) as u64,
                size_bytes: (hidden_dim * hidden_dim * 2) as u64,
            });
        }

        // MLP
        for proj in &["gate_proj", "up_proj", "down_proj"] {
            let intermediate = hidden_dim * 4;
            let shape = if proj == &"down_proj" {
                vec![hidden_dim, intermediate]
            } else {
                vec![intermediate, hidden_dim]
            };
            tensors.push(TensorInfo {
                name: format!("model.layers.{i}.mlp.{proj}.weight"),
                shape: shape.clone(),
                dtype: DataType::F16,
                num_elements: (shape[0] * shape[1]) as u64,
                size_bytes: (shape[0] * shape[1] * 2) as u64,
            });
        }
    }

    // LM head
    tensors.push(TensorInfo {
        name: "lm_head.weight".to_string(),
        shape: vec![vocab_size, hidden_dim],
        dtype: DataType::F16,
        num_elements: (vocab_size * hidden_dim) as u64,
        size_bytes: (vocab_size * hidden_dim * 2) as u64,
    });

    tensors
}

/// Get layer-by-layer breakdown.
pub fn layer_breakdown(info: &ModelInfo) -> Vec<LayerSummary> {
    let mut layers: HashMap<usize, LayerSummary> = HashMap::new();

    for tensor in &info.tensors {
        // Extract layer number from tensor name
        if let Some(layer_num) = extract_layer_number(&tensor.name) {
            let entry = layers.entry(layer_num).or_insert(LayerSummary {
                layer_num,
                tensor_count: 0,
                param_count: 0,
                size_bytes: 0,
            });

            entry.tensor_count += 1;
            entry.param_count += tensor.num_elements;
            entry.size_bytes += tensor.size_bytes;
        }
    }

    let mut result: Vec<_> = layers.into_values().collect();
    result.sort_by_key(|l| l.layer_num);
    result
}

fn extract_layer_number(name: &str) -> Option<usize> {
    name.split('.').find_map(|part| part.parse::<usize>().ok())
}

/// Summary of a single layer.
#[derive(Debug, Clone)]
pub struct LayerSummary {
    /// Layer number
    pub layer_num: usize,
    /// Number of tensors in layer
    pub tensor_count: usize,
    /// Total parameters in layer
    pub param_count: u64,
    /// Total size in bytes
    pub size_bytes: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_detect_format() {
        assert_eq!(detect_format(Path::new("model.safetensors")), ModelFormat::SafeTensors);
        assert_eq!(detect_format(Path::new("model.gguf")), ModelFormat::Gguf);
        assert_eq!(detect_format(Path::new("model.pt")), ModelFormat::PyTorch);
        assert_eq!(detect_format(Path::new("model.unknown")), ModelFormat::Unknown);
    }

    #[test]
    fn test_data_type_size() {
        assert_eq!(DataType::F32.size(), 4);
        assert_eq!(DataType::F16.size(), 2);
        assert_eq!(DataType::I8.size(), 1);
    }

    #[test]
    fn test_estimate_params() {
        let size = 14_000_000_000u64; // ~14GB
        let params = estimate_params_from_size(size, &ModelFormat::SafeTensors);
        assert_eq!(params, 7_000_000_000); // 7B params at FP16
    }

    #[test]
    fn test_generate_mock_tensors() {
        let tensors = generate_mock_tensors(7_000_000_000);
        assert!(!tensors.is_empty());
        assert!(tensors.iter().any(|t| t.name.contains("embed")));
        assert!(tensors.iter().any(|t| t.name.contains("layers")));
    }

    #[test]
    fn test_extract_layer_number() {
        assert_eq!(extract_layer_number("model.layers.5.self_attn.q_proj.weight"), Some(5));
        assert_eq!(extract_layer_number("model.embed_tokens.weight"), None);
    }

    #[test]
    fn test_layer_breakdown() {
        let info = ModelInfo {
            path: PathBuf::from("test.safetensors"),
            size_bytes: 100,
            format: ModelFormat::SafeTensors,
            architecture: ArchitectureInfo {
                architecture: crate::architecture::Architecture::Llama,
                hidden_dim: 4096,
                num_layers: 32,
                vocab_size: 32000,
                num_heads: 32,
            },
            total_params: 7_000_000_000,
            tensors: generate_mock_tensors(7_000_000_000),
        };

        let breakdown = layer_breakdown(&info);
        assert!(!breakdown.is_empty());
    }

    #[test]
    fn test_model_info_size_human() {
        let info = ModelInfo {
            path: PathBuf::from("test.safetensors"),
            size_bytes: 14_000_000_000, // ~14GB
            format: ModelFormat::SafeTensors,
            architecture: ArchitectureInfo {
                architecture: crate::architecture::Architecture::Llama,
                hidden_dim: 4096,
                num_layers: 32,
                vocab_size: 32000,
                num_heads: 32,
            },
            total_params: 7_000_000_000,
            tensors: vec![],
        };

        let size = info.size_human();
        assert!(size.contains("GB"));
    }

    #[test]
    fn test_model_info_params_b() {
        let info = ModelInfo {
            path: PathBuf::from("test.safetensors"),
            size_bytes: 14_000_000_000,
            format: ModelFormat::SafeTensors,
            architecture: ArchitectureInfo {
                architecture: crate::architecture::Architecture::Llama,
                hidden_dim: 4096,
                num_layers: 32,
                vocab_size: 32000,
                num_heads: 32,
            },
            total_params: 7_000_000_000,
            tensors: vec![],
        };

        assert!((info.params_b() - 7.0).abs() < 0.01);
    }

    #[test]
    fn test_tensor_info_params() {
        let tensor = TensorInfo {
            name: "test".to_string(),
            shape: vec![4096, 4096],
            dtype: DataType::F16,
            num_elements: 4096 * 4096,
            size_bytes: 4096 * 4096 * 2,
        };
        assert_eq!(tensor.params(), 4096 * 4096);
    }

    #[test]
    fn test_data_type_all_sizes() {
        assert_eq!(DataType::F32.size(), 4);
        assert_eq!(DataType::F16.size(), 2);
        assert_eq!(DataType::BF16.size(), 2);
        assert_eq!(DataType::I32.size(), 4);
        assert_eq!(DataType::I8.size(), 1);
        assert_eq!(DataType::U8.size(), 1);
        assert_eq!(DataType::Unknown.size(), 0);
    }

    #[test]
    fn test_detect_format_pth() {
        assert_eq!(detect_format(Path::new("model.pth")), ModelFormat::PyTorch);
    }

    #[test]
    fn test_detect_format_bin() {
        assert_eq!(detect_format(Path::new("model.bin")), ModelFormat::PyTorch);
    }

    #[test]
    fn test_detect_format_apr() {
        assert_eq!(detect_format(Path::new("model.apr")), ModelFormat::Apr);
    }

    #[test]
    fn test_estimate_params_gguf() {
        let size = 7_000_000_000u64; // ~7GB
        let params = estimate_params_from_size(size, &ModelFormat::Gguf);
        assert_eq!(params, 7_000_000_000); // 1:1 at 8-bit
    }

    #[test]
    fn test_generate_mock_tensors_small_model() {
        let tensors = generate_mock_tensors(100_000_000); // 100M params
        assert!(!tensors.is_empty());
        // Smaller model should have smaller hidden dim
        let embed =
            tensors.iter().find(|t| t.name.contains("embed")).expect("operation should succeed");
        assert!(embed.shape[1] < 4096);
    }

    #[test]
    fn test_layer_breakdown_sorted() {
        let info = ModelInfo {
            path: PathBuf::from("test.safetensors"),
            size_bytes: 100,
            format: ModelFormat::SafeTensors,
            architecture: ArchitectureInfo {
                architecture: crate::architecture::Architecture::Llama,
                hidden_dim: 4096,
                num_layers: 32,
                vocab_size: 32000,
                num_heads: 32,
            },
            total_params: 7_000_000_000,
            tensors: generate_mock_tensors(7_000_000_000),
        };

        let breakdown = layer_breakdown(&info);
        // Verify layers are sorted
        for i in 1..breakdown.len() {
            assert!(breakdown[i].layer_num >= breakdown[i - 1].layer_num);
        }
    }
}
