//! Conversion from `.apr`/`.ald` file formats to display widgets.
//!
//! Bridges `presentar-yaml` format loaders to `ModelCard` and `DataCard` widgets.
#![allow(clippy::doc_markdown)]
#![allow(clippy::uninlined_format_args)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::redundant_closure_for_method_calls)]

use crate::data_card::{DataCard, DataColumn};
use crate::model_card::{ModelCard, ModelMetric, ModelStatus};
use presentar_yaml::formats::{AldDataset, AprModel, DType};

/// Extension trait for converting `AprModel` to `ModelCard`.
pub trait AprModelExt {
    /// Convert to a display-ready `ModelCard` widget.
    fn to_model_card(&self) -> ModelCard;
}

impl AprModelExt for AprModel {
    fn to_model_card(&self) -> ModelCard {
        // Count total parameters
        let total_params: u64 = self
            .layers
            .iter()
            .flat_map(|l| l.parameters.iter())
            .map(|t| t.numel() as u64)
            .sum();

        // Build layer summary for description
        let layer_summary: String = self
            .layers
            .iter()
            .map(|l| format!("{}({})", l.layer_type, l.parameters.len()))
            .collect::<Vec<_>>()
            .join(" → ");

        let mut card = ModelCard::new(&self.model_type)
            .version(format!("v{}", self.version))
            .framework("Aprender")
            .parameters(total_params)
            .status(ModelStatus::Published);

        if !layer_summary.is_empty() {
            card = card.description(format!("Architecture: {}", layer_summary));
        }

        // Add metrics from metadata
        if let Some(acc) = self.metadata.get("accuracy") {
            if let Ok(v) = acc.parse::<f64>() {
                card = card.metric(ModelMetric::new("accuracy", v));
            }
        }
        if let Some(loss) = self.metadata.get("loss") {
            if let Ok(v) = loss.parse::<f64>() {
                card = card.metric(ModelMetric::new("loss", v).lower_is_better());
            }
        }
        if let Some(f1) = self.metadata.get("f1_score") {
            if let Ok(v) = f1.parse::<f64>() {
                card = card.metric(ModelMetric::new("F1", v));
            }
        }

        // Add task if in metadata
        if let Some(task) = self.metadata.get("task") {
            card = card.task(task);
        }

        // Add dataset if in metadata
        if let Some(dataset) = self.metadata.get("dataset") {
            card = card.dataset(dataset);
        }

        // Add author if in metadata
        if let Some(author) = self.metadata.get("author") {
            card = card.author(author);
        }

        // Add remaining metadata
        for (k, v) in &self.metadata {
            if !["accuracy", "loss", "f1_score", "task", "dataset", "author"].contains(&k.as_str())
            {
                card = card.metadata_entry(k, v);
            }
        }

        card.tag("apr").tag("sovereign-ai")
    }
}

/// Extension trait for converting `AldDataset` to `DataCard`.
pub trait AldDatasetExt {
    /// Convert to a display-ready `DataCard` widget.
    fn to_data_card(&self, name: &str) -> DataCard;
}

impl AldDatasetExt for AldDataset {
    fn to_data_card(&self, name: &str) -> DataCard {
        // Calculate total size
        let total_bytes: usize = self.tensors.iter().map(|t| t.data.len()).sum();

        // Count total elements
        let total_elements: usize = self.tensors.iter().map(|t| t.numel()).sum();

        // Build columns from tensors
        let columns: Vec<DataColumn> = self
            .tensors
            .iter()
            .map(|t| {
                let dtype_str = match t.dtype {
                    DType::F32 => "float32",
                    DType::F64 => "float64",
                    DType::I32 => "int32",
                    DType::I64 => "int64",
                    DType::U8 => "uint8",
                };
                let shape_str = t
                    .shape
                    .iter()
                    .map(|d| d.to_string())
                    .collect::<Vec<_>>()
                    .join("×");

                DataColumn::new(&t.name, dtype_str).description(format!("Shape: [{}]", shape_str))
            })
            .collect();

        let mut card = DataCard::new(name)
            .description(format!(
                "{} elements, {} tensors, {}",
                total_elements,
                self.tensors.len(),
                format_bytes(total_bytes)
            ))
            .source("Alimentar (.ald)")
            .tag("ald")
            .tag("sovereign-ai");

        for col in columns {
            card = card.column(col);
        }

        card
    }
}

/// Format bytes as human-readable string.
fn format_bytes(bytes: usize) -> String {
    const KB: usize = 1024;
    const MB: usize = KB * 1024;
    const GB: usize = MB * 1024;

    if bytes >= GB {
        format!("{:.1} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.1} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.1} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}

/// Load an .apr file and convert to ModelCard.
///
/// # Errors
///
/// Returns error if file cannot be loaded.
pub fn load_apr_as_card(data: &[u8]) -> Result<ModelCard, presentar_yaml::FormatError> {
    let model = AprModel::load(data)?;
    Ok(model.to_model_card())
}

/// Load an .ald file and convert to DataCard.
///
/// # Errors
///
/// Returns error if file cannot be loaded.
pub fn load_ald_as_card(data: &[u8], name: &str) -> Result<DataCard, presentar_yaml::FormatError> {
    let dataset = AldDataset::load(data)?;
    Ok(dataset.to_data_card(name))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::disallowed_methods)]
mod tests {
    use super::*;
    use presentar_yaml::formats::{DType, ModelLayer, Tensor};

    #[test]
    fn test_apr_to_model_card() {
        let mut model = AprModel::new("LinearRegression");
        model.layers.push(ModelLayer {
            layer_type: "dense".to_string(),
            parameters: vec![Tensor::from_f32("weights", vec![10, 5], &[0.0; 50])],
        });
        model
            .metadata
            .insert("accuracy".to_string(), "0.95".to_string());
        model
            .metadata
            .insert("task".to_string(), "classification".to_string());

        let card = model.to_model_card();

        assert_eq!(card.get_name(), "LinearRegression");
        assert_eq!(card.get_framework(), Some("Aprender"));
        assert_eq!(card.get_parameters(), Some(50));
        assert!(card.get_tags().contains(&"apr".to_string()));
    }

    #[test]
    fn test_ald_to_data_card() {
        let mut dataset = AldDataset::new();
        dataset.add_tensor(Tensor::from_f32("features", vec![100, 10], &[0.0; 1000]));
        dataset.add_tensor(Tensor::from_f32("labels", vec![100], &[0.0; 100]));

        let card = dataset.to_data_card("mnist_sample");

        assert_eq!(card.get_name(), "mnist_sample");
        assert_eq!(card.column_count(), 2);
        assert!(card.get_tags().contains(&"ald".to_string()));
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(500), "500 B");
        assert_eq!(format_bytes(1024), "1.0 KB");
        assert_eq!(format_bytes(1536), "1.5 KB");
        assert_eq!(format_bytes(1_048_576), "1.0 MB");
        assert_eq!(format_bytes(1_073_741_824), "1.0 GB");
    }

    #[test]
    fn test_load_apr_roundtrip() {
        let mut model = AprModel::new("MLP");
        model.layers.push(ModelLayer {
            layer_type: "dense".to_string(),
            parameters: vec![Tensor::from_f32("w", vec![4, 4], &[1.0; 16])],
        });

        let bytes = model.save();
        let card = load_apr_as_card(&bytes).unwrap();

        assert_eq!(card.get_name(), "MLP");
    }

    #[test]
    fn test_load_ald_roundtrip() {
        let mut dataset = AldDataset::new();
        dataset.add_tensor(Tensor::from_f32("data", vec![10], &[1.0; 10]));

        let bytes = dataset.save();
        let card = load_ald_as_card(&bytes, "test_data").unwrap();

        assert_eq!(card.get_name(), "test_data");
    }

    // =========================================================================
    // ModelCard Metadata Tests
    // =========================================================================

    #[test]
    fn test_apr_model_with_all_metrics() {
        let mut model = AprModel::new("Classifier");
        let kernel_data = vec![0.0_f32; 73_728];
        model.layers.push(ModelLayer {
            layer_type: "conv2d".to_string(),
            parameters: vec![Tensor::from_f32(
                "kernel",
                vec![3, 3, 64, 128],
                &kernel_data,
            )],
        });
        model
            .metadata
            .insert("accuracy".to_string(), "0.972".to_string());
        model
            .metadata
            .insert("loss".to_string(), "0.083".to_string());
        model
            .metadata
            .insert("f1_score".to_string(), "0.968".to_string());

        let card = model.to_model_card();

        assert_eq!(card.get_name(), "Classifier");
        assert_eq!(card.get_parameters(), Some(73728));
    }

    #[test]
    fn test_apr_model_with_task_and_dataset() {
        let mut model = AprModel::new("BERT");
        model
            .metadata
            .insert("task".to_string(), "text-classification".to_string());
        model
            .metadata
            .insert("dataset".to_string(), "IMDB".to_string());
        model
            .metadata
            .insert("author".to_string(), "research-team".to_string());

        let card = model.to_model_card();

        assert_eq!(card.get_task(), Some("text-classification"));
        assert_eq!(card.get_dataset(), Some("IMDB"));
        assert_eq!(card.get_author(), Some("research-team"));
    }

    #[test]
    fn test_apr_model_with_custom_metadata() {
        let mut model = AprModel::new("CustomModel");
        model
            .metadata
            .insert("training_time".to_string(), "2h30m".to_string());
        model
            .metadata
            .insert("epochs".to_string(), "50".to_string());
        model
            .metadata
            .insert("learning_rate".to_string(), "0.001".to_string());

        let card = model.to_model_card();

        // Custom metadata is added as metadata entries
        assert_eq!(card.get_name(), "CustomModel");
    }

    #[test]
    fn test_apr_model_multiple_layers() {
        let mut model = AprModel::new("DeepNet");
        let w1_data = vec![0.0_f32; 200_704];
        model.layers.push(ModelLayer {
            layer_type: "dense".to_string(),
            parameters: vec![
                Tensor::from_f32("w1", vec![784, 256], &w1_data),
                Tensor::from_f32("b1", vec![256], &[0.0; 256]),
            ],
        });
        model.layers.push(ModelLayer {
            layer_type: "relu".to_string(),
            parameters: vec![],
        });
        model.layers.push(ModelLayer {
            layer_type: "dense".to_string(),
            parameters: vec![
                Tensor::from_f32("w2", vec![256, 10], &[0.0; 2560]),
                Tensor::from_f32("b2", vec![10], &[0.0; 10]),
            ],
        });

        let card = model.to_model_card();

        // 200704 + 256 + 2560 + 10 = 203530
        assert_eq!(card.get_parameters(), Some(203_530));
    }

    #[test]
    fn test_apr_model_empty_layers() {
        let model = AprModel::new("EmptyModel");

        let card = model.to_model_card();

        assert_eq!(card.get_name(), "EmptyModel");
        assert_eq!(card.get_parameters(), Some(0));
    }

    #[test]
    fn test_apr_model_invalid_metric_values() {
        let mut model = AprModel::new("BadMetrics");
        model
            .metadata
            .insert("accuracy".to_string(), "not-a-number".to_string());
        model
            .metadata
            .insert("loss".to_string(), "invalid".to_string());

        // Should not panic, just skip invalid metrics
        let card = model.to_model_card();
        assert_eq!(card.get_name(), "BadMetrics");
    }

    // =========================================================================
    // DataCard Tests
    // =========================================================================

    #[test]
    fn test_ald_empty_dataset() {
        let dataset = AldDataset::new();
        let card = dataset.to_data_card("empty");

        assert_eq!(card.get_name(), "empty");
        assert_eq!(card.column_count(), 0);
    }

    #[test]
    fn test_ald_multiple_tensors() {
        let mut dataset = AldDataset::new();
        let train_x = vec![0.0_f32; 784_000];
        let test_x = vec![0.0_f32; 78_400];
        dataset.add_tensor(Tensor::from_f32("train_x", vec![1000, 784], &train_x));
        dataset.add_tensor(Tensor::from_f32("train_y", vec![1000], &[0.0; 1000]));
        dataset.add_tensor(Tensor::from_f32("test_x", vec![100, 784], &test_x));
        dataset.add_tensor(Tensor::from_f32("test_y", vec![100], &[0.0; 100]));

        let card = dataset.to_data_card("mnist");

        assert_eq!(card.get_name(), "mnist");
        assert_eq!(card.column_count(), 4);
    }

    #[test]
    fn test_ald_different_dtypes() {
        let mut dataset = AldDataset::new();

        // Test float32 dtype (only one available via from_f32)
        dataset.add_tensor(Tensor::from_f32("float32_tensor", vec![10], &[0.0; 10]));
        dataset.add_tensor(Tensor::from_f32("float32_tensor2", vec![5, 2], &[0.0; 10]));

        // Use Tensor::new for other dtypes
        dataset.add_tensor(Tensor::new(
            "float64_tensor",
            DType::F64,
            vec![10],
            vec![0u8; 80],
        ));
        dataset.add_tensor(Tensor::new(
            "int32_tensor",
            DType::I32,
            vec![10],
            vec![0u8; 40],
        ));
        dataset.add_tensor(Tensor::new(
            "uint8_tensor",
            DType::U8,
            vec![10],
            vec![0u8; 10],
        ));

        let card = dataset.to_data_card("multi_dtype");

        assert_eq!(card.column_count(), 5);
    }

    // =========================================================================
    // format_bytes Edge Cases
    // =========================================================================

    #[test]
    fn test_format_bytes_boundaries() {
        // Exact boundaries
        assert_eq!(format_bytes(0), "0 B");
        assert_eq!(format_bytes(1023), "1023 B");
        assert_eq!(format_bytes(1024), "1.0 KB");
        assert_eq!(format_bytes(1024 * 1024 - 1), "1024.0 KB");
        assert_eq!(format_bytes(1024 * 1024), "1.0 MB");
        assert_eq!(format_bytes(1024 * 1024 * 1024 - 1), "1024.0 MB");
        assert_eq!(format_bytes(1024 * 1024 * 1024), "1.0 GB");
    }

    #[test]
    fn test_format_bytes_large_values() {
        // 10 GB
        assert_eq!(format_bytes(10 * 1024 * 1024 * 1024), "10.0 GB");
        // 100 GB
        assert_eq!(format_bytes(100 * 1024 * 1024 * 1024), "100.0 GB");
    }

    #[test]
    fn test_format_bytes_fractional() {
        // 1.5 KB = 1536 bytes
        assert_eq!(format_bytes(1536), "1.5 KB");
        // 2.5 MB
        assert_eq!(format_bytes(2_621_440), "2.5 MB");
        // 3.25 GB
        assert_eq!(format_bytes(3_489_660_928), "3.2 GB");
    }

    // =========================================================================
    // Error Handling Tests
    // =========================================================================

    #[test]
    fn test_load_apr_invalid_data() {
        let invalid_data = b"not a valid apr file";
        let result = load_apr_as_card(invalid_data);
        assert!(result.is_err());
    }

    #[test]
    fn test_load_ald_invalid_data() {
        let invalid_data = b"not a valid ald file";
        let result = load_ald_as_card(invalid_data, "test");
        assert!(result.is_err());
    }

    #[test]
    fn test_load_apr_empty_data() {
        let empty_data = b"";
        let result = load_apr_as_card(empty_data);
        assert!(result.is_err());
    }

    #[test]
    fn test_load_ald_empty_data() {
        let empty_data = b"";
        let result = load_ald_as_card(empty_data, "test");
        assert!(result.is_err());
    }

    // =========================================================================
    // Tags Test
    // =========================================================================

    #[test]
    fn test_model_card_has_sovereign_ai_tag() {
        let model = AprModel::new("SovereignModel");
        let card = model.to_model_card();

        assert!(card.get_tags().contains(&"sovereign-ai".to_string()));
        assert!(card.get_tags().contains(&"apr".to_string()));
    }

    #[test]
    fn test_data_card_has_sovereign_ai_tag() {
        let dataset = AldDataset::new();
        let card = dataset.to_data_card("SovereignData");

        assert!(card.get_tags().contains(&"sovereign-ai".to_string()));
        assert!(card.get_tags().contains(&"ald".to_string()));
    }
}
