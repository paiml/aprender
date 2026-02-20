
fn tree_view_recursive(node: &TreeNode, prefix: &str, is_last: bool, result: &mut String) {
    // Branch character
    let branch = if is_last { "└── " } else { "├── " };

    // Node info
    let shape_str = node
        .shape
        .as_ref()
        .map_or(String::new(), |s| LayerInfo::format_shape(s));

    let dtype_str = node
        .dtype
        .as_ref()
        .map_or(String::new(), |d| format!(" <{d}>"));

    if node.shape.is_some() {
        result.push_str(&format!(
            "{prefix}{branch}{} {shape_str}{dtype_str}\n",
            node.name
        ));
    } else {
        result.push_str(&format!(
            "{prefix}{branch}{} [{}]\n",
            node.name, node.node_type
        ));
    }

    // Children
    let child_prefix = format!("{prefix}{}", if is_last { "    " } else { "│   " });
    for (i, child) in node.children.iter().enumerate() {
        let is_last_child = i == node.children.len() - 1;
        tree_view_recursive(child, &child_prefix, is_last_child, result);
    }
}

// ============================================================================
// Tensor Statistics
// ============================================================================

/// Statistics for a tensor
#[derive(Debug, Clone)]
pub struct TensorStatistics {
    /// Tensor name
    pub name: String,
    /// Shape
    pub shape: Vec<usize>,
    /// Data type
    pub dtype: String,
    /// Minimum value
    pub min: f64,
    /// Maximum value
    pub max: f64,
    /// Mean value
    pub mean: f64,
    /// Standard deviation
    pub std: f64,
    /// Count of NaN values
    pub nan_count: usize,
    /// Count of Inf values
    pub inf_count: usize,
    /// Count of zeros
    pub zero_count: usize,
}

impl TensorStatistics {
    /// Compute statistics from f32 tensor
    #[must_use]
    pub fn from_f32(name: impl Into<String>, shape: Vec<usize>, data: &[f32]) -> Self {
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        let mut sum = 0.0_f64;
        let mut nan_count = 0;
        let mut inf_count = 0;
        let mut zero_count = 0;

        for &val in data {
            if val.is_nan() {
                nan_count += 1;
            } else if val.is_infinite() {
                inf_count += 1;
            } else {
                if val < min {
                    min = val;
                }
                if val > max {
                    max = val;
                }
                if val == 0.0 {
                    zero_count += 1;
                }
                sum += f64::from(val);
            }
        }

        let valid_count = data.len() - nan_count - inf_count;
        let mean = if valid_count > 0 {
            sum / valid_count as f64
        } else {
            0.0
        };

        // Compute std deviation
        let mut var_sum = 0.0_f64;
        for &val in data {
            if !val.is_nan() && !val.is_infinite() {
                var_sum += (f64::from(val) - mean).powi(2);
            }
        }
        let std = if valid_count > 1 {
            (var_sum / (valid_count - 1) as f64).sqrt()
        } else {
            0.0
        };

        Self {
            name: name.into(),
            shape,
            dtype: "f32".to_string(),
            min: if min.is_infinite() {
                0.0
            } else {
                f64::from(min)
            },
            max: if max.is_infinite() {
                0.0
            } else {
                f64::from(max)
            },
            mean,
            std,
            nan_count,
            inf_count,
            zero_count,
        }
    }

    /// Format as single line summary
    #[must_use]
    pub fn summary(&self) -> String {
        let shape_str = LayerInfo::format_shape(&self.shape);
        format!(
            "{}: {} <{}> min={:.4e} max={:.4e} mean={:.4e} std={:.4e}",
            self.name, shape_str, self.dtype, self.min, self.max, self.mean, self.std
        )
    }

    /// Check for any anomalies (NaN, Inf, all zeros)
    #[must_use]
    pub fn has_anomalies(&self) -> bool {
        let total = self.shape.iter().product::<usize>();
        self.nan_count > 0 || self.inf_count > 0 || (total > 0 && self.zero_count == total)
    }
}

/// Generate statistics table for multiple tensors.
#[must_use]
pub fn statistics_table(stats: &[TensorStatistics]) -> String {
    let mut result = String::new();

    if stats.is_empty() {
        return "(no tensors)\n".to_string();
    }

    // Find max name length
    let max_name = stats.iter().map(|s| s.name.len()).max().unwrap_or(20);

    // Header
    result.push_str(&format!(
        "{:name_width$}  {:>15}  {:>12}  {:>12}  {:>12}  {:>12}  {:>5}\n",
        "Name",
        "Shape",
        "Min",
        "Max",
        "Mean",
        "Std",
        "Anom",
        name_width = max_name
    ));
    result.push_str(&format!("{:-<width$}\n", "", width = max_name + 75));

    // Rows
    for stat in stats {
        let shape_str = LayerInfo::format_shape(&stat.shape);
        let anomaly_str = if stat.has_anomalies() { "!" } else { "" };
        result.push_str(&format!(
            "{:name_width$}  {:>15}  {:>12.4e}  {:>12.4e}  {:>12.4e}  {:>12.4e}  {:>5}\n",
            stat.name,
            shape_str,
            stat.min,
            stat.max,
            stat.mean,
            stat.std,
            anomaly_str,
            name_width = max_name
        ));
    }

    result
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hex_dump_basic() {
        let data = [0x41, 0x50, 0x52, 0x31]; // "APR1"
        let dump = hex_dump(&data, &HexDumpConfig::default());
        assert!(dump.contains("41 50 52 31"));
        assert!(dump.contains("APR1"));
    }

    #[test]
    fn test_hex_dump_with_offset() {
        let data: Vec<u8> = (0..32).collect();
        let config = HexDumpConfig::default();
        let dump = hex_dump(&data, &config);
        assert!(dump.contains("00000000"));
        assert!(dump.contains("00000010"));
    }

    #[test]
    fn test_hex_dump_no_offset() {
        let data = [0x00, 0x01, 0x02];
        let mut config = HexDumpConfig::default();
        config.show_offset = false;
        let dump = hex_dump(&data, &config);
        assert!(!dump.contains("00000000"));
    }

    #[test]
    fn test_hex_dump_truncation() {
        let data: Vec<u8> = (0..100).collect();
        let mut config = HexDumpConfig::default();
        config.max_bytes = 32;
        let dump = hex_dump(&data, &config);
        assert!(dump.contains("truncated"));
        assert!(dump.contains("68 more bytes"));
    }

    #[test]
    fn test_hex_dump_non_printable() {
        let data = [0x00, 0x01, 0x7F, 0xFF];
        let dump = hex_dump(&data, &HexDumpConfig::default());
        // Non-printable characters are shown as dots
        // With 4 bytes + padding on a 16-byte line: "|....            |"
        assert!(dump.contains('|'));
        assert!(dump.contains("...."));
    }

    #[test]
    fn test_hex_dump_empty() {
        let data: [u8; 0] = [];
        let dump = hex_dump(&data, &HexDumpConfig::default());
        assert!(dump.is_empty());
    }

    #[test]
    fn test_tensor_hex_dump() {
        let tensor = [1.0_f32, 2.0, 3.0, 4.0];
        let config = HexDumpConfig::default();
        let dump = tensor_hex_dump(&tensor, &config);
        assert!(dump.contains("1.000000e0"));
        assert!(dump.contains("2.000000e0"));
    }

    #[test]
    fn test_data_flow_diagram() {
        let layers = vec![
            LayerInfo::new(
                "conv1",
                "Conv2d",
                vec![1, 3, 224, 224],
                vec![1, 64, 112, 112],
                9408,
            ),
            LayerInfo::new(
                "pool1",
                "MaxPool",
                vec![1, 64, 112, 112],
                vec![1, 64, 56, 56],
                0,
            ),
            LayerInfo::new(
                "fc",
                "Linear",
                vec![1, 64, 56, 56],
                vec![1, 1000],
                200704000,
            ),
        ];
        let diagram = data_flow_diagram(&layers);
        assert!(diagram.contains("conv1"));
        assert!(diagram.contains("Conv2d"));
        assert!(diagram.contains("Total parameters"));
    }

    #[test]
    fn test_data_flow_diagram_empty() {
        let layers: Vec<LayerInfo> = vec![];
        let diagram = data_flow_diagram(&layers);
        assert!(diagram.contains("no layers"));
    }

    #[test]
    fn test_format_params() {
        assert_eq!(format_params(500), "500");
        assert_eq!(format_params(1500), "1.5K");
        assert_eq!(format_params(1_500_000), "1.5M");
        assert_eq!(format_params(1_500_000_000), "1.5B");
    }

    #[test]
    fn test_tree_node() {
        let mut root = TreeNode::new("model", "Module");
        let mut encoder = TreeNode::new("encoder", "Block");
        encoder.add_child(TreeNode::tensor("weight", vec![512, 768], "f32"));
        encoder.add_child(TreeNode::tensor("bias", vec![512], "f32"));
        root.add_child(encoder);

        assert_eq!(root.count_nodes(), 4);
    }

    #[test]
    fn test_tree_view() {
        let mut root = TreeNode::new("model", "Module");
        root.add_child(TreeNode::tensor("weight", vec![10, 20], "f32"));
        root.add_child(TreeNode::tensor("bias", vec![10], "f32"));

        let view = tree_view(&root);
        assert!(view.contains("model"));
        assert!(view.contains("weight"));
        assert!(view.contains("bias"));
        assert!(view.contains("(10, 20)"));
    }

    #[test]
    fn test_tensor_statistics_basic() {
        let data = [1.0_f32, 2.0, 3.0, 4.0, 5.0];
        let stats = TensorStatistics::from_f32("test", vec![5], &data);

        assert_eq!(stats.name, "test");
        assert!((stats.min - 1.0).abs() < 1e-6);
        assert!((stats.max - 5.0).abs() < 1e-6);
        assert!((stats.mean - 3.0).abs() < 1e-6);
        assert!(!stats.has_anomalies());
    }

    #[test]
    fn test_tensor_statistics_with_nan() {
        let data = [1.0_f32, f32::NAN, 3.0];
        let stats = TensorStatistics::from_f32("test", vec![3], &data);

        assert_eq!(stats.nan_count, 1);
        assert!(stats.has_anomalies());
    }

    #[test]
    fn test_tensor_statistics_all_zeros() {
        let data = [0.0_f32; 10];
        let stats = TensorStatistics::from_f32("test", vec![10], &data);

        assert_eq!(stats.zero_count, 10);
        assert!(stats.has_anomalies());
    }

    #[test]
    fn test_statistics_table() {
        let stats = vec![
            TensorStatistics::from_f32("layer1.weight", vec![10, 20], &[1.0_f32; 200]),
            TensorStatistics::from_f32("layer1.bias", vec![10], &[0.5_f32; 10]),
        ];
        let table = statistics_table(&stats);
        assert!(table.contains("layer1.weight"));
        assert!(table.contains("layer1.bias"));
        assert!(table.contains("(10, 20)"));
    }

    #[test]
    fn test_hex_dump_config_compact() {
        let config = HexDumpConfig::compact();
        assert_eq!(config.bytes_per_line, 8);
        assert_eq!(config.max_bytes, 256);
    }

    #[test]
    fn test_hex_dump_config_wide() {
        let config = HexDumpConfig::wide();
        assert_eq!(config.bytes_per_line, 32);
        assert_eq!(config.group_size, 4);
    }

    #[test]
    fn test_layer_info_format_shape() {
        assert_eq!(LayerInfo::format_shape(&[]), "()");
        assert_eq!(LayerInfo::format_shape(&[10]), "(10)");
        assert_eq!(LayerInfo::format_shape(&[3, 224, 224]), "(3, 224, 224)");
    }

    #[test]
    fn test_tensor_statistics_summary() {
        let data = [1.0_f32, 2.0, 3.0];
        let stats = TensorStatistics::from_f32("test", vec![3], &data);
        let summary = stats.summary();
        assert!(summary.contains("test"));
        assert!(summary.contains("(3)"));
        assert!(summary.contains("f32"));
    }
}
