//! Gradient flow heatmap for visualizing per-layer gradients.

/// Gradient flow heatmap for visualizing per-layer gradients.
#[derive(Debug, Clone)]
pub struct GradientFlowHeatmap {
    /// Layer names
    pub(crate) layer_names: Vec<String>,
    /// Gradient magnitudes per layer (log scale)
    pub(crate) gradients: Vec<Vec<f32>>,
    /// Column labels (Q, K, V, O, FFN, etc.)
    pub(crate) column_labels: Vec<String>,
}

impl GradientFlowHeatmap {
    /// Create a new gradient flow heatmap.
    pub fn new(layer_names: Vec<String>, column_labels: Vec<String>) -> Self {
        let num_layers = layer_names.len();
        Self {
            layer_names,
            gradients: vec![vec![0.0; column_labels.len()]; num_layers],
            column_labels,
        }
    }

    /// Update gradient for a specific layer and column.
    pub fn update(&mut self, layer: usize, col: usize, grad_norm: f32) {
        if layer < self.gradients.len() && col < self.column_labels.len() {
            // Store log scale for visualization
            self.gradients[layer][col] = (grad_norm + 1e-8).max(f32::MIN_POSITIVE).ln();
        }
    }

    /// Render to string.
    pub fn render(&self) -> String {
        let heatmap_chars = ['░', '▒', '▓', '█'];

        // Find min/max for normalization
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        for row in &self.gradients {
            for &v in row {
                min = min.min(v);
                max = max.max(v);
            }
        }
        let range = max - min;

        let mut output = String::new();
        output.push_str("Gradient Flow (log scale):\n");

        // Header
        output.push_str("         ");
        for label in &self.column_labels {
            output.push_str(&format!("{label:^5}"));
        }
        output.push('\n');

        // Rows
        for (i, row) in self.gradients.iter().enumerate() {
            let name = self.layer_names.get(i).map_or("?", String::as_str);
            output.push_str(&format!("{name:>8} "));

            for &v in row {
                let normalized =
                    if range > f32::EPSILON { ((v - min) / range).clamp(0.0, 1.0) } else { 0.5 };
                let idx = (normalized * 3.0).round() as usize;
                let c = heatmap_chars[idx.min(3)];
                output.push_str(&format!("{c}{c}{c}{c} "));
            }
            output.push('\n');
        }

        output
    }
}
