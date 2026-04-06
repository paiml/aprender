//! Feature importance bar chart for terminal display.

/// Feature importance bar chart for terminal display.
#[derive(Debug, Clone)]
pub struct FeatureImportanceChart {
    /// Feature names
    pub(crate) names: Vec<String>,
    /// Importance scores
    pub(crate) scores: Vec<f32>,
    /// Bar width
    pub(crate) bar_width: usize,
    /// Number of features to show
    pub(crate) top_k: usize,
}

impl FeatureImportanceChart {
    /// Create a new feature importance chart.
    pub fn new(top_k: usize, bar_width: usize) -> Self {
        Self { names: Vec::new(), scores: Vec::new(), bar_width, top_k }
    }

    /// Update with new importance scores.
    pub fn update(&mut self, importances: &[(usize, f32)], feature_names: Option<&[String]>) {
        let mut sorted: Vec<_> = importances.to_vec();
        sorted
            .sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap_or(std::cmp::Ordering::Equal));
        sorted.truncate(self.top_k);

        self.names.clear();
        self.scores.clear();

        for (idx, score) in sorted {
            let name = feature_names
                .and_then(|n| n.get(idx))
                .cloned()
                .unwrap_or_else(|| format!("feature_{idx}"));
            self.names.push(name);
            self.scores.push(score);
        }
    }

    /// Render to string.
    pub fn render(&self) -> String {
        if self.names.is_empty() {
            return String::from("No feature importance data");
        }

        let max_name_len = self.names.iter().map(String::len).max().unwrap_or(10);
        let max_score = self.scores.iter().copied().fold(0.0f32, f32::max);

        let mut output = String::new();
        output.push_str("┌─ Feature Importance ─────────────────────────────┐\n");

        for (name, score) in self.names.iter().zip(self.scores.iter()) {
            let bar_len = if max_score > 0.0 {
                ((score / max_score) * self.bar_width as f32).round() as usize
            } else {
                0
            };
            let bar: String = "█".repeat(bar_len);
            output.push_str(&format!(
                "│  {:width$}  {:bar_width$}  {:.3}  │\n",
                name,
                bar,
                score,
                width = max_name_len,
                bar_width = self.bar_width
            ));
        }

        output.push_str("└──────────────────────────────────────────────────┘\n");
        output
    }
}
