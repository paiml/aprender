//! Column statistics for type inference

/// Statistics about a column used for type inference
#[derive(Debug, Clone, Default)]
pub struct ColumnStats {
    /// Column name
    pub name: String,
    /// Number of rows
    pub count: usize,
    /// Number of unique values
    pub unique_count: usize,
    /// Number of null/missing values
    pub null_count: usize,
    /// Whether all values are integers
    pub all_integers: bool,
    /// Whether all values are numeric
    pub all_numeric: bool,
    /// Minimum string length (if text)
    pub min_str_len: Option<usize>,
    /// Maximum string length (if text)
    pub max_str_len: Option<usize>,
    /// Average string length (if text)
    pub avg_str_len: Option<f32>,
    /// Whether values look like timestamps
    pub looks_like_datetime: bool,
    /// Whether values are arrays/lists
    pub is_array: bool,
    /// Array element count (if array)
    pub array_len: Option<usize>,
    /// Sample values for heuristic analysis
    pub sample_values: Vec<String>,
}

impl ColumnStats {
    /// Create stats for a column
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into(), ..Default::default() }
    }

    /// Cardinality ratio: unique_count / count
    pub fn cardinality_ratio(&self) -> f32 {
        if self.count == 0 {
            0.0
        } else {
            self.unique_count as f32 / self.count as f32
        }
    }

    /// Null ratio: null_count / count
    pub fn null_ratio(&self) -> f32 {
        if self.count == 0 {
            0.0
        } else {
            self.null_count as f32 / self.count as f32
        }
    }
}
