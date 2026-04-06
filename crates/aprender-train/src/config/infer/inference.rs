//! Type inference functions

use std::collections::HashSet;
use std::path::Path;

use super::config::InferenceConfig;
use super::schema::InferredSchema;
use super::stats::ColumnStats;
use super::types::FeatureType;

/// Check if a column name matches any target column pattern
fn is_target_column(name_lower: &str, target_columns: &[String]) -> bool {
    target_columns.iter().any(|t| {
        let t_lower = t.to_lowercase();
        name_lower == t_lower
            || name_lower.ends_with(&format!("_{t_lower}"))
            || name_lower.starts_with(&format!("{t_lower}_"))
    })
}

/// Infer type for a numeric target column
fn infer_target_type(stats: &ColumnStats) -> FeatureType {
    if stats.all_integers && stats.unique_count == 2 {
        FeatureType::BinaryTarget
    } else if stats.all_integers && stats.unique_count <= 100 {
        FeatureType::MultiClassTarget
    } else {
        FeatureType::RegressionTarget
    }
}

/// Infer type for a numeric non-target column
fn infer_numeric_type(stats: &ColumnStats, config: &InferenceConfig) -> FeatureType {
    if stats.all_integers && stats.cardinality_ratio() < config.categorical_threshold {
        FeatureType::Categorical
    } else {
        FeatureType::Numeric
    }
}

/// Check if sample values contain token sequences
fn has_token_sequences(sample_values: &[String]) -> bool {
    sample_values.iter().any(|s| s.split_whitespace().count() > 5)
}

/// Infer type for a string column
fn infer_string_type(stats: &ColumnStats, config: &InferenceConfig, avg_len: f32) -> FeatureType {
    if avg_len >= config.text_min_avg_len {
        return FeatureType::Text;
    }
    if stats.cardinality_ratio() < config.categorical_threshold {
        return FeatureType::Categorical;
    }
    if has_token_sequences(&stats.sample_values) {
        return FeatureType::TokenSequence;
    }
    FeatureType::Text
}

/// Infer feature type from column statistics
pub fn infer_type(stats: &ColumnStats, config: &InferenceConfig) -> FeatureType {
    let name_lower = stats.name.to_lowercase();
    let is_target = is_target_column(&name_lower, &config.target_columns);

    if config.exclude_columns.contains(&stats.name) {
        return FeatureType::Unknown;
    }

    if stats.is_array && stats.array_len.is_some() {
        return FeatureType::Embedding;
    }

    if stats.looks_like_datetime {
        return FeatureType::DateTime;
    }

    if stats.all_numeric {
        return if is_target {
            infer_target_type(stats)
        } else {
            infer_numeric_type(stats, config)
        };
    }

    if let Some(avg_len) = stats.avg_str_len {
        return infer_string_type(stats, config, avg_len);
    }

    FeatureType::Unknown
}

/// Infer schema from column statistics
pub fn infer_schema(stats: Vec<ColumnStats>, config: &InferenceConfig) -> InferredSchema {
    let mut schema = InferredSchema::default();

    for col_stats in stats {
        let feature_type = infer_type(&col_stats, config);
        schema.features.insert(col_stats.name.clone(), feature_type);
        schema.stats.insert(col_stats.name.clone(), col_stats);
    }

    schema
}

/// Check if a string looks like a numeric value
fn is_numeric_string(s: &str) -> (bool, bool) {
    let is_float = s.parse::<f64>().is_ok();
    let is_int = s.parse::<i64>().is_ok();
    (is_float, is_int)
}

/// Check if a string looks like a datetime
fn looks_like_datetime(s: &str) -> bool {
    s.contains('-')
        && s.len() >= 10
        && s.len() <= 30
        && s.chars().filter(char::is_ascii_digit).count() >= 8
}

/// Accumulator for collecting string statistics
struct StatsAccumulator<'a> {
    unique: HashSet<&'a str>,
    total_len: usize,
    min_len: usize,
    max_len: usize,
    all_numeric: bool,
    all_integers: bool,
    datetime_count: usize,
}

impl<'a> StatsAccumulator<'a> {
    fn new() -> Self {
        Self {
            unique: HashSet::new(),
            total_len: 0,
            min_len: usize::MAX,
            max_len: 0,
            all_numeric: true,
            all_integers: true,
            datetime_count: 0,
        }
    }

    fn process(&mut self, s: &'a str) {
        self.unique.insert(s);
        let len = s.len();
        self.total_len += len;
        self.min_len = self.min_len.min(len);
        self.max_len = self.max_len.max(len);

        let (is_float, is_int) = is_numeric_string(s);
        if !is_float {
            self.all_numeric = false;
            self.all_integers = false;
        } else if !is_int {
            self.all_integers = false;
        }

        if looks_like_datetime(s) {
            self.datetime_count += 1;
        }
    }
}

/// Finalize stats from accumulator
fn finalize_stats(stats: &mut ColumnStats, acc: &StatsAccumulator<'_>) {
    stats.unique_count = acc.unique.len();
    stats.all_numeric = acc.all_numeric && stats.null_count < stats.count;
    stats.all_integers = acc.all_integers && stats.null_count < stats.count;

    let non_null = stats.count - stats.null_count;
    if non_null > 0 {
        stats.min_str_len = Some(acc.min_len);
        stats.max_str_len = Some(acc.max_len);
        stats.avg_str_len = Some(acc.total_len as f32 / non_null as f32);
        stats.looks_like_datetime = acc.datetime_count as f32 / non_null as f32 > 0.5;
    }
}

/// Collect statistics from sample values (simplified in-memory analysis)
pub fn collect_stats_from_samples(name: &str, values: &[Option<&str>]) -> ColumnStats {
    let mut stats = ColumnStats::new(name);
    stats.count = values.len();

    let mut acc = StatsAccumulator::new();

    for val in values {
        match val {
            Some(s) => {
                acc.process(s);
                if stats.sample_values.len() < 10 {
                    stats.sample_values.push((*s).to_string());
                }
            }
            None => {
                stats.null_count += 1;
            }
        }
    }

    finalize_stats(&mut stats, &acc);
    stats
}

/// Placeholder: Load stats from Parquet file
/// Real implementation would use arrow-rs/parquet crate
pub fn infer_schema_from_path(
    _path: &Path,
    _config: &InferenceConfig,
) -> Result<InferredSchema, std::io::Error> {
    // In a real implementation, this would:
    // 1. Open the Parquet file
    // 2. Read schema metadata
    // 3. Sample rows for statistics
    // 4. Call infer_schema()

    // For now, return empty schema
    Ok(InferredSchema::default())
}
