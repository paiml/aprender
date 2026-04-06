//! Tests for auto-feature type inference

use super::*;
use proptest::prelude::*;

fn make_stats(
    name: &str,
    count: usize,
    unique: usize,
    all_numeric: bool,
    all_int: bool,
) -> ColumnStats {
    ColumnStats {
        name: name.to_string(),
        count,
        unique_count: unique,
        all_numeric,
        all_integers: all_int,
        ..Default::default()
    }
}

// ============================================================
// Unit Tests
// ============================================================

#[test]
fn test_infer_numeric() {
    let stats = make_stats("price", 1000, 500, true, false);
    let config = InferenceConfig::default();
    assert_eq!(infer_type(&stats, &config), FeatureType::Numeric);
}

#[test]
fn test_infer_categorical_low_cardinality() {
    // Use column name that won't match target heuristics
    let stats = make_stats("status_code", 1000, 10, true, true);
    let config = InferenceConfig::default();
    assert_eq!(infer_type(&stats, &config), FeatureType::Categorical);
}

#[test]
fn test_infer_binary_target() {
    let stats = make_stats("label", 1000, 2, true, true);
    let config = InferenceConfig::default();
    assert_eq!(infer_type(&stats, &config), FeatureType::BinaryTarget);
}

#[test]
fn test_infer_multiclass_target() {
    let stats = make_stats("target", 1000, 10, true, true);
    let config = InferenceConfig::default();
    assert_eq!(infer_type(&stats, &config), FeatureType::MultiClassTarget);
}

#[test]
fn test_infer_regression_target() {
    let stats = make_stats("y", 1000, 800, true, false);
    let config = InferenceConfig::default();
    assert_eq!(infer_type(&stats, &config), FeatureType::RegressionTarget);
}

#[test]
fn test_infer_text() {
    let mut stats = make_stats("description", 1000, 900, false, false);
    stats.avg_str_len = Some(100.0);
    let config = InferenceConfig::default();
    assert_eq!(infer_type(&stats, &config), FeatureType::Text);
}

#[test]
fn test_infer_categorical_string() {
    let mut stats = make_stats("status", 1000, 5, false, false);
    stats.avg_str_len = Some(8.0);
    let config = InferenceConfig::default();
    assert_eq!(infer_type(&stats, &config), FeatureType::Categorical);
}

#[test]
fn test_infer_datetime() {
    let mut stats = make_stats("created_at", 1000, 1000, false, false);
    stats.looks_like_datetime = true;
    let config = InferenceConfig::default();
    assert_eq!(infer_type(&stats, &config), FeatureType::DateTime);
}

#[test]
fn test_infer_embedding() {
    let mut stats = make_stats("embedding", 1000, 1000, true, false);
    stats.is_array = true;
    stats.array_len = Some(768);
    let config = InferenceConfig::default();
    assert_eq!(infer_type(&stats, &config), FeatureType::Embedding);
}

#[test]
fn test_infer_schema() {
    let stats = vec![
        make_stats("id", 1000, 1000, true, true),
        make_stats("label", 1000, 2, true, true),
        make_stats("price", 1000, 500, true, false),
    ];
    let config = InferenceConfig::default();
    let schema = infer_schema(stats, &config);

    assert_eq!(schema.features.len(), 3);
    assert_eq!(schema.features["label"], FeatureType::BinaryTarget);
    assert_eq!(schema.features["price"], FeatureType::Numeric);
}

#[test]
fn test_schema_targets() {
    let stats = vec![
        make_stats("x1", 100, 50, true, false),
        make_stats("x2", 100, 50, true, false),
        make_stats("y", 100, 80, true, false),
    ];
    let config = InferenceConfig::default();
    let schema = infer_schema(stats, &config);

    let targets = schema.targets();
    assert_eq!(targets.len(), 1);
    assert!(targets.contains(&"y"));
}

#[test]
fn test_schema_inputs() {
    let stats = vec![
        make_stats("x1", 100, 50, true, false),
        make_stats("x2", 100, 50, true, false),
        make_stats("y", 100, 80, true, false),
    ];
    let config = InferenceConfig::default();
    let schema = infer_schema(stats, &config);

    let inputs = schema.inputs();
    assert_eq!(inputs.len(), 2);
    assert!(inputs.contains(&"x1"));
    assert!(inputs.contains(&"x2"));
}

#[test]
fn test_collect_stats_numeric() {
    let values: Vec<Option<&str>> = vec![Some("1.5"), Some("2.3"), Some("3.7"), None, Some("4.1")];
    let stats = collect_stats_from_samples("price", &values);

    assert_eq!(stats.count, 5);
    assert_eq!(stats.null_count, 1);
    assert!(stats.all_numeric);
    assert!(!stats.all_integers);
}

#[test]
fn test_collect_stats_integers() {
    let values: Vec<Option<&str>> = vec![Some("1"), Some("2"), Some("3"), Some("4"), Some("5")];
    let stats = collect_stats_from_samples("count", &values);

    assert!(stats.all_numeric);
    assert!(stats.all_integers);
    assert_eq!(stats.unique_count, 5);
}

#[test]
fn test_collect_stats_datetime() {
    let values: Vec<Option<&str>> =
        vec![Some("2024-01-15"), Some("2024-02-20"), Some("2024-03-25")];
    let stats = collect_stats_from_samples("date", &values);

    assert!(stats.looks_like_datetime);
}

#[test]
fn test_cardinality_ratio() {
    let stats = ColumnStats { count: 1000, unique_count: 50, ..Default::default() };
    assert!((stats.cardinality_ratio() - 0.05).abs() < 1e-6);
}

#[test]
fn test_null_ratio() {
    let stats = ColumnStats { count: 100, null_count: 10, ..Default::default() };
    assert!((stats.null_ratio() - 0.1).abs() < 1e-6);
}

#[test]
fn test_feature_type_display() {
    assert_eq!(format!("{}", FeatureType::Numeric), "numeric");
    assert_eq!(format!("{}", FeatureType::Categorical), "categorical");
    assert_eq!(format!("{}", FeatureType::BinaryTarget), "binary_target");
}

// ============================================================
// Property Tests
// ============================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    #[test]
    fn prop_cardinality_ratio_bounded(
        count in 1usize..10000,
        unique in 1usize..10000
    ) {
        let unique = unique.min(count);
        let stats = ColumnStats {
            count,
            unique_count: unique,
            ..Default::default()
        };
        let ratio = stats.cardinality_ratio();
        prop_assert!((0.0..=1.0).contains(&ratio));
    }

    #[test]
    fn prop_null_ratio_bounded(
        count in 1usize..10000,
        null_count in 0usize..10000
    ) {
        let null_count = null_count.min(count);
        let stats = ColumnStats {
            count,
            null_count,
            ..Default::default()
        };
        let ratio = stats.null_ratio();
        prop_assert!((0.0..=1.0).contains(&ratio));
    }

    #[test]
    fn prop_numeric_low_cardinality_is_categorical(
        count in 1000usize..10000,
        unique in 2usize..20
    ) {
        // Ensure cardinality ratio < 0.05 threshold
        let unique = unique.min((count as f32 * 0.04) as usize).max(2);
        let stats = make_stats("feature_col", count, unique, true, true);
        // Use empty target columns to avoid accidental target matching
        let config = InferenceConfig {
            target_columns: vec![],
            ..Default::default()
        };
        let inferred = infer_type(&stats, &config);
        // Low cardinality integers should be categorical
        prop_assert_eq!(inferred, FeatureType::Categorical);
    }

    #[test]
    fn prop_high_cardinality_numeric_stays_numeric(
        count in 1000usize..10000,
        unique_ratio in 0.5f32..1.0
    ) {
        let unique = (count as f32 * unique_ratio) as usize;
        let stats = make_stats("col", count, unique, true, false);
        let config = InferenceConfig::default();
        let inferred = infer_type(&stats, &config);
        prop_assert_eq!(inferred, FeatureType::Numeric);
    }

    #[test]
    fn prop_binary_target_detected(
        count in 100usize..10000
    ) {
        let stats = make_stats("label", count, 2, true, true);
        let config = InferenceConfig::default();
        let inferred = infer_type(&stats, &config);
        prop_assert_eq!(inferred, FeatureType::BinaryTarget);
    }

    #[test]
    fn prop_multiclass_target_detected(
        count in 100usize..10000,
        classes in 3usize..50
    ) {
        let stats = make_stats("target", count, classes, true, true);
        let config = InferenceConfig::default();
        let inferred = infer_type(&stats, &config);
        prop_assert_eq!(inferred, FeatureType::MultiClassTarget);
    }

    #[test]
    fn prop_text_detected_by_length(
        count in 100usize..1000,
        avg_len in 50.0f32..500.0
    ) {
        let mut stats = make_stats("description", count, count, false, false);
        stats.avg_str_len = Some(avg_len);
        let config = InferenceConfig::default();
        let inferred = infer_type(&stats, &config);
        prop_assert_eq!(inferred, FeatureType::Text);
    }

    #[test]
    fn prop_embedding_detected(
        count in 100usize..1000,
        embed_dim in 64usize..2048
    ) {
        let mut stats = make_stats("embedding", count, count, true, false);
        stats.is_array = true;
        stats.array_len = Some(embed_dim);
        let config = InferenceConfig::default();
        let inferred = infer_type(&stats, &config);
        prop_assert_eq!(inferred, FeatureType::Embedding);
    }

    #[test]
    fn prop_datetime_detected(count in 100usize..1000) {
        let mut stats = make_stats("timestamp", count, count, false, false);
        stats.looks_like_datetime = true;
        let config = InferenceConfig::default();
        let inferred = infer_type(&stats, &config);
        prop_assert_eq!(inferred, FeatureType::DateTime);
    }

    #[test]
    fn prop_schema_preserves_all_columns(
        num_cols in 1usize..20
    ) {
        let stats: Vec<ColumnStats> = (0..num_cols)
            .map(|i| make_stats(&format!("col_{i}"), 100, 50, true, false))
            .collect();
        let config = InferenceConfig::default();
        let schema = infer_schema(stats, &config);
        prop_assert_eq!(schema.features.len(), num_cols);
    }

    #[test]
    fn prop_targets_and_inputs_partition(
        num_features in 1usize..10,
        num_targets in 0usize..3
    ) {
        let mut stats: Vec<ColumnStats> = (0..num_features)
            .map(|i| make_stats(&format!("x{i}"), 100, 50, true, false))
            .collect();

        // Add targets
        for i in 0..num_targets {
            stats.push(make_stats(&format!("y{i}"), 100, 80, true, false));
        }

        let mut config = InferenceConfig::default();
        config.target_columns = (0..num_targets).map(|i| format!("y{i}")).collect();

        let schema = infer_schema(stats, &config);

        let inputs = schema.inputs();
        let targets = schema.targets();

        // Inputs and targets should partition all columns
        prop_assert_eq!(inputs.len() + targets.len(), num_features + num_targets);
    }
}

// ============================================================
// Additional Coverage Tests
// ============================================================

#[test]
fn test_feature_type_display_all() {
    assert_eq!(format!("{}", FeatureType::Text), "text");
    assert_eq!(format!("{}", FeatureType::DateTime), "datetime");
    assert_eq!(format!("{}", FeatureType::Embedding), "embedding");
    assert_eq!(format!("{}", FeatureType::MultiClassTarget), "multiclass_target");
    assert_eq!(format!("{}", FeatureType::RegressionTarget), "regression_target");
    assert_eq!(format!("{}", FeatureType::TokenSequence), "token_sequence");
    assert_eq!(format!("{}", FeatureType::Unknown), "unknown");
}

#[test]
fn test_column_stats_new() {
    let stats = ColumnStats::new("my_column");
    assert_eq!(stats.name, "my_column");
    assert_eq!(stats.count, 0);
    assert_eq!(stats.unique_count, 0);
}

#[test]
fn test_cardinality_ratio_zero_count() {
    let stats = ColumnStats { count: 0, unique_count: 0, ..Default::default() };
    assert_eq!(stats.cardinality_ratio(), 0.0);
}

#[test]
fn test_null_ratio_zero_count() {
    let stats = ColumnStats { count: 0, null_count: 0, ..Default::default() };
    assert_eq!(stats.null_ratio(), 0.0);
}

#[test]
fn test_inference_config_default() {
    let config = InferenceConfig::default();
    // Default has some target columns defined
    assert!(!config.target_columns.is_empty());
    assert!(config.categorical_threshold > 0.0);
    assert!(config.text_min_avg_len > 0.0);
    assert!(config.exclude_columns.is_empty());
}

#[test]
fn test_infer_unknown_type() {
    // Empty column stats with no distinguishing features
    let stats = ColumnStats {
        name: "weird".to_string(),
        count: 0,
        unique_count: 0,
        all_numeric: false,
        all_integers: false,
        ..Default::default()
    };
    let config = InferenceConfig { target_columns: vec![], ..Default::default() };
    // With count=0, it should fall through to Unknown
    assert_eq!(infer_type(&stats, &config), FeatureType::Unknown);
}

#[test]
fn test_infer_array_type() {
    let stats = ColumnStats {
        name: "embedding".to_string(),
        count: 100,
        unique_count: 100,
        is_array: true,
        array_len: Some(768), // Large enough for embedding
        all_numeric: true,
        ..Default::default()
    };
    let config = InferenceConfig { target_columns: vec![], ..Default::default() };
    let result = infer_type(&stats, &config);
    // Numeric array with large dimension is embedding
    assert_eq!(result, FeatureType::Embedding);
}

#[test]
fn test_collect_stats_text() {
    let values: Vec<Option<&str>> = vec![
        Some("This is a long text sentence for testing purposes"),
        Some("Another lengthy text example with multiple words"),
        Some("Yet another piece of textual content"),
    ];
    let stats = collect_stats_from_samples("content", &values);

    assert!(!stats.all_numeric);
    assert!(stats.avg_str_len.is_some());
    assert!(stats.avg_str_len.expect("operation should succeed") > 30.0);
}

#[test]
fn test_collect_stats_all_nulls() {
    let values: Vec<Option<&str>> = vec![None, None, None];
    let stats = collect_stats_from_samples("nulls", &values);

    assert_eq!(stats.count, 3);
    assert_eq!(stats.null_count, 3);
}

#[test]
fn test_schema_get_feature_type() {
    let stats = vec![make_stats("x", 100, 50, true, false), make_stats("y", 100, 2, true, true)];
    let config = InferenceConfig::default();
    let schema = infer_schema(stats, &config);

    assert_eq!(schema.features.get("x"), Some(&FeatureType::Numeric));
    assert_eq!(schema.features.get("y"), Some(&FeatureType::BinaryTarget));
    assert_eq!(schema.features.get("nonexistent"), None);
}
