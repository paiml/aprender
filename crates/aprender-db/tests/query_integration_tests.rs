//! Integration tests for SQL query execution
//!
//! These tests validate the complete query pipeline:
//! SQL → Parser → Executor → Results

use arrow::array::{Float64Array, Int32Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use std::sync::Arc;
use trueno_db::query::{QueryEngine, QueryExecutor};
use trueno_db::storage::StorageEngine;

/// Helper function to create test data
fn create_test_data() -> StorageEngine {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("category", DataType::Utf8, false),
        Field::new("value", DataType::Float64, false),
        Field::new("quantity", DataType::Int32, false),
    ]));

    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5])),
            Arc::new(StringArray::from(vec!["A", "B", "A", "C", "B"])),
            Arc::new(Float64Array::from(vec![10.0, 20.0, 30.0, 40.0, 50.0])),
            Arc::new(Int32Array::from(vec![100, 200, 300, 400, 500])),
        ],
    )
    .unwrap();

    let mut storage = StorageEngine::new(vec![]);
    storage.append_batch(batch).unwrap();
    storage
}

#[test]
fn test_simple_select_all() {
    let storage = create_test_data();
    let engine = QueryEngine::new();
    let executor = QueryExecutor::new();

    let plan = engine.parse("SELECT * FROM table1").unwrap();
    let result = executor.execute(&plan, &storage).unwrap();

    assert_eq!(result.num_rows(), 5);
    assert_eq!(result.num_columns(), 4);
}

#[test]
fn test_select_specific_columns() {
    let storage = create_test_data();
    let engine = QueryEngine::new();
    let executor = QueryExecutor::new();

    let plan = engine.parse("SELECT id, value FROM table1").unwrap();
    let result = executor.execute(&plan, &storage).unwrap();

    assert_eq!(result.num_rows(), 5);
    assert_eq!(result.num_columns(), 2);
    assert_eq!(result.schema().field(0).name(), "id");
    assert_eq!(result.schema().field(1).name(), "value");
}

#[test]
fn test_aggregation_sum() {
    let storage = create_test_data();
    let engine = QueryEngine::new();
    let executor = QueryExecutor::new();

    let plan = engine.parse("SELECT SUM(value) FROM table1").unwrap();
    let result = executor.execute(&plan, &storage).unwrap();

    assert_eq!(result.num_rows(), 1);
    let sum_col = result.column(0).as_any().downcast_ref::<Float64Array>().unwrap();
    assert!((sum_col.value(0) - 150.0).abs() < 0.01);
}

#[test]
fn test_aggregation_avg() {
    let storage = create_test_data();
    let engine = QueryEngine::new();
    let executor = QueryExecutor::new();

    let plan = engine.parse("SELECT AVG(value) FROM table1").unwrap();
    let result = executor.execute(&plan, &storage).unwrap();

    assert_eq!(result.num_rows(), 1);
    let avg_col = result.column(0).as_any().downcast_ref::<Float64Array>().unwrap();
    assert!((avg_col.value(0) - 30.0).abs() < 0.01);
}

#[test]
fn test_aggregation_count() {
    let storage = create_test_data();
    let engine = QueryEngine::new();
    let executor = QueryExecutor::new();

    let plan = engine.parse("SELECT COUNT(*) FROM table1").unwrap();
    let result = executor.execute(&plan, &storage).unwrap();

    assert_eq!(result.num_rows(), 1);
    let count_col = result.column(0).as_any().downcast_ref::<arrow::array::Int64Array>().unwrap();
    assert_eq!(count_col.value(0), 5);
}

#[test]
fn test_aggregation_min_max() {
    let storage = create_test_data();
    let engine = QueryEngine::new();
    let executor = QueryExecutor::new();

    let plan = engine.parse("SELECT MIN(value), MAX(value) FROM table1").unwrap();
    let result = executor.execute(&plan, &storage).unwrap();

    assert_eq!(result.num_rows(), 1);
    let min_col = result.column(0).as_any().downcast_ref::<Float64Array>().unwrap();
    let max_col = result.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
    assert!((min_col.value(0) - 10.0).abs() < 0.01);
    assert!((max_col.value(0) - 50.0).abs() < 0.01);
}

#[test]
fn test_where_filter_greater_than() {
    let storage = create_test_data();
    let engine = QueryEngine::new();
    let executor = QueryExecutor::new();

    let plan = engine.parse("SELECT id, value FROM table1 WHERE value > 25.0").unwrap();
    let result = executor.execute(&plan, &storage).unwrap();

    // Should return rows with value 30.0, 40.0, 50.0
    assert_eq!(result.num_rows(), 3);
}

#[test]
fn test_where_filter_less_than() {
    let storage = create_test_data();
    let engine = QueryEngine::new();
    let executor = QueryExecutor::new();

    let plan = engine.parse("SELECT id, value FROM table1 WHERE value < 35.0").unwrap();
    let result = executor.execute(&plan, &storage).unwrap();

    // Should return rows with value 10.0, 20.0, 30.0
    assert_eq!(result.num_rows(), 3);
}

#[test]
fn test_where_filter_equals() {
    let storage = create_test_data();
    let engine = QueryEngine::new();
    let executor = QueryExecutor::new();

    let plan = engine.parse("SELECT id, value FROM table1 WHERE id = 3").unwrap();
    let result = executor.execute(&plan, &storage).unwrap();

    assert_eq!(result.num_rows(), 1);
    let id_col = result.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
    assert_eq!(id_col.value(0), 3);
}

#[test]
fn test_order_by_limit_top_k() {
    let storage = create_test_data();
    let engine = QueryEngine::new();
    let executor = QueryExecutor::new();

    let plan = engine.parse("SELECT id, value FROM table1 ORDER BY value DESC LIMIT 2").unwrap();
    let result = executor.execute(&plan, &storage).unwrap();

    assert_eq!(result.num_rows(), 2);

    let value_col = result.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
    // Top 2 highest values: 50.0, 40.0
    assert!((value_col.value(0) - 50.0).abs() < 0.01);
    assert!((value_col.value(1) - 40.0).abs() < 0.01);
}

#[test]
fn test_order_by_ascending() {
    let storage = create_test_data();
    let engine = QueryEngine::new();
    let executor = QueryExecutor::new();

    let plan = engine.parse("SELECT id, value FROM table1 ORDER BY value ASC LIMIT 2").unwrap();
    let result = executor.execute(&plan, &storage).unwrap();

    assert_eq!(result.num_rows(), 2);

    let value_col = result.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
    // Top 2 lowest values: 10.0, 20.0
    assert!((value_col.value(0) - 10.0).abs() < 0.01);
    assert!((value_col.value(1) - 20.0).abs() < 0.01);
}

#[test]
fn test_limit_without_order_by() {
    let storage = create_test_data();
    let engine = QueryEngine::new();
    let executor = QueryExecutor::new();

    let plan = engine.parse("SELECT id, value FROM table1 LIMIT 3").unwrap();
    let result = executor.execute(&plan, &storage).unwrap();

    assert_eq!(result.num_rows(), 3);
}

#[test]
fn test_aggregation_with_filter() {
    let storage = create_test_data();
    let engine = QueryEngine::new();
    let executor = QueryExecutor::new();

    let plan = engine.parse("SELECT SUM(value) FROM table1 WHERE value > 20.0").unwrap();
    let result = executor.execute(&plan, &storage).unwrap();

    assert_eq!(result.num_rows(), 1);
    let sum_col = result.column(0).as_any().downcast_ref::<Float64Array>().unwrap();
    // Sum of 30.0 + 40.0 + 50.0 = 120.0
    assert!((sum_col.value(0) - 120.0).abs() < 0.01);
}

#[test]
fn test_multiple_aggregations() {
    let storage = create_test_data();
    let engine = QueryEngine::new();
    let executor = QueryExecutor::new();

    let plan = engine
        .parse("SELECT COUNT(*), SUM(value), AVG(value), MIN(value), MAX(value) FROM table1")
        .unwrap();
    let result = executor.execute(&plan, &storage).unwrap();

    assert_eq!(result.num_rows(), 1);
    assert_eq!(result.num_columns(), 5);

    let count =
        result.column(0).as_any().downcast_ref::<arrow::array::Int64Array>().unwrap().value(0);
    let sum = result.column(1).as_any().downcast_ref::<Float64Array>().unwrap().value(0);
    let avg = result.column(2).as_any().downcast_ref::<Float64Array>().unwrap().value(0);
    let min = result.column(3).as_any().downcast_ref::<Float64Array>().unwrap().value(0);
    let max = result.column(4).as_any().downcast_ref::<Float64Array>().unwrap().value(0);

    assert_eq!(count, 5);
    assert!((sum - 150.0).abs() < 0.01);
    assert!((avg - 30.0).abs() < 0.01);
    assert!((min - 10.0).abs() < 0.01);
    assert!((max - 50.0).abs() < 0.01);
}

// Property-based tests using proptest
#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    fn create_property_test_data(values: Vec<f64>) -> StorageEngine {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("value", DataType::Float64, false),
        ]));

        let ids: Vec<i32> = (0..values.len() as i32).collect();

        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(Int32Array::from(ids)), Arc::new(Float64Array::from(values))],
        )
        .unwrap();

        let mut storage = StorageEngine::new(vec![]);
        storage.append_batch(batch).unwrap();
        storage
    }

    proptest! {
        /// Property: COUNT(*) returns correct number of rows
        #[test]
        fn prop_count_returns_correct_rows(
            values in prop::collection::vec(0.0f64..1000.0, 1..1000)
        ) {
            let storage = create_property_test_data(values.clone());
            let engine = QueryEngine::new();
            let executor = QueryExecutor::new();

            let plan = engine.parse("SELECT COUNT(*) FROM table1").unwrap();
            let result = executor.execute(&plan, &storage).unwrap();

            let count = result.column(0).as_any().downcast_ref::<arrow::array::Int64Array>().unwrap().value(0);
            prop_assert_eq!(count as usize, values.len());
        }

        /// Property: SUM aggregation matches scalar sum
        #[test]
        fn prop_sum_matches_scalar(
            values in prop::collection::vec(0.0f64..1000.0, 1..1000)
        ) {
            let storage = create_property_test_data(values.clone());
            let engine = QueryEngine::new();
            let executor = QueryExecutor::new();

            let plan = engine.parse("SELECT SUM(value) FROM table1").unwrap();
            let result = executor.execute(&plan, &storage).unwrap();

            let db_sum = result.column(0).as_any().downcast_ref::<Float64Array>().unwrap().value(0);
            let expected_sum: f64 = values.iter().sum();

            prop_assert!((db_sum - expected_sum).abs() < 0.001, "DB sum: {}, Expected: {}", db_sum, expected_sum);
        }

        /// Property: AVG aggregation matches scalar average
        #[test]
        fn prop_avg_matches_scalar(
            values in prop::collection::vec(1.0f64..1000.0, 1..1000)
        ) {
            let storage = create_property_test_data(values.clone());
            let engine = QueryEngine::new();
            let executor = QueryExecutor::new();

            let plan = engine.parse("SELECT AVG(value) FROM table1").unwrap();
            let result = executor.execute(&plan, &storage).unwrap();

            let db_avg = result.column(0).as_any().downcast_ref::<Float64Array>().unwrap().value(0);
            let expected_avg: f64 = values.iter().sum::<f64>() / values.len() as f64;

            prop_assert!((db_avg - expected_avg).abs() < 0.001);
        }

        /// Property: ORDER BY + LIMIT returns values in correct order
        #[test]
        fn prop_order_by_limit_sorted(
            values in prop::collection::vec(0.0f64..1000.0, 10..1000),
            limit in 1usize..50
        ) {
            let storage = create_property_test_data(values.clone());
            let engine = QueryEngine::new();
            let executor = QueryExecutor::new();

            let sql = format!("SELECT value FROM table1 ORDER BY value DESC LIMIT {limit}");
            let plan = engine.parse(&sql).unwrap();
            let result = executor.execute(&plan, &storage).unwrap();

            // Verify descending order
            let result_col = result.column(0).as_any().downcast_ref::<Float64Array>().unwrap();
            for i in 0..result.num_rows().saturating_sub(1) {
                prop_assert!(
                    result_col.value(i) >= result_col.value(i + 1),
                    "Not in descending order at index {}: {} < {}",
                    i,
                    result_col.value(i),
                    result_col.value(i + 1)
                );
            }

            // Verify count
            let expected_count = limit.min(values.len());
            prop_assert_eq!(result.num_rows(), expected_count);
        }
    }
}
