//! Coverage tests for `QueryExecutor` to achieve 90%+ coverage
//!
//! These tests cover error paths and edge cases that weren't covered
//! by the integration tests

use arrow::array::{Float32Array, Float64Array, Int32Array, Int64Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use std::sync::Arc;
use trueno_db::query::{QueryEngine, QueryExecutor};
use trueno_db::storage::StorageEngine;
use trueno_db::Error;

/// Helper function to create test data with all numeric types
fn create_multi_type_data() -> StorageEngine {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id_i32", DataType::Int32, false),
        Field::new("id_i64", DataType::Int64, false),
        Field::new("value_f32", DataType::Float32, false),
        Field::new("value_f64", DataType::Float64, false),
        Field::new("name", DataType::Utf8, false),
    ]));

    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5])),
            Arc::new(Int64Array::from(vec![10i64, 20, 30, 40, 50])),
            Arc::new(Float32Array::from(vec![1.5f32, 2.5, 3.5, 4.5, 5.5])),
            Arc::new(Float64Array::from(vec![10.5, 20.5, 30.5, 40.5, 50.5])),
            Arc::new(StringArray::from(vec!["a", "b", "c", "d", "e"])),
        ],
    )
    .unwrap();

    let mut storage = StorageEngine::new(vec![]);
    storage.append_batch(batch).unwrap();
    storage
}

// ============================================================================
// Error Cases
// ============================================================================

#[test]
fn test_execute_empty_storage() {
    let storage = StorageEngine::new(vec![]);
    let engine = QueryEngine::new();
    let executor = QueryExecutor::new();

    let plan = engine.parse("SELECT * FROM table1").unwrap();
    let result = executor.execute(&plan, &storage);

    assert!(result.is_err());
    match result.unwrap_err() {
        Error::InvalidInput(msg) => assert!(msg.contains("No data in storage")),
        _ => panic!("Expected InvalidInput error"),
    }
}

#[test]
fn test_filter_invalid_column() {
    let storage = create_multi_type_data();
    let engine = QueryEngine::new();
    let executor = QueryExecutor::new();

    let plan = engine.parse("SELECT * FROM table1 WHERE nonexistent > 10").unwrap();
    let result = executor.execute(&plan, &storage);

    assert!(result.is_err());
    match result.unwrap_err() {
        Error::InvalidInput(msg) => assert!(msg.contains("Column not found")),
        _ => panic!("Expected InvalidInput error for missing column"),
    }
}

#[test]
fn test_filter_unsupported_datatype() {
    let storage = create_multi_type_data();
    let engine = QueryEngine::new();
    let executor = QueryExecutor::new();

    // Filtering on string column should fail
    let plan = engine.parse("SELECT * FROM table1 WHERE name > 'test'").unwrap();
    let result = executor.execute(&plan, &storage);

    assert!(result.is_err());
    match result.unwrap_err() {
        Error::InvalidInput(msg) => assert!(msg.contains("Filter not supported for data type")),
        _ => panic!("Expected InvalidInput error for unsupported type"),
    }
}

#[test]
fn test_aggregation_on_string_column() {
    let storage = create_multi_type_data();
    let engine = QueryEngine::new();
    let executor = QueryExecutor::new();

    let plan = engine.parse("SELECT SUM(name) FROM table1").unwrap();
    let result = executor.execute(&plan, &storage);

    assert!(result.is_err());
    match result.unwrap_err() {
        Error::InvalidInput(msg) => {
            assert!(msg.contains("Aggregation not supported for data type"));
        }
        _ => panic!("Expected InvalidInput error for aggregation on string"),
    }
}

#[test]
fn test_order_by_invalid_column() {
    let storage = create_multi_type_data();
    let engine = QueryEngine::new();
    let executor = QueryExecutor::new();

    let plan = engine.parse("SELECT * FROM table1 ORDER BY nonexistent DESC").unwrap();
    let result = executor.execute(&plan, &storage);

    assert!(result.is_err());
    match result.unwrap_err() {
        Error::InvalidInput(msg) => assert!(msg.contains("Column not found")),
        _ => panic!("Expected InvalidInput error for invalid ORDER BY column"),
    }
}

#[test]
fn test_project_invalid_column() {
    let storage = create_multi_type_data();
    let engine = QueryEngine::new();
    let executor = QueryExecutor::new();

    let plan = engine.parse("SELECT nonexistent FROM table1").unwrap();
    let result = executor.execute(&plan, &storage);

    assert!(result.is_err());
    match result.unwrap_err() {
        Error::InvalidInput(msg) => assert!(msg.contains("Column not found")),
        _ => panic!("Expected InvalidInput error for projection"),
    }
}

#[test]
fn test_aggregation_invalid_column() {
    let storage = create_multi_type_data();
    let engine = QueryEngine::new();
    let executor = QueryExecutor::new();

    let plan = engine.parse("SELECT SUM(nonexistent) FROM table1").unwrap();
    let result = executor.execute(&plan, &storage);

    assert!(result.is_err());
    match result.unwrap_err() {
        Error::InvalidInput(msg) => assert!(msg.contains("Column not found")),
        _ => panic!("Expected InvalidInput error for aggregation on missing column"),
    }
}

// ============================================================================
// All Data Type Coverage for Aggregations
// ============================================================================

#[test]
fn test_sum_all_numeric_types() {
    let storage = create_multi_type_data();
    let engine = QueryEngine::new();
    let executor = QueryExecutor::new();

    // Int32
    let plan = engine.parse("SELECT SUM(id_i32) FROM table1").unwrap();
    let result = executor.execute(&plan, &storage).unwrap();
    assert_eq!(result.num_rows(), 1);

    // Int64
    let plan = engine.parse("SELECT SUM(id_i64) FROM table1").unwrap();
    let result = executor.execute(&plan, &storage).unwrap();
    assert_eq!(result.num_rows(), 1);

    // Float32
    let plan = engine.parse("SELECT SUM(value_f32) FROM table1").unwrap();
    let result = executor.execute(&plan, &storage).unwrap();
    assert_eq!(result.num_rows(), 1);

    // Float64
    let plan = engine.parse("SELECT SUM(value_f64) FROM table1").unwrap();
    let result = executor.execute(&plan, &storage).unwrap();
    assert_eq!(result.num_rows(), 1);
}

#[test]
fn test_avg_all_numeric_types() {
    let storage = create_multi_type_data();
    let engine = QueryEngine::new();
    let executor = QueryExecutor::new();

    // Int32
    let plan = engine.parse("SELECT AVG(id_i32) FROM table1").unwrap();
    let result = executor.execute(&plan, &storage).unwrap();
    assert_eq!(result.num_rows(), 1);

    // Int64
    let plan = engine.parse("SELECT AVG(id_i64) FROM table1").unwrap();
    let result = executor.execute(&plan, &storage).unwrap();
    assert_eq!(result.num_rows(), 1);

    // Float32
    let plan = engine.parse("SELECT AVG(value_f32) FROM table1").unwrap();
    let result = executor.execute(&plan, &storage).unwrap();
    assert_eq!(result.num_rows(), 1);

    // Float64
    let plan = engine.parse("SELECT AVG(value_f64) FROM table1").unwrap();
    let result = executor.execute(&plan, &storage).unwrap();
    assert_eq!(result.num_rows(), 1);
}

#[test]
fn test_min_max_all_numeric_types() {
    let storage = create_multi_type_data();
    let engine = QueryEngine::new();
    let executor = QueryExecutor::new();

    // Int32
    let plan = engine.parse("SELECT MIN(id_i32), MAX(id_i32) FROM table1").unwrap();
    let result = executor.execute(&plan, &storage).unwrap();
    assert_eq!(result.num_rows(), 1);
    assert_eq!(result.num_columns(), 2);

    // Int64
    let plan = engine.parse("SELECT MIN(id_i64), MAX(id_i64) FROM table1").unwrap();
    let result = executor.execute(&plan, &storage).unwrap();
    assert_eq!(result.num_rows(), 1);

    // Float32
    let plan = engine.parse("SELECT MIN(value_f32), MAX(value_f32) FROM table1").unwrap();
    let result = executor.execute(&plan, &storage).unwrap();
    assert_eq!(result.num_rows(), 1);

    // Float64
    let plan = engine.parse("SELECT MIN(value_f64), MAX(value_f64) FROM table1").unwrap();
    let result = executor.execute(&plan, &storage).unwrap();
    assert_eq!(result.num_rows(), 1);
}

// ============================================================================
// All Filter Operators on All Numeric Types
// ============================================================================

#[test]
fn test_filter_operators_int32() {
    let storage = create_multi_type_data();
    let engine = QueryEngine::new();
    let executor = QueryExecutor::new();

    // Test all operators: >, >=, <, <=, =, !=
    let operators = vec![
        ("id_i32 > 2", 3),
        ("id_i32 >= 3", 3),
        ("id_i32 < 4", 3),
        ("id_i32 <= 3", 3),
        ("id_i32 = 3", 1),
        ("id_i32 != 3", 4),
    ];

    for (filter, expected_rows) in operators {
        let sql = format!("SELECT * FROM table1 WHERE {filter}");
        let plan = engine.parse(&sql).unwrap();
        let result = executor.execute(&plan, &storage).unwrap();
        assert_eq!(
            result.num_rows(),
            expected_rows,
            "Filter '{filter}' should return {expected_rows} rows"
        );
    }
}

#[test]
fn test_filter_operators_int64() {
    let storage = create_multi_type_data();
    let engine = QueryEngine::new();
    let executor = QueryExecutor::new();

    let operators = vec![
        ("id_i64 > 20", 3),
        ("id_i64 >= 30", 3),
        ("id_i64 < 40", 3),
        ("id_i64 <= 30", 3),
        ("id_i64 = 30", 1),
        ("id_i64 != 30", 4),
    ];

    for (filter, expected_rows) in operators {
        let sql = format!("SELECT * FROM table1 WHERE {filter}");
        let plan = engine.parse(&sql).unwrap();
        let result = executor.execute(&plan, &storage).unwrap();
        assert_eq!(result.num_rows(), expected_rows);
    }
}

#[test]
fn test_filter_operators_float32() {
    let storage = create_multi_type_data();
    let engine = QueryEngine::new();
    let executor = QueryExecutor::new();

    let operators = vec![
        ("value_f32 > 2.5", 3),
        ("value_f32 >= 3.5", 3),
        ("value_f32 < 4.5", 3),
        ("value_f32 <= 3.5", 3),
        ("value_f32 = 3.5", 1),
        ("value_f32 != 3.5", 4),
    ];

    for (filter, expected_rows) in operators {
        let sql = format!("SELECT * FROM table1 WHERE {filter}");
        let plan = engine.parse(&sql).unwrap();
        let result = executor.execute(&plan, &storage).unwrap();
        assert_eq!(result.num_rows(), expected_rows);
    }
}

#[test]
fn test_filter_operators_float64() {
    let storage = create_multi_type_data();
    let engine = QueryEngine::new();
    let executor = QueryExecutor::new();

    let operators = vec![
        ("value_f64 > 20.5", 3),
        ("value_f64 >= 30.5", 3),
        ("value_f64 < 40.5", 3),
        ("value_f64 <= 30.5", 3),
        ("value_f64 = 30.5", 1),
        ("value_f64 != 30.5", 4),
    ];

    for (filter, expected_rows) in operators {
        let sql = format!("SELECT * FROM table1 WHERE {filter}");
        let plan = engine.parse(&sql).unwrap();
        let result = executor.execute(&plan, &storage).unwrap();
        assert_eq!(result.num_rows(), expected_rows);
    }
}

// ============================================================================
// ORDER BY Coverage for All Types
// ============================================================================

#[test]
fn test_order_by_all_numeric_types() {
    let storage = create_multi_type_data();
    let engine = QueryEngine::new();
    let executor = QueryExecutor::new();

    // Int32
    let plan = engine.parse("SELECT * FROM table1 ORDER BY id_i32 DESC LIMIT 2").unwrap();
    let result = executor.execute(&plan, &storage).unwrap();
    assert_eq!(result.num_rows(), 2);

    // Int64
    let plan = engine.parse("SELECT * FROM table1 ORDER BY id_i64 ASC LIMIT 2").unwrap();
    let result = executor.execute(&plan, &storage).unwrap();
    assert_eq!(result.num_rows(), 2);

    // Float32
    let plan = engine.parse("SELECT * FROM table1 ORDER BY value_f32 DESC LIMIT 3").unwrap();
    let result = executor.execute(&plan, &storage).unwrap();
    assert_eq!(result.num_rows(), 3);

    // Float64
    let plan = engine.parse("SELECT * FROM table1 ORDER BY value_f64 ASC LIMIT 3").unwrap();
    let result = executor.execute(&plan, &storage).unwrap();
    assert_eq!(result.num_rows(), 3);
}

// ============================================================================
// Edge Cases
// ============================================================================

#[test]
fn test_filter_with_invalid_parse() {
    let storage = create_multi_type_data();
    let engine = QueryEngine::new();
    let executor = QueryExecutor::new();

    // Filter expression too short (< 3 tokens)
    let plan = engine.parse("SELECT * FROM table1 WHERE invalid").unwrap();
    let result = executor.execute(&plan, &storage);

    assert!(result.is_err());
}

#[test]
fn test_filter_invalid_number_format() {
    let storage = create_multi_type_data();
    let engine = QueryEngine::new();
    let executor = QueryExecutor::new();

    // Invalid number in filter
    let plan = engine.parse("SELECT * FROM table1 WHERE id_i32 > not_a_number").unwrap();
    let result = executor.execute(&plan, &storage);

    assert!(result.is_err());
    match result.unwrap_err() {
        Error::ParseError(msg) => assert!(msg.contains("Invalid Int32 value")),
        _ => panic!("Expected ParseError"),
    }
}

#[test]
fn test_count_with_all_types() {
    let storage = create_multi_type_data();
    let engine = QueryEngine::new();
    let executor = QueryExecutor::new();

    // COUNT works on any column
    let plan = engine.parse("SELECT COUNT(id_i32) FROM table1").unwrap();
    let result = executor.execute(&plan, &storage).unwrap();
    assert_eq!(result.num_rows(), 1);

    let plan = engine.parse("SELECT COUNT(value_f32) FROM table1").unwrap();
    let result = executor.execute(&plan, &storage).unwrap();
    assert_eq!(result.num_rows(), 1);
}

#[test]
fn test_multiple_batches_combine() {
    // Test with multiple batches that need combining
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("value", DataType::Float64, false),
    ]));

    let batch1 = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1, 2])),
            Arc::new(Float64Array::from(vec![10.0, 20.0])),
        ],
    )
    .unwrap();

    let batch2 = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(vec![3, 4])),
            Arc::new(Float64Array::from(vec![30.0, 40.0])),
        ],
    )
    .unwrap();

    let mut storage = StorageEngine::new(vec![]);
    storage.append_batch(batch1).unwrap();
    storage.append_batch(batch2).unwrap();

    let engine = QueryEngine::new();
    let executor = QueryExecutor::new();

    let plan = engine.parse("SELECT SUM(value) FROM table1").unwrap();
    let result = executor.execute(&plan, &storage).unwrap();

    assert_eq!(result.num_rows(), 1);
    let sum = result.column(0).as_any().downcast_ref::<Float64Array>().unwrap().value(0);
    assert!((sum - 100.0).abs() < 0.01);
}
