//! Comprehensive property-based tests for trueno-db
//!
//! Following ruchy/trueno/aprender pattern:
//! - Test mathematical invariants
//! - Test data integrity properties
//! - Run with `ProptestConfig::with_cases(100)`
//! - Must complete in <30 seconds for pre-commit hook

use arrow::array::{Float64Array, Int32Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use proptest::prelude::*;
use std::sync::Arc;
use trueno_db::storage::StorageEngine;
use trueno_db::topk::{SortOrder, TopKSelection};

// ============================================================================
// Property Test Generators (Strategies)
// ============================================================================

/// Generate `RecordBatch` with random data
fn arb_record_batch(rows: usize) -> impl Strategy<Value = RecordBatch> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("value", DataType::Float64, false),
    ]));

    (proptest::collection::vec(0i32..10000, rows), proptest::collection::vec(0.0f64..1000.0, rows))
        .prop_map(move |(ids, values)| {
            RecordBatch::try_new(
                schema.clone(),
                vec![Arc::new(Int32Array::from(ids)), Arc::new(Float64Array::from(values))],
            )
            .unwrap()
        })
}

/// Generate k value (1..=rows)
fn arb_k_value(max_rows: usize) -> impl Strategy<Value = usize> {
    1..=max_rows.min(100)
}

// ============================================================================
// Property Tests
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    // ========================================================================
    // Top-K Selection Properties
    // ========================================================================

    /// Property: Top-K returns at most k rows
    #[test]
    fn prop_topk_returns_at_most_k_rows(
        batch in arb_record_batch(50),
        k in arb_k_value(50)
    ) {
        let result = batch.top_k(1, k, SortOrder::Descending).unwrap();
        prop_assert!(result.num_rows() <= k);
        prop_assert!(result.num_rows() <= batch.num_rows());
    }

    /// Property: Top-K descending is monotonically decreasing
    #[test]
    fn prop_topk_descending_monotonic(
        batch in arb_record_batch(50),
        k in arb_k_value(50)
    ) {
        let result = batch.top_k(1, k, SortOrder::Descending).unwrap();
        let values = result.column(1).as_any().downcast_ref::<Float64Array>().unwrap();

        for i in 0..values.len().saturating_sub(1) {
            prop_assert!(
                values.value(i) >= values.value(i + 1),
                "Not monotonic decreasing at index {}: {} < {}",
                i,
                values.value(i),
                values.value(i + 1)
            );
        }
    }

    /// Property: Top-K ascending is monotonically increasing
    #[test]
    fn prop_topk_ascending_monotonic(
        batch in arb_record_batch(50),
        k in arb_k_value(50)
    ) {
        let result = batch.top_k(1, k, SortOrder::Ascending).unwrap();
        let values = result.column(1).as_any().downcast_ref::<Float64Array>().unwrap();

        for i in 0..values.len().saturating_sub(1) {
            prop_assert!(
                values.value(i) <= values.value(i + 1),
                "Not monotonic increasing at index {}: {} > {}",
                i,
                values.value(i),
                values.value(i + 1)
            );
        }
    }

    /// Property: Top-K preserves schema
    #[test]
    fn prop_topk_preserves_schema(
        batch in arb_record_batch(50),
        k in arb_k_value(50)
    ) {
        let result = batch.top_k(1, k, SortOrder::Descending).unwrap();
        prop_assert_eq!(batch.schema(), result.schema());
        prop_assert_eq!(batch.num_columns(), result.num_columns());
    }

    /// Property: Top-K with k >= n returns all rows (sorted)
    #[test]
    fn prop_topk_k_gte_n_returns_all(
        batch in arb_record_batch(20)
    ) {
        let k = batch.num_rows() + 10;
        let result = batch.top_k(1, k, SortOrder::Descending).unwrap();
        prop_assert_eq!(result.num_rows(), batch.num_rows());
    }

    /// Property: Top-1 returns single largest element (descending)
    #[test]
    fn prop_top1_is_maximum(
        batch in arb_record_batch(50)
    ) {
        let result = batch.top_k(1, 1, SortOrder::Descending).unwrap();
        prop_assert_eq!(result.num_rows(), 1);

        let result_val = result.column(1).as_any().downcast_ref::<Float64Array>().unwrap().value(0);
        let original_vals = batch.column(1).as_any().downcast_ref::<Float64Array>().unwrap();

        // Result should be >= all original values
        for i in 0..original_vals.len() {
            prop_assert!(result_val >= original_vals.value(i));
        }
    }

    /// Property: Top-1 returns single smallest element (ascending)
    #[test]
    fn prop_top1_is_minimum(
        batch in arb_record_batch(50)
    ) {
        let result = batch.top_k(1, 1, SortOrder::Ascending).unwrap();
        prop_assert_eq!(result.num_rows(), 1);

        let result_val = result.column(1).as_any().downcast_ref::<Float64Array>().unwrap().value(0);
        let original_vals = batch.column(1).as_any().downcast_ref::<Float64Array>().unwrap();

        // Result should be <= all original values
        for i in 0..original_vals.len() {
            prop_assert!(result_val <= original_vals.value(i));
        }
    }

    // ========================================================================
    // Storage Engine Properties
    // ========================================================================

    /// Property: Storage engine preserves row count
    #[test]
    fn prop_storage_preserves_rows(
        batch in arb_record_batch(100)
    ) {
        let original_rows = batch.num_rows();
        let mut storage = StorageEngine::new(vec![]);
        storage.append_batch(batch).unwrap();

        let total_rows: usize = storage.batches().iter().map(arrow::array::RecordBatch::num_rows).sum();
        prop_assert_eq!(total_rows, original_rows);
    }

    /// Property: Morsel iterator preserves all rows
    #[test]
    fn prop_morsel_iterator_no_data_loss(
        batch in arb_record_batch(100)
    ) {
        let original_rows = batch.num_rows();
        let mut storage = StorageEngine::new(vec![]);
        storage.append_batch(batch).unwrap();

        let morsel_rows: usize = storage.morsels()
            .map(|m| m.num_rows())
            .sum();

        prop_assert_eq!(morsel_rows, original_rows);
    }

    /// Property: Multiple batches preserve total row count
    #[test]
    fn prop_multiple_batches_preserve_count(
        batch1 in arb_record_batch(30),
        batch2 in arb_record_batch(40)
    ) {
        let total_rows = batch1.num_rows() + batch2.num_rows();
        let mut storage = StorageEngine::new(vec![]);
        storage.append_batch(batch1).unwrap();
        storage.append_batch(batch2).unwrap();

        let stored_rows: usize = storage.batches().iter().map(arrow::array::RecordBatch::num_rows).sum();
        prop_assert_eq!(stored_rows, total_rows);
    }

    // ========================================================================
    // Algebra Properties
    // ========================================================================

    /// Property: Top-K commutes with Top-K (idempotent on sorted data)
    #[test]
    fn prop_topk_idempotent_on_sorted(
        batch in arb_record_batch(50),
        k in arb_k_value(50)
    ) {
        let once = batch.top_k(1, k, SortOrder::Descending).unwrap();
        let twice = once.top_k(1, k.min(once.num_rows()), SortOrder::Descending).unwrap();

        prop_assert_eq!(once.num_rows(), twice.num_rows());

        let vals_once = once.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
        let vals_twice = twice.column(1).as_any().downcast_ref::<Float64Array>().unwrap();

        for i in 0..once.num_rows() {
            prop_assert!((vals_once.value(i) - vals_twice.value(i)).abs() < 1e-10);
        }
    }
}
