//! Query execution engine
//!
//! Executes parsed SQL queries against Arrow storage using GPU/SIMD backends.
//!
//! Toyota Way Principles:
//! - Jidoka: Backend equivalence (GPU == SIMD == Scalar results)
//! - Kaizen: Top-K optimization (O(N log K) vs O(N log N))
//! - Genchi Genbutsu: Cost-based backend selection

use super::{AggregateFunction, OrderDirection, QueryPlan};
use crate::storage::StorageEngine;
use crate::topk::{SortOrder, TopKSelection};
use crate::{Backend, Error, Result};
use arrow::array::{
    Array, ArrayRef, Float32Array, Float64Array, Int32Array, Int64Array, RecordBatch,
};
use arrow::compute;
use arrow::datatypes::{DataType, Field, Schema};
use std::sync::Arc;

/// Query executor for parsed SQL queries
pub struct QueryExecutor {
    #[allow(dead_code)]
    backend: Backend,
}

impl Default for QueryExecutor {
    fn default() -> Self {
        Self::new()
    }
}

impl QueryExecutor {
    /// Create a new query executor with cost-based backend selection
    #[must_use]
    pub const fn new() -> Self {
        Self { backend: Backend::CostBased }
    }

    /// Create executor with forced backend
    #[must_use]
    pub const fn with_backend(backend: Backend) -> Self {
        Self { backend }
    }

    /// Execute a query plan against storage
    ///
    /// # Arguments
    /// * `plan` - Parsed query plan from `QueryEngine::parse()`
    /// * `storage` - Storage engine containing the data
    ///
    /// # Returns
    /// Result record batch with query results
    ///
    /// # Errors
    /// Returns error if:
    /// - Table not found in storage
    /// - Column not found in schema
    /// - Data type mismatch
    /// - Backend execution failure
    ///
    /// # Example
    /// ```rust,no_run
    /// use trueno_db::query::{QueryEngine, QueryExecutor};
    /// use trueno_db::storage::StorageEngine;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let storage = StorageEngine::load_parquet("data/events.parquet")?;
    /// let engine = QueryEngine::new();
    /// let executor = QueryExecutor::new();
    ///
    /// let plan = engine.parse("SELECT category, SUM(value) FROM events GROUP BY category")?;
    /// let result = executor.execute(&plan, &storage)?;
    ///
    /// println!("Results: {} rows", result.num_rows());
    /// # Ok(())
    /// # }
    /// ```
    pub fn execute(&self, plan: &QueryPlan, storage: &StorageEngine) -> Result<RecordBatch> {
        // Get all batches from storage
        let batches = storage.batches();
        if batches.is_empty() {
            return Err(Error::InvalidInput("No data in storage".to_string()));
        }

        // Combine batches (Phase 1: single table only)
        let combined = Self::combine_batches(batches)?;

        // Apply WHERE filter
        let filtered = if let Some(ref filter_expr) = plan.filter {
            Self::apply_filter(&combined, filter_expr)?
        } else {
            combined
        };

        // Execute aggregations if present
        let result = if plan.aggregations.is_empty() {
            // Project columns
            Self::project_columns(&filtered, &plan.columns)?
        } else {
            Self::execute_aggregations(&filtered, plan)?
        };

        // Apply ORDER BY + LIMIT (Top-K optimization)
        let result = if !plan.order_by.is_empty() {
            Self::apply_order_by_limit(&result, plan)?
        } else if let Some(limit) = plan.limit {
            // LIMIT without ORDER BY: just slice
            result.slice(0, limit.min(result.num_rows()))
        } else {
            result
        };

        Ok(result)
    }

    /// Combine multiple batches into single batch
    fn combine_batches(batches: &[RecordBatch]) -> Result<RecordBatch> {
        if batches.len() == 1 {
            return Ok(batches[0].clone());
        }

        // Use Arrow concat
        compute::concat_batches(&batches[0].schema(), batches)
            .map_err(|e| Error::StorageError(format!("Failed to combine batches: {e}")))
    }

    /// Apply WHERE filter
    fn apply_filter(batch: &RecordBatch, filter_expr: &str) -> Result<RecordBatch> {
        // Phase 1: Simple predicates only (column > value, column < value, etc.)
        // Parse filter expression: "column op value"
        let parts: Vec<&str> = filter_expr.split_whitespace().collect();
        if parts.len() < 3 {
            return Err(Error::ParseError(format!("Invalid filter expression: {filter_expr}")));
        }

        let column_name = parts[0];
        let op = parts[1];
        let value_str = parts.get(2..).unwrap_or(&[]).join(" ");

        // Find column index
        let schema = batch.schema();
        let column_index = schema
            .fields()
            .iter()
            .position(|f| f.name() == column_name)
            .ok_or_else(|| Error::InvalidInput(format!("Column not found: {column_name}")))?;

        let column = batch.column(column_index);

        // Build boolean mask based on data type
        let mask = match column.data_type() {
            DataType::Int32 => {
                let array = column
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .ok_or_else(|| Error::Other("Failed to downcast to Int32Array".to_string()))?;
                let value: i32 = value_str
                    .parse()
                    .map_err(|_| Error::ParseError(format!("Invalid Int32 value: {value_str}")))?;
                Self::build_comparison_mask_i32(array, op, value)?
            }
            DataType::Int64 => {
                let array = column
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .ok_or_else(|| Error::Other("Failed to downcast to Int64Array".to_string()))?;
                let value: i64 = value_str
                    .parse()
                    .map_err(|_| Error::ParseError(format!("Invalid Int64 value: {value_str}")))?;
                Self::build_comparison_mask_i64(array, op, value)?
            }
            DataType::Float32 => {
                let array = column.as_any().downcast_ref::<Float32Array>().ok_or_else(|| {
                    Error::Other("Failed to downcast to Float32Array".to_string())
                })?;
                let value: f32 = value_str.parse().map_err(|_| {
                    Error::ParseError(format!("Invalid Float32 value: {value_str}"))
                })?;
                Self::build_comparison_mask_f32(array, op, value)?
            }
            DataType::Float64 => {
                let array = column.as_any().downcast_ref::<Float64Array>().ok_or_else(|| {
                    Error::Other("Failed to downcast to Float64Array".to_string())
                })?;
                let value: f64 = value_str.parse().map_err(|_| {
                    Error::ParseError(format!("Invalid Float64 value: {value_str}"))
                })?;
                Self::build_comparison_mask_f64(array, op, value)?
            }
            dt => {
                return Err(Error::InvalidInput(format!(
                    "Filter not supported for data type: {dt:?}"
                )))
            }
        };

        // Apply filter using Arrow compute
        compute::filter_record_batch(batch, &mask)
            .map_err(|e| Error::StorageError(format!("Failed to apply filter: {e}")))
    }

    #[allow(clippy::unnecessary_wraps)]
    fn build_comparison_mask_i32(
        array: &Int32Array,
        op: &str,
        value: i32,
    ) -> Result<arrow::array::BooleanArray> {
        use arrow::array::BooleanArray;
        let values: Vec<bool> = (0..array.len())
            .map(|i| {
                if array.is_null(i) {
                    false
                } else {
                    let v = array.value(i);
                    match op {
                        ">" => v > value,
                        ">=" => v >= value,
                        "<" => v < value,
                        "<=" => v <= value,
                        "=" => v == value,
                        "!=" | "<>" => v != value,
                        _ => false,
                    }
                }
            })
            .collect();
        Ok(BooleanArray::from(values))
    }

    #[allow(clippy::unnecessary_wraps)]
    fn build_comparison_mask_i64(
        array: &Int64Array,
        op: &str,
        value: i64,
    ) -> Result<arrow::array::BooleanArray> {
        use arrow::array::BooleanArray;
        let values: Vec<bool> = (0..array.len())
            .map(|i| {
                if array.is_null(i) {
                    false
                } else {
                    let v = array.value(i);
                    match op {
                        ">" => v > value,
                        ">=" => v >= value,
                        "<" => v < value,
                        "<=" => v <= value,
                        "=" => v == value,
                        "!=" | "<>" => v != value,
                        _ => false,
                    }
                }
            })
            .collect();
        Ok(BooleanArray::from(values))
    }

    #[allow(clippy::unnecessary_wraps)]
    fn build_comparison_mask_f32(
        array: &Float32Array,
        op: &str,
        value: f32,
    ) -> Result<arrow::array::BooleanArray> {
        use arrow::array::BooleanArray;
        let values: Vec<bool> = (0..array.len())
            .map(|i| {
                if array.is_null(i) {
                    false
                } else {
                    let v = array.value(i);
                    match op {
                        ">" => v > value,
                        ">=" => v >= value,
                        "<" => v < value,
                        "<=" => v <= value,
                        "=" => (v - value).abs() < f32::EPSILON,
                        "!=" | "<>" => (v - value).abs() >= f32::EPSILON,
                        _ => false,
                    }
                }
            })
            .collect();
        Ok(BooleanArray::from(values))
    }

    #[allow(clippy::unnecessary_wraps)]
    fn build_comparison_mask_f64(
        array: &Float64Array,
        op: &str,
        value: f64,
    ) -> Result<arrow::array::BooleanArray> {
        use arrow::array::BooleanArray;
        let values: Vec<bool> = (0..array.len())
            .map(|i| {
                if array.is_null(i) {
                    false
                } else {
                    let v = array.value(i);
                    match op {
                        ">" => v > value,
                        ">=" => v >= value,
                        "<" => v < value,
                        "<=" => v <= value,
                        "=" => (v - value).abs() < f64::EPSILON,
                        "!=" | "<>" => (v - value).abs() >= f64::EPSILON,
                        _ => false,
                    }
                }
            })
            .collect();
        Ok(BooleanArray::from(values))
    }

    /// Project columns from batch
    fn project_columns(batch: &RecordBatch, columns: &[String]) -> Result<RecordBatch> {
        if columns.len() == 1 && columns[0] == "*" {
            return Ok(batch.clone());
        }

        let schema = batch.schema();
        let mut new_columns = Vec::new();
        let mut new_fields = Vec::new();

        for col_name in columns {
            let index = schema
                .fields()
                .iter()
                .position(|f| f.name() == col_name)
                .ok_or_else(|| Error::InvalidInput(format!("Column not found: {col_name}")))?;

            new_columns.push(batch.column(index).clone());
            new_fields.push(schema.field(index).clone());
        }

        let new_schema = Arc::new(Schema::new(new_fields));
        RecordBatch::try_new(new_schema, new_columns)
            .map_err(|e| Error::StorageError(format!("Failed to project columns: {e}")))
    }

    /// Execute aggregations
    fn execute_aggregations(batch: &RecordBatch, plan: &QueryPlan) -> Result<RecordBatch> {
        // Phase 1: Simple aggregations without GROUP BY
        if !plan.group_by.is_empty() {
            return Err(Error::InvalidInput(
                "GROUP BY aggregations not yet implemented in Phase 1".to_string(),
            ));
        }

        let mut result_columns: Vec<ArrayRef> = Vec::new();
        let mut result_fields: Vec<Field> = Vec::new();

        for (agg_func, col_name, alias) in &plan.aggregations {
            let result_name = alias.as_deref().unwrap_or(col_name);

            // Find column
            let schema = batch.schema();
            let col_index = schema
                .fields()
                .iter()
                .position(|f| f.name() == col_name || col_name == "*")
                .ok_or_else(|| Error::InvalidInput(format!("Column not found: {col_name}")))?;

            let column = batch.column(col_index);

            // Execute aggregation
            let (result_value, result_type) =
                Self::execute_single_aggregation(*agg_func, column, batch.num_rows())?;

            result_columns.push(result_value);
            result_fields.push(Field::new(result_name, result_type, false));
        }

        let result_schema = Arc::new(Schema::new(result_fields));
        RecordBatch::try_new(result_schema, result_columns)
            .map_err(|e| Error::StorageError(format!("Failed to create result batch: {e}")))
    }

    /// Execute single aggregation function
    fn execute_single_aggregation(
        func: AggregateFunction,
        column: &ArrayRef,
        num_rows: usize,
    ) -> Result<(ArrayRef, DataType)> {
        match column.data_type() {
            DataType::Int32 => {
                let array = column
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .ok_or_else(|| Error::Other("Failed to downcast to Int32Array".to_string()))?;
                Self::aggregate_i32(func, array, num_rows)
            }
            DataType::Int64 => {
                let array = column
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .ok_or_else(|| Error::Other("Failed to downcast to Int64Array".to_string()))?;
                Self::aggregate_i64(func, array, num_rows)
            }
            DataType::Float32 => {
                let array = column.as_any().downcast_ref::<Float32Array>().ok_or_else(|| {
                    Error::Other("Failed to downcast to Float32Array".to_string())
                })?;
                Self::aggregate_f32(func, array, num_rows)
            }
            DataType::Float64 => {
                let array = column.as_any().downcast_ref::<Float64Array>().ok_or_else(|| {
                    Error::Other("Failed to downcast to Float64Array".to_string())
                })?;
                Self::aggregate_f64(func, array, num_rows)
            }
            dt => {
                Err(Error::InvalidInput(format!("Aggregation not supported for data type: {dt:?}")))
            }
        }
    }

    #[allow(clippy::cast_precision_loss, clippy::cast_possible_wrap, clippy::unnecessary_wraps)]
    fn aggregate_i32(
        func: AggregateFunction,
        array: &Int32Array,
        num_rows: usize,
    ) -> Result<(ArrayRef, DataType)> {
        match func {
            AggregateFunction::Sum => {
                let sum: i64 = (0..array.len())
                    .filter(|&i| !array.is_null(i))
                    .map(|i| i64::from(array.value(i)))
                    .sum();
                Ok((Arc::new(Int64Array::from(vec![sum])), DataType::Int64))
            }
            AggregateFunction::Avg => {
                let sum: f64 = (0..array.len())
                    .filter(|&i| !array.is_null(i))
                    .map(|i| f64::from(array.value(i)))
                    .sum();
                let count = (0..array.len()).filter(|&i| !array.is_null(i)).count();
                let avg = if count > 0 { sum / count as f64 } else { 0.0 };
                Ok((Arc::new(Float64Array::from(vec![avg])), DataType::Float64))
            }
            AggregateFunction::Count => {
                Ok((Arc::new(Int64Array::from(vec![num_rows as i64])), DataType::Int64))
            }
            AggregateFunction::Min => {
                let min = (0..array.len())
                    .filter(|&i| !array.is_null(i))
                    .map(|i| array.value(i))
                    .min()
                    .unwrap_or(0);
                Ok((Arc::new(Int32Array::from(vec![min])), DataType::Int32))
            }
            AggregateFunction::Max => {
                let max = (0..array.len())
                    .filter(|&i| !array.is_null(i))
                    .map(|i| array.value(i))
                    .max()
                    .unwrap_or(0);
                Ok((Arc::new(Int32Array::from(vec![max])), DataType::Int32))
            }
        }
    }

    #[allow(clippy::cast_precision_loss, clippy::cast_possible_wrap, clippy::unnecessary_wraps)]
    fn aggregate_i64(
        func: AggregateFunction,
        array: &Int64Array,
        num_rows: usize,
    ) -> Result<(ArrayRef, DataType)> {
        match func {
            AggregateFunction::Sum => {
                let sum: i64 =
                    (0..array.len()).filter(|&i| !array.is_null(i)).map(|i| array.value(i)).sum();
                Ok((Arc::new(Int64Array::from(vec![sum])), DataType::Int64))
            }
            AggregateFunction::Avg => {
                let sum: f64 = (0..array.len())
                    .filter(|&i| !array.is_null(i))
                    .map(|i| array.value(i) as f64)
                    .sum();
                let count = (0..array.len()).filter(|&i| !array.is_null(i)).count();
                let avg = if count > 0 { sum / count as f64 } else { 0.0 };
                Ok((Arc::new(Float64Array::from(vec![avg])), DataType::Float64))
            }
            AggregateFunction::Count => {
                Ok((Arc::new(Int64Array::from(vec![num_rows as i64])), DataType::Int64))
            }
            AggregateFunction::Min => {
                let min = (0..array.len())
                    .filter(|&i| !array.is_null(i))
                    .map(|i| array.value(i))
                    .min()
                    .unwrap_or(0);
                Ok((Arc::new(Int64Array::from(vec![min])), DataType::Int64))
            }
            AggregateFunction::Max => {
                let max = (0..array.len())
                    .filter(|&i| !array.is_null(i))
                    .map(|i| array.value(i))
                    .max()
                    .unwrap_or(0);
                Ok((Arc::new(Int64Array::from(vec![max])), DataType::Int64))
            }
        }
    }

    #[allow(clippy::cast_precision_loss, clippy::cast_possible_wrap, clippy::unnecessary_wraps)]
    fn aggregate_f32(
        func: AggregateFunction,
        array: &Float32Array,
        num_rows: usize,
    ) -> Result<(ArrayRef, DataType)> {
        match func {
            AggregateFunction::Sum => {
                let sum: f32 =
                    (0..array.len()).filter(|&i| !array.is_null(i)).map(|i| array.value(i)).sum();
                Ok((Arc::new(Float32Array::from(vec![sum])), DataType::Float32))
            }
            AggregateFunction::Avg => {
                let sum: f64 = (0..array.len())
                    .filter(|&i| !array.is_null(i))
                    .map(|i| f64::from(array.value(i)))
                    .sum();
                let count = (0..array.len()).filter(|&i| !array.is_null(i)).count();
                let avg = if count > 0 { sum / count as f64 } else { 0.0 };
                Ok((Arc::new(Float64Array::from(vec![avg])), DataType::Float64))
            }
            AggregateFunction::Count => {
                Ok((Arc::new(Int64Array::from(vec![num_rows as i64])), DataType::Int64))
            }
            AggregateFunction::Min => {
                let min = (0..array.len())
                    .filter(|&i| !array.is_null(i))
                    .map(|i| array.value(i))
                    .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap_or(0.0);
                Ok((Arc::new(Float32Array::from(vec![min])), DataType::Float32))
            }
            AggregateFunction::Max => {
                let max = (0..array.len())
                    .filter(|&i| !array.is_null(i))
                    .map(|i| array.value(i))
                    .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap_or(0.0);
                Ok((Arc::new(Float32Array::from(vec![max])), DataType::Float32))
            }
        }
    }

    #[allow(clippy::cast_precision_loss, clippy::cast_possible_wrap, clippy::unnecessary_wraps)]
    fn aggregate_f64(
        func: AggregateFunction,
        array: &Float64Array,
        num_rows: usize,
    ) -> Result<(ArrayRef, DataType)> {
        match func {
            AggregateFunction::Sum => {
                let sum: f64 =
                    (0..array.len()).filter(|&i| !array.is_null(i)).map(|i| array.value(i)).sum();
                Ok((Arc::new(Float64Array::from(vec![sum])), DataType::Float64))
            }
            AggregateFunction::Avg => {
                let sum: f64 =
                    (0..array.len()).filter(|&i| !array.is_null(i)).map(|i| array.value(i)).sum();
                let count = (0..array.len()).filter(|&i| !array.is_null(i)).count();
                let avg = if count > 0 { sum / count as f64 } else { 0.0 };
                Ok((Arc::new(Float64Array::from(vec![avg])), DataType::Float64))
            }
            AggregateFunction::Count => {
                Ok((Arc::new(Int64Array::from(vec![num_rows as i64])), DataType::Int64))
            }
            AggregateFunction::Min => {
                let min = (0..array.len())
                    .filter(|&i| !array.is_null(i))
                    .map(|i| array.value(i))
                    .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap_or(0.0);
                Ok((Arc::new(Float64Array::from(vec![min])), DataType::Float64))
            }
            AggregateFunction::Max => {
                let max = (0..array.len())
                    .filter(|&i| !array.is_null(i))
                    .map(|i| array.value(i))
                    .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap_or(0.0);
                Ok((Arc::new(Float64Array::from(vec![max])), DataType::Float64))
            }
        }
    }

    /// Apply ORDER BY + LIMIT using Top-K optimization
    fn apply_order_by_limit(batch: &RecordBatch, plan: &QueryPlan) -> Result<RecordBatch> {
        if plan.order_by.is_empty() {
            return Ok(batch.clone());
        }

        // Phase 1: Single ORDER BY column only
        let (col_name, direction) = &plan.order_by[0];

        // Find column index
        let schema = batch.schema();
        let col_index = schema
            .fields()
            .iter()
            .position(|f| f.name() == col_name)
            .ok_or_else(|| Error::InvalidInput(format!("Column not found: {col_name}")))?;

        // Convert OrderDirection to SortOrder
        let sort_order = match direction {
            OrderDirection::Asc => SortOrder::Ascending,
            OrderDirection::Desc => SortOrder::Descending,
        };

        // Use Top-K if LIMIT is present, otherwise sort all
        let k = plan.limit.unwrap_or_else(|| batch.num_rows());
        batch.top_k(col_index, k, sort_order)
    }
}
