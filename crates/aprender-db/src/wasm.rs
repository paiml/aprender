//! WebAssembly bindings for trueno-db browser deployment.
//!
//! Provides JavaScript-accessible API with tiered compute:
//! - Tier 1: WebGPU compute shaders (if available)
//! - Tier 2: WASM SIMD128 via trueno backend
//! - Tier 3: Scalar fallback
//!
//! # Architecture
//!
//! ```text
//! Browser JS → wasm-bindgen → Database → trueno (SIMD128) → Results
//!                              ↓
//!                         QueryEngine
//!                              ↓
//!                         Arrow Tables
//! ```

#![cfg(target_arch = "wasm32")]

use arrow::array::{Array, Float64Array, Int32Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use js_sys::{Object, Reflect};
use std::collections::HashMap;
use std::sync::Arc;
use wasm_bindgen::prelude::*;
use web_sys::{console, window};

use wasm_bindgen_futures::JsFuture;

use crate::query::{QueryEngine, QueryExecutor};
use crate::storage::StorageEngine;

pub mod http_range;
pub mod late_materialization;
pub mod streaming_parquet;

/// Initialize WASM module with panic hook for better error messages
#[wasm_bindgen(start)]
pub fn init() {
    console_error_panic_hook::set_once();
    console::log_1(&"trueno-db WASM initialized".into());
}

/// Get library version
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Database configuration for browser deployment
#[wasm_bindgen]
#[derive(Clone, Debug)]
pub struct DatabaseConfig {
    backend: String,
    cache_size_mb: usize,
}

#[wasm_bindgen]
impl DatabaseConfig {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self { backend: "auto".to_string(), cache_size_mb: 256 }
    }

    /// Set compute backend: "auto", "webgpu", "simd128", "scalar"
    #[wasm_bindgen]
    pub fn backend(mut self, backend: &str) -> Self {
        self.backend = backend.to_string();
        self
    }

    #[wasm_bindgen]
    pub fn cache_size_mb(mut self, size: usize) -> Self {
        self.cache_size_mb = size;
        self
    }
}

impl Default for DatabaseConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// In-browser analytics database with GPU/SIMD acceleration
#[wasm_bindgen]
pub struct Database {
    #[allow(dead_code)]
    config: DatabaseConfig,
    tables: HashMap<String, StorageEngine>,
    query_engine: QueryEngine,
    executor: QueryExecutor,
}

#[wasm_bindgen]
impl Database {
    /// Create new database instance
    #[wasm_bindgen(constructor)]
    pub fn new(config: Option<DatabaseConfig>) -> Self {
        let config = config.unwrap_or_default();
        console::log_1(&format!("Database created with backend: {}", config.backend).into());

        Self {
            config,
            tables: HashMap::new(),
            query_engine: QueryEngine::new(),
            executor: QueryExecutor::new(),
        }
    }

    /// Load table from URL (supports HTTP range requests for streaming)
    #[wasm_bindgen]
    pub async fn load_table(&mut self, name: String, url: String) -> Result<(), JsValue> {
        console::log_1(&format!("Loading table '{}' from {}", name, url).into());

        // HTTP range request Parquet reader — not yet implemented
        Err(JsValue::from_str("Not yet implemented"))
    }

    /// Load table from JSON array string (for embedded demo data)
    #[wasm_bindgen]
    pub fn load_json(&mut self, name: String, json: String) -> Result<(), JsValue> {
        console::log_1(
            &format!("Loading table '{}' from JSON ({} bytes)", name, json.len()).into(),
        );

        // Parse JSON array
        let records: Vec<serde_json::Value> = serde_json::from_str(&json)
            .map_err(|e| JsValue::from_str(&format!("JSON parse error: {e}")))?;

        if records.is_empty() {
            return Err(JsValue::from_str("Empty JSON array"));
        }

        // Infer schema from first record
        let first = records
            .first()
            .and_then(|v| v.as_object())
            .ok_or_else(|| JsValue::from_str("Expected array of objects"))?;

        let mut fields: Vec<Field> = Vec::new();
        let mut column_names: Vec<String> = Vec::new();

        for (key, value) in first {
            let data_type = match value {
                serde_json::Value::Number(n) => {
                    if n.is_f64() {
                        DataType::Float64
                    } else {
                        DataType::Int32
                    }
                }
                serde_json::Value::String(_) => DataType::Utf8,
                serde_json::Value::Bool(_) => DataType::Boolean,
                _ => DataType::Utf8, // Default to string
            };
            fields.push(Field::new(key, data_type, true));
            column_names.push(key.clone());
        }

        let schema = Arc::new(Schema::new(fields.clone()));

        // Build columnar arrays
        let mut columns: Vec<Arc<dyn Array>> = Vec::new();

        for (i, field) in fields.iter().enumerate() {
            let col_name = &column_names[i];

            match field.data_type() {
                DataType::Int32 => {
                    let values: Vec<Option<i32>> = records
                        .iter()
                        .map(|r| r.get(col_name).and_then(|v| v.as_i64()).map(|n| n as i32))
                        .collect();
                    columns.push(Arc::new(Int32Array::from(values)));
                }
                DataType::Float64 => {
                    let values: Vec<Option<f64>> =
                        records.iter().map(|r| r.get(col_name).and_then(|v| v.as_f64())).collect();
                    columns.push(Arc::new(Float64Array::from(values)));
                }
                DataType::Utf8 => {
                    let values: Vec<Option<String>> = records
                        .iter()
                        .map(|r| r.get(col_name).and_then(|v| v.as_str()).map(String::from))
                        .collect();
                    columns.push(Arc::new(StringArray::from(values)));
                }
                _ => {
                    // Default: convert to string
                    let values: Vec<Option<String>> =
                        records.iter().map(|r| r.get(col_name).map(|v| v.to_string())).collect();
                    columns.push(Arc::new(StringArray::from(values)));
                }
            }
        }

        let batch = RecordBatch::try_new(schema, columns)
            .map_err(|e| JsValue::from_str(&format!("Failed to create batch: {e}")))?;

        let storage = StorageEngine::new(vec![batch]);
        self.tables.insert(name.clone(), storage);

        console::log_1(
            &format!("Table '{}' loaded: {} rows, {} columns", name, records.len(), fields.len())
                .into(),
        );
        Ok(())
    }

    /// Execute SQL query and return JSON result
    #[wasm_bindgen]
    pub fn query(&self, sql: String) -> Result<String, JsValue> {
        console::log_1(&format!("Executing query: {}", sql).into());

        // Parse SQL
        let plan = self
            .query_engine
            .parse(&sql)
            .map_err(|e| JsValue::from_str(&format!("Parse error: {e}")))?;

        // Get table
        let storage = self
            .tables
            .get(&plan.table)
            .ok_or_else(|| JsValue::from_str(&format!("Table not found: {}", plan.table)))?;

        // Execute query
        let result = self
            .executor
            .execute(&plan, storage)
            .map_err(|e| JsValue::from_str(&format!("Execution error: {e}")))?;

        // Convert to JSON
        let json = record_batch_to_json(&result)
            .map_err(|e| JsValue::from_str(&format!("JSON conversion error: {e}")))?;

        console::log_1(&format!("Query returned {} rows", result.num_rows()).into());
        Ok(json)
    }

    /// Get query execution plan (for debugging)
    #[wasm_bindgen]
    pub fn explain(&self, sql: String) -> Result<String, JsValue> {
        console::log_1(&format!("EXPLAIN: {}", sql).into());

        let plan = self
            .query_engine
            .parse(&sql)
            .map_err(|e| JsValue::from_str(&format!("Parse error: {e}")))?;

        Ok(format!("{plan:#?}"))
    }
}

/// Convert Arrow RecordBatch to JSON string
fn record_batch_to_json(batch: &RecordBatch) -> Result<String, String> {
    let schema = batch.schema();
    let mut rows: Vec<serde_json::Value> = Vec::new();

    for row_idx in 0..batch.num_rows() {
        let mut row = serde_json::Map::new();

        for (col_idx, field) in schema.fields().iter().enumerate() {
            let col = batch.column(col_idx);
            let value = array_value_to_json(col.as_ref(), row_idx)?;
            row.insert(field.name().clone(), value);
        }

        rows.push(serde_json::Value::Object(row));
    }

    serde_json::to_string_pretty(&rows).map_err(|e| e.to_string())
}

/// Extract single value from Arrow array as JSON
fn array_value_to_json(array: &dyn Array, idx: usize) -> Result<serde_json::Value, String> {
    if array.is_null(idx) {
        return Ok(serde_json::Value::Null);
    }

    match array.data_type() {
        DataType::Int32 => {
            let arr = array.as_any().downcast_ref::<Int32Array>().ok_or("downcast failed")?;
            Ok(serde_json::Value::Number(arr.value(idx).into()))
        }
        DataType::Int64 => {
            let arr = array
                .as_any()
                .downcast_ref::<arrow::array::Int64Array>()
                .ok_or("downcast failed")?;
            Ok(serde_json::Value::Number(arr.value(idx).into()))
        }
        DataType::Float32 => {
            let arr = array
                .as_any()
                .downcast_ref::<arrow::array::Float32Array>()
                .ok_or("downcast failed")?;
            let v = arr.value(idx);
            Ok(serde_json::json!(v))
        }
        DataType::Float64 => {
            let arr = array.as_any().downcast_ref::<Float64Array>().ok_or("downcast failed")?;
            let v = arr.value(idx);
            Ok(serde_json::json!(v))
        }
        DataType::Utf8 => {
            let arr = array.as_any().downcast_ref::<StringArray>().ok_or("downcast failed")?;
            Ok(serde_json::Value::String(arr.value(idx).to_string()))
        }
        _ => Ok(serde_json::Value::String(format!("<unsupported type: {:?}>", array.data_type()))),
    }
}

/// Detect available compute capabilities
#[wasm_bindgen]
pub async fn detect_capabilities() -> Result<JsValue, JsValue> {
    let window = window().ok_or_else(|| JsValue::from_str("No window"))?;
    let navigator = window.navigator();

    let caps = Object::new();

    // Check WebGPU
    let webgpu_available = check_webgpu(&navigator).await;
    Reflect::set(&caps, &"webgpu".into(), &JsValue::from(webgpu_available))?;

    // Check SIMD128 (compile-time feature)
    #[cfg(target_feature = "simd128")]
    let simd128 = true;
    #[cfg(not(target_feature = "simd128"))]
    let simd128 = false;
    Reflect::set(&caps, &"simd128".into(), &JsValue::from(simd128))?;

    // Determine best tier
    let tier = if webgpu_available {
        "webgpu"
    } else if simd128 {
        "simd128"
    } else {
        "scalar"
    };
    Reflect::set(&caps, &"tier".into(), &JsValue::from(tier))?;

    Ok(caps.into())
}

async fn check_webgpu(navigator: &web_sys::Navigator) -> bool {
    // WebGPU detection via navigator.gpu
    let gpu = navigator.gpu();
    let adapter_promise = gpu.request_adapter();
    if let Ok(adapter_val) = JsFuture::from(adapter_promise).await {
        return !adapter_val.is_null() && !adapter_val.is_undefined();
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_database_config_builder() {
        let config = DatabaseConfig::new().backend("simd128").cache_size_mb(512);

        assert_eq!(config.backend, "simd128");
        assert_eq!(config.cache_size_mb, 512);
    }
}
