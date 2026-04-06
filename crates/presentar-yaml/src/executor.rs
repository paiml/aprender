//! Expression executor for data transformations.

use crate::expression::{AggregateOp, Expression, RankMethod, Transform};
use std::collections::HashMap;

/// A generic value that can hold any data type.
#[derive(Debug, Clone, PartialEq, Default)]
pub enum Value {
    /// Null value
    #[default]
    Null,
    /// Boolean value
    Bool(bool),
    /// Numeric value
    Number(f64),
    /// String value
    String(String),
    /// Array of values
    Array(Vec<Self>),
    /// Object (key-value map)
    Object(HashMap<String, Self>),
}

impl Value {
    /// Create a new null value.
    #[must_use]
    pub const fn null() -> Self {
        Self::Null
    }

    /// Create a new boolean value.
    #[must_use]
    pub const fn bool(v: bool) -> Self {
        Self::Bool(v)
    }

    /// Create a new number value.
    #[must_use]
    pub const fn number(v: f64) -> Self {
        Self::Number(v)
    }

    /// Create a new string value.
    #[must_use]
    pub fn string(v: impl Into<String>) -> Self {
        Self::String(v.into())
    }

    /// Create a new array value.
    #[must_use]
    pub const fn array(v: Vec<Self>) -> Self {
        Self::Array(v)
    }

    /// Create a new object value.
    #[must_use]
    pub const fn object(v: HashMap<String, Self>) -> Self {
        Self::Object(v)
    }

    /// Check if value is null.
    #[must_use]
    pub const fn is_null(&self) -> bool {
        matches!(self, Self::Null)
    }

    /// Check if value is a boolean.
    #[must_use]
    pub const fn is_bool(&self) -> bool {
        matches!(self, Self::Bool(_))
    }

    /// Check if value is a number.
    #[must_use]
    pub const fn is_number(&self) -> bool {
        matches!(self, Self::Number(_))
    }

    /// Check if value is a string.
    #[must_use]
    pub const fn is_string(&self) -> bool {
        matches!(self, Self::String(_))
    }

    /// Check if value is an array.
    #[must_use]
    pub const fn is_array(&self) -> bool {
        matches!(self, Self::Array(_))
    }

    /// Check if value is an object.
    #[must_use]
    pub const fn is_object(&self) -> bool {
        matches!(self, Self::Object(_))
    }

    /// Get as boolean.
    #[must_use]
    pub const fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(v) => Some(*v),
            _ => None,
        }
    }

    /// Get as number.
    #[must_use]
    pub const fn as_number(&self) -> Option<f64> {
        match self {
            Self::Number(v) => Some(*v),
            _ => None,
        }
    }

    /// Get as string.
    #[must_use]
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(v) => Some(v),
            _ => None,
        }
    }

    /// Get as array.
    #[must_use]
    pub const fn as_array(&self) -> Option<&Vec<Self>> {
        match self {
            Self::Array(v) => Some(v),
            _ => None,
        }
    }

    /// Get as mutable array.
    #[must_use]
    pub fn as_array_mut(&mut self) -> Option<&mut Vec<Self>> {
        match self {
            Self::Array(v) => Some(v),
            _ => None,
        }
    }

    /// Get as object.
    #[must_use]
    pub const fn as_object(&self) -> Option<&HashMap<String, Self>> {
        match self {
            Self::Object(v) => Some(v),
            _ => None,
        }
    }

    /// Get field from object.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&Self> {
        match self {
            Self::Object(map) => map.get(key),
            _ => None,
        }
    }

    /// Get as array or return `ExpectedArray` error (DRY helper for apply_ methods).
    pub fn require_array(&self) -> Result<&Vec<Self>, ExecutionError> {
        self.as_array().ok_or(ExecutionError::ExpectedArray)
    }

    /// Extract numeric values from array items by field name.
    pub fn extract_numbers(&self, field: &str) -> Result<Vec<f64>, ExecutionError> {
        Ok(self
            .require_array()?
            .iter()
            .filter_map(|item| item.get(field)?.as_number())
            .collect())
    }

    /// Get array length or object key count.
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::Array(arr) => arr.len(),
            Self::Object(obj) => obj.len(),
            Self::String(s) => s.len(),
            _ => 0,
        }
    }

    /// Check if array or object is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl From<bool> for Value {
    fn from(v: bool) -> Self {
        Self::Bool(v)
    }
}

impl From<f64> for Value {
    fn from(v: f64) -> Self {
        Self::Number(v)
    }
}

impl From<i32> for Value {
    fn from(v: i32) -> Self {
        Self::Number(f64::from(v))
    }
}

impl From<i64> for Value {
    fn from(v: i64) -> Self {
        Self::Number(v as f64)
    }
}

impl From<&str> for Value {
    fn from(v: &str) -> Self {
        Self::String(v.to_string())
    }
}

impl From<String> for Value {
    fn from(v: String) -> Self {
        Self::String(v)
    }
}

impl<T: Into<Self>> From<Vec<T>> for Value {
    fn from(v: Vec<T>) -> Self {
        Self::Array(v.into_iter().map(Into::into).collect())
    }
}

/// Execution error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExecutionError {
    /// Source not found in data context
    SourceNotFound(String),
    /// Expected an array for this transform
    ExpectedArray,
    /// Expected an object
    ExpectedObject,
    /// Field not found
    FieldNotFound(String),
    /// Type mismatch
    TypeMismatch(String),
    /// Invalid transform
    InvalidTransform(String),
}

impl std::fmt::Display for ExecutionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SourceNotFound(name) => write!(f, "source not found: {name}"),
            Self::ExpectedArray => write!(f, "expected an array"),
            Self::ExpectedObject => write!(f, "expected an object"),
            Self::FieldNotFound(name) => write!(f, "field not found: {name}"),
            Self::TypeMismatch(msg) => write!(f, "type mismatch: {msg}"),
            Self::InvalidTransform(msg) => write!(f, "invalid transform: {msg}"),
        }
    }
}

impl std::error::Error for ExecutionError {}

/// Data context for expression execution.
#[derive(Debug, Clone, Default)]
pub struct DataContext {
    /// Named data sources
    sources: HashMap<String, Value>,
}

impl DataContext {
    /// Create a new empty context.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a data source.
    pub fn insert(&mut self, name: impl Into<String>, value: Value) {
        self.sources.insert(name.into(), value);
    }

    /// Get a data source.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&Value> {
        // Support dotted paths like "data.transactions"
        let parts: Vec<&str> = name.split('.').collect();
        let mut current = self.sources.get(parts[0])?;

        for part in &parts[1..] {
            current = current.get(part)?;
        }

        Some(current)
    }

    /// Check if context has a source.
    #[must_use]
    pub fn contains(&self, name: &str) -> bool {
        self.get(name).is_some()
    }
}

/// Expression executor.
#[derive(Debug, Default)]
pub struct ExpressionExecutor;

impl ExpressionExecutor {
    /// Create a new executor.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Execute an expression against a data context.
    ///
    /// # Errors
    ///
    /// Returns an error if execution fails.
    pub fn execute(&self, expr: &Expression, ctx: &DataContext) -> Result<Value, ExecutionError> {
        // Resolve source
        let mut value = ctx
            .get(&expr.source)
            .cloned()
            .ok_or_else(|| ExecutionError::SourceNotFound(expr.source.clone()))?;

        // Apply transforms
        for transform in &expr.transforms {
            value = self.apply_transform(&value, transform, ctx)?;
        }

        Ok(value)
    }

    fn apply_transform(
        &self,
        value: &Value,
        transform: &Transform,
        ctx: &DataContext,
    ) -> Result<Value, ExecutionError> {
        match transform {
            Transform::Filter {
                field,
                value: match_value,
            } => self.apply_filter(value, field, match_value),
            Transform::Select { fields } => self.apply_select(value, fields),
            Transform::Sort { field, desc } => self.apply_sort(value, field, *desc),
            Transform::Count => Ok(self.apply_count(value)),
            Transform::Sum { field } => self.apply_sum(value, field),
            Transform::Mean { field } => self.apply_mean(value, field),
            Transform::Sample { n } => self.apply_sample(value, *n),
            Transform::Percentage => self.apply_percentage(value),
            Transform::Rate { window } => self.apply_rate(value, window),
            Transform::Join { other, on } => self.apply_join(value, other, on, ctx),
            Transform::GroupBy { field } => self.apply_group_by(value, field),
            Transform::Distinct { field } => self.apply_distinct(value, field.as_deref()),
            Transform::Where {
                field,
                op,
                value: match_value,
            } => self.apply_where(value, field, op, match_value),
            Transform::Offset { n } => self.apply_offset(value, *n),
            Transform::Min { field } => self.apply_min(value, field),
            Transform::Max { field } => self.apply_max(value, field),
            Transform::First { n } | Transform::Limit { n } => self.apply_limit(value, *n),
            Transform::Last { n } => self.apply_last(value, *n),
            Transform::Flatten => self.apply_flatten(value),
            Transform::Reverse => self.apply_reverse(value),
            // New transforms
            Transform::Map { expr } => self.apply_map(value, expr),
            Transform::Reduce { initial, expr } => self.apply_reduce(value, initial, expr),
            Transform::Aggregate { field, op } => self.apply_aggregate(value, field, *op),
            Transform::Pivot {
                row_field,
                col_field,
                value_field,
            } => self.apply_pivot(value, row_field, col_field, value_field),
            Transform::CumulativeSum { field } => self.apply_cumsum(value, field),
            Transform::Rank { field, method } => self.apply_rank(value, field, *method),
            Transform::MovingAverage { field, window } => {
                self.apply_moving_avg(value, field, *window)
            }
            Transform::PercentChange { field } => self.apply_pct_change(value, field),
            Transform::Suggest { prefix, count } => self.apply_suggest(value, prefix, *count),
        }
    }

    fn apply_filter(
        &self,
        value: &Value,
        field: &str,
        match_value: &str,
    ) -> Result<Value, ExecutionError> {
        let arr = value.require_array()?;

        let filtered: Vec<Value> = arr
            .iter()
            .filter(|item| {
                if let Some(obj) = item.as_object() {
                    if let Some(val) = obj.get(field) {
                        return self.value_matches(val, match_value);
                    }
                }
                false
            })
            .cloned()
            .collect();

        Ok(Value::Array(filtered))
    }

    fn value_matches(&self, value: &Value, target: &str) -> bool {
        match value {
            Value::String(s) => s == target,
            Value::Number(n) => {
                if let Ok(t) = target.parse::<f64>() {
                    (*n - t).abs() < f64::EPSILON
                } else {
                    false
                }
            }
            Value::Bool(b) => {
                matches!((b, target), (true, "true") | (false, "false"))
            }
            _ => false,
        }
    }

    fn apply_select(&self, value: &Value, fields: &[String]) -> Result<Value, ExecutionError> {
        let arr = value.require_array()?;

        let selected: Vec<Value> = arr
            .iter()
            .map(|item| {
                if let Some(obj) = item.as_object() {
                    let mut new_obj = HashMap::new();
                    for field in fields {
                        if let Some(val) = obj.get(field) {
                            new_obj.insert(field.clone(), val.clone());
                        }
                    }
                    Value::Object(new_obj)
                } else {
                    item.clone()
                }
            })
            .collect();

        Ok(Value::Array(selected))
    }

    fn apply_sort(&self, value: &Value, field: &str, desc: bool) -> Result<Value, ExecutionError> {
        let arr = value.require_array()?;
        let mut sorted = arr.clone();

        sorted.sort_by(|a, b| {
            let a_val = a.get(field);
            let b_val = b.get(field);

            let cmp = match (a_val, b_val) {
                (Some(Value::Number(a)), Some(Value::Number(b))) => {
                    a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                }
                (Some(Value::String(a)), Some(Value::String(b))) => a.cmp(b),
                _ => std::cmp::Ordering::Equal,
            };

            if desc {
                cmp.reverse()
            } else {
                cmp
            }
        });

        Ok(Value::Array(sorted))
    }

    fn apply_limit(&self, value: &Value, n: usize) -> Result<Value, ExecutionError> {
        let arr = value.require_array()?;
        Ok(Value::Array(arr.iter().take(n).cloned().collect()))
    }

    fn apply_count(&self, value: &Value) -> Value {
        match value {
            Value::Array(arr) => Value::Number(arr.len() as f64),
            Value::Object(obj) => Value::Number(obj.len() as f64),
            Value::String(s) => Value::Number(s.len() as f64),
            _ => Value::Number(0.0),
        }
    }

    fn apply_sum(&self, value: &Value, field: &str) -> Result<Value, ExecutionError> {
        let nums = value.extract_numbers(field)?;
        Ok(Value::Number(nums.iter().sum()))
    }

    fn apply_mean(&self, value: &Value, field: &str) -> Result<Value, ExecutionError> {
        let nums = value.extract_numbers(field)?;
        if nums.is_empty() {
            return Ok(Value::Number(0.0));
        }
        Ok(Value::Number(nums.iter().sum::<f64>() / nums.len() as f64))
    }

    fn apply_sample(&self, value: &Value, n: usize) -> Result<Value, ExecutionError> {
        let arr = value.require_array()?;

        // Simple deterministic "sampling" - just take first n elements
        // Real implementation would use random sampling
        Ok(Value::Array(arr.iter().take(n).cloned().collect()))
    }

    fn apply_percentage(&self, value: &Value) -> Result<Value, ExecutionError> {
        match value {
            Value::Number(n) => Ok(Value::Number(n * 100.0)),
            _ => Err(ExecutionError::TypeMismatch(
                "percentage requires a number".to_string(),
            )),
        }
    }

    fn apply_rate(&self, value: &Value, window: &str) -> Result<Value, ExecutionError> {
        let arr = value.require_array()?;

        // Parse window (e.g., "1m", "5m", "1h")
        let window_ms = self.parse_window(window)?;

        // For rate calculation, we need timestamped data
        // Look for a "timestamp" or "time" field
        let mut values_with_time: Vec<(f64, f64)> = arr
            .iter()
            .filter_map(|item| {
                let obj = item.as_object()?;
                let time = obj
                    .get("timestamp")
                    .or_else(|| obj.get("time"))
                    .and_then(Value::as_number)?;
                let val = obj
                    .get("value")
                    .or_else(|| obj.get("count"))
                    .and_then(Value::as_number)
                    .unwrap_or(1.0);
                Some((time, val))
            })
            .collect();

        if values_with_time.len() < 2 {
            return Ok(Value::Number(0.0));
        }

        // Sort by time
        values_with_time.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        // Calculate rate over the window
        let window_ms_f64 = window_ms as f64;
        let last_time = values_with_time.last().map_or(0.0, |v| v.0);
        let window_start = last_time - window_ms_f64;

        let sum_in_window: f64 = values_with_time
            .iter()
            .filter(|(t, _)| *t >= window_start)
            .map(|(_, v)| v)
            .sum();

        // Rate per second
        let rate = sum_in_window / (window_ms_f64 / 1000.0);

        Ok(Value::Number(rate))
    }

    fn parse_window(&self, window: &str) -> Result<u64, ExecutionError> {
        let window = window.trim();
        if window.is_empty() {
            return Err(ExecutionError::InvalidTransform("empty window".to_string()));
        }

        let (num_str, unit) = if let Some(s) = window.strip_suffix("ms") {
            (s, "ms")
        } else if let Some(s) = window.strip_suffix('s') {
            (s, "s")
        } else if let Some(s) = window.strip_suffix('m') {
            (s, "m")
        } else if let Some(s) = window.strip_suffix('h') {
            (s, "h")
        } else if let Some(s) = window.strip_suffix('d') {
            (s, "d")
        } else {
            // Assume milliseconds if no unit
            (window, "ms")
        };

        let num: u64 = num_str
            .parse()
            .map_err(|_| ExecutionError::InvalidTransform(format!("invalid window: {window}")))?;

        let ms = match unit {
            "s" => num * 1000,
            "m" => num * 60 * 1000,
            "h" => num * 60 * 60 * 1000,
            "d" => num * 24 * 60 * 60 * 1000,
            // "ms" and any other unit default to num (milliseconds)
            _ => num,
        };

        Ok(ms)
    }

    fn apply_join(
        &self,
        value: &Value,
        other: &str,
        on: &str,
        ctx: &DataContext,
    ) -> Result<Value, ExecutionError> {
        let left_arr = value.require_array()?;

        // Get the other dataset from context
        let right_value = ctx
            .get(other)
            .ok_or_else(|| ExecutionError::SourceNotFound(other.to_string()))?;
        let right_arr = right_value
            .as_array()
            .ok_or(ExecutionError::ExpectedArray)?;

        // Build a lookup map for the right side (keyed by the join field)
        let mut right_lookup: HashMap<String, Vec<&Value>> = HashMap::new();
        for item in right_arr {
            if let Some(obj) = item.as_object() {
                if let Some(key_val) = obj.get(on) {
                    let key = self.value_to_string(key_val);
                    right_lookup.entry(key).or_default().push(item);
                }
            }
        }

        // Perform the join (left join - keeps all left items)
        let mut result = Vec::new();
        for left_item in left_arr {
            if let Some(left_obj) = left_item.as_object() {
                if let Some(key_val) = left_obj.get(on) {
                    let key = self.value_to_string(key_val);
                    if let Some(right_items) = right_lookup.get(&key) {
                        // Join with each matching right item
                        for right_item in right_items {
                            if let Some(right_obj) = right_item.as_object() {
                                // Merge left and right objects
                                let mut merged = left_obj.clone();
                                for (k, v) in right_obj {
                                    // Don't overwrite left values, prefix with source name
                                    if merged.contains_key(k) && k != on {
                                        merged.insert(format!("{other}_{k}"), v.clone());
                                    } else if k != on {
                                        merged.insert(k.clone(), v.clone());
                                    }
                                }
                                result.push(Value::Object(merged));
                            }
                        }
                    } else {
                        // No match, keep left item as-is (left join behavior)
                        result.push(left_item.clone());
                    }
                } else {
                    // No join key, keep as-is
                    result.push(left_item.clone());
                }
            } else {
                // Not an object, keep as-is
                result.push(left_item.clone());
            }
        }

        Ok(Value::Array(result))
    }

    fn apply_group_by(&self, value: &Value, field: &str) -> Result<Value, ExecutionError> {
        let arr = value.require_array()?;

        let mut groups: HashMap<String, Vec<Value>> = HashMap::new();

        for item in arr {
            let key = if let Some(obj) = item.as_object() {
                if let Some(val) = obj.get(field) {
                    self.value_to_string(val)
                } else {
                    "_null".to_string()
                }
            } else {
                "_null".to_string()
            };

            groups.entry(key).or_default().push(item.clone());
        }

        // Convert to array of objects with key and items
        let result: Vec<Value> = groups
            .into_iter()
            .map(|(key, items)| {
                let mut obj = HashMap::new();
                obj.insert("key".to_string(), Value::String(key));
                obj.insert("items".to_string(), Value::Array(items.clone()));
                obj.insert("count".to_string(), Value::Number(items.len() as f64));
                Value::Object(obj)
            })
            .collect();

        Ok(Value::Array(result))
    }

    fn value_to_string(&self, value: &Value) -> String {
        match value {
            Value::Null => "_null".to_string(),
            Value::Bool(b) => b.to_string(),
            Value::Number(n) => n.to_string(),
            Value::String(s) => s.clone(),
            Value::Array(_) => "_array".to_string(),
            Value::Object(_) => "_object".to_string(),
        }
    }

    fn apply_distinct(&self, value: &Value, field: Option<&str>) -> Result<Value, ExecutionError> {
        let arr = value.require_array()?;

        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut result = Vec::new();

        for item in arr {
            let key = if let Some(f) = field {
                if let Some(obj) = item.as_object() {
                    obj.get(f)
                        .map(|v| self.value_to_string(v))
                        .unwrap_or_default()
                } else {
                    self.value_to_string(item)
                }
            } else {
                self.value_to_string(item)
            };

            if seen.insert(key) {
                result.push(item.clone());
            }
        }

        Ok(Value::Array(result))
    }

    fn apply_where(
        &self,
        value: &Value,
        field: &str,
        op: &str,
        match_value: &str,
    ) -> Result<Value, ExecutionError> {
        let arr = value.require_array()?;

        let filtered: Vec<Value> = arr
            .iter()
            .filter(|item| {
                if let Some(obj) = item.as_object() {
                    if let Some(val) = obj.get(field) {
                        return self.compare_values(val, op, match_value);
                    }
                }
                false
            })
            .cloned()
            .collect();

        Ok(Value::Array(filtered))
    }

    fn compare_values(&self, value: &Value, op: &str, target: &str) -> bool {
        match op {
            "eq" | "==" | "=" => self.value_matches(value, target),
            "ne" | "!=" | "<>" => !self.value_matches(value, target),
            "gt" | ">" => {
                if let (Some(v), Ok(t)) = (value.as_number(), target.parse::<f64>()) {
                    v > t
                } else {
                    false
                }
            }
            "lt" | "<" => {
                if let (Some(v), Ok(t)) = (value.as_number(), target.parse::<f64>()) {
                    v < t
                } else {
                    false
                }
            }
            "gte" | ">=" => {
                if let (Some(v), Ok(t)) = (value.as_number(), target.parse::<f64>()) {
                    v >= t
                } else {
                    false
                }
            }
            "lte" | "<=" => {
                if let (Some(v), Ok(t)) = (value.as_number(), target.parse::<f64>()) {
                    v <= t
                } else {
                    false
                }
            }
            "contains" => {
                if let Some(s) = value.as_str() {
                    s.contains(target)
                } else {
                    false
                }
            }
            "starts_with" => {
                if let Some(s) = value.as_str() {
                    s.starts_with(target)
                } else {
                    false
                }
            }
            "ends_with" => {
                if let Some(s) = value.as_str() {
                    s.ends_with(target)
                } else {
                    false
                }
            }
            _ => false,
        }
    }

    fn apply_offset(&self, value: &Value, n: usize) -> Result<Value, ExecutionError> {
        let arr = value.require_array()?;
        Ok(Value::Array(arr.iter().skip(n).cloned().collect()))
    }

    fn apply_min(&self, value: &Value, field: &str) -> Result<Value, ExecutionError> {
        let nums = value.extract_numbers(field)?;
        let min = nums.iter().copied().fold(f64::INFINITY, f64::min);
        Ok(if min.is_infinite() {
            Value::Null
        } else {
            Value::Number(min)
        })
    }

    fn apply_max(&self, value: &Value, field: &str) -> Result<Value, ExecutionError> {
        let nums = value.extract_numbers(field)?;
        let max = nums.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        Ok(if max.is_infinite() {
            Value::Null
        } else {
            Value::Number(max)
        })
    }

    fn apply_last(&self, value: &Value, n: usize) -> Result<Value, ExecutionError> {
        let arr = value.require_array()?;
        let len = arr.len();
        let skip = len.saturating_sub(n);
        Ok(Value::Array(arr.iter().skip(skip).cloned().collect()))
    }

    fn apply_flatten(&self, value: &Value) -> Result<Value, ExecutionError> {
        let arr = value.require_array()?;

        let mut result = Vec::new();
        for item in arr {
            if let Some(inner) = item.as_array() {
                result.extend(inner.iter().cloned());
            } else {
                result.push(item.clone());
            }
        }

        Ok(Value::Array(result))
    }

    fn apply_reverse(&self, value: &Value) -> Result<Value, ExecutionError> {
        let arr = value.require_array()?;
        let mut reversed = arr.clone();
        reversed.reverse();
        Ok(Value::Array(reversed))
    }

    // =========================================================================
    // New Transform Implementations
    // =========================================================================

    fn apply_map(&self, value: &Value, expr: &str) -> Result<Value, ExecutionError> {
        let arr = value.require_array()?;

        // Simple expression evaluation: extract field if expr is "item.field"
        // For complex expressions, this would need a proper expression evaluator
        let mapped: Vec<Value> = arr
            .iter()
            .map(|item| {
                // Handle simple field access like "item.field"
                if let Some(field) = expr.strip_prefix("item.") {
                    if let Some(obj) = item.as_object() {
                        obj.get(field).cloned().unwrap_or(Value::Null)
                    } else {
                        item.clone()
                    }
                } else {
                    // Return item unchanged if we can't parse the expression
                    item.clone()
                }
            })
            .collect();

        Ok(Value::Array(mapped))
    }

    fn apply_reduce(
        &self,
        value: &Value,
        initial: &str,
        _expr: &str,
    ) -> Result<Value, ExecutionError> {
        let arr = value.require_array()?;

        // Parse initial value
        let mut acc: f64 = initial.parse().unwrap_or(0.0);

        // Simple sum reduction (a proper implementation would evaluate the expr)
        for item in arr {
            if let Some(n) = item.as_number() {
                acc += n;
            }
        }

        Ok(Value::Number(acc))
    }

    fn apply_aggregate(
        &self,
        value: &Value,
        field: &str,
        op: AggregateOp,
    ) -> Result<Value, ExecutionError> {
        let arr = value.require_array()?;

        // For grouped data, expect array of {key: ..., values: [...]}
        // For ungrouped data, operate on the field directly

        let values: Vec<f64> = arr
            .iter()
            .filter_map(|item| {
                if let Some(obj) = item.as_object() {
                    // If this is a group with "values" key
                    if let Some(Value::Array(group_values)) = obj.get("values") {
                        return Some(
                            group_values
                                .iter()
                                .filter_map(|v| v.get(field)?.as_number())
                                .collect::<Vec<_>>(),
                        );
                    }
                    // Direct field access
                    obj.get(field)?.as_number().map(|n| vec![n])
                } else {
                    None
                }
            })
            .flatten()
            .collect();

        let result = match op {
            AggregateOp::Sum => values.iter().sum(),
            AggregateOp::Count => values.len() as f64,
            AggregateOp::Mean => {
                if values.is_empty() {
                    0.0
                } else {
                    values.iter().sum::<f64>() / values.len() as f64
                }
            }
            AggregateOp::Min => values.iter().cloned().fold(f64::INFINITY, f64::min),
            AggregateOp::Max => values.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            AggregateOp::First => values.first().copied().unwrap_or(0.0),
            AggregateOp::Last => values.last().copied().unwrap_or(0.0),
        };

        Ok(Value::Number(result))
    }

    fn apply_pivot(
        &self,
        value: &Value,
        row_field: &str,
        col_field: &str,
        value_field: &str,
    ) -> Result<Value, ExecutionError> {
        let arr = value.require_array()?;

        // Build pivot table
        let mut rows: HashMap<String, HashMap<String, f64>> = HashMap::new();

        for item in arr {
            if let Some(obj) = item.as_object() {
                let row_key = obj
                    .get(row_field)
                    .map(|v| self.value_to_string(v))
                    .unwrap_or_default();
                let col_key = obj
                    .get(col_field)
                    .map(|v| self.value_to_string(v))
                    .unwrap_or_default();
                let val = obj
                    .get(value_field)
                    .and_then(|v| v.as_number())
                    .unwrap_or(0.0);

                rows.entry(row_key)
                    .or_default()
                    .entry(col_key)
                    .and_modify(|v| *v += val)
                    .or_insert(val);
            }
        }

        // Convert to array of objects
        let result: Vec<Value> = rows
            .into_iter()
            .map(|(row_key, cols)| {
                let mut obj = HashMap::new();
                obj.insert(row_field.to_string(), Value::String(row_key));
                for (col_key, val) in cols {
                    obj.insert(col_key, Value::Number(val));
                }
                Value::Object(obj)
            })
            .collect();

        Ok(Value::Array(result))
    }

    fn apply_cumsum(&self, value: &Value, field: &str) -> Result<Value, ExecutionError> {
        let arr = value.require_array()?;

        let mut running_sum = 0.0;
        let result: Vec<Value> = arr
            .iter()
            .map(|item| {
                if let Some(obj) = item.as_object() {
                    let val = obj.get(field).and_then(|v| v.as_number()).unwrap_or(0.0);
                    running_sum += val;

                    let mut new_obj = obj.clone();
                    new_obj.insert(format!("{field}_cumsum"), Value::Number(running_sum));
                    Value::Object(new_obj)
                } else {
                    item.clone()
                }
            })
            .collect();

        Ok(Value::Array(result))
    }

    fn apply_rank(
        &self,
        value: &Value,
        field: &str,
        method: RankMethod,
    ) -> Result<Value, ExecutionError> {
        let arr = value.require_array()?;

        // Extract values with indices
        let mut indexed: Vec<(usize, f64)> = arr
            .iter()
            .enumerate()
            .filter_map(|(i, item)| item.as_object()?.get(field)?.as_number().map(|n| (i, n)))
            .collect();

        // Sort by value (descending for ranking)
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Assign ranks based on method
        let mut ranks = vec![0.0; arr.len()];
        match method {
            RankMethod::Dense => {
                let mut rank = 0;
                let mut prev_val: Option<f64> = None;
                for (i, val) in indexed {
                    if prev_val != Some(val) {
                        rank += 1;
                    }
                    ranks[i] = rank as f64;
                    prev_val = Some(val);
                }
            }
            RankMethod::Ordinal => {
                for (rank, (i, _)) in indexed.iter().enumerate() {
                    ranks[*i] = (rank + 1) as f64;
                }
            }
            RankMethod::Average => {
                let mut i = 0;
                while i < indexed.len() {
                    let val = indexed[i].1;
                    let start = i;
                    while i < indexed.len() && (indexed[i].1 - val).abs() < f64::EPSILON {
                        i += 1;
                    }
                    let avg_rank =
                        (start + 1..=i).map(|r| r as f64).sum::<f64>() / (i - start) as f64;
                    for j in start..i {
                        ranks[indexed[j].0] = avg_rank;
                    }
                }
            }
        }

        // Add rank to each object
        let result: Vec<Value> = arr
            .iter()
            .enumerate()
            .map(|(i, item)| {
                if let Some(obj) = item.as_object() {
                    let mut new_obj = obj.clone();
                    new_obj.insert(format!("{field}_rank"), Value::Number(ranks[i]));
                    Value::Object(new_obj)
                } else {
                    item.clone()
                }
            })
            .collect();

        Ok(Value::Array(result))
    }

    fn apply_moving_avg(
        &self,
        value: &Value,
        field: &str,
        window: usize,
    ) -> Result<Value, ExecutionError> {
        let arr = value.require_array()?;

        let values: Vec<f64> = arr
            .iter()
            .filter_map(|item| item.as_object()?.get(field)?.as_number())
            .collect();

        let result: Vec<Value> = arr
            .iter()
            .enumerate()
            .map(|(i, item)| {
                if let Some(obj) = item.as_object() {
                    let start = i.saturating_sub(window - 1);
                    let window_values = &values[start..=i.min(values.len() - 1)];
                    let ma = if window_values.is_empty() {
                        0.0
                    } else {
                        window_values.iter().sum::<f64>() / window_values.len() as f64
                    };

                    let mut new_obj = obj.clone();
                    new_obj.insert(format!("{field}_ma{window}"), Value::Number(ma));
                    Value::Object(new_obj)
                } else {
                    item.clone()
                }
            })
            .collect();

        Ok(Value::Array(result))
    }

    fn apply_pct_change(&self, value: &Value, field: &str) -> Result<Value, ExecutionError> {
        let arr = value.require_array()?;

        let values: Vec<f64> = arr
            .iter()
            .filter_map(|item| item.as_object()?.get(field)?.as_number())
            .collect();

        let result: Vec<Value> = arr
            .iter()
            .enumerate()
            .map(|(i, item)| {
                if let Some(obj) = item.as_object() {
                    let pct = if i == 0 || values.get(i - 1).map_or(true, |&prev| prev == 0.0) {
                        0.0
                    } else {
                        let prev = values[i - 1];
                        let curr = values.get(i).copied().unwrap_or(prev);
                        (curr - prev) / prev * 100.0
                    };

                    let mut new_obj = obj.clone();
                    new_obj.insert(format!("{field}_pct_change"), Value::Number(pct));
                    Value::Object(new_obj)
                } else {
                    item.clone()
                }
            })
            .collect();

        Ok(Value::Array(result))
    }

    /// Apply suggestion transform for N-gram/autocomplete models.
    ///
    /// The input `value` should be a model object with a `model_type` field.
    /// This is a stub implementation - actual model inference is handled by
    /// the runtime layer which injects results into the context.
    ///
    /// In production, the runtime loads the .apr model and provides suggestions
    /// through a callback or pre-computed context value.
    #[allow(clippy::unnecessary_wraps)] // Returns Result for API consistency with other transforms
    fn apply_suggest(
        &self,
        value: &Value,
        prefix: &str,
        count: usize,
    ) -> Result<Value, ExecutionError> {
        // Check if value is a model object with pre-computed suggestions
        if let Some(obj) = value.as_object() {
            // If the model has pre-computed suggestions for this prefix, use them
            if let Some(suggestions) = obj.get("_suggestions") {
                if let Some(arr) = suggestions.as_array() {
                    return Ok(Value::Array(arr.iter().take(count).cloned().collect()));
                }
            }

            // If this is a model reference, return placeholder suggestions
            // The runtime layer should populate _suggestions before execution
            if obj.contains_key("model_type") || obj.contains_key("source") {
                // Return empty array - runtime should inject actual suggestions
                return Ok(Value::Array(vec![]));
            }
        }

        // For testing/demo: if value is an array of suggestion objects, filter by prefix
        if let Some(arr) = value.as_array() {
            let filtered: Vec<Value> = arr
                .iter()
                .filter(|item| {
                    if let Some(obj) = item.as_object() {
                        if let Some(text) = obj.get("text").and_then(|v| v.as_str()) {
                            return text.starts_with(prefix);
                        }
                    }
                    false
                })
                .take(count)
                .cloned()
                .collect();
            return Ok(Value::Array(filtered));
        }

        // Fallback: return empty suggestions
        Ok(Value::Array(vec![]))
    }
}
