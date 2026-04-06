//! Query operations for SQLite Backend.
//!
//! Contains search, list, and parameter filtering methods.

use super::backend::SqliteBackend;
use super::types::{Experiment, FilterOp, ParamFilter, ParameterValue, Run};
use crate::storage::{Result, RunStatus, StorageError};
use chrono::{DateTime, Utc};
use rusqlite::params;
use std::collections::HashMap;

/// Parse a RunStatus from its stored string
fn str_to_status(s: &str) -> RunStatus {
    match s {
        "pending" => RunStatus::Pending,
        "running" => RunStatus::Running,
        "completed" => RunStatus::Success,
        "failed" => RunStatus::Failed,
        "cancelled" => RunStatus::Cancelled,
        _ => RunStatus::Failed,
    }
}

/// Parse an RFC3339 timestamp string, falling back to now
fn parse_timestamp(s: &str) -> DateTime<Utc> {
    s.parse().unwrap_or_else(|_| Utc::now())
}

impl SqliteBackend {
    /// Log a parameter for a run
    pub fn log_param(&self, run_id: &str, key: &str, value: ParameterValue) -> Result<()> {
        let conn = self.lock_conn()?;

        // Verify run exists
        let exists: bool = conn
            .query_row("SELECT EXISTS(SELECT 1 FROM runs WHERE id = ?1)", [run_id], |row| {
                row.get(0)
            })
            .map_err(|e| StorageError::Backend(format!("Failed to check run: {e}")))?;

        if !exists {
            return Err(StorageError::RunNotFound(run_id.to_string()));
        }

        let value_json = value.to_json();
        let type_name = value.type_name();

        conn.execute(
            "INSERT OR REPLACE INTO params (run_id, key, value, type) VALUES (?1, ?2, ?3, ?4)",
            params![run_id, key, value_json, type_name],
        )
        .map_err(|e| StorageError::Backend(format!("Failed to log param: {e}")))?;

        Ok(())
    }

    /// Log multiple parameters for a run
    pub fn log_params(
        &self,
        run_id: &str,
        params_map: HashMap<String, ParameterValue>,
    ) -> Result<()> {
        let conn = self.lock_conn()?;

        // Verify run exists
        let exists: bool = conn
            .query_row("SELECT EXISTS(SELECT 1 FROM runs WHERE id = ?1)", [run_id], |row| {
                row.get(0)
            })
            .map_err(|e| StorageError::Backend(format!("Failed to check run: {e}")))?;

        if !exists {
            return Err(StorageError::RunNotFound(run_id.to_string()));
        }

        for (key, value) in &params_map {
            let value_json = value.to_json();
            let type_name = value.type_name();

            conn.execute(
                "INSERT OR REPLACE INTO params (run_id, key, value, type) VALUES (?1, ?2, ?3, ?4)",
                rusqlite::params![run_id, key, value_json, type_name],
            )
            .map_err(|e| StorageError::Backend(format!("Failed to log param: {e}")))?;
        }

        Ok(())
    }

    /// Get parameters for a run
    pub fn get_params(&self, run_id: &str) -> Result<HashMap<String, ParameterValue>> {
        let conn = self.lock_conn()?;

        // Verify run exists
        let exists: bool = conn
            .query_row("SELECT EXISTS(SELECT 1 FROM runs WHERE id = ?1)", [run_id], |row| {
                row.get(0)
            })
            .map_err(|e| StorageError::Backend(format!("Failed to check run: {e}")))?;

        if !exists {
            return Err(StorageError::RunNotFound(run_id.to_string()));
        }

        let mut stmt = conn
            .prepare("SELECT key, value FROM params WHERE run_id = ?1")
            .map_err(|e| StorageError::Backend(format!("Failed to prepare params query: {e}")))?;

        let rows = stmt
            .query_map([run_id], |row| {
                let key: String = row.get(0)?;
                let value_json: String = row.get(1)?;
                Ok((key, value_json))
            })
            .map_err(|e| StorageError::Backend(format!("Failed to query params: {e}")))?;

        let mut result = HashMap::new();
        for row in rows {
            let (key, value_json) =
                row.map_err(|e| StorageError::Backend(format!("Failed to read param row: {e}")))?;
            if let Some(value) = ParameterValue::from_json(&value_json) {
                result.insert(key, value);
            }
        }

        Ok(result)
    }

    /// Search runs by parameter filters
    pub fn search_runs_by_params(&self, filters: &[ParamFilter]) -> Result<Vec<Run>> {
        if filters.is_empty() {
            // No filters — return all runs
            return self.list_all_runs();
        }

        // Get all runs, then filter in Rust (same semantics as the HashMap version).
        // The filter logic requires cross-type matching rules that are complex to
        // express in SQL, so we fetch candidate runs and apply filters in memory.
        let all_runs = self.list_all_runs()?;
        let mut results = Vec::new();

        for run in all_runs {
            let run_params = self.get_params(&run.id)?;

            let matches = filters.iter().all(|filter| {
                if let Some(value) = run_params.get(&filter.key) {
                    Self::param_matches(value, &filter.op, &filter.value)
                } else {
                    false
                }
            });

            if matches {
                results.push(run);
            }
        }

        Ok(results)
    }

    /// Check if a parameter value matches a filter
    fn param_matches(value: &ParameterValue, op: &FilterOp, filter_value: &ParameterValue) -> bool {
        match (value, filter_value, op) {
            // Float comparisons
            (ParameterValue::Float(v), ParameterValue::Float(fv), FilterOp::Eq) => {
                (v - fv).abs() < f64::EPSILON
            }
            (ParameterValue::Float(v), ParameterValue::Float(fv), FilterOp::Ne) => {
                (v - fv).abs() >= f64::EPSILON
            }
            (ParameterValue::Float(v), ParameterValue::Float(fv), FilterOp::Gt) => v > fv,
            (ParameterValue::Float(v), ParameterValue::Float(fv), FilterOp::Lt) => v < fv,
            (ParameterValue::Float(v), ParameterValue::Float(fv), FilterOp::Gte) => v >= fv,
            (ParameterValue::Float(v), ParameterValue::Float(fv), FilterOp::Lte) => v <= fv,

            // Int comparisons
            (ParameterValue::Int(v), ParameterValue::Int(fv), FilterOp::Eq) => v == fv,
            (ParameterValue::Int(v), ParameterValue::Int(fv), FilterOp::Ne) => v != fv,
            (ParameterValue::Int(v), ParameterValue::Int(fv), FilterOp::Gt) => v > fv,
            (ParameterValue::Int(v), ParameterValue::Int(fv), FilterOp::Lt) => v < fv,
            (ParameterValue::Int(v), ParameterValue::Int(fv), FilterOp::Gte) => v >= fv,
            (ParameterValue::Int(v), ParameterValue::Int(fv), FilterOp::Lte) => v <= fv,

            // String comparisons
            (ParameterValue::String(v), ParameterValue::String(fv), FilterOp::Eq) => v == fv,
            (ParameterValue::String(v), ParameterValue::String(fv), FilterOp::Ne) => v != fv,
            (ParameterValue::String(v), ParameterValue::String(fv), FilterOp::Contains) => {
                v.contains(fv.as_str())
            }
            (ParameterValue::String(v), ParameterValue::String(fv), FilterOp::StartsWith) => {
                v.starts_with(fv.as_str())
            }

            // Bool comparisons
            (ParameterValue::Bool(v), ParameterValue::Bool(fv), FilterOp::Eq) => v == fv,
            (ParameterValue::Bool(v), ParameterValue::Bool(fv), FilterOp::Ne) => v != fv,

            // Everything else (unsupported ops, cross-type, List/Dict) → false
            _ => false,
        }
    }

    /// Get an experiment by ID
    pub fn get_experiment(&self, experiment_id: &str) -> Result<Experiment> {
        let conn = self.lock_conn()?;

        let row = conn
            .query_row(
                "SELECT id, name, description, config, tags, created_at, updated_at FROM experiments WHERE id = ?1",
                [experiment_id],
                |row| {
                    let id: String = row.get(0)?;
                    let name: String = row.get(1)?;
                    let description: Option<String> = row.get(2)?;
                    let config_str: Option<String> = row.get(3)?;
                    let tags_str: Option<String> = row.get(4)?;
                    let created_str: String = row.get(5)?;
                    let updated_str: String = row.get(6)?;
                    Ok((id, name, description, config_str, tags_str, created_str, updated_str))
                },
            )
            .map_err(|e| match e {
                rusqlite::Error::QueryReturnedNoRows => {
                    StorageError::ExperimentNotFound(experiment_id.to_string())
                }
                _ => StorageError::Backend(format!("Failed to get experiment: {e}")),
            })?;

        let (id, name, description, config_str, tags_str, created_str, updated_str) = row;
        let config = config_str.and_then(|s| serde_json::from_str(&s).ok());
        let tags: HashMap<String, String> =
            tags_str.and_then(|s| serde_json::from_str(&s).ok()).unwrap_or_default();

        Ok(Experiment {
            id,
            name,
            description,
            config,
            tags,
            created_at: parse_timestamp(&created_str),
            updated_at: parse_timestamp(&updated_str),
        })
    }

    /// Get a run by ID
    pub fn get_run(&self, run_id: &str) -> Result<Run> {
        self.query_run_by_id(run_id)
    }

    /// List all experiments
    pub fn list_experiments(&self) -> Result<Vec<Experiment>> {
        let conn = self.lock_conn()?;

        let mut stmt = conn
            .prepare("SELECT id, name, description, config, tags, created_at, updated_at FROM experiments ORDER BY created_at DESC")
            .map_err(|e| StorageError::Backend(format!("Failed to prepare query: {e}")))?;

        let rows = stmt
            .query_map([], |row| {
                let id: String = row.get(0)?;
                let name: String = row.get(1)?;
                let description: Option<String> = row.get(2)?;
                let config_str: Option<String> = row.get(3)?;
                let tags_str: Option<String> = row.get(4)?;
                let created_str: String = row.get(5)?;
                let updated_str: String = row.get(6)?;
                Ok((id, name, description, config_str, tags_str, created_str, updated_str))
            })
            .map_err(|e| StorageError::Backend(format!("Failed to list experiments: {e}")))?;

        let mut result = Vec::new();
        for row in rows {
            let (id, name, description, config_str, tags_str, created_str, updated_str) =
                row.map_err(|e| StorageError::Backend(format!("Failed to read row: {e}")))?;
            let config = config_str.and_then(|s| serde_json::from_str(&s).ok());
            let tags: HashMap<String, String> =
                tags_str.and_then(|s| serde_json::from_str(&s).ok()).unwrap_or_default();
            result.push(Experiment {
                id,
                name,
                description,
                config,
                tags,
                created_at: parse_timestamp(&created_str),
                updated_at: parse_timestamp(&updated_str),
            });
        }

        Ok(result)
    }

    /// List runs for an experiment
    pub fn list_runs(&self, experiment_id: &str) -> Result<Vec<Run>> {
        let conn = self.lock_conn()?;

        // Verify experiment exists
        let exists: bool = conn
            .query_row(
                "SELECT EXISTS(SELECT 1 FROM experiments WHERE id = ?1)",
                [experiment_id],
                |row| row.get(0),
            )
            .map_err(|e| StorageError::Backend(format!("Failed to check experiment: {e}")))?;

        if !exists {
            return Err(StorageError::ExperimentNotFound(experiment_id.to_string()));
        }

        let mut stmt = conn
            .prepare("SELECT id, experiment_id, status, start_time, end_time, tags FROM runs WHERE experiment_id = ?1 ORDER BY start_time")
            .map_err(|e| StorageError::Backend(format!("Failed to prepare query: {e}")))?;

        Self::collect_runs_from_stmt(&mut stmt, params![experiment_id])
    }

    // ── Internal helpers ──────────────────────────────────────────────────

    fn query_run_by_id(&self, run_id: &str) -> Result<Run> {
        let conn = self.lock_conn()?;

        let row = conn
            .query_row(
                "SELECT id, experiment_id, status, start_time, end_time, tags FROM runs WHERE id = ?1",
                [run_id],
                |row| {
                    let id: String = row.get(0)?;
                    let experiment_id: String = row.get(1)?;
                    let status_str: String = row.get(2)?;
                    let start_str: Option<String> = row.get(3)?;
                    let end_str: Option<String> = row.get(4)?;
                    let tags_str: Option<String> = row.get(5)?;
                    Ok((id, experiment_id, status_str, start_str, end_str, tags_str))
                },
            )
            .map_err(|e| match e {
                rusqlite::Error::QueryReturnedNoRows => {
                    StorageError::RunNotFound(run_id.to_string())
                }
                _ => StorageError::Backend(format!("Failed to get run: {e}")),
            })?;

        Ok(Self::row_to_run(row))
    }

    fn list_all_runs(&self) -> Result<Vec<Run>> {
        let conn = self.lock_conn()?;

        let mut stmt = conn
            .prepare("SELECT id, experiment_id, status, start_time, end_time, tags FROM runs ORDER BY start_time")
            .map_err(|e| StorageError::Backend(format!("Failed to prepare query: {e}")))?;

        Self::collect_runs_from_stmt(&mut stmt, [])
    }

    fn collect_runs_from_stmt<P: rusqlite::Params>(
        stmt: &mut rusqlite::Statement<'_>,
        params: P,
    ) -> Result<Vec<Run>> {
        let rows = stmt
            .query_map(params, |row| {
                let id: String = row.get(0)?;
                let experiment_id: String = row.get(1)?;
                let status_str: String = row.get(2)?;
                let start_str: Option<String> = row.get(3)?;
                let end_str: Option<String> = row.get(4)?;
                let tags_str: Option<String> = row.get(5)?;
                Ok((id, experiment_id, status_str, start_str, end_str, tags_str))
            })
            .map_err(|e| StorageError::Backend(format!("Failed to query runs: {e}")))?;

        let mut result = Vec::new();
        for row in rows {
            let tuple =
                row.map_err(|e| StorageError::Backend(format!("Failed to read run row: {e}")))?;
            result.push(Self::row_to_run(tuple));
        }
        Ok(result)
    }

    fn row_to_run(
        row: (String, String, String, Option<String>, Option<String>, Option<String>),
    ) -> Run {
        let (id, experiment_id, status_str, start_str, end_str, tags_str) = row;
        let start_time = start_str.map_or_else(Utc::now, |s| parse_timestamp(&s));
        let end_time = end_str.map(|s| parse_timestamp(&s));
        let tags: HashMap<String, String> =
            tags_str.and_then(|s| serde_json::from_str(&s).ok()).unwrap_or_default();

        Run {
            id,
            experiment_id,
            status: str_to_status(&status_str),
            start_time,
            end_time,
            params: HashMap::new(),
            tags,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_param_matches_all_filter_ops() {
        let float_val = ParameterValue::Float(5.0);
        let float_filter = ParameterValue::Float(5.0);
        let string_val = ParameterValue::String("hello world".to_string());
        let string_filter = ParameterValue::String("hello".to_string());

        for op in &[
            FilterOp::Eq,
            FilterOp::Ne,
            FilterOp::Gt,
            FilterOp::Lt,
            FilterOp::Gte,
            FilterOp::Lte,
            FilterOp::Contains,
            FilterOp::StartsWith,
        ] {
            match op {
                FilterOp::Eq => {
                    assert!(SqliteBackend::param_matches(&float_val, op, &float_filter));
                }
                FilterOp::Ne => {
                    assert!(!SqliteBackend::param_matches(&float_val, op, &float_filter));
                }
                FilterOp::Gt => {
                    assert!(!SqliteBackend::param_matches(&float_val, op, &float_filter));
                }
                FilterOp::Lt => {
                    assert!(!SqliteBackend::param_matches(&float_val, op, &float_filter));
                }
                FilterOp::Gte => {
                    assert!(SqliteBackend::param_matches(&float_val, op, &float_filter));
                }
                FilterOp::Lte => {
                    assert!(SqliteBackend::param_matches(&float_val, op, &float_filter));
                }
                FilterOp::Contains => {
                    // Unsupported for Float, but supported for String
                    assert!(!SqliteBackend::param_matches(&float_val, op, &float_filter));
                    assert!(SqliteBackend::param_matches(&string_val, op, &string_filter));
                }
                FilterOp::StartsWith => {
                    assert!(!SqliteBackend::param_matches(&float_val, op, &float_filter));
                    assert!(SqliteBackend::param_matches(&string_val, op, &string_filter));
                }
            }
        }
    }

    #[test]
    fn test_param_matches_cross_type_returns_false() {
        let float_val = ParameterValue::Float(42.0);
        let int_filter = ParameterValue::Int(42);
        // Cross-type comparisons always return false
        assert!(!SqliteBackend::param_matches(&float_val, &FilterOp::Eq, &int_filter));
        assert!(!SqliteBackend::param_matches(&float_val, &FilterOp::Ne, &int_filter));
        assert!(!SqliteBackend::param_matches(&float_val, &FilterOp::Gt, &int_filter));
        assert!(!SqliteBackend::param_matches(&float_val, &FilterOp::Lt, &int_filter));
    }

    #[test]
    fn test_param_matches_list_and_dict_always_false() {
        let list_val = ParameterValue::List(vec![ParameterValue::Int(1)]);
        let list_filter = ParameterValue::List(vec![ParameterValue::Int(1)]);
        let dict_val =
            ParameterValue::Dict(HashMap::from([("k".to_string(), ParameterValue::Int(1))]));
        let dict_filter =
            ParameterValue::Dict(HashMap::from([("k".to_string(), ParameterValue::Int(1))]));

        // List and Dict do not support any filter operations - verify with match
        let test_cases: Vec<(&ParameterValue, &ParameterValue, &FilterOp)> = vec![
            (&list_val, &list_filter, &FilterOp::Eq),
            (&list_val, &list_filter, &FilterOp::Ne),
            (&dict_val, &dict_filter, &FilterOp::Eq),
            (&dict_val, &dict_filter, &FilterOp::Ne),
        ];

        for (val, filt, op) in &test_cases {
            let result = SqliteBackend::param_matches(val, op, filt);
            match (val, filt, op) {
                (ParameterValue::List(..), ParameterValue::List(..), _) => {
                    assert!(!result, "List should not match with any op");
                }
                (ParameterValue::Dict(..), ParameterValue::Dict(..), _) => {
                    assert!(!result, "Dict should not match with any op");
                }
                _ => unreachable!(),
            }
        }
    }

    // ── Additional coverage tests ──

    #[test]
    fn test_str_to_status_all_variants() {
        assert_eq!(str_to_status("pending"), crate::storage::RunStatus::Pending);
        assert_eq!(str_to_status("running"), crate::storage::RunStatus::Running);
        assert_eq!(str_to_status("completed"), crate::storage::RunStatus::Success);
        assert_eq!(str_to_status("failed"), crate::storage::RunStatus::Failed);
        assert_eq!(str_to_status("cancelled"), crate::storage::RunStatus::Cancelled);
    }

    #[test]
    fn test_str_to_status_unknown_defaults_to_failed() {
        assert_eq!(str_to_status("unknown"), crate::storage::RunStatus::Failed);
        assert_eq!(str_to_status(""), crate::storage::RunStatus::Failed);
        assert_eq!(str_to_status("RUNNING"), crate::storage::RunStatus::Failed);
    }

    #[test]
    fn test_parse_timestamp_valid() {
        let ts = parse_timestamp("2026-03-08T12:00:00Z");
        assert_eq!(ts.year(), 2026);
        assert_eq!(ts.month(), 3);
    }

    #[test]
    fn test_parse_timestamp_invalid_falls_back() {
        let ts = parse_timestamp("not-a-date");
        // Should fall back to now
        let now = chrono::Utc::now();
        let diff = (now - ts).num_seconds().abs();
        assert!(diff < 5); // Within 5 seconds of now
    }

    #[test]
    fn test_param_matches_int_all_ops() {
        let v5 = ParameterValue::Int(5);
        let v3 = ParameterValue::Int(3);
        let v5_dup = ParameterValue::Int(5);
        let v7 = ParameterValue::Int(7);

        assert!(SqliteBackend::param_matches(&v5, &FilterOp::Eq, &v5_dup));
        assert!(!SqliteBackend::param_matches(&v5, &FilterOp::Eq, &v3));

        assert!(!SqliteBackend::param_matches(&v5, &FilterOp::Ne, &v5_dup));
        assert!(SqliteBackend::param_matches(&v5, &FilterOp::Ne, &v3));

        assert!(SqliteBackend::param_matches(&v5, &FilterOp::Gt, &v3));
        assert!(!SqliteBackend::param_matches(&v3, &FilterOp::Gt, &v5));
        assert!(!SqliteBackend::param_matches(&v5, &FilterOp::Gt, &v5_dup));

        assert!(SqliteBackend::param_matches(&v3, &FilterOp::Lt, &v5));
        assert!(!SqliteBackend::param_matches(&v5, &FilterOp::Lt, &v3));

        assert!(SqliteBackend::param_matches(&v5, &FilterOp::Gte, &v5_dup));
        assert!(SqliteBackend::param_matches(&v7, &FilterOp::Gte, &v5));
        assert!(!SqliteBackend::param_matches(&v3, &FilterOp::Gte, &v5));

        assert!(SqliteBackend::param_matches(&v5, &FilterOp::Lte, &v5_dup));
        assert!(SqliteBackend::param_matches(&v3, &FilterOp::Lte, &v5));
        assert!(!SqliteBackend::param_matches(&v7, &FilterOp::Lte, &v5));
    }

    #[test]
    fn test_param_matches_string_eq_ne() {
        let hello = ParameterValue::String("hello".to_string());
        let world = ParameterValue::String("world".to_string());
        let hello_dup = ParameterValue::String("hello".to_string());

        assert!(SqliteBackend::param_matches(&hello, &FilterOp::Eq, &hello_dup));
        assert!(!SqliteBackend::param_matches(&hello, &FilterOp::Eq, &world));

        assert!(!SqliteBackend::param_matches(&hello, &FilterOp::Ne, &hello_dup));
        assert!(SqliteBackend::param_matches(&hello, &FilterOp::Ne, &world));
    }

    #[test]
    fn test_param_matches_string_contains() {
        let full = ParameterValue::String("hello world".to_string());
        let sub = ParameterValue::String("world".to_string());
        let missing = ParameterValue::String("xyz".to_string());

        assert!(SqliteBackend::param_matches(&full, &FilterOp::Contains, &sub));
        assert!(!SqliteBackend::param_matches(&full, &FilterOp::Contains, &missing));
    }

    #[test]
    fn test_param_matches_string_starts_with() {
        let full = ParameterValue::String("hello world".to_string());
        let prefix = ParameterValue::String("hello".to_string());
        let wrong = ParameterValue::String("world".to_string());

        assert!(SqliteBackend::param_matches(&full, &FilterOp::StartsWith, &prefix));
        assert!(!SqliteBackend::param_matches(&full, &FilterOp::StartsWith, &wrong));
    }

    #[test]
    fn test_param_matches_bool_eq_ne() {
        let t = ParameterValue::Bool(true);
        let f = ParameterValue::Bool(false);
        let t_dup = ParameterValue::Bool(true);

        assert!(SqliteBackend::param_matches(&t, &FilterOp::Eq, &t_dup));
        assert!(!SqliteBackend::param_matches(&t, &FilterOp::Eq, &f));

        assert!(!SqliteBackend::param_matches(&t, &FilterOp::Ne, &t_dup));
        assert!(SqliteBackend::param_matches(&t, &FilterOp::Ne, &f));
    }

    #[test]
    fn test_param_matches_float_gt_lt() {
        let f5 = ParameterValue::Float(5.0);
        let f3 = ParameterValue::Float(3.0);

        assert!(SqliteBackend::param_matches(&f5, &FilterOp::Gt, &f3));
        assert!(!SqliteBackend::param_matches(&f3, &FilterOp::Gt, &f5));

        assert!(SqliteBackend::param_matches(&f3, &FilterOp::Lt, &f5));
        assert!(!SqliteBackend::param_matches(&f5, &FilterOp::Lt, &f3));
    }

    #[test]
    fn test_param_matches_float_gte_lte() {
        let f5 = ParameterValue::Float(5.0);
        let f5_dup = ParameterValue::Float(5.0);
        let f3 = ParameterValue::Float(3.0);

        assert!(SqliteBackend::param_matches(&f5, &FilterOp::Gte, &f5_dup));
        assert!(SqliteBackend::param_matches(&f5, &FilterOp::Gte, &f3));
        assert!(!SqliteBackend::param_matches(&f3, &FilterOp::Gte, &f5));

        assert!(SqliteBackend::param_matches(&f5, &FilterOp::Lte, &f5_dup));
        assert!(SqliteBackend::param_matches(&f3, &FilterOp::Lte, &f5));
        assert!(!SqliteBackend::param_matches(&f5, &FilterOp::Lte, &f3));
    }

    #[test]
    fn test_param_matches_unsupported_ops_return_false() {
        // String with Gt/Lt/Gte/Lte should return false
        let s = ParameterValue::String("hello".to_string());
        assert!(!SqliteBackend::param_matches(&s, &FilterOp::Gt, &s));
        assert!(!SqliteBackend::param_matches(&s, &FilterOp::Lt, &s));
        assert!(!SqliteBackend::param_matches(&s, &FilterOp::Gte, &s));
        assert!(!SqliteBackend::param_matches(&s, &FilterOp::Lte, &s));

        // Bool with Gt/Lt/Gte/Lte/Contains/StartsWith should return false
        let b = ParameterValue::Bool(true);
        assert!(!SqliteBackend::param_matches(&b, &FilterOp::Gt, &b));
        assert!(!SqliteBackend::param_matches(&b, &FilterOp::Lt, &b));
        assert!(!SqliteBackend::param_matches(&b, &FilterOp::Contains, &b));
        assert!(!SqliteBackend::param_matches(&b, &FilterOp::StartsWith, &b));

        // Int with Contains/StartsWith should return false
        let i = ParameterValue::Int(42);
        assert!(!SqliteBackend::param_matches(&i, &FilterOp::Contains, &i));
        assert!(!SqliteBackend::param_matches(&i, &FilterOp::StartsWith, &i));
    }

    #[test]
    fn test_row_to_run_with_all_fields() {
        let now = chrono::Utc::now().to_rfc3339();
        let tags_json = serde_json::json!({"env": "prod"}).to_string();
        let row = (
            "run-123".to_string(),
            "exp-456".to_string(),
            "running".to_string(),
            Some(now.clone()),
            Some(now.clone()),
            Some(tags_json),
        );
        let run = SqliteBackend::row_to_run(row);
        assert_eq!(run.id, "run-123");
        assert_eq!(run.experiment_id, "exp-456");
        assert_eq!(run.status, crate::storage::RunStatus::Running);
        assert!(run.end_time.is_some());
        assert_eq!(run.tags.get("env").map(String::as_str), Some("prod"));
    }

    #[test]
    fn test_row_to_run_with_none_fields() {
        let row =
            ("run-1".to_string(), "exp-1".to_string(), "pending".to_string(), None, None, None);
        let run = SqliteBackend::row_to_run(row);
        assert_eq!(run.id, "run-1");
        assert_eq!(run.status, crate::storage::RunStatus::Pending);
        assert!(run.end_time.is_none());
        assert!(run.tags.is_empty());
    }

    #[test]
    fn test_row_to_run_invalid_tags_json() {
        let row = (
            "run-1".to_string(),
            "exp-1".to_string(),
            "completed".to_string(),
            Some(chrono::Utc::now().to_rfc3339()),
            None,
            Some("not-valid-json".to_string()),
        );
        let run = SqliteBackend::row_to_run(row);
        // Invalid JSON for tags should result in empty HashMap
        assert!(run.tags.is_empty());
    }

    #[test]
    fn test_param_matches_float_ne() {
        let f5 = ParameterValue::Float(5.0);
        let f3 = ParameterValue::Float(3.0);
        let f5_dup = ParameterValue::Float(5.0);

        assert!(SqliteBackend::param_matches(&f5, &FilterOp::Ne, &f3));
        assert!(!SqliteBackend::param_matches(&f5, &FilterOp::Ne, &f5_dup));
    }

    use chrono::Datelike;

    #[test]
    fn test_parse_timestamp_rfc3339() {
        let ts = parse_timestamp("2025-06-15T10:30:00+00:00");
        assert_eq!(ts.year(), 2025);
        assert_eq!(ts.month(), 6);
        assert_eq!(ts.day(), 15);
    }

    // ── test_cov4 additional coverage tests ────────────────────────

    #[test]
    fn test_cov4_str_to_status_all_case_sensitive() {
        // Verify case sensitivity
        assert_eq!(str_to_status("Pending"), crate::storage::RunStatus::Failed); // capital P
        assert_eq!(str_to_status("COMPLETED"), crate::storage::RunStatus::Failed);
        assert_eq!(str_to_status("Running"), crate::storage::RunStatus::Failed);
        assert_eq!(str_to_status("Cancelled"), crate::storage::RunStatus::Failed);
        assert_eq!(str_to_status("FAILED"), crate::storage::RunStatus::Failed);
    }

    #[test]
    fn test_cov4_str_to_status_whitespace() {
        assert_eq!(str_to_status(" pending"), crate::storage::RunStatus::Failed);
        assert_eq!(str_to_status("pending "), crate::storage::RunStatus::Failed);
        assert_eq!(str_to_status("  "), crate::storage::RunStatus::Failed);
    }

    #[test]
    fn test_cov4_parse_timestamp_various_formats() {
        // With timezone offset
        let ts = parse_timestamp("2024-12-25T00:00:00+05:30");
        assert_eq!(ts.year(), 2024);
        assert_eq!(ts.month(), 12);

        // UTC with Z
        let ts2 = parse_timestamp("2020-01-01T00:00:00Z");
        assert_eq!(ts2.year(), 2020);

        // Negative offset
        let ts3 = parse_timestamp("2023-06-15T12:00:00-07:00");
        assert_eq!(ts3.year(), 2023);
    }

    #[test]
    fn test_cov4_parse_timestamp_empty_string() {
        let ts = parse_timestamp("");
        let now = chrono::Utc::now();
        let diff = (now - ts).num_seconds().abs();
        assert!(diff < 5);
    }

    #[test]
    fn test_cov4_parse_timestamp_partial_date() {
        // Partial date should fail and fall back to now
        let ts = parse_timestamp("2025-01");
        let now = chrono::Utc::now();
        let diff = (now - ts).num_seconds().abs();
        assert!(diff < 5);
    }

    #[test]
    fn test_cov4_row_to_run_all_statuses() {
        for (status_str, expected) in &[
            ("pending", crate::storage::RunStatus::Pending),
            ("running", crate::storage::RunStatus::Running),
            ("completed", crate::storage::RunStatus::Success),
            ("failed", crate::storage::RunStatus::Failed),
            ("cancelled", crate::storage::RunStatus::Cancelled),
            ("unknown", crate::storage::RunStatus::Failed),
        ] {
            let row = (
                "run-x".to_string(),
                "exp-x".to_string(),
                status_str.to_string(),
                Some("2026-01-01T00:00:00Z".to_string()),
                None,
                None,
            );
            let run = SqliteBackend::row_to_run(row);
            assert_eq!(run.status, *expected, "Status for '{status_str}'");
        }
    }

    #[test]
    fn test_cov4_row_to_run_with_end_time() {
        let row = (
            "r1".to_string(),
            "e1".to_string(),
            "completed".to_string(),
            Some("2026-01-01T00:00:00Z".to_string()),
            Some("2026-01-01T01:00:00Z".to_string()),
            None,
        );
        let run = SqliteBackend::row_to_run(row);
        assert!(run.end_time.is_some());
        let end = run.end_time.unwrap();
        assert_eq!(end.hour(), 1);
    }

    #[test]
    fn test_cov4_row_to_run_with_complex_tags() {
        let tags =
            serde_json::json!({"env": "staging", "model": "qwen2", "version": "1.0"}).to_string();
        let row = (
            "r1".to_string(),
            "e1".to_string(),
            "running".to_string(),
            Some("2026-01-01T00:00:00Z".to_string()),
            None,
            Some(tags),
        );
        let run = SqliteBackend::row_to_run(row);
        assert_eq!(run.tags.len(), 3);
        assert_eq!(run.tags.get("env").map(String::as_str), Some("staging"));
        assert_eq!(run.tags.get("model").map(String::as_str), Some("qwen2"));
    }

    #[test]
    fn test_cov4_row_to_run_empty_tags_json() {
        let row = (
            "r1".to_string(),
            "e1".to_string(),
            "pending".to_string(),
            None,
            None,
            Some("{}".to_string()),
        );
        let run = SqliteBackend::row_to_run(row);
        assert!(run.tags.is_empty());
    }

    #[test]
    fn test_cov4_row_to_run_invalid_start_time() {
        let row = (
            "r1".to_string(),
            "e1".to_string(),
            "completed".to_string(),
            Some("not-a-date".to_string()),
            Some("also-not-a-date".to_string()),
            None,
        );
        let run = SqliteBackend::row_to_run(row);
        // Invalid timestamps fall back to now
        let now = chrono::Utc::now();
        let diff = (now - run.start_time).num_seconds().abs();
        assert!(diff < 5);
    }

    #[test]
    fn test_cov4_param_matches_int_contains_false() {
        let v = ParameterValue::Int(42);
        assert!(!SqliteBackend::param_matches(&v, &FilterOp::Contains, &v));
    }

    #[test]
    fn test_cov4_param_matches_int_starts_with_false() {
        let v = ParameterValue::Int(42);
        assert!(!SqliteBackend::param_matches(&v, &FilterOp::StartsWith, &v));
    }

    #[test]
    fn test_cov4_param_matches_bool_all_unsupported_ops() {
        let t = ParameterValue::Bool(true);
        let f = ParameterValue::Bool(false);

        // Bool only supports Eq and Ne
        assert!(SqliteBackend::param_matches(&t, &FilterOp::Eq, &t));
        assert!(!SqliteBackend::param_matches(&t, &FilterOp::Eq, &f));
        assert!(!SqliteBackend::param_matches(&t, &FilterOp::Ne, &t));
        assert!(SqliteBackend::param_matches(&t, &FilterOp::Ne, &f));

        // All others unsupported
        assert!(!SqliteBackend::param_matches(&t, &FilterOp::Gte, &t));
        assert!(!SqliteBackend::param_matches(&t, &FilterOp::Lte, &t));
    }

    #[test]
    fn test_cov4_param_matches_string_all_unsupported_ops() {
        let s = ParameterValue::String("test".to_string());
        // String does not support Gt, Lt, Gte, Lte
        assert!(!SqliteBackend::param_matches(&s, &FilterOp::Gt, &s));
        assert!(!SqliteBackend::param_matches(&s, &FilterOp::Lt, &s));
        assert!(!SqliteBackend::param_matches(&s, &FilterOp::Gte, &s));
        assert!(!SqliteBackend::param_matches(&s, &FilterOp::Lte, &s));
    }

    #[test]
    fn test_cov4_param_matches_float_epsilon_boundary() {
        let a = ParameterValue::Float(1.0);
        let b = ParameterValue::Float(1.0 + f64::EPSILON * 0.5);
        // Should be Eq because diff < EPSILON
        assert!(SqliteBackend::param_matches(&a, &FilterOp::Eq, &b));
    }

    #[test]
    fn test_cov4_param_matches_float_ne_different() {
        let a = ParameterValue::Float(1.0);
        let b = ParameterValue::Float(1.1);
        assert!(SqliteBackend::param_matches(&a, &FilterOp::Ne, &b));
        assert!(!SqliteBackend::param_matches(&a, &FilterOp::Eq, &b));
    }

    #[test]
    fn test_cov4_param_matches_string_contains_empty() {
        let full = ParameterValue::String("hello".to_string());
        let empty = ParameterValue::String(String::new());
        // Every string contains empty string
        assert!(SqliteBackend::param_matches(&full, &FilterOp::Contains, &empty));
    }

    #[test]
    fn test_cov4_param_matches_string_starts_with_empty() {
        let full = ParameterValue::String("hello".to_string());
        let empty = ParameterValue::String(String::new());
        assert!(SqliteBackend::param_matches(&full, &FilterOp::StartsWith, &empty));
    }

    #[test]
    fn test_cov4_param_matches_string_contains_full() {
        let s = ParameterValue::String("abcdef".to_string());
        let s2 = ParameterValue::String("abcdef".to_string());
        assert!(SqliteBackend::param_matches(&s, &FilterOp::Contains, &s2));
    }

    #[test]
    fn test_cov4_param_matches_cross_type_all_combinations() {
        let float = ParameterValue::Float(1.0);
        let int = ParameterValue::Int(1);
        let string = ParameterValue::String("1".to_string());
        let bool_val = ParameterValue::Bool(true);

        // Float vs Int
        assert!(!SqliteBackend::param_matches(&float, &FilterOp::Eq, &int));
        // Float vs String
        assert!(!SqliteBackend::param_matches(&float, &FilterOp::Eq, &string));
        // Float vs Bool
        assert!(!SqliteBackend::param_matches(&float, &FilterOp::Eq, &bool_val));
        // Int vs String
        assert!(!SqliteBackend::param_matches(&int, &FilterOp::Eq, &string));
        // Int vs Bool
        assert!(!SqliteBackend::param_matches(&int, &FilterOp::Eq, &bool_val));
        // String vs Bool
        assert!(!SqliteBackend::param_matches(&string, &FilterOp::Eq, &bool_val));
    }

    #[test]
    fn test_cov4_param_matches_list_all_ops() {
        let list = ParameterValue::List(vec![ParameterValue::Int(1), ParameterValue::Int(2)]);
        for op in &[
            FilterOp::Eq,
            FilterOp::Ne,
            FilterOp::Gt,
            FilterOp::Lt,
            FilterOp::Gte,
            FilterOp::Lte,
            FilterOp::Contains,
            FilterOp::StartsWith,
        ] {
            assert!(
                !SqliteBackend::param_matches(&list, op, &list),
                "List should not match with {op:?}"
            );
        }
    }

    #[test]
    fn test_cov4_param_matches_dict_all_ops() {
        let dict = ParameterValue::Dict(HashMap::from([("k".to_string(), ParameterValue::Int(1))]));
        for op in &[
            FilterOp::Eq,
            FilterOp::Ne,
            FilterOp::Gt,
            FilterOp::Lt,
            FilterOp::Gte,
            FilterOp::Lte,
            FilterOp::Contains,
            FilterOp::StartsWith,
        ] {
            assert!(
                !SqliteBackend::param_matches(&dict, op, &dict),
                "Dict should not match with {op:?}"
            );
        }
    }

    use chrono::Timelike;

    #[test]
    fn test_cov4_row_to_run_params_always_empty() {
        // row_to_run always initializes params as empty HashMap
        let row = ("r1".to_string(), "e1".to_string(), "running".to_string(), None, None, None);
        let run = SqliteBackend::row_to_run(row);
        assert!(run.params.is_empty());
    }

    #[test]
    fn test_cov4_row_to_run_no_start_time_uses_now() {
        let row = (
            "r1".to_string(),
            "e1".to_string(),
            "pending".to_string(),
            None, // no start_time → uses now
            None,
            None,
        );
        let run = SqliteBackend::row_to_run(row);
        let now = chrono::Utc::now();
        let diff = (now - run.start_time).num_seconds().abs();
        assert!(diff < 5, "start_time should be near now when None, diff={diff}");
    }
}
