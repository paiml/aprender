//! Data integrity preflight checks.

use std::collections::HashMap;

use super::{CheckResult, CheckType, PreflightCheck};

impl PreflightCheck {
    // =========================================================================
    // Built-in Data Integrity Checks
    // =========================================================================

    /// Check for NaN values in data
    pub fn no_nan_values() -> Self {
        Self::new(
            "no_nan_values",
            CheckType::DataIntegrity,
            "Ensures no NaN values exist in the dataset",
            |data, _ctx| {
                let mut nan_count = 0;
                let mut nan_locations = Vec::new();

                for (row_idx, row) in data.iter().enumerate() {
                    for (col_idx, val) in row.iter().enumerate() {
                        if val.is_nan() {
                            nan_count += 1;
                            if nan_locations.len() < 5 {
                                nan_locations.push(format!("({row_idx}, {col_idx})"));
                            }
                        }
                    }
                }

                if nan_count == 0 {
                    CheckResult::passed("No NaN values found")
                } else {
                    CheckResult::failed_with_details(
                        format!("Found {nan_count} NaN values"),
                        format!("First locations: {}", nan_locations.join(", ")),
                    )
                }
            },
        )
    }

    /// Check for infinite values in data
    pub fn no_inf_values() -> Self {
        Self::new(
            "no_inf_values",
            CheckType::DataIntegrity,
            "Ensures no infinite values exist in the dataset",
            |data, _ctx| {
                let mut inf_count = 0;
                let mut inf_locations = Vec::new();

                for (row_idx, row) in data.iter().enumerate() {
                    for (col_idx, val) in row.iter().enumerate() {
                        if val.is_infinite() {
                            inf_count += 1;
                            if inf_locations.len() < 5 {
                                inf_locations.push(format!("({row_idx}, {col_idx})"));
                            }
                        }
                    }
                }

                if inf_count == 0 {
                    CheckResult::passed("No infinite values found")
                } else {
                    CheckResult::failed_with_details(
                        format!("Found {inf_count} infinite values"),
                        format!("First locations: {}", inf_locations.join(", ")),
                    )
                }
            },
        )
    }

    /// Check minimum number of samples
    pub fn min_samples(min: usize) -> Self {
        Self::new(
            "min_samples",
            CheckType::DataIntegrity,
            format!("Ensures at least {min} samples exist"),
            move |data, ctx| {
                let min_required = ctx.min_samples.unwrap_or(min);
                let actual = data.len();

                if actual >= min_required {
                    CheckResult::passed(format!("Found {actual} samples (minimum: {min_required})"))
                } else {
                    CheckResult::failed(format!(
                        "Only {actual} samples found (minimum: {min_required})"
                    ))
                }
            },
        )
    }

    /// Check minimum number of features
    pub fn min_features(min: usize) -> Self {
        Self::new(
            "min_features",
            CheckType::DataIntegrity,
            format!("Ensures at least {min} features exist"),
            move |data, ctx| {
                let min_required = ctx.min_features.unwrap_or(min);
                let actual = data.first().map_or(0, Vec::len);

                if actual >= min_required {
                    CheckResult::passed(format!(
                        "Found {actual} features (minimum: {min_required})"
                    ))
                } else {
                    CheckResult::failed(format!(
                        "Only {actual} features found (minimum: {min_required})"
                    ))
                }
            },
        )
    }

    /// Check for consistent row lengths
    pub fn consistent_dimensions() -> Self {
        Self::new(
            "consistent_dimensions",
            CheckType::DataIntegrity,
            "Ensures all rows have the same number of features",
            |data, _ctx| {
                if data.is_empty() {
                    return CheckResult::skipped("No data to check");
                }

                let expected_len = data[0].len();
                let mut inconsistent = Vec::new();

                for (idx, row) in data.iter().enumerate() {
                    if row.len() != expected_len {
                        inconsistent.push(format!("row {idx}: {} features", row.len()));
                        if inconsistent.len() >= 5 {
                            break;
                        }
                    }
                }

                if inconsistent.is_empty() {
                    CheckResult::passed(format!(
                        "All {} rows have {expected_len} features",
                        data.len()
                    ))
                } else {
                    CheckResult::failed_with_details(
                        format!("Inconsistent dimensions (expected {expected_len} features)"),
                        inconsistent.join(", "),
                    )
                }
            },
        )
    }

    /// Check feature variance (detect constant features)
    pub fn no_constant_features() -> Self {
        Self::new(
            "no_constant_features",
            CheckType::DataIntegrity,
            "Ensures no features have zero variance",
            |data, _ctx| {
                if data.is_empty() || data[0].is_empty() {
                    return CheckResult::skipped("No data to check");
                }

                let n_features = data[0].len();
                let mut constant_features = Vec::new();

                for col in 0..n_features {
                    let values: Vec<f64> = data.iter().map(|row| row[col]).collect();
                    let first = values[0];

                    if values.iter().all(|v| (*v - first).abs() < f64::EPSILON) {
                        constant_features.push(col);
                    }
                }

                if constant_features.is_empty() {
                    CheckResult::passed("No constant features found")
                } else {
                    CheckResult::warning(format!(
                        "Found {} constant feature(s): {:?}",
                        constant_features.len(),
                        constant_features
                    ))
                }
            },
        )
        .optional()
    }

    /// Check for label balance (classification)
    pub fn label_balance(max_imbalance_ratio: f64) -> Self {
        Self::new(
            "label_balance",
            CheckType::DataIntegrity,
            format!("Ensures class imbalance ratio <= {max_imbalance_ratio}"),
            move |data, _ctx| {
                if data.is_empty() || data[0].is_empty() {
                    return CheckResult::skipped("No data to check");
                }

                // Assume last column is label
                let labels: Vec<i64> =
                    data.iter().map(|row| *row.last().unwrap_or(&0.0) as i64).collect();

                let mut counts: HashMap<i64, usize> = HashMap::new();
                for label in &labels {
                    *counts.entry(*label).or_default() += 1;
                }

                if counts.is_empty() {
                    return CheckResult::skipped("No labels found");
                }

                let max_count = *counts.values().max().unwrap_or(&0);
                let min_count = *counts.values().min().unwrap_or(&0);

                if min_count == 0 {
                    return CheckResult::failed("One or more classes have zero samples");
                }

                let ratio = max_count as f64 / min_count as f64;

                if ratio <= max_imbalance_ratio {
                    CheckResult::passed(format!(
                        "Class imbalance ratio {ratio:.2} <= {max_imbalance_ratio}"
                    ))
                } else {
                    CheckResult::warning(format!(
                        "Class imbalance ratio {ratio:.2} > {max_imbalance_ratio}"
                    ))
                }
            },
        )
        .optional()
    }
}
